'''
Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this computer program. 
Using this computer program means that you agree to the terms in the LICENSE file (https://ringnet.is.tue.mpg.de/license). 
Any use not explicitly granted by the LICENSE is prohibited.
Copyright 2020 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its 
Max Planck Institute for Intelligent Systems. All rights reserved.
More information about the NoW Challenge is available at https://ringnet.is.tue.mpg.de/challenge.
For comments or questions, please email us at ringnet@tue.mpg.de
'''

import os
import numpy as np
from math import sqrt
import chumpy as ch
from psbody.mesh import Mesh
from smpl_webuser.posemapper import Rodrigues
from sbody.alignment.objectives import sample_from_mesh
from sbody.mesh_distance import ScanToMesh
from psbody.mesh.meshviewer import MeshViewer
from scipy.sparse.linalg import cg


def rigid_scan_2_mesh_alignment(scan, mesh, visualize=False):
    options = {'sparse_solver': lambda A, x: cg(A, x, maxiter=2000)[0]}
    options['disp'] = 1.0
    options['delta_0'] = 0.1
    options['e_3'] = 1e-4

    s = ch.ones(1)
    r = ch.zeros(3)
    R = Rodrigues(r)
    t = ch.zeros(3)
    trafo_mesh = s*(R.dot(mesh.v.T)).T + t

    sampler = sample_from_mesh(scan, sample_type='vertices')
    s2m = ScanToMesh(scan, trafo_mesh, mesh.f, scan_sampler=sampler, signed=False, normalize=False)

    if visualize:       
        #Visualization code
        mv = MeshViewer()
        mv.set_static_meshes([scan])
        tmp_mesh = Mesh(trafo_mesh.r, mesh.f)
        tmp_mesh.set_vertex_colors('light sky blue')
        mv.set_dynamic_meshes([tmp_mesh])
        def on_show(_):
            tmp_mesh = Mesh(trafo_mesh.r, mesh.f)
            tmp_mesh.set_vertex_colors('light sky blue')
            mv.set_dynamic_meshes([tmp_mesh])
    else:
        def on_show(_):
            pass

    ch.minimize(fun={'dist': s2m, 's_reg': 100*(ch.abs(s)-s)}, x0=[s, r, t], callback=on_show, options=options)
    return s,Rodrigues(r),t

def compute_mask(grundtruth_landmark_points):
    """
    Computes a circular area around the center of the face.
    :param grundtruth_landmark_points: Landmarks of the ground truth scans
    :return face center and mask radius
    """

    #  Take the nose-bottom and go upwards a bit:
    nose_bottom = np.array(grundtruth_landmark_points[4])
    nose_bridge = (np.array(grundtruth_landmark_points[1]) + np.array(grundtruth_landmark_points[2])) / 2  # between the inner eye corners
    face_centre = nose_bottom + 0.3 * (nose_bridge - nose_bottom)
    # Compute the radius for the face mask:
    outer_eye_dist = np.linalg.norm(np.array(grundtruth_landmark_points[0]) - np.array(grundtruth_landmark_points[3]))
    nose_dist = np.linalg.norm(nose_bridge - nose_bottom)
    # mask_radius = 1.2 * (outer_eye_dist + nose_dist) / 2
    mask_radius = 1.4 * (outer_eye_dist + nose_dist) / 2
    return (face_centre, mask_radius)

def crop_face_scan(groundtruth_vertices, groundtruth_faces, grundtruth_landmark_points):
    """
    Crop face scan to a circular area around the center of the face.
    :param groundtruth_vertices: An n x 3 numpy array of vertices from a ground truth scan.
    :param groundtruth_faces: Faces of the ground truth scan
    :param grundtruth_landmark_points: A 7 x 3 list with annotations of the ground truth scan.
    return: Cropped face scan
    """

    # Compute mask
    face_centre, mask_radius = compute_mask(grundtruth_landmark_points)

    # Compute mask vertex indiced
    dist = np.linalg.norm(groundtruth_vertices-face_centre, axis=1)
    ids = np.where(dist <= mask_radius)[0]

    # Mask scan
    masked_gt_scan = Mesh(v=groundtruth_vertices, f=groundtruth_faces)
    masked_gt_scan.keep_vertices(ids)
    return masked_gt_scan

def write_alignment_check_files(
    check_alignment_output_dir,
    masked_gt_scan,
    predicted_mesh_vertices,
    predicted_mesh_vertices_aligned,
    predicted_mesh_faces,
):
    os.makedirs(check_alignment_output_dir, exist_ok=True)
    _join = lambda x: os.path.join(check_alignment_output_dir, x)
    masked_gt_scan.write_obj(_join("gt_scan_val.obj"))
    Mesh(predicted_mesh_vertices, predicted_mesh_faces).write_obj(
        _join("predicted.obj")
    )
    Mesh(predicted_mesh_vertices_aligned, predicted_mesh_faces).write_obj(
        _join("predicted_aligned.obj")
    )

def compute_rigid_alignment(masked_gt_scan, grundtruth_landmark_points, 
                            predicted_mesh_vertices, predicted_mesh_faces, predicted_mesh_landmark_points, check_alignment_output_dir=None):
    """
    Computes the rigid alignment between the 
    :param masked_gt_scan: Masked face area mesh
    :param grundtruth_landmark_points: A 7 x 3 list with annotations of the ground truth scan.
    :param predicted_mesh_vertices: An m x 3 numpy array of vertices from a predicted mesh.
    :param predicted_mesh_faces: A k x 3 numpy array of vertex indices composing the predicted mesh.
    :param predicted_mesh_landmark_points: A 7 x 3 list containing the annotated 3D point locations in the predicted mesh.
    """

    grundtruth_landmark_points = np.array(grundtruth_landmark_points)
    predicted_mesh_landmark_points = np.array(predicted_mesh_landmark_points)

    d, Z, tform = procrustes(grundtruth_landmark_points, predicted_mesh_landmark_points, scaling=True, reflection='best')

    # Use tform to transform all vertices in predicted_mesh_vertices to the ground truth reference space:
    predicted_mesh_vertices_aligned = tform['scale']*(tform['rotation'].T.dot(predicted_mesh_vertices.T).T) + tform['translation']

    # Refine rigid alignment
    s , R, t = rigid_scan_2_mesh_alignment(masked_gt_scan, Mesh(predicted_mesh_vertices_aligned, predicted_mesh_faces))
    predicted_mesh_vertices_aligned = s*(R.dot(predicted_mesh_vertices_aligned.T)).T + t

    if check_alignment_output_dir is not None:
        write_alignment_check_files(
            check_alignment_output_dir,
            masked_gt_scan,
            predicted_mesh_vertices,
            predicted_mesh_vertices_aligned,
            predicted_mesh_faces,
        )

    return (predicted_mesh_vertices_aligned, masked_gt_scan)

def compute_errors(groundtruth_vertices, groundtruth_faces, grundtruth_landmark_points, predicted_mesh_vertices,
                    predicted_mesh_faces, predicted_mesh_landmark_points, check_alignment_output_dir=None):
    """
    This script computes the reconstruction error between an input mesh and a ground truth mesh.
    :param groundtruth_vertices: An n x 3 numpy array of vertices from a ground truth scan.
    :param grundtruth_landmark_points: A 7 x 3 list with annotations of the ground truth scan.
    :param predicted_mesh_vertices: An m x 3 numpy array of vertices from a predicted mesh.
    :param predicted_mesh_faces: A k x 3 numpy array of vertex indices composing the predicted mesh.
    :param predicted_mesh_landmark_points: A 7 x 3 list containing the annotated 3D point locations in the predicted mesh.
    :param check_alignment_output_dir [optional]: If provided, will write aligned and GT mesh to this dir for verification.
    :return: A list of distances (errors)
    """

    # Crop face scan
    masked_gt_scan = crop_face_scan(groundtruth_vertices, groundtruth_faces, grundtruth_landmark_points)

    # Rigidly align predicted mesh with the ground truth scan
    predicted_mesh_vertices_aligned, masked_gt_scan = compute_rigid_alignment(  masked_gt_scan, grundtruth_landmark_points, 
                                                                                predicted_mesh_vertices, predicted_mesh_faces, 
                                                                                predicted_mesh_landmark_points,
                                                                                check_alignment_output_dir)

    # Compute error
    sampler = sample_from_mesh(masked_gt_scan, sample_type='vertices')
    s2m = ScanToMesh(masked_gt_scan, predicted_mesh_vertices_aligned, predicted_mesh_faces, scan_sampler=sampler, signed=False, normalize=False)
    return s2m.r

def procrustes(X, Y, scaling=True, reflection='best'):
    """
    Taken from https://github.com/patrikhuber/fg2018-competition

    A port of MATLAB's `procrustes` function to Numpy.
    Code from: https://stackoverflow.com/a/18927641.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n, m = X.shape
    ny, my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0 ** 2.).sum()
    ssY = (Y0 ** 2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m - my)), 0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA ** 2

        # transformed coords
        Z = normX * traceTA * np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my, :]
    c = muX - b * np.dot(muY, T)

    # transformation values
    tform = {'rotation': T, 'scale': b, 'translation': c}

    return d, Z, tform
