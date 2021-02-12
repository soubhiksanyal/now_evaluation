import numpy as np
from math import sqrt
import chumpy as ch
from psbody.mesh import Mesh
from smpl_webuser.posemapper import Rodrigues
from sbody.alignment.objectives import sample_from_mesh
from sbody.mesh_distance import ScanToMesh
from psbody.mesh.meshviewer import MeshViewer
from scipy.sparse.linalg import cg


def rigid_scan_2_mesh_alignment(scan, mesh):
    options = {'sparse_solver': lambda A, x: cg(A, x, maxiter=2000)[0]}
    options['disp'] = 1.0
    options['delta_0'] = 0.1
    options['e_3'] = 1e-4
    # scan_fname = '/ps/dynamics10/FaceTalk/FaceTalk_170912_03278_TA/sentence09/meshes/sentence09.000001.obj'
    # mesh_fname = '/ps/body/projects/faces/timo/new_template_revised/body_template_mesh/head_eyes_6.obj'
    # scan = Mesh(filename=scan_fname)
    # mesh = Mesh(filename=mesh_fname)
    # mesh.v[:] *= 1000 #our scans are in mm, our model meshes in m
    # only use this if no initial rigid alignment from landmarks is given
    # mesh.v[:] = mesh.v - (np.mean(mesh.v, axis=0) - np.mean(scan.v, axis=0))
    s = ch.ones(1)
    r = ch.zeros(3)
    R = Rodrigues(r)
    t = ch.zeros(3)
    trafo_mesh = s*(R.dot(mesh.v.T)).T + t
    # num_samples = 1000
    # sample_type = 'vertices'
    # sample_seed = 0
    # scan_sampler = sample_from_mesh(scan, num_samples=num_samples, sample_type=sample_type, seed=sample_seed)
    sampler = sample_from_mesh(scan, sample_type='vertices')
    s2m = ScanToMesh(scan, trafo_mesh, mesh.f, scan_sampler=sampler, signed=False, normalize=False)
    #Visualization code
    # mv = MeshViewer()
    # mv.set_static_meshes([scan])
    # tmp_mesh = Mesh(trafo_mesh.r, mesh.f)
    # tmp_mesh.set_vertex_colors('light sky blue')
    # mv.set_dynamic_meshes([tmp_mesh])
    # def on_show(_):
    #     tmp_mesh = Mesh(trafo_mesh.r, mesh.f)
    #     tmp_mesh.set_vertex_colors('light sky blue')
    #     mv.set_dynamic_meshes([tmp_mesh])
    # ch.minimize(fun={'dist': s2m, 's_reg': 0.001*(s-0)}, x0=[s, r, t], callback=on_show, options=options)
    ch.minimize(fun={'dist': s2m, 's_reg': 0.001*(s-0)}, x0=[s, r, t], options=options)
    # ch.minimize(fun={'dist': s2m, 's_reg': 0.001*(s-0)}, x0=[s, r, t], callback=on_show, options=options)
    # import pdb; pdb.set_trace()
    return s,Rodrigues(r),t


def compute_errors(groundtruth_vertices, groundtruth_faces, grundtruth_landmark_points, predicted_mesh_vertices,
                                       predicted_mesh_faces,
                                       predicted_mesh_landmark_points):
    """
    This script computes the reconstruction error between an input mesh and a ground truth mesh.
    :param groundtruth_vertices: An n x 3 numpy array of vertices from a ground truth scan.
    :param grundtruth_landmark_points: A 7 x 3 list with annotations of the ground truth scan.
    :param predicted_mesh_vertices: An m x 3 numpy array of vertices from a predicted mesh.
    :param predicted_mesh_faces: A k x 3 numpy array of vertex indices composing the predicted mesh.
    :param predicted_mesh_landmark_points: A 7 x 3 list containing the annotated 3D point locations in the predicted mesh.
    :return: A list of distances (errors)
    """

    # from body.alignment.objectives import sample_from_mesh
    # from psbody.mesh import Mesh
    # from body.ch.mesh_distance import ScanToMesh

    # Do procrustes based on the 7 points:

    d, Z, tform = procrustes(np.array(grundtruth_landmark_points), np.array(predicted_mesh_landmark_points),
                             scaling=True, reflection='best')
    # Use tform to transform all vertices in predicted_mesh_vertices to the ground truth reference space:
    predicted_mesh_vertices_aligned = []
    # import ipdb; ipdb.set_trace()
    for v in predicted_mesh_vertices:
        s = tform['scale']
        R = tform['rotation']
        t = tform['translation']
        transformed_vertex = s * np.dot(v, R) + t
        predicted_mesh_vertices_aligned.append(transformed_vertex)

    # Compute the mask: A circular area around the center of the face. Take the nose-bottom and go upwards a bit:
    nose_bottom = np.array(grundtruth_landmark_points[4])
    nose_bridge = (np.array(grundtruth_landmark_points[1]) + np.array(
        grundtruth_landmark_points[2])) / 2  # between the inner eye corners
    face_centre = nose_bottom + 0.3 * (nose_bridge - nose_bottom)
    # Compute the radius for the face mask:
    outer_eye_dist = np.linalg.norm(np.array(grundtruth_landmark_points[0]) - np.array(grundtruth_landmark_points[3]))
    nose_dist = np.linalg.norm(nose_bridge - nose_bottom)
    # mask_radius = 1.2 * (outer_eye_dist + nose_dist) / 2
    mask_radius = 1.4 * (outer_eye_dist + nose_dist) / 2

    # Find all the vertex indices in the ground truth scan that lie within the mask area:
    vertex_indices_mask = []  # vertex indices in the source mesh (the ground truth scan)
    points_on_groundtruth_scan_to_measure_from = []
    for vertex_idx, vertex in enumerate(groundtruth_vertices):
        dist = np.linalg.norm(vertex - face_centre) # We use Euclidean distance for the mask area for now.
        if dist <= mask_radius:
            vertex_indices_mask.append(vertex_idx)
            points_on_groundtruth_scan_to_measure_from.append(vertex)
    assert len(vertex_indices_mask) == len(points_on_groundtruth_scan_to_measure_from)

    predicted_mesh_vertices_aligned_array = np.array(predicted_mesh_vertices_aligned)
    aligned_mesh = Mesh(v=predicted_mesh_vertices_aligned_array, f=predicted_mesh_faces)

    gt_scan = Mesh(v=groundtruth_vertices, f=groundtruth_faces)
    gt_scan.keep_vertices(vertex_indices_mask)

    s , r, t = rigid_scan_2_mesh_alignment(gt_scan, aligned_mesh)
    trafo_mesh = s*(r.dot(aligned_mesh.v.T)).T + t

    # Check the mesh alignment is correct or not
    # gt_scan.write_obj('gt_scan_val.obj')
    # aligned_mesh.write_obj('aligned_mesh_before_scan2mesh_alignment.obj')
    # rigidly_aligned_mesh = Mesh(v=trafo_mesh, f=aligned_mesh.f)
    # rigidly_aligned_mesh.write_obj('aligned_mesh.obj')

    sampler = sample_from_mesh(gt_scan, sample_type='vertices')
    s2m = ScanToMesh(gt_scan, trafo_mesh, aligned_mesh.f, scan_sampler=sampler, signed=False, normalize=False)
    # s2m = ScanToMesh(gt_scan, aligned_mesh.v, aligned_mesh.f, signed=False, normalize=False)
    # import ipdb; ipdb.set_trace()
    distances = s2m.r#[sqrt(d2) for d2 in squared_distances]

    return distances
    

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
