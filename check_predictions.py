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
import sys
import numpy as np
from scan2mesh_computations import crop_face_scan as crop_face_scan
from scan2mesh_computations import compute_rigid_alignment as compute_rigid_alignment
from psbody.mesh import Mesh

def load_pp(fname):
    lamdmarks = np.zeros([7,3]).astype(np.float32)
    # import ipdb; ipdb.set_trace()
    with open(fname, 'r') as f:
        lines = f.readlines()
        for j in range(8,15): # for j in xrange(9,15):
            # import ipdb; ipdb.set_trace()
            line_contentes = lines[j].split(' ')
            # Check the .pp file to get to accurately pickup the columns for x , y and z coordinates
            for i in range(len(line_contentes)):
                if line_contentes[i].split('=')[0] == 'x':
                    x_content = float((line_contentes[i].split('=')[1]).split('"')[1])
                elif line_contentes[i].split('=')[0] == 'y':
                    y_content = float((line_contentes[i].split('=')[1]).split('"')[1])
                elif line_contentes[i].split('=')[0] == 'z':
                    z_content = float((line_contentes[i].split('=')[1]).split('"')[1])
                else:
                    pass
            lamdmarks[j-8, :] = (np.array([x_content, y_content, z_content]).astype(np.float32))
            # import ipdb; ipdb.set_trace()
    return lamdmarks

def load_txt(fname):
    landmarks = []#np.zeros([7,3]).astype(np.float32)
    with open(fname, 'r') as f:
        lines = f.read().splitlines()
    # import ipdb; ipdb.set_trace()
    line = []
    for i in range(len(lines)): # For Jiaxiang_Shang
        line.append(lines[i].split(' '))
    # import ipdb; ipdb.set_trace()
    landmarks = np.array(line, dtype=np.float32)
    lmks = landmarks
    return lmks

def save_obj(path, v, f, c):
    with open(path, 'w') as file:
        for i in range(len(v)):
            file.write('v %f %f %f %f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2], c[i, 0], c[i, 1], c[i, 2]))
        file.write('\n')

        for i in range(len(f)):
            file.write('f %d %d %d\n' % (f[i, 0], f[i, 1], f[i, 2]))
    file.close()


def check_mesh_import_export(pred_mesh_filename):
    """
    Import and export predicted mesh to check if mesh is properly loaded
    """
    if not os.path.exists(pred_mesh_filename):
        print('Predicted mesh not found - %s' % pred_mesh_filename)
        return

    # Load and export the predicted mesh
    predicted_mesh = Mesh(filename=pred_mesh_filename)
    predicted_mesh.write_obj('./predicted_mesh_export.obj')    

def check_mesh_alignment(pred_mesh_filename, pred_lmk_filename, gt_mesh_filename, gt_lmk_filename):
    """
    Compute rigid alignment between the predicted mesh and the ground truth scan.
    :param pred_mesh_filename: path of the predicted mesh to be aligned
    :param pred_lmk_filename: path of the landmarks of the predicted mesh
    :param gt_mesh_filename: path of the ground truth scan
    :param gt_lmk_filename: path of the ground truth landmark file
    """

    if not os.path.exists(pred_mesh_filename):
        print('Predicted mesh not found - %s' % pred_mesh_filename)
        return
    if not os.path.exists(pred_lmk_filename):
        print('Predicted mesh landmarks not found - %s' % pred_lmk_filename)
        return  
    if not os.path.exists(gt_mesh_filename):
        print('Ground truth scan not found - %s' % gt_mesh_filename)
        return
    if not os.path.exists(gt_lmk_filename):
        print('Ground truth scan landmarks not found - %s' % gt_lmk_filename)
        return  

    # Load ground truth data
    groundtruth_scan = Mesh(filename=gt_mesh_filename)
    grundtruth_landmark_points = load_pp(gt_lmk_filename)

    # Load predicted data
    predicted_mesh = Mesh(filename=pred_mesh_filename)
    pred_lmk_ext = os.path.splitext(pred_lmk_filename)[-1]
    if pred_lmk_ext =='.txt':
        predicted_lmks = load_txt(pred_lmk_filename)
    elif pred_lmk_ext == '.npy':
        predicted_lmks = np.load(pred_lmk_filename)
    else:
        print('Unable to load predicted landmarks, must be of format txt or npy')
        return

    # Crop face scan
    masked_gt_scan = crop_face_scan(groundtruth_scan.v, groundtruth_scan.f, grundtruth_landmark_points)

    # Rigidly align predicted mesh with the ground truth scan
    predicted_mesh_vertices_aligned, masked_gt_scan = compute_rigid_alignment(  masked_gt_scan, grundtruth_landmark_points, 
                                                                                predicted_mesh.v, predicted_mesh.f, predicted_lmks)

    # Output cropped scan
    masked_gt_scan.write_obj('./cropped_scan.obj')

    # Output cropped aligned mesh
    Mesh(predicted_mesh_vertices_aligned, predicted_mesh.f).write_obj('./predicted_mesh_aligned.obj')


def main(argv):
    if len(argv) < 2:
        return

    pred_mesh_filename = argv[1]
    if len(argv) == 2:
        check_mesh_import_export(pred_mesh_filename)
    elif len(argv) == 5:
        pred_lmk_filename = argv[2]
        gt_mesh_filename = argv[3]
        gt_lmk_filename = argv[4]
        check_mesh_alignment(pred_mesh_filename, pred_lmk_filename, gt_mesh_filename, gt_lmk_filename)
    else:
        print('Number of parameters wrong - %d != (%d or %d)' % (len(argv), 2, 5))
        return

if __name__ == '__main__':
    main(sys.argv)