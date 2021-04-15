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
from glob import glob
import sys
import numpy as np
import scan2mesh_computations as s2m_opt
import matplotlib.pyplot as plt
from psbody.mesh import Mesh
import chumpy as ch

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

def compute_error_metric(gt_path, gt_lmk_path, predicted_mesh_path, predicted_lmk_path):
    groundtruth_scan = Mesh(filename=gt_path)
    grundtruth_landmark_points = load_pp(gt_lmk_path)
    predicted_mesh = predicted_mesh_path
    predicted_mesh_landmark_points = predicted_lmk_path
    distances =  s2m_opt.compute_errors(groundtruth_scan.v, groundtruth_scan.f, grundtruth_landmark_points, predicted_mesh.v,
                                          predicted_mesh.f, predicted_mesh_landmark_points)
    return np.stack(distances)

def metric_computation(dataset_folder,
                       predicted_mesh_folder,
                       gt_mesh_folder=None,
                       gt_lmk_folder=None,
                       image_set='val',
                       imgs_list=None,
                       challenge='',
                       error_out_path=None,
                       method_identifier=''):
    '''
    :param dataset_folder: Path to root of the dataset, which contains images, scans and lanmarks
    :param predicted_mesh_folder: Path to predicted restuls to be evaluated
    :param gt_mesh_folder: Optional. Path to the GT scans. If not specified, it will be looked for in the dataset folder
    :param gt_lmk_folder: Optional. Path to the GT landmarks. If not specified, it will be looked for in the dataset folder
    :param image_set: 'val' or 'test'. This specifies which images will be used. Ignored if imgs_list is specified
    :param imgs_list: Optional. Path to file with image list to be used
    :param challenge:
    :param error_out_path: Optional. Path to results folder. If None, results will be saved to predicted_mesh_folder
    :param method_identifier: Optional. Will be used to name the output file
    '''
    # Path of the meshes predicted for the NoW challenge
    if not os.path.isdir(predicted_mesh_folder):
        raise RuntimeError(f"Predicted mesh folder does not exist. '{predicted_mesh_folder}'")

    if (imgs_list is None and gt_mesh_folder is None and gt_lmk_folder is None) and not os.path.isdir(dataset_folder):
        raise RuntimeError(f"Dataset folder does not exist. '{dataset_folder}'")

    # Image list, for the NoW validation data, this file can be downloaded from here: https://ringnet.is.tue.mpg.de/downloads
    if imgs_list is None or imgs_list == '':
        if image_set == 'val':
            imgs_list = os.path.join(dataset_folder, "imagepathsvalidation.txt")
        elif image_set == 'test':
            imgs_list = os.path.join(dataset_folder, "imagepathstest.txt")
        else:
            print(f"Invalid image set identifier '{image_set}'.")

    # Path of the ground truth scans
    gt_mesh_folder = gt_mesh_folder or os.path.join(dataset_folder, 'scans')
    # Path of the ground truth scan landmarks
    gt_lmk_folder = gt_lmk_folder or os.path.join(dataset_folder, 'scans_lmks_onlypp')

    # Output path for the computed error
    if error_out_path is None or error_out_path == '':
        error_out_path = os.path.join(predicted_mesh_folder, "results")

    os.makedirs(error_out_path, exist_ok=True)

    # If empty, error across all challenges (i.e. multiview_neutral, multiview_expressions, multiview_occlusions, or selfie) is computed. 
    # If challenge \in {'multiview_neutral', 'multiview_expressions', 'multiview_occlusions', 'selfie'}, only results of the specified challenge are considered
    valid_challenges = ['', 'multiview_neutral', 'multiview_expressions', 'multiview_occlusions', 'selfie']
    if challenge not in valid_challenges:
        raise ValueError(f"Invalid challenge value '{challenge}'. Accepted values are: {', '.join(valid_challenges)}")

    if not os.path.exists(predicted_mesh_folder):
        print('Predicted mesh path not found - %s' % predicted_mesh_folder)
        return
    if not os.path.exists(imgs_list):
        print('Image list not found - %s' % imgs_list)
        return
    if not os.path.exists(error_out_path):
        os.makedirs(error_out_path)

    distance_metric = []
    
    with open(imgs_list,'r') as f:
        lines = f.read().splitlines()

    for i in range(len(lines)):
        subject = lines[i].split('/')[-3]
        experiments = lines[i].split('/')[-2]
        filename = lines[i].split('/')[-1]
        
        if challenge != '':
            if experiments != challenge: 
                continue

        print('Processing %d of %d (%s, %s, %s)' % (i+1, len(lines), subject, experiments, filename))

        predicted_mesh_path = os.path.join(predicted_mesh_folder, subject, experiments, filename[:-4] + '.obj')
        predicted_landmarks_path_npy = os.path.join(predicted_mesh_folder, subject, experiments, filename[:-4] + '.npy')
        predicted_landmarks_path_txt = os.path.join(predicted_mesh_folder, subject, experiments, filename[:-4] + '.txt')       

        gt_mesh_path = glob(os.path.join(gt_mesh_folder, subject, '*.obj'))[0]
        gt_lmk_path = (glob(os.path.join(gt_lmk_folder, subject, '*.pp'))[0])

        if not os.path.exists(predicted_mesh_path):
            print('Predicted mesh not found - Resulting error is insufficient for comparison')
            print(predicted_mesh_path)
            continue
        if not os.path.exists(predicted_landmarks_path_npy) and not os.path.exists(predicted_landmarks_path_txt):
            print('Predicted mesh landmarks not found - Resulting error is insufficient for comparison')
            continue

        predicted_mesh = Mesh(filename=predicted_mesh_path)

        if os.path.exists(predicted_landmarks_path_npy):
            predicted_lmks = np.load(predicted_landmarks_path_npy)
        else:
            predicted_lmks = load_txt(predicted_landmarks_path_txt)

        distances = compute_error_metric(gt_mesh_path, gt_lmk_path, predicted_mesh, predicted_lmks)
        print(distances)
        
        distance_metric.append(distances)
    computed_distances = {'computed_distances': distance_metric}
    if challenge == '':
        np.save(os.path.join(error_out_path, '%s_computed_distances.npy' % method_identifier), computed_distances)
    else: 
        np.save(os.path.join(error_out_path, '%s_computed_distances_%s.npy' % (method_identifier, challenge)), computed_distances)

if __name__ == '__main__':
    # Computation of the s2m error

    nargs = len(sys.argv)

    dataset_folder = sys.argv[1]
    predicted_mesh_folder = sys.argv[2]

    if nargs > 3:
        image_set = sys.argv[3]
    else:
        image_set = 'val'

    if nargs > 4:
        error_out_path = sys.argv[4]
    else:
        error_out_path = None

    if nargs > 5:
        method_identifier = sys.argv[5]
    else:
        method_identifier = ''

    if nargs > 6:
        gt_mesh_folder = sys.argv[6]
    else:
        gt_mesh_folder = None

    if nargs > 7:
        gt_lmk_folder = sys.argv[7]
    else:
        gt_lmk_folder = None

    imgs_list = None
    challenge = ''

    metric_computation(dataset_folder, predicted_mesh_folder, gt_mesh_folder, gt_lmk_folder, image_set, imgs_list,
                       challenge=challenge,
                       error_out_path=error_out_path,
                       method_identifier=method_identifier
                       )
