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

def cumulative_error(errors, nbins=100000):
    errors = errors.ravel()
    values, base = np.histogram(errors, bins=nbins) #values, base = np.histogram(1000*errors, bins=nbins)
    cumsum = np.array(np.cumsum(values), dtype=float)
    cumulative = 100.0*cumsum/float(errors.shape[0])
    return (base[:-1], cumulative)

def generating_cumulative_error_plots():
    """
    Generate cumulative error plots for a list of errors. 
    """

    # List of method identifiers, used as method name within the polot
    method_identifiers = []
    # List of paths to the error files (must be of same order than the method identifiers)
    method_error_fnames = []

    # Output cumulative error image filename
    out_fname = ''

    method_errors = []
    for fname in method_error_fnames:
        method_errors.append(np.load(fname, allow_pickle=True, encoding="latin1").item()['computed_distances'])

    for i in range(len(method_identifiers)):
        print('%s - median: %f, mean: %f, std: %f' % (method_identifiers[i], np.median(np.hstack(method_errors[i])), np.mean(np.hstack(method_errors[i])), np.std(np.hstack(method_errors[i]))))

    cumulative_errors = []
    for error in method_errors:
        cumulative_errors.append(cumulative_error(np.hstack(error)))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim([0, 8])
    ax.set_xticks(np.arange(0, 8, 1.0))
    ax.set_ylim([0, 100])
    ax.set_yticks(np.arange(0, 101, 20.0))

    for i, method_id in enumerate(method_identifiers):
        plt.plot(cumulative_errors[i][0], cumulative_errors[i][1], label = method_id)

    plt.xlabel('Error [mm]', fontsize=12)
    plt.ylabel('Percentage', fontsize=12)
    lgd = ax.legend(loc='lower right')
    plt.savefig(out_fname)

def compute_error_metric(gt_path, gt_lmk_path, predicted_mesh_path, predicted_lmk_path):
    groundtruth_scan = Mesh(filename=gt_path)
    grundtruth_landmark_points = load_pp(gt_lmk_path)
    predicted_mesh = predicted_mesh_path
    predicted_mesh_landmark_points = predicted_lmk_path
    distances =  s2m_opt.compute_errors(groundtruth_scan.v, groundtruth_scan.f, grundtruth_landmark_points, predicted_mesh.v,
                                          predicted_mesh.f, predicted_mesh_landmark_points)
    return np.stack(distances)

def metric_computation():
    """
    Compute the NoW 3D reconstruction error. 
    """

    # Path of the meshes predicted for the NoW challenge
    predicted_mesh_folder = ''
    # Identifier of the method which is used as filename for the output error file
    method_identifier = ''

    # Output path for the computed error
    error_out_path = ''

    # If empty, error across all challenges (i.e. multiview_neutral, multiview_expressions, multiview_occlusions, or selfie) is computed. 
    # If challenge \in {'multiview_neutral', 'multiview_expressions', 'multiview_occlusions', 'selfie'}, only results of the specified challenge are considered
    challenge = ''

    # Path of the ground truth scans
    gt_mesh_folder = ''
    # Path of the ground truth scan landmarks
    gt_lmk_folder = ''
    # Image list, for the NoW validation data, this file can be downloaded from here: https://ringnet.is.tue.mpg.de/downloads
    imgs_list = ''

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

        gt_mesh_path = glob(gt_mesh_folder + subject + '/' + '*.obj')[0]
        gt_lmk_path = (glob(gt_lmk_folder + subject + '/' + '*.pp')[0])

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
    metric_computation()

    # Generate cumulative error plots for multiple error files
    # generating_cumulative_error_plots()
