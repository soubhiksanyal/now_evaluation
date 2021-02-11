import os
from glob import glob
import sys
# import pymesh
import numpy as np
import scan2mesh_computations as s2m_opt
import matplotlib.pyplot as plt
from psbody.mesh import Mesh
from experiments.tianye.align_d3dfacs.find_opt_landmark_indexes import solve_barycentric_coordinates_for_closest_points, mesh_points_by_barycentric_coordinates
import chumpy as ch
from psbody.smpl.serialization import load_model
from psbody.smpl.verts import verts_decorated
from plyfile import PlyData, PlyElement

def load_pp(fname):
    lamdmarks = np.zeros([7,3]).astype(np.float32)
    # import ipdb; ipdb.set_trace()
    with open(fname, 'r') as f:
        lines = f.readlines()
        for j in xrange(8,15): # for j in xrange(9,15):
            # import ipdb; ipdb.set_trace()
            line_contentes = lines[j].split(' ')
            # Check the .pp file to get to accurately pickup the columns for x , y and z coordinates
            for i in xrange(len(line_contentes)):
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
    for i in xrange(len(lines)): # For Jiaxiang_Shang
        line.append(lines[i].split(' '))
    # import ipdb; ipdb.set_trace()
    landmarks = np.array(line, dtype=np.float32)
    lmks = landmarks
    return lmks

def read_ply(fname):
    plydata = PlyData.read(fname)
    v = np.stack([plydata['vertex']['x'],plydata['vertex']['y'],plydata['vertex']['z']],axis=1)
    f = np.stack( plydata['face']['vertex_indices'].ravel())
    mesh = Mesh(v=v,f=f)
    return mesh

def cumulative_error(errors, nbins=100000):
    errors = errors.ravel()
    values, base = np.histogram(errors, bins=nbins) #values, base = np.histogram(1000*errors, bins=nbins)
    cumsum = np.array(np.cumsum(values), dtype=float)
    cumulative = 100.0*cumsum/float(errors.shape[0])
    return (base[:-1], cumulative)

def generating_cumulative_error_plots():
    prnet_with_crop = np.load('/ps/scratch/ssanyal/NoW_Dataset/comaprisons/NoW_website_validation/recheck_cvpr_submission/PRNet/selfie.npy')
    extreme_3D_withcrop = np.load('/ps/scratch/ssanyal/NoW_Dataset/comaprisons/NoW_website_validation/recheck_cvpr_submission/3dmm_cnn/selfie.npy')
    our_pred = np.load("/ps/scratch/ssanyal/NoW_Dataset/comaprisons/NoW_website_validation/recheck_cvpr_submission/HMR_VGG2Ring_contranstive_R6_npy_resnet_fc3_dropout_Elr1e-04_kp-weight60_shp-weight1_encod_512_512_decode_l1_shp100exp50_nostg3_invrt_config_resfixed_alpha_0.5_srw_1e4_erw_1e4_scratch_batch32_68641/selfie.npy")

    prnet_with_crop = prnet_with_crop[()]
    extreme_3D_withcrop = extreme_3D_withcrop[()]
    our_pred = our_pred[()]
    
    prnet_with_crop_distance_metric = prnet_with_crop['computed_distances']
    extreme_3D_withcrop_distance_metric = extreme_3D_withcrop['computed_distances']
    our_pred_distance_metric = our_pred['computed_distances']#our_pred['distance_metric']#

    prnet_with_crop_cumul_error = cumulative_error(np.hstack(prnet_with_crop_distance_metric))
    extreme_3D_withcrop_cumul_error = cumulative_error(np.hstack(extreme_3D_withcrop_distance_metric))
    our_pred_cumul_error = cumulative_error(np.hstack(our_pred_distance_metric))

    print 'PRNet with crop median: ', np.median(np.hstack(prnet_with_crop_distance_metric))
    print 'Hassner with crop median: ', np.median(np.hstack(extreme_3D_withcrop_distance_metric))
    print 'Our without crop median: ', np.median(np.hstack(our_pred_distance_metric))

    print 'PRNet with crop std: ', np.std(np.hstack(prnet_with_crop_distance_metric))
    print 'Hassner with crop std: ', np.std(np.hstack(extreme_3D_withcrop_distance_metric))
    print 'Our without crop std: ', np.std(np.hstack(our_pred_distance_metric))

    print 'PRNet with crop average: ', np.average(np.hstack(prnet_with_crop_distance_metric))
    print 'Hassner with crop average: ', np.average(np.hstack(extreme_3D_withcrop_distance_metric))
    print 'Our without crop average: ', np.average(np.hstack(our_pred_distance_metric))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim([0, 8])
    ax.set_xticks(np.arange(0, 8, 1.0))
    ax.set_ylim([0, 100])
    ax.set_yticks(np.arange(0, 101, 20.0))
    plt.plot(prnet_with_crop_cumul_error[0], prnet_with_crop_cumul_error[1],'r', label = 'PRNet [ECCV 2018]')
    plt.plot(extreme_3D_withcrop_cumul_error[0], extreme_3D_withcrop_cumul_error[1], 'g', label = '3DMM-CNN [CVPR 2017]')
    plt.plot(our_pred_cumul_error[0], our_pred_cumul_error[1], 'b', label = 'Ours')
    plt.xlabel('Error [mm]', fontsize=12)
    plt.ylabel('Percentage', fontsize=12)
    lgd = ax.legend(loc='lower right')
    plt.savefig('/ps/scratch/ssanyal/NoW_Dataset/comaprisons/NoW_website_validation/recheck_cvpr_submission/selfie.png')

def compute_error_metric(gt_path, gt_lmk_path, predicted_mesh_path, predicted_lmk_path):
    groundtruth_scan = Mesh(filename=gt_path)
    grundtruth_landmark_points = load_pp(gt_lmk_path)
    predicted_mesh = predicted_mesh_path
    predicted_mesh_landmark_points = predicted_lmk_path
    distances =  s2m_opt.compute_errors(groundtruth_scan.v, groundtruth_scan.f, grundtruth_landmark_points, predicted_mesh.v,
                                          predicted_mesh.f, predicted_mesh_landmark_points)
    return np.stack(distances)


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def metric_computation():
    predicted_mesh_folder =  '/ps/scratch/face2d3d/texture_in_the_wild_code/NoW_validation/results/previous_works_validation_results/msra/NoW_test_pred_meshes_original_BFM/'# '/ps/project/face2d3d/comparisons/NoW_website/will_smith_york/NoW_result_n/' #'/ps/face2d3d/face2mesh/results/stirling_comparison_new/'
    #### imgs_list = '/ps/face2d3d/comparisons/FAMOS_results/image_path_original_path.txt'
    # imgs_list = '/ps/scratch/ssanyal/NoW_Dataset/comaprisons/NoW_website/image_paths.txt'
    imgs_list = '/ps/project/now_challenge/Now_Test_set/image_paths.txt'
    
    gt_mesh_folder = '/ps/project/face2d3d/benchmark_dataset/extended_benchmark/release_soubhik/full_data/scans/'
    gt_lmk_folder = '/ps/project/face2d3d/benchmark_dataset/extended_benchmark/release_soubhik/full_data/scans_lmks/'
    
    tot_dist = 0.0
    img_count = 0
    distance_metric = []
    
    with open(imgs_list,'r') as f:
        lines = f.read().splitlines()

    for i in xrange(len(lines)):#xrange(5): #
        print i
        subject = lines[i].split('/')[-3]
        experiments = lines[i].split('/')[-2]
        filename = lines[i].split('/')[-1]
        
        predicted_mesh = predicted_mesh_folder + subject + '/' + experiments + '/' + filename[:-4] + '.obj'
        predicted_landmarks_path = predicted_mesh_folder + subject + '/' + experiments + '/' + filename[:-4] + '.npy'
        
        gt_mesh_path = glob(gt_mesh_folder + subject + '/' + '*.obj')[0]
        gt_lmk_path = (glob(gt_lmk_folder + subject + '/' + '*.pp')[0])

        mesh = Mesh(filename=predicted_mesh)#read_ply(predicted_mesh)#
        predicted_lmks = np.load(predicted_landmarks_path)#load_txt_will_smith(predicted_landmarks_path)#load_txt(predicted_landmarks_path)#

        distances = compute_error_metric(gt_mesh_path, gt_lmk_path, mesh, predicted_lmks)
        print distances
        
        distance_metric.append(distances)
    computed_distances = {'computed_distances': distance_metric}
    np.save("/ps/project/now_challenge/Now_Test_set/comparisons/msra_computed_distances.npy", computed_distances)

def different_challenges():
    imgs_list = '/ps/scratch/ssanyal/NoW_Dataset/comaprisons/NoW_website_validation/recheck_cvpr_submission/image_paths.txt'
    # imgs_list = '/ps/project/face2d3d/benchmark_dataset/extended_benchmark/release_soubhik/NoW_validation/resized_img_paths.txt'
    tot_dist = 0.0
    img_count = 0
    distance_metric = []
    pre_dist = np.load('/ps/scratch/ssanyal/NoW_Dataset/comaprisons/NoW_website_validation/recheck_cvpr_submission/HMR_VGG2Ring_contranstive_R6_npy_resnet_fc3_dropout_Elr1e-04_kp-weight60_shp-weight1_encod_512_512_decode_l1_shp100exp50_nostg3_invrt_config_resfixed_alpha_0.5_srw_1e4_erw_1e4_scratch_batch32_68641/ring_distance_full.npy')
    pre_dist = pre_dist[()]
    pre_dist = pre_dist['distance_metric'] #pre_dist['computed_distances']#
    with open(imgs_list,'r') as f:
        lines = f.read().splitlines()

    for i in xrange(len(lines)):#xrange(1): #
        print i
        print lines[i]
        subject = lines[i].split('/')[-3]
        experiments = lines[i].split('/')[-2]
        filename = lines[i].split('/')[-1]
        if experiments=='selfie': #multiview_neutral #multiview_expressions #multiview_occlusions #selfie
            pass
        else:
            continue
        # import ipdb; ipdb.set_trace()
        distances = pre_dist[i]
        print distances
        distance_metric.append(distances)
    computed_distances = {'computed_distances': distance_metric}
    np.save("/ps/scratch/ssanyal/NoW_Dataset/comaprisons/NoW_website_validation/recheck_cvpr_submission/HMR_VGG2Ring_contranstive_R6_npy_resnet_fc3_dropout_Elr1e-04_kp-weight60_shp-weight1_encod_512_512_decode_l1_shp100exp50_nostg3_invrt_config_resfixed_alpha_0.5_srw_1e4_erw_1e4_scratch_batch32_68641/selfie.npy", computed_distances)


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    metric_computation()
    # generating_cumulative_error_plots()
    # different_challenges()
