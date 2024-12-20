"""
Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this computer program. 
Using this computer program means that you agree to the terms in the LICENSE file (https://ringnet.is.tue.mpg.de/license). 
Any use not explicitly granted by the LICENSE is prohibited.
Copyright 2020 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its 
Max Planck Institute for Intelligent Systems. All rights reserved.
More information about the NoW Challenge is available at https://ringnet.is.tue.mpg.de/challenge.
For comments or questions, please email us at ringnet@tue.mpg.de
"""

import os
import sys
import argparse
from glob import glob
import numpy as np
import chumpy as ch
import scan2mesh_computations as s2m_opt
import scan2mesh_computations_metrical as s2m_opt_metrical
import matplotlib.pyplot as plt
from psbody.mesh import Mesh
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import contextlib
import os
import functools


def suppress_output(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with open(os.devnull, "w") as devnull:
            old_stdout = os.sys.stdout
            old_stderr = os.sys.stderr
            os.sys.stdout = devnull
            os.sys.stderr = devnull
            try:
                return func(*args, **kwargs)
            finally:
                os.sys.stdout = old_stdout
                os.sys.stderr = old_stderr

    return wrapper


def clean_degenerated_faces(mesh):
    edge1 = mesh.v[mesh.f[:, 1]] - mesh.v[mesh.f[:, 0]]
    edge2 = mesh.v[mesh.f[:, 2]] - mesh.v[mesh.f[:, 0]]

    edge1_length = np.linalg.norm(edge1, axis=-1)[:, np.newaxis]
    edge1_length[edge1_length == 0.0] = 1.0
    edge2_length = np.linalg.norm(edge2, axis=-1)[:, np.newaxis]
    edge2_length[edge2_length == 0.0] = 1.0
    edge1 = edge1 / edge1_length
    edge2 = edge2 / edge2_length
    normals = np.cross(edge1, edge2, axis=-1)
    normal_length = np.linalg.norm(normals, axis=-1)
    invalid_faces = np.where(normal_length == 0.0)[0]
    valid_faces = np.setdiff1d(np.arange(mesh.f.shape[0]), invalid_faces)
    return Mesh(mesh.v, mesh.f[valid_faces, :])


def load_pp(fname):
    lamdmarks = np.zeros([7, 3]).astype(np.float32)
    # import ipdb; ipdb.set_trace()
    with open(fname, "r") as f:
        lines = f.readlines()
        for j in range(8, 15):  # for j in xrange(9,15):
            # import ipdb; ipdb.set_trace()
            line_contentes = lines[j].split(" ")
            # Check the .pp file to get to accurately pickup the columns for x , y and z coordinates
            for i in range(len(line_contentes)):
                if line_contentes[i].split("=")[0] == "x":
                    x_content = float((line_contentes[i].split("=")[1]).split('"')[1])
                elif line_contentes[i].split("=")[0] == "y":
                    y_content = float((line_contentes[i].split("=")[1]).split('"')[1])
                elif line_contentes[i].split("=")[0] == "z":
                    z_content = float((line_contentes[i].split("=")[1]).split('"')[1])
                else:
                    pass
            lamdmarks[j - 8, :] = np.array([x_content, y_content, z_content]).astype(
                np.float32
            )
            # import ipdb; ipdb.set_trace()
    return lamdmarks


def load_txt(fname):
    landmarks = []  # np.zeros([7,3]).astype(np.float32)
    with open(fname, "r") as f:
        lines = f.read().splitlines()
    # import ipdb; ipdb.set_trace()
    line = []
    for i in range(len(lines)):
        line.append(lines[i].split(" "))
    # import ipdb; ipdb.set_trace()
    landmarks = np.array(line, dtype=np.float32)
    lmks = landmarks
    return lmks


def compute_error_metric(
    gt_path,
    gt_lmk_path,
    predicted_mesh_path,
    predicted_lmk_path,
    predicted_mesh_unit=None,
    metrical_eval=False,
    check_alignment=False,
):
    groundtruth_scan = Mesh(filename=gt_path)
    grundtruth_landmark_points = load_pp(gt_lmk_path)
    predicted_mesh = predicted_mesh_path
    predicted_mesh_landmark_points = predicted_lmk_path
    if metrical_eval:
        distances = s2m_opt_metrical.compute_errors(
            groundtruth_scan.v,
            groundtruth_scan.f,
            grundtruth_landmark_points,
            predicted_mesh.v,
            predicted_mesh.f,
            predicted_mesh_landmark_points,
            predicted_mesh_unit,
            check_alignment,
        )
    else:
        distances = s2m_opt.compute_errors(
            groundtruth_scan.v,
            groundtruth_scan.f,
            grundtruth_landmark_points,
            predicted_mesh.v,
            predicted_mesh.f,
            predicted_mesh_landmark_points,
            check_alignment,
        )
    return np.stack(distances)


@suppress_output
def process_single_file(args):
    # Unpack the arguments
    (
        i,
        lines,
        predicted_mesh_folder,
        gt_mesh_folder,
        gt_lmk_folder,
        challenge,
        clean_mesh,
        predicted_mesh_unit,
        metrical_eval,
        check_alignment,
    ) = args

    subject = lines[i].split("/")[-3]
    experiments = lines[i].split("/")[-2]
    filename = lines[i].split("/")[-1]

    if challenge != "":
        if experiments != challenge:
            return None

    print(
        "Processing %d of %d (%s, %s, %s)"
        % (i + 1, len(lines), subject, experiments, filename)
    )

    predicted_mesh_path_obj = os.path.join(
        predicted_mesh_folder, subject, experiments, filename[:-4] + ".obj"
    )
    predicted_mesh_path_ply = os.path.join(
        predicted_mesh_folder, subject, experiments, filename[:-4] + ".ply"
    )
    predicted_landmarks_path_npy = os.path.join(
        predicted_mesh_folder, subject, experiments, filename[:-4] + ".npy"
    )
    predicted_landmarks_path_txt = os.path.join(
        predicted_mesh_folder, subject, experiments, filename[:-4] + ".txt"
    )

    gt_mesh_path = glob(os.path.join(gt_mesh_folder, subject, "*.obj"))[0]
    gt_lmk_path = glob(os.path.join(gt_lmk_folder, subject, "*.pp"))[0]

    if os.path.exists(predicted_mesh_path_obj):
        predicted_mesh = Mesh(filename=predicted_mesh_path_obj)
    elif os.path.exists(predicted_mesh_path_ply):
        predicted_mesh = Mesh(filename=predicted_mesh_path_ply)
    else:
        print(
            "Predicted mesh not found - Resulting error is insufficient for comparison"
        )
        print(predicted_mesh_path_obj + " " + predicted_mesh_path_ply)
        return None

    if not os.path.exists(predicted_landmarks_path_npy) and not os.path.exists(
        predicted_landmarks_path_txt
    ):
        print(
            "Predicted mesh landmarks not found - Resulting error is insufficient for comparison"
        )
        return None

    if clean_mesh:
        predicted_mesh = clean_degenerated_faces(predicted_mesh)

    if os.path.exists(predicted_landmarks_path_npy):
        predicted_lmks = np.load(predicted_landmarks_path_npy)
        if predicted_lmks.shape[0] == 1:
            predicted_lmks = predicted_lmks[0]
        assert (
            predicted_lmks.shape[0] == 7
        ), f"predicted_lmks.shape={predicted_lmks.shape} / should be (7,3)"
    else:
        predicted_lmks = load_txt(predicted_landmarks_path_txt)

    distances = compute_error_metric(
        gt_mesh_path,
        gt_lmk_path,
        predicted_mesh,
        predicted_lmks,
        predicted_mesh_unit=predicted_mesh_unit,
        metrical_eval=metrical_eval,
        check_alignment=check_alignment,
    )
    np.random.shuffle(distances)

    return distances


def metric_computation(
    dataset_folder,
    predicted_mesh_folder,
    gt_mesh_folder=None,
    gt_lmk_folder=None,
    image_set="val",
    imgs_list=None,
    challenge="",
    error_out_path=None,
    method_identifier="",
    predicted_mesh_unit=None,
    metrical_eval=False,
    clean_mesh=False,
    check_alignment=False,
):
    """
    :param dataset_folder: Path to root of the dataset, which contains images, scans and lanmarks
    :param predicted_mesh_folder: Path to predicted results to be evaluated
    :param gt_mesh_folder: Optional. Path to the GT scans. If not specified, it will be looked for in the dataset folder
    :param gt_lmk_folder: Optional. Path to the GT landmarks. If not specified, it will be looked for in the dataset folder
    :param image_set: 'val' or 'test'. This specifies which images will be used. Ignored if imgs_list is specified
    :param imgs_list: Optional. Path to file with image list to be used
    :param challenge:
    :param error_out_path: Optional. Path to results folder. If None, results will be saved to predicted_mesh_folder
    :param method_identifier: Optional. Will be used to name the output file
    """
    # Path of the meshes predicted for the NoW challenge
    if not os.path.isdir(predicted_mesh_folder):
        raise RuntimeError(
            f"Predicted mesh folder does not exist. '{predicted_mesh_folder}'"
        )

    if (
        imgs_list is None and gt_mesh_folder is None and gt_lmk_folder is None
    ) and not os.path.isdir(dataset_folder):
        raise RuntimeError(f"Dataset folder does not exist. '{dataset_folder}'")

    # Image list, for the NoW validation data, this file can be downloaded from here: https://ringnet.is.tue.mpg.de/downloads
    if imgs_list is None or imgs_list == "":
        if image_set == "val":
            imgs_list = os.path.join(dataset_folder, "imagepathsvalidation.txt")
        elif image_set == "test":
            imgs_list = os.path.join(dataset_folder, "imagepathstest.txt")
        else:
            print(f"Invalid image set identifier '{image_set}'.")

    # Path of the ground truth scans
    gt_mesh_folder = gt_mesh_folder or os.path.join(dataset_folder, "scans")
    # Path of the ground truth scan landmarks
    gt_lmk_folder = gt_lmk_folder or os.path.join(dataset_folder, "scans_lmks_onlypp")

    # Output path for the computed error
    if error_out_path is None or error_out_path == "":
        error_out_path = os.path.join(predicted_mesh_folder, "results")

    os.makedirs(error_out_path, exist_ok=True)

    # If empty, error across all challenges (i.e. multiview_neutral, multiview_expressions, multiview_occlusions, or selfie) is computed.
    # If challenge \in {'multiview_neutral', 'multiview_expressions', 'multiview_occlusions', 'selfie'}, only results of the specified challenge are considered
    valid_challenges = [
        "",
        "multiview_neutral",
        "multiview_expressions",
        "multiview_occlusions",
        "selfie",
    ]
    if challenge not in valid_challenges:
        raise ValueError(
            f"Invalid challenge value '{challenge}'. Accepted values are: {', '.join(valid_challenges)}"
        )

    if not os.path.exists(predicted_mesh_folder):
        print("Predicted mesh path not found - %s" % predicted_mesh_folder)
        return
    if not os.path.exists(imgs_list):
        print("Image list not found - %s" % imgs_list)
        return
    if not os.path.exists(error_out_path):
        os.makedirs(error_out_path)

    distance_metric = []

    with open(imgs_list, "r") as f:
        lines = f.read().splitlines()

    num_missing_files = 0

    # Prepare arguments for parallel processing
    args_list = [
        (
            i,
            lines,
            predicted_mesh_folder,
            gt_mesh_folder,
            gt_lmk_folder,
            challenge,
            clean_mesh,
            predicted_mesh_unit,
            metrical_eval,
            check_alignment,
        )
        for i in range(len(lines))
    ]

    # Use ProcessPoolExecutor for parallel processing
    nproc = os.cpu_count() * 4  # nproc * 4 seemed optimal
    with ProcessPoolExecutor(max_workers=nproc) as executor:
        future_to_result = {
            executor.submit(process_single_file, args): i
            for i, args in enumerate(args_list)
        }

        # Use tqdm to show progress bar
        for future in tqdm(
            as_completed(future_to_result),
            total=len(future_to_result),
            desc="Processing",
            unit="file",
        ):
            result = future.result()
            if result is not None:
                distance_metric.append(result)
            else:
                num_missing_files += 1

    computed_distances = {
        "computed_distances": distance_metric,
        "num_missing_files": num_missing_files,
    }
    if challenge == "":
        np.save(
            os.path.join(
                error_out_path, "%s_computed_distances.npy" % method_identifier
            ),
            computed_distances,
        )
    else:
        np.save(
            os.path.join(
                error_out_path,
                "%s_computed_distances_%s.npy" % (method_identifier, challenge),
            ),
            computed_distances,
        )

    print(f"Computed distances saved to {error_out_path}")
    print(f"Number of missing files: {num_missing_files}")
    print(f"Number of computed distances: {len(distance_metric)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NoW evaluation")
    parser.add_argument(
        "--predicted_mesh_folder",
        type=str,
        default="",
        help=" Path to predicted results to be evaluated",
    )
    parser.add_argument(
        "--dataset_folder",
        type=str,
        default="",
        help="Path to root of the dataset, which contains images, scans and lanmarks. One myst specify either the dataset_folder or gt_mesh_folder and gt_lmk_folder.",
    )
    parser.add_argument(
        "--image_set",
        type=str,
        default="val",
        help="val or test. This specifies which images will be used. Ignored if imgs_list argument is specified",
    )
    parser.add_argument(
        "--error_out_path",
        type=str,
        default=None,
        help="(Optional) Path to results folder. If None, results will be saved to predicted_mesh_folder",
    )
    parser.add_argument(
        "--method_identifier",
        type=str,
        default="RECON",
        help="(Optional) Will be used to name the output file",
    )
    parser.add_argument(
        "--gt_mesh_folder",
        type=str,
        default=None,
        help="(Optional) Path to the GT scans. If not specified, it will be looked for in the dataset folder",
    )
    parser.add_argument(
        "--gt_lmk_folder",
        type=str,
        default=None,
        help="(Optional) Path to the GT landmarks. If not specified, it will be looked for in the dataset folder",
    )
    parser.add_argument(
        "--imgs_list",
        type=str,
        default=None,
        help="(Optional) Path to file with image list to be used",
    )
    parser.add_argument(
        "--metrical_evaluation",
        type=bool,
        default=False,
        help="Flag if metrical or non-metrical evaluation protocol is used.",
    )
    parser.add_argument(
        "--predicted_mesh_unit",
        type=str,
        default="m",
        help="(Optional) Unit of measurements of the reconstructions. Required for metrical evaluations. Supported: [mm, cm, m]",
    )
    parser.add_argument(
        "--challenge",
        type=str,
        default="",
        help="(Optional) Error computation for one of the challenges only. Supported:  [multiview_neutral, multiview_expressions, multiview_occlusions, selfie]",
    )
    parser.add_argument(
        "--check_alignment",
        type=bool,
        default=False,
        help="(Optional) Outputs the predicted meshes rigidly aligned with the scans to check the rigid alignment",
    )

    args = parser.parse_args()
    dataset_folder = args.dataset_folder
    predicted_mesh_folder = args.predicted_mesh_folder
    image_set = args.image_set
    error_out_path = args.error_out_path
    method_identifier = args.method_identifier
    gt_mesh_folder = args.gt_mesh_folder
    gt_lmk_folder = args.gt_lmk_folder
    imgs_list = args.imgs_list
    metrical_evaluation = args.metrical_evaluation
    predicted_mesh_unit = args.predicted_mesh_unit
    challenge = args.challenge
    check_alignment = args.check_alignment

    if metrical_evaluation:
        method_identifier += "_metrical"

    metric_computation(
        dataset_folder,
        predicted_mesh_folder,
        gt_mesh_folder,
        gt_lmk_folder,
        image_set,
        imgs_list,
        challenge=challenge,
        error_out_path=error_out_path,
        method_identifier=method_identifier,
        predicted_mesh_unit=predicted_mesh_unit,
        metrical_eval=metrical_evaluation,
        check_alignment=check_alignment,
    )
