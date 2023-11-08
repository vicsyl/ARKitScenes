import itertools
import argparse
import math
import os
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

import utils.box_utils as box_utils
from common.common_transforms import evaluate_pose
from common.common_transforms import get_deviation_from_axis, get_deviation_from_plane, X_AXIS, Z_AXIS, Y_AXIS
from common.fitting import fit_min_area_rect
from common.vanishing_point import get_directions, get_vp
from data_utils import append_entry, save
from pnp_utils import *
from utils.taxonomy import class_names, ARKitDatasetConfig
from utils.tenFpsDataLoader import TenFpsDataLoader, extract_gt

dc = ARKitDatasetConfig()
from pathlib import Path
from pyquaternion import Quaternion
import traceback

ARKIT_PERMUTE = np.array([
     [0., 0., 1.],
     [1., 0., 0.],
     [0., 1., 0.]])


class DataPrepareConf:
    visualize_main_directions = False
    visualize_vanishing_point = True
    visualize_rest = False


def get_main_directions(centers_2d, K, R, t):

    centers_3d = unproject_k_r_t(centers_2d, K, R, t)

    centers_3d_plus_y = centers_3d + np.array([0.0, 1.0, 0.0])
    centers_plus_gt_y = project_from_frame_R_t(K, R, t, centers_3d_plus_y).T[:, :2]
    dirs_gt_y = centers_plus_gt_y - centers_2d

    centers_3d_plus_x = centers_3d + np.array([1.0, 0.0, 0.0])
    centers_plus_gt_x = project_from_frame_R_t(K, R, t, centers_3d_plus_x).T[:, :2]
    dirs_gt_x = centers_plus_gt_x - centers_2d

    centers_3d_plus_z = centers_3d + np.array([0.0, 0.0, 1.0])
    centers_plus_gt_z = project_from_frame_R_t(K, R, t, centers_3d_plus_z).T[:, :2]
    dirs_gt_z = centers_plus_gt_z - centers_2d

    return dirs_gt_x, dirs_gt_y, dirs_gt_z


def get_vps(K, R_gt, t_gt, boxes_2d):

    # TODO - is this necessary?
    boxes_2d = np.array(boxes_2d)
    all_chosen_2d_dirs, all_heights, all_centers_2d = get_directions(boxes_2d)

    _, all_dirs_gt_ys, _ = get_main_directions(all_centers_2d, K, R_gt, t_gt)

    pure_R_y_reals = []
    vp_homo_reals = []
    pure_R_y_gts = []
    vp_homo_gts = []
    centers_2d_used_vps = []

    for comb in itertools.combinations(range(boxes_2d.shape[0]), 2):

        centers_2d_used = all_centers_2d[list(comb)]
        centers_2d_used_vps.append(centers_2d_used)
        pure_R_y_real, vp_homo_real = get_vp(all_chosen_2d_dirs, centers_2d_used)
        pure_R_y_reals.append(pure_R_y_real)
        vp_homo_reals.append(vp_homo_real.tolist())

        _, dirs_gt_y, _ = get_main_directions(centers_2d_used, K, R_gt, t_gt)
        pure_R_y_gt, vp_homo_gt = get_vp(dirs_gt_y, centers_2d_used)
        pure_R_y_gts.append(pure_R_y_gt)
        vp_homo_gts.append(vp_homo_gt.tolist())

    return all_centers_2d, all_chosen_2d_dirs, all_heights, all_dirs_gt_ys, \
           centers_2d_used_vps, vp_homo_reals, pure_R_y_reals, vp_homo_gts, pure_R_y_gts


# def permute_me_R_t(perm, r, t):
#     return perm @ r @ np.linalg.inv(perm), perm @ t
#
#
# def permute_me_column_vectors(perm, matrix, line_vectors):
#     raise NotImplementedError
#
#
# def permute_me_row_vectors(perm, matrix, row_vectors):
#     line_vectors_ret = None
#     matrix_ret = None
#     if row_vectors is not None:
#         assert row_vectors.shape[1] == 3
#         line_vectors_ret = (perm @ row_vectors.T).T
#         assert line_vectors_ret.shape[1] == 3
#     if matrix is not None:
#         matrix_ret = perm @ matrix @ np.linalg.inv(perm)
#     return matrix_ret, line_vectors_ret
#
#
def visualize(frame,
              boxes_2d,
              projections,
              mask_pts_in_box,
              centers_proj_in_2d,
              boxes_crns,
              scene_id,
              file_name):

    def vis_directions_from_center(center, dir_l, fmt, linewidth=2):
        dir_vis = np.vstack((center, center + dir_l))
        ax.plot(dir_vis[:, 0], dir_vis[:, 1], fmt, linewidth=linewidth)

    def vis_directions_from_boxes(dir_l, centers_2d_loc, fmt, linewidth=2):
        dir_vis1 = np.zeros((boxes_2d.shape[0], 2, 2))
        dir_vis1[:, 0] = boxes_2d.sum(axis=1) / 4
        assert np.all(dir_vis1[:, 0] == centers_2d_loc)
        dir_vis1[:, 1] = boxes_2d.sum(axis=1) / 4 + dir_l / 2
        for i in range(dir_vis1.shape[0]):
            ax.plot(dir_vis1[i, :, 0], dir_vis1[i, :, 1], fmt, linewidth=linewidth)

    K = frame["intrinsics"]
    pcd = frame["pcd"]
    pose = frame["pose"]
    R_gt, t_gt = R_t_from_frame_pose(pose)

    # # ARKIT permute
    # R_gt_permuted, t_gt_permuted = permute_me_R_t(ARKIT_PERMUTE, R_gt, t_gt)
    # _, pcd_permuted = permute_me_row_vectors(ARKIT_PERMUTE, None, pcd)

    # test projections (project_from_frame_R_t vs. project_from_frame)
    projections_test = project_from_frame_R_t(K, R_gt, t_gt, pcd)
    assert np.allclose(projections, projections_test)

    # projections_test_perm = project_from_frame_R_t(K, R_gt_permuted, t_gt_permuted, pcd_permuted)
    # assert np.allclose(projections, projections_test_perm)

    # Show image.
    _, ax = plt.subplots(1, 1, figsize=(9, 16))
    img = frame['image']
    ax.imshow(img)
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)

    # 2D bbox
    for b_i in range(len(boxes_2d)):
        for v_i in range(3):
            ax.plot(boxes_2d[b_i][v_i:v_i+2][:, 0], boxes_2d[b_i][v_i:v_i+2][:, 1], "y-.", linewidth=2)
        ax.plot(boxes_2d[b_i][[0, 3]][:, 0], boxes_2d[b_i][[0, 3]][:, 1], "y-.", linewidth=2)

    # the main axes
    if DataPrepareConf.visualize_main_directions:
        img_center, _ = unproject_center_r_t(R_gt, t_gt, K)
        img_center = img_center[:, :2]
        dirs_gt_x, dirs_gt_y, dirs_gt_z = get_main_directions(img_center, K, R_gt, t_gt)
        ax.plot(img_center[:, 0:1], img_center[:, 1:2], "rx", markersize=20, markeredgewidth=2)
        vis_directions_from_center(img_center, dirs_gt_x, fmt="r-", linewidth=3)
        vis_directions_from_center(img_center, dirs_gt_y, fmt="g-", linewidth=3)
        # 3rd coord -> vertical
        # 3rd coord -> y

        vis_directions_from_center(img_center, dirs_gt_z, fmt="b-", linewidth=3)

    if DataPrepareConf.visualize_vanishing_point and boxes_2d.shape[0] > 1:

        # # TODO iterate through all tuples?
        # if boxes_2d.shape[0] > 2:
        #     print("boxes_2d.shape[0] > 2 (vanishing points)")
        # boxes_2d = boxes_2d[:2]

        #         centers_2d, \
        #         pure_R_y_real, \
        #         chosen_2d_dirs, \
        #         vp_homo, \
        #         pure_R_y_gt, \
        #         dirs_gt_x, \
        #         dirs_gt_y, \
        #         dirs_gt_z, \
        #         vp_homo_gt =

        all_centers_2d, all_chosen_2d_dirs, all_heights, all_dirs_gt_ys, \
               centers_2d_used_vps, vp_homo_reals, pure_R_y_reals, vp_homo_gts, pure_R_y_gts = get_vps(K, R_gt, t_gt, boxes_2d)

        # boxes_2d - all directions
        for i in range(2):
            # [sample, axis]
            dir = boxes_2d[:, i] - boxes_2d[:, i + 1]
            vis_directions_from_boxes(dir, all_centers_2d, "k-", linewidth=6)

        # centers
        ax.plot(all_centers_2d[:, 0], all_centers_2d[:, 1], "rx", markersize=20, markeredgewidth=2)

        # chosen dirs
        vis_directions_from_boxes(all_chosen_2d_dirs, all_centers_2d, "r-", linewidth=4)

        # dirs_gt
        # vis_directions_from_boxes(dirs_gt_x, centers_2d, "y-.", linewidth=4)
        vis_directions_from_boxes(all_dirs_gt_ys, all_centers_2d, "m-.", linewidth=4)
        # vis_directions_from_boxes(dirs_gt_z, centers_2d, "k-.", linewidth=4)

        def viso_vps(pure_R_ys, vp_homos, fmt):

            # in closure: centers_2d_used_vps

            for centers_2d_used_vp, pure_R_y, vp_homo in zip(centers_2d_used_vps, pure_R_ys, vp_homos):
                if pure_R_y:
                    print("vp viso skipped, pure r_y")
                else:
                    for i in range(2):
                        ax.plot([centers_2d_used_vp[i, 0], vp_homo[0]], [centers_2d_used_vp[i, 1], vp_homo[1]], fmt, linewidth=6)

        viso_vps(pure_R_y_reals, vp_homo_reals, "r-")
        viso_vps(pure_R_y_gts, vp_homo_gts, "b-")

    if DataPrepareConf.visualize_rest:
        color = ["b", "r", "g", "y", "m", "c", "k", "b", "r"]
        print(f"size: {mask_pts_in_box.shape[1]}")
        for b_i in range(mask_pts_in_box.shape[1]):
            colr = color[b_i % 9]
            # point clouds per objects
            proj_to_use = projections[:, mask_pts_in_box[:, b_i]]
            fmt = f"{colr}o"
            ax.plot(proj_to_use[0], proj_to_use[1], fmt, markersize=2)

            centers_display = centers_proj_in_2d[:, b_i: b_i + 1]
            centers_display = centers_display[:, centers_display[2] == 1.0]
            ax.plot(centers_display[0], centers_display[1], f"{colr}^", markersize="9")

            one_box = boxes_crns[b_i].T
            crns_display = one_box[:, one_box[2] == 1.0]
            # 3D box vertices
            ax.plot(crns_display[0], crns_display[1], fmt, markersize="7")

            # 3D box frame
            fmt_bf = f"{colr}-"
            for fr in range(3):
                tt = fr + 1
                if tt < crns_display.shape[1] and fr < crns_display.shape[1]:
                    ax.plot(crns_display[0, [fr, tt]], crns_display[1, [fr, tt]], fmt_bf, linewidth=2)
                tt = fr + 4
                if tt < crns_display.shape[1] and fr < crns_display.shape[1]:
                    ax.plot(crns_display[0, [fr, tt]], crns_display[1, [fr, tt]], fmt_bf, linewidth=2)

                tt = fr + 5
                fr2 = fr + 4
                if tt < crns_display.shape[1] and fr2 < crns_display.shape[1]:
                    ax.plot(crns_display[0, [fr2, tt]], crns_display[1, [fr2, tt]], fmt_bf, linewidth=2)

                tt = fr + 5
                fr2 = fr + 1
                if tt < crns_display.shape[1] and fr2 < crns_display.shape[1]:
                    ax.plot(crns_display[0, [fr2, tt]], crns_display[1, [fr2, tt]], fmt_bf, linewidth=2)

            #
            if crns_display.shape[1] >= 4:
                ax.plot(crns_display[0, [0, 3]], crns_display[1, [0, 3]], fmt_bf, linewidth=2)
            if crns_display.shape[1] >= 8:
                ax.plot(crns_display[0, [4, 7]], crns_display[1, [4, 7]], fmt_bf, linewidth=2)

    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    ax.axis("off")
    Path(f"imgs/{scene_id}/ok").mkdir(parents=True, exist_ok=True)
    Path(f"imgs/{scene_id}/all").mkdir(parents=True, exist_ok=True)
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()
    print(f"{file_name} saved")


def get_scene_ids_gts(data_root):
    ret = []
    for p in Path(data_root).iterdir():
        if p.is_dir() and set(p.name).issubset(set("0123456789")):
            gt = p.joinpath(f"{p.name}_3dod_annotation.json")
            if gt.exists() and gt.is_file():
                ret.append((p.name, gt.absolute().__str__()))

    return ret


def main():
    # --data_root ../train/Training
    # --scene_id 47333462
    # --gt_path ../train/Training/47333462/47333462_3dod_annotation.json
    # --output_dir ../train/Training/47333462/47333462_online_prepared_data/ --frame_rate 5
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        default="./threedod/download/3dod/Training/",
        help="input folder with ./scene_id"
        "extracted by unzipping scene_id.tar.gz"
    )

    parser.add_argument(
        "--gt_path",
        default="../../threedod/sample_data/47331606/47331606_3dod_annotation.json",
        help="gt path to annotation json file",
    )

    # this should be 1
    parser.add_argument("--frame_rate", default=1, type=int, help="sampling rate of frames")
    parser.add_argument(
        "--output_dir", default="../sample_data/online_prepared_data/", help="directory to save the data and annoation"
    )
    parser.add_argument("--vis", action="store_true", default=False)
    parser.add_argument("--min_corners", type=int, default=6)
    parser.add_argument("--min_scenes", type=int, default=0)
    parser.add_argument("--max_scenes", type=int, default=None)
    parser.add_argument("--ang_tol", type=float, default=None)
    parser.add_argument("--min_obj", type=int, default=2)
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()
    print(f"Args:\{args}")

    ang_tol = args.ang_tol
    min_objects = args.min_obj
    print(f"ang_tol: {ang_tol}")
    out_hocon_dir = "./hocons"
    Path(out_hocon_dir).mkdir(parents=True, exist_ok=True)
    suffix = f"_min={args.min_scenes}" if args.min_scenes != 0 else ""
    suffix += f"_max={args.max_scenes}" if args.max_scenes is not None else ""
    ang_tol_pint = int(ang_tol) if ang_tol is not None and round(ang_tol) == ang_tol else ang_tol
    ang_infix = f"_ang={ang_tol_pint}" if ang_tol_pint is not None else ""
    save_file_path = f"{out_hocon_dir}/ARKitScenes=obj={min_objects}{suffix}{ang_infix}"
    print(f"Will save into: {save_file_path}")

    scenes = get_scene_ids_gts(args.data_root)
    if args.min_scenes is not None:
        scenes = scenes[:args.max_scenes]
    scenes = scenes[args.min_scenes:]

    print(f"{len(scenes)} scenes:")
    print("\n".join([str(s) for s in scenes]))

    objects_counts_map = defaultdict(int)
    data_entries = []
    # counts
    all_frames = 0
    all_R_y = 0
    all_x_hor = 0
    all_z_hor = 0

    savepoint_indices = set([10, 20, 50, 100, 500])

    start_time = time.time()
    for scene_index, (scene_id, gt_path) in enumerate(scenes):

        try:
            skipped, boxes_corners, centers_3d, sizes, labels, uids = extract_gt(gt_path)
            if skipped:
                print(f"scene {scene_id} skipped")
                continue
        except:
            print("exception caught (see below), skipping the scene")
            traceback.print_exc()
            print("exception caught (see above), skipping the scene")

        objects = boxes_corners.shape[0]
        if objects < 2:
            print(f"two few objects ({objects}) in scene {scene_id} skipped")
            continue

        data_path = os.path.join(args.data_root, scene_id, f"{scene_id}_frames")
        print(f"scene index: {scene_index}")
        print(f"data_path: {os.path.abspath(data_path)}")
        loader = TenFpsDataLoader(
            dataset_cfg=None,
            class_names=class_names,
            root_path=data_path,
            world_coordinate=True
        )

        start_time_scene = time.time()
        # for frame_index in range(len(loader)):
        for frame_index in range(338, 339):

            all_frames += 1

            if frame_index % args.frame_rate != 0:
                continue

            frame = loader[frame_index]
            image_path = frame["image_path"]
            pcd = frame["pcd"]
            pose = frame["pose"]
            rgb = frame["color"]
            boxes = box_utils.corners_to_boxes(boxes_corners)
            # remove boxes with < 20 points
            # (n, m)
            mask_pts_in_box = box_utils.points_in_boxes(pcd, boxes_corners)
            # (m, )
            pts_cnt = np.sum(mask_pts_in_box, axis=0)
            mask_box = pts_cnt > 20
            # only used here: obj_names.append(gt_labels['types'][obj_i])
            gt_labels = {
                "bboxes": boxes,
                "bboxes_mask": mask_box,
                "types": labels,
                "uids": uids,
                "pose": pose,
            }

            # projections = K @ (R @ pcd.T + t_gt)
            K = frame["intrinsics"]
            R_gt_old, t_gt = R_t_from_frame_pose(pose)

            # projections are fine !!!
            projections = project_from_frame(K, pose, pcd).T
            projections_2 = project_from_frame_R_t(K, R_gt_old, t_gt, pcd)
            assert np.allclose(projections, projections_2)

            def change_x_3d(x_3d):
                assert x_3d.shape[1] == 3
                R_x_m_half_pi = np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, -1.0, 0.0],
                ])
                return (R_x_m_half_pi @ x_3d.T).T

            def change_r(r_l):
                R_x_half_pi = np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0],
                    [0.0, 1.0, 0.0],
                ])
                return r_l @ R_x_half_pi

            R_gt = change_r(R_gt_old)

            # pcd_new used only here
            pcd_new = change_x_3d(pcd)
            projections_3 = project_from_frame_R_t(K, R_gt, t_gt, pcd_new)
            assert np.allclose(projections, projections_3)

            boxes_corners_used = boxes_corners.reshape(-1, 3)
            boxes_crns = project_from_frame(K, pose, boxes_corners_used)

            # test, if OK, then boxes_crns are fine
            boxes_corners_used_new = change_x_3d(boxes_corners_used)
            boxes_crns_new = project_from_frame_R_t(K, R_gt, t_gt, boxes_corners_used_new).T
            assert np.allclose(boxes_crns_new, boxes_crns)

            boxes_crns = boxes_crns.reshape(-1, 8, 3)
            # this should be fine, too
            centers_proj_in_2d = project_from_frame(K, pose, centers_3d).T

            # now the tests will be against new, rectified R_gt
            def is_close(a, b):
                if ang_tol is None:
                    return True
                else:
                    return math.fabs(a - b) < ang_tol

            R_y_dev = get_deviation_from_axis(R_gt, Y_AXIS)
            x_hor_dev = get_deviation_from_plane(R_gt, X_AXIS, Y_AXIS)
            z_hor_dev = get_deviation_from_plane(R_gt, Z_AXIS, Y_AXIS)
            if args.verbose:
                print(f"R_y_dev: {R_y_dev} ", end="")
                print(f"x_hor_dev: {x_hor_dev} ", end="")
                print(f"z_hor_dev: {z_hor_dev} ")
            is_R_y = is_close(R_y_dev, 0)
            is_x_hor = is_close(x_hor_dev, 0)
            is_z_hor = is_close(z_hor_dev, 0)

            if is_R_y:
                all_R_y += 1
            if is_z_hor:
                all_z_hor += 1
            if is_x_hor:
                all_x_hor += 1

            if args.verbose:
                print(f"is_R_y: {is_R_y} ", end="")
                print(f"is_x_hor: {is_x_hor} ", end="")
                print(f"is_z_hor: {is_z_hor} ")
            if not is_R_y and not is_x_hor and not is_z_hor and not args.vis:
                if args.verbose:
                    print("Frame skipped, not pose not aligned")
                continue

            # def in_img(xy_np):
            #     img = frame['image']
            #     return xy_np[0] >= 0 and xy_np[0] < img.shape[0] and xy_np[1] >= 0 and xy_np[1] < img.shape[1]
            #
            # def vector_ok(xyl):
            #     return xyl[2] == 1.0 and in_img(xyl[:2])

            def vectors_ok(xyl_c):
                img = frame['image']
                in_front = xyl_c[2] == 1.0
                mask = in_front
                mask = np.logical_and(mask, xyl_c[0] >= 0)
                mask = np.logical_and(mask, xyl_c[1] >= 0)
                mask = np.logical_and(mask, xyl_c[0] < img.shape[0])
                mask = np.logical_and(mask, xyl_c[1] < img.shape[1])
                return mask

            # centers_2d_data = centers_3d[:, centers_3d[2] == 1.0]
            # already present
            # K = frame["intrinsics"]
            R_gt_q_l_old = Quaternion._from_matrix(R_gt_old).elements.tolist()
            R_gt_q_l = Quaternion._from_matrix(R_gt).elements.tolist()
            boxes_2d_newly_added = []

            # this is new -> new 2D bboxes
            x_i_old = []
            x_i = []

            X_i = []
            X_i_up = []
            X_i_down = []

            # apparently not used
            all_2d_corners_old = []

            obj_names = []
            # scene_token = scene_id
            all_orientations = []
            widths_heights_old = []
            lwhs = []
            corners_counts = []
            # sample_data_token = frame_index
            # both_2D_3D = list(range(n))
            for obj_i in range(centers_proj_in_2d.shape[1]):
                # box_center_ok = centers_proj_in_2d[2, obj_i] == 1.0 and in_img(centers_proj_in_2d[:2, obj_i])

                one_box = boxes_crns[obj_i].T
                mask_ok = vectors_ok(one_box)
                proj_to_use = projections[:, mask_pts_in_box[:, obj_i]]
                min_projections = 100
                corners_count = sum(mask_ok).item()
                if corners_count >= args.min_corners and proj_to_use.shape[1] >= min_projections:

                    corners_counts.append(corners_count)

                    # new 2D bboxes...
                    pixels_to_fit = projections[:, mask_pts_in_box[:, obj_i]][:2].transpose().astype(int)
                    box = fit_min_area_rect(pixels_to_fit)
                    boxes_2d_newly_added.append(box)
                    new_center_2d = box.sum(axis=0) / 4
                    x_i.append(new_center_2d)

                    # old 2D bboxes...
                    min_2dx = np.min(proj_to_use[0]).item()
                    max_2dx = np.max(proj_to_use[0]).item()
                    min_2dy = np.min(proj_to_use[1]).item()
                    max_2dy = np.max(proj_to_use[1]).item()
                    c_x_old = (min_2dx + max_2dx) / 2
                    c_y_old = (min_2dy + max_2dy) / 2
                    x_i_old.append([c_x_old, c_y_old])
                    two_d_corners_old = [
                        [max_2dx, c_y_old],
                        [max_2dx, max_2dy],
                        [c_x_old, max_2dy],
                        [min_2dx, max_2dy],
                        [min_2dx, c_y_old],
                        [min_2dx, min_2dy],
                        [c_x_old, min_2dy],
                        [max_2dx, min_2dy],
                    ]
                    all_2d_corners_old.append(two_d_corners_old)
                    widths_heights_old.append([max_2dx - min_2dx, max_2dy - min_2dy])


                    # 3D
                    center_3d = centers_3d[obj_i]
                    X_i.append(center_3d)
                    # boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading]
                    #             with (x, y, z) is the box center
                    #             (dx, dy, dz) as the box size
                    #             and heading as the clockwise rotation angle
                    box_3d = boxes[obj_i]
                    assert np.all(box_3d[:3] == center_3d)

                    # TODO check
                    half_dimension_up = box_3d[4] / 2
                    X_i_up.append(center_3d + half_dimension_up)
                    X_i_down.append(center_3d - half_dimension_up)
                    # east, north - east, north, etc..(x0, x1), (y0, y1)

                    obj_names.append(gt_labels['types'][obj_i])
                    # TODO - for gravity: yes, but use box_3d[:, 6] as heading: the clockwise rotation angle
                    # lw unused, h should be dz and therefore box_3d[5]
                    lwhs.append(box_3d[3:6].tolist())
                    default_orientation = Quaternion._from_matrix(np.eye(3)).elements.tolist()
                    all_orientations.append(default_orientation)

            obj_count = len(X_i)
            objects_counts_map[obj_count] += 1

            # demo_vp disabled
            # centers_2d = None
            # chosen_2d_dirs = None
            if obj_count >= min_objects and (is_R_y or is_x_hor or is_z_hor):

                vis_file_path = f"imgs/{scene_id}/ok/ok_{frame_index}.png"

                # centers_2d, \
                # pure_R_y_real, \
                # chosen_2d_dirs, \
                # vp_homo, \
                # pure_R_y_gt, \
                # _, \
                # dirs_gt, \
                # _, \
                # vp_homo_gt = get_vps(K, R_gt, t_gt, boxes_2d_newly_added)

                # CONTINUE!!!
                all_centers_2d, \
                all_chosen_2d_dirs, \
                all_heights, \
                all_dirs_gt_ys, \
                centers_2d_used_vps, \
                vp_homo_reals, \
                pure_R_y_reals, \
                vp_homo_gts, \
                pure_R_y_gts = get_vps(K, R_gt, t_gt, boxes_2d_newly_added)
                # FIXME: REDUNDANCY
                assert np.allclose(all_centers_2d, x_i)

                extra_map = {
                    # all
                    "all_centers_2d": all_centers_2d.tolist(),
                    "all_chosen_2d_dirs": all_chosen_2d_dirs.tolist(),
                    "all_dirs_gt_ys": all_dirs_gt_ys.tolist(),
                    # 2 over n
                    "pure_R_y_reals": pure_R_y_reals,
                    "vp_homo_reals": vp_homo_reals if vp_homo_reals is not None else None,
                    "pure_R_y_gts": pure_R_y_gts,
                    "vp_homo_gts": vp_homo_gts if vp_homo_gts is not None else None,

                    # old
                    "R_cs_l": R_gt_q_l_old,
                    "x_i_old": x_i_old
                }

                # CHANGES:
                # R_cs_l (+ _old)
                # x_i (+ _old)
                # boxes_2d (newly_added)

                append_entry(data_entries,
                             # new -> visualization, orig image
                             orig_img_path=image_path,
                             vis_img_path=vis_file_path,
                             corners_counts=corners_counts,
                             # new
                             x_i=np.array(x_i),  # np.ndarray(n, 2)
                             X_i=np.asarray(X_i),  # np.ndarray(n, 3)
                             # new
                             boxes_2d=np.array(boxes_2d_newly_added),
                             K=K,  # np.ndarray(3, 3)
                             # new
                             R_cs_l=R_gt_q_l,  # list[4] : quaternion
                             t_cs_l=t_gt[:, 0].tolist(),  # list[3]: meters
                             # new
                             R_ego_l=R_gt_q_l,  # list[4] : quaternion
                             t_ego_l=t_gt[:, 0].tolist(),  # list[3]: meters
                             X_i_up_down=np.array([X_i_up, X_i_down]),  # np.ndarray(2, n, 3): first index: # center + height/2, center - height/2

                             # Apparently this is not used...
                             two_d_cmcs=None,  # list[n] of np.array(2, 8) # last index: east, north-east, north, etc.. (x0, x1), (y0, y1)
                             # two_d_cmcs=[np.array(l).T for l in all_2d_corners_old],  # list[n] of np.array(2, 8) # last index: east, north-east, north, etc.. (x0, x1), (y0, y1)

                             names=obj_names,  # list[n] types of objects
                             scene_token=scene_id,  # (e.g. 'trABmlDfsN1z6XCSJgFQxO')
                             lwhs=lwhs,  # list[n] of list[3] : length, width, height
                             orientations=all_orientations,  # list[n] of list[4]: quaternion
                             # TODO test against widths_heights_new
                             widths_heights=np.hstack((all_heights[:, None], all_heights[:, None])).tolist(),  # list[n] of list[2]
                             widths_heights_new=widths_heights_old,  # list[n] of list[2]
                             sample_data_token=frame_index,  # (e.g. 'a1LHTHCD_RydavtlH93q8Q-cam-right')
                             # IMHO unused
                             both_2D_3D=list(range(obj_count)),
                             extra_map=extra_map)

            else:
                vis_file_path = f"imgs/{scene_id}/all/all_{frame_index}.png"
                if args.verbose:
                    print("Frame skipped")
                    print(f"obj_count >= min_objects: {obj_count >= min_objects}")
                    print(f"is_R_y: {is_R_y}; is_x_hor: {is_x_hor} is_z_hor: {is_x_hor}")

            # CONTINUE: monitor the return value
            # CONTINUE: caching... ?

            if args.vis:
                print(f"R:\n{R_gt}")
                rot_err, pos_err = evaluate_pose(np.eye(3), t_gt, R_gt, t_gt)
                print(f"rot_err:\n{rot_err}")
                visualize(frame,
                          np.array(boxes_2d_newly_added),
                          projections,
                          mask_pts_in_box,
                          centers_proj_in_2d,
                          boxes_crns,
                          scene_id,
                          vis_file_path)


        elapased = time.time() - start_time_scene
        print(f"elapsed time for scene {scene_id}: %f sec" % elapased)

        if savepoint_indices.__contains__(scene_index + 1) and scene_index + 1 != len(scenes):
            sp_file_path = f"{out_hocon_dir}/ARKitScenes=obj={min_objects}{suffix}{ang_infix}_sp={scene_index + 1}"
            save(sp_file_path, data_entries, objects_counts_map, vars(args))

    elapased = time.time() - start_time
    print(f"total time: %f sec" % elapased)

    print("Saving to hocon")
    save(save_file_path, data_entries, objects_counts_map, vars(args))

    print(f"all_frames: {all_frames}")
    print(f"all_R_y: {all_R_y}")
    print(f"all_x_hor: {all_x_hor}")
    print(f"all_z_hor: {all_z_hor}")


# TODO

# remove filter... OK
# add the fitting to the object
# visualize that ... OK
# clean up - (quickly) ...
# --min_corners=6 --min_obj=1 --data_root ../../download/3dod/Training --vis --frame_rate 1
# ignore?? TODO: sky direction: a) try out some visos b) just filter on UP

# continue:
# - all needed data / per scene !!
# - how to run it on the server?
# - clean up and externalize
# - run to generate the data
# - which frames / scenes it will be used?
# - measure the time to read the data
# - adopt to original project (new dataset)
# pose - normalize (assert), deviations..., filters...

if __name__ == "__main__":
    main()
