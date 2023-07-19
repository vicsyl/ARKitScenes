import argparse
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np

import utils.box_utils as box_utils
from data_utlls import append_entry, save_to_hocon
from pnp_utils import project_from_frame, R_t_from_frame_pose, get_deviation_from_plane, get_deviation_from_axis
from utils.taxonomy import class_names, ARKitDatasetConfig
from utils.tenFpsDataLoader import TenFpsDataLoader, extract_gt

dc = ARKitDatasetConfig()
from pathlib import Path
from pyquaternion import Quaternion


def get_scene_ids_gts(data_root):
    ret = []
    for p in Path(data_root).iterdir():
        if p.is_dir() and set(p.name).issubset(set("0123456789")):
            gt = p.joinpath(f"{p.name}_3dod_annotation.json")
            if gt.exists() and gt.is_file():
                ret.append((p.name, gt.absolute().__str__()))

    return ret


if __name__ == "__main__":
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
    parser.add_argument("--max_scenes", type=int, default=None)
    parser.add_argument("--ang_tol", type=int, default=5)
    parser.add_argument("--min_obj", type=int, default=2)

    args = parser.parse_args()
    scenes = get_scene_ids_gts(args.data_root)
    if args.max_scenes is not None:
        scenes = scenes[:args.max_scenes]

    ang_tol = args.ang_tol
    min_objects = args.min_obj

    print(f"ang_tol: {ang_tol}")
    print(f"{len(scenes)} scenes:")
    print("\n".join([str(s) for s in scenes]))

    objects_counts_map = {}
    data_entries = []
    # counts
    all_frames = 0
    all_R_y = 0
    all_x_hor = 0
    all_z_hor = 0

    start_time = time.time()
    for scene_id, gt_path in scenes:

        skipped, boxes_corners, centers_3d, sizes, labels, uids = extract_gt(gt_path)
        if skipped:
            print(f"scene {scene_id} skipped")
            continue

        objects = boxes_corners.shape[0]
        if objects < 2:
            print(f"two few objects ({objects}) in scene {scene_id} skipped")
            continue

        data_path = os.path.join(args.data_root, scene_id, f"{scene_id}_frames")
        print(f"data_path: {os.path.abspath(data_path)}")
        loader = TenFpsDataLoader(
            dataset_cfg=None,
            class_names=class_names,
            root_path=data_path,
            world_coordinate=True
        )

        start_time_scene = time.time()
        for frame_index in range(len(loader)):

            all_frames += 1

            if frame_index % args.frame_rate != 0:
                continue

            print(f"Processing frame: {frame_index}")

            frame = loader[frame_index]
            image_path = frame["image_path"]
            image = frame["image"]
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
            gt_labels = {
                "bboxes": boxes,
                "bboxes_mask": mask_box,
                "types": labels,
                "uids": uids,
                "pose": pose,
            }

            # projections = K @ (R @ pcd.T + t_gt)
            K = frame["intrinsics"]
            projections = project_from_frame(K, pose, pcd).T

            R_gt, t_gt = R_t_from_frame_pose(pose)

            # test
            projections2 = K @ (R_gt @ pcd.T + t_gt)
            projections2 = projections2 / projections2[2]
            assert np.all(projections2 == projections)

            boxes_crns = project_from_frame(K, pose, boxes_corners.reshape(-1, 3))
            boxes_crns = boxes_crns.reshape(-1, 8, 3)
            centers_proj_in_2d = project_from_frame(K, pose, centers_3d).T

            X = np.array([1.0, 0, 0])
            Y = np.array([0, 1.0, 0])
            Z = np.array([0, 0, 1.0])

            def is_close(a, b):
                tol = ang_tol * math.pi / 180
                return math.fabs(a - b) < tol

            R_y_dev = get_deviation_from_axis(R_gt, Y)
            x_hor_dev = get_deviation_from_plane(R_gt, X, Y)
            z_hor_dev = get_deviation_from_plane(R_gt, Z, Y)
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

            print(f"is_R_y: {is_R_y} ", end="")
            print(f"is_x_hor: {is_x_hor} ", end="")
            print(f"is_z_hor: {is_z_hor} ")
            if not is_R_y and not is_x_hor and not is_z_hor:
                print("Frame skipped, not pose not aligned")
                continue

            def in_img(xy_np):
                img = frame['image']
                return xy_np[0] >= 0 and xy_np[0] < img.shape[0] and xy_np[1] >= 0 and xy_np[1] < img.shape[1]

            def vector_ok(xyl):
                return xyl[2] == 1.0 and in_img(xyl[:2])

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
            R_gt_q_l = Quaternion._from_matrix(R_gt).elements.tolist()
            x_i = []
            X_i = []
            X_i_up = []
            X_i_down = []
            all_2d_corners = []
            obj_names = []
            # scene_token = scene_id
            all_orientations = []
            widths_heights = []
            lwhs = []
            # sample_data_token = frame_index
            # both_2D_3D = list(range(n))
            for obj_i in range(centers_proj_in_2d.shape[1]):
                # box_center_ok = centers_proj_in_2d[2, obj_i] == 1.0 and in_img(centers_proj_in_2d[:2, obj_i])

                one_box = boxes_crns[obj_i].T
                mask_ok = vectors_ok(one_box)
                minimal_corners = 6
                proj_to_use = projections[:, mask_pts_in_box[:, obj_i]]
                min_projections = 100
                if sum(mask_ok) >= minimal_corners and proj_to_use.shape[1] >= min_projections:
                    min_2dx = np.min(proj_to_use[0])
                    max_2dx = np.max(proj_to_use[0])
                    min_2dy = np.min(proj_to_use[1])
                    max_2dy = np.max(proj_to_use[1])

                    x_i.append(centers_proj_in_2d[:2, obj_i])
                    center = centers_3d[obj_i]
                    X_i.append(center)

                    # boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading]
                    #             with (x, y, z) is the box center
                    #             (dx, dy, dz) as the box size
                    #             and heading as the clockwise rotation angle
                    box_3d = boxes[obj_i]
                    assert np.all(box_3d[:3] == center)
                    # TODO check
                    half_dimension_up = box_3d[4] / 2
                    X_i_up.append(center + half_dimension_up)
                    X_i_down.append(center - half_dimension_up)
                    # east, north - east, north, etc..(x0, x1), (y0, y1)
                    c_x = (min_2dx + max_2dx) / 2
                    c_y = (min_2dy + max_2dy) / 2
                    two_d_corners = [
                        [max_2dx, c_y],
                        [max_2dx, max_2dy],
                        [c_x, max_2dy],
                        [min_2dx, max_2dy],
                        [min_2dx, c_y],
                        [min_2dx, min_2dy],
                        [c_x, min_2dy],
                        [max_2dx, min_2dy],
                    ]
                    all_2d_corners.append(two_d_corners)
                    obj_names.append(gt_labels['types'][obj_i])
                    # TODO - for gravity: yes, but use box_3d[:, 6] as heading: the clockwise rotation angle
                    # lw unused, h should be dz and therefore box_3d[5]
                    lwhs.append(box_3d[3:6].tolist())
                    all_orientations.append(Quaternion._from_matrix(np.eye(3)).elements.tolist())
                    # old AND new
                    widths_heights.append([max_2dx - min_2dx, max_2dy - min_2dy])

            obj_count = len(X_i)
            if not objects_counts_map.__contains__(obj_count):
                objects_counts_map[obj_count] = 0
            objects_counts_map[obj_count] += 1

            if obj_count >= min_objects:
                append_entry(data_entries,
                             x_i=np.array(x_i), # np.ndarray(n, 2)
                             X_i=np.asarray(X_i), # np.ndarray(n, 3)
                             K=K, # np.ndarray(3, 3)
                             # TODO
                             R_cs_l=R_gt_q_l, # list[4] : quaternion
                             t_cs_l=t_gt[:, 0].tolist(), # list[3]: meters
                             R_ego_l=R_gt_q_l, # list[4] : quaternion
                             t_ego_l=t_gt[:, 0].tolist(), # list[3]: meters
                             X_i_up_down=np.array([X_i_up, X_i_down]), # np.ndarray(2, n, 3): first index: # center + height/2, center - height/2
                             # TODO: NOW just pass it in a more convenient way
                             two_d_cmcs=[np.array(l).T for l in all_2d_corners], # list[n] of np.array(2, 8) # last index: east, north-east, north, etc.. (x0, x1), (y0, y1)
                             names=obj_names, # list[n] types of objects
                             scene_token=scene_id, # (e.g. 'trABmlDfsN1z6XCSJgFQxO')
                             lwhs=lwhs, # list[n] of list[3] : length, width, height
                             orientations=all_orientations, # list[n] of list[4]: quaternion
                             widths_heights=widths_heights, # list[n] of list[2]
                             widths_heights_new=widths_heights, # list[n] of list[2]
                             sample_data_token=frame_index, # (e.g. 'a1LHTHCD_RydavtlH93q8Q-cam-right')
                             # IMHO unused
                             both_2D_3D=list(range(obj_count)))

            if args.vis:

                _, ax = plt.subplots(1, 1, figsize=(9, 16))

                #_, ax = plt.subplots(1, 1)
                # with open(image_path, "rb") as fid:
                #     data = Image.open(fid)
                #     data.load()

                # Show image.
                img = frame['image']
                ax.imshow(img)
                # ax.imshow(data)
                ax.set_xlim(0, img.shape[0])
                ax.set_ylim(img.shape[1], 0)

                ax.plot(projections[0],
                        projections[1],
                        'r1',
                        markersize=1)

                # mask_pts_in_box_w = box_utils.points_in_boxes(pcd, boxes_corners)
                # TODO - arbitrary number of objects
                color = ["b", "r", "g", "y", "m", "c", "k", "b", "r"]
                for b_i in range(9):
                    proj_to_use = projections[:, mask_pts_in_box[:, b_i]]
                    fmt = f"{color[b_i]}x"
                    fmt2 = f"{color[b_i]}o"
                    # print(f"fmt: {fmt}")

                    ax.plot(proj_to_use[0], proj_to_use[1], fmt, markersize=4)

                    centers_display = centers_proj_in_2d[:, b_i: b_i + 1]
                    centers_display = centers_display[:, centers_display[2] == 1.0]
                    ax.plot(centers_display[0], centers_display[1], fmt2, markersize="20")

                    one_box = boxes_crns[b_i].T
                    crns_display = one_box[:, one_box[2] == 1.0]
                    ax.plot(crns_display[0], crns_display[1], fmt, markersize="15")

                np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
                ax.axis("off")
                Path(f"imgs/{scene_id}").mkdir(parents=True, exist_ok=True)
                plt.savefig(f"imgs/{scene_id}/all_{frame_index}.png")

        elapased = time.time() - start_time_scene
        print(f"{scene_id}: elapsed time: %f sec" % elapased)
    elapased = time.time() - start_time
    print(f"total time: %f sec" % elapased)

    print("Saving to hocon")
    out_hocon_dir = "./hocons"
    Path(out_hocon_dir).mkdir(parents=True, exist_ok=True)

    suffix = f"_max={args.max_scenes}" if args.max_scenes is not None else ""
    fp = f"{out_hocon_dir}/ARKitScenes=obj={min_objects}{suffix}.conf"

    save_to_hocon(fp, data_entries, objects_counts_map, vars(args))

    print(f"all_frames: {all_frames}")
    print(f"all_R_y: {all_R_y}")
    print(f"all_x_hor: {all_x_hor}")
    print(f"all_z_hor: {all_z_hor}")

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