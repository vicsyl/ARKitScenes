#TODO: remove sys.path based import

#!/usr/bin/env python3

import argparse
import math

import numpy as np
import os
import time

import utils.box_utils as box_utils
import utils.rotation as rotation
from utils.taxonomy import class_names, ARKitDatasetConfig
from utils.tenFpsDataLoader import TenFpsDataLoader, extract_gt
import utils.visual_utils as visual_utils
import matplotlib.pyplot as plt
from PIL import Image
from pnp_utils import project_from_frame, R_t_from_frame_pose, get_deviation_from_plane, get_deviation_from_axis

dc = ARKitDatasetConfig()
from pathlib import Path


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

    args = parser.parse_args()
    scenes = get_scene_ids_gts(args.data_root)
    print(f"scenes: {scenes}")

    start_time = time.time()
    for scene_id, gt_path in scenes:

        skipped, boxes_corners, centers, sizes, labels, uids = extract_gt(gt_path)
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
        for i in range(len(loader)):

            if i % args.frame_rate != 0:
                continue

            print(f"Processing frame: {i}")

            frame = loader[i]
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
            projections2 = project_from_frame(K, pose, pcd).T
            boxes_crns = project_from_frame(K, pose, boxes_corners.reshape(-1, 3))
            boxes_crns = boxes_crns.reshape(-1, 8, 3)
            centers_2d = project_from_frame(K, pose, centers).T

            # pcd_ort = np.vstack(
            #     (pcd.T, np.ones((1, pcd.T.shape[1])))
            # )
            #projections = K @ ((np.linalg.inv(pose) @ pcd_ort)[:3])

            R_gt, t = R_t_from_frame_pose(pose)

            X = np.array([1.0, 0, 0])
            Y = np.array([0, 1.0, 0])
            Z = np.array([0, 0, 1.0])

            def is_close(a, b):
                ang_tol = 2
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
            print(f"is_R_y: {is_R_y} ", end="")
            print(f"is_x_hor: {is_x_hor} ", end="")
            print(f"is_z_hor: {is_z_hor} ")

            projections = K @ (R_gt @ pcd.T + t)
            projections = projections / projections[2]
            assert np.all(projections2 == projections)

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

                    centers_display = centers_2d[:, b_i: b_i + 1]
                    centers_display = centers_display[:, centers_display[2] == 1.0]
                    ax.plot(centers_display[0], centers_display[1], fmt2, markersize="20")

                    one_box = boxes_crns[b_i].T
                    crns_display = one_box[:, one_box[2] == 1.0]
                    ax.plot(crns_display[0], crns_display[1], fmt, markersize="15")

                np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
                ax.axis("off")
                Path(f"imgs/{scene_id}").mkdir(parents=True, exist_ok=True)
                plt.savefig(f"imgs/{scene_id}/all_{i}.png")

        elapased = time.time() - start_time_scene
        print(f"{scene_id}: elapsed time: %f sec" % elapased)
    elapased = time.time() - start_time
    print(f"total time: %f sec" % elapased)


# continue:
# - all needed data / per scene !!
# - how to run it on the server?
# - clean up and externalize
# - run to generate the data
# - which frames / scenes it will be used?
# - measure the time to read the data
# - adopt to original project (new dataset)

# pose - normalize (assert), deviations..., filters...