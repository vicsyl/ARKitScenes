import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common.common_transforms import project_from_frame_R_t
from common.vanishing_point import get_main_directions, get_vps, change_r_arkit


# TODO make it arkit agnostic
from decomposition import RotSampler, RSampling, Rot, Smp


def visualize(frame,
              K,
              R_gt,
              R_gt_new,
              t_gt,
              lwhs,
              boxes_2d,
              boxes_8_points_2d_old,
              widths_heights_old,
              widths_heights_new,
              x_i,
              x_i_new,
              X_i,
              X_i_new,
              X_i_up,
              X_i_down,
              projections,
              mask_pts_in_box,
              centers_proj_in_2d,
              boxes_crns,
              scene_id,
              file_name,
              show,
              visualize_main_directions=False,
              visualize_vanishing_point=False,
              visualize_rest=False):

    def vis_directions_from_center(center, dir_l, fmt, label, linewidth=2):
        dir_vis = np.vstack((center, center + dir_l))
        ax.plot(dir_vis[:, 0], dir_vis[:, 1], fmt, linewidth=linewidth, label=label)

    def vis_directions_from_boxes(dir_l, centers_2d_loc, fmt, linewidth=2):
        dir_vis1 = np.zeros((boxes_2d.shape[0], 2, 2))
        dir_vis1[:, 0] = boxes_2d.sum(axis=1) / 4
        assert np.all(dir_vis1[:, 0] == centers_2d_loc)
        dir_vis1[:, 1] = boxes_2d.sum(axis=1) / 4 + dir_l / 2
        for i in range(dir_vis1.shape[0]):
            ax.plot(dir_vis1[i, :, 0], dir_vis1[i, :, 1], fmt, linewidth=linewidth)

    printoptions = np.get_printoptions()
    np.set_printoptions(formatter={'float': lambda x: "{0:.3g}".format(x), "max_line_width": np.inf}, linewidth=np.inf)

    K = frame["intrinsics"]
    # pose = frame["pose"]
    # R_gt, t_gt = R_t_from_frame_pose(pose)

    # Show image.
    _, ax = plt.subplots(1, 1, figsize=(9, 16))
    img = frame['image']
    ax.imshow(img)
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)

    # 2D bbox vis
    def vis_2d_box(boxes_2d_l, fmt, label):
        label_on = True
        for b_i in range(len(boxes_2d_l)):
            for v_i in range(3):
                ax.plot(boxes_2d_l[b_i][v_i:v_i + 2][:, 0], boxes_2d_l[b_i][v_i:v_i + 2][:, 1], fmt, linewidth=2)
            ax.plot(boxes_2d_l[b_i][[0, 3]][:, 0], boxes_2d_l[b_i][[0, 3]][:, 1], fmt, linewidth=2, label=label if label_on else None)
            if label_on:
                label_on = False

    vis_2d_box(boxes_2d, "y-.", "new boxes")
    if len(boxes_8_points_2d_old) > 0:
        boxes_8_points_2d_old = np.transpose(np.array(boxes_8_points_2d_old)[:, :, 1::2], axes=(0, 2, 1))
        vis_2d_box(boxes_8_points_2d_old, "r-.", "old boxes")

    def plot_pempty_np(ar, fmt, label, markersize=6, markeredgewidth=3):
        if ar.shape[0] > 0:
            ax.plot(ar[:, 0], ar[:, 1], fmt, markersize=markersize, markeredgewidth=markeredgewidth, label=label)

    plot_pempty_np(x_i, fmt="rx", label="x_i old")
    plot_pempty_np(x_i_new, fmt="bx", label="x_i new")
    x_proj = project_from_frame_R_t(K, R_gt, t_gt[:, 0], X_i).T
    plot_pempty_np(x_proj, fmt="mx", label="X_i", markersize=10, markeredgewidth=3)
    x_proj = project_from_frame_R_t(K, R_gt, t_gt[:, 0], X_i_down).T
    plot_pempty_np(x_proj, fmt="bx", label="X_i down", markersize=10, markeredgewidth=3)
    x_proj = project_from_frame_R_t(K, R_gt, t_gt[:, 0], X_i_up).T
    plot_pempty_np(x_proj, fmt="yx", label="X_i up", markersize=10, markeredgewidth=3)

    def unproject_center_r_t_local(R_gt, t_gt, K_for_center):
        center_homo = np.array([[0.0], [0.0], [1.0]])
        X = np.linalg.inv(R_gt) @ (center_homo - t_gt)
        X = X.T
        test = False
        # FIXME assumes K = np.eye(3)
        if test:
            back = project_from_frame_R_t(np.eye(3), R_gt, t_gt, X)
            assert np.allclose(back, center_homo)

        center = project_from_frame_R_t(K_for_center, R_gt, t_gt[:, 0], X).T
        return center, X

    # the main axes
    if visualize_main_directions:
        def vis_main_dir(R_l, t_l, K_l, line_fmt, legend_suffix):
            img_center, _ = unproject_center_r_t_local(R_l, t_l, K_l)
            img_center = img_center[:, :2]
            dirs_gt_x, dirs_gt_y, dirs_gt_z = get_main_directions(img_center, K, R_l, t_l[:, 0])
            ax.plot(img_center[:, 0:1], img_center[:, 1:2], "rx", markersize=20, markeredgewidth=2)
            vis_directions_from_center(img_center, dirs_gt_x, fmt=f"r{line_fmt}", label=f"x-{legend_suffix}", linewidth=3)
            vis_directions_from_center(img_center, dirs_gt_y, fmt=f"g{line_fmt}", label=f"y-{legend_suffix}", linewidth=3)
            vis_directions_from_center(img_center, dirs_gt_z, fmt=f"b{line_fmt}", label=f"z-{legend_suffix}", linewidth=3)

        vis_main_dir(R_gt, t_gt, K, "-", "orig")
        R_gt_new = change_r_arkit(R_gt)
        vis_main_dir(R_gt_new, t_gt, K, "--", "new")
        plt.legend()

    def log_title(s):
        print_data = True
        return f"{s}\n" if print_data else ""

    sampler = RotSampler([
        RSampling(Rot.Y, 1, Smp.UNIFORM),
        RSampling(Rot.X, 1, Smp.UNIFORM),
        RSampling(Rot.Z, 1, Smp.UNIFORM),
    ])

    title = ""
    title += log_title(f"b2d: {boxes_2d}")
    # title += log_title(f"boxes:{np.vstack((boxes_2d, boxes_2d, boxes_2d))}")
    pretty_gt, _, _ = sampler.decompose_and_info(R_gt)
    title += log_title(f"R_gt={pretty_gt}")
    pretty_new, _, _ = sampler.decompose_and_info(R_gt_new)
    title += log_title(f"R_gt_new={pretty_new}")

    title += log_title(f"widths_heights_old:{np.array(widths_heights_old)}")
    title += log_title(f"x_i/centers_old: {x_i}")
    title += log_title(f"widths_heights_new:{np.array(widths_heights_new)}")
    title += log_title(f"x_i/centers_new: {x_i_new}")
    title += log_title(f"lwhs: {np.array(lwhs)}")

    if False or visualize_vanishing_point and boxes_2d.shape[0] > 0:

        all_centers_2d, all_chosen_2d_dirs, all_heights, all_dirs_gt_ys, \
        centers_2d_used_vps, vp_homo_reals, pure_R_y_reals, vp_homo_gts, pure_R_y_gts = get_vps(K, R_gt, t_gt, boxes_2d)

        # max_line_width ?
        title += log_title(f"2d_bb_centers:\n{all_centers_2d}\n")
        title += log_title(f"all_heights:\n{all_heights}\n")

        all_chosen_2d_dirs_normed = all_chosen_2d_dirs / np.linalg.norm(all_chosen_2d_dirs, axis=1)[:, None]
        title += log_title(f"detected boxes vertical directions:\n{all_chosen_2d_dirs_normed}\n")

        all_dirs_gt_ys_normed = all_dirs_gt_ys / np.linalg.norm(all_dirs_gt_ys, axis=1)[:, None]
        title += log_title(f"gt projected vertical directions:\n{all_dirs_gt_ys_normed}\n")

        # something like this
        dir_errors = []
        for gt, detected in zip(all_dirs_gt_ys_normed, all_chosen_2d_dirs_normed):
            alpha = np.arccos(gt.T @ detected).item()
            alpha = alpha * math.pi / 180
            alpha = min(alpha, 180 - alpha)
            dir_errors.append(alpha)
        dir_errors = np.array(dir_errors)
        title += log_title(f"vertical direction errors[deg]:\n{dir_errors}\n")

        # boxes_2d - all directions
        for i in range(2):
            # [sample, axis]
            dir = boxes_2d[:, i] - boxes_2d[:, i + 1]
            vis_directions_from_boxes(dir, all_centers_2d, "c-", linewidth=2)

        # centers
        ax.plot(all_centers_2d[:, 0], all_centers_2d[:, 1], "rx", markersize=10, markeredgewidth=2)

        # chosen dirs
        vis_directions_from_boxes(all_chosen_2d_dirs, all_centers_2d, "r-", linewidth=5)
        vis_directions_from_boxes(all_dirs_gt_ys, all_centers_2d, "b-.", linewidth=3)

        def viso_vps(pure_R_ys, vp_homos, fmt, linewidth):

            # in closure: centers_2d_used_vps

            for centers_2d_used_vp, pure_R_y, vp_homo in zip(centers_2d_used_vps, pure_R_ys, vp_homos):
                if pure_R_y:
                    print("vp viso skipped, pure r_y")
                else:
                    for i in range(2):
                        ax.plot([centers_2d_used_vp[i, 0], vp_homo[0]], [centers_2d_used_vp[i, 1], vp_homo[1]], fmt,
                                linewidth=linewidth)

        viso_vps(pure_R_y_reals, vp_homo_reals, "r-", 5)
        viso_vps(pure_R_y_gts, vp_homo_gts, "b-.", 3)

    if visualize_rest:
        color = ["b", "r", "g", "y", "m", "c", "k", "b", "r"]
        # print(f"size: {mask_pts_in_box.shape[1]}")
        for b_i in range(mask_pts_in_box.shape[1]):
            colr = color[b_i % 9]
            # point clouds per objects
            proj_to_use = projections[:, mask_pts_in_box[:, b_i]]
            fmt = f"{colr}o"
            ax.plot(proj_to_use[0], proj_to_use[1], fmt, markersize=2)

            centers_display = centers_proj_in_2d[:, b_i: b_i + 1]
            centers_display = centers_display[:, centers_display[2] == 1.0]
            ax.plot(centers_display[0], centers_display[1], f"{colr}x", markersize="15", markeredgewidth=4)

            if centers_display.shape[1] != 0:
                title += log_title(f"centers_display:\n{centers_display[:2, 0]}\n")

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

    plt.title(title)
    # ax.axis("off")
    Path(f"imgs/{scene_id}/ok").mkdir(parents=True, exist_ok=True)
    Path(f"imgs/{scene_id}/all").mkdir(parents=True, exist_ok=True)
    plt.savefig(file_name, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    print(f"{file_name} saved")
    np.set_printoptions(**printoptions)
