import numpy as np


def append_entry(entries_list,
                 orig_img_path,
                 vis_img_path,
                 corners_counts,
                 x_i,
                 X_i,
                 boxes_2d,
                 K,
                 R_cs_l,
                 t_cs_l,
                 R_ego_l,
                 t_ego_l,
                 X_i_up_down,
                 two_d_cmcs,
                 names,
                 scene_token,
                 lwhs,
                 orientations,
                 widths_heights,
                 widths_heights_new,
                 sample_data_token,
                 both_2D_3D,
                 extra_map={}):
    """
    Args:
        entries_list:
        corners_counts: list[]: int
        x_i: np.ndarray(n, 2)
        X_i: np.ndarray(n, 3)
        K: np.ndarray(3, 3)
        # TODO a) quaternion b) cs/ego
        R_cs_l: list[4] : quaternion
        t_cs_l: list[3] : meters
        R_ego_l: list[4] : quaternion
        t_ego_l: list[3] : meters
        X_i_up_down: np.ndarray(2, n, 3): first index: # center + height/2, center - height/2
        # TODO: NOW just pass it in a more convenient way
        two_d_cmcs: list[n] of np.array(2, 8) # last index: east, north-east, north, etc.. (x0, x1), (y0, y1)
        names: list[n] types of objects
        scene_token: scene_token (e.g. 'trABmlDfsN1z6XCSJgFQxO')
        lwhs: list[n] of list[3] : length, width, height
        orientations: list[n] of list[4]: quaternion
        widths_heights: list[n] of list[2]
        widths_heights_new: list[n] of list[2]
        sample_data_token: sample_data_token (e.g. 'a1LHTHCD_RydavtlH93q8Q-cam-right')
        both_2D_3D: list[n] : object ids: AFAIK unused
    Returns:
    """

    entry_map = {"x_i": x_i,
               "boxes_2d": boxes_2d,
               "vis_file_path": vis_img_path,
               "orig_file_path": orig_img_path,
               "corners_counts": corners_counts,
               # east, north-east, north, etc.. (x0, x1), (y0, y1)
               "x_i_corners_mps": two_d_cmcs,
               "X_i": X_i,
               # center + height/2, center - height/2
               "X_i_up_down": X_i_up_down,
               "orientations": orientations,
               "lwhs": lwhs,
               "widths_heights": widths_heights,
               "widths_heights_new": widths_heights_new,
               "names": names,
               "R_cs": R_cs_l,
               "t_cs": t_cs_l,
               "R_ego": R_ego_l,
               "t_ego": t_ego_l,
               "K": K,
               "scene": scene_token,
               "sample_data": sample_data_token,
               "instances": both_2D_3D}

    for k, v in extra_map.items():
        entry_map[k] = v

    entries_list.append(entry_map)
