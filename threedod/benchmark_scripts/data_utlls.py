from pyhocon import ConfigFactory, HOCONConverter
import numpy as np


def save_to_hocon(fp, entries, objects_counts_map=None, conf_attribute_map={}):

    at_least_objects_counts_map = {}
    # TODO test with None
    if objects_counts_map is not None:
        at_least_n_sum = 0
        for obj, count in objects_counts_map.items():
            if obj > 10:
                at_least_n_sum += count
        for at_least in range(10, 0, -1):
            if objects_counts_map.__contains__(at_least):
                at_least_n_sum += objects_counts_map[at_least]
            at_least_objects_counts_map[str(at_least)] = at_least_n_sum

    data = {'metropolis_data': entries,
            'samples_with_at_least_n_objects': at_least_objects_counts_map,
            'config': conf_attribute_map
            }

    conf_tree = ConfigFactory.from_dict(data)
    with open(fp, "w") as fd:
        fd.write(HOCONConverter.to_hocon(conf_tree))
    print(f"data saved to {fp}")


def append_entry(entries_list,
                 x_i,
                 X_i,
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
                 both_2D_3D):
    """
    Args:
        entries_list:
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


    x_i = x_i.tolist()
    X_i = X_i.tolist()
    X_i_up_down = X_i_up_down.tolist()
    K = K.tolist()
    two_d_cmcs = np.transpose(np.array([t.T for t in two_d_cmcs]), (1, 0, 2)).tolist()

    p3p_map = {"x_i": x_i,
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
    entries_list.append(p3p_map)