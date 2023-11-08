import numpy as np

from common.vanishing_point import normalize_projections


def R_t_from_frame_pose(frame_pose):
    inv = np.linalg.inv(frame_pose)
    R = inv[:3, :3]
    # columns or rows?
    t = inv[:3, 3:4]
    return R, t


def project_from_frame(K, frame_pose, row_points):
    columns = row_points.T
    augmented = np.vstack(
        (columns, np.ones((1, columns.shape[1])))
    )
    projections = K @ ((np.linalg.inv(frame_pose) @ augmented)[:3])
    projections = normalize_projections(projections)
    return projections.T
