import numpy as np


def get_deviation(R, unit_vector):
    assert np.isclose(np.linalg.norm(unit_vector), 1.0)
    y = R @ unit_vector
    deviation = np.arccos(y @ unit_vector)
    return deviation


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
    in_front = np.sign(projections[2])
    projections = projections / projections[2]
    projections[2] *= in_front
    return projections.T

