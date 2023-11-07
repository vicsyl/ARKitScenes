import numpy as np


def R_t_from_frame_pose(frame_pose):
    inv = np.linalg.inv(frame_pose)
    R = inv[:3, :3]
    # columns or rows?
    t = inv[:3, 3:4]
    return R, t


def normalize_projections(projections):
    # TODO everything in front??!!
    in_front = np.sign(projections[2])
    projections = projections / projections[2]
    projections[2] *= in_front
    return projections


def project_from_frame(K, frame_pose, row_points):
    columns = row_points.T
    augmented = np.vstack(
        (columns, np.ones((1, columns.shape[1])))
    )
    projections = K @ ((np.linalg.inv(frame_pose) @ augmented)[:3])
    projections = normalize_projections(projections)
    return projections.T


# TODO docstring
def unproject_center_r_t(R_gt, t_gt, K_for_center):
    center_homo = np.array([[0.0], [0.0], [1.0]])
    X = np.linalg.inv(R_gt) @ (center_homo - t_gt)
    X = X.T
    test = True
    if test:
        back = project_from_frame_R_t(np.eye(3), R_gt, t_gt, X)
        assert np.allclose(back, center_homo)

    center = project_from_frame_R_t(K_for_center, R_gt, t_gt, X).T
    return center, X


# TODO docstring
def unproject_k_r_t(x_i, K, R_gt, t_gt):

    x_homo = np.zeros((x_i.shape[0], 3))
    x_homo[:, :2] = x_i
    x_homo[:, 2] = 1.0

    X = np.linalg.inv(R_gt) @ (np.linalg.inv(K) @ x_homo.T - t_gt)
    X = X.T
    test = True
    if test:
        back = project_from_frame_R_t(K, R_gt, t_gt, X).T
        assert np.allclose(x_homo, back)
    return X


# TODO docstring
def project_from_frame_R_t(K, R_gt, t_gt, row_points):
    assert row_points.shape[1] == 3
    projections = K @ (R_gt @ row_points.T + t_gt)
    projections = normalize_projections(projections)
    return projections
