import numpy as np
import math


def get_deviation_from_plane(R, unit_axis, plane_normal):

    assert np.isclose(np.linalg.norm(unit_axis), 1.0)
    assert np.isclose(np.linalg.norm(plane_normal), 1.0)
    axis = R.T @ unit_axis
    deviation = math.fabs(math.pi / 2 - math.acos(axis @ plane_normal))
    return deviation


def get_deviation_from_axis(R, unit_axis):
    assert np.isclose(np.linalg.norm(unit_axis), 1.0)
    y = R @ unit_axis
    deviation = np.arccos(y @ unit_axis)
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


def get_R_alpha(axis_idx, alpha):

    s = math.sin(alpha)
    c = math.cos(alpha)

    if axis_idx == 0:
        return np.array([
            [1.0, 0, 0],
            [0, c, -s],
            [0, s, c],
        ])
    elif axis_idx == 1:
        return np.array([
            [c, 0, s],
            [0, 1.0, 0],
            [-s, 0, c],
        ])
    elif axis_idx == 2:
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1.0],
        ])
    else:
        raise ValueError(f"really?: {axis_idx}")


def test():

    X = np.array([1.0, 0, 0])
    Y = np.array([0, 1.0, 0])
    Z = np.array([0, 0, 1.0])

    for alp1 in range(0, 359, 5):
        a1_r = alp1 / 180 * math.pi
        R_y = get_R_alpha(1, a1_r)
        assert np.allclose(get_deviation_from_axis(R_y, Y), 0)
        for alp2 in range(0, 359, 5):
            a2_r = alp2 / 180 * math.pi
            R_x = get_R_alpha(0, a2_r)
            R_z = get_R_alpha(2, a2_r)
            R_yx = R_x @ R_y
            R_yz = R_z @ R_y
            assert np.allclose(get_deviation_from_plane(R_yx, X, Y), 0)
            assert np.allclose(get_deviation_from_plane(R_yz, Z, Y), 0)
            if alp2 % 180 != 0:
                assert not np.allclose(get_deviation_from_axis(R_yx, Y), 0)
                assert not np.allclose(get_deviation_from_axis(R_yz, Y), 0)
                assert not np.allclose(get_deviation_from_plane(R_yx, Z, Y), 0)
                assert not np.allclose(get_deviation_from_plane(R_yz, X, Y), 0)


if __name__ == "__main__":
    test()
