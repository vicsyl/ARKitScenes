import numpy as np


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

