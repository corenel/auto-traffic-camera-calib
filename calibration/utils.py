import numpy as np


def get_angle_between(u, v):
    """
    Get angle between two vectors

    :param u: first vector
    :param v: second vector
    :return: angle (in radian)
    """
    # cosine of the angle
    c = np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)
    # if you really want the angle
    angle = np.arccos(np.clip(c, -1, 1))
    return angle


def compute_z_avg(calibs):
    """
    Compute average z of calibrations

    :param calibs: set of calibrations
    :return: average z
    """
    return np.mean(calibs[:, :, 2], axis=0)


def compute_average_rotation_matrix(z_avg):
    return 0


def reproject_to_ground(p, calib, camera):
    camera.set_R(calib[:, :3])
    camera.set_t(calib[:, 3])
    return camera.image_to_world(p, z=0)


def distance_to_camera(p, calib):
    point_to_camera = p + calib[:2, :]
    return np.linalg.norm(point_to_camera[:2])


def compute_displacement(calibs, p, camera):
    displacements = []
    for calib_idx in range(calibs.shape[0]):
        p_i = reproject_to_ground(p, calibs[calib_idx], camera)
        d_i = distance_to_camera(p_i, calibs[calib_idx])
        displacements.append(d_i)
    return displacements
