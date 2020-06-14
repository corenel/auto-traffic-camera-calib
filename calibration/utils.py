import cv2
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


def compute_average_rotation_matrix(calibs):
    common_unit_vector = np.array([0, 0, 1])
    # compute average normalized rotated vector
    rotated_vectors = []
    for box_idx in range(calibs.shape[0]):
        rotated_vector = np.matmul(calibs[box_idx][:, :3], common_unit_vector)
        rotated_vectors.append(rotated_vector / np.linalg.norm(rotated_vector))
    mean_rotated_vector = np.mean(np.stack(rotated_vectors), axis=0)
    mean_rotated_vector /= np.linalg.norm(mean_rotated_vector)
    # compute average rotation vector
    axis = np.cross(mean_rotated_vector, common_unit_vector)
    angle = np.dot(mean_rotated_vector, common_unit_vector)
    rvec = angle * axis
    # get basis with z_avg
    z_avg = compute_z_avg(calibs)
    basis_1, basis_2, _ = basis(z_avg)
    # get representation of rvec under new basis
    bases = np.stack([basis_1, basis_2, z_avg],
                     axis=1)
    rvec_repr = np.linalg.inv(bases) * np.eye(3) * rvec
    rvec_new = rvec_repr[:, 0] * basis_1 + rvec_repr[:, 1] * basis_2 + rvec_repr[:, 2] * z_avg
    # transform rvec to rotation matrix
    rmat = cv2.Rodrigues(rvec_new)[0]
    return rmat


def reproject_to_ground(p, calib, camera):
    camera.R = calib[:, :3]
    camera.t = calib[:, 3].reshape(3, 1)
    camera.update_P()
    return camera.image_to_world(p, z=0)


def distance_to_camera(p, calib):
    point_to_camera = p + calib[:, 3]
    return np.linalg.norm(point_to_camera[:2])


def compute_displacement(calibs, p, camera):
    displacements = []
    for calib_idx in range(calibs.shape[0]):
        p_i = reproject_to_ground(p, calibs[calib_idx], camera)
        d_i = distance_to_camera(p_i, calibs[calib_idx])
        displacements.append(d_i)
    return displacements


def basis(v):
    v = v / np.linalg.norm(v)
    if v[0] > 0.9:
        b1 = np.asarray([0.0, 1.0, 0.0])
    else:
        b1 = np.asarray([1.0, 0.0, 0.0])
    b1 -= v * np.dot(b1, v)
    b1 /= np.linalg.norm(b1)
    b2 = np.cross(v, b1)
    return b1, b2, v
