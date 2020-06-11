import numpy as np
import calibration.utils as utils


def orientation_filter(calibs, n):
    """
    Filter using orientation

    :param calibs: set of calibrations
    :param n: percent of calibration to be reserved (in [0,1])
    :return: set of filtered calibrations
    """
    # compute theta
    z_avg = utils.compute_z_avg(calibs)
    z = calibs[:, :, 2]
    theta = [utils.get_angle_between(z[i], z_avg) for i in range(z.shape[0])]
    # get lowest n% elements
    k = int(n * len(theta))
    theta_idx = np.argpartition(theta, kth=k)[:k]
    return calibs[theta_idx]


def displacement_filter(calibs, p, camera, n=0.50):
    """
    Filter using displacement

    :param calibs: set of calibrations
    :param n: percent of calibration to be reserved (in [0,1])
    :param p: center point of interest region (in image coord)
    :return: set of filtered calibrations
    """
    # compute displacements
    displacements = utils.compute_displacement(calibs, p, camera)
    # get middle n% elements
    num_outliers = int(n * calibs.shape[0])
    start_idx = num_outliers // 2
    end_idx = calibs.shape[0] - num_outliers // 2
    filtered_idx = np.argsort(displacements)[start_idx:end_idx]
    return calibs[filtered_idx]
