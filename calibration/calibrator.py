import numpy as np
import calibration.utils as utils
import calibration.filters as filters
from calibration.camera import Camera


def compute_focus_region_mid_point():
    return 1


class Calibration:
    def __init__(self) -> None:
        super().__init__()
        self.camera = Camera()

    def filter_and_average(self, calibs: np.array):
        """
        AutoCalib filtering and averaging function

        :param calibs: set of calibrations of the shape [N, 3, 4]
        :return: estimated calibration of the shape [3, 4]
        """
        # get mid point
        p = compute_focus_region_mid_point()

        # filtering
        calibs = filters.orientation_filter(calibs, 0.75)
        calibs = filters.displacement_filter(calibs, p, self.camera, 0.50)
        calibs = filters.orientation_filter(calibs, 0.75)

        # compute average R
        z_avg = utils.compute_z_avg(calibs)
        r_avg = utils.compute_average_rotation_matrix(z_avg)
        calibs[:, :, :3] = r_avg

        # get the middle calibration
        d = utils.compute_displacement(calibs, p, self.camera)
        median_idx = np.argsort(d)[len(d) // 2]
        calib_est = calibs[median_idx]

        return calib_est
