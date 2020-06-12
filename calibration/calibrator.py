import numpy as np
import calibration.utils as utils
import calibration.filters as filters
from calibration.camera import Camera
from assets.real_world_keypoints import real_world_keypoints
from detection.keypoint.utils import orientation_to_keypoints


def compute_focus_region_mid_point():
    return 1


class Calibration:
    def __init__(self) -> None:
        super().__init__()
        self.camera = Camera()

    def calibrate(self, keypoints, orientations):
        calibs = []
        for box_idx in range(orientations.shape[0]):
            for car_model, model_keypoints in real_world_keypoints.items():
                visible_keypoints = keypoints[box_idx][
                    orientation_to_keypoints[int(orientations[box_idx])]]
                visible_objects = model_keypoints[orientation_to_keypoints[int(
                    orientations[box_idx])]]
                ret, rmat, tvec = self.camera.calibrate_extrinsic(
                    visible_objects, visible_keypoints)
                calibs.append(np.concatenate([rmat, tvec], axis=1))
        return np.stack(calibs)

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
        # z_avg = utils.compute_z_avg(calibs)
        r_avg = utils.compute_average_rotation_matrix(calibs)
        calibs[:, :, :3] = r_avg

        # get the middle calibration
        d = utils.compute_displacement(calibs, p, self.camera)
        median_idx = np.argsort(d)[len(d) // 2]
        calib_est = calibs[median_idx]

        return calib_est
