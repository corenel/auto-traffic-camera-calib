import numpy as np

import calibration.filters as filters
import calibration.utils as utils
from assets.real_world_keypoints import real_world_keypoints
from calibration.camera import Camera
from detection.keypoint.utils import orientation_to_keypoints


def compute_focus_region_mid_point():
    return np.array([[1296], [1024]], dtype=np.float)


class Calibration:
    def __init__(self, camera_matrix=None, dist_coeff=None) -> None:
        super().__init__()
        self.camera = Camera()
        if camera_matrix is not None:
            self.camera.set_K(camera_matrix)
        if dist_coeff is not None:
            self.camera.opencv_dist_coeff = dist_coeff
        self.calib_est = None

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

        if len(calibs) > 0:
            return np.stack(calibs)
        else:
            return np.empty((0, 3, 4))

    def filter_and_average(self, calibs: np.array):
        """
        AutoCalib filtering and averaging function

        :param calibs: set of calibrations of the shape [N, 3, 4]
        :return: estimated calibration of the shape [3, 4]
        """
        if calibs.shape[0] == 0:
            return None

        # get mid point
        p = compute_focus_region_mid_point()

        # filtering
        calibs = filters.orientation_filter(calibs, 0.75)
        calibs = filters.displacement_filter(calibs, p, self.camera, 0.50)
        calibs = filters.orientation_filter(calibs, 0.75)

        # compute average R
        r_avg = utils.compute_average_rotation_matrix(calibs)
        calibs[:, :, :3] = r_avg

        # get the middle calibration
        d = utils.compute_displacement(calibs, p, self.camera)
        median_idx = np.argsort(d)[len(d) // 2]
        calib_est = calibs[median_idx]

        return calib_est

    def image_to_world(self, p, calib):
        self.camera.set_extrinsic(calib)
        return self.camera.image_to_world(p, z=0)

    def world_to_image(self, p, calib):
        self.camera.set_extrinsic(calib)
        return self.camera.world_to_image(p)
