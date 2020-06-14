import unittest

import cv2
import numpy as np

from calibration.calibrator import Calibration
from detection.keypoint import KeypointDetector
from .test_object_detector import detector as object_detector, image as object_image

CHECKPOINT_PATH = '/home/yuthon/Workspace/Vehicle_Key_Point_Orientation_Estimation/' \
                  'checkpoints/stage2/best_fine_kp_checkpoint.pth.tar'
MEAN_STD_PATH = '/home/yuthon/Workspace/Vehicle_Key_Point_Orientation_Estimation/' \
                'data/VeRi/mean.pth.tar'
TEST_IMAGE_DIR = '/home/yuthon/Downloads/test_images'

model = KeypointDetector(checkpoint_path=CHECKPOINT_PATH,
                         mean_std_path=MEAN_STD_PATH)


class TestCalibrator(unittest.TestCase):
    def test_calibration(self):
        detections = object_detector.detect(frame=object_image)
        frame_rgb = cv2.cvtColor(object_image, cv2.COLOR_BGR2RGB)
        keypoints, orientations = model.detect(frame_rgb,
                                               detections[0],
                                               visualize=True)
        calibrator = Calibration()
        calibrator.camera.set_K(np.array([[1, 0, 1296], [0, 1, 1024], [0, 0, 1]]))
        calibs = calibrator.calibrate(keypoints.cpu().numpy(),
                                      orientations.cpu().numpy())
        calib_est = calibrator.filter_and_average(calibs)
        print(calib_est)


if __name__ == '__main__':
    unittest.main()
