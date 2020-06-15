import argparse
import glob
import os

import cv2
import numpy as np
from tqdm import tqdm

from assets.camera_paramters import camera_matrix, distortion_coefficients
from calibration import Calibration
from detection.keypoint import KeypointDetector
from detection.object import ObjectDetector


def main():
    parser = argparse.ArgumentParser('Calibrate camera from image files')
    parser.add_argument('--input_dir',
                        '-i',
                        required=True,
                        help='path to image directory')
    parser.add_argument(
        '--object_checkpoint',
        default=
        '/home/yuthon/Workspace/Yet-Another-EfficientDet-Pytorch/weights',
        help='checkpoint directory for object detector')
    parser.add_argument(
        '--keypoint_checkpoint',
        default=
        '/home/yuthon/Workspace/Vehicle_Key_Point_Orientation_Estimation/'
        'checkpoints/stage2/best_fine_kp_checkpoint.pth.tar',
        help='checkpoint path for keypoint detector')
    parser.add_argument(
        '--keypoint_mean_std',
        default=
        '/home/yuthon/Workspace/Vehicle_Key_Point_Orientation_Estimation/'
        'data/VeRi/mean.pth.tar',
        help='mean-std path for keypoint datasets')
    opt = parser.parse_args()

    print('Initializing')
    object_detector = ObjectDetector(checkpoint_dir=opt.object_checkpoint)
    keypoint_detector = KeypointDetector(
        checkpoint_path=opt.keypoint_checkpoint,
        mean_std_path=opt.keypoint_mean_std)
    calibrator = Calibration(camera_matrix=camera_matrix,
                             dist_coeff=distortion_coefficients)

    print('Detecting')
    image_files = glob.glob(os.path.join(opt.input_dir, '*.jpg'))
    calib_candidates = []
    for image_idx, image_file in enumerate(tqdm(image_files)):
        if image_idx >= 10:
            break
        image = cv2.imread(image_file)
        detections = object_detector.detect(frame=image)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        keypoints, orientations = keypoint_detector.detect(image_rgb,
                                                           detections[0],
                                                           visualize=True)
        calibs = calibrator.calibrate(keypoints.cpu().numpy(),
                                      orientations.cpu().numpy())
        if calibs.shape[0] != 0:
            calib_candidates.append(calibs)

    print('Averaging')
    calib_candidates = np.concatenate(calib_candidates, axis=0)
    calib_est = calibrator.filter_and_average(calib_candidates)
    print(calib_est)


if __name__ == '__main__':
    main()
