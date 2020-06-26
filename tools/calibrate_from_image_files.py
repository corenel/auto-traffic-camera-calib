import argparse
import glob
import os
import pickle

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from calibration import Calibration
from detection.keypoint import KeypointDetector
from detection.keypoint.utils import visualize_keypoint
from detection.object import ObjectDetector
from detection.object.utils.utils import visualize_bbox
from misc.utils import image_resize


def main():
    parser = argparse.ArgumentParser('Calibrate camera from image files')
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('--priors',
                        '-p',
                        type=str,
                        default='assets/vehicle_keypoint_priors.yaml')
    parser.add_argument('--input_dir',
                        '-i',
                        type=str,
                        required=True,
                        help='path to image directory')
    parser.add_argument(
        '--object_checkpoint',
        type=str,
        default=
        '/home/yuthon/Workspace/Yet-Another-EfficientDet-Pytorch/weights',
        help='checkpoint directory for object detector')
    parser.add_argument(
        '--keypoint_checkpoint',
        type=str,
        default=
        '/home/yuthon/Workspace/Vehicle_Key_Point_Orientation_Estimation/'
        'checkpoints/stage2/best_fine_kp_checkpoint.pth.tar',
        help='checkpoint path for keypoint detector')
    parser.add_argument(
        '--keypoint_mean_std',
        type=str,
        default=
        '/home/yuthon/Workspace/Vehicle_Key_Point_Orientation_Estimation/'
        'data/VeRi/mean.pth.tar',
        help='mean-std path for keypoint datasets')
    parser.add_argument('--visualize', action='store_true')
    opt = parser.parse_args()

    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    with open(opt.priors) as f:
        priors = yaml.load(f, Loader=yaml.FullLoader)['vehicle_keypoints']

    print('Initializing')
    object_detector = ObjectDetector(checkpoint_dir=opt.object_checkpoint)
    keypoint_detector = KeypointDetector(
        checkpoint_path=opt.keypoint_checkpoint,
        mean_std_path=opt.keypoint_mean_std)
    calibrator = Calibration(camera_matrix=np.array(
        config['camera']['intrinsic']),
                             dist_coeff=config['camera']['distortion'],
                             priors=priors)

    print('Detecting')
    image_files = glob.glob(os.path.join(opt.input_dir, '*.bmp'))
    calib_candidates = []
    for image_idx, image_file in enumerate(tqdm(image_files)):
        # if image_idx >= 10:
        #     break
        # detection
        image = cv2.imread(image_file)
        detections = object_detector.detect(frame=image, visualize=False)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        keypoints, orientations = keypoint_detector.detect(image_rgb,
                                                           detections[0],
                                                           visualize=False)
        # visualization
        if opt.visualize:
            result = image.copy()
            for box_idx in range(keypoints.shape[0]):
                result = visualize_bbox(
                    image=result,
                    roi=detections[0]['rois'][box_idx],
                    class_id=detections[0]['class_ids'][box_idx],
                    score=detections[0]['scores'][box_idx],
                    obj_list=object_detector.obj_list)
                for kp_idx in range(keypoints.shape[1]):
                    result = visualize_keypoint(
                        image=result,
                        coord=keypoints[box_idx][kp_idx],
                        kp_idx=kp_idx,
                        orientation=orientations[box_idx])
            cv2.imshow('detection result', image_resize(result, width=1280))
            cv2.waitKey(0)
        # calibration
        calibs = calibrator.calibrate(keypoints.cpu().numpy(),
                                      orientations.cpu().numpy())
        if calibs.shape[0] != 0:
            calib_candidates.append(calibs)

    print('Averaging')
    calib_candidates = np.concatenate(calib_candidates, axis=0)
    calib_est = calibrator.filter_and_average(calib_candidates)
    print(calib_est)

    with open('calib.pkl', 'wb') as f:
        pickle.dump({
            'candidates': calib_candidates,
            'estimation': calib_est
        }, f)


if __name__ == '__main__':
    main()
