import unittest
import cv2

from torch.utils.data import DataLoader

from detection.keypoint import KeypointDetector
from detection.keypoint.datasets import TestDataset
from .test_object_detector import detector as object_detector, image as object_image

from detection.keypoint.utils import visualize_keypoint
from detection.object.utils.utils import visualize_bbox

CHECKPOINT_PATH = '/home/yuthon/Workspace/Vehicle_Key_Point_Orientation_Estimation/' \
                  'checkpoints/stage2/best_fine_kp_checkpoint.pth.tar'
MEAN_STD_PATH = '/home/yuthon/Workspace/Vehicle_Key_Point_Orientation_Estimation/' \
                'data/VeRi/mean.pth.tar'
TEST_IMAGE_DIR = '/home/yuthon/Downloads/test_images'

model = KeypointDetector(checkpoint_path=CHECKPOINT_PATH,
                         mean_std_path=MEAN_STD_PATH)


class TestKeypointDetector(unittest.TestCase):
    def test_inference_from_bboxesimage_directory(self):
        test_set = TestDataset(root=TEST_IMAGE_DIR,
                               mean=model.mean,
                               std=model.std)
        test_loader = DataLoader(test_set,
                                 shuffle=False,
                                 batch_size=16,
                                 num_workers=16)
        keypoints, orientations = model.do_detect(test_loader, visualize=True)
        # print(keypoints.shape)
        # print(orientations.shape)
        # print(keypoints[0])

    def test_inference_from_bboxes(self):
        detections = object_detector.detect(frame=object_image)
        frame_rgb = cv2.cvtColor(object_image, cv2.COLOR_BGR2RGB)
        keypoints, orientations = model.detect(frame_rgb,
                                               detections[0],
                                               visualize=True)
        result = object_image.copy()
        for box_idx in range(keypoints.shape[0]):
            result = visualize_bbox(
                image=result,
                roi=detections[0]['rois'][box_idx],
                class_id=detections[0]['class_ids'][box_idx],
                score=detections[0]['scores'][box_idx],
                obj_list=object_detector.obj_list)
            for kp_idx in range(keypoints.shape[1]):
                result = visualize_keypoint(image=result,
                                            coord=keypoints[box_idx][kp_idx],
                                            kp_idx=kp_idx,
                                            orientation=orientations[box_idx])
        cv2.imwrite('/tmp/joint_result.png', result)

    if __name__ == '__main__':
        unittest.main()
