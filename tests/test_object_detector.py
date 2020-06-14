import unittest

import cv2

from detection.object import ObjectDetector

CHECKPOINT_DIR = '/home/yuthon/Workspace/Yet-Another-EfficientDet-Pytorch/weights'
TEST_IMAGE_PATH = '/media/Data/supcon-traffic-surveillance/frames/100000/100000_00001.jpg'

image = cv2.imread(TEST_IMAGE_PATH)
detector = ObjectDetector(checkpoint_dir=CHECKPOINT_DIR)


class TestObjectDetector(unittest.TestCase):
    def test_inference_from_image_file(self):
        out = detector.detect(frame=image, visualize=True)


if __name__ == '__main__':
    unittest.main()
