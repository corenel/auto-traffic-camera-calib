import unittest

from torch.utils.data import DataLoader

from detection.keypoint import KeypointDetector
from detection.keypoint.datasets import TestDataset

CHECKPOINT_PATH = '/home/yuthon/Workspace/Vehicle_Key_Point_Orientation_Estimation/' \
                  'checkpoints/stage2/best_fine_kp_checkpoint.pth.tar'
MEAN_STD_PATH = '/home/yuthon/Workspace/Vehicle_Key_Point_Orientation_Estimation/' \
                'data/VeRi/mean.pth.tar'
TEST_IMAGE_DIR = '/home/yuthon/Downloads/test_images'


class TestKeypointDetector(unittest.TestCase):
    def test_inference(self):
        model = KeypointDetector(checkpoint_path=CHECKPOINT_PATH,
                                 mean_std_path=MEAN_STD_PATH)
        test_set = TestDataset(root=TEST_IMAGE_DIR,
                               mean=model.mean,
                               std=model.std)
        test_loader = DataLoader(test_set,
                                 shuffle=False,
                                 batch_size=16,
                                 num_workers=16)
        keypoints, orientations = model.do_detect(test_loader, visualize=True)
        print(keypoints.shape)
        print(orientations.shape)
        print(keypoints[0])


if __name__ == '__main__':
    unittest.main()
