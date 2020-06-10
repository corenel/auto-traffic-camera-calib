import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from detection.keypoint.datasets import TensorDetectionDataset
from detection.keypoint.models import KeyPointModel
from detection.keypoint.transforms import Denormalize
from detection.keypoint.utils import get_preds, visualize_results, process_keypoints


class KeypointDetector:
    def __init__(self, checkpoint_path, mean_std_path) -> None:
        super().__init__()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net = KeyPointModel()
        self.net = self.net.to(device)

        print('Total number of Parameters = %s' %
              sum(p.numel() for p in self.net.parameters()))
        print('Total number of trainable Parameters = %s' %
              sum(p.numel() for p in self.net.parameters() if p.requires_grad))

        checkpoint = torch.load(checkpoint_path)
        self.net.load_state_dict(checkpoint['net_state_dict'])
        print('Resumed Checkpoint :{} is Loaded!'.format(checkpoint_path))

        # if torch.cuda.device_count() > 1 and args.mGPU:
        #     net = torch.nn.DataParallel(net)
        self.net.eval()

        mean_std = torch.load(mean_std_path)
        self.mean = mean_std['mean'].numpy()
        self.std = mean_std['std'].numpy()
        self.denormalize = Denormalize(self.mean, self.std)
        self.dataset = TensorDetectionDataset(image=None,
                                              bboxes=None,
                                              mean=self.mean,
                                              std=self.std)

    def detect(self, image, bboxes, visualize=False):
        # image should be in RGB format
        self.dataset.initialize(image=image, bboxes=bboxes)
        test_loader = DataLoader(self.dataset,
                                 shuffle=False,
                                 batch_size=16,
                                 num_workers=16)
        return self.do_detect(test_loader, visualize=visualize, bboxes=bboxes)

    def do_detect(self, data_loader, visualize=False, bboxes=None):
        orientations = []
        keypoints = []
        with torch.no_grad():
            with tqdm(total=len(data_loader),
                      ncols=0,
                      file=sys.stdout,
                      desc='Stage 2 Evaluation...') as pbar:
                for i, in_batch in enumerate(data_loader):
                    image_in1, image_in2 = in_batch
                    if torch.cuda.is_available():
                        image_in1, image_in2 = \
                            image_in1.cuda(), image_in2.cuda()

                    coarse_kp, fine_kp, orientation = self.net(
                        image_in1, image_in2)

                    predicted_keypoints = get_preds(fine_kp).cpu()
                    keypoints.append(predicted_keypoints)
                    # print('fine_kp: {}'.format(fine_kp.shape))
                    print('predicted_keypoints: {}'.format(
                        predicted_keypoints.shape))

                    _, predicted_orientations = torch.max(orientation.data, 1)
                    predicted_orientations = predicted_orientations.cpu()
                    orientations.append(predicted_orientations)
                    print('predicted_orientations: {}'.format(
                        predicted_orientations.shape))

                    if visualize:
                        print(predicted_orientations)
                        visualize_results(predicted_keypoints,
                                          predicted_orientations, image_in1, i,
                                          self.denormalize)

                    pbar.update()
        keypoints = torch.cat(keypoints, dim=0)
        orientations = torch.cat(orientations, dim=0)
        if bboxes is not None:
            keypoints = process_keypoints(bboxes, keypoints, orientations)
        return keypoints, orientations
