import time
from collections import defaultdict
import os
import torch
import cv2
import numpy as np
from torch.backends import cudnn
from detection.object.backbone import EfficientDetBackbone
from detection.object.efficientdet.utils import BBoxTransform, ClipBoxes
from detection.object.utils.utils import preprocess, invert_affine, postprocess, preprocess_video
from detection.object.efficientdet.config import COCO_CLASSES


class ObjectDetector:
    obj_list = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', '', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack',
        'umbrella', '', '', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '',
        'toilet', '', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]
    valid_obj_list = [2]
    valid_width_ratio = 0.05
    valid_height_ratio = 0.05
    # tf bi-linear interpolation is different from any other's, just make do
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]

    def __init__(self, checkpoint_dir) -> None:
        super().__init__()
        compound_coef = 2
        force_input_size = None  # set None to use default size

        self.threshold = 0.3
        self.iou_threshold = 0.2

        self.use_cuda = True
        self.use_float16 = False
        cudnn.fastest = True
        cudnn.benchmark = True

        self.input_size = self.input_sizes[
            compound_coef] if force_input_size is None else force_input_size

        # load model
        self.model = EfficientDetBackbone(compound_coef=compound_coef,
                                          num_classes=len(self.obj_list))
        self.model.load_state_dict(
            torch.load(
                os.path.join(checkpoint_dir,
                             f'efficientdet-d{compound_coef}.pth')))
        self.model.requires_grad_(False)
        self.model.eval()

        if self.use_cuda:
            self.model = self.model.cuda()
        if self.use_float16:
            self.model = self.model.half()

        # Box
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()

    def visualize(self, preds, imgs):
        for i in range(len(imgs)):
            if len(preds[i]['rois']) == 0:
                return imgs[i]

            for j in range(len(preds[i]['rois'])):
                (x1, y1, x2, y2) = preds[i]['rois'][j].astype(np.int)
                cv2.rectangle(imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
                obj = self.obj_list[preds[i]['class_ids'][j]]
                score = float(preds[i]['scores'][j])

                cv2.putText(imgs[i], '{}, {:.3f}'.format(obj,
                                                         score), (x1, y1 + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            return imgs[i]

    def filter_detections(self, detections, image_width, image_height):
        # only filter by class id
        # mask = np.in1d(detections[det_idx]['class_ids'], np.asarray(self.valid_obj_list))
        # detections[det_idx]['rois'] = detections[det_idx]['rois'][mask]
        # detections[det_idx]['class_ids'] = detections[det_idx]['class_ids'][mask]
        # detections[det_idx]['scores'] = detections[det_idx]['scores'][mask]

        for det_idx in range(len(detections)):
            rois, class_ids, scores = [], [], []
            for box_idx in range(detections[det_idx]['rois'].shape[0]):
                class_id = detections[det_idx]['class_ids'][box_idx]
                box = detections[det_idx]['rois'][box_idx]
                box_width = box[2] - box[0]
                box_height = box[3] - box[1]
                if class_id in self.valid_obj_list and \
                        box_width >= image_width * self.valid_width_ratio and \
                        box_height >= image_height * self.valid_height_ratio:
                    rois.append(box)
                    class_ids.append(class_id)
                    scores.append(detections[det_idx]['scores'][box_idx])
            detections[det_idx] = {
                'rois': np.stack(rois) if len(rois) > 0 else np.empty((0, 4)),
                'class_ids': np.stack(class_ids) if len(class_ids) > 0 else np.empty((0, 1)),
                'scores': np.stack(scores) if len(scores) > 0 else np.empty((0, 1))
            }
        return detections

    def detect(self, frame, visualize=False):
        # frame pre-processing
        ori_imgs, framed_imgs, framed_metas = preprocess_video(
            frame, max_size=self.input_size)

        if self.use_cuda:
            x = torch.stack(
                [torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
        else:
            x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

        x = x.to(
            torch.float32 if not self.use_float16 else torch.float16).permute(
            0, 3, 1, 2)

        # model predict
        with torch.no_grad():
            features, regression, classification, anchors = self.model(x)

            detections = postprocess(x, anchors, regression, classification,
                                     self.regressBoxes, self.clipBoxes,
                                     self.threshold, self.iou_threshold)
        detections = self.filter_detections(detections,
                                            image_width=framed_metas[0][0],
                                            image_height=framed_metas[0][1])
        detections = invert_affine(framed_metas, detections)

        # result
        if visualize:
            img_show = self.visualize(detections, ori_imgs)
            # cv2.imshow('detection result', img_show)
            # cv2.waitKey(0)
            cv2.imwrite('/tmp/result.png', img_show)

        return detections
