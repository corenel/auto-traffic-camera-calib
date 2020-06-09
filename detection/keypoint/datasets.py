import os

import numpy as np
import scipy.ndimage as ndimage
from skimage import io
from torch.utils.data import Dataset

from . import transforms

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
                  '.tiff', '.webp')


def has_file_allowed_extension(filename, extensions):
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(directory, extensions):
    def is_valid_file(x):
        return has_file_allowed_extension(x, extensions)

    instances = []
    directory = os.path.expanduser(directory)
    for root, _, fnames in sorted(os.walk(directory, followlinks=True)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            if is_valid_file(path):
                instances.append(path)
    return sorted(instances)


class TestDataset(Dataset):
    def __init__(self, root, mean, std):
        super(TestDataset, self).__init__()

        self.struct = ndimage.generate_binary_structure(2, 1)
        self.Normalize = transforms.Normalize(mean, std)
        self.Rescale = transforms.Rescale()
        self.ToTensor = transforms.ToTensor()

        self.root = root
        self.image_files = make_dataset(root, extensions=IMG_EXTENSIONS)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        if not os.path.exists(self.image_files[index]):
            im_path = os.path.join(self.root, self.image_files[index])
        else:
            im_path = self.image_files[index]
        image = io.imread(im_path)
        image = image.astype(np.float)

        image = self.Normalize(image)
        image_in1, image_in2 = self.Rescale(image)

        return (self.ToTensor(image_in1.transpose(2, 0, 1)),
                self.ToTensor(image_in2.transpose(2, 0, 1)))


class TensorDetectionDataset(Dataset):
    def __init__(self, image, bboxes, mean, std):
        super().__init__()

        self.Normalize = transforms.Normalize(mean, std)
        self.Rescale = transforms.Rescale()
        self.ToTensor = transforms.ToTensor()

        self.image, self.bboxes = None, None
        self.initialize(image, bboxes)

    def initialize(self, image, bboxes):
        self.image, self.bboxes = image, bboxes

    @staticmethod
    def crop_image(image, bbox):
        if len(image.shape) == 3 and image.shape[2] == 3:
            return image[int(bbox[1]):int(bbox[3]),
                   int(bbox[0]):int(bbox[2]), :]
        elif len(image.shape) == 3 and image.shape[0] == 3:
            return image[:,
                   int(bbox[1]):int(bbox[3]),
                   int(bbox[0]):int(bbox[2])]
        else:
            raise RuntimeError(
                f'Unsupported image shape to crop: {image.shape}')

    def __len__(self):
        if self.bboxes is not None:
            return self.bboxes['rois'].shape[0]
        else:
            return 0

    def __getitem__(self, index):
        bbox = self.bboxes['rois'][index]
        image = self.crop_image(self.image, bbox).copy().astype(np.float)
        image = self.Normalize(image)
        image_in1, image_in2 = self.Rescale(image)

        return (self.ToTensor(image_in1.transpose(2, 0, 1)),
                self.ToTensor(image_in2.transpose(2, 0, 1)))
