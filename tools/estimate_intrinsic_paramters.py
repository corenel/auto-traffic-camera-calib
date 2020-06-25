import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Estimate camera intrinsic matrix')
    parser.add_argument('--cmos_width',
                        default=7.40,
                        type=float,
                        help='horizontal length of camera CMOS (in mm)')
    parser.add_argument('--cmos_height',
                        default=5.55,
                        type=float,
                        help='vertical length of camera CMOS (in mm)')
    parser.add_argument('--focal_length',
                        required=True,
                        type=float,
                        help='focal length of camera (in mm)')
    parser.add_argument('--image_width',
                        default=3840,
                        type=int,
                        help='image width of captured frame')
    parser.add_argument('--image_height',
                        default=2160,
                        type=int,
                        help='image height of captured frame')
    opt = parser.parse_args()

    K = np.eye(3, 3)
    # f_x
    K[0][0] = opt.image_width * opt.focal_length / opt.cmos_width
    # f_y
    K[1][1] = opt.image_height * opt.focal_length / opt.cmos_height
    # c_x
    K[0][2] = opt.image_width / 2
    # c_y
    K[1][2] = opt.image_height / 2
    print(K)
