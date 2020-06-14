from warnings import warn

import cv2
import numpy as np
from scipy.interpolate import griddata


def p2e(projective):
    """
    Convert 2d or 3d projective to euclidean coordinates.

    :param projective: projective coordinate(s)
    :type projective: numpy.array, shape=(3 or 4, n)
    :return: euclidean coordinate(s)
    :rtype: numpy.array, shape=(2 or 3, n)
    """
    assert (type(projective) == np.ndarray)
    assert ((projective.shape[0] == 4) | (projective.shape[0] == 3))
    return (projective / projective[-1, :])[0:-1, :]


def e2p(euclidean):
    """
    Convert 2d or 3d euclidean to projective coordinates.

    :param euclidean: projective coordinate(s)
    :type euclidean: numpy.array, shape=(2 or 3, n)
    :return: projective coordinate(s)
    :rtype: numpy.array, shape=(3 or 4, n)
    """
    assert (type(euclidean) == np.ndarray)
    assert ((euclidean.shape[0] == 3) | (euclidean.shape[0] == 2))
    return np.vstack((euclidean, np.ones((1, euclidean.shape[1]))))


class Camera:
    """
    Projective camera model

    Refer to https://github.com/smidm/camera.py
    """

    def __init__(self):
        # camera intrinsic parameters
        self.K = np.eye(3)
        self.R = np.eye(3)
        self.t = np.zeros((3, 1))
        self.P = None
        self.update_P()
        self.opencv_dist_coeff = None
        self.calibration_type = 'opencv'

    def set_K(self, K):
        """
        Set K and update P.

        :param K: intrinsic camera parameters
        :type K: numpy.array, shape=(3, 3)
        """
        self.K = K
        self.update_P()

    def set_R(self, R):
        """
        Set camera extrinsic parameters and updates P.

        :param R: camera extrinsic parameters matrix
        :type R: numpy.array, shape=(3, 3)
        """
        self.R = R
        self.update_P()

    def set_t(self, t):
        """
        Set camera translation and updates P.

        :param t: camera translation vector
        :type t: numpy.array, shape=(3, 1)
        """
        self.t = t
        self.update_P()

    def update_P(self):
        """Update camera P matrix from K, R and t."""
        self.P = self.K.dot(np.hstack((self.R, self.t)))

    def undistort_image(self, img, K_undistortion=None):
        """
        Transform grayscale image such that radial distortion is removed.

        :param img: input image
        :type img: np.ndarray, shape=(n, m) or (n, m, 3)
        :param K_undistortion: camera matrix for undistorted view, None for self.K
        :type K_undistortion: array-like, shape=(3, 3)
        :return: transformed image
        :rtype: np.ndarray, shape=(n, m) or (n, m, 3)
        """
        if K_undistortion is None:
            K_undistortion = self.K
        if self.calibration_type == 'opencv':
            return cv2.undistort(img,
                                 self.K,
                                 self.opencv_dist_coeff,
                                 newCameraMatrix=K_undistortion)
        elif self.calibration_type == 'opencv_fisheye':
            return cv2.fisheye.undistortImage(img,
                                              self.K,
                                              self.opencv_dist_coeff,
                                              Knew=K_undistortion)
        else:
            xx, yy = np.meshgrid(np.arange(img.shape[1]),
                                 np.arange(img.shape[0]))
            img_coords = np.array([xx.ravel(), yy.ravel()])
            y_l = self.undistort(img_coords, K_undistortion)
            if img.ndim == 2:
                return griddata(y_l.T,
                                img.ravel(), (xx, yy),
                                fill_value=0,
                                method='linear')
            else:
                channels = [
                    griddata(y_l.T,
                             img[:, :, i].ravel(), (xx, yy),
                             fill_value=0,
                             method='linear') for i in range(img.shape[2])
                ]
                return np.dstack(channels)

    def undistort(self, distorted_image_coords, K_undistortion=None):
        """
        Remove distortion from image coordinates.

        :param distorted_image_coords: real image coordinates
        :type distorted_image_coords: numpy.array, shape=(2, n)
        :param K_undistortion: camera matrix for undistorted view, None for self.K
        :type K_undistortion: array-like, shape=(3, 3)
        :return: linear image coordinates
        :rtype: numpy.array, shape=(2, n)
        """
        assert distorted_image_coords.shape[0] == 2
        assert distorted_image_coords.ndim == 2
        if K_undistortion is None:
            K_undistortion = self.K
        if self.calibration_type == 'opencv':
            undistorted_image_coords = cv2.undistortPoints(
                distorted_image_coords.T.reshape((1, -1, 2)),
                self.K,
                self.opencv_dist_coeff,
                P=K_undistortion).reshape(-1, 2).T
        elif self.calibration_type == 'opencv_fisheye':
            undistorted_image_coords = cv2.fisheye.undistortPoints(
                distorted_image_coords.T.reshape((1, -1, 2)),
                self.K,
                self.opencv_dist_coeff,
                P=K_undistortion).reshape(-1, 2).T
        else:
            warn('undistortion not implemented')
            undistorted_image_coords = distorted_image_coords
        assert undistorted_image_coords.shape[0] == 2
        assert undistorted_image_coords.ndim == 2
        return undistorted_image_coords

    def calibrate_extrinsic(self, object_points, image_points):
        ret, rvec, tvec = cv2.solvePnP(object_points, image_points, self.K,
                                       self.opencv_dist_coeff)
        rmat = cv2.Rodrigues(rvec)[0]
        # self.set_R(rmat)
        # self.set_t(tvec)
        return ret, rmat, tvec

    def world_to_image(self, world):
        """
        Project world coordinates to image coordinates.

        :param world: world points in 3d projective or euclidean coordinates
        :type world: numpy.array, shape=(3 or 4, n)
        :return: projective image coordinates
        :rtype: numpy.array, shape=(3, n)
        """
        assert (type(world) == np.ndarray)
        if self.calibration_type == 'opencv' or self.calibration_type == 'opencv_fisheye':
            if world.shape[0] == 4:
                world = p2e(world)
            if self.calibration_type == 'opencv':
                distorted_image_coords = cv2.projectPoints(
                    world.T, self.R, self.t, self.K,
                    self.opencv_dist_coeff)[0].reshape(-1, 2).T
            else:
                distorted_image_coords = cv2.fisheye.projectPoints(
                    world.T.reshape((1, -1, 3)),
                    cv2.Rodrigues(self.R)[0], self.t, self.K,
                    self.opencv_dist_coeff)[0].reshape(-1, 2).T
            return e2p(distorted_image_coords)
        else:
            warn('world_to_image not implemented')

    def image_to_world(self, image_px, z, undistort=True):
        """
        Project image points with defined world z to world coordinates.

        :param image_px: image points
        :type image_px: numpy.array, shape=(2 or 3, n)
        :param z: world z coordinate of the projected image points
        :type z: float
        :param undistort: do undistortion
        :type undistort: bool
        :return: n projective world coordinates
        :rtype: numpy.array, shape=(3, n)
        """
        if image_px.shape[0] == 3:
            image_px = p2e(image_px)
        image_undistorted = self.undistort(image_px) if undistort else image_px
        tmpP = np.hstack(
            (self.P[:, [0, 1]],
             self.P[:, 2, np.newaxis] * z + self.P[:, 3, np.newaxis]))
        world_xy = p2e(np.linalg.pinv(tmpP).dot(e2p(image_undistorted)))
        return np.vstack((world_xy, z * np.ones(image_px.shape[1])))
