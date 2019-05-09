# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import cv2
import skvideo.io


class ImageSource(object):
    """
    Source of natural images to be added to a simulated environment.
    """
    def get_image(self):
        """
        Returns:
            an RGB image of [h, w, 3] with a fixed shape.
        """
        pass

    def reset(self):
        """ Called when an episode ends. """
        pass


class FixedColorSource(ImageSource):
    def __init__(self, shape, color):
        """
        Args:
            shape: [h, w]
            color: a 3-tuple
        """
        self.arr = np.zeros((shape[0], shape[1], 3))
        self.arr[:, :] = color

    def get_image(self):
        return np.copy(self.arr)


class RandomColorSource(ImageSource):
    def __init__(self, shape):
        """
        Args:
            shape: [h, w]
        """
        self.shape = shape
        self.reset()

    def reset(self):
        self._color = np.random.randint(0, 256, size=(3,))

    def get_image(self):
        arr = np.zeros((self.shape[0], self.shape[1], 3))
        arr[:, :] = self._color
        return arr


class NoiseSource(ImageSource):
    def __init__(self, shape, strength=50):
        """
        Args:
            shape: [h, w]
            strength (int): the strength of noise, in range [0, 255]
        """
        self.shape = shape
        self.strength = strength

    def get_image(self):
        return np.maximum(np.random.randn(
            self.shape[0], self.shape[1], 3) * self.strength, 0)


class RandomImageSource(ImageSource):
    def __init__(self, shape, filelist):
        """
        Args:
            shape: [h, w]
            filelist: a list of image files
        """
        self.shape_wh = shape[::-1]
        self.filelist = filelist
        self.reset()

    def reset(self):
        fname = np.random.choice(self.filelist)
        im = cv2.imread(fname, cv2.IMREAD_COLOR)
        im = im[:, :, ::-1]
        im = cv2.resize(im, self.shape_wh)
        self._im = im

    def get_image(self):
        return self._im


class RandomVideoSource(ImageSource):
    def __init__(self, shape, filelist):
        """
        Args:
            shape: [h, w]
            filelist: a list of video files
        """
        self.shape_wh = shape[::-1]
        self.filelist = filelist
        self.reset()

    def reset(self):
        fname = np.random.choice(self.filelist)
        self.frames = skvideo.io.vread(fname)
        self.frame_idx = 0

    def get_image(self):
        if self.frame_idx >= self.frames.shape[0]:
            self.reset()
        im = self.frames[self.frame_idx][:, :, ::-1]
        self.frame_idx += 1
        im = im[:, :, ::-1]
        im = cv2.resize(im, self.shape_wh)
        return im
