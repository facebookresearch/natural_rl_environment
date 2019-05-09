# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


class BackgroundMatting(object):
    """
    Produce a mask of a given image which will be replaced by natural signals.
    """
    def get_mask(self, img):
        """
        Take an image of [H, W, 3]. Returns a mask of [H, W]
        """
        raise NotImplementedError()


class BackgroundMattingWithColor(BackgroundMatting):
    """
    Produce a mask by masking the given color. This is a simple strategy
    but effective for many games.
    """
    def __init__(self, color):
        """
        Args:
            color: a (r, g, b) tuple
        """
        self._color = color

    def get_mask(self, img):
        return img == self._color
