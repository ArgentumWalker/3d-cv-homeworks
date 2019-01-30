#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims
import queue

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli
from sklearn.neighbors import KDTree

__DEBUG__ = True


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


class _TrackedCorner:
    def __init__(self, id, position, deg=4, positions=11):
        self.id = id
        self.positions = queue.deque(maxlen=max(positions, deg+1))
        self.deg = deg
        self.approximation_coefficients = []
        self.update(position)
        self.__position = None

    def get_raw_position(self):
        return self.positions[-1]

    def get_position(self):
        if self.__position is None:
            self.__position = self._predict(len(self.positions) - 1)
        return self.__position

    def update(self, position):
        self.positions.append(position)
        self.approximation_coefficients = np.polyfit([i for i in range(len(self.positions))],
                                                      np.array(self.positions),
                                                      min(self.deg, len(self.positions) - 1))
        self.__position = None

    def predict_next(self):
        return self._predict(len(self.positions))

    def _predict(self, x):
        deg = len(self.approximation_coefficients)
        x_pow = np.array([x**(deg - i) for i in range(1, deg)] + [1])
        return x_pow.dot(self.approximation_coefficients)


class _CornerTracker:
    def __init__(self, first_frame, max_corners, quality, min_distance, lkWinSize, lkMaxLevel, block_size=4):
        self.block_size = block_size
        self.min_distance = max(min_distance, self.block_size * 1.5)
        self.max_corners = max_corners
        self.quality = quality
        self.image_size = first_frame.shape

        self.corners = []
        self.corners_positions = []
        self.kdtree = None

        self._next_corner_index = 0
        self._old_grayscale = None
        self.lkWinSize = lkWinSize
        self.lkMaxLevel = lkMaxLevel
        self.update(first_frame)

    def update(self, grayscale):
        if __DEBUG__:
            print()
            print("Next step")
        if len(self.corners) > 0:
            self._track_corners(grayscale)
            self._filter_corners()
        if len(self.corners) < self.max_corners:
            self._find_corners(grayscale)
        print(">> Total corners:", len(self.corners))
        self._old_grayscale = grayscale

    def _filter_corners(self):
        remain_corners = []
        for corner in self.corners:
            if self.__check_corner(corner):
                remain_corners.append(corner)
        if __DEBUG__:
            print(">> Removed", len(self.corners) - len(remain_corners), "corners by filter")
        self.corners = remain_corners
        self._update_corner_positions()

    def __check_corner(self, corner):
        predicted_position = corner.predict_next()
        if predicted_position[0] < self.min_distance or predicted_position[0] >= self.image_size[1] - self.min_distance:
            return False
        if predicted_position[1] < self.min_distance or predicted_position[1] >= self.image_size[0] - self.min_distance:
            return False
        if self.kdtree is not None:
            dst, ids = self.kdtree.query([corner.get_position()], k=2)
            for id, d in zip(ids[0], dst[0]):
                if self.corners[id].id != corner.id and d < self.min_distance:
                    return False
        return True

    def _track_corners(self, grayscale):
        new_positions, st, err = cv2.calcOpticalFlowPyrLK((255 * self._old_grayscale).astype(np.uint8),
                                                 (255 * grayscale).astype(np.uint8), np.array([np.array(self.corners_positions)], dtype=np.float32).round(),
                                                 None, winSize=self.lkWinSize, maxLevel=self.lkMaxLevel)
        if __DEBUG__:
            print(">> Removed", len([i for i in st if i == 0]), "corners by tracker")
        self.corners = [corner for i, corner in enumerate(self.corners) if st[i][0] == 1]
        new_positions = [pos for i, pos in enumerate(new_positions[0]) if st[i][0] == 1]
        for corner, position in zip(self.corners, new_positions):
            corner.update(position)
        self._update_corner_positions()

    def _update_corner_positions(self):
        self.corners_positions = np.array([corner.get_position() for corner in self.corners])
        if len(self.corners) > 0:
            self.kdtree = KDTree(self.corners_positions, leaf_size=4, metric="l2")
        else:
            self.kdtree = None

    def _find_corners(self, grayscale):
        target_corners = self.max_corners - len(self.corners)
        raw_corners = cv2.goodFeaturesToTrack(grayscale, self.max_corners, self.quality, self.min_distance, self.block_size)
        new_corners = []
        for candidate_corner in raw_corners:
            if self.__check_corner(_TrackedCorner(-1, candidate_corner[0])):
                new_corners.append(candidate_corner)
                if len(new_corners) >= target_corners:
                    break
        if __DEBUG__:
            print(">> Added", len(new_corners), "new corners")
        for corner in new_corners:
            self.corners.append(_TrackedCorner(self._next_corner_index, corner[0]))
            self._next_corner_index += 1
        self._update_corner_positions()

    def get_current_corners(self):
        return FrameCorners(
            np.array([corner.id for corner in self.corners]),
            np.array([corner.get_position() for corner in self.corners]),
            np.array([self.block_size] * len(self.corners))
        )


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    image_0 = frame_sequence[0]
    corners_count = 500
    corners_tracker = _CornerTracker(image_0, corners_count, 0.05, max(*(image_0.shape[:2])) * 2 / corners_count, (20, 20), 3)
    builder.set_corners_at_frame(0, corners_tracker.get_current_corners())
    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        corners_tracker.update(image_1)
        builder.set_corners_at_frame(frame, corners_tracker.get_current_corners())


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
