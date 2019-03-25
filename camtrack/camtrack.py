#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Tuple

import numpy as np

from cv2 import findFundamentalMat, findHomography, RANSAC, findEssentialMat, decomposeEssentialMat, solvePnPRansac

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import *


def _get_first_pose(corner_storage, instrinsic_mat, triangulation_params):
    corners1 = corner_storage[0]
    frame_results = []
    for k in range(1, len(corner_storage)):
        corners2 = corner_storage[k]
        correspondence = build_correspondences(corners1, corners2)

        if len(correspondence.points_1) < 5:
            continue

        E, mask = findEssentialMat(correspondence.points_1, correspondence.points_2, instrinsic_mat)

        if not (E.shape[0] == 3 and E.shape[1] == 3):
            continue

        nonzero = np.nonzero(1 - mask.reshape(-1))[0]
        filtered_correspondences = build_correspondences(corners1, corners2, nonzero)
        inliers = len(nonzero)

        if inliers / len(correspondence.points_1) > 0.8:
            continue

        R1, R2, t1 = decomposeEssentialMat(E)
        t1 = t1.reshape(-1)

        variants = [Pose(R1.T, R1.T @ t1), Pose(R1.T, R1.T @ -t1), Pose(R2.T, R2.T @ t1), Pose(R2.T, R2.T @ -t1)]
        sizes = []
        for p in variants:
            _, ids = triangulate_correspondences(filtered_correspondences, eye3x4(), pose_to_view_mat3x4(p), instrinsic_mat, triangulation_params)
            sizes.append(len(ids))

        id = np.argmax(sizes)

        frame_results.append((k, variants[id], sizes[id]))
    return max(*frame_results, key=lambda x: x[2])


def _build_cloud(corners1, frame_mat1, corners2, frame_mat2, instrinsic_mat, triangulation_params):
    correspondences = build_correspondences(corners1, corners2)
    poss, ids = triangulate_correspondences(correspondences, frame_mat1, frame_mat2,instrinsic_mat, triangulation_params)
    return list(zip(ids, poss))


def _track_camera(corner_storage: CornerStorage, intrinsic_mat: np.ndarray) -> Tuple[List[np.ndarray], PointCloudBuilder]:
    tr_params = TriangulationParameters(max_reprojection_error=1., min_triangulation_angle_deg=4., min_depth=0.1)
    frame_matrices = [None] * len(corner_storage)
    pt_id2pos = {}
    not_added_frames = set(range(len(corner_storage)))

    ### First frame
    i, pose, _ = _get_first_pose(corner_storage, intrinsic_mat, tr_params)
    frame_matrices[0] = eye3x4()
    frame_matrices[i] = pose_to_view_mat3x4(pose)
    not_added_frames.remove(0)
    not_added_frames.remove(i)

    pts = _build_cloud(corner_storage[0], frame_matrices[0], corner_storage[i], frame_matrices[i], intrinsic_mat, tr_params)
    for pi, pos in pts:
        pt_id2pos[pi] = pos

    ### Other frames
    while len(not_added_frames) > 0:
        # Frame with best possible quality
        qualities = []
        ids = []
        for i in not_added_frames:
            quality = 0
            for pi in corner_storage[i].ids:
                if pi[0] in pt_id2pos:
                    quality += 1
            qualities.append(quality)
            ids.append(i)

        # Compute frame mat
        retval = False
        best_id = ids[0]
        while not retval:
            best_id = np.argmax(qualities)
            qualities[best_id] = -1
            best_id = ids[best_id]

            pts_ids = []
            pts = []
            for i, pi in enumerate(corner_storage[best_id].ids):
                if pi[0] in pt_id2pos:
                    pts_ids.append(pi[0])
                    pts.append(corner_storage[best_id].points[i])
            pts_3d = []
            for pi in pts_ids:
                pts_3d.append(pt_id2pos[pi])

            retval, rvec, tvec, inliers = solvePnPRansac(np.array(pts_3d).reshape(-1, 1, 3), np.array(pts).reshape(-1, 1, 2), intrinsic_mat, None)

            if not retval:
                continue

            inliers = set(inliers.flatten())
            removed_ctn = 0
            for i, pi in enumerate(pts_ids):
                if i not in inliers:
                    pt_id2pos.pop(pi)
                    removed_ctn += 1
            frame_matrices[best_id] = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
            not_added_frames.remove(best_id)

        # Update 3d points
        added_points = 0
        for delta in range(max(len(corner_storage) - best_id, best_id), 0, -1):
            alt_id = best_id + delta
            if alt_id < len(corner_storage) and frame_matrices[alt_id] is not None:
                pts = _build_cloud(corner_storage[alt_id], frame_matrices[alt_id], corner_storage[best_id],
                                   frame_matrices[best_id], intrinsic_mat, tr_params)
                for pi, pos in pts:
                    if pi not in pt_id2pos:
                        pt_id2pos[pi] = pos
                        added_points += 1

            alt_id = best_id - delta
            if alt_id > 0 and frame_matrices[alt_id] is not None:
                pts = _build_cloud(corner_storage[alt_id], frame_matrices[alt_id], corner_storage[best_id],
                                   frame_matrices[best_id], intrinsic_mat, tr_params)
                for pi, pos in pts:
                    if pi not in pt_id2pos:
                        pt_id2pos[pi] = pos
                        added_points += 1
        print("Added frame", best_id, "with", len(inliers), "inliers.", removed_ctn,
              "points were removed and", added_points, "were added.")

    ids = np.array(list(pt_id2pos.keys()))

    return frame_matrices, PointCloudBuilder(ids, np.array([pt_id2pos[pi] for pi in ids]))


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str) \
        -> Tuple[List[Pose], PointCloud]:
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )
    view_mats, point_cloud_builder = _track_camera(
        corner_storage,
        intrinsic_mat
    )
    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    create_cli(track_and_calc_colors)()
