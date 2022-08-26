import math
from dataclasses import dataclass

import numpy as np
from numpy import ndarray
from sklearn.linear_model import LinearRegression
from load_poses import load_poses, PoseLandmark
from statistics import mean


@dataclass
class Point:
    x: float
    y: float
    z: float = 0.0


@dataclass
class TimeSeriesPoint:
    x: [float]
    y: [float]
    z: [float]


def derivative_of_two_joints(pose_set: [PoseLandmark], joint_indices: [int]) -> ndarray:
    return np.gradient([LinearRegression().fit(
        np.array([pose[index].x for index in joint_indices]).reshape(-1, 1),
        np.array([pose[index].y for index in joint_indices])
    ).coef_[0] for pose in pose_set
                        ], 1)


def slope_of_two_joints(pose_set: [PoseLandmark], joint_indices: [int]) -> [float]:
    return [LinearRegression().fit(
        np.array([pose[index].x for index in joint_indices]).reshape(-1, 1),
        np.array([pose[index].y for index in joint_indices])
    ).coef_[0] for pose in pose_set
            ]


def pose_centers(pose_set: [[PoseLandmark]]) -> [Point]:
    return [Point(
        x=mean([landmark.x for landmark in pose]),
        y=mean([landmark.y for landmark in pose]),
        z=mean([landmark.z for landmark in pose])
    ) for pose in pose_set]


def pose_centers_subset(pose_set: [PoseLandmark], key_points: [int]) -> [Point]:
    return [Point(
        x=mean([landmark.x for landmark in pose if landmark.joint_index in key_points]),
        y=mean([landmark.y for landmark in pose if landmark.joint_index in key_points]),
        z=mean([landmark.z for landmark in pose if landmark.joint_index in key_points])
    ) for pose in pose_set]


def size_by_regression() -> [float]:
    pose_set = full_pose_as_regression()
    centers = pose_centers(pose_set)
    return [
        mean([
            math.dist([point.x, point.y], [centers[frame].x, centers[frame].y])
            for point in pose
        ]) for frame, pose in enumerate(pose_set)
    ]


def full_pose_as_regression() -> [[PoseLandmark]]:
    mapped = load_poses()
    by_joint = [binomial_regression_of_joint(joint) for joint in split_by_joint(mapped)]
    return [[PoseLandmark(
        joint_index=joint_index,
        x=regressed[frame].x,
        y=regressed[frame].y
    ) for joint_index, regressed in enumerate(by_joint)] for frame in range(len(mapped))]


def binomial_regression_of_joint(joint_frames: TimeSeriesPoint) -> [Point]:
    frames = range(len(joint_frames.x))
    regression_inputs = np.array(frames).reshape(-1, 1)
    regression_outputs = np.array([[joint_frames.x[frame], joint_frames.y[frame]] for frame in frames])
    model = LinearRegression().fit(regression_inputs, regression_outputs)
    return [Point(x=point[0], y=point[1]) for point in model.predict(regression_inputs)]


def poses_as_outlines(frames: [[PoseLandmark]]) -> [TimeSeriesPoint]:
    return [joint_list_to_outline(joint_list) for joint_list in frames]


def split_by_joint(joint_maps: [[PoseLandmark]]):
    output = [TimeSeriesPoint(x=[], y=[], z=[]) for _ in range(33)]
    for frame in joint_maps:
        for keypoint in frame:
            append_values(output[keypoint.joint_index], keypoint)
    return output


def index_joint_list(input_list: [PoseLandmark]) -> dict[int, PoseLandmark]:
    return {
        joint.joint_index: joint
        for joint in input_list
    }


def distance_3d(first: dict[str, float], second: dict[str, float]) -> float:
    return ((second['x'] - first['x']) ** 2 + (second['y'] - first['y']) ** 2 + (second['z'] - first['z']) ** 2) ** 0.5


def midpoint_3d(first: PoseLandmark, second: PoseLandmark) -> PoseLandmark:
    return PoseLandmark(
        x=(first.x + second.x) / 2,
        y=(first.y + second.y) / 2,
        z=(first.z + second.z) / 2
    )


# pose points into 3d figure (fancier)
def joint_list_to_outline(input_list: [PoseLandmark]) -> TimeSeriesPoint:
    output = TimeSeriesPoint(x=[], y=[], z=[])
    joints = index_joint_list(input_list)

    append_values(output, joints[10])
    append_values(output, joints[8])
    append_values(output, joints[6])
    append_values(output, joints[5])
    append_values(output, joints[4])
    append_values(output, joints[1])
    append_values(output, joints[2])
    append_values(output, joints[3])
    append_values(output, joints[7])
    append_values(output, joints[9])
    append_values(output, joints[0])
    append_values(output, joints[10])
    append_values(output, joints[9])

    mid_mouth = midpoint_3d(joints[10], joints[9])
    mid_shoulders = midpoint_3d(joints[11], joints[12])

    append_values(output, mid_mouth)
    append_values(output, mid_shoulders)

    append_values(output, joints[11])
    append_values(output, joints[13])
    append_values(output, joints[15])
    append_values(output, joints[21])
    append_values(output, joints[19])
    append_values(output, joints[17])
    append_values(output, joints[15])
    append_values(output, joints[13])
    append_values(output, joints[11])
    append_values(output, joints[23])
    append_values(output, joints[25])
    append_values(output, joints[27])
    append_values(output, joints[29])
    append_values(output, joints[31])
    append_values(output, joints[27])
    append_values(output, joints[25])
    append_values(output, joints[23])
    append_values(output, joints[24])
    append_values(output, joints[26])
    append_values(output, joints[28])
    append_values(output, joints[30])
    append_values(output, joints[32])
    append_values(output, joints[28])
    append_values(output, joints[26])
    append_values(output, joints[24])
    append_values(output, joints[12])
    append_values(output, joints[14])
    append_values(output, joints[16])
    append_values(output, joints[22])
    append_values(output, joints[20])
    append_values(output, joints[18])
    append_values(output, joints[16])
    append_values(output, joints[14])
    append_values(output, joints[12])

    append_values(output, mid_shoulders)

    return output


# Turns pose points into simple 3d stick figure
def joint_list_to_outline_simple(input_list: [PoseLandmark]) -> TimeSeriesPoint:
    output = TimeSeriesPoint(x=[], y=[], z=[])
    joints = index_joint_list(input_list)

    append_values(output, joints[10])
    append_values(output, joints[8])
    append_values(output, joints[6])
    append_values(output, joints[5])
    append_values(output, joints[4])
    append_values(output, joints[1])
    append_values(output, joints[2])
    append_values(output, joints[3])
    append_values(output, joints[7])
    append_values(output, joints[9])
    append_values(output, joints[0])
    append_values(output, joints[10])
    append_values(output, joints[9])
    append_values(output, joints[11])
    append_values(output, joints[13])
    append_values(output, joints[15])
    append_values(output, joints[21])
    append_values(output, joints[19])
    append_values(output, joints[17])
    append_values(output, joints[15])
    append_values(output, joints[13])
    append_values(output, joints[11])
    append_values(output, joints[23])
    append_values(output, joints[25])
    append_values(output, joints[27])
    append_values(output, joints[29])
    append_values(output, joints[31])
    append_values(output, joints[27])
    append_values(output, joints[25])
    append_values(output, joints[23])
    append_values(output, joints[24])
    append_values(output, joints[26])
    append_values(output, joints[28])
    append_values(output, joints[30])
    append_values(output, joints[32])
    append_values(output, joints[28])
    append_values(output, joints[26])
    append_values(output, joints[24])
    append_values(output, joints[12])
    append_values(output, joints[14])
    append_values(output, joints[16])
    append_values(output, joints[22])
    append_values(output, joints[20])
    append_values(output, joints[18])
    append_values(output, joints[16])
    append_values(output, joints[14])
    append_values(output, joints[12])
    append_values(output, joints[10])

    return output


def append_values(destination: TimeSeriesPoint, value: PoseLandmark):
    destination.x.append(value.x)
    destination.y.append(value.y)
    destination.z.append(value.z)


if __name__ == '__main__':
    pass
