import math
from dataclasses import dataclass

import numpy as np
from numpy import ndarray
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from load_poses import load_poses, PoseLandmark
from statistics import mean, stdev, median


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


def looping_pose_interpolated(pose_map: [[PoseLandmark]], feature: [float]) -> [[PoseLandmark]]:
    mean_value = mean(feature)
    cycles = []
    in_progress = []
    skipped = []
    has_up = False
    has_down = False
    for frame, slope in enumerate(feature):
        if len(in_progress) == 0:
            in_progress.append(frame)
            continue

        is_up = slope > mean_value
        was_up = feature[in_progress[-1]] > mean_value

        if is_up != was_up:
            if len(skipped) == 0:
                skipped = in_progress
            elif was_up:
                has_up = True
            else:
                has_down = True

            if has_up and has_down:
                cycles.append(in_progress)
                has_up = False
                has_down = False
                in_progress = []

        in_progress.append(frame)

    relative_cycles = [
        [frame, (frame - frames[0]) / len(frames)]
        for frames in cycles
        for frame in frames
    ]
    cycle_length = median([len(cycle) for cycle in cycles])
    binned_frames = [[
        [cycle[0], cycle[1]]
        for cycle in relative_cycles
        if int(cycle_length * cycle[1]) == frame
    ] for frame in range(cycle_length)
    ]
    by_joint = split_by_joint(pose_map)
    output = [[] for _ in range(cycle_length * 10)]
    for frame_index, frame in enumerate(binned_frames):
        regression_inputs = np.array([value[1] for value in frame]).reshape(-1, 1)
        for joint_index in range(33):
            regression_outputs = np.array([[
                by_joint[joint_index].x[value[0]],
                by_joint[joint_index].y[value[0]]
            ] for value in frame
            ])
            model = LinearRegression().fit(
                regression_inputs,
                regression_outputs
            )
            indices = np.linspace(
                (frame_index / cycle_length),
                ((frame_index + 1) / cycle_length) - (1 / cycle_length),
                10
            ).reshape(10, 1)
            predictions = model.predict(indices)
            for sub_index, point in enumerate(predictions):
                output[(frame_index * 10) + sub_index].append(PoseLandmark(
                    joint_index=joint_index,
                    x=point[0],
                    y=point[1]
                ))

    return output


def normalized_looping_pose(pose_map: [[PoseLandmark]], feature: [float]) -> [[PoseLandmark]]:
    mean_value = mean(feature)
    cycles = []
    in_progress = []
    skipped = []
    has_up = False
    has_down = False
    for frame, slope in enumerate(feature):
        if len(in_progress) == 0:
            in_progress.append(frame)
            continue

        is_up = slope > mean_value
        was_up = feature[in_progress[-1]] > mean_value

        if is_up != was_up:
            if len(skipped) == 0:
                skipped = in_progress
            elif was_up:
                has_up = True
            else:
                has_down = True

            if has_up and has_down:
                cycles.append(in_progress)
                has_up = False
                has_down = False
                in_progress = []

        in_progress.append(frame)

    relative_cycles = [
        [frame, (frame - frames[0]) / len(frames)]
        for frames in cycles
        for frame in frames
    ]
    normalized = [[cycle[0], cycle[1] + index] for index in range(3) for cycle in relative_cycles]
    cycle_length = int(median([len(cycle) for cycle in cycles]))
    by_joint = split_by_joint(pose_map)
    output = [[] for _ in range(cycle_length)]
    regression_inputs = np.array([
        [value[1], np.sin(value[1]), np.cos(value[1])]
        for value in normalized
    ]).reshape(-1, 3)
    for joint_index in range(33):
        regression_outputs = np.array([[
            by_joint[joint_index].x[value[0]],
            by_joint[joint_index].y[value[0]]
        ] for value in normalized
        ])
        polynomial_regression = PolynomialFeatures(degree=6)
        polynomial_inputs = polynomial_regression.fit_transform(regression_inputs)
        model = LinearRegression()
        model.fit(polynomial_inputs, regression_outputs)
        indices = np.array([[index, np.sin(index), np.cos(index)] for index in [
            (frame / cycle_length) + 0.5 for frame in range(cycle_length * 2)
        ]]).reshape(cycle_length * 2, 3)
        positions_to_predict = polynomial_regression.transform(indices)
        predictions = model.predict(positions_to_predict)
        for frame in range(cycle_length):
            ratio = 0.25
            output[frame].append(PoseLandmark(
                joint_index=joint_index,
                x=predictions[frame][0] * ratio + predictions[frame + cycle_length][0] * (1 - ratio),
                y=predictions[frame][1] * ratio + predictions[frame + cycle_length][1] * (1 - ratio)
            ))

    return output


def normalized_symmetric_pose(pose_map: [[PoseLandmark]], feature: [float]) -> [[PoseLandmark]]:
    mean_value = mean(feature)
    ups = []
    downs = []
    in_progress = []
    skipped = []
    for frame, slope in enumerate(feature):
        in_progress.append(frame)
        if len(in_progress) == 1:
            continue

        is_up = slope > mean_value
        was_up = feature[in_progress[-2]] > mean_value
        if is_up != was_up:
            if len(skipped) == 0:
                skipped = in_progress
            elif was_up:
                ups.append(in_progress)
            else:
                downs.append(in_progress)
            in_progress = in_progress[-2:]

    normalized_ups = [
        [frame, (frame - frames[1]) / (len(frames) - 2)]
        for frames in ups
        for frame in frames
    ]
    normalized_downs = [
        [frame, (frame - frames[1]) / (len(frames) - 2)]
        for frames in downs
        for frame in frames
    ]
    half_cycle_length = int(median([len(cycle) for cycle in ups + downs]) - 2)
    by_joint = split_by_joint(pose_map)
    output = [[] for _ in range(half_cycle_length * 2)]
    for series_index, series in enumerate([normalized_ups, normalized_downs]):
        offset = series_index * half_cycle_length
        regression_inputs = np.array([value[1] for value in series]).reshape(-1, 1)
        for joint_index in range(33):
            regression_outputs = np.array([[
                by_joint[joint_index].x[value[0]],
                by_joint[joint_index].y[value[0]]
            ] for value in series
            ])
            polynomial_regression = PolynomialFeatures(degree=4)
            polynomial_inputs = polynomial_regression.fit_transform(regression_inputs)
            model = LinearRegression()
            model.fit(polynomial_inputs, regression_outputs)
            indices = np.array([
                frame / (half_cycle_length - 1) for frame in range(half_cycle_length)
            ]).reshape(half_cycle_length, 1)
            positions_to_predict = polynomial_regression.transform(indices)
            for index, point in enumerate(model.predict(positions_to_predict)):
                output[index + offset].append(PoseLandmark(joint_index=joint_index, x=point[0], y=point[1]))
    return output


def normalized_pose(
        reference_pose: [[TimeSeriesPoint]],
        target_pose: [[TimeSeriesPoint]],
        scales: [float]
) -> [[PoseLandmark]]:
    mean_scale = mean(scales)
    reference_map = split_by_joint(reference_pose)
    target_map = split_by_joint(target_pose)
    mean_position = [Point(
        x=mean(reference_map[joint_index].x),
        y=mean(reference_map[joint_index].y)
    ) for joint_index in range(33)]
    return [[
        PoseLandmark(
            joint_index=joint_index,
            x=((target_map[joint_index].x[frame] -
                reference_map[joint_index].x[frame]) / (scales[frame] / mean_scale)) + mean_position[joint_index].x,
            y=((target_map[joint_index].y[frame] -
                reference_map[joint_index].y[frame]) / (scales[frame] / mean_scale)) + mean_position[joint_index].y
        )
        for joint_index in range(33)
    ] for frame in range(len(reference_pose))]


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
