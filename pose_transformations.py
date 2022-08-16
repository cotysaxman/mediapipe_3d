import math
import numpy as np
from sklearn.linear_model import LinearRegression
from load_poses import load_poses
from statistics import mean


def derivative_of_two_joints(pose_set, joint_indices):
    return np.gradient([LinearRegression().fit(
        np.array([pose[index]["x"] for index in joint_indices]).reshape(-1, 1),
        np.array([pose[index]["y"] for index in joint_indices])
    ).coef_[0] for pose in pose_set
    ], 1)


def slope_of_two_joints(pose_set, joint_indices):
    return [LinearRegression().fit(
        np.array([pose[index]["x"] for index in joint_indices]).reshape(-1, 1),
        np.array([pose[index]["y"] for index in joint_indices])
    ).coef_[0] for pose in pose_set
    ]


def pose_centers(pose_set):
    return [{key: mean([point[key] for point in pose]) for key in pose[0].keys()} for pose in pose_set]


def pose_centers_subset(pose_set, keypoints):
    return [{
        key: mean([point[key] for index, point in enumerate(pose) if index in keypoints])
        for key in pose[0].keys()
    } for pose in pose_set]


def size_by_regression():
    pose_set = full_pose_as_regression()
    centers = pose_centers(pose_set)
    return [
        mean([
            math.dist([point["x"], point["y"]], [centers[index]["x"], centers[index]["y"]])
            for index, point in enumerate(pose)
        ]) for pose in pose_set
    ]


def full_pose_as_regression():
    mapped = maps_of_poses()
    by_joint = split_by_joint(mapped)
    as_regression = [[{
        "joint_index": index,
        "x": regressed["x"],
        "y": regressed["y"],
        "z": 0.0,
        "visibility": 0.0
    } for regressed in binomial_regression_of_joint(joint)
    ] for index, joint in enumerate(by_joint)
    ]
    as_regressed_complete_frames = [
        [regressed[frame] for regressed in as_regression] for frame in range(len(maps_of_poses()))
    ]

    return as_regressed_complete_frames


def binomial_regression_of_joint(joint_frames):
    frames = range(len(joint_frames["x"]))
    regression_inputs = np.array(frames).reshape(-1, 1)
    regression_outputs = np.array([[joint_frames["x"][frame], joint_frames["y"][frame]] for frame in frames])
    model = LinearRegression().fit(regression_inputs, regression_outputs)
    return [{"x": point[0], "y": point[1]} for point in model.predict(regression_inputs)]


def poses_as_outlines(frames):
    return [joint_list_to_outline(joint_list) for joint_list in frames]


def maps_of_poses():
    raw_poses = load_poses()
    return [[raw_joint_to_map(joint) for joint in frame] for frame in raw_poses]


def split_by_joint(joint_maps):
    output = []
    for frame in joint_maps:
        for keypoint in frame:
            joint_index = keypoint["joint_index"]
            if joint_index >= len(output):
                output.append({"x": [], "y": [], "z": [], "visibility": []})
            output[joint_index]["x"].append(keypoint["x"])
            output[joint_index]["y"].append(keypoint["y"])
            output[joint_index]["z"].append(keypoint["z"])
            output[joint_index]["visibility"].append(keypoint["visibility"])
    return output


def raw_joint_to_map(raw_input):
    output = {
        "joint_index": int(raw_input[0]),
        "x": float(raw_input[1]),
        "y": float(raw_input[2]),
        "z": float(raw_input[3]),
        "visibility": float(raw_input[4])
    }
    return output


def index_joint_list(input_list):
    joints = {}
    for joint in input_list:
        joints[int(joint['joint_index'])] = {'x': joint['x'], 'y': joint['y'], 'z': joint['z']}

    return joints


def distance_3d(first, second):
    return ((second['x'] - first['x']) ** 2 + (second['y'] - first['y']) ** 2 + (second['z'] - first['z']) ** 2) ** 0.5


def midpoint_3d(first, second):
    return {
        'x': (first['x'] + second['x']) / 2,
        'y': (first['y'] + second['y']) / 2,
        'z': (first['z'] + second['z']) / 2
    }


# pose points into 3d figure (fancier)
def joint_list_to_outline(input_list):
    output = {'x': [], 'y': [], 'z': []}
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
def joint_list_to_outline_simple(input_list):
    output = {'x': [], 'y': [], 'z': []}
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


def append_values(destination, value):
    for key in value.keys():
        destination[key].append(value[key])


if __name__ == '__main__':
    pass
