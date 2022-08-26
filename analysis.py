import matplotlib.pyplot as plt

from load_poses import load_poses
from pose_transformations import slope_of_two_joints
from statistics import mean, stdev


if __name__ == '__main__':
    pose_map = load_poses()
    shoulder_slope = slope_of_two_joints(pose_map, [11, 12])

    fig, ax = plt.subplots()
    ax.set_title("shoulder slope")

    mean_slope = mean(shoulder_slope)
    st_dev = stdev(shoulder_slope)
    xs = range(len(shoulder_slope))

    ax.plot(xs, shoulder_slope)
    ax.plot(xs, [mean_slope] * len(xs))
    ax.plot(xs, [mean_slope + st_dev / 2] * len(xs))
    ax.plot(xs, [mean_slope - st_dev / 2] * len(xs))

    plt.show()
