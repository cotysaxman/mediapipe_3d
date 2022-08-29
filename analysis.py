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

    ups = []
    downs = []
    in_progress = []
    for frame, slope in enumerate(shoulder_slope):
        if len(in_progress) > 0:
            is_up = slope > mean_slope
            was_up = shoulder_slope[in_progress[-1]] > mean_slope
            if is_up != was_up:
                if was_up:
                    ups.append(in_progress)
                else:
                    downs.append(in_progress)
                in_progress = []
        in_progress.append(frame)

    normalized_ups = [[(frame + 0.5 - frames[0]) / len(frames), shoulder_slope[frame]] for frames in ups for frame in frames]
    normalized_downs = [[(frame + 0.5 - frames[0]) / len(frames), shoulder_slope[frame]] for frames in downs for frame in frames]

    flattened_ups = [frame for frames in ups for frame in frames]
    flattened_downs = [frame for frames in downs for frame in frames]

    # ax.plot(xs, shoulder_slope)
    # ax.plot(xs, [mean_slope] * len(xs))
    # ax.plot(xs, [mean_slope + st_dev / 2] * len(xs))
    # ax.plot(xs, [mean_slope - st_dev / 2] * len(xs))
    # ax.plot(flattened_ups, [shoulder_slope[frame] for frame in flattened_ups])
    # ax.plot(flattened_downs, [shoulder_slope[frame] for frame in flattened_downs])

    ax.plot([0, 1], [mean_slope] * 2)
    ax.plot([0, 1], [mean_slope + st_dev / 2] * 2)
    ax.plot([0, 1], [mean_slope - st_dev / 2] * 2)
    ax.scatter([frame[0] for frame in normalized_ups], [frame[1] for frame in normalized_ups])
    ax.scatter([frame[0] for frame in normalized_downs], [frame[1] for frame in normalized_downs])

    plt.show()
