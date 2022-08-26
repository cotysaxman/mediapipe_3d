import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes

from load_poses import load_poses
from pose_transformations import poses_as_outlines, full_pose_as_regression, pose_centers, \
    pose_centers_subset, size_by_regression, slope_of_two_joints, TimeSeriesPoint, Point
from statistics import mean, stdev


class Plot2D:
    def __init__(
            self,
            axes: Axes,
            line: [TimeSeriesPoint],
            animation_titles: [str] = None,
            additional_points: dict[int, [Point]] = None,
            center_on: [TimeSeriesPoint] = None,
            pose_sizes: [float] = None
    ):
        if animation_titles is None:
            animation_titles = ["Pose"] * len(line)
        if additional_points is None:
            additional_points = []
        if center_on is None:
            center_on = [[0.5, 0.5] * len(line)]
        if pose_sizes is None:
            pose_sizes = [1] * len(line)
        self.ax = axes
        self.ax.set_aspect(aspect=str(16 / 9))
        self.titles = animation_titles
        self.sizes = pose_sizes
        self.update_axes(self.sizes[0], self.titles[0])
        self.line, = self.ax.plot(line[0].x, line[0].y)
        self.all_poses: [TimeSeriesPoint] = line
        self.extra_points: [Point] = additional_points
        self.point_plot = self.ax.scatter([], [])
        self.offsets = center_on

    def update_axes(self, size, title):
        self.ax.set_title(title)
        multiplier = self.ax.get_aspect()
        padded_size = size * 1.5
        self.ax.set_xlim(left=0 - (padded_size / multiplier), right=padded_size / multiplier)
        self.ax.set_ylim(bottom=padded_size * multiplier, top=0 - (padded_size * multiplier))

    def __call__(self, i):
        outline = self.all_poses[i]
        x_offset = self.offsets[i][0]
        y_offset = self.offsets[i][1]
        self.update_axes(self.sizes[i], self.titles[i])
        self.line.set_data(
            [value - x_offset for value in outline.x],
            [value - y_offset for value in outline.y]
        )
        self.point_plot.set_paths([])
        if i in self.extra_points.keys():
            self.point_plot = self.ax.scatter(
                [point.x - x_offset for point in self.extra_points[i]],
                [point.y - y_offset for point in self.extra_points[i]],
            )
        return [self.line, self.point_plot]


if __name__ == '__main__':
    pose_map = load_poses()
    regressed_pose = full_pose_as_regression()
    regressed_outline = poses_as_outlines(regressed_pose)
    poses = poses_as_outlines(pose_map)
    average_position = pose_centers(regressed_pose)
    torso_average = pose_centers_subset(regressed_pose, [11, 12, 23, 24])
    pose_size = size_by_regression()
    shoulder_slope = slope_of_two_joints(pose_map, [11, 12])
    mean_slope = mean(shoulder_slope)
    st_dev = stdev(shoulder_slope)
    total_frames = len(poses)

    title_labels = ["Scaled/Centered Poses With Steps"]
    titles = ["{title} {frame}/{total_frames}".format(
        title=title_labels[frame // total_frames],
        frame=frame % total_frames,
        total_frames=total_frames
    ) for frame in range(total_frames)
    ]

    points = {frame: [
        pose_map[frame % total_frames][30]
        if slope > (mean_slope + st_dev / 2)
        else pose_map[frame % total_frames][29]
    ] for frame, slope in enumerate(
        [shoulder_slope[frame % total_frames] for frame in range(total_frames)]
    ) if abs(slope - mean_slope) > (st_dev / 2)}

    origins = [[point.x, point.y] for point in torso_average]

    fig, ax = plt.subplots()
    instance = Plot2D(
        ax,
        poses,
        animation_titles=titles,
        additional_points=points,
        center_on=origins,
        pose_sizes=pose_size
    )
    pose_animation = FuncAnimation(fig, instance, frames=total_frames, interval=25)
    plt.show()
