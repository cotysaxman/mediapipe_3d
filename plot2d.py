import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pose_transformations import poses_as_outlines, full_pose_as_regression, maps_of_poses, pose_centers, \
    pose_centers_subset, size_by_regression, slope_of_two_joints


class Plot2D:
    def __init__(self, axes, line, titles=None, points=None, center_on=None, sizes=None):
        if titles is None:
            titles = ["Pose" for _ in range(line)]
        if points is None:
            points = [[{}]]
        if center_on is None:
            center_on = [[0.5, 0.5] for _ in range(len(line))]
        if sizes is None:
            sizes = [1 for _ in range(len(line))]
        self.ax = axes
        self.ax.set_aspect(16 / 9)
        self.titles = titles
        self.sizes = sizes
        self.update_axes(self.sizes[0], self.titles[0])
        self.line, = self.ax.plot(line[0]['x'], line[0]['y'])
        self.all_poses = line
        self.extra_points = points
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
            [value - x_offset for value in outline['x']],
            [value - y_offset for value in outline['y']]
        )
        self.point_plot.set_paths([])
        if i < len(self.extra_points) and len(self.extra_points[i]) > 0:
            self.point_plot = self.ax.scatter(
                [point["x"] - x_offset for point in self.extra_points[i]
                 if "x" in point.keys() and "y" in point.keys()],
                [point["y"] - y_offset for point in self.extra_points[i]
                 if "x" in point.keys() and "y" in point.keys()],
            )
        return [self.line, self.point_plot]


if __name__ == '__main__':
    pose_map = maps_of_poses()
    regressed_pose = full_pose_as_regression()
    regressed_outline = poses_as_outlines(regressed_pose)
    poses = poses_as_outlines(pose_map)
    average_position = pose_centers(regressed_pose)
    torso_average = pose_centers_subset(regressed_pose, [11, 12, 23, 24])
    pose_size = size_by_regression()
    shoulder_slope = slope_of_two_joints(pose_map, [11, 12])

    total_frames = len(poses)

    title_labels = ["Scaled/Centered Poses With Steps"]
    titles = ["{title} {frame}/{total_frames}".format(
        title=title_labels[frame // total_frames],
        frame=frame % total_frames,
        total_frames=total_frames
    ) for frame in range(total_frames)
    ]

    points = [[
        pose_map[frame % total_frames][30]
        if shoulder_slope[frame % total_frames] > 0.015
        else (pose_map[frame % total_frames][29]
              if shoulder_slope[frame % total_frames] < -0.015
              else {}
              )
    ] for frame in range(total_frames)]

    origins = [[point["x"], point["y"]] for point in torso_average]

    sizes = pose_size

    fig, ax = plt.subplots()
    instance = Plot2D(
        ax,
        poses,
        titles=titles,
        points=points,
        center_on=origins,
        sizes=sizes
    )
    pose_animation = FuncAnimation(fig, instance, frames=total_frames, interval=25)
    plt.show()
