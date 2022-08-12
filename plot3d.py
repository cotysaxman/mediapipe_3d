import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from load_poses import load_poses
from pose_transformations import poses_as_outlines


class Plot3D:
    def __init__(self, ax, poses):
        self.ax = ax
        self.ax.set_title("Pose")
        self.limit_axes()
        self.line, = self.ax.plot3D(poses[0]['x'], poses[0]['z'], poses[0]['y'])
        self.all_poses = poses

    def limit_axes(self):
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(1, 0)

    def __call__(self, i):
        outline = self.all_poses[i]
        self.line.set_data_3d(outline['x'], outline['z'], outline['y'])
        return self.line


if __name__ == '__main__':
    poses = poses_as_outlines(load_poses())
    ax = plt.axes(projection='3d')
    fig = ax.figure
    instance = Plot3D(ax, poses)
    pose_animation = FuncAnimation(fig, instance, frames=len(poses), interval=10)
    plt.show()
