import matplotlib.pyplot as plt
import csv


def limit_axes(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(1, 0)


def plot_points():
    ax = plt.axes(projection='3d')
    ax.set_title("Pose")

    while True:
        with open('output.txt', 'r') as point_csv:
            reader = csv.reader(point_csv)
            i = 0
            for row in reader:
                # i += 1
                # if i % 3 != 0:
                #     continue
                plt.cla()
                limit_axes(ax)
                joints = []
                for joint in row:
                    joints.append(raw_joint_to_map(joint))
                outline = joint_list_to_outline(joints)
                ax.plot3D(outline['x'], outline['z'], outline['y'])
                plt.draw()
                plt.pause(0.0001)

    plt.show()


# Example input:
# "(0, x: 0.49779224395751953
# y: 0.49814850091934204
# z: 0.16941595077514648
# visibility: 0.9988901019096375
# )"
def raw_joint_to_map(raw_input):
    output = {}
    no_parens = raw_input.strip('()')
    # print(no_parens)
    joint_index, remainder = no_parens.split(',')
    remainder = remainder.rstrip('\n')
    # print(remainder)
    output["joint_index"] = joint_index
    values = remainder.split('\n')
    for value in values:
        # print(value)
        k, v = value.split(':')

        output[k.strip(' ')] = float(v.strip(' '))

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
    plot_points()
