import csv


def load_poses():
    with open('output.txt', 'r') as point_csv:
        reader = csv.reader(point_csv)
        frames = []
        joints = []
        for joint in reader:
            joint_index = joint[0]
            if int(joint_index) == 0 and joints != []:
                frames.append(joints)
                joints = []
            joints.append(joint)
        frames.append(joints)
        return frames
