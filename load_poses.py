import csv
from dataclasses import dataclass
from typing import Iterable


@dataclass
class PoseLandmark:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 0.0
    joint_index: int = -1


def chunked(sequence: Iterable, size: int) -> list:
    temp = []
    for item in sequence:
        temp.append(item)
        if len(temp) == size:
            yield temp
            temp = []
    if len(temp):
        yield temp


def load_poses():
    with open('output.txt', 'r') as point_csv:
        reader = csv.reader(point_csv)
        frames = []
        raw_data = chunked(reader, 33)
        for raw_pose in raw_data:
            frames.append([
                PoseLandmark(
                    x=float(landmark[1]),
                    y=float(landmark[2]),
                    z=float(landmark[3]),
                    visibility=float(landmark[4]),
                    joint_index=int(landmark[0])
                ) for landmark in raw_pose
            ])
        return frames
