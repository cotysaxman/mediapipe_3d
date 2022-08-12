# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
import mediapipe as mp
import csv


def calculate_pose():
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    filename = 'oldinput.mp4'
    capture = cv2.VideoCapture(filename)
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    largest_dimension = max(width, height)
    fps = capture.get(cv2.CAP_PROP_FPS)
    writer = cv2.VideoWriter_fourcc(*'mp4v')
    result = cv2.VideoWriter('output.mp4', writer, fps, (int(width), int(height)))
    result_text = open('output.txt', 'w')
    result_writer = csv.writer(result_text)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while capture.isOpened():
            has_frame, frame = capture.read()
            if not has_frame:
                print("Ignoring empty camera frame.")
                break
            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame)
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            cv2.imshow('MediaPipe Pose', cv2.flip(frame, 1))
            result.write(frame)
            for index, landmark in enumerate(results.pose_landmarks.landmark):
                row = [str(index),
                       str((landmark.x * width) / largest_dimension),
                       str((landmark.y * height) / largest_dimension),
                       str(landmark.z),
                       str(landmark.visibility)]
                result_writer.writerow(row)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    result_text.close()
    capture.release()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    calculate_pose()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
