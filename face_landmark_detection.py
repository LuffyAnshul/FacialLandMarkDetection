import cv2
import dlib
from imutils import face_utils

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        # left = face.left()
        # right = face.right()
        # bottom = face.bottom()
        # top = face.top()
        # cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)

        landmarks = predictor(gray, face)

        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -2)

    cv2.imshow("Gray Frame", frame)
    key = cv2.waitKey(1)

    if key == 27 or key == ord('q'):
        break
