import cv2
import numpy as np
import dlib
from imutils import face_utils
import time
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
buzzer_pin = 36
LED_PIN = 29
GPIO.setup(buzzer_pin, GPIO.OUT)
GPIO.setup(LED_PIN, GPIO.OUT)

def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist

def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)

    if ratio > 0.25:
        return 2
    elif ratio > 0.21 and ratio <= 0.25:
        return 1
    else:
        return 0

sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)
delay = 15

GPIO.setup(12, GPIO.IN, pull_up_down=GPIO.PUD_UP)
output_pins = [11, 15, 16, 18, 22]
for pin in output_pins:
    GPIO.setup(pin, GPIO.OUT)

def input_callback(channel):
    if GPIO.input(channel):
        for pin in output_pins:
            GPIO.output(pin, GPIO.HIGH)
    else:
        for pin in output_pins:
            GPIO.output(pin, GPIO.LOW)

GPIO.add_event_detect(12, GPIO.BOTH, callback=input_callback)

cap = cv2.VideoCapture(0)
hog_face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = hog_face_detector(gray)

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        face_frame = frame.copy()
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        left_blink = blinked(
            landmarks[36], landmarks[37], landmarks[38],
            landmarks[41], landmarks[40], landmarks[39]
        )
        right_blink = blinked(
            landmarks[42], landmarks[43], landmarks[44],
            landmarks[47], landmarks[46], landmarks[45]
        )

        if left_blink == 0 or right_blink == 0:
            sleep += 1
            drowsy = 0
            active = 0
            if sleep > 1:
                status = "SLEEPING !!!"
                GPIO.output(buzzer_pin, GPIO.HIGH)
                GPIO.output(LED_PIN, GPIO.HIGH)
                time.sleep(0.2)
                color = (0, 0, 255)

        elif left_blink == 1 or right_blink == 1:
            sleep = 0
            active = 0
            drowsy += 1
            if drowsy > 1:
                status = "Drowsy !"
                GPIO.output(buzzer_pin, GPIO.HIGH)
                GPIO.output(LED_PIN, GPIO.HIGH)
                time.sleep(0.2)
                color = (0, 0, 255)

        else:
            drowsy = 0
            sleep = 0
            active += 1
            if active > 1:
                status = "Active :)"
                time.sleep(0.2)
                GPIO.output(buzzer_pin, GPIO.LOW)
                GPIO.output(LED_PIN, GPIO.LOW)
                color = (0, 0, 255)

        cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        for n in range(0, 68):
            (x, y) = landmarks[n]
            cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

GPIO.cleanup()
cv2.destroyAllWindows()