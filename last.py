import cv2
import time
import numpy as np
import speech_recognition as sr
from Foundation import NSUserNotification, NSUserNotificationCenter

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def show_notification(title, subtitle, text):
    notification = NSUserNotification.alloc().init()
    notification.setTitle_(title)
    notification.setSubtitle_(subtitle)
    notification.setInformativeText_(text)

    notification_center = NSUserNotificationCenter.defaultUserNotificationCenter()
    notification_center.deliverNotification_(notification)

EAR_THRESHOLD = 0.25
def calculate_ear(eye):
    (x, y, w, h) = eye
    top_y = y + h // 4
    bottom_y = y + h * 3 // 4

    eye_height = bottom_y - top_y

    eye_width = w

    return eye_height / eye_width

def detect_face_and_lighting():
    cap = cv2.VideoCapture(0)

    previous_distance = None
    base_ear = None
    blink_counter = 0
    prev_ear_time = time.time()
    prev_blink_reset_time = time.time()
    notification_shown = False

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            face_size = faces[0][2] * faces[0][3]
            distance = 60 / (face_size ** 0.5)

            if distance <= 0.09:
                show_notification('Предупреждение', ' Недопустимое расстояние от экрана до глаз', 'Отдалитесь')
            elif distance > 0.09 and distance < 0.12:
                show_notification('Предупреждение', 'Расстояние от экрана на грани', 'В таком положении ваши глаза устают быстрее')

            previous_distance = distance

        avg_intensity = np.average(gray)
        if avg_intensity < 50:
            show_notification('Предупреждение', 'Освещенность помещения', 'Включите свет')

        if len(faces) == 1:
            (x, y, w, h) = faces[0]

            left_eye_region = gray[y:y+h//2, x:x+w//2]
            right_eye_region = gray[y:y+h//2, x+w//2:x+w]

            left_eyes = eye_cascade.detectMultiScale(left_eye_region, 1.1, 3)
            right_eyes = eye_cascade.detectMultiScale(right_eye_region, 1.1, 3)

            if len(left_eyes) == 1 and len(right_eyes) == 1:
                (left_eye_x, left_eye_y, left_eye_w, left_eye_h) = left_eyes[0]
                (right_eye_x, right_eye_y, right_eye_w, right_eye_h) = right_eyes[0]

                left_eye_width = left_eye_w
                left_eye_height = left_eye_h
                right_eye_width = right_eye_w
                right_eye_height = right_eye_h

                eye_center_x = x + w // 2
                eye_center_y = y + h // 4
                eye_distance = np.sqrt((left_eye_x + left_eye_w // 2 - eye_center_x)**2 + (left_eye_y + left_eye_h // 2 - eye_center_y)**2)

                left_eye_top = (left_eye_x + left_eye_w // 4, left_eye_y)
                left_eye_bottom = (left_eye_x + left_eye_w // 4 * 3, left_eye_y + left_eye_h)
                right_eye_top = (right_eye_x + right_eye_w // 4, right_eye_y)
                right_eye_bottom = (right_eye_x + right_eye_w // 4 * 3, right_eye_y + right_eye_h)

                left_eye_vertical = np.sqrt((left_eye_top[0] - left_eye_bottom[0])**2 + (left_eye_top[1] - left_eye_bottom[1])**2)
                right_eye_vertical = np.sqrt((right_eye_top[0] - right_eye_bottom[0])**2 + (right_eye_top[1] - right_eye_bottom[1])**2)

                left_ear = left_eye_vertical / (2.0 * eye_distance)
                right_ear = right_eye_vertical / (2.0 * eye_distance)
                ear = (left_ear + right_ear) / 2.0

                if base_ear is None:
                    base_ear = ear

                if ear < base_ear * 0.95 or ear > base_ear * 1.05:
                    blink_counter += 1

                if time.time() - prev_ear_time >= 1:
                    prev_ear_time = time.time()
                    print("EAR: {:.2f}".format(ear))

            elif len(left_eyes) == 0:
                blink_counter += 1

            elif len(right_eyes) == 0:
                blink_counter += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if time.time() - prev_blink_reset_time >= 30:
            prev_blink_reset_time = time.time()
            blink_counter = 0
            notification_shown = False

        if blink_counter > 50 and not notification_shown:
            show_notification("Усталось глаз", "Сделайте перерыв", "")
            blink_counter = 0
            notification_shown = True
            time.sleep(5)

    cap.release()
    cv2.destroyAllWindows()

def start_with_voice_command():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    print("Say 'антон' to start the program...")

    recognized_command = False

    while not recognized_command:
        with microphone:
            recognizer.adjust_for_ambient_noise(microphone, duration=2)

            try:
                print("Listening...")
                audio = recognizer.listen(microphone, 5, 5)

                print("Started recognition...")
                recognized_data = recognizer.recognize_google(audio, language="ru").lower()

                if "антон" in recognized_data:
                    print("Starting the program...")
                    recognized_command = True
                    detect_face_and_lighting()
                else:
                    print("Did not recognize 'антон'. Keep trying...")

            except sr.WaitTimeoutError:
                print("Can you check if your microphone is on, please?")
            except sr.UnknownValueError:
                print("Unable to recognize speech. Keep trying...")
            except sr.RequestError:
                print("Check your Internet Connection, please")

if __name__ == "__main__":
    start_with_voice_command()
