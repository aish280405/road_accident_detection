import cv2
from detection import AccidentDetectionModel
import numpy as np
import os
import time
import csv
import smtplib
from email.message import EmailMessage

# ================== EMAIL CONFIG ==================
EMAIL_SENDER = "anamolyalert@gmail.com"
EMAIL_PASSWORD = "nceubqmfdbbdvehi"  
EMAIL_RECEIVER = "akarshkumar2004@gmail.com"

def send_email(image_path, timestamp, confidence):
    msg = EmailMessage()
    msg["Subject"] = "🚨 Accident Detected!"
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER

    msg.set_content(f"""
Accident detected!

Time: {timestamp} seconds
Confidence: {confidence}%

Check attached image.
""")

    with open(image_path, "rb") as f:
        img_data = f.read()
        msg.add_attachment(
            img_data,
            maintype="image",
            subtype="jpeg",
            filename=os.path.basename(image_path)
        )

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
        smtp.send_message(msg)

    print("📧 Email sent successfully!")


# ================== MODEL ==================
model = AccidentDetectionModel("model.json", 'model_weights.h5')
font = cv2.FONT_HERSHEY_SIMPLEX

# ================== STORAGE ==================
SAVE_FOLDER = "accident_frames"
os.makedirs(SAVE_FOLDER, exist_ok=True)

csv_file = open("accident_log.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["timestamp_sec", "confidence", "image_path"])

# ================== CONTROL ==================
last_saved_time = 0
SAVE_COOLDOWN = 2  # seconds
email_sent = False  # important flag


def _get_video_path():
    show_dir = "show"
    supported_extensions = (".mp4", ".avi", ".mov", ".mkv")

    for filename in sorted(os.listdir(show_dir)):
        if filename.lower().endswith(supported_extensions):
            return os.path.join(show_dir, filename)

    raise FileNotFoundError("No video file was found inside the 'show' folder.")


def startapplication():
    global last_saved_time, email_sent

    video_path = _get_video_path()
    print("Using video:", video_path)

    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        raise RuntimeError(f"Unable to open video file: {video_path}")

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Preprocess
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(rgb_frame, (250, 250))

        # Prediction
        pred, prob = model.predict_accident(roi[np.newaxis, :, :])
        confidence = round(np.max(prob) * 100, 2)

        # TimestampA
        timestamp = video.get(cv2.CAP_PROP_POS_MSEC) / 1000

        print(f"Prediction: {pred}, Confidence: {confidence}%, Time: {round(timestamp,2)}s")

        current_time = time.time()

        # ================== ACCIDENT DETECTED ==================
        if pred == "Accident":

            # Save frame with cooldown
            if current_time - last_saved_time > SAVE_COOLDOWN:
                filename = f"{SAVE_FOLDER}/accident_{round(timestamp,2)}s_{confidence}.jpg"
                cv2.imwrite(filename, frame)

                csv_writer.writerow([round(timestamp, 2), confidence, filename])
                print(f"🚨 Accident saved: {filename}")

                last_saved_time = current_time

                # Send email ONLY once per event
                if not email_sent:
                    try:
                        send_email(filename, round(timestamp, 2), confidence)
                        email_sent = True
                    except Exception as e:
                        print("❌ Email failed:", e)

        # ================== RESET WHEN NORMAL ==================
        else:
            email_sent = False

        # ================== DISPLAY ==================
        color = (0, 0, 255) if pred == "Accident" else (0, 255, 0)

        cv2.rectangle(frame, (0, 0), (420, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"{pred} {confidence}%", (20, 35),
                    font, 1, color, 2)

        cv2.putText(frame, f"Time: {round(timestamp,2)}s",
                    (20, 55), font, 0.6, (255, 255, 255), 2)

        cv2.imshow('Accident Detection', frame)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    video.release()
    csv_file.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    startapplication()