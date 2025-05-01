import logging
from config import ENV
import sys
import cv2
import sqlite3
import numpy as np
import face_recognition
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def pi_eye_detect():
    try:
        from picamera2 import Picamera2
        # Attempt to initialize the PiCamera2
        camera = Picamera2()
        camera.start()  # Start the preview to check if the camera works
        logger.info("Raspberry Pi camera detected!")
        camera.close()
        return True
    except ImportError:
        logger.warning("Picamera2 library not found. Falling back to generic camera detection.")
    except Exception as e:
        logger.error(f"Raspberry Pi camera not detected: {e}. Falling back to generic camera detection.")
    
    # If PiCamera is not detected, fall back to generic detection
    return generic_eye_detect()

def generic_eye_detect():
    camera = cv2.VideoCapture(0)
    if camera.isOpened():
        logger.info("Camera detected!")
        camera.release()  # Release the camera resource
        return True
    else:
        logger.warning("No camera detected.")
        return False

def detect_eyes():
    if ENV.upper() == "PI":
        return pi_eye_detect()
    else:
        return generic_eye_detect()
DB_PATH = "faces.db"

def setup_database():
    """Initialize the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            encoding BLOB NOT NULL,
            label TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    logger.info("Database initialized.")

def save_face_to_db(encoding, label="unknown"):
    """Save a face encoding to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO faces (encoding, label) VALUES (?, ?)", (encoding.tobytes(), label))
    conn.commit()
    conn.close()
    logger.info(f"New face saved to database with label: {label}")

def load_known_faces():
    """Load all known face encodings from the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT encoding, label FROM faces")
    rows = cursor.fetchall()
    conn.close()
    known_encodings = [np.frombuffer(row[0], dtype=np.float64) for row in rows]
    known_labels = [row[1] for row in rows]
    return known_encodings, known_labels

def detect_and_enroll_faces():
    """Run the camera feed, detect faces, and enroll new ones."""
    known_encodings, known_labels = load_known_faces()

    if ENV.upper() == "PI":
        try:
            from picamera2 import Picamera2, Preview
            picam2 = Picamera2()
            config = picam2.create_preview_configuration(main={"size": (640, 480)})
            picam2.configure(config)
            picam2.start()
            time.sleep(2)
            logger.info("Using Raspberry Pi camera for face detection.")
            
            while True:
                frame = picam2.capture_array()  # Capture a frame as a NumPy array

                # Convert the frame to RGB (face_recognition expects RGB images)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Detect face locations and encodings
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                for face_encoding, face_location in zip(face_encodings, face_locations):
                    # Check if the face is already known
                    matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
                    if not any(matches):
                        # New face detected
                        logger.info("New face detected. Enrolling...")
                        save_face_to_db(face_encoding)
                        known_encodings.append(face_encoding)
                        known_labels.append("unknown")

                        # Draw a rectangle around the face
                        top, right, bottom, left = face_location
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(frame, "Unknown", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Display the frame
                #cv2.imshow("Face Detection", frame)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            picam2.stop()
            cv2.destroyAllWindows()

        except ImportError:
            logger.error("Picamera2 library not found. Falling back to OpenCV.")
        except Exception as e:
            logger.error(f"Error using Raspberry Pi camera: {e}. Falling back to OpenCV.")

    # Fallback to OpenCV if not on Raspberry Pi or Pi camera is not detected
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        logger.error("No camera detected. Exiting the program.")
        sys.exit(1)

    logger.info("Using OpenCV for face detection.")
    while True:
        ret, frame = video_capture.read()
        if not ret:
            logger.error("Failed to capture frame from camera.")
            break

        # Convert the frame to RGB (face_recognition expects RGB images)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face locations and encodings
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Check if the face is already known
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
            if not any(matches):
                # New face detected
                logger.info("New face detected. Enrolling...")
                save_face_to_db(face_encoding)
                known_encodings.append(face_encoding)
                known_labels.append("unknown")

                # Draw a rectangle around the face
                top, right, bottom, left = face_location
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, "Unknown", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Face Detection", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()

def main():
    logger.info("Starting the eyes module.")
    if not detect_eyes():
        logger.error("No camera detected. Exiting the program.")
        sys.exit(1)
    logger.info("Camera detected.")
    setup_database()
    detect_and_enroll_faces()

if __name__ == "__main__":
    main()