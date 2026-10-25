from pathlib import Path

import cv2
import face_recognition

PROJECT_DIR = Path(__file__).resolve().parent.parent
TRAINING_DIR = PROJECT_DIR / "trainingimages"

# these are the camera settings
CAMERA_INDEX = 0
FRAME_SCALE = 0.25
TOLERANCE = 0.50
PROCESS_EVERY_N_FRAMES = 60

GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

known_encodings = []
known_names = []

def load_training_images():
    if not TRAINING_DIR.exists():
        raise FileNotFoundError(f"Training directory '{TRAINING_DIR}' does not exist.")

    extensions = {".jpg", ".jpeg", ".png", ".webp"}

    for image_path in sorted(TRAINING_DIR.iterdir()):
        if image_path.suffix.lower() not in extensions:
            continue
        
        # jacob_1.png -> jacob
        name = image_path.stem.rsplit("_", 1)

        if len(name) != 2 or not name[1].isdigit():
            print(f"skipping: {image_path.name} bcuz of invalid format")
            continue

        person_name = name[0].upper()

        image = face_recognition.load_image_file(str(image_path))
        encodings = face_recognition.face_encodings(image)

        if len(encodings) != 1:
            print(f"skipping {image_path.name} expected 1 face found {len(encodings)}")
            continue

        known_encodings.append(encodings[0])
        known_names.append(person_name)

        print(f"loaded {image_path.name} for {person_name}")

    if not known_encodings:
        raise RuntimeError("No valid training images found. Please check the trainingimages directory.")

    print(f"\n loaded {len(known_encodings)} training images")

def recognize_face(face_encoding):
    # compare against stored photos
    distances = face_recognition.face_distance(
        known_encodings, face_encoding
    )

    best_index = distances.argmin()
    best_distance = float(distances[best_index])

    if best_distance > TOLERANCE:
        return "Unknown", 0

    name = known_names[best_index]

    confidence = max(
        0, min(100, round((1 - best_distance) * 100))
    )

    return name, confidence

def main():
    load_training_images()

    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera with index {CAMERA_INDEX} or check camera permissions.")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    frame_count = 0
    cached_faces = []

    print("started , q to quit")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read frame from camera. Exiting.")
                break

            frame_count += 1

            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                small_frame = cv2.resize(
                    frame,  (0,0),
                    fx=FRAME_SCALE, fy=FRAME_SCALE
                )

                rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                locations = face_recognition.face_locations(rgb_frame, model="hog")
                encodings = face_recognition.face_encodings(rgb_frame, locations)

                cached_faces = []

                for location, encoding in zip(locations, encodings):
                    top, right, bottom, left = location

                    name, confidence = recognize_face(encoding)

                    cached_faces.append((
                        int(top / FRAME_SCALE),
                        int(right / FRAME_SCALE),
                        int(bottom / FRAME_SCALE),
                        int(left / FRAME_SCALE),
                        name,
                        confidence
                    ))

            for top, right, bottom, left, name, confidence in cached_faces:
                label = (
                    f"{name} ({confidence}%)"
                    if name != "Unknown"
                    else "Unknown"
                )

                cv2.rectangle(frame, (left, top), (right, bottom), GREEN, 2)

                (text_width, text_height), baseline = (
                    cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, 2
                    )
                )

                label_top = max(0, top - text_height - baseline - 8)

                cv2.rectangle(
                    frame,
                    (left, label_top),
                    (left + text_width + 10, top),
                    GREEN,
                    cv2.FILLED,
                )

                cv2.putText(
                    frame, label,
                    (left + 5, top - baseline - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0), 2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Facial Recognition System", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()