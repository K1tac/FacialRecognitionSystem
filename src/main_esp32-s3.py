import http.client
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import face_recognition

PROJECT_DIR = Path(__file__).resolve().parent.parent
TRAINING_DIR = PROJECT_DIR / "trainingimages"

ESP32_IP = "192.168.1.84"
FRAME_PATH = "/frame"

KNOWN_SIZES = {
    160 * 120 * 2: (160, 120),
    176 * 144 * 2: (176, 144),
    240 * 176 * 2: (240, 176),
    320 * 240 * 2: (320, 240),
    400 * 296 * 2: (400, 296),
    640 * 480 * 2: (640, 480),
}

DETECT_MAX_WIDTH = 320
DISPLAY_TARGET_WIDTH = 960
TOLERANCE = 0.50
RECONNECT_DELAY = 2.0
WINDOW_NAME = "Facial Recognition System"

GREEN = (0, 255, 0)

known_encodings = []
known_names = []


class StreamReader:
    def __init__(self, host, path):
        self.host = host
        self.path = path
        self.frame = None
        self.index = 0
        self.lock = threading.Lock()
        self.running = True
        self.connected = False
        self.conn = None

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _open(self):
        self._close()
        self.conn = http.client.HTTPConnection(self.host, timeout=5)

    def _close(self):
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None

    @staticmethod
    def _decode(data, width, height):
        raw = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 2)
        swapped = np.ascontiguousarray(raw[:, :, ::-1])
        return cv2.cvtColor(swapped, cv2.COLOR_BGR5652BGR)

    def _run(self):
        self._open()

        while self.running:
            try:
                self.conn.request("GET", self.path)
                response = self.conn.getresponse()
                data = response.read()

                size = KNOWN_SIZES.get(len(data))

                if size is None:
                    print(f"Unexpected frame size: {len(data)} bytes")
                    time.sleep(0.2)
                    continue

                frame = self._decode(data, size[0], size[1])

                with self.lock:
                    self.frame = frame
                    self.index += 1

                if not self.connected:
                    print(f"Connected to {self.host} at {size[0]}x{size[1]}")
                    self.connected = True

            except Exception as e:
                if self.connected:
                    print(f"Stream interrupted: {e}")

                self.connected = False
                self._open()
                time.sleep(RECONNECT_DELAY)

        self._close()

    def read(self):
        with self.lock:
            if self.frame is None:
                return None, 0
            return self.frame, self.index

    def stop(self):
        self.running = False
        self.thread.join(timeout=6)


class Recognizer:
    def __init__(self, stream):
        self.stream = stream
        self.faces = []
        self.frame_size = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        last_index = 0

        while self.running:
            frame, index = self.stream.read()

            if frame is None or index == last_index:
                time.sleep(0.005)
                continue

            last_index = index

            height, width = frame.shape[:2]
            scale = min(1.0, DETECT_MAX_WIDTH / width)

            if scale < 1.0:
                small = cv2.resize(
                    frame, (0, 0), fx=scale, fy=scale,
                    interpolation=cv2.INTER_AREA
                )
            else:
                small = frame

            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            upsample = 1 if small.shape[1] <= 200 else 0

            locations = face_recognition.face_locations(
                rgb, number_of_times_to_upsample=upsample, model="hog"
            )
            encodings = face_recognition.face_encodings(rgb, locations)

            results = []

            for (top, right, bottom, left), encoding in zip(locations, encodings):
                name, confidence = recognize_face(encoding)
                results.append((
                    top / scale / height,
                    right / scale / width,
                    bottom / scale / height,
                    left / scale / width,
                    name,
                    confidence,
                ))

            with self.lock:
                self.faces = results

    def get(self):
        with self.lock:
            return list(self.faces)

    def stop(self):
        self.running = False
        self.thread.join(timeout=3)


def load_training_images():
    if not TRAINING_DIR.exists():
        raise FileNotFoundError(f"Training directory '{TRAINING_DIR}' does not exist.")

    extensions = {".jpg", ".jpeg", ".png", ".webp"}

    for image_path in sorted(TRAINING_DIR.iterdir()):
        if image_path.suffix.lower() not in extensions:
            continue

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
    distances = face_recognition.face_distance(known_encodings, face_encoding)

    best_index = int(distances.argmin())
    best_distance = float(distances[best_index])

    if best_distance > TOLERANCE:
        return "Unknown", 0

    name = known_names[best_index]
    confidence = max(0, min(100, round((1 - best_distance) * 100)))

    return name, confidence


def draw_face(frame, top, right, bottom, left, name, confidence):
    label = f"{name} ({confidence}%)" if name != "Unknown" else "Unknown"

    cv2.rectangle(frame, (left, top), (right, bottom), GREEN, 2)

    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
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


def main():
    load_training_images()

    stream = StreamReader(ESP32_IP, FRAME_PATH)
    recognizer = Recognizer(stream)

    waiting = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(
        waiting, "Connecting to ESP32...",
        (150, 240), cv2.FONT_HERSHEY_SIMPLEX,
        0.9, (255, 255, 255), 2, cv2.LINE_AA,
    )

    last_index = 0
    last_tick = time.perf_counter()
    fps = 0.0

    print("started , q to quit")

    try:
        while True:
            frame, index = stream.read()

            if frame is None:
                cv2.imshow(WINDOW_NAME, waiting)
                if cv2.waitKey(100) & 0xFF == ord("q"):
                    break
                continue

            if index == last_index:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            last_index = index

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_tick, 1e-6))
            last_tick = now

            height, width = frame.shape[:2]
            display_scale = max(1, DISPLAY_TARGET_WIDTH // width)

            display = cv2.resize(
                frame,
                (width * display_scale, height * display_scale),
                interpolation=cv2.INTER_LINEAR,
            )

            dh, dw = display.shape[:2]

            for top, right, bottom, left, name, confidence in recognizer.get():
                draw_face(
                    display,
                    int(top * dh), int(right * dw),
                    int(bottom * dh), int(left * dw),
                    name, confidence,
                )

            cv2.putText(
                display, f"{fps:.1f} FPS",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, GREEN, 2, cv2.LINE_AA,
            )

            cv2.imshow(WINDOW_NAME, display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        recognizer.stop()
        stream.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()