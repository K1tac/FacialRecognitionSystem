import sys
from pathlib import Path

import face_recognition
import numpy as np
from PIL import Image, ImageOps

TRAINING_DIR = Path(__file__).resolve().parent.parent / "trainingimages"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
EXIF_ORIENTATION_TAG = 274
MAX_SIDE = 1000
USE_CNN = "--cnn" in sys.argv


def count_faces(array, upsample=1, model="hog"):
    locations = face_recognition.face_locations(
        array, number_of_times_to_upsample=upsample, model=model
    )
    return len(locations)


def downscale(image, max_side=MAX_SIDE):
    scale = max_side / max(image.size)
    if scale >= 1:
        return image
    new_size = (round(image.width * scale), round(image.height * scale))
    return image.resize(new_size, Image.LANCZOS)


def find_working_rotation(image):
    for degrees in (90, 180, 270):
        rotated = np.array(image.rotate(degrees, expand=True))
        if count_faces(rotated) == 1:
            return degrees
    return None


def main():
    if not TRAINING_DIR.exists():
        raise SystemExit(f"{TRAINING_DIR} does not exist")

    header = f"{'file':<16}{'size':<12}{'exif':<6}{'raw':<5}{'fix':<5}{'up2':<5}{'small':<7}{'rot':<6}"
    if USE_CNN:
        header += "cnn"
    print(header)
    print("-" * len(header))

    for path in sorted(TRAINING_DIR.iterdir()):
        if path.suffix.lower() not in EXTENSIONS:
            continue

        original = Image.open(path)
        orientation = original.getexif().get(EXIF_ORIENTATION_TAG, "-")
        size_text = f"{original.width}x{original.height}"

        raw_array = np.array(original.convert("RGB"))
        fixed_image = ImageOps.exif_transpose(original).convert("RGB")
        fixed_array = np.array(fixed_image)
        small_array = np.array(downscale(fixed_image))

        raw_count = count_faces(raw_array)
        fixed_count = count_faces(fixed_array)
        up2_count = count_faces(fixed_array, upsample=2)
        small_count = count_faces(small_array, upsample=2)

        rotation_text = "-"
        if fixed_count == 0 and up2_count == 0 and small_count == 0:
            rotation = find_working_rotation(fixed_image)
            rotation_text = f"{rotation}" if rotation is not None else "none"

        line = (
            f"{path.name:<16}{size_text:<12}{str(orientation):<6}"
            f"{raw_count:<5}{fixed_count:<5}{up2_count:<5}{small_count:<7}{rotation_text:<6}"
        )

        if USE_CNN:
            line += str(count_faces(small_array, model="cnn"))

        print(line)

    print()
    print("raw   = hog on image exactly as load_image_file sees it")
    print("fix   = hog after applying EXIF rotation")
    print("up2   = hog after EXIF fix with upsample=2")
    print("small = hog on EXIF-fixed image downscaled to 1000px, upsample=2")
    print("rot   = manual rotation that finally works (only checked if everything else is 0)")


if __name__ == "__main__":
    main()