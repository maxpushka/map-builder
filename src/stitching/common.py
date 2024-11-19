from dataclasses import dataclass
import re
import cv2
import numpy as np


# Function to crop image around a given bounding box
def crop_image(image):
    if image is None:
        return None

    # Define the bounding box coordinates for cropping
    CROP_POINTS = np.array(
        [
            (3840, 40),
            (620, 40),
            (620, 2130),
            (3840, 2130),
        ],
        dtype=np.float32,
    )

    # Get the bounding rectangle for the crop area
    x, y, w, h = cv2.boundingRect(CROP_POINTS.astype(np.int32))

    # Crop the region of interest (ROI) from the image
    cropped_image = image[y : y + h, x : x + w]

    return cropped_image


@dataclass
class Coordinates:
    zone: str
    square: str
    easting: int
    northing: int


def parse_coordinates(filename: str) -> Coordinates | None:
    match = re.match(r"(\d+_\w)_(\w+)_(\d{2})(\d{3})_(\d{2})(\d{3})", filename)
    if not match:
        return None
    return Coordinates(
        zone=match.group(1),
        square=match.group(2),
        easting=int(match.group(3)),
        northing=int(match.group(5)),
    )
