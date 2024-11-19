from dataclasses import dataclass
import re
import cv2
import numpy as np
from enum import Enum


class MRGSRelativePosition(Enum):
    NO_RELATION = 0
    ABOVE = 1
    BELOW = 2
    LEFT = 3
    RIGHT = 4
    ABOVE_LEFT = 5
    ABOVE_RIGHT = 6
    BELOW_LEFT = 7
    BELOW_RIGHT = 8
    CENTER = 9

class MGRSCoordinates:
    zone: str
    square: str
    easting: int
    northing: int

    def __init__(self, filename: str):
        match = re.match(r"(\d+_\w)_(\w+)_(\d{2})(\d{3})_(\d{2})(\d{3})", filename)
        if not match:
            raise ValueError(
                f"Filename format does not match expected pattern: {filename}"
            )
        self.zone = match.group(1)
        self.square = match.group(2)
        self.easting = int(match.group(3))
        self.northing = int(match.group(5))

    def position(self, other: "MGRSCoordinates") -> MRGSRelativePosition:
        if self.zone != other.zone or self.square != other.square:
            return MRGSRelativePosition.NO_RELATION

        easting_diff = other.easting - self.easting
        northing_diff = other.northing - self.northing

        if easting_diff == 0 and northing_diff == 0:
            return MRGSRelativePosition.CENTER
        elif easting_diff == 0 and northing_diff > 0:
            return MRGSRelativePosition.ABOVE
        elif easting_diff == 0 and northing_diff < 0:
            return MRGSRelativePosition.BELOW
        elif easting_diff > 0 and northing_diff == 0:
            return MRGSRelativePosition.RIGHT
        elif easting_diff < 0 and northing_diff == 0:
            return MRGSRelativePosition.LEFT
        elif easting_diff > 0 and northing_diff > 0:
            return MRGSRelativePosition.ABOVE_RIGHT
        elif easting_diff > 0 and northing_diff < 0:
            return MRGSRelativePosition.BELOW_RIGHT
        elif easting_diff < 0 and northing_diff > 0:
            return MRGSRelativePosition.ABOVE_LEFT
        elif easting_diff < 0 and northing_diff < 0:
            return MRGSRelativePosition.BELOW_LEFT
        return MRGSRelativePosition.NO_RELATION



@dataclass
class Tile:
    name: str
    image: np.ndarray
    mgrs_coord: MGRSCoordinates
    top_left: np.array
    top_right: np.array
    bottom_left: np.array
    bottom_right: np.array


# Function to crop image around a given bounding box
def crop_image(tile: Tile, visualize=False):
    if tile is None or tile.image is None:
        raise ValueError("Tile or image is None")

    # Define the bounding box coordinates for cropping
    CROP_POINTS = np.array(
        [
            (3840, 40),
            (620, 40),
            (620, 2130),
            (3840, 2130),
        ],
        dtype=np.int32,
    )

    # Get the bounding rectangle for the crop area
    x, y, w, h = cv2.boundingRect(CROP_POINTS.astype(np.int32))
    delta = np.array([x, y])

    # Crop the region of interest (ROI) from the image
    cropped_image = tile.image[y : y + h, x : x + w]

    # Update the tile coordinates to reflect the cropped region
    tile.top_left -= delta
    tile.top_right -= delta
    tile.bottom_left -= delta
    tile.bottom_right -= delta

    tile.image = cropped_image

    # Add circles to the corners of the cropped image for visualization
    if visualize:
        cv2.circle(cropped_image, tile.top_left, 10, (255, 0, 0), -1)  # Blue in BGR
        cv2.circle(cropped_image, tile.top_right, 10, (0, 255, 0), -1)  # Green in BGR
        cv2.circle(cropped_image, tile.bottom_left, 10, (0, 0, 255), -1)  # Red in BGR
        cv2.circle(
            cropped_image, tile.bottom_right, 10, (0, 255, 255), -1
        )  # Yellow in BGR
        cv2.imshow("Cropped Image", cropped_image)
        cv2.waitKey(0)

    return tile
