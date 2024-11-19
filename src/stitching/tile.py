from enum import Enum
import cv2
import numpy as np


class RelativePosition(Enum):
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


class GridLineIntersections:
    top_left: np.array
    top_right: np.array
    bottom_left: np.array
    bottom_right: np.array

    def __init__(self):
        self.top_left = np.array([1135, 386])
        self.top_right = np.array([2644, 277])
        self.bottom_left = np.array([1244, 1895])
        self.bottom_right = np.array([2753, 1786])


class Tile:
    image: np.ndarray
    grid: dict[str, GridLineIntersections]

    def __init__(self, image: np.ndarray, coord: str, do_crop=True):
        self.image = image
        self.grid = {coord: GridLineIntersections()}
        if do_crop:
            self._crop_image()

    @classmethod
    def from_tile(cls, image: np.ndarray, grid: dict[str, GridLineIntersections]):
        tile = cls(image, "", do_crop=False)
        tile.grid = grid
        return tile

    # Function to crop image around a given bounding box
    def _crop_image(self, visualize=False):
        if self.image is None:
            raise ValueError("Tile or image is None")
        if len(self.grid) > 1:
            raise ValueError("Crop is applied for a one-image tile only")

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
        cropped_image = self.image[y : y + h, x : x + w]

        # Get the grid lines for the tile
        mgrs, grid = next(iter(self.grid.items()))

        # Update the tile coordinates to reflect the cropped region
        grid.top_left -= delta
        grid.top_right -= delta
        grid.bottom_left -= delta
        grid.bottom_right -= delta

        self.image = cropped_image
        self.grid = {mgrs: grid}

        # Add circles to the corners of the cropped image for visualization
        if visualize:
            cv2.circle(cropped_image, grid.top_left, 10, (0, 0, 255), -1)  # Red in BGR
            cv2.circle(
                cropped_image, grid.top_right, 10, (0, 255, 0), -1
            )  # Green in BGR
            cv2.circle(
                cropped_image, grid.bottom_left, 10, (255, 0, 0), -1
            )  # Blue in BGR
            cv2.circle(
                cropped_image, grid.bottom_right, 10, (0, 255, 255), -1
            )  # Yellow in BGR
            cv2.imshow("Cropped Image", cropped_image)
            cv2.waitKey(0)
