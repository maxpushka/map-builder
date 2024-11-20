from enum import Enum
import cv2
import numpy as np

from coordinates import MGRSCoordinate


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
    grid: dict[MGRSCoordinate, GridLineIntersections]

    def __init__(self, image: np.ndarray, coord: MGRSCoordinate, do_crop=True):
        self.image = image
        self.grid = {coord: GridLineIntersections()}
        if do_crop:
            self._crop_image()

    @classmethod
    def from_tile(cls, image: np.ndarray, grid: dict[MGRSCoordinate, GridLineIntersections]):
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
                (620, 2124),
                (3840, 2124),
            ],
            dtype=np.int32,
        )

        # Get the bounding rectangle for the crop area
        x, y, w, h = cv2.boundingRect(CROP_POINTS)
        delta = np.array([x, y])

        # Crop the region of interest (ROI) from the image
        cropped_image = self.image[y : y + h, x : x + w]

        # Create an alpha channel mask based on the crop points
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [CROP_POINTS - delta], 255)  # Adjust points for mask region

        # Ensure the cropped image has an alpha channel
        if cropped_image.shape[2] == 3:  # If RGB, add alpha channel
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2BGRA)

        # Resize the mask to match the cropped image's dimensions if necessary
        mask = cv2.resize(mask, (cropped_image.shape[1], cropped_image.shape[0]))

        # Apply the mask to the alpha channel (outside area becomes transparent)
        cropped_image[:, :, 3] = mask

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
            cv2.circle(cropped_image, grid.top_left, 10, (0, 0, 255, 255), -1)  # Red
            cv2.circle(cropped_image, grid.top_right, 10, (0, 255, 0, 255), -1)  # Green
            cv2.circle(
                cropped_image, grid.bottom_left, 10, (255, 0, 0, 255), -1
            )  # Blue
            cv2.circle(
                cropped_image, grid.bottom_right, 10, (0, 255, 255, 255), -1
            )  # Yellow
            cv2.imshow("Cropped Image with Transparency", cropped_image)
            cv2.waitKey(0)
