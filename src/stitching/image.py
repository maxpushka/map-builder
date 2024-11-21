from typing import Callable
import cv2
import numpy as np

from coordinates import MGRSCoordinate, RelativePosition
from tile import GridLineIntersections, Tile


def stitch_tiles(tile_a: Tile, tile_b: Tile, opacity=1.0) -> Tile:
    # Ensure the larger tile is always first
    if (
        tile_a.image.shape[0] * tile_a.image.shape[1]
        < tile_b.image.shape[0] * tile_b.image.shape[1]
    ):
        tile_a, tile_b = tile_b, tile_a

    # Find the nearest grid intersection between the two tiles
    position, adjacent_grid_a, adjacent_grid_b = None, None, None
    for mgrs_a, grid_a in tile_a.grid.items():
        for direction in RelativePosition:
            if direction == RelativePosition.NO_RELATION:
                continue
            nearest_mgrs = mgrs_a.nearest(direction)
            if nearest_mgrs in tile_b.grid:
                position = direction
                adjacent_grid_a = grid_a
                adjacent_grid_b = tile_b.grid[nearest_mgrs]
                break
    if position is None or adjacent_grid_a is None or adjacent_grid_b is None:
        raise ValueError("No common grid intersection found between the two tiles")

    # Calculate x and y offsets based on points of interest in `Tile`
    offset = _compute_offset(position, adjacent_grid_a, adjacent_grid_b)

    # Extract the calculated offsets
    x_offset, y_offset = offset
    height_a, width_a = tile_a.image.shape[:2]
    height_b, width_b = tile_b.image.shape[:2]

    # Calculate the bounding box of the combined image
    canvas_width = (
        max(width_a, width_a + x_offset) if x_offset > 0 else width_a - x_offset
    )
    canvas_height = (
        max(height_a, height_a + y_offset) if y_offset > 0 else height_a - y_offset
    )

    # Create a blank canvas with the calculated dimensions and 4 channels (RGBA),
    # and initialize it as fully transparent
    canvas = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)

    # Place tile_a on the canvas, adjusting for negative offsets
    x_a, y_a = (max(-x_offset, 0), max(-y_offset, 0))
    canvas[y_a : y_a + height_a, x_a : x_a + width_a] = tile_a.image

    # Calculate the position to place tile_b on the canvas based on offsets
    x_b, y_b = (x_a + x_offset, y_a + y_offset)

    # Overlay tile_b on the canvas at the calculated position with specified opacity
    tile_b_region = canvas[y_b : y_b + height_b, x_b : x_b + width_b]
    blended = cv2.addWeighted(
        tile_b_region[:, :, :3], 1 - opacity, tile_b.image[:, :, :3], opacity, 0
    )

    # Combine alpha channel of tile_b with existing canvas alpha
    alpha_b = tile_b.image[:, :, 3] * opacity
    alpha_existing = tile_b_region[:, :, 3] * (1 - opacity)
    combined_alpha = np.clip(alpha_b + alpha_existing, 0, 255)

    # Assign RGB and alpha values back to canvas
    tile_b_region[:, :, :3] = blended
    tile_b_region[:, :, 3] = combined_alpha

    # Adjust tile_a's grid based on its placement offsets on the canvas
    adjusted_grid_a = {}
    for mgrs_a, grid_a in tile_a.grid.items():
        adjusted_intersections = GridLineIntersections()
        adjusted_intersections.top_left = _clamp_to_canvas(
            grid_a.top_left + np.array([x_a, y_a]), canvas.shape
        )
        adjusted_intersections.top_right = _clamp_to_canvas(
            grid_a.top_right + np.array([x_a, y_a]), canvas.shape
        )
        adjusted_intersections.bottom_left = _clamp_to_canvas(
            grid_a.bottom_left + np.array([x_a, y_a]), canvas.shape
        )
        adjusted_intersections.bottom_right = _clamp_to_canvas(
            grid_a.bottom_right + np.array([x_a, y_a]), canvas.shape
        )
        adjusted_grid_a[mgrs_a] = adjusted_intersections

    # Adjust tile_b's grid based on its placement offsets on the canvas
    adjusted_grid_b = {}
    for mgrs_b, grid_b in tile_b.grid.items():
        adjusted_intersections = GridLineIntersections()
        adjusted_intersections.top_left = _clamp_to_canvas(
            grid_b.top_left + np.array([x_b, y_b]), canvas.shape
        )
        adjusted_intersections.top_right = _clamp_to_canvas(
            grid_b.top_right + np.array([x_b, y_b]), canvas.shape
        )
        adjusted_intersections.bottom_left = _clamp_to_canvas(
            grid_b.bottom_left + np.array([x_b, y_b]), canvas.shape
        )
        adjusted_intersections.bottom_right = _clamp_to_canvas(
            grid_b.bottom_right + np.array([x_b, y_b]), canvas.shape
        )
        adjusted_grid_b[mgrs_b] = adjusted_intersections

    # Merge the adjusted grids
    merged_grid = {**adjusted_grid_a, **adjusted_grid_b}

    # Create and return a new Tile with the merged image and grid
    return Tile.from_tile(canvas, merged_grid)


def _compute_offset(
    position: RelativePosition,
    grid_a: GridLineIntersections,
    grid_b: GridLineIntersections,
) -> np.ndarray:
    if position == RelativePosition.ABOVE:
        offset = grid_a.top_left - grid_b.bottom_left
    elif position == RelativePosition.BELOW:
        offset = grid_a.bottom_left - grid_b.top_left
    elif position == RelativePosition.LEFT:
        offset = grid_a.top_left - grid_b.top_right
    elif position == RelativePosition.RIGHT:
        offset = grid_a.top_right - grid_b.top_left
    elif position == RelativePosition.ABOVE_LEFT:
        offset = grid_a.top_left - grid_b.bottom_right
    elif position == RelativePosition.ABOVE_RIGHT:
        offset = grid_a.top_right - grid_b.bottom_left
    elif position == RelativePosition.BELOW_LEFT:
        offset = grid_a.bottom_left - grid_b.top_right
    elif position == RelativePosition.BELOW_RIGHT:
        offset = grid_a.bottom_right - grid_b.top_left
    elif position == RelativePosition.CENTER:
        offset = 0
    else:
        raise ValueError(f"Invalid relative position: {position}")
    return offset


def _clamp_to_canvas(point, canvas_shape):
    """Clamp a point to the canvas dimensions."""
    y_max, x_max = canvas_shape[:2]
    return np.array(
        [max(0, min(point[0], x_max - 1)), max(0, min(point[1], y_max - 1))]
    )


if __name__ == "__main__":
    # Load the images and parse coordinates
    # NOTE: download the images manually!
    # They're not provided in the repo.
    root = "/Users/maxpushka/dev/github.com/maxpushka/map-builder/src/stitching/images/map/a"
    name_a = f"{root}/37_T_FJ_00500_00500.png"
    name_b = f"{root}/37_T_FJ_00500_01500.png"
    name_c = f"{root}/37_T_FJ_01500_00500.png"
    name_d = f"{root}/37_T_FJ_01500_01500.png"

    # Create Tile instances
    to_mgrs: Callable[[str], MGRSCoordinate] = (
        lambda name: MGRSCoordinate.from_filename(name.split("/")[-1].split(".")[0])
    )
    tile_a = Tile(cv2.imread(name_a), to_mgrs(name_a))
    tile_b = Tile(cv2.imread(name_b), to_mgrs(name_b))
    tile_c = Tile(cv2.imread(name_c), to_mgrs(name_c))
    tile_d = Tile(cv2.imread(name_d), to_mgrs(name_d))

    opacity = 1

    def display_result(tile: Tile):
        window_name = "Image Overlay"
        cv2.imshow(window_name, tile.image)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow(window_name)
        if key == ord("s"):
            cv2.imwrite("combined_image.png", tile.image)

    # Overlay images based on corner alignment
    # Case 1: Merge equally sized tiles
    # tile_ab = stitch_images(tile_a, tile_b, opacity)
    # tile_cd = stitch_images(tile_c, tile_d, opacity)
    # combined = stitch_images(tile_ab, tile_cd, opacity)

    # Case 2: Merge tiles with different sizes
    tile_ab = stitch_tiles(tile_a, tile_b, opacity)
    tile_abc = stitch_tiles(
        tile_c, tile_ab, opacity
    )  # here tile C is smaller than tile AB
    combined = stitch_tiles(tile_abc, tile_d, opacity)

    # Case 3: Merge tiles of irregular shapes
