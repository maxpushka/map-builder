import os
import hashlib
import random
import string
from typing import Callable, Tuple
import cv2
import numpy as np

from coordinates import MGRSCoordinate, RelativePosition
from tile import GridLineIntersections, Tile


def stitch_tiles(tile_a: Tile, tile_b: Tile, cache_dir: str, opacity=1.0) -> Tile:
    # Ensure the larger tile is always first
    if (
        tile_a.image().shape[0] * tile_a.image().shape[1]
        < tile_b.image().shape[0] * tile_b.image().shape[1]
    ):
        tile_a, tile_b = tile_b, tile_a

    # Find the nearest grid intersection between the two tiles
    position, adjacent_grid_a, adjacent_grid_b = _find_intersection(tile_a, tile_b)

    # Calculate x and y offsets based on points of interest in `Tile`
    offset = _compute_offset(position, adjacent_grid_a, adjacent_grid_b)

    # Stitch the two tiles together
    canvas, a_position, b_position = _stitch_tiles(tile_a, tile_b, offset, opacity)

    # Merge the grids of the two tiles
    merged_grid = _merge_grids(tile_a, tile_b, canvas, a_position, b_position)

    # Write the merged image to the disk to avoid keeping it in memory
    canvas_path = f"{cache_dir}/{_generate_random_hash()}.npy"
    np.save(canvas_path, canvas)
    cv2.imwrite(canvas_path.replace(".npy", ".png"), canvas)

    # Create and return a new Tile with the merged image and grid
    return Tile.from_tile(canvas_path, merged_grid)


def _find_intersection(
    tile_a: Tile, tile_b: Tile
) -> Tuple[RelativePosition, GridLineIntersections, GridLineIntersections]:
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
    return (position, adjacent_grid_a, adjacent_grid_b)


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


def _stitch_tiles(
    tile_a: Tile, tile_b: Tile, offset: np.ndarray, opacity: float
) -> Tuple[cv2.typing.MatLike, tuple[int, int], tuple[int, int]]:
    # Extract the calculated offsets
    x_offset, y_offset = offset
    height_a, width_a = tile_a.image().shape[:2]
    height_b, width_b = tile_b.image().shape[:2]

    # Calculate the bounding box of the final canvas
    canvas_width = (
        max(width_a, width_a + x_offset) if x_offset > 0 else width_a - x_offset
    )
    canvas_height = (
        max(height_a, height_a + y_offset) if y_offset > 0 else height_a - y_offset
    )

    # Create two blank canvases for tile_a and tile_b with the same dimensions
    canvas_a = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)
    canvas_b = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)

    # Place tile_a on canvas_a
    x_a, y_a = max(-x_offset, 0), max(-y_offset, 0)
    canvas_a[y_a : y_a + height_a, x_a : x_a + width_a] = tile_a.image()

    # Place tile_b on canvas_b with bounds checking
    x_b, y_b = x_a + x_offset, y_a + y_offset
    x_b_start = max(x_b, 0)
    y_b_start = max(y_b, 0)
    x_b_end = min(x_b + width_b, canvas_width)
    y_b_end = min(y_b + height_b, canvas_height)

    tile_b_x_start = max(0, -x_b)
    tile_b_y_start = max(0, -y_b)
    tile_b_x_end = tile_b_x_start + (x_b_end - x_b_start)
    tile_b_y_end = tile_b_y_start + (y_b_end - y_b_start)

    canvas_b[y_b_start:y_b_end, x_b_start:x_b_end] = tile_b.image()[
        tile_b_y_start:tile_b_y_end, tile_b_x_start:tile_b_x_end
    ]

    # Merge the two canvases so that non-alpha pixels from canvas_b overwrite those in canvas_a
    blended_canvas = canvas_a.copy()
    alpha_b = canvas_b[:, :, 3] > 0  # Mask where canvas_b has non-transparent pixels
    blended_canvas[alpha_b] = canvas_b[alpha_b]

    # Crop the blended canvas to remove transparent areas
    alpha_channel = blended_canvas[:, :, 3]
    non_empty_rows = np.nonzero(alpha_channel.max(axis=1))[0]
    non_empty_cols = np.nonzero(alpha_channel.max(axis=0))[0]

    if non_empty_rows.size > 0 and non_empty_cols.size > 0:
        crop_top, crop_bottom = non_empty_rows[0], non_empty_rows[-1] + 1
        crop_left, crop_right = non_empty_cols[0], non_empty_cols[-1] + 1
        cropped_canvas = blended_canvas[crop_top:crop_bottom, crop_left:crop_right]
        return (
            cropped_canvas,
            (x_a - crop_left, y_a - crop_top),
            (x_b - crop_left, y_b - crop_top),
        )
    else:
        # If no non-transparent pixels are found, return an empty canvas
        return blended_canvas, (x_a, y_a), (x_b, y_b)


def _merge_grids(
    tile_a: Tile,
    tile_b: Tile,
    canvas: cv2.typing.MatLike,
    a_position: tuple[int, int],
    b_position: tuple[int, int],
):
    # Adjust tile_a's grid based on its placement offsets on the canvas
    adjusted_grid_a = {}
    x_a, y_a = a_position
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
    x_b, y_b = b_position
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
    return merged_grid


def _clamp_to_canvas(point, canvas_shape):
    """Clamp a point to the canvas dimensions."""
    y_max, x_max = canvas_shape[:2]
    return np.array(
        [max(0, min(point[0], x_max - 1)), max(0, min(point[1], y_max - 1))]
    )


def _generate_random_hash():
    # Generate a random string
    random_string = "".join(random.choices(string.ascii_letters + string.digits, k=16))
    # Create a hash object
    hash_object = hashlib.sha256(random_string.encode())
    return hash_object.hexdigest()


def display_result(
    tile: Tile, filename: str = "output.png", window_name: str = "Image Overlay"
):
    cv2.imshow(window_name, tile.image())
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyWindow(window_name)
    if key == ord("s"):
        cv2.imwrite(filename, tile.image())


if __name__ == "__main__":
    ROOT_DIR = "."
    
    # Load the images and parse coordinates
    # NOTE: download the images manually!
    # They're not provided in the repo.
    CACHE_DIR = f"{ROOT_DIR}/cache"
    os.makedirs(CACHE_DIR, exist_ok=True)

    DATA_DIR = f"{ROOT_DIR}/images"
    name_a = f"{DATA_DIR}/37TFJ0050000500.png"
    name_b = f"{DATA_DIR}/37TFJ0050001500.png"
    name_c = f"{DATA_DIR}/37TFJ0150000500.png"
    name_d = f"{DATA_DIR}/37TFJ0150001500.png"

    # Create Tile instances
    to_mgrs: Callable[[str], MGRSCoordinate] = (
        lambda name: MGRSCoordinate.from_filename(name.split("/")[-1].split(".")[0])
    )
    tile_a = Tile(name_a, to_mgrs(name_a))
    tile_b = Tile(name_b, to_mgrs(name_b))
    tile_c = Tile(name_c, to_mgrs(name_c))
    tile_d = Tile(name_d, to_mgrs(name_d))

    opacity = 1

    # Overlay images based on corner alignment

    # Case 1: Merge equally sized tiles
    tile_ab = stitch_tiles(tile_a, tile_b, CACHE_DIR, opacity)
    tile_cd = stitch_tiles(tile_c, tile_d, CACHE_DIR, opacity)
    equally_sized_tiles = stitch_tiles(tile_ab, tile_cd, CACHE_DIR, opacity)
    display_result(
        equally_sized_tiles, "equally_sized_image.png", "Equally Sized Image Overlay"
    )

    # Case 2: Merge tiles with different sizes
    tile_ab = stitch_tiles(tile_a, tile_b, CACHE_DIR, opacity)
    tile_abc = stitch_tiles(
        tile_c, tile_ab, CACHE_DIR, opacity
    )  # here tile C is smaller than tile AB
    different_sized_tiles = stitch_tiles(tile_abc, tile_d, CACHE_DIR, opacity)
    display_result(
        different_sized_tiles,
        "different_sized_image.png",
        "Different Sized Image Overlay",
    )

    # Case 3: Merge tiles of irregular shapes
