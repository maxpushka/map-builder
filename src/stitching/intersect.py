from dataclasses import dataclass
import cv2
import numpy as np
from common import MGRSCoordinates, MRGSRelativePosition, Tile, crop_image


def stitch_images(tile_a: Tile, tile_b: Tile, opacity=1.0):
    # Determine the relative position of tile_b with respect to tile_a
    position = tile_a.mgrs_coord.position(tile_b.mgrs_coord)

    # Calculate x and y offsets based on points of interest in `Tile`
    if position == MRGSRelativePosition.ABOVE:
        offset = tile_a.top_left - tile_b.bottom_left
    elif position == MRGSRelativePosition.BELOW:
        offset = tile_a.bottom_left - tile_b.top_left
    elif position == MRGSRelativePosition.LEFT:
        offset = tile_a.top_left - tile_b.top_right
    elif position == MRGSRelativePosition.RIGHT:
        offset = tile_a.top_right - tile_b.top_left
    elif position == MRGSRelativePosition.ABOVE_LEFT:
        offset = tile_a.top_left - tile_b.bottom_right
    elif position == MRGSRelativePosition.ABOVE_RIGHT:
        offset = tile_a.top_right - tile_b.bottom_left
    elif position == MRGSRelativePosition.BELOW_LEFT:
        offset = tile_a.bottom_left - tile_b.top_right
    elif position == MRGSRelativePosition.BELOW_RIGHT:
        offset = tile_a.bottom_right - tile_b.top_left
    else:
        raise ValueError(f"Invalid relative position: {position}")

    # Extract the calculated offsets
    x_offset, y_offset = offset
    print(f"x_offset: {x_offset}, y_offset: {y_offset}")
    height_a, width_a = tile_a.image.shape[:2]
    height_b, width_b = tile_b.image.shape[:2]

    # Calculate the bounding box of the combined image
    canvas_width = (
        max(width_a, width_a + x_offset) if x_offset > 0 else width_a - x_offset
    )
    canvas_height = (
        max(height_a, height_a + y_offset) if y_offset > 0 else height_a - y_offset
    )

    # Create a blank canvas with the calculated dimensions
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # Place tile_a on the canvas, adjusting for negative offsets
    x_a, y_a = (max(-x_offset, 0), max(-y_offset, 0))
    canvas[y_a : y_a + height_a, x_a : x_a + width_a] = tile_a.image

    # Calculate the position to place tile_b on the canvas based on offsets
    x_b, y_b = (x_a + x_offset, y_a + y_offset)

    # Overlay tile_b on the canvas at the calculated position with specified opacity
    blended = cv2.addWeighted(
        canvas[y_b : y_b + height_b, x_b : x_b + width_b],
        1 - opacity,
        tile_b.image[:height_b, :width_b],
        opacity,
        0,
    )
    canvas[y_b : y_b + height_b, x_b : x_b + width_b] = blended

    return canvas


if __name__ == "__main__":
    # Load the images and parse coordinates
    name_a = "/Users/maxpushka/dev/github.com/maxpushka/map-builder/src/stitching/images/map/a/37_T_FJ_01500_01500.png"
    name_b = "/Users/maxpushka/dev/github.com/maxpushka/map-builder/src/stitching/images/map/a/37_T_FJ_01500_00500.png"
    image_a = cv2.imread(name_a)
    image_b = cv2.imread(name_b)
    coord_a = MGRSCoordinates(name_a.split("/")[-1].split(".")[0])
    coord_b = MGRSCoordinates(name_b.split("/")[-1].split(".")[0])
    print(coord_a.position(coord_b))

    # Create Tile instances
    tile_a = Tile(
        name=name_a,
        image=image_a,
        mgrs_coord=coord_a,
        top_left=(1135, 386),
        top_right=(2644, 277),
        bottom_left=(1244, 1895),
        bottom_right=(2753, 1786),
    )
    tile_b = Tile(
        name=name_b,
        image=image_b,
        mgrs_coord=coord_b,
        top_left=(1135, 386),
        top_right=(2644, 277),
        bottom_left=(1244, 1895),
        bottom_right=(2753, 1786),
    )

    # Crop images and update coordinates
    tile_a = crop_image(tile_a)
    tile_b = crop_image(tile_b)

    # Overlay images based on corner alignment
    combined_image = stitch_images(tile_a, tile_b, opacity=0.5)

    # Display the result
    cv2.imshow("Image Overlay", combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
