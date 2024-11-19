import cv2
import numpy as np
import pyautogui

from common import crop_image, parse_coordinates


# Function to blend images with given offsets
def overlay_images(image_a, image_b, x_offset, y_offset, opacity=1.0):
    # Calculate canvas size to fit both images fully
    canvas_width = max(image_a.shape[1], image_b.shape[1] + abs(x_offset))
    canvas_height = max(image_a.shape[0], image_b.shape[0] + abs(y_offset))

    # Create a blank canvas based on the calculated size
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # Place image_a at the center of the canvas
    canvas[: image_a.shape[0], : image_a.shape[1]] = image_a

    # Calculate the overlay area and handle edges
    x1, y1 = max(0, x_offset), max(0, y_offset)
    x2, y2 = min(canvas.shape[1], x_offset + image_b.shape[1]), min(
        canvas.shape[0], y_offset + image_b.shape[0]
    )
    canvas_section = canvas[y1:y2, x1:x2]

    # Calculate the region of image_b to overlay
    x1_b, y1_b = max(0, -x_offset), max(0, -y_offset)
    x2_b, y2_b = x1_b + (x2 - x1), y1_b + (y2 - y1)
    image_b_section = image_b[y1_b:y2_b, x1_b:x2_b]

    # Blend the images in the region
    blended_section = cv2.addWeighted(
        canvas_section, 1 - opacity, image_b_section, opacity, 0
    )
    canvas[y1:y2, x1:x2] = blended_section

    return canvas


if __name__ == "__main__":
    # Load the images
    name_a = "images/map/a/37_T_FJ_01500_00500.png"
    name_b = "images/map/a/37_T_FJ_01500_01500.png"
    image_a = cv2.imread(name_a)
    image_b = cv2.imread(name_b)

    do_crop = True
    if do_crop:
        image_a = crop_image(image_a)
        image_b = crop_image(image_b)

    # Determine initial offsets based on the coordinates
    coord_a = parse_coordinates(name_a.split("/")[-1].split(".")[0])
    coord_b = parse_coordinates(name_b.split("/")[-1].split(".")[0])
    if coord_a.easting < coord_b.easting and coord_a.northing == coord_b.northing:
        x_offset, y_offset = 1510, -110
    elif coord_a.easting > coord_b.easting and coord_a.northing == coord_b.northing:
        x_offset, y_offset = 1510, -110
        name_a, name_b = name_b, name_a
        image_a, image_b = image_b, image_a
    elif coord_a.easting == coord_b.easting and coord_a.northing > coord_b.northing:
        x_offset, y_offset = 110, 1510
    elif coord_a.easting == coord_b.easting and coord_a.northing < coord_b.northing:
        x_offset, y_offset = 110, 1510
        name_a, name_b = name_b, name_a
        image_a, image_b = image_b, image_a
    else:
        raise ValueError("No intersection found between the images")

    # Set opacity for the second image
    opacity_toggle = True
    opacity_default = 0.5
    opacity = opacity_default
    scale = 1.0

    # Display window and handle key events
    cv2.namedWindow("Image Overlay")
    while True:
        # Overlay images with current offsets and scaled images
        combined_image = overlay_images(
            image_a, image_b, x_offset, y_offset, opacity
        )
        combined_image = cv2.resize(
            combined_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
        )
        cv2.imshow("Image Overlay", combined_image)

        # Capture key press
        key = cv2.waitKey(10) & 0xFF

        if key == ord("w"):
            y_offset -= 10
        elif key == ord("s"):
            y_offset += 10
        elif key == ord("a"):
            x_offset -= 10
        elif key == ord("d"):
            x_offset += 10
        elif key == ord("e"):
            print(f"Current offsets: X = {x_offset}, Y = {y_offset}")
        elif key == ord("c"):
            cv2.imwrite("result.png", combined_image)
        elif key == ord("o"):
            if opacity_toggle:
                opacity = 1.0
            else:
                opacity = opacity_default
            opacity_toggle = not opacity_toggle
        elif key == ord("z"):
            scale += 0.1  # Increase zoom
        elif key == ord("x") and scale > 0.1:
            scale -= 0.1  # Decrease zoom
        elif key == ord("q") or key == 27:  # Escape key
            break

    cv2.destroyAllWindows()
