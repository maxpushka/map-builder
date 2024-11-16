import pyautogui
import pygetwindow
import pydirectinput
import time
import os
from coordinate_iterator import MGRSCoordinateIterator


window_title = "Digital Combat Simulator"

# Define screen coordinates for automation steps.
# Run cursor_tracking.py to find out
# actual coordinates on your machine.
zone_input_field_coords = (104, 120)
letter_input_field_coords = (133, 120)
square_input_field_coords = (170, 120)
northing_input_field_coords = (233, 120)
easting_input_field_coords = (333, 120)
input_field_coords = (
    zone_input_field_coords,
    letter_input_field_coords,
    square_input_field_coords,
    northing_input_field_coords,
    easting_input_field_coords,
)

ok_button_coords = (336, 174)
alt_mode_button_coords = (1005, 2143)
sat_mode_button_coords = (959, 2144)

coord_iterator = MGRSCoordinateIterator(
    start_zone=37,
    end_zone=37,
    start_letter="T",
    end_letter="T",
    start_square="FJ",
    end_square="FJ",
)


def wait_for_dcs_window(log=True):
    active_window = False
    while True:
        if (not active_window) and log:
            print("Waiting for DCS window to be in focus...")
        active_window = pygetwindow.getActiveWindow()
        if active_window and window_title in active_window.title:
            if active_window and log:
                print("DCS window is in focus.")
            break
        time.sleep(0.1)


def wait_button_input(enabled=True):
    if enabled:
        time.sleep(0.01)


def wait_map_mode_switch(enabled=True):
    if enabled:
        time.sleep(0.01)


if __name__ == "__main__":
    # Create directories for screenshots
    alt_prefix = "alt"
    # sat_prefix = "sat"
    os.makedirs(alt_prefix, exist_ok=True)
    # os.makedirs(sat_prefix, exist_ok=True)

    previous_coord = [None] * len(input_field_coords)

    # Main loop to automate input and screenshot capture
    for coord in coord_iterator:
        start_time = time.time()
        wait_for_dcs_window()

        alt_filename = f"{alt_prefix}/{'_'.join(coord)}.png"
        # sat_filename = f"{sat_prefix}/{'_'.join(coord)}.png"
        if os.path.exists(alt_filename):# and os.path.exists(sat_filename):
            continue

        # Step 1: Enter MGRS coordinates and navigate with tabs
        print(f"Entering MGRS coordinates: {coord}")
        for i, (item, input_coords) in enumerate(zip(coord, input_field_coords)):
            # Check if the value has changed from the previous iteration
            if previous_coord[i] == item:
                continue
            pydirectinput.doubleClick(*input_coords)
            pyautogui.write(str(item))  # Type the MGRS part
            previous_coord[i] = item  # Update the last value
        pydirectinput.click(*ok_button_coords)  # Click OK button
        wait_button_input()

        # # Step 2: Take a screenshot in Altitude mode
        # # pydirectinput.click(*alt_mode_button_coords)
        # wait_map_mode_switch()
        pyautogui.screenshot(alt_filename)
        print(time.time() - start_time)

        # # Step 3: Take a screenshot in Satellite mode
        # pydirectinput.click(*sat_mode_button_coords)
        # wait_map_mode_switch()
        # pyautogui.screenshot(sat_filename)
