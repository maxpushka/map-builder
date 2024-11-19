# import mgrs


# class MGRSIterator:
#     def __init__(self, start_mgrs, step_size=1000):
#         self.mgrs_converter = mgrs.MGRS()
#         self.step_size = step_size  # Step size in meters

#         # Convert initial MGRS to UTM coordinates
#         self.zone, self.hemisphere, self.easting, self.northing = (
#             self.mgrs_converter.MGRSToUTM(start_mgrs)
#         )

#     def __iter__(self):
#         return self

#     def __next__(self):
#         # Increment UTM easting by step_size (move eastward by step_size meters)
#         self.easting += self.step_size

#         # Convert updated UTM coordinates back to MGRS
#         new_mgrs = self.mgrs_converter.UTMToMGRS(
#             self.zone, self.hemisphere, self.easting, self.northing
#         )

#         return new_mgrs


# if __name__ == "__main__":
#     start_mgrs = "37TGJ3850068500"  # Replace with your starting MGRS coordinate
#     iterator = MGRSIterator(start_mgrs, step_size=1000)  # Step size in meters

#     for i, coord in zip(range(10), iterator):  # Limit to 10 steps for demonstration
#         print(coord)


class MGRSCoordinateIterator:
    def __init__(
        self,
        start_zone,
        end_zone,
        start_letter,
        end_letter,
        start_square,
        end_square,
        skip_low_value_func=None,
    ):
        self.start_zone = start_zone
        self.end_zone = end_zone
        self.start_letter = start_letter
        self.end_letter = end_letter
        self.start_square = start_square
        self.end_square = end_square
        self.current_zone = start_zone
        self.current_letter = ord(start_letter)
        self.grid_squares_list: list[str] = self.grid_squares()
        self.current_square_index = self.grid_squares_list.index(start_square)
        self.easting = 0
        self.northing = 0
        self.skip_low_value_func = skip_low_value_func

    def grid_squares(self):
        # The letter I is excluded from MGRS grid squares
        # because of potential confusion with the number 1,
        # which can lead to misinterpretation of coordinates.
        # This exclusion of I (along with sometimes O
        # to avoid confusion with 0 in some systems)
        # is a convention that helps ensure clarity in MGRS
        # and other military coordinate systems where accurate,
        # unambiguous readings are crucial.
        GRID_ZONE_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"
        squares = []
        for first_letter in GRID_ZONE_LETTERS:
            for second_letter in GRID_ZONE_LETTERS:
                squares.append(f"{first_letter}{second_letter}")
        return squares

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            # Check if we have completed all zones, letters, and squares
            if self.current_zone > self.end_zone:
                raise StopIteration
            if self.current_letter > ord(self.end_letter):
                raise StopIteration
            if (
                self.current_square_index >= len(self.grid_squares_list)
                or self.grid_squares_list[self.current_square_index] > self.end_square
            ):
                raise StopIteration

            # Format the current tile in 1-meter precision, centered in 1km tiles (XX500)
            easting_str = f"{self.easting:02}500"
            northing_str = f"{self.northing:02}500"
            coord = (
                f"{self.current_zone}",
                chr(self.current_letter),
                self.grid_squares_list[self.current_square_index],
                easting_str,
                northing_str,
            )

            # Apply the filter function to skip low-value coordinates
            if self.skip_low_value_func and self.skip_low_value_func(coord):
                self._increment_coordinates()
                continue

            # Proceed to the next coordinate
            self._increment_coordinates()
            return coord

    def _increment_coordinates(self):
        # Increment easting and northing for 1km tiles
        self.easting += 1
        if self.easting > 99:  # Reset easting and move northing
            self.easting = 0
            self.northing += 1
            if self.northing > 99:  # Reset northing, move to the next square
                self.northing = 0
                self.current_square_index += 1
                if (
                    self.current_square_index >= len(self.grid_squares_list)
                    or self.grid_squares_list[self.current_square_index]
                    > self.end_square
                ):
                    self.current_square_index = self.grid_squares_list.index(
                        self.start_square
                    )
                    self.current_letter += 1  # Move to the next letter
                    if self.current_letter > ord(self.end_letter):
                        self.current_letter = ord(self.start_letter)
                        self.current_zone += 1


# Example filter function for skipping low-value coordinates
def skip_sea_coordinates(coord):
    zone, letter, square, easting, northing = coord
    # Example criteria: Skip certain squares with mostly sea regions

    if zone != "37" or letter != "T" or square != "FH":
        return True  # Skip all coordinates outside the '37 T FH' square

    e = int(easting[0])
    n = int(northing[0])

    _1 = 0 <= e <= 4 and 0 <= n <= 6
    _2 = 5 <= e <= 9 and 0 <= n <= 2
    _3 = 5 <= e <= 6 and 3 <= n <= 4
    _4 = e == 5 and n == 5
    _5 = e == 7 and n == 3
    low_value = _1 or _2 or _3 or _4 or _5
    return low_value


# Example usage:
if __name__ == "__main__":
    low_value_coords = [
        ("37", "T", "FH", "00500", "60500"),
        ("37", "T", "FH", "00500", "00500"),
        ("37", "T", "FH", "20500", "40500"),
        ("37", "T", "FH", "50500", "50500"),
        ("37", "T", "FH", "70500", "30500"),
        ("37", "T", "FH", "80500", "20500"),
    ]
    for coord in low_value_coords:
        assert skip_sea_coordinates(coord) == True
    assert skip_sea_coordinates(("37", "T", "FH", "8", "3")) == False

    coord_iterator = MGRSCoordinateIterator(
        start_zone=37,
        end_zone=37,
        start_letter="T",
        end_letter="T",
        start_square="FH",
        end_square="FH",
        skip_low_value_func=skip_sea_coordinates,
    )

    with open("coordinates.txt", "w") as file:
        for coord in coord_iterator:
            file.write(f"{coord}\n")
