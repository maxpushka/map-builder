class MGRSCoordinateIterator:
    def __init__(self, start_zone, end_zone, start_letter, end_letter, start_square, end_square):
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
        # Dynamically generate two-letter grid squares (AA to EZ, skipping 'I' in both positions)
        squares = []
        for first_letter in GRID_ZONE_LETTERS:
            for second_letter in GRID_ZONE_LETTERS:
                squares.append(f"{first_letter}{second_letter}")
        return squares

    def __iter__(self):
        return self

    def __next__(self):
        # Check if we have completed all zones, letters, and squares
        if self.current_zone > self.end_zone:
            raise StopIteration
        if self.current_letter > ord(self.end_letter):
            raise StopIteration
        if self.current_square_index >= len(self.grid_squares_list) or \
           self.grid_squares_list[self.current_square_index] > self.end_square:
            raise StopIteration

        # Format the current tile in 1-meter precision, centered in 1km tiles (XX500)
        easting_str = f"{self.easting:02}500"
        northing_str = f"{self.northing:02}500"
        coord = (f"{self.current_zone}", chr(self.current_letter), self.grid_squares_list[self.current_square_index], easting_str, northing_str)

        # Increment easting and northing for 1km tiles
        self.easting += 1
        if self.easting > 99:  # Reset easting and move northing
            self.easting = 0
            self.northing += 1
            if self.northing > 99:  # Reset northing, move to the next square
                self.northing = 0
                self.current_square_index += 1
                if self.current_square_index >= len(self.grid_squares_list) or \
                   self.grid_squares_list[self.current_square_index] > self.end_square:
                    self.current_square_index = self.grid_squares_list.index(self.start_square)
                    self.current_letter += 1  # Move to the next letter
                    if self.current_letter > ord(self.end_letter):
                        self.current_letter = ord(self.start_letter)
                        self.current_zone += 1

        return coord


# Example usage:
if __name__ == "__main__":
    coord_iterator = MGRSCoordinateIterator(
        start_zone=37,
        end_zone=37,
        start_letter="T",
        end_letter="T",
        start_square="FJ",
        end_square="FJ",
    )

    with open('coordinates.txt', 'w') as file:
        for coord in coord_iterator:
            file.write(f"{coord}\n")
