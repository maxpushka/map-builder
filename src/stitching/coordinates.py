from enum import Enum
import re
import mgrs


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


class MGRSCoordinate:
    def __init__(self, zone: str, square: str, easting: int, northing: int):
        self.zone = zone  # e.g., "37T"
        self.square = square  # e.g., "FJ"
        self.easting = easting  # in meters, e.g., 42500
        self.northing = northing  # in meters, e.g., 17500

    @classmethod
    def from_filename(cls, filename: str):
        filename = filename.replace(".png", "").replace("_", "")
        match = re.match(r"(\d+\w)(\w+)(\d{5})(\d{5})", filename)
        if not match:
            return None
        zone = match.group(1)
        square = match.group(2)
        easting = int(match.group(3))
        northing = int(match.group(4))
        return cls(zone, square, easting, northing)

    def truncate(self, granularity: int):
        """Truncate easting and northing to a specific granularity."""
        easting = (self.easting // granularity) * granularity
        northing = (self.northing // granularity) * granularity
        return MGRSCoordinate(self.zone, self.square, easting, northing)

    def to_tuple(self, include_granularity=False):
        """Convert to a tuple for grouping operations."""
        if include_granularity:
            return (self.zone, self.square, self.easting, self.northing)
        return (self.zone, self.square)

    def __str__(self):
        """String representation for debugging or grouping."""
        return f"{self.zone}{self.square}{self.easting:05}{self.northing:05}"

    def __eq__(self, other):
        if not isinstance(other, MGRSCoordinate):
            return False
        return (
            self.zone == other.zone
            and self.square == other.square
            and self.easting == other.easting
            and self.northing == other.northing
        )

    def __hash__(self):
        return hash((self.zone, self.square, self.easting, self.northing))

    def position(self, other: "MGRSCoordinate") -> RelativePosition:
        m = mgrs.MGRS()

        # Convert both MGRS strings to UTM coordinates
        zone_a, hemisphere_a, easting_a, northing_a = m.MGRSToUTM(self.__str__())
        zone_b, hemisphere_b, easting_b, northing_b = m.MGRSToUTM(str(other))

        # If zones or hemispheres are different, they have no direct relation in our comparison
        if zone_a != zone_b or hemisphere_a != hemisphere_b:
            return RelativePosition.NO_RELATION

        # Calculate relative positions by comparing easting and northing values
        easting_diff = easting_b - easting_a
        northing_diff = northing_b - northing_a

        direction_map = {
            (0, 0): RelativePosition.CENTER,
            (0, 1): RelativePosition.ABOVE,
            (0, -1): RelativePosition.BELOW,
            (1, 0): RelativePosition.RIGHT,
            (-1, 0): RelativePosition.LEFT,
            (1, 1): RelativePosition.ABOVE_RIGHT,
            (1, -1): RelativePosition.BELOW_RIGHT,
            (-1, 1): RelativePosition.ABOVE_LEFT,
            (-1, -1): RelativePosition.BELOW_LEFT,
        }

        # Normalize diffs to -1, 0, or 1 to simplify matching
        easting_key = 1 if easting_diff > 0 else -1 if easting_diff < 0 else 0
        northing_key = 1 if northing_diff > 0 else -1 if northing_diff < 0 else 0

        return direction_map.get(
            (easting_key, northing_key), RelativePosition.NO_RELATION
        )

    def nearest(self, dir: RelativePosition, step_size=1000) -> "MGRSCoordinate":
        m = mgrs.MGRS()

        # Convert MGRS coordinate to UTM coordinates
        zone, hemisphere, easting, northing = m.MGRSToUTM(str(self))

        # Define movement based on direction
        movements = {
            RelativePosition.ABOVE: (0, step_size),
            RelativePosition.BELOW: (0, -step_size),
            RelativePosition.LEFT: (-step_size, 0),
            RelativePosition.RIGHT: (step_size, 0),
            RelativePosition.ABOVE_LEFT: (-step_size, step_size),
            RelativePosition.ABOVE_RIGHT: (step_size, step_size),
            RelativePosition.BELOW_LEFT: (-step_size, -step_size),
            RelativePosition.BELOW_RIGHT: (step_size, -step_size),
            RelativePosition.CENTER: (0, 0),
        }

        # Get the movement deltas for the specified direction
        delta_easting, delta_northing = movements.get(dir, (0, 0))

        # Apply the deltas to get the new UTM coordinates
        new_easting = easting + delta_easting
        new_northing = northing + delta_northing

        # Convert back to MGRS and return
        return MGRSCoordinate.from_filename(
            m.UTMToMGRS(zone, hemisphere, new_easting, new_northing)
        )


if __name__ == "__main__":
    a = MGRSCoordinate("37T", "FJ", 15000, 5000)
    b = MGRSCoordinate("37T", "FJ", 15000, 5000)
    d = {a: 12}
    print(b in d)
