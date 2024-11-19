import mgrs

from tile import RelativePosition


def position(mgrs_a: str, mgrs_b: str) -> RelativePosition:
    m = mgrs.MGRS()

    # Convert both MGRS strings to UTM coordinates
    zone_a, hemisphere_a, easting_a, northing_a = m.MGRSToUTM(mgrs_a)
    zone_b, hemisphere_b, easting_b, northing_b = m.MGRSToUTM(mgrs_b)

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

    return direction_map.get((easting_key, northing_key), RelativePosition.NO_RELATION)


def nearest(mgrs_coord: str, dir: RelativePosition, step_size=1000) -> str:
    m = mgrs.MGRS()

    # Convert MGRS coordinate to UTM coordinates
    zone, hemisphere, easting, northing = m.MGRSToUTM(mgrs_coord)

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
    return m.UTMToMGRS(zone, hemisphere, new_easting, new_northing)
