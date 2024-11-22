import os
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2

from coordinates import MGRSCoordinate, RelativePosition
from image import display_result, stitch_tiles
from tile import Tile


def build_tree(data_dir: str) -> Dict:
    filenames: List[str] = os.listdir(data_dir)

    tree: defaultdict[
        str,
        defaultdict[
            str,
            defaultdict[
                Tuple[bool, bool],
                defaultdict[
                    Tuple[int, int],
                    defaultdict[
                        Tuple[bool, bool], defaultdict[Tuple[bool, bool], List[str]]
                    ],
                ],
            ],
        ],
    ] = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
            )
        )
    )

    quarter_1km_threshold = {
        (0, 0): (2, 2),
        (0, 1): (2, 7),
        (1, 0): (7, 2),
        (1, 1): (7, 7),
    }
    for filename in filenames:
        coord = MGRSCoordinate.from_filename(filename)
        # Group by zone -> square (100km block) -> 10km quarter -> 10km block -> 5km quarter -> 1km quarter -> (easting, northing)
        block_10km = coord.easting // 10000, coord.northing // 10000
        quarter_10km = block_10km[0] >= 5, block_10km[1] >= 5

        block_5km = coord.easting // 1000 % 10, coord.northing // 1000 % 10
        quarter_5km = block_5km[0] >= 5, block_5km[1] >= 5

        b = quarter_1km_threshold[quarter_5km]
        quarter_1km = block_5km[0] >= b[0], block_5km[1] >= b[1]

        tile = Tile(os.path.join(data_dir, filename), coord)
        tree[coord.zone][coord.square][quarter_10km][block_10km][quarter_5km][
            quarter_1km
        ].append(tile)
    return tree


def print_tree(tree: Dict, level=0):
    indent = "  " * level
    for key, value in tree.items():
        print(f"{indent}{key}:")
        if isinstance(value, dict):
            print_tree(value, level + 1)
        else:
            print(f"{indent}  {value}")


def walk_tree(tree: Dict, stitch_multiple_tiles, cache_dir: str) -> Tile:
    """
    Recursively traverse the tree and reduce it to a single Tile by applying stitch_multiple_tiles
    on the leaf nodes and propagating upward.
    """
    if not isinstance(tree, dict):
        # If tree is not a dict, it's a leaf node; assume it's a list of Tile objects.
        return stitch_multiple_tiles(tree, cache_dir)

    # Traverse all branches and replace their subtrees with the stitched results
    reduced_tree = {}
    for key, subtree in tree.items():
        reduced_tree[key] = walk_tree(subtree, stitch_multiple_tiles, cache_dir)

    # At this level, stitch the results of all branches to produce a single Tile
    stitched_tile = stitch_multiple_tiles(list(reduced_tree.values()), cache_dir)
    return stitched_tile


def stitch_multiple_tiles(tiles: List[Tile], cache_dir: str) -> Tile:
    """
    Merges a list of tiles into a single tile, ensuring that indirectly adjacent tiles
    are merged correctly after iterations.

    :param tiles: A list of Tile objects to be stitched together.
    :param cache_dir: Directory to store intermediate results.
    :return: A single Tile object resulting from stitching all input tiles.
    """
    if len(tiles) == 1:
        return tiles[0]  # If only one tile, return it as is.

    # Helper function to check adjacency and stitch two tiles
    def find_and_merge(tiles: List[Tile]) -> List[Tile]:
        merged = []
        used_indices = set()

        for i, tile_a in enumerate(tiles):
            if i in used_indices:
                continue
            for j, tile_b in enumerate(tiles):
                if i != j and j not in used_indices:
                    position = (
                        tile_a.grid.keys()
                        .__iter__()
                        .__next__()
                        .position(tile_b.grid.keys().__iter__().__next__())
                    )
                    if position not in [
                        RelativePosition.ABOVE,
                        RelativePosition.BELOW,
                        RelativePosition.LEFT,
                        RelativePosition.RIGHT,
                        RelativePosition.CENTER,
                    ]:
                        continue
                    merged_tile = stitch_tiles(tile_a, tile_b, cache_dir)
                    # display_result(merged_tile)
                    used_indices.update([i, j])
                    merged.append(merged_tile)
                    break
            else:
                # If not merged, keep the tile for the next iteration
                if i not in used_indices:
                    merged.append(tile_a)
        return merged

    # Iteratively merge tiles until only one tile remains
    current_tiles = tiles
    while len(current_tiles) > 1:
        current_tiles = find_and_merge(current_tiles)

    return current_tiles[0]


# Example usage
if __name__ == "__main__":
    CACHE_DIR = "./cache"
    os.makedirs(CACHE_DIR, exist_ok=True)
    DATA_DIR = "./images"

    tree = build_tree(DATA_DIR)
    print("Organized Tree:")
    print_tree(tree)
    output = walk_tree(tree, stitch_multiple_tiles, CACHE_DIR)
    cv2.imwrite("output_tree.png", output.image())
