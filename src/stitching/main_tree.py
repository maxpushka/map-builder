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
    Iteratively merges a list of tiles into a single tile by stitching adjacent tiles together.
    If tiles are not immediately adjacent, merging may make them adjacent in subsequent iterations.
    
    Args:
        tiles: List of Tile objects to be stitched together.
        cache_dir: Directory to cache intermediate results.

    Returns:
        A single Tile object representing the merged result.
    """
    if len(tiles) == 1:
        # If there's only one tile, return it as is.
        return tiles[0]

    merged_tiles = tiles[:]
    while len(merged_tiles) > 1:
        new_merged_tiles = []
        used_indices = set()

        for i in range(len(merged_tiles)):
            if i in used_indices:
                continue

            current_tile = merged_tiles[i]
            merged = False

            for j in range(i + 1, len(merged_tiles)):
                if j in used_indices:
                    continue

                other_tile = merged_tiles[j]
                relation = current_tile.grid.keys().__iter__().__next__().position(
                    other_tile.grid.keys().__iter__().__next__()
                )
                if relation != RelativePosition.NO_RELATION:
                    # Stitch tiles if they are adjacent
                    current_tile = stitch_tiles(current_tile, other_tile, cache_dir)
                    used_indices.add(j)
                    merged = True

            new_merged_tiles.append(current_tile)
            used_indices.add(i)

        # Check if no merges happened; this could indicate an error in positioning logic
        if len(new_merged_tiles) == len(merged_tiles):
            raise ValueError("Some tiles could not be merged. Check for gaps or invalid positioning.")

        merged_tiles = new_merged_tiles

    return merged_tiles[0]


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
