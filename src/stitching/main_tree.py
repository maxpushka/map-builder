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
    Recursively traverse the tree and reduce it to a single Tile while logging progress.
    """
    total_tasks = count_leaves(tree)
    progress = {
        "completed": 0
    }  # Use a mutable object to allow modification within nested scopes

    def walk(tree: Dict) -> Tile:
        if not isinstance(tree, dict):
            # If tree is not a dict, it's a leaf node; assume it's a list of Tile objects.
            progress["completed"] += len(tree)
            print(f"Progress: {progress['completed']}/{total_tasks} tasks completed")
            return stitch_multiple_tiles(tree, cache_dir)

        # Traverse all branches and replace their subtrees with the stitched results
        reduced_tree = {}
        for key, subtree in tree.items():
            reduced_tree[key] = walk(subtree)

        # At this level, stitch the results of all branches to produce a single Tile
        stitched_tile = stitch_multiple_tiles(list(reduced_tree.values()), cache_dir)
        return stitched_tile

    # Start walking the tree
    return walk(tree)


def count_leaves(tree: Dict) -> int:
    """
    Count the total number of leaf nodes in the tree.

    Args:
        tree: Nested dictionary representing the tree structure.

    Returns:
        Total number of leaf nodes.
    """
    if not isinstance(tree, dict):
        # If the tree is not a dict, it's a list of leaf nodes.
        return len(tree)

    total_leaves = 0
    for subtree in tree.values():
        total_leaves += count_leaves(subtree)
    return total_leaves


def stitch_multiple_tiles(tiles: List[Tile], cache_dir: str) -> Tile:
    """
    Iteratively merges a list of tiles into a single tile by stitching adjacent tiles together.
    Tiles are popped from the list and put back if they can't be merged immediately.

    Args:
        tiles: List of Tile objects to be stitched together.
        cache_dir: Directory to cache intermediate results.

    Returns:
        A single Tile object representing the merged result.
    """
    if len(tiles) == 1:
        # If there's only one tile, return it as is.
        return tiles[0]

    while len(tiles) > 1:
        i = 0
        merged = False

        while len(tiles) != 1:
            print(f"len tiles: {i, len(tiles), i < len(tiles)}")
            current_tile = tiles.pop(i)
            merge_success = False

            for j in range(len(tiles)):
                other_tile = tiles[j]
                

                # Check adjacency by comparing grids from both tiles
                for current_grid_key in current_tile.grid.keys():
                    for other_grid_key in other_tile.grid.keys():
                        print("relation: computing...")
                        relation = current_grid_key.position(other_grid_key)
                        print(f"relation: {relation}")
                        if relation in [
                            RelativePosition.ABOVE,
                            RelativePosition.BELOW,
                            RelativePosition.LEFT,
                            RelativePosition.RIGHT,
                        ]:  # Merge the tiles if they are adjacent
                            print("Stitched!")
                            stitched_tile = stitch_tiles(
                                current_tile, other_tile, cache_dir
                            )
                            tiles[j] = (
                                stitched_tile  # Replace the other_tile with the stitched_tile
                            )
                            merge_success = True
                            merged = True
                            break
                    if merge_success:
                        break  # Exit the grid comparison loop after a successful merge

                if merge_success:
                    break  # Exit the tile comparison loop after a successful merge

            if not merge_success:
                # If the current tile couldn't be merged, put it back at the end of the list
                tiles.append(current_tile)
            else:
                # Only move to the next tile if no merge occurred
                continue

        # If no merges occurred in this pass, raise an error
        if not merged:
            raise ValueError(
                "Some tiles could not be merged. Check for gaps or invalid positioning."
            )

    # Return the final merged tile
    return tiles[0]


# Example usage
if __name__ == "__main__":
    ROOT_DIR = "."
    
    CACHE_DIR = f"{ROOT_DIR}/cache"
    os.makedirs(CACHE_DIR, exist_ok=True)

    DATA_DIR = f"{ROOT_DIR}/images"
    tree = build_tree(DATA_DIR)
    print("Organized Tree:")
    print_tree(tree)
    output = walk_tree(tree, stitch_multiple_tiles, CACHE_DIR)
    cv2.imwrite("output_tree.png", output.image())
