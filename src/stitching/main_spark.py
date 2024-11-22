import os
from typing import Iterable, Union
import cv2
import numpy as np
from pyspark import RDD
from pyspark.sql import SparkSession

from coordinates import MGRSCoordinate, RelativePosition
from tile import Tile
from image import stitch_tiles as stitch

# Define the Spark context and session
spark = (
    SparkSession.builder.appName("MapBuilder")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config(
        "spark.sql.catalog.spark_catalog",
        "org.apache.spark.sql.delta.catalog.DeltaCatalog",
    )
    .config(
        "spark.jars.packages",
        "io.delta:delta-core_2.12:1.1.0,"
        "org.apache.hadoop:hadoop-aws:3.2.2,"
        "com.amazonaws:aws-java-sdk-bundle:1.12.180",
    )
    .config(
        "spark.hadoop.fs.s3a.aws.credentials.provider",
        "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
    )
    .getOrCreate()
)

# Configure Spark to use AWS S3
spark._jsc.hadoopConfiguration().set(
    "fs.s3.impl", "org.apache.hadoop.fs.s3.NativeS3FileSystem"
)
spark._jsc.hadoopConfiguration().set(
    "fs.s3.aws.credentials.provider",
    "com.amazonaws.auth.InstanceProfileCredentialsProvider,com.amazonaws.auth.DefaultAWSCredentialsProviderChain",
)
spark._jsc.hadoopConfiguration().set("com.amazonaws.services.s3.enableV4", "true")

spark._jsc.hadoopConfiguration().set(
    "fs.s3a.access.key", os.environ["AWS_ACCESS_KEY_ID"]
)
spark._jsc.hadoopConfiguration().set(
    "fs.s3a.secret.key", os.environ["AWS_SECRET_ACCESS_KEY"]
)


def load_and_parse_tiles(path) -> RDD[Tile]:
    tiles_rdd = spark.sparkContext.binaryFiles(path).map(
        lambda x: (x[0].split("/")[-1], x[1])
    )
    tiles_rdd = tiles_rdd.map(parse_tile).filter(lambda x: x is not None)
    return tiles_rdd


def parse_tile(record: tuple[str, np.ndarray]) -> Union[Tile, None]:
    filename, image_binary = record
    # Decode the binary data into an OpenCV Mat (image)
    image = cv2.imdecode(np.frombuffer(image_binary, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Failed to decode image from file: {filename}")
        return None

    # Extract coordinates from the filename
    coords = MGRSCoordinate.from_filename(filename)
    if not coords:
        print(f"Invalid coordinates for file: {filename}")
        return None

    return Tile(image, coords)


def are_adjacent(tile_a: Tile, tile_b: Tile) -> bool:
    """Check if two tiles are spatially adjacent."""
    for coord_a in tile_a.grid.keys():
        for coord_b in tile_b.grid.keys():
            if coord_a.position(coord_b) != RelativePosition.NO_RELATION:
                return True
    return False


def merge_adjacent_tiles(tiles: Iterable[Tile]) -> Union[Tile, None]:
    """Merge tiles that are part of a continuous block, considering all possible grid coordinates."""
    if not tiles:
        return None

    # Flatten all coordinates from all tiles to create a comprehensive sort order
    sorted_tiles = sorted(
        tiles,
        key=lambda t: sorted(
            (coord.zone, coord.square, coord.easting, coord.northing)
            for coord in t.grid.keys()
        ),
    )

    # Iteratively merge tiles until no further merging is possible
    merged_tiles = sorted_tiles
    while len(merged_tiles) > 1:
        new_merged_tiles = []
        skip_indices = set()

        for i, tile in enumerate(merged_tiles):
            if i in skip_indices:
                continue

            merged = tile
            for j, other_tile in enumerate(merged_tiles):
                if (
                    i != j
                    and j not in skip_indices
                    and any(
                        coord_a.position(coord_b) != RelativePosition.NO_RELATION
                        for coord_a in merged.grid.keys()
                        for coord_b in other_tile.grid.keys()
                    )
                ):
                    # Merge the adjacent tile
                    merged = stitch(merged, other_tile)
                    skip_indices.add(j)

            new_merged_tiles.append(merged)

        # If no further merging occurred, break out of the loop
        if len(new_merged_tiles) == len(merged_tiles):
            break

        merged_tiles = new_merged_tiles

    # After merging, there should be only one tile left if all are connected
    return merged_tiles[0] if merged_tiles else None


# Update merging stages to use adjacency-aware merging
def merge_1km_to_10km(tiles_rdd: RDD[Tile]) -> RDD[Tile]:
    def extract_10km_key(tile: Tile):
        # Extract the first (and only) coordinate from the grid dictionary
        coord = next(iter(tile.grid.keys()))
        truncated = coord.truncate(10_000)
        return truncated.to_tuple(include_granularity=True)

    km_10_blocks = tiles_rdd.map(
        lambda tile: (extract_10km_key(tile), tile)
    ).groupByKey()

    km_10_merged = km_10_blocks.mapValues(
        lambda tiles: merge_adjacent_tiles(tiles)
    ).filter(lambda x: x[1] is not None)

    return km_10_merged


def merge_10km_to_100km(km_10_merged: RDD[Tile]) -> RDD[Tile]:
    def extract_100km_key(key):
        coord = MGRSCoordinate(key[0], key[1], key[2], key[3])
        truncated = coord.truncate(100_000)
        return truncated.to_tuple(include_granularity=True)

    km_100_blocks = km_10_merged.map(
        lambda x: (extract_100km_key(x[0]), x[1])
    ).groupByKey()

    km_100_merged = km_100_blocks.mapValues(
        lambda tiles: merge_adjacent_tiles(tiles)
    ).filter(lambda x: x[1] is not None)

    return km_100_merged


def merge_100km_to_zone(km_100_merged: RDD[Tile]) -> RDD[Tile]:
    def extract_zone_key(key):
        # Reconstruct MGRSCoordinate from the key tuple
        coord = MGRSCoordinate(key[0], key[1], key[2], key[3])
        return coord.to_tuple()

    zone_blocks = km_100_merged.map(
        lambda x: (extract_zone_key(x[0]), x[1])
    ).groupByKey()

    zone_merged = zone_blocks.mapValues(
        lambda tiles: merge_adjacent_tiles(tiles)
    ).filter(lambda x: x[1] is not None)

    return zone_merged


def merge_all_zones(zone_merged: RDD[Tile]) -> Union[Tile, None]:
    """Merge all zone-level maps into a single final map, ensuring adjacency."""
    zone_tiles = zone_merged.values().collect()

    # Ensure merging only adjacent zones
    final_map = merge_adjacent_tiles(zone_tiles)
    return final_map


# Main function to orchestrate the pipeline
def main():
    # Step 1: Load and parse tiles
    tiles_rdd = load_and_parse_tiles("s3a://dcs-map-builder-20241117/raw/")

    # Step 2: Merge 1km blocks into 10km blocks
    km_10_merged = merge_1km_to_10km(tiles_rdd)

    # Step 3: Merge 10km blocks into 100km blocks
    km_100_merged = merge_10km_to_100km(km_10_merged)

    # Step 4: Merge 100km blocks into zone-level maps
    zone_merged = merge_100km_to_zone(km_100_merged)

    # Step 5: Final assembly of all zones
    final_map = merge_all_zones(zone_merged)

    # Save the final map
    final_map.save("s3a://dcs-map-builder-20241117/output/output.png")


# Run the main function
if __name__ == "__main__":
    main()
