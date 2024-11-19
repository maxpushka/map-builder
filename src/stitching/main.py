from pyspark import SparkContext
from pyspark.sql import SparkSession
from PIL import Image
import cv2
import re

# Define the Spark context and session
sc = SparkContext("local", "ImageStitchingPipeline")
spark = SparkSession(sc)


class StitchingError(Exception):
    def __init__(self, status):
        super().__init__(f"Can't stitch images, error code = {status}")
        self.status = status


def stitch_images(images):
    # Stitching
    stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS)
    status, pano = stitcher.stitch(images)

    if status != cv2.Stitcher_OK:
        raise StitchingError(status)

    return pano


# Parse MGRS coordinates from filename
def parse_coordinates(filename):
    match = re.match(r"(\d+_\w)_(\w+)_(\d{2})(\d{3})_(\d{2})(\d{3})", filename)
    if match:
        zone = match.group(1)
        square = match.group(2)
        easting = match.group(3)
        northing = match.group(5)
        return zone, square, int(easting), int(northing)
    return None


# Load images and parse their MGRS coordinates
def load_and_parse_images(path):
    images_rdd = sc.binaryFiles(path).map(lambda x: (x[0].split("/")[-1], x[1]))
    images_rdd = images_rdd.map(parse_image).filter(lambda x: x is not None)
    return images_rdd


def parse_image(record):
    filename, image_data = record
    coords = parse_coordinates(filename)
    if not coords:
        return None
    zone, square, easting, northing = coords
    # Crop image around the bounding box
    cropped_image_data = crop_image(image_data)
    return (zone, square, easting, northing), cropped_image_data


# Stage 1: Merge 1km blocks into 10km blocks
def merge_1km_to_10km(images_rdd):
    km_10_blocks = images_rdd.map(
        lambda x: ((x[0][0], x[0][1], x[0][2] // 10, x[0][3] // 10), x[1])
    ).groupByKey()
    km_10_merged = km_10_blocks.mapValues(lambda images: stitch_images(list(images)))
    return km_10_merged


# Stage 2: Merge 10km blocks into 100km blocks
def merge_10km_to_100km(km_10_merged):
    km_100_blocks = km_10_merged.map(lambda x: ((x[0][0], x[0][1]), x[1])).groupByKey()
    km_100_merged = km_100_blocks.mapValues(lambda images: stitch_images(list(images)))
    return km_100_merged


# Stage 3: Merge 100km blocks into zone-level map
def merge_100km_to_zone(km_100_merged):
    zone_blocks = km_100_merged.map(lambda x: (x[0][0], x[1])).groupByKey()
    zone_merged = zone_blocks.mapValues(lambda images: stitch_images(list(images)))
    return zone_merged


# Final assembly if multiple zones need merging
def final_assembly(zone_merged):
    final_map = stitch_images(zone_merged.values().collect())
    return final_map


# Main function to orchestrate the pipeline
def main():
    # Step 1: Load and parse images
    images_rdd = load_and_parse_images("s3://raw/")

    # Step 2: Merge 1km blocks into 10km blocks
    km_10_merged = merge_1km_to_10km(images_rdd)

    # Step 3: Merge 10km blocks into 100km blocks
    km_100_merged = merge_10km_to_100km(km_10_merged)

    # Step 4: Merge 100km blocks into zone-level maps
    zone_merged = merge_100km_to_zone(km_100_merged)

    # Step 5: Final assembly of all zones
    final_map = final_assembly(zone_merged)

    # Save the final map
    final_map.save("s3://result/final_map.png")


# Run the main function
if __name__ == "__main__":
    main()
