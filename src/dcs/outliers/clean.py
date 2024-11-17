import os
import shutil
import pandas as pd

from common import pca_dbscan, feature_file

# Heuristics for outliers.
# Each line is defined by two points (x1, y1) and (x2, y2).
LINE1 = ((-11, 20), (-7, 0))  # Line 1
LINE2 = ((-6.491185, -0.5775821), (10.09225, -2.941133))  # Line 2


# Function to calculate if a point (x, y) is below a line defined by two points
def is_below_line(point, line):
    (x1, y1), (x2, y2) = line
    px, py = point
    # Calculate slope (m) and intercept (b) of the line
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    # Check if point is below the line
    return py < (m * px + b)


if __name__ == "__main__":
    # Load features and image paths from CSV
    df = pd.read_csv(feature_file)
    feature_list = df.iloc[:, 1:].values  # Skip first column (file paths)
    image_paths = df["file_path"].tolist()

    # Load PCA results from feature list
    _, X_pca_2d, _ = pca_dbscan(feature_list)

    # Identify files to move based on whether their PCA projections are below both lines
    files_to_move = []
    for i, row in enumerate(X_pca_2d):
        # Treat PCA results tuple as a point in 2D space
        # and check if they are below both lines
        if is_below_line(row, LINE1) and is_below_line(row, LINE2):
            files_to_move.append(image_paths[i])

    # Move files that meet the criteria to the drop directory
    for file_path in files_to_move:
        try:
            # Destination directory is located in the same directory as the feature file
            # Example: ./alt/image.png -> ./alt/drop/image.png
            destination_dir = os.path.normpath(
                os.path.join(os.path.dirname(file_path) + "/", "drop/")
            )
            destination = os.path.normpath(
                os.path.join(destination_dir, os.path.basename(file_path))
            )
            os.makedirs(destination_dir, exist_ok=True)

            print(f"Moving: {file_path} -> {destination}")
            shutil.move(file_path, destination)
            print(f"Moved: {file_path} -> {destination}")
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error moving {file_path}: {e}")
    print(f"File move process complete. Moved {len(files_to_move)} files")
