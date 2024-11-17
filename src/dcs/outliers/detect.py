from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np
import os
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
from PIL import Image
import base64
import io
import time

from common import pca_dbscan, feature_file, image_paths


# Function to extract features
def extract_features(image_path):
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error loading image: {image_path}")
        return None

    # Resize image to a fixed size (e.g., 128x128)
    resized_image = cv2.resize(image, (128, 128))

    # Calculate mean and standard deviation of pixel intensities
    mean_intensity = np.mean(resized_image)
    std_intensity = np.std(resized_image)

    # Edge detection using Canny
    edges = cv2.Canny(resized_image, 100, 200)
    edge_density = np.sum(edges) / (128 * 128)  # Normalized edge density

    # Histogram of pixel intensities (normalized)
    hist = cv2.calcHist([resized_image], [0], None, [256], [0, 256])
    hist = hist / hist.sum()  # Normalize histogram

    # Flatten histogram for feature vector
    hist_flattened = hist.flatten()

    # Combine all features
    features = np.array(
        [mean_intensity, std_intensity, edge_density] + hist_flattened.tolist()
    )
    return (image_path, features)


# Load features from CSV if available, else extract and save
if os.path.exists(feature_file):
    # Load features and image paths from CSV
    df = pd.read_csv(feature_file)
    feature_list = df.iloc[:, 1:].values  # Skip first column (file paths)
    image_paths = df["file_path"].tolist()
    print(f"Loaded features from {feature_file}")
else:
    print(f"Feature file {feature_file} not found. Extracting features from images...")
    # Extract features concurrently
    feature_list = []

    with ThreadPoolExecutor() as executor:
        future_to_path = {
            executor.submit(extract_features, path): path for path in image_paths
        }

        start_time = time.time()
        for i, future in enumerate(as_completed(future_to_path), start=1):
            image_path, features = future.result()
            if features is not None:
                feature_list.append([image_path] + list(features))
        elapsed_time = time.time() - start_time
        print(f"Feature extraction finished in {elapsed_time:.2f} seconds")

    # Convert to DataFrame and save
    columns = ["file_path"] + [f"feature_{i}" for i in range(len(feature_list[0]) - 1)]
    df = pd.DataFrame(feature_list, columns=columns)
    df.to_csv(feature_file, index=False)
    feature_list = df.iloc[:, 1:].values
    print("Extracted features and saved to features.csv")

dbscan_labels, X_pca_2d, X_pca_3d = pca_dbscan(feature_list)

# Create DataFrames for plotting
df_plot_2d = pd.DataFrame(X_pca_2d, columns=["PCA1", "PCA2"])
df_plot_2d["Cluster"] = dbscan_labels
df_plot_2d["file_path"] = image_paths

df_plot_3d = pd.DataFrame(X_pca_3d, columns=["PCA1", "PCA2", "PCA3"])
df_plot_3d["Cluster"] = dbscan_labels
df_plot_3d["file_path"] = image_paths

# Initialize Dash app
app = Dash(__name__)

# Layout with both 2D and 3D graphs
app.layout = html.Div(
    [
        dcc.Graph(
            id="2d-scatter-plot",
            figure=px.scatter(
                df_plot_2d,
                x="PCA1",
                y="PCA2",
                color="Cluster",
                title="2D PCA with DBSCAN Clustering",
            ),
        ),
        dcc.Graph(
            id="3d-scatter-plot",
            figure=px.scatter_3d(
                df_plot_3d,
                x="PCA1",
                y="PCA2",
                z="PCA3",
                color="Cluster",
                title="3D PCA with DBSCAN Clustering",
            ),
        ),
        html.Div(id="hover-image", style={"textAlign": "center", "marginTop": 20}),
    ]
)


# Hover callback for both plots
@app.callback(
    Output("hover-image", "children"),
    [Input("2d-scatter-plot", "hoverData"), Input("3d-scatter-plot", "hoverData")],
)
def display_hover_image(hoverData2D, hoverData3D):
    hoverData = hoverData2D if hoverData2D else hoverData3D
    if hoverData is None:
        return html.Div("Hover over a point to see the image.")

    try:
        point_index = hoverData["points"][0].get("pointNumber")
        if point_index is None:
            return html.Div("Hover data not available.")

        # Get file path of hovered point
        image_path = df_plot_2d["file_path"].iloc[point_index]
        print(f"Hovered image: {image_path}")

        # Load and encode image for display
        image = Image.open(image_path)
        buffered = io.BytesIO()
        image.thumbnail((128, 128))
        image.save(buffered, format="PNG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Display the image and file name
        return html.Div(
            [
                html.P(image_path, style={"fontSize": "16px", "fontWeight": "bold"}),
                html.Img(
                    src=f"data:image/png;base64,{encoded_image}",
                    style={"width": "128px", "height": "128px"},
                ),
            ]
        )

    except (KeyError, IndexError) as e:
        return html.Div(f"Error retrieving hover data: {e}")


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
