# Map Builder

## Overview

This project implements a comprehensive system for building large-scale geographic maps through automated data collection and advanced image stitching algorithms. The system consists of two main components:

1. **Data Collection System (DCS)**: Automated screenshot capture from Digital Combat Simulator for geographic map tiles
2. **Image Stitching Pipeline**: Advanced algorithms for seamlessly combining map tiles into coherent large-scale maps

## Features

### Data Collection (`src/dcs/`)
- **Automated Screenshot Capture**: Programmatic interface with Digital Combat Simulator for systematic map tile collection
- **MGRS Coordinate System**: Military Grid Reference System integration for precise geographic positioning
- **Outlier Detection**: Statistical analysis to identify and filter corrupted or invalid map tiles
- **Coordinate Iteration**: Intelligent traversal of geographic coordinate spaces with sea coordinate filtering

### Image Stitching (`src/stitching/`)
- **Hierarchical Tree Structure**: Efficient organization of map tiles using multi-level geographic indexing
- **Advanced Tile Merging**: Sophisticated algorithms for seamless image blending with alpha channel support
- **Memory-Efficient Processing**: Optimized handling of large image datasets with caching mechanisms
- **Spark Integration**: Distributed processing capabilities for large-scale map generation

## Technical Implementation

### Key Algorithms
- **Tile Positioning**: Precise placement of map tiles using geographic coordinate calculations
- **Image Blending**: Alpha-channel based merging with edge detection and seamless transitions
- **Hierarchical Indexing**: Multi-level geographic organization (Zone → Square → Block → Quarter → Tile)
- **Memory Management**: Efficient caching system for processed tiles

### Technologies Used
- **Python**: Core implementation language
- **OpenCV**: Computer vision and image processing
- **PySpark**: Distributed computing for large datasets
- **NumPy**: Numerical computations and array operations
- **PIL/Pillow**: Image manipulation and format handling
- **MGRS**: Military Grid Reference System coordinate handling

## Project Structure

```
src/
├── dcs/                   # Data collection system
│   ├── dataset/           # Screenshot automation and coordinate management
│   └── outliers/          # Data quality analysis and filtering
└── stitching/             # Image processing and map assembly
    ├── cache/             # Processed tile cache
    ├── images/            # Sample output images
    └── raw/               # Raw input map tiles
```

## Research Applications

This project contributes to several areas of computer science research:

- **Computer Vision**: Advanced image stitching and blending techniques
- **Geographic Information Systems**: Automated map generation and coordinate system integration
- **Distributed Computing**: Scalable processing of large geographic datasets
- **Data Quality Assurance**: Automated outlier detection in visual datasets

## Academic Context

This project demonstrates practical applications of:
- Algorithm design for geographic data processing
- Distributed system architecture for large-scale image processing
- Computer vision techniques for seamless image composition
- Automated data collection and quality assurance methodologies

## Usage

The system is designed for research purposes and requires:
- Digital Combat Simulator for data collection
- Python environment with specified dependencies
- Sufficient storage for large image datasets

For detailed implementation examples, refer to the main processing scripts in each module.
