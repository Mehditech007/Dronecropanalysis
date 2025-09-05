# Drone Crop Health Analysis Project

This project demonstrates a simple **real‑time crop health analysis** pipeline
using a DJI Tello drone.  It includes Python scripts to analyse static images
or stream frames from the Tello and estimate vegetation health using a
greenness index.  A companion script generates synthetic aerial images so
that the analysis can be tested without access to a real drone.

## Background

In precision agriculture, vegetation indices like the **Normalized Difference
Vegetation Index (NDVI)** are widely used to assess plant health.  NDVI
measures the difference between near‑infrared (NIR) and red reflectance
and ranges from −1 to 1.  Healthy plants absorb visible light for
photosynthesis and reflect most of the NIR spectrum【255148486829847†L65-L91】.
High NDVI values (0.5–1.0) indicate vigorous vegetation while low values
suggest stress or sparse cover【255148486829847†L97-L110】.  Consumer drones with
RGB cameras cannot measure NIR directly, but simple indices derived
from colour channels can still provide useful proxies for plant
greenness.  This project uses a **greenness ratio** (green channel divided by
the sum of all channels) to approximate vegetation cover.

## Project Contents

| File | Description |
| --- | --- |
| **`drone_crop_analysis.py`** | Main module that analyses images or a live video stream to estimate crop health.  It connects to a DJI Tello drone via the [`djitellopy` library](https://github.com/damiafuentes/DJITelloPy), retrieves frames and overlays a colour mask to highlight healthy (green) and unhealthy (brown/red) regions.  The script can also analyse static images. |
| **`generate_synthetic_image.py`** | Utility to generate synthetic aerial images with alternating rows of healthy and stressed crops.  Use this to test the analysis without a drone. |
| **`requirements.txt`** | Python dependencies for the project. |

## Prerequisites

* Python 3.7 or higher
* [`opencv-python`](https://pypi.org/project/opencv-python/) for image processing
* [`numpy`](https://pypi.org/project/numpy/) for numerical computations
* [`djitellopy`](https://pypi.org/project/djitellopy/) for communicating with the Tello drone (optional if you only analyse static images)

To install the dependencies, run:

```bash
pip install -r requirements.txt
```

Note: you only need `djitellopy` if you intend to run the real‑time
streaming mode with a Tello drone.  The package wraps the official
DJI Tello SDK and implements all Tello commands, including retrieving
video streams and state packets【381994597102161†L97-L105】.

## Generating a Synthetic Test Image

If you do not have access to a drone or real aerial imagery, you can
create a synthetic test image:

```bash
python generate_synthetic_image.py --output synthetic_field.png --width 640 --height 480
```

This script will produce a PNG file with alternating rows of green and
brown colours representing healthy and stressed crops.  Random noise is
added to make the image more realistic.  You can adjust the width,
height and number of rows via command‑line options.

## Analysing a Static Image

Use the `--image` flag to analyse any saved aerial image:

```bash
python drone_crop_analysis.py --image synthetic_field.png
```

The script will load the image, compute the greenness ratio for each
pixel, classify the overall scene as **healthy** or **unhealthy** based on
the proportion of vegetation, and display the result with a colour
overlay.  Green pixels represent vegetated areas, whereas red pixels
indicate non‑vegetated or stressed regions.

## Analysing the Drone Video Stream

To run the analysis live on a DJI Tello drone, first connect your
computer to the drone’s Wi‑Fi network.  Then execute:

```bash
python drone_crop_analysis.py --realtime
```

The script will connect to the Tello using `djitellopy`, start
streaming frames, apply the greenness ratio on each frame and display
the overlay in real time.  Press `q` to exit the stream.  Ensure you
have a clear view of the crops and adequate lighting for reliable
analysis.

## Further Improvements

* **NDVI Sensors**: For accurate vegetation health assessment, drones with
  NIR sensors or modified cameras should be used so that genuine NDVI can
  be computed.  The simple greenness index implemented here is only a
  proxy and may misclassify certain scenes.
* **Model Calibration**: The threshold for classifying healthy vs.
  unhealthy crops is currently fixed (0.4).  Collecting labelled data
  from your fields and calibrating the threshold or training a
  supervised model would improve accuracy.
* **Mapping and Georeferencing**: Integrating GPS data and stitching
  frames into orthomosaics would enable mapping vegetation patterns
  across entire fields.

## Citation

This project references open information about vegetation indices and
the DJI Tello SDK.  NDVI measures the difference between near‑infrared
and red reflectance and high values indicate healthy vegetation【255148486829847†L65-L91】【255148486829847†L97-L110】.  The
`djitellopy` library wraps the official Tello SDK, enabling Python
control of the drone and access to its video stream【381994597102161†L97-L105】.