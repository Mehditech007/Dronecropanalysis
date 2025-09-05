"""
Real‑Time Crop Health Analysis with DJI Tello
============================================

This module provides a simple example of how to analyse drone imagery for
vegetation health using a standard RGB camera.  It computes a
colour‑based vegetation index on each frame and labels the scene as
``healthy`` or ``unhealthy`` based on the proportion of green pixels.
Although professional NDVI requires near‑infrared sensors, simple
indices derived from RGB values can still provide useful insights.  The
code is intended to run either on a synthetic test image or in real
time using the DJI Tello drone via the `djitellopy` library.

The underlying concept of vegetation indices is well described in
precision‑agriculture literature.  NDVI, for example, measures the
difference between near‑infrared (NIR) and red reflectance, scaled to
−1..1: NDVI = (NIR − Red)/(NIR + Red).  Healthy vegetation absorbs
visible light for photosynthesis and reflects much of the NIR
spectrum【255148486829847†L65-L91】.  Values above 0.5 typically indicate healthy
plants, while lower scores suggest stress or sparse vegetation【255148486829847†L97-L110】.

Even without NIR, RGB cameras can approximate plant greenness by
computing indices such as the Excess Green Index (ExG) or simply the
ratio of green channel intensity.  This example uses a simple greenness
ratio to demonstrate the concept.

To run the real‑time analysis you will need a DJI Tello drone and
Python package `djitellopy`.  The package wraps the official Tello SDK
and supports connecting to the drone, retrieving a video stream and
controlling movement【381994597102161†L97-L105】.  Install it via
``pip install djitellopy``.  Note that video streaming is available on
the consumer Tello only when connected directly to its Wi‑Fi network.

Usage
-----

There are two primary entry points:

* ``analyse_image(image_path: str) -> None`` – analyse a static image and
  display the results.
* ``analyse_stream() -> None`` – connect to a Tello drone, stream
  frames and overlay health predictions in real time.

Example:

```bash
python drone_crop_analysis.py --image synthetic_field.png
```

or, to run live on the drone:

```bash
python drone_crop_analysis.py --realtime
```

References
----------
* The ``djitellopy`` project provides a Python interface to the DJI Tello
  drone using the official Tello SDK and supports video streaming【381994597102161†L97-L105】.
* The Farmonaut blog explains NDVI and how high values (0.5–0.7) indicate
  healthy vegetation, whereas low values suggest stress【255148486829847†L65-L110】.
"""

import argparse
from dataclasses import dataclass
from typing import Tuple
import cv2
import numpy as np

try:
    from djitellopy import Tello  # type: ignore
    _DJITELLOPY_AVAILABLE = True
except ImportError:
    # If djitellopy is not installed, set flag to False.  The realtime mode will not work.
    _DJITELLOPY_AVAILABLE = False


@dataclass
class AnalysisResult:
    classification: str
    vegetation_ratio: float
    mask: np.ndarray


def compute_vegetation_ratio(frame: np.ndarray, threshold: float = 0.4) -> AnalysisResult:
    """Compute a simple vegetation ratio and classify frame health.

    The function converts the image to RGB, computes the greenness ratio for
    each pixel (green channel divided by the sum of all channels) and
    thresholds this ratio to produce a binary vegetation mask.  The overall
    vegetation ratio is the proportion of pixels deemed vegetated.  A simple
    classification of ``healthy`` or ``unhealthy`` is returned based on
    whether the ratio exceeds the supplied threshold.

    Parameters
    ----------
    frame : numpy.ndarray
        Image in BGR colour space as returned by OpenCV.
    threshold : float, optional
        Threshold on the vegetation ratio to decide health status.

    Returns
    -------
    AnalysisResult
        Dataclass containing the classification label, the computed
        vegetation ratio and the binary mask used to compute the ratio.
    """
    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
    # Avoid division by zero
    sum_channels = np.maximum(rgb.sum(axis=2), 1e-6)
    green_ratio = rgb[:, :, 1] / sum_channels

    # Simple threshold to identify vegetation pixels
    vegetation_mask = (green_ratio > 0.4).astype(np.uint8)

    vegetation_ratio = vegetation_mask.mean()
    classification = 'healthy' if vegetation_ratio >= threshold else 'unhealthy'

    return AnalysisResult(classification, float(vegetation_ratio), vegetation_mask)


def overlay_analysis(frame: np.ndarray, result: AnalysisResult) -> np.ndarray:
    """Overlay the analysis mask and label onto the frame for visualisation.

    Parameters
    ----------
    frame : numpy.ndarray
        Original BGR frame.
    result : AnalysisResult
        Result returned by ``compute_vegetation_ratio``.

    Returns
    -------
    numpy.ndarray
        Frame with overlay drawn.
    """
    overlay = frame.copy()
    # Create a coloured mask: green for vegetation, red for non‑vegetation
    mask_colored = np.zeros_like(frame)
    mask_colored[result.mask == 1] = (0, 255, 0)  # green in BGR
    mask_colored[result.mask == 0] = (0, 0, 255)  # red

    # Blend the mask with the original frame
    alpha = 0.4
    cv2.addWeighted(mask_colored, alpha, overlay, 1 - alpha, 0, overlay)

    # Write classification text
    text = f"{result.classification.upper()} ({result.vegetation_ratio*100:.1f}% vegetation)"
    cv2.putText(
        overlay,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )
    return overlay


def analyse_image(image_path: str) -> None:
    """Analyse a static image for vegetation health and display the results.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    """
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    result = compute_vegetation_ratio(frame)
    overlay = overlay_analysis(frame, result)
    # Convert BGR to RGB for display in notebook or OpenCV window
    cv2.imwrite('analysis_result.png', overlay)
    # Display via cv2 window if available
    cv2.imshow('Crop Health Analysis', overlay)
    print(f"Classification: {result.classification}, vegetation ratio: {result.vegetation_ratio:.4f}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def analyse_stream() -> None:
    """Connect to a DJI Tello drone and analyse the video stream in real time.

    Requires the ``djitellopy`` package and an available Tello drone.  Frames
    are captured from the drone, analysed and displayed until the user
    presses the ``q`` key.
    """
    if not _DJITELLOPY_AVAILABLE:
        raise ImportError("djitellopy is required for real‑time streaming")
    tello = Tello()
    tello.connect()
    print(f"Battery: {tello.get_battery()}%")
    tello.streamon()
    frame_reader = tello.get_frame_read()

    try:
        while True:
            frame = frame_reader.frame
            result = compute_vegetation_ratio(frame)
            overlay = overlay_analysis(frame, result)
            cv2.imshow('Drone Crop Health', overlay)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        tello.streamoff()
        tello.end()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Crop Health Analysis via Tello')
    parser.add_argument('--image', type=str, default=None, help='Path to a static image to analyse')
    parser.add_argument('--realtime', action='store_true', help='Analyse live video stream from Tello')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.realtime:
        analyse_stream()
    elif args.image is not None:
        analyse_image(args.image)
    else:
        print("Please specify either --image <path> or --realtime")


if __name__ == '__main__':
    main()