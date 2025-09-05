"""
Synthetic Crop Field Generator
=============================

This utility script produces a synthetic aerial image of a crop field.
The generated image consists of alternating rows of healthy and unhealthy
vegetation simulated by bands of green and brown colours.  It is
intended for testing the real‑time crop health analysis module without
access to a real drone.  The concept mirrors how vegetation indices
such as NDVI evaluate plant health.  Healthy vegetation reflects
near‑infrared and appears green, whereas stressed plants are duller
【255148486829847†L65-L91】【255148486829847†L97-L110】.  In the synthetic image,
greener rows represent higher vegetation ratios, while brown rows mimic
unhealthy patches.

Usage
-----

Run this script from the command line to generate an image:

```
python generate_synthetic_image.py --output synthetic_field.png --width 640 --height 480
```

The image can then be analysed using the ``drone_crop_analysis.py``
module in ``--image`` mode.

Dependencies
------------

This script depends only on ``numpy`` and ``opencv-python`` (``cv2``).
Install them with ``pip install numpy opencv-python`` if they are not
already available.

"""

import argparse
from typing import Tuple
import numpy as np
import cv2


def generate_field(width: int, height: int, num_rows: int = 10) -> np.ndarray:
    """Create a synthetic crop field image.

    The image is constructed with horizontal bands representing rows of
    crops.  Half of the rows use a bright green colour to signify
    healthy vegetation, while the other half use a brownish colour to
    simulate stressed plants.  Random noise is added to make the
    appearance more natural.

    Parameters
    ----------
    width : int
        Width of the output image in pixels.
    height : int
        Height of the output image in pixels.
    num_rows : int, optional
        Number of crop rows to simulate.  Must divide evenly into
        ``height``.

    Returns
    -------
    numpy.ndarray
        Generated image in BGR format.
    """
    row_height = height // num_rows
    image = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(num_rows):
        start_y = i * row_height
        end_y = start_y + row_height
        # Alternate healthy (green) and unhealthy (brown) rows
        if i % 2 == 0:
            base_color = np.array([34, 139, 34], dtype=np.uint8)  # forest green (BGR)
        else:
            base_color = np.array([42, 42, 165], dtype=np.uint8)  # brownish (BGR)
        # Fill the row with base colour and add random variation
        noise = np.random.randint(-10, 10, size=(row_height, width, 3), dtype=np.int16)
        row = np.clip(base_color + noise, 0, 255).astype(np.uint8)
        image[start_y:end_y] = row

    return image


def main(args: argparse.Namespace) -> None:
    image = generate_field(args.width, args.height)
    cv2.imwrite(args.output, image)
    print(f"Synthetic image saved to {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic crop field image")
    parser.add_argument("--output", type=str, required=True, help="Output filename for the generated image")
    parser.add_argument("--width", type=int, default=640, help="Width of the image in pixels")
    parser.add_argument("--height", type=int, default=480, help="Height of the image in pixels")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())