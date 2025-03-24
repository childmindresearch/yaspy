import math
from typing import Literal

import numpy as np
from PIL import Image, ImageOps


def montage(
    images: list[Image.Image | None] | list[list[Image.Image | None]],
    pad: int | None = None,
    color: str | tuple[int, ...] | None = None,
    ha: Literal["left", "center", "right"] = "center",
    va: Literal["top", "center", "bottom"] = "center",
    shareh: bool = False,
    sharew: bool = False,
) -> Image.Image:
    """Create a montage of images arranged in a grid.

    This function aligns images into a grid while handling different sizes and
    missing images (`None`). It optionally pads images and ensures alignment
    based on horizontal and vertical alignment parameters.

    Parameters
    ----------
    images : list[list[Image.Image | None]] or list[Image.Image | None]
        A list of lists representing rows of images, or a single flat list for a single
        row. `None` values represent empty slots in the montage.
    pad : int, optional
        Number of pixels to pad around each image.
    color : str or tuple[int, ...], optional
        Background color for padding and empty slots. If not provided, it is
        inferred from the background of the first non-`None` image.
    ha : {'left', 'center', 'right'}, default='center'
        Horizontal alignment of images within each column.
    va : {'top', 'center', 'bottom'}, default='center'
        Vertical alignment of images within each row.
    shareh : bool, default=False
        If True, all images share the same height. Otherwise, only images within the
        same row share height.
    sharew : bool, default=False
        If True, all images share the same width. Otherwise, only images within the
        same column share width.

    Returns
    -------
    grid: Image.Image
        A single PIL image containing the assembled montage.

    See Also
    --------
    image_grid : Lower-level function for arranging images in a grid.
    """
    if not isinstance(images[0], list):
        images = [images]

    # Get background color of first image and set as background.
    first_img: Image.Image = next(
        img for row in images for img in row if img is not None
    )
    if color is None:
        color = tuple(np.asarray(first_img)[0, 0])
    mode = first_img.mode

    # Convert to centering argument for padding.
    hc = {"left": 0.0, "center": 0.5, "right": 1.0}[ha]
    vc = {"top": 0.0, "center": 0.5, "bottom": 1.0}[va]
    centering = (hc, vc)

    # Pad each row with None to make a ragged grid.
    ncol = max(len(row) for row in images)
    images = [row + (ncol - len(row)) * [None] for row in images]

    # Pad each image on all sides.
    if pad:
        images = [
            [
                ImageOps.expand(img, pad, fill=color) if img is not None else None
                for img in row
            ]
            for row in images
        ]

    # Get max widths of each column and max heights of each row.
    # Then, each image is resized/padded to the aligned size of its row/column.
    sizes = np.array(
        [[img.size if img is not None else (0, 0) for img in row] for row in images]
    )
    widths = np.max(sizes[:, :, 0], axis=0)
    heights = np.max(sizes[:, :, 1], axis=1)

    if sharew:
        widths = np.full_like(widths, widths.max())
    if shareh:
        heights = np.full_like(heights, heights.max())

    # Resize/pad each image to the appropriate size.
    pad_images = []
    for ii, row in enumerate(images):
        for jj, img in enumerate(row):
            size = widths[jj], heights[ii]
            if img is None:
                # Fill with blank background image.
                img = Image.new(mode, size, color=color)
            else:
                # Resize/pad to target size.
                img = ImageOps.pad(img, size, color=color, centering=centering)
            pad_images.append(img)

    # Finally, make the image grid. This is a simple function that just pastes the
    # image and doesn't handle padding or alignment at all.
    grid = image_grid(pad_images, ncol=ncol, color=color)
    return grid


def image_grid(
    images: list[Image.Image],
    ncol: int,
    color: str | tuple[int, ...] | None = None,
) -> Image.Image:
    """Arrange images in a simple grid without resizing or padding.

    This function places images in a grid of `ncol` columns, filling rows
    as needed. It does not resize or align images; they are pasted in
    their original sizes.

    Parameters
    ----------
    images : list[Image.Image]
        List of PIL images to arrange in a grid.
    ncol : int
        Number of columns in the grid.
    color : str or tuple[int, ...], optional
        Background color for the grid. If not provided, it is inferred from
        the background of the first image.

    Returns
    -------
    grid : Image.Image
        A single PIL image containing the arranged grid.

    See Also
    --------
    montage : Higher-level function for arranging images with padding and alignment.
    """
    if color is None:
        color = tuple(np.asarray(images[0])[0, 0])

    widths, heights = zip(*(img.size for img in images))
    width = max(widths)
    height = max(heights)
    nrow = math.ceil(len(images) / ncol)

    left, upper, right, lower = 0, 0, 0, 0
    grid = Image.new(images[0].mode, size=(ncol * width, nrow * height), color=color)

    for ii, img in enumerate(images):
        grid.paste(img, (left, upper))
        right = max(right, left + img.width)
        lower = max(lower, upper + img.height)
        if (ii + 1) % ncol == 0:
            left, upper = 0, lower
        else:
            left, upper = left + img.width, upper

    grid = grid.crop((0, 0, right, lower))
    return grid
