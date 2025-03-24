from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, NamedTuple

import nibabel as nib
import numpy as np
import pyvista as pv
from matplotlib import image
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, Normalize, to_rgba
from matplotlib.transforms import IdentityTransform
from PIL import Image, ImageOps

VIEW_CAMERA_POS_MAP = {
    ("lh", "lateral"): ((-500, 0, 0), (0, 0, 0), (0, 0, 1)),
    ("lh", "medial"): ((500, 0, 0), (0, 0, 0), (0, 0, 1)),
    ("lh", "posterior"): ((0, -500, 0), (0, 0, 0), (0, 0, 1)),
    ("lh", "anterior"): ((0, 500, 0), (0, 0, 0), (0, 0, 1)),
    ("lh", "inferior"): ((0, 0, -500), (0, 0, 0), (-1, 0, 0)),
    ("lh", "superior"): ((0, 0, 500), (0, 0, 0), (1, 0, 0)),
    ("rh", "lateral"): ((500, 0, 0), (0, 0, 0), (0, 0, 1)),
    ("rh", "medial"): ((-500, 0, 0), (0, 0, 0), (0, 0, 1)),
    ("rh", "posterior"): ((0, -500, 0), (0, 0, 0), (0, 0, 1)),
    ("rh", "anterior"): ((0, 500, 0), (0, 0, 0), (0, 0, 1)),
    ("rh", "inferior"): ((0, 0, -500), (0, 0, 0), (1, 0, 0)),
    ("rh", "superior"): ((0, 0, 500), (0, 0, 0), (-1, 0, 0)),
}

# This is the width of the left lateral view / window size after cropping.
# We scale window size by this so that the cropped images are about the requested size.
WINDOW_SCALE = 5 / 3


class View(StrEnum):
    """Enumeration of standard surface views."""

    LATERAL = "lateral"
    MEDIAL = "medial"
    POSTERIOR = "posterior"
    ANTERIOR = "anterior"
    INFERIOR = "inferior"
    SUPERIOR = "superior"


class CameraPos(NamedTuple):
    """PyVista camera position parameters.

    Attributes
    ----------
    position : tuple of float
        Camera position in 3D space.
    focal_point : tuple of float
        Point in 3D space at which the camera is looking.
    viewup : tuple of float
        Upward direction vector for the camera.

    See also
    --------
    https://docs.pyvista.org/api/plotting/_autosummary/pyvista.cameraposition
    """

    position: tuple[float, float, float]
    focal_point: tuple[float, float, float]
    viewup: tuple[float, float, float]


class Surface(NamedTuple):
    """Surface mesh representation.

    Attributes
    ----------
    points : ndarray of shape (n_points, 3)
        Array of 3D coordinates representing surface vertices.
    faces : ndarray of shape (n_faces, 3)
        Array of vertex indices forming triangular faces.
    """

    points: np.ndarray
    faces: np.ndarray


class Overlay(ScalarMappable):
    """Color-mapped 1D overlay for surface visualization.

    Parameters
    ----------
    values : ndarray of shape (n_points,) or (n_points, 3) or (n_points, 4)
        Overlay values, either scalar (for colormap mapping) or RGB(A) colors.
    cmap : str or Colormap, default=None
        Colormap used for scalar overlay values.
    norm : str or Normalize, default=None
        Normalization for mapping scalar values to colors.
    vmin : float, default=None
        Minimum value for normalization.
    vmax : float, default=None
        Maximum value for normalization.
    alpha : float, default=None
        Alpha transparency value.
    """

    def __init__(
        self,
        values: np.ndarray,
        cmap: str | Colormap | None = None,
        norm: str | Normalize | None = None,
        *,
        vmin: float | None = None,
        vmax: float | None = None,
        alpha: float | None = None,
    ):
        super().__init__(norm=norm, cmap=cmap)
        if not (values.ndim == 1 or (values.ndim == 2 and values.shape[1] in {3, 4})):
            raise ValueError(
                "Invalid overlay values; "
                "expected shape (n_points,) or (n_points, {3, 4})."
            )
        # Prepend singleton dimension so that the pass-through of RGB(A) values works.
        self.set_array(values[None])
        self.set_clim(vmin=vmin, vmax=vmax)
        self._alpha = alpha

    def pixel_values(self) -> np.ndarray:
        """Compute RGBA pixel values from overlay.

        Returns
        -------
        ndarray of shape (n_points, 4)
            RGBA color values for each surface point.
        """
        return self.to_rgba(self.get_array(), alpha=self._alpha).squeeze(0)


class Plotter:
    """Surface plotter using PyVista for 3D visualization.

    Parameters
    ----------
    surf : Path or Surface
        Path to a surface file (.gii) or a preloaded Surface instance.
    hemi : {'lh', 'rh'}, default='lh'
        Hemisphere specification ('lh' for left, 'rh' for right).
    sulc : Path or ndarray, default=None
        Path to a sulcal depth file or an array of sulcal depth values.
    color : any, default=(0.6, 0.6, 0.6)
        Base color of the surface.
    width : int, default=256
        Image width for rendering.
    """

    def __init__(
        self,
        surf: Path | Surface,
        hemi: Literal["lh", "rh"] = "lh",
        sulc: Path | np.ndarray | None = None,
        color: Any = (0.6, 0.6, 0.6),
        width: int = 256,
    ):
        if isinstance(surf, (str, Path)):
            surf = _read_surface(surf)
        self._surf = Surface(*surf)
        self._hemi = hemi
        self._color = color
        self._width = width

        n_points = len(self._surf.points)
        if sulc is not None:
            if isinstance(sulc, (str, Path)):
                sulc = _read_shape(sulc)
            if sulc.shape != (n_points,):
                raise ValueError(
                    f"sulc data doesn't match surface; expected shape ({n_points},)."
                )
            self._base_overlay = _sulc_overlay(sulc)
        else:
            self._base_overlay = _constant_overlay(n_points, color)

        self._poly = _surface_to_polydata(surf)
        self._plotter = pv.Plotter(
            window_size=(int(WINDOW_SCALE * width), int(WINDOW_SCALE * 0.75 * width)),
            off_screen=True,
        )
        self._overlays: list[Overlay] = []

    def overlay(
        self,
        values: np.ndarray,
        cmap: str | Colormap | None = None,
        norm: str | Normalize | None = None,
        *,
        vmin: float | None = None,
        vmax: float | None = None,
        alpha: float | None = None,
    ) -> Overlay:
        """Apply an overlay to the surface.

        Parameters
        ----------
        values : ndarray of shape (n_points,) or (n_points, 3) or (n_points, 4)
            Scalar values to overlay on the surface.
        cmap : str or Colormap, default=None
            Colormap used for overlay.
        norm : str or Normalize, default=None
            Normalization for mapping values to colors.
        vmin : float, default=None
            Minimum value for normalization.
        vmax : float, default=None
            Maximum value for normalization.
        alpha : float, default=None
            Alpha transparency value.

        Returns
        -------
        overlay : Overlay
            The applied overlay object. Is instance of `ScalarMappable`, so can be used
            for creating colorbars.
        """
        n_points = len(self._surf.points)
        if len(values) != n_points:
            raise ValueError(f"Overlay doesn't match surface; expected {n_points=}.")

        # Invalidate the plotter. Note, plotter.clear() also clears shading properties.
        self._plotter.actors.clear()
        overlay = Overlay(
            values=values, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, alpha=alpha
        )
        self._overlays.append(overlay)
        return overlay

    def _render(self) -> np.ndarray:
        """Combine the overlays and render the 3D scene."""
        layers = [self._base_overlay.pixel_values()]
        layers += [overlay.pixel_values() for overlay in self._overlays]
        composite = _alpha_composite(layers)

        self._plotter.actors.clear()
        self._plotter.add_mesh(
            self._poly.copy(),
            scalars=composite,
            rgb=True,
            show_scalar_bar=False,
        )

    def screenshot(
        self,
        view: View | CameraPos = View.LATERAL,
        pad: int | None = None,
        color: Any | None = None,
    ) -> Image.Image:
        """Capture a screenshot of the surface.

        Parameters
        ----------
        view : View or CameraPos, default=View.LATERAL
            Viewpoint or camera position for rendering.
        pad : int, default=None
            Padding around the image.
        color : any, default=None
            Background color.

        Returns
        -------
        img: Image.Image
            The rendered surface image.
        """
        if isinstance(view, (View, str)):
            camera_pos = VIEW_CAMERA_POS_MAP[(self._hemi, View(view).value)]
        else:
            camera_pos = view

        if len(self._plotter.renderer.actors) == 0:
            self._render()

        self._plotter.camera_position = camera_pos
        self._plotter.render()
        img = self._plotter.screenshot(return_img=True, transparent_background=True)

        img = _crop_transparent_background(img)
        img = Image.fromarray(img)

        if pad is not None:
            img = ImageOps.expand(img, pad, (255, 255, 255, 0))

        if color is not None:
            bg = Image.new(img.mode, img.size, color=color)
            img = Image.alpha_composite(bg, img)
        return img

    def imshow(
        self,
        view: View | CameraPos = View.LATERAL,
        pad: int | None = None,
        color: Any | None = None,
        extent: tuple[float, float, float, float] | None = None,
    ) -> Overlay | None:
        """Display the rendered surface using matplotlib.

        Parameters
        ----------
        view : View or CameraPos, default=View.LATERAL
            Viewpoint or camera position for rendering.
        pad : int, default=None
            Padding around the image.
        color : any, default=None
            Background color.
        extent : tuple of float, default=None
            Extent for image positioning in the plot.

        Returns
        -------
        overlay: Overlay or None
            The last applied overlay if available, else None.
        """
        img = self.screenshot(view, pad=pad, color=color)
        plt.imshow(img, aspect="equal", extent=extent)
        if len(self._overlays) > 0:
            return self._overlays[-1]
        return None

    def clear(self) -> None:
        """Clear all overlays from the plotter."""
        self._overlays.clear()
        self._plotter.actors.clear()


def _read_surface(path: Path) -> Surface:
    """Read a surface from a file."""
    path = Path(path)
    match path.suffix:
        case ".gii":
            surf = nib.load(path)
            points = surf.darrays[0].data
            faces = surf.darrays[1].data
            surf = Surface(points, faces)
        case _:
            raise ValueError(
                f"Unsupported surface format: {path}. Only .gii supported."
            )
    return surf


def _read_shape(path: Path) -> np.ndarray:
    """Read a surface metric/shape from a file."""
    path = Path(path)
    match path.suffix:
        case ".gii":
            shape = nib.load(path).darrays[0].data
        case _:
            raise ValueError(
                f"Unsupported surface format: {path}. Only .gii supported."
            )
    return shape


def _surface_to_polydata(surf: Surface) -> pv.PolyData:
    """Convert a surface (points, faces) to a pyvista mesh."""
    points, faces = surf
    if not points.ndim == faces.ndim == 2 and points.shape[1] == faces.shape[1] == 3:
        raise ValueError("Invalid surface points/faces. Expected two Nx3 arrays.")

    # prepend number of points and flatten, pyvista format
    # https://docs.pyvista.org/examples/00-load/create-poly#sphx-glr-examples-00-load-create-poly-py
    faces = np.concatenate(
        [np.full((len(faces), 1), 3, dtype=faces.dtype), faces], axis=1
    )
    poly = pv.PolyData(points, faces.flatten())
    return poly


def _constant_overlay(n_points: int, color: Any) -> Overlay:
    """Create a constant RGB overlay."""
    rgba = to_rgba(color)
    values = np.tile(np.asarray(rgba), (n_points, 1))
    return Overlay(values)


def _sulc_overlay(values: np.ndarray, cmin: float = 0.4, cmax: float = 0.6) -> Overlay:
    """Create a binary sulcal depth overlay."""
    return Overlay(np.where(values < 0, cmin, cmax), cmap="gray", vmin=0.0, vmax=1.0)


def _alpha_composite(layers: list[np.ndarray]) -> np.ndarray:
    """Make alpha blend of a stack of RGBA layers."""
    assert len(layers) > 0, "expected at least one layer"
    assert all(
        layer.ndim == 1 or (layer.ndim == 2 and layer.shape[1] in {3, 4})
        for layer in layers
    ), "expected layers to be shape (n_points,) or (n_points, {3, 4})"

    sizes = set(len(layer) for layer in layers)
    assert len(sizes) == 1
    size = sizes.pop()

    # Make all layers appear like image array, shape (1, width).
    layers = [layer[None] for layer in layers]

    # Make composite.
    output = np.zeros((1, size, 4), dtype=layers[0].dtype)
    transform = IdentityTransform()
    for layer in layers:
        image.resample(layer, output, transform=transform, interpolation=image.NEAREST)

    output = output.squeeze(0)
    return output


def _crop_transparent_background(image: np.ndarray) -> np.ndarray:
    """Crop out transparent background from image."""
    assert image.ndim == 3 and image.shape[-1] == 4
    bg_mask = image[..., 3] == 0
    row_ind, col_ind = np.where(~bg_mask)
    y1, y2 = row_ind.min(), row_ind.max()
    x1, x2 = col_ind.min(), col_ind.max()
    cropped = image[y1 : y2 + 1][:, x1 : x2 + 1]
    return cropped
