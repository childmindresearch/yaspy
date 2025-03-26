import logging
import time

import matplotlib
import nilearn.plotting as nilplt
import numpy as np
from matplotlib import pyplot as plt
from neuromaps.datasets import fetch_fslr
from PIL import Image
from tqdm import tqdm

import yaspy

matplotlib.use("agg")

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M:%S",
)


def main():
    surfaces = fetch_fslr()
    surf_path, _ = surfaces["inflated"]
    sulc_path, _ = surfaces["sulc"]

    logging.info("Benchmarking yaspy overlay screenshots...")
    n_frames = 1000
    rng = np.random.default_rng(42)
    plotter = yaspy.Plotter(surf_path, hemi="lh", sulc=sulc_path)
    n_points = len(plotter._surf.points)
    tic = time.monotonic()
    for ii in tqdm(range(n_frames)):
        plotter.clear()
        values = rng.normal(size=(n_points,))
        plotter.overlay(values, vmin=-2.5, vmax=2.5)
        plotter.screenshot()
    rt = time.monotonic() - tic
    logging.info("Yaspy overlay screenshot FPS: %.1f", n_frames / rt)

    logging.info("Benchmarking nilearn plot surf with overlay...")
    n_frames = 20
    rng = np.random.default_rng(42)
    surf = plotter._surf
    f = plt.figure()
    tic = time.monotonic()
    for ii in tqdm(range(n_frames)):
        f.clear()
        values = rng.normal(size=(n_points,))
        nilplt.plot_surf(surf, surf_map=values, vmin=-2.5, vmax=2.5, figure=f)

        # Render the figure as an in memory PIL Image, for consistency.
        # Note that without at least calling f.canvas.draw(), matplotlib doesn't
        # actually render the image.
        f.canvas.draw()
        buf = np.asarray(f.canvas.renderer.buffer_rgba())
        Image.fromarray(buf)
    rt = time.monotonic() - tic
    logging.info("nilearn plot_surf overlay FPS: %.1f", n_frames / rt)


if __name__ == "__main__":
    main()
