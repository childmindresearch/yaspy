# yaspy

Yaspy (yet another surface plotting python library) is a tool for making neuroimaging surface plots, from simple screenshots to multi-panel montages to publication ready figures. It is built on top of [PyVista](https://docs.pyvista.org/), [Matplotlib](https://matplotlib.org/), and [Pillow](https://pillow.readthedocs.io/en/stable/index.html).

<p align="left">
  <img src="doc/_static/img/rsfc_principal_gradient.png" height="300">
</p>

## Installation

Install from github with pip via

```bash
pip install git+https://github.com/childmindresearch/yaspy.git
```

## Example

```python
import yaspy
from neuromaps.datasets import fetch_fslr

surfaces = fetch_fslr()
surf_path, _ = surfaces["inflated"]
sulc_path, _ = surfaces["sulc"]

plotter_lh = yaspy.Plotter(surf_path, hemi="lh", sulc=sulc_path)
plotter_lh.screenshot(view="lateral")
```

<p align="left">
  <img src="doc/_static/img/example.png" height="200">
</p>

## Tutorial

See our [tutorial notebook](examples/tutorial.ipynb) for an in-depth tour of what you can do with yaspy.

## Related libraries

- [nilearn](https://nilearn.github.io/dev/modules/generated/nilearn.plotting.plot_surf.html)
- [brainspace](https://brainspace.readthedocs.io/en/latest/python_doc/api_doc/brainspace.plotting.html)
- [surfplot](https://github.com/danjgale/surfplot)
- [brainplotlib](https://github.com/feilong/brainplotlib)
- [pycortex](https://github.com/gallantlab/pycortex)
