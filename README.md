<img src ="/images/GatorGlaciologyLogo-01.jpg" width="100" align = "right">

[![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](<https://gatorglaciology.github.io/gstatsimbook/intro.html>)  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/GatorGlaciology/gstatsimbook/HEAD?urlpath=lab)  ![PyPI](https://img.shields.io/pypi/v/gstatsim?label=pypi%20package) ![PyPI - Downloads](https://img.shields.io/pypi/dm/gstatsim) [![Twitter](https://img.shields.io/twitter/url/https/twitter.com/gstatsim.svg?style=social&label=Follow%20%40GatorGlaciology)](https://twitter.com/GatorGlaciology)


</p>

# GStatSim
GStatSim is a Python package specifically designed for geostatistical interpolation and simulation. It is inspired by open source geostatistical resources such as GeostatsPy and SciKit-GStat. The functions are intended to address the challenges of working with large data sets, non-linear trends, variability in measurement density, and non-stationarity. These tools are part of our ongoing effort to develop and adapt open-access geostatistical functions.

We have created Jupyter Book tutorials here: https://gatorglaciology.github.io/gstatsimbook/intro.html

In its current state, the demos focus on the geostatistical simulation of subglacial topography. However, these protocols could be applied to a number geoscientific topics.

We will continuously develop new tools and tutorials to address specific technical challenges in geostatistics. Do you have feedback or suggestions? Specific things that we should account for? Feel free to contact me at emackie@ufl.edu. Our goal is to create tools that are useful and accessible, so we welcome your thoughts and insight.

<img src ="/images/GStat-sim_master_figure.png" align = "center">

# Features

## Functions
Some of the tools in GStatSim:

* **skrige** - Simple kriging
* **okrige** - Ordinary kriging
* **skrige_sgs** - Sequential Gaussian simulation using simple kriging
* **okrige_sgs** - Sequential Gaussian simulation using ordinary kriging
* **cluster_sgs** - Sequential Gaussian simulation where different variograms are used in different areas
* **cokrige_mm1** - Cokriging (kriging with a secondary constraint) under Markov assumptions 
* **cosim_mm1** - Cosimulation under Markov assumptions

## Demos
We have created tutorials that are designed to provide an intuitive understanding of geostatistical methods and to demonstrate how these methods are used. The current demos are:

* **1_Experimental_Variogram.ipynb** - Demonstration of experimental variogram calculation to quantify spatial relationships.
* **2_Variogram_model.ipynb** - A tutorial on fitting a variogram model to an experimental variogram.
* **3_Simple_kriging_and_ordinary_kriging.ipynb** - Demonstration of simple kriging and ordinary kriging interpolation.
* **4_Sequential_Gaussian_Simulation.ipynb** - An introduction to stochastic simulation.
* **5_Variogram_interpolation_comparison.ipynb** - A demonstration of kriging and SGS with different variogram models.
* **6_interpolation_with_anisotropy.ipynb** - A demonstration of kriging and SGS with anisotropy.
* **7_non-stationary_SGS_example1.ipynb** - A tutorial on SGS with multiple variograms. This demo uses k-means clustering to divide the conditioning data into groups that are each assigned their own variogram.
* **8_non-stationary_SGS_example2.ipynb** - SGS using multiple variograms where the clusters are determined automatically.
* **9_interpolation_with_a_trend.ipynb** - Kriging and SGS in the presence of a large-scale trend.
* **10_cokriging_and_cosimulation_MM1.ipynb** - Kriging and SGS using secondary constraints.



# Contributors
(Emma) Mickey MacKie, University of Florida

Michael Field, University of Florida

Lijing Wang, Stanford University

(Zhen) David Yin, Stanford University

Nathan Schoedl, University of Florida

Matthew Hibbs, University of Florida

# Usage

Install GStatSim with *pip install gstatsim*

or

*git clone https://github.com/GatorGlaciology/GStatSim*



# Package dependencies
* Numpy
* Pandas
* Scipy
* tqdm
* Sklearn

# Requirements for visualization and variogram analysis
* Matplotlib
* SciKit-GStat

These can all be installed using the command *pip install (package name)*. We have included a plot_utils.py file with plotting routines.

# Educational use

GStatSim is well-suited for educational use. Please contact us if you plan on using GStatSim material in a course so we can track the impact of our work.

# Cite as

MacKie, E. J., Field, M., Wang, L., Yin, Z., Schoedl, N., Hibbs, M., & Zhang, A. (2023). GStatSim V1. 0: a Python package for geostatistical interpolation and conditional simulation. EGUsphere, 1-27.

or

@article{mackie2022gstatsim,
  title={GStatSim V1. 0: a Python package for geostatistical interpolation and conditional simulation},
  author={MacKie, Emma Johanne and Field, Michael and Wang, Lijing and Yin, Zhen and Schoedl, Nathan and Hibbs, Matthew and Zhang, Allan},
  journal={EGUsphere},
  pages={1--27},
  year={2023},
  publisher={Copernicus GmbH}
}

# Datasets

The demos use radar bed measurements from the Center for the Remote Sensing of Ice Sheets (CReSIS, 2020).

CReSIS. 2020. Radar depth sounder, Lawrence, Kansas, USA. Digital Media. http://data.cresis.ku.edu/.
