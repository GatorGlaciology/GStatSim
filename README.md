<img src ="GStat-Sim/Images/GatorGlaciologyLogo-01.jpg" width="100" align = "right">

# GStatSim
GStatSim is a Python package specifically designed for geostatistical interpolation and simulation. It is inspired by open source geostatistical resources such as GeostatsPy and SciKit-GStat. The functions are intended to address the challenges of working with datasets with large crossover errors, non-linear trends, variability in measurement density, and non-stationarity. These tools are part of our ongoing effort to develop and adapt open-access geostatistical functions.

In its current state, the demos focus on the geostatistical simulation of subglacial topography. However, these protocols could be applied to a number topics in glaciology, or geoscientific problems in general.

We will continuously develop new tools and tutorials to address specific technical challenges in geostatistics. Do you have feedback or suggestions? Specific things that we should account for? Feel free to contact me at emackie@ufl.edu. Our goal is to create tools that are useful and accessible, so we welcome your thoughts and insight.

<img src ="GStat-Sim/Images/GStat-sim_master_figure.png" align = "center">

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
* **5_interpolation_with_anisotropy.ipynb** - A demonstration of kriging and SGS with anisotropy.
* **6_non-stationary_SGS_example1.ipynb** - A tutorial on SGS with multiple variograms. This demo uses k-means clustering to divide the conditioning data
* **7_non-stationary_SGS_example2.ipynb** - SGS using multiple variograms where the clusters are determined automatically.
* **8_interpolation_with_a_trend.ipynb** - Kriging and SGS in the presence of a large-scale trend
* **9_cokriging_and_cosimulation_MM1.ipynb** - Kriging and SGS using secondary constraints
into groups that are each assigned their own variogram.


# Contributors
(Emma) Mickey MacKie, University of Florida
Michael Field, University of Florida
Lijing Wang, Stanford University
(Zhen) David Yin, Stanford University
Nathan Schoedl, University of Florida
Matthew Hibbs, University of Florida

# Useage
Install GStatSim with *pip install GStatSim*

## Package dependencies
* Numpy
* Pandas
* Scipy
* tqdm
* Sklearn

## Requirements for visualization and variogram analysis
* Matplotlib
* earthpy
* SciKit-GStat

These can all be installed using the command *pip install (package name)*

## Cite as

MacKie, Emma, Field, Michael, Wang, Lijing, Yin, Zhen, Schoedl, Nathan, & Hibbs, Matthew. (2022). GStatSim (1.0). Zenodo. https://doi.org/10.5281/zenodo.7230276

or

@software{mackie_emma_2022_7230276,
  author       = {MacKie, Emma and
                  Field, Michael and
                  Wang, Lijing and
                  Yin, Zhen and
                  Schoedl, Nathan and
                  Hibbs, Matthew},
  title        = {GStatSim},
  month        = oct,
  year         = 2022,
  publisher    = {Zenodo},
  version      = {1.0},
  doi          = {10.5281/zenodo.7230276},
  url          = {https://doi.org/10.5281/zenodo.7230276}
}

# Datasets

The demos use radar bed measurements from the Center for the Remote Sensing of Ice Sheets (CReSIS, 2020).

CReSIS. 2020. Radar depth sounder, Lawrence, Kansas, USA. Digital Media. http://data.cresis.ku.edu/.
