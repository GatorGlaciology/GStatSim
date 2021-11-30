# GlacierStats
GlacierStats is a collection of functions and demos specifically designed for stochastic methods in glaciology. It is inspired by open source geostatistical resources such as GeostatsPy and Geostatistics Lessons. In my own research, I have found that geostatistical tools designed for industry applications do not have the flexibility to address the unique combination of challenges in ice sheet problems, including large crossover errors, spatially variable measurement uncertainty, extremely large datasets, non-linear trends, variability in measurement density, and non-stationarity. These tools are part of our ongoing effort to develop and adapt geostatistical techniques and other machine learning methods to glaciology.

In its current state, the demos focus on the geostatistical simulation of subglacial topography. However, these protocols could be applied to a number topics in glaciology, or remote sensing problems in general.

We will continuously develop new tools and tutorials to address specific technical challenges in glaciology. Do you have feedback or suggestions? Specific things that we should account for? Feel free to contact me at mackie3@stanford.edu. Our goal is to create tools that are useful and accessible, so we welcome your thoughts and insight.

# Features

## Functions
Some of the tools in GlacierStats:

* **axis_var** - Obtain the variogram for the major or minor axis
* **skrige** - Simple kriging
* **okrige** - Ordinary kriging
* **sgsim** - Sequential Gaussian simulation

## Demos
We have created tutorials that are designed to provide an intuitive understanding of geostatistical methods and to demonstrate how these methods can be used for ice sheet analysis. The current demos are:

* **Experimental_Variogram.ipynb** - Demonstration of experimental variogram calculation to quantify spatial relationships.
* **Variogram_model.ipynb** - A tutorial on fitting a variogram model to an experimental variogram.
* **Outlier_removal.ipynb** - Outlier removal to minimize artifacts from measurement crossover errors.
* **Kriging.ipynb** - Demonstration of simple kriging and ordinary kriging interpolation with commentary on glaciology-specific considerations.
* **Sequential_Gaussian_simulation.ipynb** - An introduction to stochastic simulation of subglacial topography.


# The author
(Emma) Mickey MacKie is an assistant professor at the University of Florida.

# Useage
The functions are in the GlacierStats.py document. Just download the GlacierStats folder and make sure the GlacierStats.py script is in your working directory. The datasets for the demos are in the Data folder.

## Dependencies
* Numpy
* Pandas
* Math
* Scipy
* Matplotlib
* tqdm
* Sklearn
* earthpy
* GeostatsPy
* random

These can all be installed using the command *pip install (package name)*

# Datasets

The demos use radar bed measurements from the Center for the Remote Sensing of Ice Sheets (CReSIS, 2020) and elevation data from BedMachine Greenland (Morlighem et al., 2017).

CReSIS. 2020. REPLACE_WITH_RADAR_NAME Data, Lawrence, Kansas, USA. Digital Media. http://data.cresis.ku.edu/.

Morlighem, M., Williams, C. N., Rignot, E., An, L., Arndt, J. E., Bamber, J. L., ... & Fenty, I. (2017). BedMachine v3: Complete bed topography and ocean bathymetry mapping of Greenland from multibeam echo sounding combined with mass conservation. Geophysical research letters, 44(21), 11-051.
