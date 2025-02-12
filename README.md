# Scinter3
Library for analysis and simulations of scintillation data with a focus on series of single-dish observations and two screen theory.

## Compilation of C++ part
Some computationally demanding parts of code were written in C++ instead of Python and collected in the file lib_scinter.cpp. The code provides a library to be accessed in Python which was compiled on Ubuntu using the following command:
> g++ -Wall -O2 -fopenmp -shared -Wl,-soname,lib_scinter -o lib_scinter.so -fPIC lib_scinter.cpp

The compiled file lib_scinter.so is provided. If you are using an incompatible OS, you need to find a way to recompile lib_scinter.cpp.

## Required Python packages
- numpy
- scipy
- astropy
- progressbar
- matplotlib
- skimage
- ruamel.yaml<0.18.0

### Extra packages for some examples
- emcee
- corner

## Usage
Proper documentation is still missing and many parts may change in the future. Some tutorial scripts and jupyter notebooks are provided.
