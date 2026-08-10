# Sparse_GP_with_derivatives

Code repository for the paper: A Streaming Sparse Cholesky Method for Derivative-Informed Gaussian Process Surrogates Within Digital Twin Applications.

Link to the paper: https://www.sciencedirect.com/science/article/pii/S0021999126006182

Authors: Shridhar Vashishtha*, Krishna Prasath Logakannan*, Jacob Hochhalter, Shandian Zhe, Robert M. Kirby

S.Vashishtha and KP. Logakannan contributed equally to the writing, testing, and debugging of the code.

*Vashishtha, S., Logakannan, K. P., Hochhalter, J., Zhe, S., & Kirby, R. M. (2026). A Streaming Sparse Cholesky Method for Derivative-Informed Gaussian Process Surrogates Within Digital Twin Applications. Journal of Computational Physics, 115268. https://doi.org/10.1016/j.jcp.2026.115268*

**  **

## Overview
This repository contains the implementation of the experiments presented in our paper.  

The experiments can be run by executing the `.py` files.

## Installation of required libraries

```bash
# Create a virtual environment
python -m venv venv

# Activate the environment
# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

## Follow the instructions to use Cythonized version of the code (efficient in parallel processing)
python Setup.py build_ext --inplace

Note: Cython version is tested on macOS. 




