# Sparse_GP_with_derivatives

Code repository for the paper: A Streaming Sparse Cholesky Method for Derivative-Informed Gaussian Process Surrogates Within Digital Twin Applications

*(This work is still under review. Link to the paper will be made available when the article is ready to be published)*

**  **

## Overview
This repository contains the implementation of the experiments presented in our paper.  

The experiments can be run by executing the `.py` files located in the **`Experiments/`** folder.

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

## Follow the insttruction to use Cythonized version of code (efficient in parallel processing)
cd SGP
python Setup.py build_ext --inplace


Note: Cython version is tested on macos. 

## Citation format (when we publish the paper):



