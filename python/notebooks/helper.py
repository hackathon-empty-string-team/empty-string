# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # A file for Helper functions

# %%
import numpy as np
import os


# %%
# Uniform sampling over the filenames WITHOUT replacement (i.e. we get no duplicates)
# Import: from helper import getUniformSamples
def getUniformSamples(dir, n):
    """
    Takes: The input directory of the dataset and the number of samples to be taken from it
    Returns: A python dictionary of the form {file_id: file_name}
    """
    fnames = os.listdir(dir)
    
    assert(len(fnames) <= n, "Requesting more samples than files in directory")
    
    fnames = np.random.choice(fnames, n, replace=False)
    
    return {i: f for i, f in enumerate(fnames)}  
