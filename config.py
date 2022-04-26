# config.py
"""Configuration file containing project directives for file and folder names,
plot customizations, and so forth.

New users should set the BASE_FOLDER variable to the location of the data,
preferably as an absolute path. Other global variables specify the naming
conventions for the various data files.
"""
import os
import numpy as np
import matplotlib.pyplot as plt


# Configuration ===============================================================

# File structure --------------------------------------------------------------

BASE_FOLDER = "."                           # Base folder for data files.
FIGURES_FOLDER = "figures"                  # Saved figures folder.

SNAPSHOT_DATA_FILE = "snapshots.h5"         # Snapshot data file.
BASIS_FILE = "bases.h5"                     # POD basis file.
LOG_FILE = "log.log"                        # Log file.


def _rfmt(rs):
    return 'r' + '-'.join(f"{r:0>2}" for r in rs)


# Simulation geometry ---------------------------------------------------------

FHN_SOLVER_DEFAULTS = dict(nx=512, nt=4000, L=1, tf=4, downsample=10)

HEAT_SOLVER_DEFAULTS = dict(nx=500, nt=1000, L=1, t0=0., tf=1.)


def fhn_input(t):
    """Input signal for FitzHugh-Nagumo equations."""
    return -5e4 * t**3 * np.exp(-15*t)


# Matplotlib configuration ----------------------------------------------------

# Plotting defaults.
plt.rc("figure", figsize=(8, 3))
plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rc("legend", edgecolor='none', frameon=False)
plt.rc("animation", html="jshtml", embed_limit=50)

# Dark palette colors
chalk = (220/255, 220/255, 220/255)
accent_orange = (236/255, 138/255, 63/255)


# Validation and setup ========================================================

# Check that the base folder exists.
BASE_FOLDER = os.path.abspath(BASE_FOLDER)
if not os.path.exists(BASE_FOLDER):
    raise NotADirectoryError(BASE_FOLDER + " (set config.BASE_FOLDER)")

# Create the figures folder if needed.
if not os.path.isdir(FIGURES_FOLDER):
    os.mkdir(FIGURES_FOLDER)

# Prepend BASE_FOLDER to other data files.
SNAPSHOT_DATA_FILE = os.path.join(BASE_FOLDER, SNAPSHOT_DATA_FILE)
BASIS_FILE = os.path.join(BASE_FOLDER, BASIS_FILE)
LOG_FILE = os.path.join(BASE_FOLDER, LOG_FILE)
