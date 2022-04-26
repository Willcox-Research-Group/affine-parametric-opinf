# utils.py
"""Utilities for reading/saving data and saving figures."""

import os
import sys
import time
import h5py
import signal   # may be Unix only
import logging
import itertools
import collections
import numpy as np
import matplotlib.pyplot as plt

import config


# Logging =====================================================================

def init_logger():
    """Initialize the logger."""
    # Remove all old logging handlers.
    logger = logging.getLogger()
    for hdlr in logger.handlers[:]:
        logger.removeHandler(hdlr)

    # Get the log filename and append a newline.
    logfile = config.LOG_FILE
    with open(logfile, 'a') as lf:
        lf.write('\n')

    # Get a new logging handler to the log file.
    handler = logging.FileHandler(logfile, 'a')
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    handler.setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    print(f"Logging to {logfile}")

    # Log the session header.
    if hasattr(sys.modules["__main__"], "__file__"):
        _front = f"({os.path.basename(sys.modules['__main__'].__file__)})"
        _end = time.strftime('%Y-%m-%d %H:%M:%S')
        _mid = '-' * (79 - len(_front) - len(_end) - 20)
        header = f"NEW SESSION {_front} {_mid} {_end}"
    else:
        header = f"NEW SESSION {time.strftime(' %Y-%m-%d %H:%M:%S'):->61}"
    logging.info(header)


init_logger()


class timed_block:
    """Context manager for timing a block of code and reporting the timing.

    >>> with timed_block("This is a test"):
    ...     # Code to be timed
    ...     time.sleep(2)
    ...
    This is a test...done in 2.00 s.

    >>> with timed_block("Another test", timelimit=3):
    ...     # Code to be timed and halted within the specified time limit.
    ...     i = 0
    ...     while True:
    ...         i += 1
    Another test...TIMED OUT after 3.00 s.
    """
    verbose = True

    @staticmethod
    def _signal_handler(signum, frame):
        raise TimeoutError("timed out!")

    @property
    def timelimit(self):
        return self._timelimit

    def __init__(self, message, timelimit=None):
        self.message = message
        self._end = '\n' if '\r' not in message else ''
        self._timelimit = timelimit

    def __enter__(self):
        """Print the message and record the current time."""
        if self.verbose:
            print(f"{self.message}...", end='', flush=True)
        self._tic = time.time()
        if self._timelimit is not None:
            signal.signal(signal.SIGALRM, self._signal_handler)
            signal.alarm(self._timelimit)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Calculate and report the elapsed time."""
        self._toc = time.time()
        if self._timelimit is not None:
            signal.alarm(0)
        elapsed = self._toc - self._tic
        if exc_type:    # Report an exception if present.
            if self._timelimit is not None and exc_type is TimeoutError:
                print(f"TIMED OUT after {elapsed:.2f} s.",
                      flush=True, end=self._end)
                logging.info(f"TIMED OUT after {elapsed:.2f} s.")
                return True
            print(f"{exc_type.__name__}: {exc_value}")
            logging.info(self.message.strip())
            logging.error(f"({exc_type.__name__}) {exc_value} "
                          f"(raised after {elapsed:.6f} s)")
        else:           # If no exception, report execution time.
            if self.verbose:
                print(f"done in {elapsed:.2f} s.", flush=True, end=self._end)
            logging.info(f"{self.message.strip()}...done in {elapsed:.6f} s.")
        self.elapsed = elapsed
        return


# Data management =============================================================

class DataNotFoundError(FileNotFoundError):
    """Exception to be raised when attempting to load a missing data file."""
    pass


def _checkexists(filename):
    """Check that the file `filename` exists; if not, raise an exception."""
    if not os.path.isfile(filename):
        raise DataNotFoundError(filename)
    return filename


def save_basis(group, Vr, svdvals=None):
    """Save a POD basis and (optionally) the associated singular values.

    Parameters
    ----------
    group : str
        Name under which to save this set of data.
    Vr : (n, r) ndarray
        POD basis matrix of rank r.
    svdvals : (n,) ndarray
        Singular values of the POD basis (optional).
    """
    save_path = config.BASIS_FILE
    msg = '' if svdvals is None else " and singular values"
    with timed_block(f"Saving basis{msg}"):
        with h5py.File(save_path, 'a') as hf:
            # Delete existing group of the same name if present.
            if group in hf:
                del hf[group]
            # Create new group and add data.
            gp = hf.create_group(group)
            gp.create_dataset("basis", data=Vr)
            if svdvals is not None:
                gp.create_dataset("svdvals", data=svdvals)

    logging.info(f"Basis{msg} saved to {save_path}.\n")


def load_basis(group, r, svdvals=False):
    """Load POD basis and (optionally) associated singular values.

    Parameters
    ----------
    group : str
        Name of data set to load (set previously in save_training_data()).
    r : int
        Number of retained POD modes to load.
    svdvals : bool
        If True, also return the POD singular values.

    Returns
    -------
    Vr : (n, r) ndarray
        POD basis matrix of order r.
    svdvals : (n,) ndarray
        Singular values of the POD basis. Only returned if svdvals=True.
    """
    data_path = _checkexists(config.BASIS_FILE)
    with timed_block(f"Loading basis '{group}' from {data_path}"):
        with h5py.File(data_path, 'r') as hf:

            # Check data shapes.
            basisname = f"{group}/basis"
            rmax = hf[basisname].shape[1]
            if r is None:
                r = rmax
            if rmax < r:
                raise ValueError(f"basis only has {rmax} columns")

            # Get the correct subsets of the saved data.
            Vr = hf[basisname][:, :r]
            return (Vr, hf[f"{group}/svdvals"][:]) if svdvals else Vr


class FHNDataManager:
    """File structure manager for FitzHugh-Nagumo experiments."""
    _valid_labels = {
        "train",
    }

    def __init__(self, label):
        """Set the experiment label."""
        if label not in self._valid_labels:
            raise ValueError(f"unrecognized label '{label}'")
        self.label = label

        self.base_folder = os.path.join(config.BASE_FOLDER, f"fhn_{label}")
        if not os.path.isdir(self.base_folder):
            os.mkdir(self.base_folder)

    def __repr__(self):
        """String representation: base folder"""
        return f"FHNDataManager('{self.base_folder}')"

    @property
    def solverfile(self):
        """Full-order training data, loaded with fhn.FHNROMSolver.load()."""
        return os.path.join(self.base_folder, f"fhn_{self.label}.h5")

    @property
    def trainingdatafile(self):
        """Full-order training data, downsampled."""
        return os.path.join(self.base_folder, "traindata.h5")

    @property
    def testingdatafile(self):
        """Full-order testing data, downsampled."""
        return os.path.join(self.base_folder, "testdata.h5")

    @property
    def multiplotfile(self):
        """Results for all basis sizes."""
        return os.path.join(self.base_folder, "multiplot.h5")

    def resultsfile(self, train, intrusive, rs):
        """Results for a single ROM."""
        filenameparts = ["results"]
        filenameparts.append("train" if train else "test")
        if intrusive:
            filenameparts.append("intrusive")
        filenameparts.append(config._rfmt(rs))
        filename = "_".join(filenameparts) + ".h5"
        return os.path.join(self.base_folder, filename)

    def romfile(self, rs):
        """Learned pOpInf AffineFHNROM."""
        return os.path.join(self.base_folder, f"rom_{config._rfmt(rs)}.h5")

    def regsfile(self, rs):
        """Regularization hyperparameters."""
        return os.path.join(self.base_folder, f"regs_{config._rfmt(rs)}.npy")

    def training_parameters(self):
        """HARD CODED training parameters"""
        if self.label == "train":
            ignore = np.array([
                [0.025, 0.55, 2.5, 0.010],
                [0.035, 0.65, 2.5, 0.025],
            ])
            paramgen = itertools.product(
                [.025, .035, .045, .055, .065, .075],
                [0.25, 0.35, 0.45, 0.55, 0.65, 0.75],
                [2, 2.5],
                [.010, .015, .020, .025, .030, .035, .040]
            )
            params = [µ for µ in paramgen if not in2Darray(ignore, µ)]
        else:
            raise ValueError(self.label)
        return np.array(params)

    def testing_parameters(self):
        """HARD CODED testing parameters"""
        if self.label == "train":
            params = list(itertools.product(
                np.round(np.arange(.025, .080, .005), 3),
                np.round(np.arange(0.25, 0.80, 0.05), 2),
                [2, 2.25, 2.5],
                np.round(np.arange(.010, .041, .001), 3)))
        else:
            raise ValueError(self.label)
        return np.array(params)

    def fullcomparison_parameters(self):
        """HARD CODED parameters at which to compare FOM / ROM solutions."""
        if self.label == "train":
            params = [
                [.030, .60, 2.50, .037],
                [.040, .60, 2.00, .012],
                [.050, .30, 2.25, .023],
            ]
        else:
            raise ValueError(self.label)
        return np.array(params)

    def _trialchunks(self, params, trialsize):
        indices = np.arange(trialsize, params.shape[0], trialsize)
        return {f"trial{i:0>3}": chunk
                for i, chunk in enumerate(np.split(params, indices))}

    def training_trials(self):
        """Split the training set into chunks for distribution."""
        return self._trialchunks(self.training_parameters(), 35)

    def testing_trials(self):
        """Split the testing set into chunks for distribution."""
        return self._trialchunks(self.testing_parameters(), 100)

    def _loc(self, params, µ, trialsize):
        diffs = np.linalg.norm(params - µ, axis=1)
        loc = np.argmin(diffs)
        if diffs[loc] > 1e-16:
            return "not found"
        trial, index = divmod(loc, trialsize)
        return f"trial{trial:0>2}_{index:0>2}"

    def loc_train(self, µ):
        """Locate a parameter within the training set."""
        return self._loc(self.training_parameters(), µ, 35)

    def loc_test(self, µ):
        """Locate a parameter withing the testing set."""
        return self._loc(self.testing_parameters(), µ, 100)

    def basis_sizes(self, low=3, high=12):
        """Get basis sizes corresponding to levels of residual energy."""
        _, svals1 = load_basis(f"{self.label}_1", None, True)
        _, svals2 = load_basis(f"{self.label}_2", None, True)
        resid1 = 1 - np.cumsum(svals1**2)/np.sum(svals1**2)
        resid2 = 1 - np.cumsum(svals2**2)/np.sum(svals2**2)
        energies = [float(f"1e-{i:d}") for i in range(low, high+1)]
        return [(np.count_nonzero(resid1 > level) + 1,
                 np.count_nonzero(resid2 > level) + 1) for level in energies]

    def get_regularizations(self):
        """Load regularization hyperparameters for each basis size."""
        regularizations = {}
        for rs in self.basis_sizes():
            filename = self.regsfile(rs)
            if not os.path.isfile(filename):
                print(f"{filename} not found")
                continue
            regs = np.load(filename)
            print(f"rs = {rs}: {regs}")
            regularizations[rs] = regs
        return regularizations

    def get_unstables(self, flatten=False):
        """Search results for parameters where the pOpInf ROM is unstable."""
        unstables = collections.defaultdict(list)
        for rs in self.basis_sizes():
            filename = self.resultsfile(False, False, rs)
            if not os.path.isfile(filename):
                print(f"{filename} not found")
                continue
            with h5py.File(filename, 'r') as hf:
                for trial in hf["romerrors"]:
                    for µ in hf[f"romerrors/{trial}/unstables"][:]:
                        print(f"rs = {rs}, unstable at µ = {µ}")
                        unstables[rs].append(µ.tolist())
        if flatten:
            unstables = np.unique(np.vstack(list(unstables.values())), axis=0)
        return unstables


# Figure management ===========================================================

def save_figure(figname):
    """Save the current matplotlib figure to the figures folder."""
    save_path = os.path.join(config.FIGURES_FOLDER, figname)
    # plt.show()  # Uncomment to display figure before saving.
    with timed_block(f"Saving {save_path}"):
        plt.savefig(save_path, bbox_inches="tight", dpi=1200)
        plt.close(plt.gcf())


# Misc ========================================================================

def in2Darray(arr, vec):
    return np.abs(arr - vec).sum(axis=1).min() == 0
