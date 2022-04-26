# new_rom.py
"""Do the hyperparameter search for a new data set (FH-N)."""
import os
import logging
import numpy as np
import multiprocessing as mp

import utils
import fhn


# Adaptive ODE solver tolerances.
TOL = 1e-6


def generate_data(label, numchunks=None):
    dm = utils.FHNDataManager(label)
    training_parameters = dm.training_parameters()

    print("\n**** TRAINING DATA ****", end="\n\n")
    solver = fhn.FHNROMSolver()

    if numchunks is not None:
        for chunk in np.split(training_parameters, numchunks, axis=0):
            solver.add_snapshot_sets(chunk, atol=TOL, rtol=TOL)
            solver.save(dm.solverfile, overwrite=True)
    else:
        solver.add_snapshot_sets(training_parameters, atol=TOL, rtol=TOL)
        solver.save(dm.solverfile, overwrite=True)

    solver._pod_basis(saveas=label)


def main(label, rs, timelimit=12, fromregsfile=False, µ_test=None):
    """Hyperparameter grid search for FH-N OpInf ROM."""
    dm = utils.FHNDataManager(label)
    if not os.path.isfile(dm.solverfile):
        raise FileNotFoundError(f"{dm.solverfile} (run generate_data())")

    solver = fhn.FHNROMSolver.load(dm.solverfile)
    print(f"Loaded training parameters from {dm.solverfile}")
    print(f"Retaining r1 = {rs[0]:d} and r2 = {rs[1]:d} POD modes")

    if fromregsfile:
        λs = np.load(dm.regsfile(rs))
        rom = solver.train_rom(rs, bases=label, trialtimelimit=timelimit,
                               regguess=λs, µ_test=µ_test, atol=TOL, rtol=TOL)
    else:
        rom = solver.train_rom(rs, bases=label, trialtimelimit=timelimit,
                               gridsearch=[np.linspace(-4, 5, 30),
                                           np.linspace(-4, 5, 30)],
                               µ_test=µ_test, atol=TOL, rtol=TOL)
    if not hasattr(rom, "reg"):
        message = f"SEARCH FAILED, NO REGS SAVED ('{label}', rs={rs})"
        print(message)
        logging.info(message)
        return

    # Save the optimal regularization parameters.
    regsfile = dm.regsfile(rs)
    np.save(regsfile, rom.reg)
    print(f"Regularization parameters saved to {regsfile}.")
    logging.info(f"Hyperparameters: {rom.reg}; saved to {regsfile}")

    # Save the optimal ROM.
    romfile = dm.romfile(rs)
    rom.save(romfile, overwrite=True)
    print(f"Learned pOpInf ROM saved to {romfile}")

    return rom


if __name__ == "__main__":
    _label_ = "train"

    generate_data(_label_)

    def _distributed(rs):
        main(_label_, rs, timelimit=350, fromregsfile=False)

    basis_sizes = utils.FHNDataManager(_label_).basis_sizes()
    with mp.Pool(processes=min([len(basis_sizes), mp.cpu_count()])) as pool:
        pool.map(_distributed, basis_sizes)
