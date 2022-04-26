# plot_fhn.py
"""Plot FitzHugh-Nagumo numerical results."""
import os
import re
import h5py
import time
import itertools
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors

import rom_operator_inference as opinf

import config
import utils
import fhn


# Plotting settings ===========================================================

def init_settings():
    """Turn on custom matplotlib settings."""
    plt.rc("figure", figsize=(12,4), dpi=1200)
    plt.rc("axes", titlesize="xx-large", labelsize="xx-large", linewidth=.5)
    plt.rc("xtick", labelsize="large")
    plt.rc("ytick", labelsize="large")
    plt.rc("legend", fontsize="xx-large", frameon=False, edgecolor="none")


# Data generation =============================================================

def _globalize_lock(L):
    global lock
    lock = L


def _printable_param(µ):
    α, β, δ, ε = µ
    return f"[{α:.4f}  {β:.3f}  {δ:.2f}  {ε:.3f}]"


def _get_trialnames(datafile):
    """Get the list of trial names in the specified H5 file."""

    def _sortkey(g):
        out = re.findall(r"\w+?(\d+)", g)
        if not out:
            raise ValueError(f"regex failed for {g}")
        return out[0]

    with h5py.File(datafile, 'r') as hf:
        if "romerrors" in hf:
            hf = hf["romerrors"]
        groups = [g for g in hf if isinstance(hf[g], h5py.Group)]
    return sorted((g for g in groups if "trial" in g), key=_sortkey)


def _generate_test_data_trial(µ_test, datafile, trial_name,
                              parallel=True, chunksize=None):
    """Do one round of things..."""
    with h5py.File(datafile, 'r') as hf:
        if trial_name in hf:
            group = hf[trial_name]
            params_done = group["parameters"][:]
            print(f"{params_done.shape[0]} already found")
            µ_test = np.array([µ for µ in µ_test
                               if not utils.in2Darray(params_done, µ)])
        if len(µ_test) == 0:
            print(f"Group '{trial_name}' complete")
            return

    print(f"Data generation group '{trial_name}'")

    # Evaluate the FOM at each parameter.
    parameters, snapshots, execution_time = [], [], []
    solver = fhn.FHNSolver(**config.FHN_SOLVER_DEFAULTS)
    n = len(µ_test)
    for i, µ in enumerate(µ_test):
        start = time.process_time()
        Ufom_µ = solver.full_order_solve(µ, config.fhn_input)[0]
        elapsed = time.process_time() - start

        if parallel:
            lock.acquire()
        print(f"{trial_name} ({i+1:d}/{n}):",
              f"finished high-fidelity solve in {elapsed:0>6.2f} s at µ =",
              _printable_param(µ), flush=True)
        if parallel:
            lock.release()

        parameters.append(µ)
        snapshots.append(Ufom_µ)
        execution_time.append(elapsed)

        # Update data from this trial.
        if (chunksize and (i+1) % chunksize == 0) or ((i+1) == len(µ_test)):
            if parallel:
                lock.acquire()
            with utils.timed_block(f"{trial_name} saving to {datafile}"):
                with h5py.File(datafile, 'a') as hf:
                    if trial_name in hf:
                        gp = hf[trial_name]
                        parameters = np.vstack([gp["parameters"][:],
                                                parameters])
                        snapshots = np.vstack([gp["snapshots"][:], snapshots])
                        execution_time = np.concatenate([gp["cputime"][:],
                                                         execution_time])
                        del gp
                        del hf[trial_name]
                    group = hf.create_group(trial_name)
                    group.create_dataset("parameters", data=parameters)
                    group.create_dataset("snapshots", data=snapshots)
                    group.create_dataset("cputime", data=execution_time)
            if parallel:
                lock.release()

            parameters, snapshots, execution_time = [], [], []


def generate_test_data(label, train=False, serial=False):
    """Generate full-order FOM trajectories to test the ROM against.
    THIS IS EXPENSIVE AND RESULTS IN A LARGE FILE (i.e., do this ONCE).

    Parameters
    ----------
    label : str
        Label of the data set.
    train : bool
        If True, create training data; if False, create testing data.
    serial : bool
        If True, generate data in parallel.
    """
    dm = utils.FHNDataManager(label)
    parameters = dm.training_trials() if train else dm.testing_trials()
    datafile = dm.trainingdatafile if train else dm.testingdatafile

    # Initialize save file if needed.
    if not os.path.isfile(datafile):
        with h5py.File(datafile, 'w'):
            pass

    # Assign trial names for each parameter group.
    n_trials = len(parameters)

    # Distribute tasks.
    if serial:
        for trial_name, µ_test in parameters.items():
            _generate_test_data_trial(µ_test, datafile, trial_name, False)
    else:
        trial_names, params = zip(*parameters.items())
        args = zip(params,
                   (datafile for _ in range(n_trials)),
                   trial_names,
                   (True for _ in range(n_trials)))
        with mp.Pool(initializer=_globalize_lock, initargs=(mp.Lock(),),
                     processes=min([n_trials, mp.cpu_count()])) as pool:
            pool.starmap(_generate_test_data_trial, args)


def _get_solver_and_rom(label, rs, intrusive, loadsolver=False):
    """Load a FHN solver object and a trained ROM."""
    dm = utils.FHNDataManager(label)

    # Load the solver and basis.
    if loadsolver:
        solver = fhn.FHNROMSolver.load(dm.solverfile)
        print(f"Loaded snapshot data from {dm.solverfile}")
    else:
        solver = fhn.FHNROMSolver(**config.FHN_SOLVER_DEFAULTS)
    V1, svals1 = utils.load_basis(f"{label}_1", rs[0], svdvals=True)
    V2, svals2 = utils.load_basis(f"{label}_2", rs[1], svdvals=True)

    # If it exists, load the pOpInf ROM from file.
    if not intrusive:
        λs = np.load(dm.regsfile(rs))
        romfile = dm.romfile(rs)
        if os.path.isfile(romfile):
            print(f"Loading pOpInf ROM from {romfile}")
            rom = fhn.AffineFHNROM.load(romfile)
            rom.reg = λs
            return solver, rom, (svals1, svals2)

    # Compute reduced-order model.
    bases = (V1[:,:rs[0]], V2[:,:rs[1]])
    if intrusive:
        print("Computing intrusive ROM")
        rom = fhn.AffineFHNROM_Intrusive().fit(bases, solver._A1)
    else:
        print("Computing pOpInf ROM")
        rom = solver.train_rom(rs, bases=bases,
                               ROMClass=fhn.AffineFHNROM, reg=λs)
        rom.save(romfile)

    return solver, rom, (svals1, svals2)


def _generate_plot_data_trial(trial_name, testfile, datafile,
                              label, rs, intrusive, parallel=True):
    with h5py.File(datafile, 'r') as hf:
        if trial_name in hf:
            print(f"Group '{trial_name}' already present")
            return
    solver, rom, [svals1, svals2] = _get_solver_and_rom(label, rs, intrusive)

    proj_errors, rom_errors, execution_time, unstables = [], [], [], []
    print(f"\n**** TRIAL '{trial_name}' ****")
    with h5py.File(testfile, 'r') as hf:
        µ_test = hf[f"{trial_name}/parameters"][:]
    n = len(µ_test)

    V = rom.Vr
    VT = V.T
    for i, µ in enumerate(µ_test):
        # Load the full-order snapshots.
        with h5py.File(testfile, 'r') as hf:
            Ufom_µ = hf[f"{trial_name}/snapshots"][i]
        u0 = Ufom_µ[:,0]

        # Projection Error
        projerr = opinf.post.Lp_error(Ufom_µ, V @ (VT @ Ufom_µ), solver.t)[1]

        # Reduced-order solve.
        start = time.process_time()
        try:
            Urom_µ = rom.predict(µ, u0, solver.t_dense, config.fhn_input)
        except ValueError:
            print("UNSTABLE! Coarsening time mesh...", flush=True)
            unstables.append(µ)
            start = time.process_time()
            try:
                Urom_µ = rom.predict(µ, u0, solver.t, config.fhn_input)
            except ValueError:
                print("STILL UNSTABLE!!", flush=True)
                romerr = 1e6
            else:
                elapsed = time.process_time() - start
                execution_time.append(elapsed)
                romerr = opinf.post.Lp_error(Ufom_µ, Urom_µ, solver.t)[1]
                print(f"done in {elapsed:.2f} s.", flush=True)
        else:
            elapsed = time.process_time() - start
            Urom_µ = Urom_µ[:,::solver.downsample]
            execution_time.append(elapsed)
            romerr = opinf.post.Lp_error(Ufom_µ, Urom_µ, solver.t)[1]
            print(f"done in {elapsed:.2f} s.", flush=True)

        if parallel:
            lock.acquire()
        print(f"{trial_name} ({i+1:d}/{n}): µ = ", _printable_param(µ),
              f"\nProjection error: {projerr:.2%}",
              f"\tROM error: {romerr:.2%}", sep='', flush=True)
        if parallel:
            lock.release()

        # Compute and record relative space-time errors.
        rom_errors.append(romerr)
        proj_errors.append(projerr)

    # Save data from this trial.
    if parallel:
        lock.acquire()
    with utils.timed_block(f"\nSaving {trial_name} to {datafile}"):
        groupname = f"romerrors/{trial_name}"
        with h5py.File(datafile, 'a') as hf:
            if groupname in hf:
                del hf[groupname]
            group = hf.create_group(f"romerrors/{trial_name}")
            group.create_dataset("parameters", data=µ_test)
            group.create_dataset("projection_error", data=proj_errors)
            group.create_dataset("rom_error", data=rom_errors)
            group.create_dataset("cputime", data=execution_time)
            group.create_dataset("unstables", data=unstables)
    if parallel:
        lock.release()


def _generate_romfom_comparison(label, datafile, solver, rom):
    dm = utils.FHNDataManager(label)
    full_comparison_parameters = dm.fullcomparison_parameters()

    print("Evaluating FOM/ROM on experimental parameters")
    fom_snaps, rom_snaps, params = [], [], []
    for µ in full_comparison_parameters:
        print(f"\nµ = {µ}")

        # Full-order solve.
        with utils.timed_block("High-fidelity solve"):
            fom_µ = solver.full_order_solve(µ, config.fhn_input)[0]

        # Reduced-order solve.
        u0 = np.zeros(solver.x.size*2)
        with utils.timed_block("Low-fidelity solve"):
            try:
                rom_µ = rom.predict(µ, u0, solver.t_dense, config.fhn_input)
                rom_µ = rom_µ[:,::solver.downsample]
            except ValueError:
                print(f"UNSTABLE! at µ = {µ}")
            else:
                params.append(µ)
                fom_snaps.append(fom_µ)
                rom_snaps.append(rom_µ)

    # Save full-order data.
    with utils.timed_block(f"\nsaving FOM/ROM data to {datafile}"):
        with h5py.File(datafile, 'a') as hf:
            if "fullcomparison" in hf:
                del hf["fullcomparison"]
            group = hf.create_group("fullcomparison")
            group.create_dataset("parameters", data=params)
            group.create_dataset("fom_snapshots", data=fom_snaps)
            group.create_dataset("rom_snapshots", data=rom_snaps)


def generate_plot_data(label, rs, intrusive=False, trainindices=(0,1,2),
                       train=False, overwrite=False, serial=False):
    """Generate and save all of the data needed to plot the figures.

    Paramters
    ---------
    label : str
        Name of data set / ROM to test. If label="default", and rs=(8,6),
        the following files should exist in the base data folder:
        * fhn_default.h5: FHNSolver training data.
        * bases.h5: Basis file with datasets "default_1" and "default_2".
        * fhnregs_default_r08-06.npy: Regularization parameters for OpInf.
        The results file "fhn_results_default_r08-06.h5" is then created.
    rs : (int,int)
        Size of the POD bases to use.
    intrusive : bool
        If True, use the intrusive ROM instead of the OpInf ROM.
    trainindices : list(int)
        Indices of the FOM training data to save.
    train : bool
        If True, compare to training data; if False, compare to testing data.
    overwrite : bool
        If False and the results file already exists, raise an error.
    """
    dm = utils.FHNDataManager(label)

    # Ensure file with FOM test data exists.
    testfile = dm.trainingdatafile if train else dm.testingdatafile
    if not os.path.isfile(testfile):
        raise FileNotFoundError(f"{testfile} (run generate_test_data() first)")
    print(f"Comparing results to data in {testfile}")

    # Protect against overwriting previous data.
    datafile = dm.resultsfile(train, intrusive, rs)
    if os.path.isfile(datafile) and not overwrite:
        raise FileExistsError(f"{datafile} (use overwrite=True to ignore)")
    print(f"Results will be written to {datafile}")

    # Load full-order training data and POD basis.
    solver, rom, [svals1, svals2] = _get_solver_and_rom(label, rs, intrusive,
                                                        loadsolver=True)

    # Save training data and basis data.
    mask = np.array(trainindices, dtype=np.int)
    with utils.timed_block(f"saving FOM/POD data to {datafile}"):
        with h5py.File(datafile, 'w') as hf:
            # Full-order training data (only snapshots to be plotted).
            group = hf.create_group("trainingdata")
            group.create_dataset("parameters", data=solver.parameters[mask])
            group.create_dataset("snapshots", data=solver.snapshots[mask])

            # ROM data.
            group = hf.create_group("reducedmodel")
            group.create_dataset("basis1", data=rom.Vr1)
            group.create_dataset("basis2", data=rom.Vr2)
            group.create_dataset("svals1", data=svals1)
            group.create_dataset("svals2", data=svals2)

            hf.create_group("romerrors")

    # Evaluate the ROM at the experiment parameters.
    trialnames = _get_trialnames(testfile)
    if serial:
        for trialname in trialnames:
            _generate_plot_data_trial(trialname, testfile, datafile, label,
                                      rs, intrusive, parallel=not serial)
    else:
        n_trials = len(trialnames)
        args = zip(trialnames,
                   (testfile for _ in range(n_trials)),
                   (datafile for _ in range(n_trials)),
                   (label for _ in range(n_trials)),
                   (rs for _ in range(n_trials)),
                   (intrusive for _ in range(n_trials)),
                   (True for _ in range(n_trials)))
        with mp.Pool(initializer=_globalize_lock, initargs=(mp.Lock(),),
                     processes=min([n_trials, mp.cpu_count()])) as pool:
            pool.starmap(_generate_plot_data_trial, args)

    # Evaluate ROM and FOM at select test parameters.
    _generate_romfom_comparison(label, datafile, solver, rom)


def process_multi(label, residual_energies, basis_sizes,
                  train=False, quant=.25):
    """Extract statistical data from multiple results files."""
    if len(residual_energies) != len(basis_sizes):
        raise ValueError("len(residual_energies) != len(basis_sizes)")

    dm = utils.FHNDataManager(label)
    opinf_errors, intrusive_errors, projection_errors = [], [], []
    if not train:
        _training = dm.training_parameters()
        _sensitive = np.load("bad_params_30.npy")

    print(f"Processing {'train' if train else 'test'}ing results")
    for rom_errors, intrusive in [(opinf_errors, False),
                                  (intrusive_errors, True),
                                  (projection_errors, True)]:
        errlabel = "rom_error"
        if rom_errors is projection_errors:
            errlabel = "projection_error"
        for rs in basis_sizes:
            datafilepath = dm.resultsfile(train, intrusive, rs)
            if not train:
                _to_stack = [_training, _sensitive]
                ignores = np.unique(np.vstack(_to_stack), axis=0)
                print(f"rs = {rs}: ignoring {ignores.shape[0]} points")

            with h5py.File(datafilepath, 'r') as hf:
                r1 = hf["reducedmodel/basis1"].shape[1]
                r2 = hf["reducedmodel/basis2"].shape[1]
                assert rs[0] == r1
                assert rs[1] == r2

                # ROM relative errors and execution times.
                errors = []
                for trial_name in sorted(hf["romerrors"],
                                         key=lambda s: int(s[5:])):
                    mask = slice(None)
                    if not train:
                        µ_test = hf[f"romerrors/{trial_name}/parameters"][:]
                        mask = np.ones(µ_test.shape[0], dtype=np.bool)
                        for i,µ in enumerate(µ_test):
                            if utils.in2Darray(ignores, µ):
                                mask[i] = False
                    errs = hf[f"romerrors/{trial_name}/{errlabel}"][:]
                    errors += errs[mask].tolist()
            rom_errors.append(errors)
            print(f"rs = {rs}: {len(errors)} parameters checked")

        # Calculate error statistics.
        mins, q1s, medians, q3s, maxs = [], [], [], [], []
        max_stables, geomeans, arithmeans = [], [], []

        for err in rom_errors:
            e = np.array(err)
            quantiles = np.quantile(e, [0, quant, .5, 1 - quant, 1])
            mins.append(quantiles[0])
            q1s.append(quantiles[1])
            medians.append(quantiles[2])
            q3s.append(quantiles[3])
            maxs.append(quantiles[4])

            estab = e[e < 1e6]
            max_stables.append(np.max(estab))
            geomeans.append(np.exp(np.mean(np.log(estab))))
            arithmeans.append(np.mean(estab))

        # Save error statistics.
        with h5py.File(dm.multiplotfile, 'a') as hf:
            if "residual_energies" not in hf:
                hf.create_dataset("residual_energies", data=residual_energies)
            if not np.allclose(residual_energies, hf["residual_energies"][:]):
                raise ValueError("residual_energies "
                                 "does not match existing file")

            if "basis_sizes" not in hf:
                hf.create_dataset("basis_sizes", data=basis_sizes)
            if not np.allclose(basis_sizes, hf["basis_sizes"][:]):
                raise ValueError("basis_sizes does not match existing file")

            name = "train" if train else "test"
            if "projection" in errlabel:
                name = f"{name}_projection"
            else:
                name = f"{name}_intrusive" if intrusive else f"{name}_opinf"
            if name in hf:
                del hf[name]
            group = hf.create_group(name)

            group.create_dataset("mins", data=mins)
            group.create_dataset("q1s", data=q1s)
            group.create_dataset("medians", data=medians)
            group.create_dataset("q3s", data=q3s)
            group.create_dataset("maxs", data=maxs)

            group.create_dataset("max_stables", data=max_stables)
            group.create_dataset("geomeans", data=geomeans)
            group.create_dataset("arithmeans", data=arithmeans)


# Plotting routines ===========================================================

def _param_labels(params, sep=''):
    return ", ".join([fr"$\alpha={params[0]:.3f}$",
                      fr"$\beta={params[1]:.2f}$",
                      fr"{sep}$\gamma={params[2]:.2f}$",
                      fr"$\varepsilon={params[3]:.3f}$"])


def plot_svdvals(svalss, rs, rmax, level=1e-7):
    """Plot residual energy of singular values."""
    fig, ax = plt.subplots(1, 1, figsize=(6,4))
    indices = np.arange(1,rmax+1)
    resid = []
    for i, [svdvals, r] in enumerate(zip(svalss, rs)):
        resid.append(1 - np.cumsum(svdvals[:rmax]**2)/np.sum(svdvals**2))
        ax.axvline(r, lw=.5, c=f"C{i:d}")
        ax.semilogy(indices, resid[i], '.-', mew=0, ms=10, lw=1,
                    label=rf"$u_{{{i+1:d}}}$", zorder=10)
    ax.axhline(level, lw=.5, c="gray")

    ax.text(4, resid[0][2], r"$u_{1}$", color="C0", ha="center", va="bottom")
    ax.text(3, resid[1][3], r"$u_{2}$", color="C1", ha="center", va="center")
    ax.text(rs[0]+.5, 2e-14, fr"$r_{{1}} = {rs[0]}$",
            color="C0", fontsize="large", ha="left", va="bottom")
    ax.text(rs[1]-.5, 2e-14, fr"$r_{{2}} = {rs[1]}$",
            color="C1", fontsize="large", ha="right", va="bottom")
    ax.text(rmax-.5, 2.5*level,
            fr"$1-\mathcal{{E}}(r_{{\ell}})=10^{{{int(np.log10(level))}}}$",
            ha="right", color="gray", fontsize="large")

    ax.set_xlim(1, rmax)
    if rmax == 20:
        ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_yticks([1e-14, 1e-11, 1e-8, 1e-5, 1e-2]),
    ax.set_ylim(1e-14, 1e-2)
    ax.set_xlabel(r"Number of retained modes")
    ax.set_ylabel(r"Energy of residual modes")


def plot_phase(ax, µ, u, lw=1.5, freq=10, isromcompare=False):
    """Plot phase portrait."""
    n = u.shape[0] // 2
    locs = np.logspace(0, np.log10(n-1), 10, dtype=np.int)
    color = plt.cm.viridis_r(np.linspace(.2, 1, len(locs)))
    for i, c in zip(reversed(locs), reversed(color)):
        u1 = u[i,:]
        u2 = u[i+n,:]
        if isromcompare:
            ax.plot(u1, u2, 'k--', lw=1, alpha=.7)
        else:
            ax.plot(u1, u2, '-', color=c, lw=lw)
    ax.set_xlabel(r"$u_{1}(x_{j},t)$")
    ax.set_ylabel(r"$u_{2}(x_{j},t)$")
    ax.set_xlim(-.4, 1.6)
    ax.set_ylim(0, .23)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, .1, .2])
    ax.set_title(_param_labels(µ, '\n'), fontsize="x-large")


def plot_phase_gallery(parameters, snapshots, romcompare=None):
    N = len(parameters)
    fig, axes = plt.subplots(1, N, sharey=True, figsize=(4*N,3))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for µ, U, ax in zip(parameters, snapshots, axes):
        plot_phase(ax, µ, U, lw=(2.5 if romcompare is not None else 2))
    if romcompare is not None:
        for µ, U, ax in zip(parameters, romcompare, axes):
            plot_phase(ax, µ, U, isromcompare=True)
    if len(axes) > 1:
        for ax in axes[1:]:
            ax.set_ylabel("")
    fig.subplots_adjust(wspace=.05)
    xc = np.linspace(0.2, 1, 400)
    cmap = mplcolors.LinearSegmentedColormap.from_list("viridis_r_short",
                                                       plt.cm.viridis_r(xc))
    mappable = plt.cm.ScalarMappable(norm=mplcolors.Normalize(vmin=0, vmax=1),
                                     cmap=cmap)
    cbar = fig.colorbar(mappable, ax=axes, pad=0.015)
    cbar.set_ticks([0, .25, .5, .75, 1])
    cbar.set_label(r"spatial coordinate $x$")

    if romcompare is not None:
        leg = axes[-1].legend(["FOM", "ROM"], ncol=2,
                              loc="upper right", fontsize="large")
        lines = leg.get_lines()
        lines[0].set_c(plt.cm.viridis_r(.275))
        lines[0].set_linewidth(3)
        lines[1].set_c('k')
        lines[1].set_linestyle('--')
        lines[1].set_linewidth(2)


def plot_spacetime(XX, TT, u, µ):
    u1, u2 = np.split(u, 2, axis=0)
    if XX.ndim == 1:
        XX, TT = np.meshgrid(XX, TT)

    fig, [ax1, ax2] = plt.subplots(1, 2)
    c1 = ax1.pcolormesh(XX, TT, u1.T, shading="nearest",
                        vmin=-.5, vmax=1.75, cmap="viridis")
    c2 = ax2.pcolormesh(XX, TT, u2.T, shading="nearest",
                        vmin=0, vmax=.25, cmap="cividis")

    for ax in [ax1, ax2]:
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$t$")
    ax1.set_title(r"$u_{1}(x,t)$")
    ax2.set_title(r"$u_{2}(x,t)$")

    fig.colorbar(c1, ax=ax1, pad=.05)
    fig.colorbar(c2, ax=ax2, pad=.05)
    fig.subplots_adjust(wspace=.1)


def plot_errors(epsrange, errors, anchors):
    """Plot relative space-time errors of the ROM."""
    fig, ax = plt.subplots(1, 1, figsize=(12,5.5))
    lss = itertools.cycle(['-', '--', '-.', ':'])
    for i in range(len(errors)):
        µ = anchors[i]
        ax.semilogy(epsrange, errors[i], ls=next(lss), c=f"C{i:d}", lw=3,
                    label=fr"$\mu = ({µ[0]:.3f}, {µ[1]:.2f}, {µ[2]:.0f},"
                          r" \varepsilon)$")

    # Set limits and labels.
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel("Relative error")
    ax.set_xticks([0.01, 0.02, 0.03, 0.04])
    ax.set_xlim(0, 0.040)
    ax.set_yticks([1e-6, 1e-4, 1e-2, 1e0])
    ax.grid()

    # Legend to the right of the plot.
    fig.subplots_adjust(right=.725)
    leg = ax.legend(loc="center right", ncol=1,
                    bbox_to_anchor=(1,.5), bbox_transform=fig.transFigure)
    for line in leg.get_lines():
        line.set_linewidth(3)


def plot_fomrom_u1u2(t, Ufom, Urom, µ, freq=10):
    fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(6,4))
    n = Ufom.shape[0] // 2
    locs = range(0, n, freq)
    color = plt.cm.viridis(np.linspace(.5, 1, len(locs)))
    for i, c in zip(reversed(locs), reversed(color)):
        u1 = Urom[i,:]
        u2 = Urom[i+n,:]
        ax1.plot(t, u1, '-', color=c, lw=2.5)
        ax2.plot(t, u2, '-', color=c, lw=2.5)
        u1 = Ufom[i,:]
        u2 = Ufom[i+n,:]
        ax1.plot(t, u1, '--', color='k', lw=1)
        ax2.plot(t, u2, '--', color='k', lw=1)

    ax1.set_ylabel(r"$u_{1}(x,t)$")
    ax2.set_ylabel(r"$u_{2}(x,t)$")
    ax1.set_ylim(-.5, 1.75)
    ax2.set_ylim(0, .25)
    for ax in [ax1, ax2]:
        ax.set_xlim(0, 1)
        ax.set_xticks([0, .5, 1])
    ax1.set_title(_param_labels(µ))


def plot_size_vs_error(residual_energies, basis_sizes,
                       train_opinf, test_opinf,
                       train_intrusive, test_intrusive,
                       train_projection=None, test_projection=None):
    """Plot residual energy vs geometric mean of relative errors.

    Parameters
    ----------
    residual_energies : (m,) ndarray
        Residual energy levels used to pick the basis sizes.
    basis_sizes : (m,2) ndarary
        Basis sizes for each model, (r1, r2).
    train_opinf : list(3 ndarrays)
        Relative errors on the training set for each OpInf ROM.
    test_opinf : list(3 ndarrays)
        Relative errors on the testing set for each OpInf ROM.
    train_intrusive : list(3 ndarrays)
        Relative errors on the training set for each intrusive ROM.
    test_intrusive : list(3 ndarrays)
        Relative errors on the testing set for each intrusive ROM.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12,4), sharex=True)

    for datasets, ls, ms, ax in [
        (train_projection, "C3s-.", 4, axes[0]),
        (test_projection, "C3s-.", 4, axes[1]),
        (train_intrusive, "C5d--", 5, axes[0]),
        (test_intrusive, "C5d--", 5, axes[1]),
        (train_opinf, "C0.-", 10, axes[0]),
        (test_opinf, "C0.-", 10, axes[1]),
    ]:
        if datasets is None:
            continue
        low, q1, mid, q3, high = datasets

        if not ls.startswith("C3"):
            ax.loglog(residual_energies, mid, ls, ms=ms, lw=1)
            ax.fill_between(residual_energies, q1, q3,
                            color=ls[:2], alpha=.2, lw=0)
        else:
            ax.loglog(residual_energies, low, ls, marker=None, lw=1)
            ax.loglog(residual_energies, mid, ls, marker=None, lw=1)
        ax.loglog(residual_energies, high, ls, marker=None, lw=1)

    axes[0].text(8e-5, 3e-3, "pOpInf", color="C0", ha="right", va="center")
    axes[0].text(8e-5, 2e-2, "Intrusive", color="C5", ha="left", va="center")
    axes[1].text(8e-5, 3e-3, "pOpInf", color="C0", ha="right", va="center")
    axes[1].text(8e-5, 2e-2, "Intrusive", color="C5", ha="left", va="center")

    for ax in axes:
        ax.grid(color="gray", lw=.5, axis="y")
        ax.set_xticks(residual_energies[::2])
        ax.set_xlim(residual_energies[0], residual_energies[-1])
        ax.set_xlabel(r"Residual energy $1 - \mathcal{E}(r_{\ell})$")
        ax.set_yticks([1e-6, 1e-4, 1e-2])
        ax.set_ylim(1e-7, 2)
    axes[0].set_yticklabels([fr"${p}\%$"
                             for p in ["0.0001", "0.01", "1"]])
    axes[0].set_ylabel("ROM relative error")
    axes[1].set_yticklabels([])

    axes[0].set_title("Training set")
    axes[1].set_title("Testing set")

    fig.subplots_adjust(hspace=.025, wspace=.025)

    return fig, axes


def plot_train_test_param_space(label):
    if label == "alphaepsilon":
        train_params = np.array(list(itertools.product(
            [.025, .050, .075],
            [.001, .003, .005, .010, .020, .030, .040])))
        test_params = np.array(list(itertools.product(
            [.0250, .0375, .0500, .0625, .0750],
            np.round(np.concatenate([np.arange(.001, .011, .001),
                                     np.arange(.012, .042, .002)]), 3))))
    else:
        raise ValueError(label)

    fig, ax = plt.subplots(1, 1)
    ax.plot(test_params[:,1], test_params[:,0], 'C1o', mew=0, ms=8)
    ax.plot(train_params[:,1], train_params[:,0], 'k*', mew=0, ms=12)
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel(r"$\alpha$")


# Main routine ================================================================

def main(label, rs):
    """Plot and save all figures."""
    dm = utils.FHNDataManager(label)

    # Ensure required data exists ---------------------------------------------
    datafile = dm.resultsfile(False, False, rs)
    if not os.path.isfile(datafile):
        raise FileNotFoundError(f"{datafile} (call generate_plot_data()")

    # Load data ---------------------------------------------------------------
    fullcomparison = False
    with h5py.File(datafile, 'r') as hf:
        # Full-order training data.
        training_parameters = hf["trainingdata/parameters"][:]
        training_snapshots = hf["trainingdata/snapshots"][:]

        # POD basis metadata.
        svals1 = hf["reducedmodel/svals1"][:]
        svals2 = hf["reducedmodel/svals2"][:]
        r1 = hf["reducedmodel/basis1"].shape[1]
        r2 = hf["reducedmodel/basis2"].shape[1]
        assert rs[0] == r1
        assert rs[1] == r2

        # FOM/ROM full comparisons.
        if "fullcomparison" in hf:
            fullcomparison = True
            group = hf["fullcomparison"]
            fom_snaps = group["fom_snapshots"][:]
            rom_snaps = group["rom_snapshots"][:]
            snap_params = group["parameters"][:]

    # Generate plots ----------------------------------------------------------
    init_settings()

    # Sample snapshot phase portraits.
    plot_phase_gallery(training_parameters, training_snapshots)
    utils.save_figure("fhn_training_snapshots.pdf")

    if len(training_parameters) > 3:
        plot_phase_gallery(training_parameters[-1:], training_snapshots[-1:])
        utils.save_figure("fhn_training_supplement.pdf")

    # Singular value decay.
    plot_svdvals((svals1, svals2), (r1, r2), rmax=25, level=1e-7)
    utils.save_figure("fhn_svdval_decay.pdf")

    # FOM and ROM solutions together.
    if fullcomparison:
        plot_phase_gallery(snap_params, fom_snaps, rom_snaps)
        utils.save_figure("fhn_fomromphase.pdf")


def main_multi(label):
    dm = utils.FHNDataManager(label)
    train_opinf, test_opinf, train_intrusive, test_intrusive = [], [], [], []
    train_projection, test_projection = [], []

    with h5py.File(dm.multiplotfile, 'r') as hf:
        residual_energies = hf["residual_energies"][:]
        basis_sizes = hf["basis_sizes"][:]

        for name, data in [
            ("train_intrusive", train_intrusive),
            ("test_intrusive", test_intrusive),
            ("train_opinf", train_opinf),
            ("test_opinf", test_opinf),
            ("train_projection", train_projection),
            ("test_projection", test_projection),
        ]:
            group = hf[name]
            for dset in ["mins", "q1s", "medians", "q3s", "max_stables"]:
                data.append(group[dset][:])

    init_settings()
    fig, axes = plot_size_vs_error(residual_energies, basis_sizes,
                                   train_opinf, test_opinf,
                                   train_intrusive, test_intrusive,
                                   None, None)

    utils.save_figure(f"fhn_basisVerror_{label}.pdf")


# =============================================================================

if __name__ == "__main__":
    pass

    # label = "train"
    # basis_sizes = [
    #     (3, 2),
    #     (4, 4),
    #     (7, 5),
    #     (9, 7),
    #     (12, 9),
    #     (14, 11),
    #     (16, 13),
    #     (19, 15),
    #     (21, 17),
    #     (24, 19),
    # ]
    # residual_energies = [float(f"1e-{i:d}") for i in range(3, 13)]
    # particular_rs = (12, 9)
    # trainindices = [138, 293, 457]

    # -----------------------------------------------------------------------

    # Generate all results.
    # for train in [True, False]:
    #     generate_test_data(label, train=train, serial=False)
    #     print(f"\nDONE WITH {'TRAINING' if train else 'TESTING'} DATA\n")
    #     for rs in basis_sizes:
    #         for intrusive in [False]:
    #             generate_plot_data(label=label, rs=rs,
    #                                trainindices=trainindices,
    #                                train=train, intrusive=intrusive,
    #                                overwrite=True, serial=False)
    #     process_multi(label, residual_energies, basis_sizes, train, .10)

    # Generate specific results.
    # generate_plot_data(label=label, rs=particular_rs,
    #                    trainindices=trainindices, overwrite=True,
    #                    train=False, intrusive=False)
    # dm = utils.FHNDataManager(label)
    # dfile = dm.resultsfile(False, False, particular_rs)
    # solver, rom, _ = _get_solver_and_rom(label, particular_rs, False)
    # _generate_romfom_comparison(label, dfile, solver, rom)

    # Make plots.
    # main_multi(label)
    # main(label, particular_rs)
