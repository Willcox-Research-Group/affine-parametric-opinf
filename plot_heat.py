# plot_heat.py
"""Plot heat equation numerical results."""

import os
import glob
import h5py
import time
import tqdm
import itertools
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import matplotlib.colors as mplcolors

import rom_operator_inference as opinf

import config
import utils
import heat


# Base folder for all data.
BASE_FOLDER = os.path.join(config.BASE_FOLDER, "heat")
if not os.path.isdir(BASE_FOLDER):
    os.mkdir(BASE_FOLDER)


# Plotting settings ===========================================================

def init_settings():
    """Turn on custom matplotlib settings."""
    plt.rc("figure", figsize=(12,4))
    plt.rc("axes", titlesize="xx-large", labelsize="xx-large", linewidth=.5)
    plt.rc("xtick", labelsize="large")
    plt.rc("ytick", labelsize="large")
    plt.rc("legend", fontsize="xx-large", frameon=False, edgecolor="none")


# Data generation =============================================================

def generate_test_data(nx=1000, nt=1500, tf=1.5, xbar=2/3, gridsize=40):
    """Generate full-order FOM trajectories to test the ROM against.
    THIS IS EXPENSIVE AND RESULTS IN A LARGE FILE (i.e., do this ONCE).
    """
    datafile = os.path.join(BASE_FOLDER, "heat_testdata.h5")
    solver = heat.HomogeneousHeatROM(nx=nx, nt=nt, tf=tf, xbar=xbar)
    u0 = solver.initial_conditions()

    # Full-order model results for each test parameter.
    parameters, snapshots, execution_time = [], [], []
    µ_test = np.linspace(.1, 2.5, gridsize)
    total = gridsize**2

    for i,µ in enumerate(itertools.product(µ_test, µ_test)):

        start = time.time()
        U_fom = solver.full_order_solve(µ, u0)[0]
        elapsed = time.time() - start

        print(f"\rµ = ({µ[0]:.3f}, {µ[1]:.3f}):",
              f"done in {elapsed:.2f} s ({i+1:0>4d}/{total})", end='')

        parameters.append(µ)
        snapshots.append(U_fom)
        execution_time.append(elapsed)
    print()

    # Save data.
    with utils.timed_block(f"saving data to {datafile}"):
        with h5py.File(datafile, 'w') as hf:
            # Domain / solver info.
            group = hf.create_group("domain")
            group.create_dataset("nx", data=[nx])
            group.create_dataset("nt", data=[nt])
            group.create_dataset("tf", data=[tf])
            group.create_dataset("xbar", data=[xbar])

            # Full-order data.
            hf.create_dataset("parameters", data=parameters)
            hf.create_dataset("snapshots", data=snapshots)
            hf.create_dataset("cputime", data=execution_time)


def _romtrial(solver, µ, Ufom, opinf_rom, intru_rom):
    u0 = Ufom[:,0]

    # Projection error.
    time_domain = solver.t
    U_proj = opinf_rom.Vr @ (opinf_rom.Vr.T @ Ufom)
    error_proj = opinf.post.Lp_error(Ufom, U_proj, time_domain)[1]

    # Reduced-order solve (OpInf).
    try:
        start = time.time()
        Urom_opinf = solver.predict(opinf_rom, µ, u0)
        elapsed = time.time() - start
        error_opinf = opinf.post.Lp_error(Ufom, Urom_opinf, time_domain)[1]
        time_rom = elapsed
    except ValueError as e:
        print(e)
        error_opinf = 1e10

    # Reduced-order solve (intrusive).
    try:
        Urom_intru = solver.predict(intru_rom, µ, u0)
        error_intru = opinf.post.Lp_error(Ufom, Urom_intru, time_domain)[1]
    except ValueError as e:
        print(e)
        error_intru = 1e10

    return error_proj, error_opinf, error_intru, time_rom, Urom_opinf


def generate_plot_data(r, s=5, overwrite=False):
    """Generate and save all of the data needed to plot the figures."""

    # Ensure file with FOM test data exists.
    testfile = os.path.join(BASE_FOLDER, "heat_testdata.h5")
    if not os.path.isfile(testfile):
        raise FileNotFoundError(f"{testfile} (run generate_test_data() first)")
    print(f"Comparing to test data in {testfile}")

    # Protect against overwriting previous data.
    datafile = os.path.join(BASE_FOLDER, f"heat_results_r{r:0>2d}.h5")
    if os.path.isfile(datafile) and not overwrite:
        raise FileExistsError(f"{datafile} (use overwrite=True to ignore)")
    print(f"Results will be written to {datafile}")

    # Load data from the test file to be copied to the results file.
    with h5py.File(testfile, 'r') as hf:
        nx = hf["domain/nx"][0]
        nt = hf["domain/nt"][0]
        tf = hf["domain/tf"][0]
        xbar = hf["domain/xbar"][0]

        time_fom = hf["cputime"][:]
        µ_test = hf["parameters"][:]

    # Compute full-order training data and POD basis.
    solver = heat.HomogeneousHeatROM(nx=nx, nt=nt, tf=tf, xbar=xbar)
    µ_train = heat.arc_parameters(s, std=0)
    for µ in µ_train:
        solver.add_snapshot_set(µ)
    V, svdvals = solver._pod_basis()
    rom = opinf.AffineInferredContinuousROM("A")
    snapshots = solver.snapshots

    # Data matrix ranks / condition numbers as a function of r.
    Us_ = [V.T @ U for U in snapshots]
    ds, conds_r, ranks_r = [], [], []
    rs = np.arange(1, min(3*r, 100))
    for rr in tqdm.tqdm(rs):
        Usr_ = [U_[:rr,:] for U_ in Us_]
        D = rom._assemble_data_matrix(solver.parameters, solver._affines,
                                      Usr_, None)
        ds.append(D.shape[1])
        conds_r.append(np.linalg.cond(D))
        ranks_r.append(np.linalg.matrix_rank(D))

    # Data matrix ranks / condition numbers as a function of s.
    ii = np.random.permutation(s)                 # Randomize sample order.
    params = solver.parameters[ii]
    Us = snapshots[ii]
    # Us_ = [Us_[i][:r,:] for i in ii]
    conds_s, ranks_s = [], []
    ss = np.arange(1, len(Us)+1)
    for s in tqdm.tqdm(ss):
        Vr = la.svd(np.hstack(Us[:s]))[0][:,:r]   # POD basis matrix.
        Us_ = [Vr.T @ U for U in Us[:s]]          # Projected snapshots.
        D = rom._assemble_data_matrix(params[:s], solver._affines,
                                      Us_, None)  # Data matrix D.
        conds_s.append(np.linalg.cond(D))         # Condition number of D.
        ranks_s.append(np.linalg.matrix_rank(D))  # Rank of D.

    # Learn the OpInf ROM and calculate the intrusive ROM.
    opinf_rom = solver.train_rom(r)
    A1, A2 = np.zeros((solver.n, solver.n)), np.zeros((solver.n, solver.n))
    A1[1:-1,1:-1] = solver._A1.toarray()
    A2[1:-1,1:-1] = solver._A2.toarray()
    intru_rom = opinf.AffineIntrusiveContinuousROM("A")
    intru_rom.fit(opinf_rom.Vr,
                  heat.HomogeneousHeatROM._affines, dict(A=[A1, A2]))

    # Training errors.
    err_proj_train, err_opinf_train, err_intru_train = [], [], []
    time_train, reconstruct_train = [], []
    print("\nTraining data")
    for i,µ in enumerate(solver.parameters):
        Ufom = snapshots[i]
        results = _romtrial(solver, µ, Ufom, opinf_rom, intru_rom)
        err_proj_train.append(results[0])
        err_opinf_train.append(results[1])
        err_intru_train.append(results[2])
        time_train.append(results[3])
        reconstruct_train.append(results[4])
        print(f"\rµ = ({µ[0]:.3f}, {µ[1]:.3f}):",
              f"done in {results[3]:.2f} s ({i+1:d}/{s:d})",
              end='', flush=True)

    # Testing errors.
    err_proj_test, err_opinf_test, err_intru_test, time_test = [], [], [], []
    print("\nTesting data")
    with h5py.File(testfile, 'r') as hf:
        for i,µ in enumerate(µ_test):
            Ufom = hf["snapshots"][i]
            results = _romtrial(solver, µ, Ufom, opinf_rom, intru_rom)
            err_proj_test.append(results[0])
            err_opinf_test.append(results[1])
            err_intru_test.append(results[2])
            time_test.append(results[3])
            print(f"\rµ = ({µ[0]:.3f}, {µ[1]:.3f}):",
                  f"done in {results[3]:.2f} s ({i+1:d}/{µ_test.shape[0]:d})",
                  end='', flush=True)
    print()

    # Save data.
    with utils.timed_block(f"saving data to {datafile}"):
        with h5py.File(datafile, 'w') as hf:
            group = hf.create_group("domain")
            group.create_dataset("x", data=solver.x)
            group.create_dataset("t", data=solver.t)
            group.create_dataset("xbar", data=[xbar])
            group.create_dataset("fom_cputimes", data=time_fom)

            group = hf.create_group("reducedmodel")
            group.create_dataset("r", data=[r])
            group.create_dataset("basis", data=V[:,:2*r])
            group.create_dataset("svdvals", data=svdvals)
            group.create_dataset("conds_r", data=conds_r)
            group.create_dataset("ranks_r", data=ranks_r)
            group.create_dataset("conds_s", data=conds_s)
            group.create_dataset("ranks_s", data=ranks_s)
            group.create_dataset("regularization", data=[opinf_rom.reg])

            group = hf.create_group("training")
            group.create_dataset("parameters", data=solver.parameters)
            group.create_dataset("snapshots", data=snapshots)
            group.create_dataset("reconstruction", data=reconstruct_train)
            group.create_dataset("error_proj", data=err_proj_train)
            group.create_dataset("error_opinf", data=err_opinf_train)
            group.create_dataset("error_intru", data=err_intru_train)
            group.create_dataset("rom_cputime", data=time_train)

            group = hf.create_group("testing")
            group.create_dataset("parameters", data=µ_test)
            group.create_dataset("error_proj", data=err_proj_test)
            group.create_dataset("error_opinf", data=err_opinf_test)
            group.create_dataset("error_intru", data=err_intru_test)
            group.create_dataset("rom_cputime", data=time_test)


# Plotting routines ===========================================================

def plot_parameter_samples(ax, parameters):
    """Plot parameter samples in parameter space."""
    color = iter(plt.cm.Spectral(np.linspace(.1, 1, len(parameters))))
    if len(parameters) == 5:
        color = iter(["C3", "C1", "C2", "C0", "C4"])
    for i, (alpha, beta) in enumerate(parameters):
        ax.plot([alpha],[beta], '*', ms=12, color=next(color), mew=.5, mec='k',
                label=fr"$(\alpha_{i+1:d},\beta_{i+1:d}) "
                      fr"= ({alpha:.2f},{beta:.2f})$")

    ax.set_xlim(.01, 2.5)
    ax.set_ylim(.01, 2.5)
    ax.set_xticks([.5, 1, 1.5, 2])
    ax.set_yticks([.5, 1, 1.5, 2])
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\beta$")
    ax.set_title("Parameter samples")


def plot_snapshot_samples(ax, x, parameters, snapshots, tindex, xbar):
    """Plot snapshot samples at the given time index."""
    color = plt.cm.Spectral(np.linspace(.1, 1, len(parameters)))
    if len(parameters) == 5:
        color = ["C3", "C1", "C2", "C0", "C4"]
    for U, (alpha,beta), col in zip(snapshots, parameters, color):
        ax.plot(x, U[:,tindex], color=col, lw=1.5,
                label=fr"$(\alpha,\beta) = ({alpha:.2f},{beta:.2f})$")
    ax.plot(x, snapshots[0,:,0], lw=1.5, color='k', label=r"$u_{0}$")
    ax.axvline(xbar, c="k", ls="--", lw=1, label=r"$\bar{x}$")

    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, 1.05)
    ax.set_xticks([0, .2, .4, .6, .8, 1])
    ax.set_yticks([0, .2, .4, .6, .8, 1])
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(fr"$u(x,t=t_{{{tindex:d}}})$")
    ax.set_title("Example training snapshots")
    ax.text(.68, .1, r"$x = \bar{x}$", fontsize="large")


def plot_training_samples(µ, U, x, xbar):
    """Figure 1 (out of date, need to do annotations)."""
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(12,4.75),
                                   gridspec_kw={'width_ratios': [1, 2]})
    plot_parameter_samples(ax1, µ)
    plot_snapshot_samples(ax2, x, µ, U, 75, xbar)
    fig.subplots_adjust(bottom=0.3, wspace=.25)
    handles, _ = ax1.get_legend_handles_labels()
    handles.append(plt.Line2D([],[], ls='-', lw=1.5, color='k',
                              label=r"Initial condition $u_0$"))
    s = µ.shape[0]
    ncol = 4 if s == 3 else min(s//2 + 1, 4)
    leg = ax1.legend(handles=handles, loc="lower center", ncol=ncol,
                     bbox_to_anchor=(0.5, 0), bbox_transform=fig.transFigure)
    for line in leg.get_lines():
        line.set_linewidth(2.5)
        if line.get_linestyle() not in ['-', '--']:
            line.set_linestyle("-")


def plot_svdvals(ax, svdvals, r, rmax, level=1e-12):
    """Plot residual energy of singular values."""
    residual_energy = 1 - np.cumsum(svdvals[:rmax]**2)/np.sum(svdvals**2)
    rs = np.arange(1, rmax+1)

    ax.axhline(level, lw=.5, c="gray")
    ax.axvline(r, lw=.5, c="C0")
    ax.semilogy(rs, residual_energy, '.-', mew=0, ms=10, lw=1)

    ax.set_xlim(1, rmax)
    ax.set_yticks([1e-14, 1e-10, 1e-6, 1e-2]),
    ax.set_xlabel(r"Number of retained modes")
    ax.set_ylabel(r"Energy of residual modes")
    ax.text(r-.5, 2.5*ax.get_ylim()[0], fr"$r = {r}$",
            fontsize="large", color="C0", ha="right", va="bottom")
    ax.text(2, 2.5*level, r"$1 - \mathcal{E}(r) = 10^{-12}$",
            fontsize="large", color="gray", ha="left", va="bottom")


def plot_basis(ax, x, V, xbar, rmax):
    """Plot the first rmax basis vectors."""
    ax.axhline(0, c="k", lw=1)
    linestyle = itertools.cycle(["-", "--", "-.", ":",])
    for j in range(rmax):
        ax.plot(x, V[:,j], ls=next(linestyle), label=fr"$v_{{{j+1}}}(x)$")
    ax.axvline(xbar, c="k", ls="--", lw=1)     # , label=r"$\bar{x}$")

    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(-.125, .125)
    ax.set_xticks([0, .2, .4, .6, .8, 1])
    ax.set_yticks([-.1, 0, .1])
    ax.set_xlabel(r"$x$")
    ax.set_ylabel("Dominant POD basis functions")
    ax.legend(loc="upper left", ncol=2,
              frameon=False, edgecolor="none", fontsize="large")
    ax.text(.68, -.1, r"$x = \bar{x}$", fontsize="large")


def plot_basischoice(r, svdvals, V, x, xbar):
    fig, [ax1, ax2] = plt.subplots(1, 2)
    plot_svdvals(ax1, svdvals, r, 35)
    plot_basis(ax2, x, V, xbar, 6)
    fig.subplots_adjust(wspace=.25)


def plot_datacond_r(ax, conds_r, r, rmax):
    """Plot the data matrix condition number as a function of basis size."""
    rs = np.arange(1, len(conds_r)+1)

    ax.axhline(conds_r[r-1], lw=.5, c="gray")
    ax.axvline(r, lw=.5, c="gray")
    ax.semilogy(rs, conds_r, '.-', color="C3", mew=0, ms=10, lw=1)

    ax.set_xlim(rs[0], rmax)
    ax.set_ylim(1, 5e15)
    ax.set_yticks([1e2, 1e6, 1e10, 1e14])
    ax.set_xlabel(r"POD basis size $r$")
    ax.set_ylabel(r"$\kappa(\mathbf{D})$")
    ax.set_title(r"Condition number of data matrix")
    ax.text(r+.75, 1e3, fr"$r = {r}$", fontsize="large", color="gray")


def plot_datarank_r(ax, ranks_r, r, rmax):
    """Plot the numerical rank of the data matrix as a function of basis size.
    """
    qs = 2*np.arange(1, len(ranks_r)+1)

    ax.axhline(ranks_r[r-1], lw=.5, c="gray")
    ax.axvline(2*r, lw=.5, c="gray")
    ax.plot(qs, qs, '-', color="gray", lw=1)
    ax.plot(qs, ranks_r, '.-', color="C0", mew=0, ms=10, lw=1)

    ax.set_xlim(qs[0], 2*rmax)
    ax.set_ylim(0, 2*rmax)
    ax.set_xlabel(r"Data matrix dimension $q(r) = 2r$")
    ax.set_ylabel(r"$\textrm{rank}(\mathbf{D})$")
    ax.set_title(r"Numerical rank of data matrix")
    ax.text(2*r+1.5, 20, fr"$q({r}) = {2*r}$", fontsize="large", color="gray")
    ax.text(20, ranks_r[r-1]+1.5,
            r"$\textrm{rank}(\mathbf{D}) = " + fr"{ranks_r[r-1]:d}$",
            fontsize="large", color="gray")


def plot_conditioning_r(r, conds_r, ranks_r):
    fig, [ax1, ax2] = plt.subplots(1, 2)
    plot_datacond_r(ax1, conds_r, r, 45)
    plot_datarank_r(ax2, ranks_r, r, 45)
    fig.subplots_adjust(wspace=.25)


def plot_datacond_s(ax, conds_s):
    """Plot the data matrix condition number as a function of the number of
    parameter samples.
    """
    ss = np.arange(1, len(conds_s)+1)

    ax.semilogy(ss, conds_s, '.-', color="C3", mew=0, ms=10, lw=1)

    ax.set_xlim(ss[0], ss[-1])
    ax.set_xlabel(r"Number of parameter samples $s$")
    ax.set_ylabel(r"$\kappa(\mathbf{D})$")
    ax.set_title(r"Condition number of data matrix")


def plot_datarank_s(ax, ranks_s):
    """Plot the numerical rank of the data matrix as a function of basis size.
    """
    ss = np.arange(1, len(ranks_s)+1)

    ax.plot(ss, ranks_s, '.-', color="C0", mew=0, ms=10, lw=1)

    ax.set_xlim(ss[0], ss[-1])
    ax.set_xlabel(r"Number of parameter samples $s$")
    ax.set_ylabel(r"$\textrm{rank}(\mathbf{D})$")
    ax.set_title(r"Numerical rank of data matrix")


def plot_conditioning_s(conds_s, ranks_s):
    fig, [ax1, ax2] = plt.subplots(1, 2)
    plot_datacond_s(ax1, conds_s)
    plot_datarank_s(ax2, ranks_s)
    fig.subplots_adjust(wspace=.25)


def plot_parameterspace_errors(ax, µ_train, µ_test, error, vmin, vmax):
    """Plot relative errors over the 2D parameter space."""
    S = µ_train.shape[0]
    αα, ββ = µ_train.reshape(S, S, 2).T
    error = error.reshape(S, S)

    norm = mplcolors.LogNorm(vmin, vmax)

    pcm = ax.pcolormesh(αα, ββ, error,
                        shading="nearest", cmap="magma", norm=norm)
    ax.plot(µ_train[:,0], µ_train[:,1], "*", color='w', ms=12, mew=.5, mec='k',
            label=r"training samples "
                  r"$\{\mu_i = (\alpha_i,\beta_i)\}_{i=1}^{s}$")

    ax.set_aspect("equal")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\beta$")
    ax.set_xticks([.5, 1.5, 2.5])
    ax.set_yticks([.5, 1.5, 2.5])

    print(f"Error statistics: "
          f"min = {error.min():.4e}; "
          f"max = {error.max():.4e}; "
          f"geometric mean = {np.exp(np.mean(np.log(error))):.4e}")

    return pcm


def plot_errors_projandrom(µ_train, µ_test, errors_proj, errors_rom,
                           vmin=1e-7, vmax=1e-3):
    vmin, vmax = 1e-7, 1e-3             # errors_proj.min()/2, 2*errors.max()

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(9,4), sharey=True)
    pcm = plot_parameterspace_errors(ax1, µ_train, µ_test, errors_proj,
                                     vmin, vmax)
    plot_parameterspace_errors(ax2, µ_train, µ_test, errors_rom, vmin, vmax)

    ax1.set_title("Projection Error")
    ax2.set_title("Reduced-order Model Error")

    ratio = 5/6
    fig.subplots_adjust(right=ratio, wspace=0.1)
    axheight = ax1.get_position().height
    cbar_ax = fig.add_axes([ratio + ratio/30, (1-axheight)/2,
                            ratio/40, axheight])
    cbar = fig.colorbar(pcm, cax=cbar_ax, extend="both")
    cbar.set_label("Relative error")


def plot_size_vs_error(rs,
                       opinf_train, intru_train, opinf_test, intru_test,
                       proj_train=None, proj_test=None):
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    ax1, ax2 = axes

    # Training set.
    ax1.semilogy(rs, opinf_train[1], "C0-", ms=10, lw=1)
    ax1.fill_between(rs, opinf_train[0], opinf_train[2],
                     color="C0", alpha=.2, lw=0, label="OpInf")
    ax1.semilogy(rs, intru_train[1], "C5--", ms=6, lw=1)
    ax1.fill_between(rs, intru_train[0], intru_train[2],
                     color="C5", alpha=.2, lw=0, label="Intrusive")
    if proj_train is not None:
        ax1.semilogy(rs, proj_train[1], "C3:", lw=1)
        ax1.fill_between(rs, proj_train[0], proj_train[2],
                         color="C3", alpha=.2, lw=0, label="Proj")

    # Testing set.
    ax2.semilogy(rs, opinf_test[1], "C0-", ms=10, lw=1)
    ax2.fill_between(rs, opinf_test[0], opinf_test[2],
                     color="C0", alpha=.2, lw=0, label="OpInf")
    ax2.semilogy(rs, intru_test[1], "C5--", ms=6, lw=1)
    ax2.fill_between(rs, intru_test[0], intru_test[2],
                     color="C5", alpha=.2, lw=0, label="Intrusive")
    if proj_test is not None:
        ax2.semilogy(rs, proj_test[1], "C3:", lw=1)
        ax2.fill_between(rs, proj_train[0], proj_train[2],
                         color="C3", alpha=.2, lw=0, label="Proj")

    # Annotations.
    ax1.legend(loc="lower left")
    ax2.legend(loc="lower left")

    # Format axes.
    for ax in axes:
        ax.set_xlabel(r"Basis size $r$")
        ax.set_xlim(min(rs), max(rs))
        ax.set_yticks([1e-6, 1e-4, 1e-2])
        ax.set_ylim(1e-7, .1)
    ax1.set_yticklabels([fr"${p}\%$" for p in ["0.0001", "0.01", "1"]])

    ax1.set_title("Training set")
    ax2.set_title("Testing set")
    fig.subplots_adjust(wspace=.025)


def movie_training(x, parameters, snapshots, xbar, fps=50, dpi=300):
    """Make a movie of some of the training data.

    Parameters
    ----------
    fps : int
        Frames per second (speed).
    dpi : int
        Dots per inch (resolution).
    """
    # Initialize figure
    fig, ax = plt.subplots(1, 1, figsize=(8,3))

    # Plot all initial lines.
    if len(parameters) == 5:
        color = ["C3", "C1", "C2", "C0", "C4"]
    else:
        color = plt.cm.Spectral(np.linspace(.1, 1, len(parameters)))
    lines = []
    for U, (alpha,beta), col in zip(snapshots, parameters, color):
        lines.append(ax.plot(x, U[:,0], color=col, lw=1.5)[0])
    ax.axvline(xbar, c="k", ls="--", lw=1, label=r"$\bar{x}$")

    # Format the axes.
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, 1.05)
    ax.set_xticks([0, .2, .4, .6, .8, 1])
    ax.set_yticks([0, .2, .4, .6, .8, 1])
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u(x,t=t_{0})$")

    # Write frames to a file in a loop.
    writer = ani.writers["ffmpeg"](fps=fps)
    outfile = os.path.join(BASE_FOLDER, "heat_snapshots.mp4")
    with writer.saving(fig, outfile, dpi):
        for j in tqdm.tqdm(range(snapshots.shape[-1]//3)):
            for i, line in enumerate(lines):
                line.set_data(x, snapshots[i,:,j])
            ax.set_ylabel(fr"$u(x,t=t_{{{j}}})$")
            writer.grab_frame()
    print("Wrote a movie to", outfile)


def movie_reconstruct(x, parameters, snapshots, reconstruct, xbar,
                      fps=50, dpi=300):
    """Make a movie of some of the ROM results on test data.

    Parameters
    ----------
    x
        Spatial domain.
    parameters
        Parameter values being tested.
    snapshots
        True full-order data to compare to.
    reconstruct
        ROM state output, reconstructed in the full space.
    xbar : float
        Point in the domain where the diffusion changes.
    fps : int
        Frames per second (speed).
    dpi : int
        Dots per inch (resolution).
    """
    # Initialize figure
    fig, ax = plt.subplots(1, 1, figsize=(8,3))

    # Plot all initial lines.
    if len(parameters) == 5:
        color = ["C3", "C1", "C2", "C0", "C4"]
    else:
        color = plt.cm.Spectral(np.linspace(.1, 1, len(parameters)))
    rom_lines = []
    for U, (alpha,beta), col in zip(reconstruct, parameters, color):
        rom_lines.append(ax.plot(x, U[:,0], color=col, lw=2.5)[0])
    snapshot_lines = []
    for U, (alpha,beta), col in zip(snapshots, parameters, color):
        snapshot_lines.append(ax.plot(x, U[:,0], 'k--', lw=1)[0])
    ax.axvline(xbar, c="gray", ls="-", lw=1, label=r"$\bar{x}$")

    # Format the axes.
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, 1.05)
    ax.set_xticks([0, .2, .4, .6, .8, 1])
    ax.set_yticks([0, .2, .4, .6, .8, 1])
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u(x,t=t_{0})$")

    # Write frames to a file in a loop.
    writer = ani.writers["ffmpeg"](fps=fps)
    outfile = os.path.join(BASE_FOLDER, "heat_reconstruct.mp4")
    with writer.saving(fig, outfile, dpi):
        for j in tqdm.tqdm(range(snapshots.shape[-1]//3)):
            for i, line in enumerate(rom_lines):
                line.set_data(x, reconstruct[i,:,j])
            for i, line in enumerate(snapshot_lines):
                line.set_data(x, snapshots[i,:,j])
            ax.set_ylabel(fr"$u(x,t=t_{{{j}}})$")
            writer.grab_frame()
    print("Wrote a movie to", outfile)


# Main routine ================================================================

def main(r, movie=False):
    """Make figures for a single experiment.

    Parameters
    ----------
    r : int
        Basis size of the ROM to plot results for.
    """
    init_settings()

    # Load data ---------------------------------------------------------------
    datafile = os.path.join(BASE_FOLDER, f"heat_results_r{r:0>2d}.h5")
    print(f"Reading results from {datafile}")
    with h5py.File(datafile, 'r') as hf:
        group = hf["domain"]
        x = group["x"][:]
        xbar = group["xbar"][:]
        time_fom = group["fom_cputimes"][:]

        group = hf["reducedmodel"]
        assert group["r"][0] == r
        V = group["basis"][:]
        svdvals = group["svdvals"][:]
        conds_r = group["conds_r"][:]
        ranks_r = group["ranks_r"][:]
        conds_s = group["conds_s"][:]
        ranks_s = group["ranks_s"][:]

        group = hf["training"]
        µ_train = group["parameters"][:]
        U_train = group["snapshots"][:]
        Urom_train = group["reconstruction"][:]

        group = hf["testing"]
        µ_test = group["parameters"][:]
        err_proj_test = group["error_proj"][:]
        err_opinf_test = group["error_opinf"][:]
        time_test = group["rom_cputime"][:]

    print(f"FOM time: {np.mean(time_fom):.4e} ± {np.std(time_fom):.4e}")
    print(f"ROM time: {np.mean(time_test):.4e} ± {np.std(time_test):.4e}")

    # Make a movie, then quit -------------------------------------------------
    if movie:
        movie_training(x, µ_train, U_train, xbar)
        movie_reconstruct(x, µ_train, U_train, Urom_train, xbar)
        return

    # Generate plots ----------------------------------------------------------
    # Sample snapshots of training trajectories + parameter samples.
    plot_training_samples(µ_train, U_train, x, xbar)
    utils.save_figure(f"heat_training_samples_r{r:0>2d}.pdf")

    # Residual mode energy decay and basis vectors.
    plot_basischoice(r, svdvals, V, x, xbar)
    utils.save_figure(f"heat_basis_r{r:0>2d}.pdf")

    # Conditioning / rank of data matrix as a function of basis size.
    plot_conditioning_r(r, conds_r, ranks_r)
    utils.save_figure(f"heat_data_r_r{r:0>2d}.pdf")

    # Conditioning / rank of data matrix as a function of sample size.
    plot_conditioning_s(conds_s, ranks_s)
    utils.save_figure(f"heat_data_s_r{r:0>2d}.pdf")

    # Errors in parameter space.
    plot_errors_projandrom(µ_train, µ_test, err_proj_test, err_opinf_test)
    utils.save_figure(f"heat_errors_r{r:0>2d}.png")


def main_multi():
    """Make figures for multiple experiments."""
    init_settings()

    # Load data.
    basis_sizes, regularizations = [], []
    errors_proj_train, errors_proj_test = [], []
    errors_opinf_train, errors_intru_train = [], []
    errors_opinf_test, errors_intru_test = [], []
    for f in sorted(glob.glob("/storage1/popinf/heat_results_r*.h5")):
        with h5py.File(f, 'r') as hf:
            basis_sizes.append(hf["reducedmodel/r"][0])
            regularizations.append(hf["reducedmodel/regularization"][0])
            errors_proj_train.append(hf["training/error_proj"][:])
            errors_opinf_train.append(hf["training/error_opinf"][:])
            errors_intru_train.append(hf["training/error_intru"][:])
            errors_proj_test.append(hf["testing/error_proj"][:])
            errors_opinf_test.append(hf["testing/error_opinf"][:])
            errors_intru_test.append(hf["testing/error_intru"][:])

    # Calculate error statistics.
    def get_stats(errors):
        return list(zip(*[(np.min(e), np.exp(np.mean(np.log(e))), np.max(e))
                          for e in errors]))

    proj_train = get_stats(errors_proj_train)
    opinf_train = get_stats(errors_opinf_train)
    intru_train = get_stats(errors_intru_train)
    proj_test = get_stats(errors_proj_train)
    opinf_test = get_stats(errors_opinf_test)
    intru_test = get_stats(errors_intru_test)

    # Plot the data.
    plot_size_vs_error(basis_sizes,
                       opinf_train, intru_train,
                       opinf_test, intru_test,
                       proj_train, proj_test)
    utils.save_figure("heat_basisVerror.pdf")


# =============================================================================

if __name__ == "__main__":
    # generate_test_data()

    # main(24, movie=False)
    # main_multi()

    pass
