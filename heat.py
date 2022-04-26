# heat.py
"""ROMs for a 1D parametric heat equation with piecewise constant diffusion."""

import os
import h5py
import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
import scipy.sparse as sparse
import scipy.sparse.linalg as spla

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

import rom_operator_inference as opinf

import utils


# ODE integrator for FOM ======================================================

def implicit_euler_linear(t, u0, A, sparseA=False):
    """Solve the system

        du / dt = Au(t),    u(0) = u0,

    over a uniform time domain via the Implicit Euler method.

    Parameters
    ----------
    t : (k,) ndarray
        Uniform time array over which to solve the ODE.
    u0 : (n,) ndarray
        Initial condition.
    A : (n, n) ndarray
        State matrix.
    sparseA : bool
        If True, assume A is a sparse matrix and use sparse linear algebra.

    Returns
    -------
    u : (n, k) ndarray
        Solution to the ODE at time t; that is, u[:, j] is the
        computed solution corresponding to time t[j].
    """
    # Check and store dimensions.
    k = len(t)
    n = len(u0)
    assert A.shape == (n, n)
    Id = sparse.eye(n) if sparseA else np.eye(n)

    # Get time step and initialize solution array.
    dt = t[1] - t[0]
    u = np.empty((n, k))
    u[:, 0] = u0.copy()

    if sparseA:
        inv = spla.splu(sparse.csc_matrix(Id - dt*A))
        for i in range(1, k):
            u[:, i] = inv.solve(u[:, i-1])
    else:
        inv = la.lu_factor(Id - dt*A)
        for i in range(1, k):
            u[:, i] = la.lu_solve(inv, u[:, i-1])

    # Return states.
    return u


# Solver classes ==============================================================

class HeatSolver:
    """Bundles a full-order solver, data management, and plotting tools
    for a parametric heat equation with piecewise constant diffusion:

        u_t = µ(x) u_xx,    where       µ(x) = α if x < xbar else β,

    with initial conditions

        u(x, t=0;µ) = f(0)(1 - x/L)^gamma + g(0)(x/L)^gamma,    0 ≤ x ≤ L,

    and homogeneous Dirichlet boundary conditions u(0, t) = u(L, t).
    ROM learning is implemented by child classes.

    Attributes
    ----------
    parameters : (s, 2) ndarray
        Scenario parameters corresponding to each snapshot set.
    snapshots : (s, n, k) ndarray
        Temperature snapshots corresponding to each scenario parameter set.

    Scenario Parameters
    -------------------
    α : float > 0
        Diffusion constant for domain < xbar.
    β : float > 0
        Diffusion constant for domain > xbar.
    """
    # Initialization ----------------------------------------------------------
    def __init__(self, nx=1000, nt=1500, L=1, tf=1.5, xbar=2/3.):
        """Initialize the domain and set variables for storing simulation data.

        Parameters
        ----------
        nx : int
            Number of partitions in the spatial domain, so that the total
            number of degrees of freedom is nx - 1 (Dirichlet BCs).
        nt : int
            Number of partitions in the temporal domain.
        L : float
            Length of the spatial domain.
        tf : float
            Final simulation time.
        xbar : float
            The point in the domain at which the diffusion coefficient changes.
        """
        self.parameters, self.snapshots = None, None

        # Spatial domain
        self._xbar = xbar
        self.x = np.linspace(0, L, nx+1)                # Domain
        assert self._L == L                             # Length
        assert self._dx == L/nx                         # Resolution
        assert self.n == nx+1                           # Size

        # Temporal domain
        self.t = np.linspace(0, tf, nt + 1).round(8)    # Domain
        assert self._tf == tf                           # Length
        assert self._dt == round(tf/nt, 16)             # Resolution
        assert self.k == nt + 1                         # Size

        # Construct matrices used by the full-order solver.
        dof = self.n - 2
        dx2inv = 1 / self._dx**2
        diags = np.array([1, -2, 1]) * dx2inv
        A1 = sparse.diags(diags, [-1, 0, 1], (dof, dof)).todok()
        A2 = A1.copy()
        A1[self._splitindex:] = 0
        A2[:self._splitindex] = 0

        # Save matrices in sparse column format.
        self._A1 = A1.tocsc()
        self._A2 = A2.tocsc()

    # Properties --------------------------------------------------------------
    @property
    def x(self):
        return self.__x

    @x.setter
    def x(self, xx):
        """Reset the spatial domain, erasing all snapshot data."""
        self.__x = xx
        self.n = xx.size                                # Spatial DOF
        self._dx = xx[1] - xx[0]                        # Spatial resolution
        self._L = xx[-1]                                # Domain length
        self._splitindex = np.searchsorted(self.x,
                                           self._xbar)  # Index of µ split
        self.snapshots = None                           # Erase data (!!)

    @property
    def t(self):
        return self.__t

    @t.setter
    def t(self, tt):
        """Reset the temporal domain."""
        self.__t = tt
        self.k = tt.size                                # Temporal DOF
        self._dt = tt[1] - tt[0]                        # Temporal resolution
        self._tf = tt[-1]                               # Final time
        self.snapshots = None                           # Erase data (!!)

    def __len__(self):
        """Length: number of datasets."""
        return self.snapshots.shape[0] if self.snapshots is not None else 0

    def __iter__(self):
        """Iteration: parameter sets (for now)."""
        if self.parameters is None:
            raise StopIteration("no data to iterate through")
        for item in self.parameters:
            yield item

    def __getitem__(self, key):
        """Indexing: get a view of a subset of the saved data (NO COPIES)."""
        if isinstance(key, int):
            key = slice(key, key+1)
        elif not isinstance(key, slice):
            raise IndexError("key must be int or slice")
        if self.snapshots is None:
            raise IndexError("no data to select")

        newsolver = self.__class__(self.n-1, self.k-1,
                                   self._L, self._tf, self._xbar)
        newsolver.parameters = self.parameters[key]
        newsolver.snapshots = self.snapshots[key]

        return newsolver

    def extend_time(self, factor):
        """Extend / shorten the time domain, maintaining the step size."""
        t, dt = self.t, self._dt
        return np.arange(t[0], factor*(t[-1] - t[0]) + t[0] + dt, dt)

    # Initial conditions ------------------------------------------------------
    def initial_conditions(self, gamma=50, plot=False):
        """Generate the initial conditions

        u(x, t=0;µ) = (1 - x/L)^gamma + (x/L)^gamma,    0 ≤ x ≤ L.

        Parameters
        ----------
        gamma : float
            Larger gamma -> more difference between boundary and interior.
        plot : bool
            If True, display the initial condition over the spatial domain.
        """
        u0 = (1 - self.x/self._L)**gamma + (self.x/self._L)**gamma

        if plot:
            fig, ax = self.plot_space(u0)
            ax.set_ylabel(r"$u_0(x)$")
            fig.suptitle(rf"Initial condition, $\gamma={gamma:.3f}$")
            fig.tight_layout()
            plt.show()

        return u0

    # Full-order solving ------------------------------------------------------
    def full_order_solve(self, params, u0):
        """Solve the full-order model at the given parameter values.

        Parameters
        ----------
        params : (2,) ndarray
            The scenario parameters, in order:
            * α: Diffusion constant for domain < xbar
            * β: Diffusion constant for domain > xbar
        u0 : (n,) ndarray
            Initial conditions for the temperature.

        Returns
        -------
        U : (n, k) ndarray
            Solution to the PDE over the discretized space-time domain.
        """
        # Unpack scenario parameter values.
        α, β = params

        # Construct the state matrix at these parameter values.
        Aµ = α*self._A1 + β*self._A2

        # Integrate the full-order model.
        u = implicit_euler_linear(self.t, u0[1:-1], Aµ, sparseA=True)

        BCs = np.zeros_like(self.t)
        return np.vstack([BCs, u, BCs])

    def add_snapshot_set(self, params):
        """Get high-fidelity snapshots for the given parameters.
        The initial condition is always the same.

        Parameters
        ----------
        params : (2,) ndarray
            Parameters at which to simulate the full-order model.
        """
        params = np.array(params)

        # Check that the parameters are not already in the database.
        if self.parameters is not None and any(np.all(params == p)
                                               for p in self.parameters):
            raise ValueError("parameters already present in database")

        # Get initial conditions.
        u0 = self.initial_conditions(plot=False)

        # Run (and time) the full-order model
        with utils.timed_block("Full-order model solve"):
            snaps = self.full_order_solve(params, u0)

        # Add results to the snapshot sets.
        if self.snapshots is None:
            self.parameters = np.array([params])
            self.snapshots = np.array([snaps])
        else:
            self.parameters = np.vstack([self.parameters, params])
            self.snapshots = np.vstack([self.snapshots,
                                        snaps.reshape((1,) + snaps.shape)])

    # Visualization -----------------------------------------------------------
    def plot_space(self, u, ax=None):
        """Plot temperature u = u(t=fixed, x) over space."""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        else:
            fig = ax.get_figure()

        ax.plot(self.x, u)
        ax.set_xlim(self.x[0], self.x[-1])
        ax.set_xlabel(r"$x \in [0, L]$")

        return fig, ax

    def plot_time(self, u, ax=None):
        """Plot temperature u(t, x=fixed) over time."""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        else:
            fig = ax.get_figure()

        ax.plot(self.t, u)
        ax.set_xlim(self.t[0], self.t[-1])
        ax.set_xlabel(r"$t \in [t_{0}, t_{f}]$")

        return fig, ax

    def plot_spacetime(self, u, params=None):
        """Plot temperature u over space-time.

        Parameters
        ----------
        u : (n, k) ndarray
            The temperature data to plot.
        params : (2,) ndarray
            The scenario parameters, in order:
            * α: Diffusion constant for domain < xbar
            * β: Diffusion constant for domain > xbar
        """
        if u.ndim != 2:
            raise ValueError("u must be two-dimensional")

        X, T = np.meshgrid(self.x, self.t, indexing="ij")
        fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 2))

        # Plot temperature at various points.
        color = iter(plt.cm.viridis(np.linspace(.25, 1, 6)))
        for j in [0, 20, 80, 160, 320, 640]:
            ax1.plot(self.x, u[:, j],
                     color=next(color), label=rf"$u(x, t_{{{j}}})$")
        ax1.set_xlim(self.x[0], self.x[-1])
        ax1.set_xlabel(r"$x \in [0, L]$")
        ax1.axvline(self._xbar, c="k", ls="--", lw=1)

        # Plot temperature in space and time.
        cdata = ax2.pcolormesh(X, T, u, shading="nearest", cmap="magma")
        ax2.set_xlabel(r"$x \in [0, L]$")
        ax2.set_ylabel(r"$t \in [t_{0}, t_{f}]$")
        fig.colorbar(cdata, ax=ax2, extend="both")

        # Make a legend on the left side of the plots.
        fig.subplots_adjust(left=0.2, wspace=.15)
        ax1.legend(loc="center left", edgecolor='none', frameon=False,
                   bbox_to_anchor=(0, 0.5), bbox_transform=fig.transFigure)

        title = r"Temperature $u$"
        if params is not None:
            title += fr", $\alpha={params[0]:.3f}$, $\beta={params[1]:.3f}$"
        fig.suptitle(title)

        return fig, [ax1, ax2]

    def plot_parameters(self, test_parameters=None):
        if self.parameters is None:
            raise ValueError("no parameters to visualize")

        fig, ax = plt.subplots(1, 1)
        µ_train = self.parameters.T
        ax.plot(µ_train[0], µ_train[1], 'C0*',
                mew=0, label="Training parameters")
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(r"$\beta$")

        # Also plot test parameters if given.
        if test_parameters is not None:
            µ_test = np.array(test_parameters).T
            ax.plot(µ_test[0], µ_test[1], 'C1*',
                    mew=0, label="Testing parameters")
            ax.legend(loc="center left",
                      bbox_to_anchor=(1.05, 0.5),
                      edgecolor='none', frameon=False)
        else:
            ax.set_title("Training parameters")

        ax.set_aspect("equal")

    def plot_snapshots(self):
        """Plot each snapshot set in a separate figure."""
        for params, data in zip(self.parameters, self.snapshots):
            self.plot_spacetime(data, params)
        plt.show()

    def animate_field(self, profiles, labels=None):
        """Animate one or two temperature profiles in time."""
        profiles = np.array(profiles)
        if profiles.ndim == 1:
            raise ValueError("two-dimensional data required for animation")
        if profiles.ndim == 2:
            profiles = np.array([profiles])
        draw_legend = (labels is not None)
        if not draw_legend:
            labels = [None]*len(profiles)
        assert len(profiles) == len(labels)

        fig, ax = plt.subplots(1, 1, figsize=(8, 3), dpi=150)
        lines = [plt.plot([], [], label=label)[0] for label in labels]

        def init():
            for line in lines:
                line.set_data([], [])
            return lines

        def update(index):
            for line, u in zip(lines, profiles):
                line.set_data(self.x, u[:, index*10])
            ax.set_title(fr"$t = t_{{{index*10}}}$")
            return lines

        ax.set_xlim(0, self._L)
        ax.set_ylim(profiles.min()-.2, profiles.max()+.2)
        ax.axvline(self._xbar, color='k', ls='--', alpha=.5)
        ax.set_xlabel(r"$x\in[0, L]$")
        ax.set_title(r"$t = t_{0}$")
        if draw_legend:
            fig.subplots_adjust(left=0.25)
            ax.legend(loc="center left",
                      edgecolor='none', frameon=False,
                      bbox_to_anchor=(0, 0.5), bbox_transform=fig.transFigure)
        else:
            ax.set_ylabel(r"$u(x, t)$")

        a = animation.FuncAnimation(fig, update, init_func=init,
                                    frames=profiles[0].shape[1]//10,
                                    interval=30, blit=True)
        plt.close(fig)
        return HTML(a.to_jshtml())

    def animate_snapshots(self):
        """Animate each snapshots set in a single figure."""
        labels = [fr"$\alpha = {m[0]:.2f}, \beta = {m[1]:.2f}$"
                  for m in self.parameters]
        return self.animate_field(self.snapshots, labels)

    # Data I/O ----------------------------------------------------------------
    @classmethod
    def load(cls, loadfile):
        """Load data from an HDF5 file.

        Should load domain data, snapshot data, parameters with which those
        snapshots were generated, etc.
        """
        with h5py.File(loadfile, 'r') as hf:
            # Domain parameters.
            if "domain" not in hf:
                raise ValueError("invalid save format (domain/ not found)")
            nx = hf["domain/nx"][0]
            nt = hf["domain/nt"][0]
            L = hf["domain/L"][0]
            tf = hf["domain/tf"][0]
            xbar = hf["domain/xbar"][0]
            solver = cls(nx, nt, L, tf, xbar)

            # Parameter and snapshot data.
            if "data" not in hf:
                raise ValueError("invalid save format (snapshots/ not found)")
            solver.parameters = hf["data/parameters"][:]
            solver.snapshots = hf["data/snapshots"][:]

        return solver

    def save(self, savefile, overwrite=False):
        """Save current state to an HDF5 file.

        Should save domain data, snapshot data, parameters with which those
        snapshots were generated, etc.

        Parameters
        ----------
        savefile : str
            File to save to. If it does not end with '.h5', the extension will
            be tacked on to the end.
        overwrite : bool
            If True and the specified file already exists, overwrite the file.
            If False and the specified file already exists, raise an error.
        """
        if self.snapshots is None or len(self.snapshots) == 0:
            raise ValueError("no data to save")

        # Make sure the file is saved in HDF5 format.
        if not savefile.endswith(".h5"):
            savefile += ".h5"

        if os.path.isfile(savefile) and not overwrite:
            raise FileExistsError(savefile)

        with h5py.File(savefile, 'w') as hf:
            # Domain parameters.
            hf.create_dataset("domain/nx", data=[self.n-1])
            hf.create_dataset("domain/nt", data=[self.k-1])
            hf.create_dataset("domain/L", data=[self._L])
            hf.create_dataset("domain/tf", data=[self._tf])
            hf.create_dataset("domain/xbar", data=[self._xbar])

            # Snapshot data.
            hf.create_dataset("data/parameters",
                              data=np.array(self.parameters))
            hf.create_dataset("data/snapshots",
                              data=np.array(self.snapshots))


# Reduced-order Models ========================================================

class HeatROM(HeatSolver):
    """Bundles a high-fidelity solver, data management, plotting tools, and
    ROM learning for a parametric heat equation with piecewise constant
    diffusion:

        u_t = µ(x) u_xx,    where       µ(x) = α if x < xbar else β,

    with initial conditions

        u(x, t=0; µ) = (1 - x/L)^gamma + (x/L)^gamma,    0 ≤ x ≤ L,

    and homogeneous Dirichlet boundary conditions u(0, t) = u(L, t).

    Attributes
    ----------
    parameters : (s, 2) ndarray
        Scenario parameters corresponding to each snapshot set.
    snapshots : (s, n, k) ndarray
        Temperature snapshots corresponding to each scenario parameter set.

    Scenario Parameters
    -------------------
    α : float > 0
        Diffusion constant for domain < xbar.
    β : float > 0
        Diffusion constant for domain > xbar.
    """
    _modelform = "A"
    _affines = {"A": [lambda µ: µ[0], lambda µ: µ[1]]}

    # Reduced-order model construction ----------------------------------------
    def _pod_basis(self, saveas=None):
        """Compute the (global) full-rank POD basis from the snapshot sets."""
        with utils.timed_block("computing POD basis"):
            U_all = np.hstack(self.snapshots)
            V, svdvals, _ = la.svd(U_all, full_matrices=False)

        if saveas:
            utils.save_basis(saveas, V, svdvals)

        return V, svdvals

    def train_rom(self, r, reg=None, basis=None):
        """Use the stored snapshot data to compute an appropriate basis and
        train a ROM using Operator Inference.

        Parameters
        ----------
        r : int or float
            * int: Number of POD basis vectors to use (size of the ROM).
            * float: Choose size to exceed this level of cumulative energy.
        reg : float or None
            * float: Regularization hyperparameter λ.
            * None: do a gridsearch, then a 1D optimization to choose λ.
        basis : str or None
            If provided, load the POD basis with the given group name.
        """
        if self.snapshots is None:
            raise ValueError("no simulation data with which to train ROM")
        snapshots = self.snapshots
        time_domain = self.t

        # Load or compute or load POD basis matrix.
        if basis:
            Vr = utils.load_basis(basis, r)
        else:
            Vr = self._pod_basis()[0][:, :r]

        # Project the training data and estimate time derivatives.
        with utils.timed_block("projecting training data"):
            Us_ = [Vr.T @ U for U in snapshots]
            dt = time_domain[1] - time_domain[0]
            Udots_train_ = [(U_[:, 1:] - U_[:, :-1]) / dt
                            for U_ in Us_]
            Us_train_ = [U_[:, 1:] for U_ in Us_]
        u0_ = Us_[0][:, 0]

        # Calculate the mean projection error of the snapshot data.
        mean_proj_err = np.mean([
            opinf.post.Lp_error(U, Vr @ U_, time_domain)[1]
            for U, U_ in zip(snapshots, Us_)])
        print(f"MEAN PROJECTION ERROR: {mean_proj_err:.2e}")

        # Instantiate the ROM.
        rom = opinf.AffineInferredContinuousROM(self._modelform)

        # Single ROM solve, no regularization optimization.
        if reg is not None:
            with utils.timed_block(f"computing single ROM with λ={reg:5e}"):
                rom.reg = reg
                return rom.fit(Vr, self.parameters, self._affines,
                               Us_train_, Udots_train_, P=reg)

        # Several ROM solves, optimizing the regularization.
        _MAXFUN = 1e8
        with utils.timed_block("Constructing OpInf least-squares solver"):
            rom._construct_solver(None, self.parameters, self._affines,
                                  Us_train_, Udots_train_, None, P=1)
            rom.Vr = Vr

        def training_error_from_rom(log10_λ):
            """Return the training error resulting from the regularization
            parameter λ = 10^log10_λ. If the resulting model violates the
            POD bound, return "infinity".
            """
            λ = 10**log10_λ
            error = 0

            print(f"\rTesting ROM with λ={λ:e}", end='')
            rom._evaluate_solver(self._affines, λ)
            with np.warnings.catch_warnings():
                np.warnings.simplefilter("ignore")

                # Simulate on training parameters, computing error.
                for µ, U_ in zip(self.parameters, Us_):
                    try:
                        U_rom = self.predict(rom, µ, u0_, decode=False)
                    except Exception as e:
                        print(f"Prediction failed ({type(e).__name__})")
                        return _MAXFUN
                    error += opinf.post.Lp_error(U_, U_rom, time_domain)[1]
                return error / len(self.snapshots)

        # Evaluate training_error_from_rom() over a coarse logarithmic grid.
        print("starting regularization grid search")
        log10_grid = np.linspace(-16, 6, 45)
        errs = [training_error_from_rom(λ) for λ in log10_grid]
        windex = np.argmin(errs)
        λ = 10**log10_grid[windex]
        print(f"\ngrid search winner: {λ:e}")
        if windex == 0 or windex == log10_grid.size - 1:
            print("WARNING: grid search bounds should be extended")
            rom._evaluate_solver(self._affines, λ)
            rom.reg = λ
            rom.Vr = Vr
            return rom

        # Run the optimization and extract the best result.
        print("starting regularization optimization-based search")
        opt_result = opt.minimize_scalar(training_error_from_rom,
                                         bracket=log10_grid[windex-1:windex+2],
                                         method="brent")
        if opt_result.success and opt_result.fun != _MAXFUN:
            λ = 10 ** opt_result.x
            print(f"\noptimization-based search winner: {λ:e}")
            print(f"MEAN RELATIVE ERROR: {opt_result.fun:2e}")
            rom._evaluate_solver(self._affines, λ)
            rom.reg = λ
            rom.Vr = Vr
            return rom
        else:
            print("\nRegularization search optimization FAILED")

    # Reduced-order model evaluation ------------------------------------------
    def predict(self, rom, µ, u0, decode=True):
        """Integrate a reduced-order model `rom` at the parameter value `µ`
        from the initial conditions `u0`.
        """
        u0_ = rom.project(u0)
        Aµ_ = rom.A_(µ)
        u_ = implicit_euler_linear(self.t, u0_, Aµ_, sparseA=False)

        return (rom.Vr @ u_) if decode else u_
