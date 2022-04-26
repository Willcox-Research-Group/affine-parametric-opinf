# fhn.py
"""ROMs for the FitzHugh-Nagumo equation."""
import os
import h5py
import logging
import warnings
import itertools
import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
import scipy.integrate as sin
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

import rom_operator_inference as opinf

import config
import utils


# Reduced-order model classes =================================================
def kron_indices(r):
    """Construct masks for compact quadratic and cubic Kronecker."""
    r2_mask = np.zeros((r*(r+1)//2, 2), dtype=np.int)
    r3_mask = np.zeros((r*(r+1)*(r+2)//6, 3), dtype=np.int)
    r2_count = 0
    r3_count = 0
    for i in range(r):
        for j in range(i+1):
            r2_mask[r2_count, :] = (i, j)
            r2_count += 1
            for k in range(j+1):
                r3_mask[r3_count, :] = (i, j, k)
                r3_count += 1
    return r2_mask, r3_mask


class _SystemROM:
    """Base class for ROMs with separate bases for each variable."""
    def _set_bases(self, bases):
        """Store bases and dimensions."""
        self.Vr1, self.Vr2 = bases
        self.n1, self.r1 = self.Vr1.shape
        self.n2, self.r2 = self.Vr2.shape
        self._r12 = self.r1 * (self.r1 + 1) // 2
        self._r13 = self._r12 * (self.r1 + 2) // 3
        self._r12mask, self._r13mask = kron_indices(self.r1)
        self.Vr = sparse.block_diag(bases).tocsr()

    @property
    def n(self):
        return self.n1 + self.n2

    @property
    def r(self):
        return self.r1 + self.r2

    def project(self, S, label="input"):
        """Check the dimensions of S and project it if needed."""
        if S.shape[0] == self.n:
            return self.Vr.T @ S
        elif S.shape[0] != self.r:
            raise ValueError(f"{label} not aligned with basis, dimension 0")
        return S


class _EvaluatedFHNROM(_SystemROM):
    """Nonparametric reduced-order model of the FitzHugh-Nagumo system."""
    def __init__(self, bases, operators):
        """Unpack bases and (non-parametric) ROM system operators."""
        # Store bases and dimension.
        self._set_bases(bases)

        # Check that operator values are arrays.
        for arr in operators.values():
            assert isinstance(arr, np.ndarray)

        # Unpack operators.
        self.c1_ = operators["c1"]
        self.B1_ = operators["B1"]
        self.A11_ = operators["A11"]
        self.A12_ = operators["A12"]
        self.H111_ = operators["H111"]
        self.G1111_ = operators["G1111"]
        self.c2_ = operators["c2"]
        self.A21_ = operators["A21"]
        self.A22_ = operators["A22"]

        self.A_ = np.block([[self.A11_, self.A12_],
                            [self.A21_, self.A22_]])

        # Expand quadratic / cubic operator for Jacobian evaluation.
        H = opinf.utils.expand_H(self.H111_).reshape([self.r1]*3)
        self.H111_Jac = H + H.transpose(0, 2, 1)
        G = opinf.utils.expand_G(self.G1111_).reshape([self.r1]*4)
        self.G1111_Jac = G + G.transpose(0, 2, 1, 3) + G.transpose(0, 3, 1, 2)

    def nonlin_(self, t, u_, input_func):
        """Compute the nonlinear terms of the right-hand side of the ODE."""
        u1_, _ = np.split(u_, [self.r1], axis=0)
        u1_2 = np.prod(u1_[self._r12mask], axis=1)
        u1_3 = np.prod(u1_[self._r13mask], axis=1)

        du1_nonlin = self.c1_ + self.B1_*input_func(t)
        du1_nonlin += (self.H111_ @ u1_2) + (self.G1111_ @ u1_3)

        return np.concatenate([du1_nonlin, self.c2_])

    def evaluate_(self, t, u_, f):
        """Reduced-order model function, du/dt = f(t, u).

        Parameters
        ----------
        t : float
            Time, a scalar.
        u_ : (r,) ndarray
            Reduced state vector corresponding to time `t` (r = r1 + r2).
        f : func(float) -> float
            Input function that maps time `t` to a float (BCs).
        """
        return self.A_ @ u_ + self.nonlin_(t, u_, f)

    def jac_(self, t, u_):
        """Jacobian of the reduced-order model function, Df(t, u).
        Need this in order for the adaptive time stepping to be efficient.

        Parameters
        ----------
        t : float
            Time, a scalar.
        u_ : (r,) ndarray
            Reduced state vector corresponding to time `t` (r = r1 + r2).
        """
        u1_, _ = np.split(u_, [self.r1], axis=0)

        # Compute the Jacobian of the quadratic and cubic terms.
        Hu1_2 = self.H111_Jac @ u1_
        Gu1_3 = (self.G1111_Jac @ u1_) @ u1_

        # Assemble the Jacobian.
        J = self.A_.copy()
        J[:self.r1, :self.r1] += Hu1_2 + Gu1_3
        return J

    def predict(self, u0, t, f, reconstruct=True, **options):
        """Simulate the learned ROM.

        Parameters
        ----------
        u0 : (n,) or (r,) ndarray
            Initial state vector, either full order (n-vector) or projected
            to reduced order (r-vector).
        t : (nt,) ndarray
            Time domain over which to integrate the reduced-order system.
        f : callable
            Input as a function of time.
        reconstruct : bool
            If True (default), reconstruct the solution in the full space.
            If Flase, return the solution in the reduced space.
        **options
            Arguments for solver.integrate.solve_ivp().
            See https://docs.scipy.org/doc/scipy/reference/integrate.html.

        Returns
        -------
        U_ROM : (n, nt) or (r, nt) ndarray
            The approximate solution to the system over the time domain `t`.
            If the basis Vr is None, return solutions in the reduced
            r-dimensional subspace (r, nt). Otherwise, map the solutions to the
            full n-dimensional space with Vr (n, nt).
        """
        # Project initial conditions if needed.
        u0_ = self.project(u0, "u0")

        # Verify time domain.
        if t.ndim != 1:
            raise ValueError("time 't' must be one-dimensional")

        # Integrate the reduced-order model.
        def _fun(t, u_):
            return self.evaluate_(t, u_, f)

        self.sol_ = sin.solve_ivp(_fun,
                                  [t[0], t[-1]],
                                  u0_,
                                  t_eval=t,
                                  method="Radau",
                                  jac=self.jac_,
                                  **options)

        # Raise warnings if the integration failed.
        if not self.sol_.success:
            warnings.warn(self.sol_.message, sin.IntegrationWarning)

        # Reconstruct the approximation to the full-order model.
        return (self.Vr @ self.sol_.y) if reconstruct else self.sol_.y


class AffineFHNROM(_SystemROM):
    """Affine-parametric reduced-order model of the FitzHugh-Nagumo system,
    learned via affine-parametric Operator Inference (pOpInf).
    """
    affines = {
        "c1": [lambda µ: µ[0]/µ[3]],                     # α/ε
        "B1": [lambda µ: -1022*µ[3]],                    # -2ε/dx
        "A11": [lambda µ: µ[3], lambda µ: -0.1/µ[3]],    # ε - .1/ε
        "A12": [lambda µ: -1/µ[3]],                      # -1/ε
        "H111": [lambda µ: 1.1/µ[3]],                    # 1.1/ε
        "G1111": [lambda µ: -1/µ[3]],                    # -1/ε
        "c2": [lambda µ: µ[0]],                          # α
        "A21": [lambda µ: µ[1]],                         # β
        "A22": [lambda µ: -µ[2]],                        # -δ
    }

    @property
    def m(self):
        """Dimension of the temporal input (boundary conditions)."""
        return 1

    @property
    def p(self):
        """Dimension of the parameter space."""
        return 4

    def _process_fit_arguments(self, bases, params, states, ddts, inputs):
        """Do sanity checks, extract dimensions, check and fix data sizes, and
        get projected data for the Operator Inference least-squares problem.

        Parameters
        ----------
        bases : tuple of two (n, r_l) ndarrays
            Bases for the reduced supspace (e.g., POD basis matrices),
            one for each state variable.
        params : list of s (p,) ndarrays
            Parameter values corresponding to the snapshot data.
        states : list of s (n, k_i) ndarrays
            Column-wise snapshot training data (each column is a snapshot).
            The ith array states[i] corresponds to parameter value params[i].
            These snapshots represent both state variables.
        ddts : list of s (n, k_i) ndarrays (or (s, n, k) ndarray)
            Column-wise next-iteration (discrete model) or time derivative
            (continuous model) training data. The ith array, ddts[i],
            corresponds to the ith parameter, params[i].
        inputs : list of s (m, k) or (k,) ndarrays
            Inputs corresponding to the snapshots (boundary conditions).

        Returns
        -------
        states_ : list of s (r, k_i) ndarrays
            Projected state snapshots. states_[i] corresponds to µ[i].
        ddts_ : list of s (r, k_i) ndarrays
            Projected right-hand-side data. ddts_[i] corresponds to µ[i].
        inputs : list of s (m, k) ndarrays or None
            Boundary condition inputs. inputs[i] corresponds to µ[i].
        """
        # Validate parameter dimension.
        for µ in params:
            if len(µ) != self.p:
                raise ValueError(f"parameters µ must be {self.p}-dimensional")

        # Check that the number of params matches the number of training sets.
        s = len(params)
        for data, name in [
            (states, "state"),
            (ddts, "ddt"),
            (inputs, "input"),
        ]:
            if len(data) != s:
                raise ValueError(f"num params != num {name} training sets "
                                 f"({s} != {len(data)})")

        # Store basis and reduced dimension.
        self._set_bases(bases)

        # Ensure training data sets have consistent sizes (inputs always 1D).
        for i in range(s):
            ki = states[i].shape[1]
            if ddts[i].shape[1] != ki or inputs[i].shape != (ki,):
                raise ValueError(f"states[{i}], ddts[{i}], inputs[{i}]"
                                 " not aligned")

        # Project states and rhs to the reduced subspace (if not done already).
        if np.any(U.shape[0] != self.r or Udot.shape[0] != self.r
                  for U, Udot in zip(states, ddts)):
            states_ = np.array([self.project(U, "state") for U in states])
            ddts_ = np.array([self.project(Udot, "ddt") for Udot in ddts])
        else:
            states_, ddts_ = states, ddts

        return states_, ddts_

    def _assemble_data_matrix(self, params, states_, inputs):
        """Construct the Operator Inference data matrix D from projected data.

        Parameters
        ----------
        params : list of s (p,) ndarrays
            Parameter values at which the snapshot data is collected.
        states_ : list of s (r, k_i) ndarrays
            Column-wise projected snapshot training data.
            The ith array states[i] corresponds to parameter value params[i].
        inputs : list of s (m, k_i) or (k_i,) ndarrays or None
            Column-wise inputs corresponding to the snapshots (BCs).

        Returns
        -------
        D1 : (sum(k_i), q1) ndarray
            Operator Inference data matrix for the first state variable.
        D2 : (sum(k_i), q2) ndarray
            Operator Inference data matrix for the second state variable.
        """
        D1_rows, D2_rows = [], []
        θs = self.affines

        for µ, U_, f in zip(params, states_, inputs):
            # Polynomial terms of the states.
            U1_, U2_ = np.split(U_, [self.r1], axis=0)
            ones = np.ones((U_.shape[1], 1))
            # U1_2 = opinf.utils.kron2c(U1_)
            # U1_3 = opinf.utils.kron3c(U1_)
            U1_2 = np.prod(U1_[self._r12mask], axis=1)
            U1_3 = np.prod(U1_[self._r13mask], axis=1)

            # Block row of D1.
            f = f.reshape(-1, 1)
            row = []
            row += [θ(µ) * ones for θ in θs["c1"]]          # Constant
            row += [θ(µ) * f for θ in θs["B1"]]             # Input
            row += [θ(µ) * U1_.T for θ in θs["A11"]]        # Linear (u1)
            row += [θ(µ) * U2_.T for θ in θs["A12"]]        # Linear (u2)
            row += [θ(µ) * U1_2.T for θ in θs["H111"]]      # Quadratic
            row += [θ(µ) * U1_3.T for θ in θs["G1111"]]     # Cubic
            D1_rows.append(np.hstack(row))

            # Block row of D2.
            row = []
            row += [θ(µ) * ones for θ in θs["c2"]]          # Constant
            row += [θ(µ) * U1_.T for θ in θs["A21"]]        # Linear (u1)
            row += [θ(µ) * U2_.T for θ in θs["A22"]]        # Linear (u2)
            D2_rows.append(np.hstack(row))

        # Assemble block rows.
        D1 = np.vstack(D1_rows)
        D2 = np.vstack(D2_rows)

        return D1, D2

    def _assemble_rhs(self, params, states_, ddts_, inputs):
        ddt1s, ddt2s = np.split(np.hstack(ddts_), [self.r1], axis=0)
        return ddt1s.T, ddt2s.T

    def _extract_operators(self, Ohats):
        """Extract and save the inferred operators from the block-matrix
        solution to the least-squares problem, constructing AffineOperators
        as indicated by the affine structure.

        Parameters
        ----------
        Ohat1 : (r1, q1) and (r2, q2) ndarrays
            Block matrices of ROM operator coefficients (the transpose of the
            solution to the Operator Inference linear least-squares problem),
            one for each state variable.
        """
        q1 = np.cumsum([
            1,                                              # Constant
            1,                                              # Input
            self.r1,                                        # Linear (u1 (1))
            self.r1,                                        # Linear (u1 (2))
            self.r2,                                        # Linear (u2)
            self._r12                                       # Quadratic
        ])                                                  # Cubic
        q2 = np.cumsum([1, self.r1])

        # Unpack operators.
        Ohat1, Ohat2 = Ohats
        c1, B1, A11_1, A11_2, A12, H111, G1111 = np.split(Ohat1, q1, axis=1)
        c2, A21, A22 = np.split(Ohat2, q2, axis=1)

        # Check shapes.
        assert c1.shape == (self.r1, 1)                     # Constant
        assert B1.shape == (self.r1, 1)                     # Input
        assert A11_1.shape == (self.r1, self.r1)            # Linear (u1)
        assert A11_2.shape == (self.r1, self.r1)            # Linear (u1)
        assert A12.shape == (self.r1, self.r2)              # Linear (u2)
        assert H111.shape == (self.r1, self._r12)           # Quadratic
        assert G1111.shape == (self.r1, self._r13)          # Cubic
        assert c2.shape == (self.r2, 1)                     # Constant
        assert A21.shape == (self.r2, self.r1)              # Linear (u1)
        assert A22.shape == (self.r2, self.r2)              # Linear (u2)

        # Define affine operators.
        θs = self.affines
        self.c1_ = opinf.AffineOperator(θs["c1"], [c1[:, 0]])
        self.B1_ = opinf.AffineOperator(θs["B1"], [B1[:, 0]])
        self.A11_ = opinf.AffineOperator(θs["A11"], [A11_1, A11_2])
        self.A12_ = opinf.AffineOperator(θs["A12"], [A12])
        self.H111_ = opinf.AffineOperator(θs["H111"], [H111])
        self.G1111_ = opinf.AffineOperator(θs["G1111"], [G1111])
        self.c2_ = opinf.AffineOperator(θs["c2"], [c2[:, 0]])
        self.A21_ = opinf.AffineOperator(θs["A21"], [A21])
        self.A22_ = opinf.AffineOperator(θs["A22"], [A22])

    def _construct_solver(self, bases, params, states, ddts, inputs):
        """Construct a solver object mapping the regularizer P to solutions
        of the Operator Inference least-squares problem.
        """
        states_, ddts_ = self._process_fit_arguments(bases, params,
                                                     states, ddts, inputs)
        D1, D2 = self._assemble_data_matrix(params, states_, inputs)
        R1, R2 = self._assemble_rhs(params, states_, ddts_, inputs)
        self.D1, self.R1 = D1, R1

        # Solve the second problem ONCE b/c there is no regularization.
        self.Ohat2 = la.lstsq(D2, R2)[0].T

    def _evaluate_solver(self, λs):
        """Evaluate the least-squares solver with the given regularization.

        Parameters
        ----------
        λs : (float, float, float)
            Regularization hyperparameters defining the Tikhonov regularizer.
            λ1: constant, input, and linear terms.
            λ2: quadratic terms.
            λ3: cubic terms.
        """
        λ1, λ2, λ3 = λs

        # Regularizer for first problem.
        d = self.D1.shape[1]
        P1 = np.zeros(d)
        P1[:(self._r12 + self._r13)] = λ1
        P1[-(self._r12 + self._r13):] = λ2
        P1[-self._r13:] = λ3

        # Solve the first (regularization dependent) problem.
        Ohat1 = la.lstsq(np.vstack((self.D1, np.diag(P1))),
                         np.vstack((self.R1, np.zeros((d, self.r1)))))[0].T
        self._extract_operators([Ohat1, self.Ohat2])

    def fit(self, bases, params, states, ddts, inputs, λs=(0, 0, 0)):
        """Solve for the reduced model operators via ordinary least squares.

        Parameters
        ----------
        bases : tuple of two (n, r_l) ndarrays
            Bases for the reduced supspace (e.g., POD basis matrices),
            one for each state variable.
        params : list of s (p,) ndarrays
            Parameter values corresponding to the snapshot data.
        states : list of s (n, k_i) ndarrays
            Column-wise snapshot training data (each column is a snapshot).
            The ith array states[i] corresponds to parameter value params[i].
            These snapshots represent both state variables.
        ddts : list of s (n, k_i) ndarrays (or (s, n, k) ndarray)
            Column-wise next-iteration (discrete model) or time derivative
            (continuous model) training data. The ith array, ddts[i],
            corresponds to the ith parameter, params[i].
        inputs : list of s (m, k) or (k,) ndarrays
            Inputs corresponding to the snapshots (boundary conditions).
        λs : (float, float, float)
            Regularization hyperparameters defining the Tikhonov regularizer.
            λ1: constant, input, and linear terms.
            λ2: quadratic terms.
            λ3: cubic terms.

        Returns
        -------
        self
        """
        self._construct_solver(bases, params, states, ddts, inputs)
        self._evaluate_solver(λs)
        return self

    def __call__(self, µ):
        """Evaluate the ROM at the given parameter."""
        ops = {
            "c1": self.c1_(µ),
            "B1": self.B1_(µ),
            "A11": self.A11_(µ),
            "A12": self.A12_(µ),
            "H111": self.H111_(µ),
            "G1111": self.G1111_(µ),
            "c2": self.c2_(µ),
            "A21": self.A21_(µ),
            "A22": self.A22_(µ),
        }
        return _EvaluatedFHNROM([self.Vr1, self.Vr2], ops)

    def predict(self, µ, *args, **kwargs):
        return self(µ).predict(*args, **kwargs)

    def save(self, savefile, overwrite=False):
        """Save the FH-N parametric ROM in HDF5 format.

        Parameters
        ----------
        savefile : str
            HDF5 file to save data to. Should end in .h5.
        overwrite : bool
            If False and `savefile` exists, raise a FileExistsError.
        """
        if os.path.isfile(savefile) and not overwrite:
            raise FileExistsError(f"{savefile} (overwrite=True to ignore)")

        with h5py.File(savefile, 'w') as hf:
            gp = hf.create_group("basis")
            gp.create_dataset("Vr1", data=self.Vr1)
            gp.create_dataset("Vr2", data=self.Vr2)

            gp = hf.create_group("operators")
            gp.create_dataset("c1_", data=self.c1_.matrices)
            gp.create_dataset("B1_", data=self.B1_.matrices)
            gp.create_dataset("A11_", data=self.A11_.matrices)
            gp.create_dataset("A12_", data=self.A12_.matrices)
            gp.create_dataset("H111_", data=self.H111_.matrices)
            gp.create_dataset("G1111_", data=self.G1111_.matrices)
            gp.create_dataset("c2_", data=self.c2_.matrices)
            gp.create_dataset("A21_", data=self.A21_.matrices)
            gp.create_dataset("A22_", data=self.A22_.matrices)

    @classmethod
    def load(cls, loadfile):
        """Load the FH-N parametric ROM from an HDF5 file.

        Parameters
        ----------
        loadfile : str
            HDF5 file to load data from.
        """
        with h5py.File(loadfile, 'r') as hf:
            bases = (hf["basis/Vr1"][:], hf["basis/Vr2"][:])

            gp = hf["operators"]
            c1_ = gp["c1_"][:]
            B1_ = gp["B1_"][:]
            A11_ = gp["A11_"][:]
            A12_ = gp["A12_"][:]
            H111_ = gp["H111_"][:]
            G1111_ = gp["G1111_"][:]
            c2_ = gp["c2_"][:]
            A21_ = gp["A21_"][:]
            A22_ = gp["A22_"][:]

        rom = cls()
        rom._set_bases(bases)
        θs = cls.affines
        rom.c1_ = opinf.AffineOperator(θs["c1"], c1_)
        rom.B1_ = opinf.AffineOperator(θs["B1"], B1_)
        rom.A11_ = opinf.AffineOperator(θs["A11"], A11_)
        rom.A12_ = opinf.AffineOperator(θs["A12"], A12_)
        rom.H111_ = opinf.AffineOperator(θs["H111"], H111_)
        rom.G1111_ = opinf.AffineOperator(θs["G1111"], G1111_)
        rom.c2_ = opinf.AffineOperator(θs["c2"], c2_)
        rom.A21_ = opinf.AffineOperator(θs["A21"], A21_)
        rom.A22_ = opinf.AffineOperator(θs["A22"], A22_)

        return rom


class AffineFHNROM_Intrusive(AffineFHNROM):
    """Affine-parametric reduced-order model of the FitzHugh-Nagumo system,
    learned via affine-parametric Operator Inference (pOpInf).

    This variant constructs the entire ROM directly with intrusive projection.
    """
    def fit(self, bases, A11_1):
        """
        Parameters
        ----------
        bases : tuple of two (n, r_l) ndarrays
            Bases for the reduced supspace (e.g., POD basis matrices),
            one for each state variable.

        A11_1 : (n, n) ndarray
            Diffusion operator for u1, i.e., A_11 @ u1 = d^2 / dx^2 u1.
        """
        self._set_bases(bases)

        # Construct reduced order operators.
        c1 = self.Vr1.sum(axis=0)
        B1 = self.Vr1[0, :]
        A11_1 = self.Vr1.T @ A11_1 @ self.Vr1
        A11_2 = self.Vr1.T @ self.Vr1
        A12 = self.Vr1.T @ self.Vr2
        c2 = self.Vr2.sum(axis=0)
        A21 = self.Vr2.T @ self.Vr1
        A22 = self.Vr2.T @ self.Vr2

        def kr23(X):
            X2 = np.column_stack([np.kron(x, x) for x in X.T])
            X3 = np.column_stack([np.kron(xj, Xj)
                                  for xj, Xj in zip(X.T, X2.T)])
            return X2, X3

        V1T2, V1T3 = kr23(self.Vr1.T)
        H111 = opinf.utils.compress_H(self.Vr1.T @ V1T2.T)
        G1111 = opinf.utils.compress_G(self.Vr1.T @ V1T3.T)

        assert c1.shape == (self.r1,)                       # Constant
        assert B1.shape == (self.r1,)                       # Input
        assert A11_1.shape == (self.r1, self.r1)            # Linear (u1)
        assert A11_2.shape == (self.r1, self.r1)            # Linear (u1)
        assert A12.shape == (self.r1, self.r2)              # Linear (u2)
        assert H111.shape == (self.r1, self._r12)           # Quadratic
        assert G1111.shape == (self.r1, self._r13)          # Cubic
        assert c2.shape == (self.r2,)                       # Constant
        assert A21.shape == (self.r2, self.r1)              # Linear (u1)
        assert A22.shape == (self.r2, self.r2)              # Linear (u2)

        θs = self.affines
        self.c1_ = opinf.AffineOperator(θs["c1"], [c1])
        self.B1_ = opinf.AffineOperator(θs["B1"], [B1])
        self.A11_ = opinf.AffineOperator(θs["A11"], [A11_1, A11_2])
        self.A12_ = opinf.AffineOperator(θs["A12"], [A12])
        self.H111_ = opinf.AffineOperator(θs["H111"], [H111])
        self.G1111_ = opinf.AffineOperator(θs["G1111"], [G1111])
        self.c2_ = opinf.AffineOperator(θs["c2"], [c2])
        self.A21_ = opinf.AffineOperator(θs["A21"], [A21])
        self.A22_ = opinf.AffineOperator(θs["A22"], [A22])

        return self


# Solver classes ==============================================================

class FHNSolver:
    """Bundles a high-fidelity solver, data management, and plotting tools
    for the FitzHugh-Nagumo equations:

        ε u1_t = ε^2 u1_xx + u1(u1 - 0.1)(1 - u1) - u2 + α,
          u2_t = β u1 - δ u2 + α,                           0 ≤ x ≤ 1, t ≥ 0,

    with initial conditions u1(x, 0) = u2(x, 0) = 0 and boundary conditions

        u1_x(0, t) = -50000t^3 exp(-15t),        u2_x(1, t) = 0,       t ≥ 0.

    ROM learning is implemented by child classes.

    Attributes
    ----------
    parameters : (s, 2) ndarray
        Scenario parameters corresponding to each snapshot set.
    snapshots : (s, n, k) ndarray
        Temperature snapshots corresponding to each scenario parameter set.
    derivatives : (s, n, k) ndarray
        Time derivatives of state snapshots for each scenario parameter set.

    Scenario Parameters
    -------------------
    α : float > 0
    β : float > 0
    δ : float > 0
    ε : float > 0
    """
    NUM_VARIABLES = 2

    # Initialization ----------------------------------------------------------
    def __init__(self, nx=512, nt=4000, L=1, tf=4, downsample=10):
        """Initialize the domain and set variables for storing simulation data.

        Parameters
        ----------
        nx : int
            Number of points in the spatial domain, so that the total
            number of degrees of freedom is 2nx (Neumann BCs).
        nt : int or float
            * int: Number of intervals in the temporal domain.
            * float: Time step δt.
        L : float
            Length of the spatial domain.
        tf : float
            Final simulation time.
        downsample : int
            Downsample solutions by this factor.
            Hence the number of snapshots per simulation is nt / downsample.
        """
        self.parameters, self.inputs = None, None
        self.snapshots, self.derivatives = None, None

        # Spatial domain
        self.x = np.linspace(0, L, nx)                  # Domain
        assert self._L == L                             # Length
        assert self._dx == L/(nx-1)                     # Resolution
        assert self.n == nx                             # Size

        # Particular solution for boundary conditions.
        self._wbar = self.x*(1 - .5*self.x**2)

        # Temporal domain
        self.downsample = int(downsample)
        if nt < 1:
            nt = int(tf / nt)
        self.t_dense = np.linspace(0, tf, nt+1)         # Domain
        assert self._tf == tf                           # Length
        assert self._dt == round(tf/nt, 16)             # Resolution
        assert self.k == (nt // self.downsample) + 1    # Size
        assert self.t_dense.size == nt + 1              # Dense size

        # Construct the state matrices used by the full-order solver.
        δx2inv = 1 / self._dx**2
        diags = np.array([1, -2, 1]) * δx2inv
        A = sparse.diags(diags, [-1, 0, 1], (self.n, self.n)).todok()

        # Neumann boundary terms.
        twoδx2inv = 2 * δx2inv
        A[0, 0] = -twoδx2inv
        A[0, 1] = twoδx2inv
        A[-1, -1] = -twoδx2inv
        A[-1, -2] = twoδx2inv

        self._A1 = A.todia()
        self._A2 = sparse.eye(self.n)

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
        self.snapshots = None                           # Erase data (!!)

    @property
    def t_dense(self):
        return self.__t

    @t_dense.setter
    def t_dense(self, tt):
        """Reset the temporal domain."""
        self.__t = tt
        self.k = self.t.size                            # Temporal DOF
        self._dt = tt[1] - tt[0]                        # Temporal resolution
        self._tf = tt[-1]                               # Final time
        self.snapshots = None                           # Erase data (!!)

    @property
    def t(self):
        return self.t_dense[::self.downsample]

    def __len__(self):
        """Length: number of datasets."""
        return self.snapshots.shape[0] if self.snapshots is not None else 0

    def __getitem__(self, key):
        """Indexing: get a view of a subset of the saved data (NO COPIES)."""
        if isinstance(key, int):
            key = slice(key, key+1)
        if self.snapshots is None:
            raise IndexError("no data to select")

        newsolver = self.__class__(self.n, self.k-1, self._L, self._tf)
        for attr in ["parameters", "inputs", "snapshots", "derivatives"]:
            setattr(newsolver, attr, getattr(self, attr)[key])

        return newsolver

    def extend_time(self, factor):
        """Extend / shorten the time domain, maintaining the step size."""
        t, dt = self.t_dense, self._dt
        return np.arange(t[0], factor*(t[-1] - t[0]) + t[0] + dt, dt)

    # Initial conditions ------------------------------------------------------
    def initial_conditions(self):
        """Generate the (zero) initial conditions."""
        return np.zeros(self.n)

    # Full-order solving ------------------------------------------------------
    def full_order_solve(self, params, f, **options):
        """Solve the full-order model at the given parameter values.

        Parameters
        ----------
        params : (4,) ndarray
            Scenario parameters α, β, δ, ε.
        f : callable
            Left Neumann boundary condition, a function of time.
        options : dict
            Options for the ODE solver scipy.integrate.solve_ivp().

        Returns
        -------
        U : (n, k) ndarray
            Solution to the PDE over the discretized space-time domain.
        dU : (n, k) ndarray
            Time derivatives over the discretized space-time domain.
        """
        # Unpack scenario parameters and set initial conditions.
        α, β, δ, ε = params
        u0 = np.zeros(self.n*self.NUM_VARIABLES)

        # Construct A1 for these parameters.
        A1 = (ε*self._A1 - (.1/ε)*self._A2).tocsr()
        Id = sparse.diags(np.ones_like(self.x))

        def full_order_model(t, u):
            """FitzHugh-Nagumo Equations"""
            u1, u2 = np.split(u, 2, axis=0)
            du1dt = (A1 @ u1) + (1.1*(u1**2) - u1**3 - u2 + α)/ε
            du1dt[0] = du1dt[0] - 2*ε*f(t)/self._dx
            du2dt = β*u1 - δ*u2 + α
            return np.concatenate([du1dt, du2dt], axis=0)

        def fom_jacobian(t, u):
            u1, u2 = np.split(u, 2, axis=0)
            du11 = A1 + (sparse.diags(2.2*u1) - sparse.diags(3*u1**2))/ε
            du12 = -Id/ε
            du21 = β*Id
            du22 = -δ*Id
            J = sparse.bmat([[du11, du12], [du21, du22]])
            return J

        # Integrate the full-order model.
        U = sin.solve_ivp(full_order_model,
                          [self.t[0], self.t[-1]],
                          u0,
                          method="Radau",
                          jac=fom_jacobian,
                          vectorized=True,
                          t_eval=self.t_dense,
                          **options).y

        # Estimate time derivatives and downsample snapshots.
        dt = self.t_dense[1] - self.t_dense[0]
        dU = opinf.pre.xdot_uniform(U, dt, order=6)
        dU = dU[:, ::self.downsample]
        U = U[:, ::self.downsample]
        return U, dU

    def add_snapshot_set(self, params=None, f=None, **options):
        """Get high-fidelity snapshots for the given parameters.
        The initial condition is always the same.

        Parameters
        ----------
        params : (4,) ndarray
            Parameters at which to simulate the full-order model.
        """
        if params is None:
            # alpha=.05, beta=.5, delta=2, epsilon=.015
            params = [.05, .5, 2, .015]
        params = np.array(params)

        if f is None:
            f = config.fhn_input

        # Check that the parameters are not already in the database.
        if self.parameters is not None:
            if np.min(la.norm(self.parameters - params, axis=1)) == 0:
                raise ValueError("parameters already present in database")

        # Run (and time) the full-order model
        with utils.timed_block(f"Full-order model solve at µ = {params}"):
            snaps, dts = self.full_order_solve(params, f, **options)

        # Add results to the snapshot sets.
        if self.snapshots is None:
            self.parameters = np.array([params])
            self.inputs = np.array([f(self.t)])
            self.snapshots = np.array([snaps])
            self.derivatives = np.array([dts])
        else:
            self.parameters = np.vstack([self.parameters, params])
            self.inputs = np.vstack([self.inputs, f(self.t)])
            self.snapshots = np.vstack([self.snapshots,
                                        snaps.reshape((1,)+snaps.shape)])
            self.derivatives = np.vstack([self.derivatives,
                                          dts.reshape((1,)+dts.shape)])

    def add_snapshot_sets(self, params, f=None, **options):
        """Get high-fidelity snapshots for multiple given parameters.
        The initial condition is always the same.

        Parameters
        ----------
        params : (s, 4) ndarray
            Parameters at which to simulate the full-order model.
        f : callable
            Input function (boundary condition).
        """
        parameters = np.array(params)
        if f is None:
            f = config.fhn_input

        # Solve the full-order model at the specified parameters.
        snapshots, derivatives, inputs = [], [], []
        for i, µ in enumerate(parameters):
            print(f"({i+1:0>3d}/{parameters.shape[0]:0>3d})", end=' ')
            with utils.timed_block(f"High-fidelity solve at µ = {µ}"):
                snaps, dts = self.full_order_solve(µ, f, **options)
                snapshots.append(snaps)
                derivatives.append(dts)
                inputs.append(f(self.t))

        # Add results to the snapshot sets.
        if self.snapshots is None:
            self.parameters = parameters
            self.inputs = np.array(inputs)
            self.snapshots = np.array(snapshots)
            self.derivatives = np.array(derivatives)
        else:
            self.parameters = np.concatenate([self.parameters, parameters])
            self.inputs = np.concatenate([self.inputs, inputs])
            self.snapshots = np.concatenate([self.snapshots, snapshots])
            self.derivatives = np.concatenate([self.derivatives, derivatives])

    # Visualization -----------------------------------------------------------
    @staticmethod
    def _param_labels(params):
        return ", ".join([fr"$\alpha={params[0]:.3f}$",
                          fr"$\beta={params[1]:.3f}$",
                          fr"$\delta={params[2]:.3f}$",
                          fr"$\varepsilon={params[3]:.3f}$"])

    def plot_space(self, u, axes=None):
        """Plot variables u1(t=fixed, x) and u2(t=fixed, x) over space."""
        if axes is None:
            fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        else:
            fig = axes[0].get_figure()

        u1, u2 = np.split(u, self.NUM_VARIABLES)

        axes[0].plot(self.x, u1)
        axes[1].plot(self.x, u2)
        axes[0].set_ylabel(r"$u_{1}(t_{j}, x)$")
        axes[1].set_ylabel(r"$u_{2}(t_{j}, x)$")
        axes[0].set_xlim(self.x[0], self.x[-1])
        axes[1].set_xlim(self.x[0], self.x[-1])
        axes[1].set_xlabel(r"$x \in [0, L]$")

        return fig, axes

    def plot_time(self, u=0, nlocs=10, axes=None):
        """Plot u1 and u2 individually in time.

        Parameters
        ----------
        u : (2n, k) ndarray or int
            * (2n, k) ndarray: snapshot set to plot.
            * int: index of snapshot set to plot in stored data.
        nlocs : int
            Number of lines to draw.
        ax : (plt.Axes, plt.Axes)
            Axes on which to draw.

        Returns
        -------
        fig : plt.Figure
            Figure that was drawn on.
        axes : (plt.Axes, plt.Axes)
            Axes that were drawn on.
        """
        if axes is None:
            fig, axes = plt.subplots(2, 1, figsize=(6, 3), sharex=True)
        else:
            fig = axes[0].get_figure()

        if isinstance(u, int):
            u = self.snapshots[u]

        colors = plt.cm.viridis_r(np.linspace(.2, 1, nlocs))
        xlocs = np.logspace(0, np.log10(self.n-1), nlocs, dtype=np.int)
        for i, c in zip(reversed(xlocs), reversed(colors)):
            u1 = u[i, :]
            u2 = u[i+self.n, :]
            axes[0].plot(self.t, u1, lw=1, color=c)
            axes[1].plot(self.t, u2, lw=1, color=c)

        axes[0].set_ylabel(r"$u_{1}(x_{j}, t)$")
        axes[1].set_ylabel(r"$u_{2}(x_{j}, t)$")
        axes[1].set_xlabel(r"$t \in [t_{0}, t_{f}]$")
        for ax in axes:
            ax.set_xlim(self.t[0], self.t[-1])
        fig.align_ylabels(axes)

        return fig, axes

    def plot_phase(self, u=0, nlocs=10, ax=None):
        """Plot u1 versus u2 in time at several points in the domain.

        Parameters
        ----------
        u : (2n, k) ndarray or int
            * (2n, k) ndarray: snapshot set to plot.
            * int: index of snapshot set to plot in stored data.
        nlocs : int
            Number of lines to draw.
        ax : plt.Axes
            Axes on which to draw.

        Returns
        -------
        fig : plt.Figure
            Figure that was drawn on.
        ax : plt.Axes
            Axes that was drawn on.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
        else:
            fig = ax.get_figure()

        if isinstance(u, int):
            u = self.snapshots[u]

        colors = plt.cm.viridis_r(np.linspace(.2, 1, nlocs))
        xlocs = np.logspace(0, np.log10(self.n-1), nlocs, dtype=np.int)
        for i, c in zip(reversed(xlocs), reversed(colors)):
            u1 = u[i, :]
            u2 = u[i+self.n, :]
            ax.plot(u1, u2, '.-', color=c, markevery=self.k//40, lw=1)
        ax.set_xlabel(r"$u_{1}(x_{j}, t)$")
        ax.set_ylabel(r"$u_{2}(x_{j}, t)$")
        ax.set_xlim(-.5, 1.5)
        ax.set_ylim(0, .2)

        return fig, ax

    @staticmethod
    def _subplot_grid():
        fig = plt.figure(constrained_layout=True, figsize=(9, 4))
        spec = fig.add_gridspec(nrows=2, ncols=2, hspace=.25, wspace=.25,
                                width_ratios=[1.5, 1],
                                height_ratios=[1, 1], bottom=.25)
        ax1 = fig.add_subplot(spec[0, 0])
        ax2 = fig.add_subplot(spec[1, 0])
        ax3 = fig.add_subplot(spec[:, 1])
        return fig, [ax1, ax2, ax3]

    def plot(self, u=0, nloc=10):
        """Plot time and phase together.

        Parameters
        ----------
        u : (2n, k) ndarray or int
            * (2n, k) ndarray: snapshot set to plot.
            * int: index of snapshot set to plot in stored data.
        nloc : int
            Number of lines to draw.

        Returns
        -------
        fig : plt.Figure
            Figure that was drawn on.
        ax : (plt.Axes, plt.Axes, plt.Axes)
            Axes that were drawn on.
        """
        fig, [ax1, ax2, ax3] = self._subplot_grid()
        self.plot_time(u, nloc, [ax1, ax2])
        self.plot_phase(u, nloc, ax3)
        ax1.set_xticklabels('')
        if isinstance(u, int):
            fig.suptitle(self._param_labels(self.parameters[u]))

        return fig, [ax1, ax2, ax3]

    def plot_spacetime(self, u=0, params=None):
        """Plot u1 and u2 in space-time.

        Parameters
        ----------
        u : (2n, k) ndarray or int
            * (2n, k) ndarray: snapshot set to plot.
            * int: index of snapshot set to plot in stored data.
        params : (4,) ndarray
            Scenario parameters α, β, δ, ε.
        """
        if isinstance(u, int):
            u = self.snapshots[u]
        if u.ndim != 2:
            raise ValueError("u must be two-dimensional")

        X, T = np.meshgrid(self.x, self.t, indexing="ij")
        fig, axes = plt.subplots(1, self.NUM_VARIABLES, figsize=(12, 2))

        for ul, ax in zip(np.split(u, self.NUM_VARIABLES), axes):
            cdata = ax.pcolormesh(X, T, ul, shading="nearest", cmap="magma")
            ax.set_xlabel(r"$x \in [0, L]$")
            ax.set_ylabel(r"$t \in [t_{0}, t_{f}]$")
            fig.colorbar(cdata, ax=ax, extend="both")

        if params is not None:
            fig.suptitle(self._param_labels(params))

        return fig, axes

    def animate_phase(self, xloc=100):
        """Animate phase profiles."""
        fig, ax = plt.subplots(1, 1, figsize=(6, 4.5), dpi=150)
        line = plt.plot([], [], '.-', markevery=100)[0]

        def init():
            line.set_data([], [])
            return (line,)

        def update(index):
            line.set_data(self.snapshots[index, xloc, :],
                          self.snapshots[index, xloc+self.n, :])
            ax.set_title(self._param_labels(self.parameters[index]))
            return (line,)

        ax.set_xlim(-.5, 1.5)
        ax.set_ylim(0, .2)
        ax.set_xlabel(r"$u_{1}(x_{j}, t)$")
        ax.set_ylabel(r"$u_{2}(x_{j}, t)$")

        a = animation.FuncAnimation(fig, update, init_func=init,
                                    frames=len(self), interval=100, blit=True)
        plt.close(fig)
        return HTML(a.to_jshtml())

    def animate_time(self, xloc=100):
        """Animate time plots."""
        fig, axes = plt.subplots(2, 1, figsize=(6, 3), sharex=True)
        lines = [ax.plot([], [])[0] for ax in axes]

        def init():
            for line in lines:
                line.set_data([], [])
            return lines

        def update(index):
            lines[0].set_data(self.t, self.snapshots[index, xloc, :])
            lines[1].set_data(self.t, self.snapshots[index, xloc+self.n, :])
            fig.suptitle(self._param_labels(self.parameters[index]))
            return lines

        axes[0].set_ylim(-.4, 1.4)
        axes[1].set_ylim(-.01, .21)
        axes[0].set_ylabel(r"$u_{1}(x_{j}, t)$")
        axes[1].set_ylabel(r"$u_{2}(x_{j}, t)$")
        axes[1].set_xlim(self.t[0], self.t[-1])
        axes[1].set_xlabel(r"$t \in [t_{0}, t_{f}]$")
        a = animation.FuncAnimation(fig, update, init_func=init,
                                    frames=len(self), interval=100, blit=True)
        plt.close(fig)
        return HTML(a.to_jshtml())

    def animate(self, nlocs=10, saveas=None):
        """Animate time and phase plots."""
        xlocs = np.logspace(0, np.log10(self.n-1), nlocs, dtype=np.int)

        fig, axes = self._subplot_grid()
        colors = plt.cm.viridis_r(np.linspace(.2, 1, nlocs))
        lines = []
        for c in reversed(colors):
            lines.append(axes[0].plot([], [], '-', color=c, lw=3)[0])
            lines.append(axes[1].plot([], [], '-', color=c, lw=3)[0])
            lines.append(axes[2].plot([], [], '.-',
                                      color=c, markevery=200, lw=2)[0])
            lines.append(axes[2].plot([], [], 'o', color=c, markersize=10)[0])

        def init():
            for line in lines:
                line.set_data([], [])
            return lines

        def update(index):
            linecount = 0
            for loc in xlocs[::-1]:
                u1 = self.snapshots[index, loc, :]
                u2 = self.snapshots[index, loc+self.n, :]
                lines[linecount+0].set_data(self.t, u1)
                lines[linecount+1].set_data(self.t, u2)
                lines[linecount+2].set_data(u1, u2)
                lines[linecount+3].set_data([u1[-1]], [u2[-1]])
                linecount += 4
            fig.suptitle(self._param_labels(self.parameters[index]))
            return lines

        axes[0].set_ylim(-.5, 1.74)
        axes[1].set_ylim(-.01, .25)
        axes[0].set_xlim(self.t[0], self.t[-1])
        axes[1].set_xlim(self.t[0], self.t[-1])
        axes[0].set_ylabel(r"$u_{1}(x_{j}, t)$")
        axes[1].set_ylabel(r"$u_{2}(x_{j}, t)$")
        axes[1].set_xlabel(r"$t \in [t_{0}, t_{f}]$")
        axes[2].set_xlim(-.4, 1.6)
        axes[2].set_ylim(0, .275)
        axes[2].set_xlabel(r"$u_{1}(x_{j}, t)$")
        axes[2].set_ylabel(r"$u_{2}(x_{j}, t)$")
        fig.align_ylabels(axes[:2])

        # Return the animation as an embeddable JS-HTML.
        if saveas is None:
            a = animation.FuncAnimation(fig, update, init_func=init,
                                        frames=len(self), interval=100,
                                        blit=True)
            plt.close(fig)
            return HTML(a.to_jshtml())

        # Write the animation to an external file.
        writer = animation.writers["ffmpeg"](fps=10)
        outfile = os.path.join(config.BASE_FOLDER, saveas)
        with writer.saving(fig, outfile, dpi=300):
            init()
            for i in range(len(self)):
                update(i)
                writer.grab_frame()
        plt.close(fig)

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
            group = hf["domain"]
            nx = group["nx"][0]
            nt = group["nt"][0]
            L = group["L"][0]
            tf = group["tf"][0]
            if "downsample" in group:
                downsample = group["downsample"][0]
            else:
                downsample = 10
            solver = cls(nx, nt, L, tf, downsample)

            # Parameter and snapshot data.
            if "data" not in hf:
                raise ValueError("invalid save format (snapshots/ not found)")
            solver.parameters = hf["data/parameters"][:]
            solver.inputs = hf["data/inputs"][:]
            solver.snapshots = hf["data/snapshots"][:]
            solver.derivatives = hf["data/derivatives"][:]

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
            print(f"Saving to {savefile}")

        if os.path.isfile(savefile) and not overwrite:
            raise FileExistsError(savefile)

        with h5py.File(savefile, 'w') as hf:
            # Domain parameters.
            hf.create_dataset("domain/nx", data=[self.n])
            hf.create_dataset("domain/nt", data=[self.t_dense.size-1])
            hf.create_dataset("domain/L", data=[self._L])
            hf.create_dataset("domain/tf", data=[self._tf])
            hf.create_dataset("domain/downsample", data=[self.downsample])

            # Snapshot data.
            hf.create_dataset("data/parameters",
                              data=np.array(self.parameters))
            hf.create_dataset("data/inputs",
                              data=np.array(self.inputs))
            hf.create_dataset("data/snapshots",
                              data=np.array(self.snapshots))
            hf.create_dataset("data/derivatives",
                              data=np.array(self.derivatives))


class FHNROMSolver(FHNSolver):
    """Bundles a high-fidelity solver, data management, plotting tools, and
    parametric Operator Inference ROMs for the FitzHugh-Nagumo equations:

        ε u1_t = ε^2 u1_xx + u1(u1 - 0.1)(1 - u1) - u2 + α,
          u2_t = β u1 - δ u2 + α,                           0 ≤ x ≤ 1, t ≥ 0,

    with initial conditions u1(x, 0) = u2(x, 0) = 0 and boundary conditions

        u1_x(0, t) = -50000t^3 exp(-15t),        u2_x(1, t) = 0,       t ≥ 0.

    Attributes
    ----------
    parameters : (s, 2) ndarray
        Scenario parameters corresponding to each snapshot set.
    snapshots : (s, n, k) ndarray
        Temperature snapshots corresponding to each scenario parameter set.

    Scenario Parameters
    -------------------
    α : float > 0
    β : float > 0
    δ : float > 0
    ε : float > 0
    """
    def _pod_basis(self, saveas=None):
        """Compute the (global) full-rank POD basis from the snapshot sets,
        with one basis for each variable (u1 and u2).

        Parameters
        ----------
        saveas : str or None
            If given, save the POD basis and singular values.

        Returns
        -------
        V1 : (n, k) ndarray
            POD basis for u1, where each column is a basis vector.
        svdvals : (n,) ndarray
            Singular values of the POD basis for u1.
        V2 : (n, k) ndarray
            POD basis for u2, where each column is a basis vector.
        svdvals : (n,) ndarray
            Singular values of the POD basis for u1.
        """
        # Compute a separate basis for each variable.
        with utils.timed_block("computing POD bases"):
            U_all = np.hstack(self.snapshots)
            U1, U2 = np.split(U_all, 2, axis=0)
            V1, svdvals1, _ = la.svd(U1, full_matrices=False)
            V2, svdvals2, _ = la.svd(U2, full_matrices=False)

        # Save each basis / svdval pair if requested.
        if saveas:
            utils.save_basis(saveas+"_1", V1, svdvals1)
            utils.save_basis(saveas+"_2", V2, svdvals2)

        return V1, svdvals1, V2, svdvals2

    def train_rom(self, rs, reg=None, bases=None,
                  gridsearch=None, regguess=None, trialtimelimit=15,
                  µ_test=None, **options):
        """Use the stored snapshot data to compute an appropriate basis and
        train a ROM using parametric Operator Inference.

        Parameters
        ----------
        rs : list(int or float)
            * int: Number of POD basis vectors to use for the ith variable.
            * float: Choose size to exceed this level of cumulative energy.
        reg : (3,) ndarray or float or None
            * ndarray: Regularization hyperparameters λ1, λ2, λ3.
                Valid if ROMClass is AffineFHNROM, AffineFHNROM_Hybrid1,
                AffineFHNROM_Hybrid2, or AffineFHNROM_Hybrid3.
            * float: Regularization hyperparameter λ.
                Valid if ROMClass is AffineFHNROM_Hybrid4.
            * None: do a gridsearch, then a 1D optimization to choose λ.
        basis : (n, r) ndarray, str, or None
            * (n, r) ndarray: POD basis matrix.
            * str: group name of previously computed basis.
            * None: compute the basis from the snapshot data.
        gridsearch : list(ndarrays) or ndarray or None
            Space to search for a regularization hyperparameter.
            * list(ndarrays): Grids to search for hyperparameters λ2, λ3.
            * ndarray: Grid to search for hyperparameter λ.
            If provided, then the argument `regguess` is ignored.
        regguess : (2,) ndarray or None
            Initial guess for the regularization hyperparameter search.
            Ignored if argument `gridsearch` is provided.
        """
        if self.snapshots is None:
            raise ValueError("no simulation data with which to train ROM")
        if len(self) < 2:
            raise ValueError("at least two trajectories required for learning")
        num_snaps = len(self.snapshots)

        # Load or compute or unpack POD basis matrices.
        r1, r2 = rs
        if isinstance(bases, str):
            V1 = utils.load_basis(bases+"_1", r1)
            V2 = utils.load_basis(bases+"_2", r2)
        elif bases is None:
            V1, _, V2, _ = self._pod_basis()
        else:
            V1, V2 = bases
        V1, V2 = V1[:, :r1], V2[:, :r2]
        bases = (V1, V2)

        # Project the training data and calculate the derivative.
        with utils.timed_block("projecting training data"):
            Us_, dUs_ = [], []
            proj_errors = []
            for i in range(num_snaps):
                U = self.snapshots[i]
                U1, U2 = np.split(U, 2, axis=0)
                U1_, U2_ = V1.T @ U1, V2.T @ U2
                Us_.append(np.vstack([U1_, U2_]))
                U_proj = np.vstack([V1 @ U1_, V2 @ U2_])
                proj_errors.append(opinf.post.Lp_error(U, U_proj,
                                                       self.t)[1])
                dU = self.derivatives[i]
                dU1, dU2 = np.split(dU, 2, axis=0)
                dUs_.append(np.vstack([V1.T @ dU1, V2.T @ dU2]))
        print(f"Average Relative Projection Error: {np.mean(proj_errors):.2%}",
              f"(geometric mean: {np.exp(np.mean(np.log(proj_errors))):.2%})")

        # Instantiate the ROM.
        rom = AffineFHNROM()

        # Single ROM solve, no regularization optimization.
        if reg is not None:
            with utils.timed_block("computing single ROM"):
                rom.reg = reg
                return rom.fit(bases, self.parameters,
                               Us_, dUs_, self.inputs, reg)

        # Several ROM solves, optimizing the regularization.
        _MAXFUN = 1e12
        self._one_works = False
        with utils.timed_block("constructing OpInf least-squares solver"):
            rom._construct_solver(bases, self.parameters,
                                  Us_, dUs_, self.inputs)
        if µ_test is None:
            µ_test = []
        u0_ = np.zeros(sum(rs))

        def training_error_from_rom(log10_λs):
            """Return the training error resulting from the regularization
            parameter(s) λ = 10^log10_λ. If the resulting model is unstable,
            return "infinity".
            """
            λs = 10**log10_λs
            λs[0] = 0
            errors = []
            with utils.timed_block(f"\nTesting ROM with rs={rs}, λ={λs}",
                                   timelimit=trialtimelimit):
                try:
                    rom._evaluate_solver(λs)
                except Exception as e:
                    print("Solver evaluation failed",
                          f"({type(e).__name__}: {e})...", end='')
                    return _MAXFUN

                # Simulate on testing parameters, check for stability.
                for µ in µ_test:
                    try:
                        with np.warnings.catch_warnings():
                            np.warnings.simplefilter("ignore")
                            U_rom = rom.predict(µ, u0_,
                                                self.t,
                                                config.fhn_input,
                                                reconstruct=False,
                                                **options)
                        nsteps = U_rom.shape[1]
                        if nsteps != self.t.size:
                            raise ValueError(f"unstable ({nsteps} steps)")
                    except Exception as e:
                        print("Prediction on test parameter failed",
                              f"({type(e).__name__}: {e})...", end='')
                        return _MAXFUN

                # Simulate on training parameters, computing error.
                for nparam, (µ, U_) in enumerate(zip(self.parameters, Us_)):
                    try:
                        with np.warnings.catch_warnings():
                            np.warnings.simplefilter("ignore")
                            U_rom = rom.predict(µ, U_[:, 0],
                                                self.t,
                                                config.fhn_input,
                                                reconstruct=False,
                                                **options)
                        nsteps = U_rom.shape[1]
                        if nsteps != self.t.size:
                            raise ValueError(f"unstable after {nsteps} steps")
                    except Exception as e:
                        print(f"Prediction on param {nparam:d} failed",
                              f"({type(e).__name__}: {e})...", end='')
                        return _MAXFUN

                    errors.append(opinf.post.Lp_error(U_, U_rom, self.t)[1])

                # Report error.
                self._one_works = True
                avgerr = np.mean(errors)
                geoerr = np.exp(np.mean(np.log(errors)))
                print(f"\nSTABLE ROM (rs = {rs}),",
                      f"Average Relative Error = {avgerr:.2%}",
                      f"(geometric mean: {geoerr:.2%})")
                logging.info(f"STABLE ROM (rs = {rs}): λs={λs}, "
                             f"AvgRelError = {avgerr:.2%}")
                return avgerr
            return _MAXFUN

        # Evaluate training_error_from_rom() over a coarse logarithmic grid.
        if regguess is not None:
            # Center the gridsearch at an initial guess.
            λs = np.log10(regguess[1:])
            gridsearch = [np.linspace(λs[0] - 1, λs[0] + 1, 11),
                          np.linspace(λs[1] - 1, λs[1] + 1, 11)]

        gridsize = np.prod([len(grid) for grid in gridsearch])
        print(f"\n**** GRID SEARCH **** "
              f"({gridsize:d} trials)")
        errs = {}
        log10_grid2, log10_grid3 = gridsearch
        λ1 = 0
        product_grid = itertools.product(log10_grid2, log10_grid3)
        for ind, (λ2, λ3) in enumerate(product_grid):
            λs = (λ1, λ2, λ3)
            errs[λs] = training_error_from_rom(np.array(λs))
            print(f"(Trial {ind+1:d}, rs={(r1, r2)})")
        err2λs = {e: λs for λs, e in errs.items()}
        λs = err2λs[min(err2λs.keys())]
        if not self._one_works:
            print("grid search failed! (no stable ROMs)")
            return rom
        print(f"\n**** GRID SEARCH WINNER **** : {10**np.array(λs)}")
        print(f"AgvRelError: {errs[λs]:.2%}")

        # Run the optimization.
        print("\n**** OPTIMIZATION-BASED SEARCH ****")
        λ1 = λs[0]

        def to_minimize(twohyperparams):
            arg = np.concatenate(([λ1], twohyperparams))
            return training_error_from_rom(arg)

        opt_result = opt.minimize(to_minimize, np.array(λs[1:]),
                                  method="Nelder-Mead",
                                  options=dict(maxiter=100,
                                               fatol=1e-3, xatol=1e-3))

        # Extract the result.
        if not opt_result.success:
            print("WARNING: optimizer 'unsuccessful'")
        if opt_result.fun != _MAXFUN:
            λs = np.concatenate(([0], 10**opt_result.x))
            print(f"\n**** OPTIMIZATION-BASED SEARCH WINNER **** : {λs}")
            print(f"Average Relative Error: {opt_result.fun:.2%}")
            rom._evaluate_solver(λs)
            rom.reg = λs
            return rom
        else:
            print("Regularization search optimization FAILED")
