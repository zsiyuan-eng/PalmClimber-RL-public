"""
ClimbingMPC - Model Predictive Control for the palm tree climbing robot.

System model (simplified, continuous-time around upright):
    x = [h, dh, theta_x, dtheta_x, theta_y, dtheta_y]  (6 states)
    u = [w1..w6]  wheel velocities (6 inputs)

Climbing dynamics (nominal model):
    Wheel velocity creates slip relative to vertical body velocity. Slip is
    converted into traction and smoothly saturated by mu * normal force.
    Tilt moments come from force imbalance, not raw wheel speed difference.

We discretize with forward Euler at dt=0.02s and use CasADi/IPOPT to
solve the nonlinear optimal control problem (NLP).

Friction mu is the nominal value; the real environment randomizes it.
The RL residual compensates for this mismatch.
"""


import numpy as np

try:
    import casadi as cs
    _HAS_CASADI = True
except ImportError:
    _HAS_CASADI = False
    print("[ClimbingMPC] casadi not installed -- using fallback heuristic controller")


# Robot parameters
R_WHEEL  = 0.025   # wheel radius (m)
R_TREE   = 0.12    # tree radius (m)
MU_NOM   = 0.70    # nominal friction coefficient
B_DAMP   = 0.15    # linear damping (drag friction)
MASS     = 3.2     # total base-only robot mass (kg)
J_TILT   = 0.08    # moment of inertia around tilt axes (kg*m^2)
G        = 9.81
NORMAL_FORCE_TOTAL = 50.0
K_SLIP   = 60.0
L_SIDE   = 0.16
L_FB     = 0.14
K_TILT   = 3.0
C_TILT   = 0.4
WHEEL_AZIMUTHS = np.deg2rad([0, 60, 120, 180, 240, 300])
WHEEL_UNIT_X = np.cos(WHEEL_AZIMUTHS)
WHEEL_UNIT_Y = np.sin(WHEEL_AZIMUTHS)

# MPC parameters
N_HORIZON = 20     # prediction steps
DT        = 0.015  # seconds per step (= 10 x 0.0015 s MuJoCo substeps)

# Cost weights
Q_HEIGHT  = 5.0
Q_VEL     = 0.5
Q_TILT    = 20.0   # tilt is expensive -- stability first
Q_TILTRATE= 5.0
R_CTRL    = 0.001  # control effort — was 0.01, too expensive vs height benefit

U_MAX = 22.0  # rad/s MPC nominal limit; env clips final residual command at +/-25
U_MIN = -22.0


class ClimbingMPC:
    """
    Solves a finite-horizon OCP at each timestep using CasADi + IPOPT.
    Falls back to a simple proportional law if CasADi is unavailable.
    """

    def __init__(self, n_horizon=N_HORIZON, dt=DT, mu_nom=MU_NOM, verbose=False, require_casadi=False):
        self.N   = n_horizon
        self.dt  = dt
        self.mu  = mu_nom
        self._verbose = verbose
        self._solver  = None
        self._last_x_opt = None
        self._last_u_opt = None
        self._last_opt_cost = None
        self._last_solver_stats = {}
        self._last_used_fallback = True
        self._cost_expr = None
        self.require_casadi = require_casadi

        if require_casadi and not _HAS_CASADI:
            raise RuntimeError(
                "CasADi is required for this MPC run but is not installed. "
                "Install casadi in the training Python environment or run without --require-casadi."
            )

        if _HAS_CASADI:
            self._build_solver()
        else:
            print("[ClimbingMPC] running in fallback mode (proportional controller)")

    # ------------------------------------------------------------------
    def _dynamics(self, x, u):
        """
        CasADi symbolic one-step forward model.
        x: [h, dh, tx, dtx, ty, dty]
        u: [w1..w6]
        """
        h    = x[0]
        dh   = x[1]
        tx   = x[2]   # tilt around x
        dtx  = x[3]
        ty   = x[4]   # tilt around y
        dty  = x[5]

        v_wheel = R_WHEEL * u
        slip = v_wheel - dh
        f_raw = K_SLIP * slip
        f_max = self.mu * NORMAL_FORCE_TOTAL / 6.0
        forces = f_max * cs.tanh(f_raw / f_max)
        f_up = cs.sum1(forces)

        # Wheel order matches mujoco_models/scene.xml:
        # [0, 60, 120, 180, 240, 300] deg around the tree.
        tau_x = L_FB * cs.dot(forces, cs.DM(WHEEL_UNIT_Y))
        tau_y = L_SIDE * cs.dot(forces, cs.DM(WHEEL_UNIT_X))

        # state derivatives
        ddh  = (f_up - MASS * G - B_DAMP * dh) / MASS
        ddtx = (tau_x - K_TILT * tx - C_TILT * dtx) / J_TILT
        ddty = (tau_y - K_TILT * ty - C_TILT * dty) / J_TILT

        x_dot = cs.vertcat(dh, ddh, dtx, ddtx, dty, ddty)
        return x + self.dt * x_dot   # forward Euler

    def _build_solver(self):
        nx, nu = 6, 6
        N = self.N

        opti = cs.Opti()

        # decision variables
        X = opti.variable(nx, N + 1)
        U = opti.variable(nu, N)

        # parameters: initial state + target height
        x0     = opti.parameter(nx)
        h_ref  = opti.parameter()

        # cost
        cost = 0
        Q = cs.diag(cs.DM([Q_HEIGHT, Q_VEL, Q_TILT, Q_TILTRATE, Q_TILT, Q_TILTRATE]))
        R = R_CTRL * cs.DM.eye(nu)

        for k in range(N):
            x_ref = cs.vertcat(h_ref, 0, 0, 0, 0, 0)   # target: at height, upright, still
            dx = X[:, k] - x_ref
            cost += cs.mtimes([dx.T, Q, dx]) + cs.mtimes([U[:, k].T, R, U[:, k]])
            # dynamics constraint
            opti.subject_to(X[:, k + 1] == self._dynamics(X[:, k], U[:, k]))

        # terminal cost (heavier)
        x_ref_T = cs.vertcat(h_ref, 0, 0, 0, 0, 0)
        dx_T = X[:, N] - x_ref_T
        cost += 5 * cs.mtimes([dx_T.T, Q, dx_T])

        # control bounds
        opti.subject_to(opti.bounded(U_MIN, U, U_MAX))

        # tilt safety constraint
        opti.subject_to(opti.bounded(-0.35, X[2, :], 0.35))
        opti.subject_to(opti.bounded(-0.35, X[4, :], 0.35))

        # initial state
        opti.subject_to(X[:, 0] == x0)

        opti.minimize(cost)
        self._cost_expr = cost   # stored for sol.value() diagnostics

        opts = {
            "ipopt.print_level":   0,
            "ipopt.max_iter":      80,
            "ipopt.tol":           1e-4,
            "print_time":          False,
            "ipopt.warm_start_init_point": "yes",
        }
        opti.solver("ipopt", opts)

        self._opti   = opti
        self._X      = X
        self._U      = U
        self._x0_par = x0
        self._href   = h_ref
        self._solver = True

        # warm start at nominal climb speed so first IPOPT solve doesn't
        # get trapped in the u≈0 local minimum
        self._u_prev = np.full((nu, N), 10.0)

    # ------------------------------------------------------------------
    def solve(self, state: dict, target_height: float) -> np.ndarray:
        """
        state keys: height, velocity, tilt_x, tilt_y, tilt_rate_x, tilt_rate_y
        returns: wheel velocities array (6,)
        """
        x0_val = np.array([
            state.get("height",      0.5),
            state.get("velocity",    0.0),
            state.get("tilt_x",      0.0),
            state.get("tilt_rate_x", 0.0),
            state.get("tilt_y",      0.0),
            state.get("tilt_rate_y", 0.0),
        ])

        if not _HAS_CASADI or self._solver is None:
            self._last_u_opt = None
            self._last_x_opt = None
            self._last_opt_cost = None
            self._last_solver_stats = {}
            self._last_used_fallback = True
            return self._fallback(x0_val, target_height)

        try:
            self._opti.set_value(self._x0_par, x0_val)
            self._opti.set_value(self._href,   target_height)

            # warm start from previous solution
            self._opti.set_initial(self._U, self._u_prev)
            X_init = np.zeros((6, self.N + 1))
            X_init[0, :] = np.linspace(x0_val[0], target_height, self.N + 1)
            self._opti.set_initial(self._X, X_init)

            sol = self._opti.solve()
            u_opt = np.array(sol.value(self._U))
            x_opt = np.array(sol.value(self._X))
            self._u_prev = u_opt
            self._last_u_opt = u_opt
            self._last_x_opt = x_opt
            stats = sol.stats()
            self._last_solver_stats = {
                "success":       bool(stats.get("success", False)),
                "iter_count":    int(stats.get("iter_count", 0)),
                "return_status": str(stats.get("return_status", "")),
            }
            try:
                self._last_opt_cost = float(sol.value(self._cost_expr))
            except Exception:
                self._last_opt_cost = None
            self._last_used_fallback = False
            return u_opt[:, 0].flatten()   # first action

        except Exception as e:
            if self._verbose:
                print(f"[MPC] solver failed: {e}, using fallback")
            self._last_u_opt = None
            self._last_x_opt = None
            self._last_opt_cost = None
            self._last_solver_stats = {}
            self._last_used_fallback = True
            return self._fallback(x0_val, target_height)

    def _fallback(self, x0, h_target) -> np.ndarray:
        """Simple P controller as fallback when casadi not available."""
        h, dh, _tx, _dtx, ty, _dty = x0
        err_h   = h_target - h
        u_base  = float(np.clip(3.0 * err_h - 1.5 * dh, -U_MAX, U_MAX))
        # tilt correction: reduce speed on high-side wheels
        u_tilt_y = float(np.clip(-5.0 * ty, -5, 5))
        # wheels 1,3,5 (lower) vs 2,4,6 (upper) -- crude differential
        u = np.array([
            u_base + u_tilt_y,   # w1
            u_base - u_tilt_y,   # w2
            u_base + u_tilt_y,   # w3
            u_base - u_tilt_y,   # w4
            u_base + u_tilt_y,   # w5
            u_base - u_tilt_y,   # w6
        ])
        return np.clip(u, U_MIN, U_MAX)

    # ------------------------------------------------------------------
    def get_diagnostics(self) -> dict:
        """
        Lightweight scalar diagnostics from the last solve().
        Safe to call every env step — no heavy computation.
        """
        used_fb = self._last_used_fallback
        stats   = self._last_solver_stats

        u0_mean     = 0.0
        u0_abs_mean = 0.0
        if self._last_u_opt is not None:
            u0 = self._last_u_opt[:, 0]
            u0_mean     = float(np.mean(u0))
            u0_abs_mean = float(np.mean(np.abs(u0)))

        pred_h_final  = 0.0
        pred_h_gain   = 0.0
        pred_dh_1step = 0.0
        if self._last_x_opt is not None:
            pred_h_final  = float(self._last_x_opt[0, -1])
            pred_h_gain   = float(self._last_x_opt[0, -1] - self._last_x_opt[0, 0])
            pred_dh_1step = float(self._last_x_opt[0, 1]  - self._last_x_opt[0, 0])

        return {
            "mpc_solver_success":  int(stats.get("success", False)),
            "mpc_solver_iters":    int(stats.get("iter_count", 0)),
            "mpc_used_fallback":   int(used_fb),
            "mpc_u0_mean":         u0_mean,
            "mpc_u0_abs_mean":     u0_abs_mean,
            "mpc_pred_h_gain":     pred_h_gain,
            "mpc_pred_h_final":    pred_h_final,
            "mpc_pred_dh_1step":   pred_dh_1step,
        }

    def compute_cost_breakdown(self, target_height: float) -> dict:
        """
        Per-component cost from the last successful solve.
        EXPENSIVE — call only in debug scripts, never during training.
        Returns {} if no successful solve is cached.
        """
        if self._last_x_opt is None or self._last_u_opt is None:
            return {}

        x   = self._last_x_opt   # (6, N+1)
        u   = self._last_u_opt   # (6, N)
        N   = u.shape[1]
        h_ref = float(target_height)

        height_cost   = float(sum(Q_HEIGHT   * (x[0, k] - h_ref) ** 2 for k in range(N)))
        vel_cost      = float(sum(Q_VEL      *  x[1, k] ** 2           for k in range(N)))
        tilt_cost     = float(sum(Q_TILT     * (x[2, k] ** 2 + x[4, k] ** 2) for k in range(N)))
        tiltrate_cost = float(sum(Q_TILTRATE * (x[3, k] ** 2 + x[5, k] ** 2) for k in range(N)))
        control_cost  = float(sum(R_CTRL     *  float(np.sum(u[:, k] ** 2))   for k in range(N)))

        # terminal cost (5× weight applied inside the MPC)
        term_h    = 5 * Q_HEIGHT   * (x[0, N] - h_ref) ** 2
        term_v    = 5 * Q_VEL      *  x[1, N] ** 2
        term_tilt = 5 * Q_TILT     * (x[2, N] ** 2 + x[4, N] ** 2)
        term_tr   = 5 * Q_TILTRATE * (x[3, N] ** 2 + x[5, N] ** 2)
        terminal_cost = float(term_h + term_v + term_tilt + term_tr)

        total = height_cost + vel_cost + tilt_cost + tiltrate_cost + control_cost + terminal_cost

        return {
            "height_cost":    height_cost,
            "vel_cost":       vel_cost,
            "tilt_cost":      tilt_cost,
            "tiltrate_cost":  tiltrate_cost,
            "control_cost":   control_cost,
            "terminal_cost":  terminal_cost,
            "total_cost":     total,
            "stored_opt_cost": self._last_opt_cost,
            "pred_h_traj":    self._last_x_opt[0, :].tolist(),
            "pred_u0":        self._last_u_opt[:, 0].tolist(),
        }

    # ------------------------------------------------------------------
    def predict_trajectory(self, state: dict, target_height: float, steps=None):
        """
        Return the optimized MPC state trajectory when the NLP solver succeeds.

        If CasADi/IPOPT is unavailable or the solve falls back, this returns a
        nominal rollout that repeats the first fallback control.
        """
        steps = self.N if steps is None else int(steps)
        x0 = np.array([
            state.get("height", 0.5), state.get("velocity", 0.0),
            state.get("tilt_x", 0.0), state.get("tilt_rate_x", 0.0),
            state.get("tilt_y", 0.0), state.get("tilt_rate_y", 0.0),
        ])
        u = self.solve(state, target_height)
        if self._last_x_opt is not None:
            n_steps = min(steps, self._last_x_opt.shape[1] - 1)
            return self._last_x_opt[:, : n_steps + 1].T

        traj = [x0.copy()]
        for _ in range(steps):
            # Fallback visualization only: repeat the first control.
            x = traj[-1].copy()
            h, dh, tx, dtx, ty, dty = x
            v_wheel = R_WHEEL * u
            slip = v_wheel - dh
            f_raw = K_SLIP * slip
            f_max = self.mu * NORMAL_FORCE_TOTAL / 6.0
            forces = f_max * np.tanh(f_raw / max(f_max, 1e-6))
            f_up = np.sum(forces)
            tau_x = L_FB * float(np.dot(forces, WHEEL_UNIT_Y))
            tau_y = L_SIDE * float(np.dot(forces, WHEEL_UNIT_X))
            ddh  = (f_up - MASS * G - B_DAMP * dh) / MASS
            x[0] += self.dt * dh
            x[1] += self.dt * ddh
            x[2] += self.dt * dtx
            x[3] += self.dt * ((tau_x - K_TILT * tx - C_TILT * dtx) / J_TILT)
            x[4] += self.dt * dty
            x[5] += self.dt * ((tau_y - K_TILT * ty - C_TILT * dty) / J_TILT)
            traj.append(x.copy())
        return np.array(traj)
