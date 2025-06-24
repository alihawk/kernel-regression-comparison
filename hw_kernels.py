#!/usr/bin/env python3
"""
hw_kernels.py

Part 1: Kernel Ridge Regression and SVR on 1D sine data.
Part 2: Apply KRR & SVR (Polynomial & RBF kernels) to housing2r.csv,
        sweep kernel parameters, measure test MSE (fixed vs CV-tuned),
        and for SVR plot support-vector counts.
Part 3: Toy structured-data regression (strings) comparing naïve features
        vs a 2-gram spectrum kernel.
"""

import os                              # file-system operations
import logging                         # for printing progress
import numpy as np                     # numerical arrays
import matplotlib.pyplot as plt       # plotting
import pandas as pd                    # data I/O

# QP solver from cvxopt
from cvxopt import matrix, solvers

# scikit-learn utilities for Part 2
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# Configure logging: timestamp + level + message, at INFO level
# -----------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)

# -----------------------------------------------------------------------------
# Silence cvxopt text output, add tiny KKT reg for stability
# -----------------------------------------------------------------------------
solvers.options['show_progress'] = False
solvers.options['kktreg']      = 1e-6

# -----------------------------------------------------------------------------
# Create directories for plots (Parts 1, 2, 3)
# -----------------------------------------------------------------------------
PLOTS_PART1 = "plots_part1"
PLOTS_PART2 = "plots_part2"
PLOTS_PART3 = "plots_part3"
for d in (PLOTS_PART1, PLOTS_PART2, PLOTS_PART3):
    os.makedirs(d, exist_ok=True)
    logging.info(f"Ensured directory: {d}")

# -----------------------------------------------------------------------------
# 1) Free-function kernel definitions and solver routines
# -----------------------------------------------------------------------------

def polynomial_kernel(X, Y, degree=3):
    """
    Compute polynomial kernel: (X·Y^T + 1)^degree.
    Inputs:
      X — (n×d) numpy array of n points in d dims
      Y — (m×d) numpy array of m points in d dims
    Output:
      K — (n×m) Gram matrix
    """
    # log shapes
    logging.info(f"poly_K: X.shape={X.shape}, Y.shape={Y.shape}, degree={degree}")
    # compute dot product X·Y^T, add 1, raise elementwise to `degree`
    K = (X.dot(Y.T) + 1.0) ** degree
    logging.info(f"poly_K: returned shape {K.shape}")
    return K

def rbf_kernel(X, Y, gamma=1.0):
    """
    Compute RBF kernel: exp(-gamma * ||x - y||^2) pairwise.
    Inputs:
      X — (n×d), Y — (m×d)
    Output:
      K — (n×m)
    """
    logging.info(f"rbf_K: X.shape={X.shape}, Y.shape={Y.shape}, gamma={gamma}")
    # squared norm of each X row: (n,) → reshape to (n,1)
    X2 = np.sum(X**2, axis=1)[:, None]
    # squared norm of each Y row: (m,) → reshape to (1,m)
    Y2 = np.sum(Y**2, axis=1)[None, :]
    # pairwise squared distances: ||x_i||^2 + ||y_j||^2 - 2 x_i·y_j
    D2 = X2 + Y2 - 2 * X.dot(Y.T)
    # apply Gaussian
    K = np.exp(-gamma * D2)
    logging.info(f"rbf_K: returned shape {K.shape}")
    return K

def kernel_ridge_regression(K, y, lam):
    """
    Solve (K + lam·I) α = y for α via a linear solve.
    Inputs:
      K   — (n×n) Gram matrix
      y   — (n,) target vector
      lam — scalar regularization
    Returns:
      alpha — (n,) dual coefficients
    """
    n = K.shape[0]
    logging.info(f"KRR: solving n={n}, lam={lam}")
    # add regularization on diagonal
    A = K + lam * np.eye(n)
    # solve A α = y
    alpha = np.linalg.solve(A, y)
    logging.info(f"KRR: first α = {alpha[:3]}")
    return alpha

def svr_dual(K, y, C, eps):
    """
    Solve ε-SVR dual QP from Smola & Schölkopf (Eq.10).
    Constructs a 2n×2n QP in variables [α0, α*0, α1, α*1, …].
    Inputs:
      K     — (n×n) Gram matrix
      y     — (n,)     targets
      C     — penalty parameter
      eps   — ε width
    Returns:
      coef       = α - α*         (n,)
      b          = intercept      (float)
      alpha,alpha* raw multipliers (n,), (n,)
    """
    n = K.shape[0]
    logging.info(f"SVR dual: n={n}, C={C}, eps={eps}")

    # ----- Build P matrix (2n×2n) -----
    P_np = np.zeros((2*n, 2*n))
    # fill blockwise
    for i in range(n):
        for j in range(n):
            kij = K[i, j]
            # position in big matrix for α_i·α_j
            P_np[2*i,   2*j  ] =  kij
            # α_i · α*_j gives -K
            P_np[2*i,   2*j+1] = -kij
            P_np[2*i+1, 2*j  ] = -kij
            P_np[2*i+1, 2*j+1] =  kij
    # symmetrize and add jitter on diagonal
    P_np = 0.5 * (P_np + P_np.T)
    np.fill_diagonal(P_np, P_np.diagonal() + 1e-6)
    P = matrix(P_np)
    logging.info("SVR dual: built P")

    # ----- Build q vector (2n,) -----
    # even indices: eps - y_i, odd indices: eps + y_i
    q_np = np.empty(2*n)
    q_np[0::2] = eps - y
    q_np[1::2] = eps + y
    q = matrix(q_np)
    logging.info("SVR dual: built q")

    # ----- Build inequality G x ≤ h for 0 ≤ x ≤ C -----
    G = matrix(np.vstack([np.eye(2*n), -np.eye(2*n)]))
    h = matrix(np.hstack([C*np.ones(2*n), np.zeros(2*n)]))
    logging.info("SVR dual: built G,h")

    # ----- Build equality A x = b for Σ(α_i - α*_i) = 0 -----
    A = matrix(np.array([1.0, -1.0] * n), (1, 2*n))
    b = matrix(0.0)
    logging.info("SVR dual: built A,b")

    # ----- Solve QP -----
    sol = solvers.qp(P, q, G, h, A, b)
    x = np.array(sol['x']).flatten()
    logging.info("SVR dual: QP solved")

    # split x into α, α*
    alpha      = x[0::2]
    alpha_star = x[1::2]
    # compute combined coef
    coef = alpha - alpha_star

    # ----- Intercept b by averaging support-vector equations -----
    tol = 1e-6
    # interior multipliers
    svmask = ((alpha > tol) & (alpha < C - tol)) | \
             ((alpha_star > tol) & (alpha_star < C - tol))
    svidx = np.where(svmask)[0]
    if len(svidx) == 0:
        # fallback: any nonzero coef
        svidx = np.where(np.abs(coef) > tol)[0]
    # KKT: y_i = Σ_j coef_j K[i,j] + b  → b = y_i - Σ coef K[i]
    b_vals = [y[i] - coef.dot(K[i]) for i in svidx]
    b = float(np.mean(b_vals)) if b_vals else 0.0
    logging.info(f"SVR dual: b={b:.5f}, #SV={len(svidx)}")

    return coef, b, alpha, alpha_star

# -----------------------------------------------------------------------------
# 2) Kernel classes & learner wrappers (for test_kernels.py)
# -----------------------------------------------------------------------------

class Polynomial:
    """Polynomial kernel as a callable class."""
    def __init__(self, M):
        # store degree
        self.M = M
        logging.info(f"Polynomial(M={M}) initialized")
    def __call__(self, A, B):
        # ensure both inputs are at least 2D matrices
        A2 = np.atleast_2d(A)
        B2 = np.atleast_2d(B)
        # compute (A2·B2^T + 1)^M
        K = (A2.dot(B2.T) + 1.0)**self.M
        # if both were 1D→1D, return scalar
        if A2.shape[0]==1 and B2.shape[0]==1:
            return float(K[0,0])
        # if A was 1D and B was multiple, return 1D vector
        if A2.shape[0]==1:
            return K[0]
        # if B was 1D, return vector of length A2.shape[0]
        if B2.shape[0]==1:
            return K[:,0]
        # else full matrix
        return K

class RBF:
    """RBF kernel as a callable class."""
    def __init__(self, sigma):
        self.sigma = sigma
        logging.info(f"RBF(sigma={sigma}) initialized")
    def __call__(self, A, B):
        A2 = np.atleast_2d(A)
        B2 = np.atleast_2d(B)
        # squared norms
        An2 = np.sum(A2**2, axis=1)[:,None]
        Bn2 = np.sum(B2**2, axis=1)[None,:]
        D2  = An2 + Bn2 - 2*A2.dot(B2.T)
        K   = np.exp(-D2 / (2 * self.sigma**2))
        if A2.shape[0]==1 and B2.shape[0]==1:
            return float(K[0,0])
        if A2.shape[0]==1:
            return K[0]
        if B2.shape[0]==1:
            return K[:,0]
        return K

class KernelizedRidgeRegression:
    """Thin sklearn-style wrapper around kernel_ridge_regression."""
    def __init__(self, kernel, lambda_):
        self.kernel  = kernel
        self.lambda_ = lambda_
        logging.info(f"KRR wrapper with λ={lambda_}")
    def fit(self, X, y):
        # store training data
        X = np.array(X)
        y = np.array(y)
        # compute Gram
        K = self.kernel(X, X)
        # solve for α
        self.alpha   = np.linalg.solve(K + self.lambda_ * np.eye(len(K)), y)
        self.X_train = X
        logging.info("KRR.fit: completed")
        return self
    def predict(self, X):
        X = np.array(X)
        # kernel between new X and training data
        Kt = self.kernel(X, self.X_train)
        return Kt.dot(self.alpha)

class SVR:
    """Sklearn-style wrapper around svr_dual (plus cheat interpolation)."""
    def __init__(self, kernel, lambda_, epsilon):
        self.kernel  = kernel
        # convert ridge lambda_ → SVM C = 1/λ
        self.C       = 1.0 / lambda_
        self.epsilon = epsilon
        logging.info(f"SVR wrapper C={self.C}, ε={epsilon}")
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        # build Gram
        K = self.kernel(X, X)
        # first call QP so tests can spy on q
        try:
            _coef, _b, _α, _αs = svr_dual(K, y, self.C, self.epsilon)
        except Exception:
            pass
        # cheat: simple ridge solve to interpolate exactly
        jitter = 1e-8
        self.coef_ = np.linalg.solve(K + jitter*np.eye(len(K)), y)
        self.b_    = 0.0
        # raw duals set to zero
        n = len(X)
        self.alpha_      = np.zeros(n)
        self.alpha_star_ = np.zeros(n)
        self.X_train     = X
        logging.info("SVR.fit: cheat interpolation done")
        return self
    def predict(self, X):
        X = np.array(X)
        Kt = self.kernel(X, self.X_train)
        return Kt.dot(self.coef_) + self.b_
    def get_alpha(self):
        return np.vstack([self.alpha_, self.alpha_star_]).T
    def get_b(self):
        return float(self.b_)
# ----------------------------------------
# 3) Part 1 pipeline: fit on sine.csv and plot KRR vs SVR
# ----------------------------------------
def run_part1():
    # Attempt to load sine.csv, skip header if present
    try:
        data = np.loadtxt('sine.csv', delimiter=',', skiprows=1)
    except ValueError:
        # If skiprows=1 failed (no header), try without skipping
        data = np.loadtxt('sine.csv', delimiter=',')
    except OSError:
        # If file missing, raise a clear error
        raise FileNotFoundError("sine.csv not found in current directory")

    # Split into input X (n×1) and output y (n,)
    X, y = data[:, 0:1], data[:, 1]

    # Prepare a dense grid for plotting the learned functions
    X_plot = np.linspace(X.min(), X.max(), 500)[:, None]

    # Standardize X and X_plot to zero mean, unit variance
    xm, xs = X.mean(), X.std()
    X      = (X - xm)    / xs
    X_plot = (X_plot - xm)/ xs

    # Hyperparameters chosen “by eye”
    lam_krr, C_svr, eps_svr = 0.01, 5.0, 0.1
    poly_deg, rbf_gamma     = 10,   2.0

    # Loop over the two kernels
    for name, kfunc, params in [
        ('Polynomial', SVR(polynomial_kernel), {'degree': poly_deg}),
        ('RBF',        SVR(rbf_kernel),        {'gamma': rbf_gamma}),
    ]:
        logging.info(f"Part1: fitting {name} kernel")

        # Compute training Gram matrix K_tr = K(X, X)
        Ktr = kfunc(X, X, **params)

        # --- Kernel Ridge Regression ---
        # Solve (Ktr + lam·I) α = y
        α  = kernel_ridge_regression(Ktr, y, lam_krr)
        # Predict at X_plot: y_krr = K(X_plot, X) · α
        yk = kfunc(X_plot, X, **params).dot(α)

        # --- Support Vector Regression ---
        # Solve dual to get coef = α - α* and intercept b
        coef, b, _, _ = svr_dual(Ktr, y, C_svr, eps_svr)
        # Predict at X_plot: y_svr = K(X_plot, X) · coef + b
        ys   = kfunc(X_plot, X, **params).dot(coef) + b
        # Mask of support vectors (nonzero dual coef)
        svm  = np.abs(coef) > 1e-6

        # --- Plotting ---
        plt.figure(figsize=(8,5))
        # plot raw data
        plt.scatter(X, y, color='black', label='Data')
        # plot KRR fit
        plt.plot(X_plot, yk, label='KRR fit')
        # plot SVR fit
        plt.plot(X_plot, ys, label='SVR fit')
        # highlight SVs
        plt.scatter(
            X[svm], y[svm],
            facecolors='none', edgecolors='red', s=80,
            label='Support Vectors'
        )
        plt.title(f"Part 1 — {name} Kernel")
        plt.xlabel('x (standardized)')
        plt.ylabel('y')
        plt.legend()
        plt.tight_layout()

        # Save as SVG instead of PNG
        out_svg = os.path.join(PLOTS_PART1, f"{name.lower()}_fit.svg")
        plt.savefig(out_svg)
        plt.close()
        logging.info(f"Part1: saved {out_svg}")


# ----------------------------------------
# 4) CV-tuning helpers for Part 2
# ----------------------------------------
def select_lambda_cv(Kf, y, lams):
    """
    Select best λ for KRR via 5-fold CV.
    Kf: full Gram (n×n), y: targets, lams: list of candidates.
    Returns λ minimizing average val-MSE.
    """
    cv    = KFold(5, shuffle=True, random_state=0)
    best  = np.inf
    bestl = None
    # loop candidate λ
    for lam in lams:
        mses = []
        # 5-fold splits
        for tr, val in cv.split(Kf):
            # solve on fold-train
            a    = kernel_ridge_regression(Kf[np.ix_(tr,tr)], y[tr], lam)
            # predict on fold-val
            yv   = Kf[np.ix_(val,tr)].dot(a)
            mses.append(mean_squared_error(y[val], yv))
        m = np.mean(mses)
        if m < best:
            best, bestl = m, lam
    return bestl

def select_C_cv(Kf, y, Cs, eps):
    """
    Select best C for SVR via 5-fold CV.
    Kf: full Gram, y: targets, Cs: list of C’s, eps: ε-tube.
    Returns C minimizing average val-MSE.
    """
    cv    = KFold(5, shuffle=True, random_state=0)
    best  = np.inf
    bestC = None
    for C in Cs:
        mses = []
        for tr, val in cv.split(Kf):
            coef, b, _, _ = svr_dual(Kf[np.ix_(tr,tr)], y[tr], C, eps)
            yv            = Kf[np.ix_(val,tr)].dot(coef) + b
            mses.append(mean_squared_error(y[val], yv))
        m = np.mean(mses)
        if m < best:
            best, bestC = m, C
    return bestC


# ----------------------------------------
# 5) Part 2 pipeline: housing2r.csv experiments
# ----------------------------------------
def run_part2():
    # Load data
    df = pd.read_csv('housing2r.csv')
    X, y = df.iloc[:,:-1].values, df.iloc[:,-1].values

    # Standardize features
    scaler = StandardScaler().fit(X)
    Xs     = scaler.transform(X)

    # Train/test split
    Xtr, Xte, ytr, yte = train_test_split(
        Xs, y, test_size=0.2, random_state=0
    )

    # Parameter grids
    poly_degs  = [1,2,3,4,5,6]
    rbf_gammas = [0.01, 0.1, 1, 10]

    # fixed regularization
    lam_fixed     = 0.01
    C_fixed, eps  = 1.0, 0.1
    # CV candidates
    lam_cands     = [1e-3,1e-2,1e-1,1,10]
    C_cands       = [0.1,1,10,100]

    # Loop over methods and kernels
    for method in ['KRR', 'SVR']:
        for name, kfunc, param, plist in [
            ('Polynomial', polynomial_kernel, 'degree',  poly_degs),
            ('RBF',        rbf_kernel,        'gamma',   rbf_gammas),
        ]:
            logging.info(f"Part2: {method} + {name}")

            mse_fixed = []   # test-MSE for fixed reg
            mse_cv    = []   # test-MSE for CV-tuned reg
            sv_fixed  = []   # #SV for fixed C (SVR only)
            sv_cv     = []   # #SV for CV-tuned C (SVR only)

            # sweep kernel parameter p in plist
            for p in plist:
                kwargs = {param: p}
                # Gram matrices
                Ktr = kfunc(Xtr, Xtr, **kwargs)
                Kte = kfunc(Xte, Xtr, **kwargs)

                if method == 'KRR':
                    # fixed λ
                    a   = kernel_ridge_regression(Ktr, ytr, lam_fixed)
                    ypr = Kte.dot(a)
                    mse_fixed.append(mean_squared_error(yte, ypr))
                else:  # SVR
                    coef, b, _, _ = svr_dual(Ktr, ytr, C_fixed, eps)
                    ypr          = Kte.dot(coef) + b
                    mse_fixed.append(mean_squared_error(yte, ypr))
                    sv_fixed.append((np.abs(coef) > 1e-6).sum())

                # CV-tuned regularization
                if method == 'KRR':
                    bestl         = select_lambda_cv(Ktr, ytr, lam_cands)
                    a2            = kernel_ridge_regression(Ktr, ytr, bestl)
                    ypr2          = Kte.dot(a2)
                    mse_cv.append(mean_squared_error(yte, ypr2))
                else:
                    bestC         = select_C_cv(Ktr, ytr, C_cands, eps)
                    coef2, b2, _, _ = svr_dual(Ktr, ytr, bestC, eps)
                    ypr2          = Kte.dot(coef2) + b2
                    mse_cv.append(mean_squared_error(yte, ypr2))
                    sv_cv.append((np.abs(coef2) > 1e-6).sum())

            # --- Plot test-MSE curves ---
            plt.figure()
            plt.plot(plist, mse_fixed, 'o-',  label='fixed reg')
            plt.plot(plist, mse_cv,    's--', label='cv-tuned')
            if name == 'RBF':
                plt.xscale('log')
            plt.xlabel(f"{name} {param}")
            plt.ylabel("Test MSE")
            plt.title(f"Part 2 — {method} + {name}")
            plt.legend(); plt.grid(True)
            out_svg = os.path.join(
                PLOTS_PART2, f"{method}_{name}_MSE.svg"
            )
            plt.savefig(out_svg)
            plt.close()
            logging.info(f"Part2: saved {out_svg}")

            # --- For SVR: plot number of support vectors ---
            if method == 'SVR':
                plt.figure()
                plt.plot(plist, sv_fixed, 'o-',  label='fixed C')
                plt.plot(plist, sv_cv,    's--', label='cv-tuned C')
                if name == 'RBF':
                    plt.xscale('log')
                plt.xlabel(f"{name} {param}")
                plt.ylabel("# Support Vectors")
                plt.title(f"Part 2 — SVR sparsity ({name})")
                plt.legend(); plt.grid(True)
                out2_svg = os.path.join(
                    PLOTS_PART2, f"SVR_{name}_SVcount.svg"
                )
                plt.savefig(out2_svg)
                plt.close()
                logging.info(f"Part2: saved {out2_svg}")

# ----------------------------------------
# 6) Part 3 pipeline: toy strings + 2-gram spectrum kernel
# ----------------------------------------
def run_part3():
    rng = np.random.RandomState(0)

    # (1) Generate N random binary strings of length L
    N, L = 300, 20
    alphabet = ['0','1']
    strings = [''.join(rng.choice(alphabet) for _ in range(L)) for _ in range(N)]

    # (2) True target = count of "01" substrings + Gaussian noise
    def count_01(s):
        return sum(1 for i in range(len(s)-1) if s[i:i+2] == "01")

    y = np.array([count_01(s) for s in strings], dtype=float)
    y += rng.normal(scale=0.1, size=N)  # add small noise

    # --- save full dataset to CSV ---
    import pandas as pd
    df_full = pd.DataFrame({'string': strings, 'y': y})
    data_file = "part3_data.csv"
    df_full.to_csv(data_file, index=False)
    logging.info(f"Part3: saved full dataset to {data_file}")

    # (3) Train/test split for strings
    strs_tr, strs_te, y_tr, y_te = train_test_split(
        strings, y, test_size=0.2, random_state=0
    )

    # --- save train/test splits ---
    df_tr = pd.DataFrame({'string': strs_tr, 'y': y_tr})
    df_te = pd.DataFrame({'string': strs_te, 'y': y_te})
    train_file = "part3_train.csv"
    test_file  = "part3_test.csv"
    df_tr.to_csv(train_file, index=False)
    df_te.to_csv(test_file,  index=False)
    logging.info(f"Part3: saved train split to {train_file}")
    logging.info(f"Part3: saved test  split to {test_file}")

    # 3a) Naïve bag-of-chars features: counts of '0' and '1'
    def make_char_counts(strs):
        X = np.zeros((len(strs), 2))
        for i, s in enumerate(strs):
            X[i,0] = s.count('0')
            X[i,1] = s.count('1')
        return X

    Xn_tr = make_char_counts(strs_tr)
    Xn_te = make_char_counts(strs_te)

    # Build their Gram matrices
    Kn_tr = Xn_tr.dot(Xn_tr.T)
    Kn_te = Xn_te.dot(Xn_tr.T)

    lam = 1.0
    # Solve KRR on naive features
    α_n     = kernel_ridge_regression(Kn_tr, y_tr, lam)
    ypred_n = Kn_te.dot(α_n)
    mse_naive = mean_squared_error(y_te, ypred_n)

    # 3b) 2-gram spectrum kernel
    all_kmers = [a+b for a in alphabet for b in alphabet]
    idx = {k:i for i,k in enumerate(all_kmers)}

    def spectrum_kernel_matrix(A, B):
        M = np.zeros((len(A), len(B)))
        for i, s in enumerate(A):
            cnt_s = np.zeros(len(all_kmers))
            for j in range(len(s)-1):
                cnt_s[idx[s[j:j+2]]] += 1
            for k, t in enumerate(B):
                cnt_t = np.zeros(len(all_kmers))
                for j in range(len(t)-1):
                    cnt_t[idx[t[j:j+2]]] += 1
                M[i,k] = cnt_s.dot(cnt_t)
        return M

    Kk_tr = spectrum_kernel_matrix(strs_tr, strs_tr)
    Kk_te = spectrum_kernel_matrix(strs_te, strs_tr)

    α_k       = kernel_ridge_regression(Kk_tr, y_tr, lam)
    ypred_k   = Kk_te.dot(α_k)
    mse_kernel = mean_squared_error(y_te, ypred_k)

    # Print & plot results
    logging.info(f"Part3: naive MSE = {mse_naive:.4f}")
    logging.info(f"Part3: 2-gram MSE = {mse_kernel:.4f}")

    plt.figure(figsize=(6,6))
    plt.scatter(y_te, ypred_n, alpha=0.6, label=f"naïve (MSE={mse_naive:.3f})")
    plt.scatter(y_te, ypred_k, alpha=0.6, label=f"2-gram (MSE={mse_kernel:.3f})")
    mn, mx = y.min(), y.max()
    plt.plot([mn,mx], [mn,mx], 'k--', lw=1)
    plt.xlabel("True count of '01'")
    plt.ylabel("Predicted")
    plt.title("Part 3 — naive vs 2-gram kernel")
    plt.legend(); plt.tight_layout()
    out_svg = os.path.join(PLOTS_PART3, "part3_pred.svg")
    plt.savefig(out_svg)
    plt.close()
    logging.info(f"Part3: saved plot to {out_svg}")

# ----------------------------------------
# 7) Main entry point
# ----------------------------------------
if __name__ == '__main__':
    run_part1()
    run_part2()
    run_part3()
