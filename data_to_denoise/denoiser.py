import numpy as np
import scipy.linalg
import mosek.fusion as mf

def annihilation_operator(N):
    a = np.zeros((N, N), dtype=complex)
    for n in range(1, N):
        a[n-1, n] = np.sqrt(n)
    return a

def displacement_operator(alpha, a, ad):
    A = alpha * ad - np.conjugate(alpha) * a
    return scipy.linalg.expm(A)

def parity_operator(N):
    diag = np.array([1 if n % 2 == 0 else -1 for n in range(N)], dtype=complex)
    return np.diag(diag)

def construct_E(alpha, a, ad, P):
    D = displacement_operator(alpha, a, ad)
    DP = np.dot(D, P)
    DPD = np.dot(DP, np.conjugate(D.T))
    E = 0.5 * (np.eye(P.shape[0]) + DPD)
    return E.real

def reconstruct_and_truncate(W_array, alpha_array, N_large, N_final):
    p_target = 0.5 * (1 + (np.pi / 2.0) * np.array(W_array))
    a = annihilation_operator(N_large)
    ad = np.conjugate(a.T)
    P = parity_operator(N_large)
    E_list = []
    for alpha in alpha_array:
        E = construct_E(alpha, a, ad, P)
        E_list.append(E)
    num_obs = len(E_list)
    with mf.Model("density_matrix_reconstruction") as M:
        X = M.variable("rho", [N_large, N_large], mf.Domain.inPSDCone())
        M.constraint("trace", mf.Expr.trace(X), mf.Domain.equalsTo(1))
        residuals = []
        for k in range(num_obs):
            Ek = E_list[k]
            expr = mf.Expr.dot(mf.Matrix.dense(Ek), X)
            residuals.append(mf.Expr.sub(expr, p_target[k]))
        R = mf.Expr.vstack(*residuals)
        M.objective("least_squares", mf.ObjectiveSense.Minimize, mf.Expr.sumSquares(R))
        M.solve()
        rho_large = X.level()
    rho_trunc = rho_large[:N_final, :N_final]
    rho_trunc = rho_trunc / np.trace(rho_trunc)
    return rho_trunc

if __name__ == "__main__":
    W_array = np.array([0.2, -0.1, 0.05])
    alpha_array = np.array([0.1 + 0.2j, 0.3 - 0.1j, -0.2 + 0.05j])
   