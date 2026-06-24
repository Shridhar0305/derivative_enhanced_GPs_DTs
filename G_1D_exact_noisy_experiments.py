import numpy as np
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Cov import GaussianCovariance_generic
from meas import PointMeasurement,dPointMeasurement,ddPointMeasurement
from meas import dddPointMeasurement,ddddPointMeasurement
import argparse
from scipy.optimize import minimize
import scipy
import pyoti.sparse as oti


def parse_commandline():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", help="α", type=float, default=1.0)
    parser.add_argument("--m", help="m", type=int, default=3)
    # parser.add_argument("--kernel", type=str, default="Matern5half")
    parser.add_argument("--kernel", type=str, default="Gaussian")
    parser.add_argument("--sigma", help="lengthscale", type=float, default=0.14)
    parser.add_argument("--h", help="grid size", type=float, default=10)
    parser.add_argument("--nugget", type=float, default=1e-5)
    parser.add_argument("--GNsteps", type=int, default=3)
    parser.add_argument("--rho_big", type=float, default=2.0)
    parser.add_argument("--rho_small", type=float, default=2.0)
    parser.add_argument("--k_neighbors", type=int, default=2)
    parser.add_argument("--compare_exact", type=bool, default=False)
    return parser.parse_args()
#     args, _ = parser.parse_known_args()
#     return args

def rhs(x, der_ind):
    # oti perturbation
    d=x.shape[0]
    n=x.shape[1]
    x_oti=oti.zeros([d,n])
    for i in range(d):
        x_oti[i,:]=oti.transpose(oti.array(x[i].T)+oti.e([i+1],order=4))
    oti_eqn=((x_oti[0]**2)/4000)-oti.cos(x_oti[0])+1

    ans=np.zeros(n*len(der_ind))
    for i in range(len(der_ind)):
        ans[i*n:(i*n)+n]=oti_eqn.get_deriv(der_ind[i]).flatten()
          
    return ans


def get_Gram_matrices(cov, X_domain,X_test, sol_now):
    d = X_domain.shape[0]
    N_domain = X_domain.shape[1]
    # get linearized PDEs correponding measurements
    meas_0 = [PointMeasurement(X_domain[:, i]) for i in range(N_domain)]
    meas_d=[dPointMeasurement(X_domain[:, i],j) for j in range(d) for i in range(N_domain)]
    meas_dd=[ddPointMeasurement(X_domain[:, i],(j,j)) for j in range(d) for i in range(N_domain)]
    meas_ddd=[dddPointMeasurement(X_domain[:, i],(j,j,j)) for j in range(d) for i in range(N_domain)]
    meas_dddd=[ddddPointMeasurement(X_domain[:, i],(j,j,j,j)) for j in range(d) for i in range(N_domain)]
    measurements=meas_0+meas_d+meas_dd+meas_ddd+meas_dddd
    
#     noise_levels = [0.05, 0.04, 0.03, 0.02, 0.01]
    noise_levels = [0.01, 0.02, 0.03, 0.04, 0.05]
    block_sizes = [len(meas_0), len(meas_d), len(meas_dd), len(meas_ddd), len(meas_dddd)]
    Theta_train = np.zeros((len(measurements),len(measurements)))
    
    for i in range(len(measurements)):
        for j in range(i,len(measurements)):
            Theta_train[i,j]=cov(measurements[i],measurements[j])
            if i != j:
                Theta_train[j, i] = Theta_train[i, j]
                
    start_idx = 0
    for blk_idx, blk_size in enumerate(block_sizes):
        end_idx = start_idx + blk_size
        Theta_train[start_idx:end_idx, start_idx:end_idx] += \
            (noise_levels[blk_idx] ** 2) * np.eye(blk_size)
        start_idx = end_idx

    N_test= X_test.shape[1]
    meas_test=[PointMeasurement(X_test[:, i]) for i in range(N_test)]
    Theta_test = np.zeros((len(meas_test),len(measurements)))

    for i in range(len(meas_test)):
        for j in range(len(measurements)):
            Theta_test[i,j]=cov(meas_test[i],measurements[j])
        

    return Theta_train,Theta_test


def iterGPR_exact(cov, X_domain,X_test, sol_init, nugget: float) -> np.ndarray:
    N_domain = X_domain.shape[1]
    d = X_domain.shape[0]

    Theta_train, Theta_test = get_Gram_matrices(cov, X_domain, X_test,sol_init)

    weights=np.linalg.solve((Theta_train + nugget * np.eye(Theta_train.shape[0])),sol_init)
    
    eg,ev=np.linalg.eig(Theta_train)
    condA = np.linalg.cond(Theta_train)
    max_eig = np.max(np.abs(eg))
    min_eig = np.min(np.abs(eg))


    print("\n--- Conditioning Information ---")
    print(f"Condition number: {condA:.3e}")
    print(f"Largest eigenvalue: {max_eig:.3e}")
    print(f"Smallest eigenvalue: {min_eig:.3e}")
    print(f"Eigenvalue ratio (max/min): {max_eig/min_eig:.3e}")
    print("--------------------------------\n")
    
    v=Theta_test @ weights

    return v,condA,np.max(eg),np.min(eg)


if __name__=="__main__":
    args = parse_commandline()
    h = int(args.h)
    rg=np.pi
    x = np.linspace(-rg, rg, h).reshape(-1,1)
    data_test=np.transpose(np.loadtxt("test_G1_pi.csv",delimiter=",",encoding='utf-8-sig').reshape(-1,1))
    data=x.T
    der_indices=[[0], [1], [1, 1], [1, 1, 1], [1, 1, 1, 1]]

    truth=rhs(data,[0])
    truth_with_der=rhs(data,der_indices)
    print("truth_with_der",np.shape(truth_with_der))

    print(np.shape(data_test))

    test_truth=rhs(data_test,[0])
    N_domain=np.shape(truth_with_der)
    sol_init=np.zeros(N_domain)
    nugget = args.nugget
    GNsteps = args.GNsteps

    rho=args.rho_big
    k_neighbors = args.k_neighbors
    
    def run_exact(pa):
        len_sc,nugget=pa
        cov = GaussianCovariance_generic(len_sc)
        sol_1,cond,max_e,min_e=iterGPR_exact(cov,data,data_test, truth_with_der,nugget)
        ac=np.mean((sol_1-test_truth)**2)
        return ac
    

    bounds = [(0.0001, 100), (1e-12, 1e-8)]
    num_len_sc_points = 50
    num_nugget_points = 1

    len_sc_values = np.linspace(1, 10, num_len_sc_points)
    nugget_values = np.logspace(np.log10(bounds[1][0]), np.log10(bounds[1][1]), num_nugget_points)

    # --- Run exact optimization with grid search for initial guess ---
    print("\n\n--- Running optimization for run_exact ---")
    print(f"--- Performing grid search for initial guess ({num_len_sc_points}x{num_nugget_points} points) ---")
    
    best_ig_exact = None
    best_fun_exact_grid = np.inf

    for ls in len_sc_values:
        for n in nugget_values:
            print(f"  Testing: lengthscale={ls}, nugget={n}")
            fun = run_exact([ls, n])
            if fun < best_fun_exact_grid:
                best_fun_exact_grid = fun
                best_ig_exact = [ls, n]
    
    print(f"\nBest initial guess from grid search for run_exact: {best_ig_exact} with MSE: {best_fun_exact_grid}")

    print("\n--- Running optimization for run_exact with best initial guess ---")
    best_result_exact = scipy.optimize.minimize(run_exact, best_ig_exact, bounds=bounds)

    print("\n\n--- run_exact Optimization Complete ---")
    if best_result_exact.success:
        print("Best result found for run_exact:")
        print(best_result_exact)
    else:
        print("Optimization failed for run_exact.")
        print(best_result_exact)
