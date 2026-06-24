import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from SGP.Cov import GaussianCovariance_generic
from SGP.Factors import ExplicitKLFactorization,ImplicitKLFactorization
from SGP.meas import PointMeasurement,dPointMeasurement,ddPointMeasurement,dddPointMeasurement,ddddPointMeasurement
from scipy.sparse.linalg import LinearOperator,spsolve_triangular
import scipy.sparse.linalg as lg
import argparse
import pyoti.sparse as oti
from SGP.supernode_converter import convert_measurements_to_list_of_dicts
import time
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import cg


def parse_commandline():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma", help="lengthscale", type=float, default=4.0)
    parser.add_argument("--h", help="grid size", type=float, default=10)
    parser.add_argument("--nugget", type=float, default=1e-5)
    parser.add_argument("--rho_big", type=float, default=10.0)
    parser.add_argument("--rho_small", type=float, default=10.0)
    parser.add_argument("--k_neighbors", type=int, default=1)
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

def incomplete_cholesky_zero_fill(A, S):
    n = A.shape[0]
    L = np.zeros_like(A)

    for i in range(n):
        sum_diag = 0.0
        for k in range(i):
            if S[i, k]:
                sum_diag += L[i, k] ** 2
        L[i, i] = np.sqrt(max(A[i, i] - sum_diag, 0.0))

        for j in range(i + 1, n):
            if not S[j, i]:
                continue
            sum_off = 0.0
            for k in range(i):
                if S[j, k] and S[i, k]:
                    sum_off += L[j, k] * L[i, k]
            if L[i, i] > 1e-12:
                L[j, i] = (A[j, i] - sum_off) / L[i, i]
            else:
                L[j, i] = 0.0
    return L


def add_heterogeneous_noise_and_factorize(K_der, noise_blocks, bandwidth=2):
    n = K_der.shape[0]
    R_diag = np.zeros(n)
    for (start, end, sigma2) in noise_blocks:
        R_diag[start:end] = sigma2
    R_inv_sqrt = np.zeros_like(R_diag)
    R_inv_sqrt[R_diag > 0] = 1.0 / np.sqrt(R_diag[R_diag > 0])
    R_inv_sqrt_mat = np.diag(R_inv_sqrt)
    K_tilde = R_inv_sqrt_mat @ K_der @ R_inv_sqrt_mat
    A = K_tilde + np.eye(n)
    S = np.abs(np.subtract.outer(np.arange(n), np.arange(n))) <= bandwidth
    L_tilde = incomplete_cholesky_zero_fill(A, S)
    L_prime = R_inv_sqrt_mat @ L_tilde

    return L_prime


def get_Gram_matrices(cov, X_domain,X_test,sol_now):
    d = X_domain.shape[0]
    N_domain = X_domain.shape[1]

    meas_0 = [PointMeasurement(X_domain[:, i]) for i in range(N_domain)]
    meas_d=[dPointMeasurement(X_domain[:, i],j) for j in range(d) for i in range(N_domain)]

    meas_dd=[ddPointMeasurement(X_domain[:, i],(j,j)) for j in range(d) for i in range(N_domain)]
    meas_ddd=[dddPointMeasurement(X_domain[:, i],(j,j,j)) for j in range(d) for i in range(N_domain)]
    meas_dddd=[ddddPointMeasurement(X_domain[:, i],(j,j,j,j)) for j in range(d) for i in range(N_domain)]

#     ind_2=[(0, 0), (0, 1), (1,1), (0,2), (1,2),(2,2)]
#     meas_dd=[]
#     for j in range(len(ind_2)):
#         for i in range(N_domain):
#             meas_dd.append(ddPointMeasurement(X_domain[:, i],ind_2[j]))

#     ind_3=[(0,0,0),(0,0,1), (0,1,1),(1,1,1), (0,0,2), (0,1,2) , (1,1,2),(0,2,2), (1,2,2) , (2,2,2)]
#     meas_ddd=[]
#     for j in range(len(ind_3)):
#         for i in range(N_domain):
#             meas_ddd.append(dddPointMeasurement(X_domain[:, i],ind_3[j]))

#     ind_4=[(0,0,0,0),(0,0,0,1),(0,0,1,1),(0,1,1,1),(1,1,1,1),(0,0,0,2),(0,0,1,2),(0,1,1,2),
#            (1,1,1,2),(0,0,2,2),(0,1,2,2),(1,1,2,2),(0,2,2,2),(1,2,2,2),(2,2,2,2)]
#     meas_dddd=[]
#     for j in range(len(ind_4)):
#         for i in range(N_domain):
#             meas_dddd.append(ddddPointMeasurement(X_domain[:, i],ind_4[j]))

    measurements=meas_0+meas_d+meas_dd+meas_ddd+meas_dddd
    
    Theta_train = np.zeros((len(measurements),len(measurements)))
    for i in range(len(measurements)):
        for j in range(i,len(measurements)):
            Theta_train[i,j]=cov(measurements[i],measurements[j])
            if i != j:
                Theta_train[j, i] = Theta_train[i, j]

    N_test= X_test.shape[1]
    meas_test=[PointMeasurement(X_test[:, i]) for i in range(N_test)]
    Theta_test = np.zeros((len(meas_test),len(measurements)))

    for i in range(len(meas_test)):
        for j in range(len(measurements)):
            Theta_test[i,j]=cov(meas_test[i],measurements[j])

    Theta_PP=np.zeros((len(meas_test),len(meas_test)))
    for i in range(len(meas_test)):
        for j in range(len(meas_test)):
            Theta_PP[i,j]=cov(meas_test[i],meas_test[j])


    return Theta_train,Theta_test,Theta_PP


def iterGPR_exact(cov, X_domain,X_test, sol_init, nugget: float):

    Theta_train, Theta_test, Theta_PP = get_Gram_matrices(cov, X_domain,X_test, sol_init)

    in_TT=np.linalg.inv(Theta_train)

    weights=np.linalg.solve((Theta_train + nugget * np.eye(Theta_train.shape[0])),sol_init)
    v=Theta_test @ weights
    cov_pp=Theta_PP-Theta_test @ (in_TT @ Theta_test.T)

    return v, np.diag(cov_pp)

    
# @profile
def fast_gpt_cg(cov,X_dom,X_test,rhs_with_der, sol_init,nugget,rho=4.0,k_neighbors=3,lamb=1.5,alpha=1.0):
    N=np.shape(X_dom)[1]

    d=np.shape(X_dom)[0]

    meas_0=[PointMeasurement(X_dom[:, i]) for i in range(N)]
    meas_d=[dPointMeasurement(X_dom[:, i],j) for j in range(d) for i in range(N)]
    
    meas_dd=[ddPointMeasurement(X_dom[:, i],(j,j)) for j in range(d) for i in range(N)]
    meas_ddd=[dddPointMeasurement(X_dom[:, i],(j,j,j)) for j in range(d) for i in range(N)]
    meas_dddd=[ddddPointMeasurement(X_dom[:, i],(j,j,j,j)) for j in range(d) for i in range(N)]

#     ind_2=[(0, 0), (0, 1), (1,1), (0,2), (1,2),(2,2)]
#     meas_dd=[]
#     for j in range(len(ind_2)):
#         for i in range(N):
#             meas_dd.append(ddPointMeasurement(X_dom[:, i],ind_2[j]))

#     ind_3=[(0,0,0),(0,0,1), (0,1,1),(1,1,1), (0,0,2), (0,1,2) , (1,1,2),(0,2,2), (1,2,2) , (2,2,2)]
#     meas_ddd=[]
#     for j in range(len(ind_3)):
#         for i in range(N):
#             meas_ddd.append(dddPointMeasurement(X_dom[:, i],ind_3[j]))
   

#     ind_4=[(0,0,0,0),(0,0,0,1),(0,0,1,1),(0,1,1,1),(1,1,1,1),(0,0,0,2),(0,0,1,2),(0,1,1,2),
#            (1,1,1,2),(0,0,2,2),(0,1,2,2),(1,1,2,2),(0,2,2,2),(1,2,2,2),(2,2,2,2)]
#     meas_dddd=[]
#     for j in range(len(ind_4)):
#         for i in range(N):
#             meas_dddd.append(ddddPointMeasurement(X_dom[:, i],ind_4[j]))
    meas=[meas_0, meas_d, meas_dd, meas_ddd, meas_dddd]

    print("before implicir")
    implicit_factor,_,_=ImplicitKLFactorization.implicit_kl_factorization_for_d(cov,meas,rho,k_neighbors)
    print(implicit_factor)
    print(len(implicit_factor.supernodes.supernodes))
    print("after implicit")
    explicit_factor=ExplicitKLFactorization.Explicit_from_implicit(implicit_factor,nugget=nugget)

    U=explicit_factor.U
    L = U.transpose().tocsc() 
    P = explicit_factor.P
    
    block_sizes = [
        len(meas_0),
        len(meas_d),
        len(meas_dd),
        len(meas_ddd),
        len(meas_dddd)
    ]

    noise_levels = [0.01, 0.02, 0.03, 0.04, 0.05] 
    noise_blocks = []
    start = 0
    for size, sigma2 in zip(block_sizes, noise_levels):
        end = start + size
        noise_blocks.append((start, end, sigma2))
        start = end
    K_der_approx = (L @ U).toarray() 
    L_prime = add_heterogeneous_noise_and_factorize(K_der_approx, noise_blocks)
    Sigma_inv_approx = L_prime @ L_prime.T
    rhs_now=rhs_with_der
    
    # Θinv_rhs=np.zeros((len(meas_0)+len(meas_d)+len(meas_dd)+len(meas_ddd)+len(meas_dddd)))
    Θinv_rhs=sol_init
    Θinv_rhs[P]=U@(L @ rhs_now[P])

    tmp=np.zeros(np.shape(Θinv_rhs)[0])
    tmp[P]=lg.inv(L)@(lg.inv(U)@Θinv_rhs[P])


    # measurements for testing
    N_test=np.shape(X_test)[0]
    meas_test = [PointMeasurement(X_test[:, i]) for i in range(X_test.shape[1])]

    # build theta (prediction, train)
    meas_train=meas_0+meas_d+meas_dd+meas_ddd+meas_dddd
    reorde_meas_trian=[meas_train[i] for i in P]

    Theta_test=np.zeros((len(meas_test),len(reorde_meas_trian)))
    for i in range(len(meas_test)):
        for j in range(len(reorde_meas_trian)):
            Theta_test[i,j]=cov(meas_test[i],reorde_meas_trian[j])

    Theta_PP=np.zeros((len(meas_test),len(meas_test)))
    for i in range(len(meas_test)):
        for j in range(len(meas_test)):
            Theta_PP=cov(meas_test[i],meas_test[j])
    

    y_predicted=np.zeros(len(meas_test))

    y_predicted=Theta_test @ Θinv_rhs[P]
    cov_pred=Theta_PP-Theta_test @ (U @ (L @ Theta_test.T))
    cov_dia=np.diag(cov_pred)

    return tmp,y_predicted,cov_dia

if __name__=="__main__":

    args = parse_commandline()
    # grid size
    h = int(args.h)
    rg=np.pi
    x = np.linspace(-rg, rg, h).reshape(-1,1)
    data_test=np.transpose(np.loadtxt("test_G1_pi.csv",delimiter=",",encoding='utf-8-sig').reshape(-1,1))
    data = x.T
    der_indices = [[0], [1], [1, 1], [1, 1, 1], [1, 1, 1, 1]]
    truth=rhs(data,[0])
    truth_with_der=rhs(data,der_indices)
    
    print("truth_with_der",np.shape(truth_with_der))

    print(np.shape(data_test))
#     test_truth=rhs(data_test,[0])

    N_domain=np.shape(truth_with_der)

    lengthscale = args.sigma
    cov = GaussianCovariance_generic(lengthscale)

    sol_init=np.zeros(N_domain)
    nugget = args.nugget

    rho=args.rho_big
    k_neighbors = args.k_neighbors
    st=time.time()
    sol,sol_1,cov_pred=fast_gpt_cg(cov,data,data_test, truth_with_der,sol_init,nugget,rho=rho,k_neighbors=k_neighbors,lamb=1.5,alpha=1.0)
    print("total time",time.time()-st)

    test_truth=rhs(data_test,[0])
    print("mse",np.mean((truth_with_der - sol)**2))
    print("test_mse",np.mean((test_truth - sol_1)**2))

    if args.compare_exact==True:
        sol_exact,cov_exact=iterGPR_exact(cov,data,data_test,truth_with_der,nugget)
        print("test_exact",np.mean((test_truth - sol_exact)**2))