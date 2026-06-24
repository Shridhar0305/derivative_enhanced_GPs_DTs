import numpy as np
import pandas as pd
import sys
import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Cov import GaussianCovariance_generic
from Factors import ExplicitKLFactorization,ImplicitKLFactorization
from meas import PointMeasurement,dPointMeasurement,ddPointMeasurement,dddPointMeasurement,ddddPointMeasurement
from Cov_g import gaussian_cov_generic as g
from supernode_converter import convert_measurements_to_list_of_dicts
import scipy.sparse.linalg as lg
import argparse
import time

def parse_commandline():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma", help="lengthscale", type=float, default=2.78)
    parser.add_argument("--h", help="no_of_points", type=float, default=27)
    parser.add_argument("--nugget", type=float, default=1e-12)
    parser.add_argument("--rho", type=float, default=3.0)
    parser.add_argument("--k_neighbors", type=int, default=1)
    parser.add_argument("--threads",type=int,default=1) # threads for parallel processing
    parser.add_argument("--type",type=str,default="points") # used for specifying the way derivatives are ordered
    # use "None" for no derivatives, "points" for grouping derivatives by points, "meas" for grouping by derivatives
    parser.add_argument("--compare_exact", type=bool, default=True)
    return parser.parse_args()

# building matrices
def get_Gram_matrices(cov, X_domain,X_test):
    d = X_domain.shape[0]
    N_domain = X_domain.shape[1]

    # different measurements for covariance calculation
    meas_0 = [PointMeasurement(X_domain[:, i]) for i in range(N_domain)]
    meas_d=[dPointMeasurement(X_domain[:, i],j) for j in range(d) for i in range(N_domain)]

    # can use the commented lines below for 1d problem
    # meas_dd=[ddPointMeasurement(X_domain[:, i],(j,j)) for j in range(d) for i in range(N_domain)]
    # meas_ddd=[dddPointMeasurement(X_domain[:, i],(j,j,j)) for j in range(d) for i in range(N_domain)]
    # meas_dddd=[ddddPointMeasurement(X_domain[:, i],(j,j,j,j)) for j in range(d) for i in range(N_domain)]

    ind_2=[(0, 0), (0, 1), (1,1), (0,2), (1,2),(2,2)]
    meas_dd=[]
    for j in range(len(ind_2)):
        for i in range(N_domain):
            meas_dd.append(ddPointMeasurement(X_domain[:, i],ind_2[j]))

    ind_3=[(0,0,0),(0,0,1), (0,1,1),(1,1,1), (0,0,2), (0,1,2) , (1,1,2),(0,2,2), (1,2,2) , (2,2,2)]
    meas_ddd=[]
    for j in range(len(ind_3)):
        for i in range(N_domain):
            meas_ddd.append(dddPointMeasurement(X_domain[:, i],ind_3[j]))

    ind_4=[(0,0,0,0),(0,0,0,1),(0,0,1,1),(0,1,1,1),(1,1,1,1),(0,0,0,2),(0,0,1,2),(0,1,1,2),
           (1,1,1,2),(0,0,2,2),(0,1,2,2),(1,1,2,2),(0,2,2,2),(1,2,2,2),(2,2,2,2)]
    meas_dddd=[]
    for j in range(len(ind_4)):
        for i in range(N_domain):
            meas_dddd.append(ddddPointMeasurement(X_domain[:, i],ind_4[j]))

    measurements=meas_0+meas_d+meas_dd+meas_ddd+meas_dddd
    N_test= X_test.shape[1]
    meas_test=[PointMeasurement(X_test[:, i]) for i in range(N_test)]
    
    cov2=g(cov.length_scale)
    measurements=convert_measurements_to_list_of_dicts(measurements)
    Theta_train=cov2.build_symmetric(measurements)
    
    meas_test=convert_measurements_to_list_of_dicts(meas_test)
    Theta_test=cov2.build_test(meas_test,measurements)

    # Code to build the matrix in pure python 
    # Theta_train = np.zeros((len(measurements),len(measurements)))
    # for i in range(len(measurements)):
    #     for j in range(i,len(measurements)):
    #         Theta_train[i,j]=cov(measurements[i],measurements[j])
    #         if i != j:
    #             Theta_train[j, i] = Theta_train[i, j]

    # Theta_test = np.zeros((len(meas_test),len(measurements)))
    # for i in range(len(meas_test)):
    #     for j in range(len(measurements)):
    #         Theta_test[i,j]=cov(meas_test[i],measurements[j])


    return Theta_train,Theta_test


def iterGPR_exact(cov, X_domain,X_test, sol_init, nugget: float):

    Theta_train, Theta_test = get_Gram_matrices(cov, X_domain,X_test)

    weights=np.linalg.solve((Theta_train + nugget * np.eye(Theta_train.shape[0])),sol_init)
    v=Theta_test @ weights
    return v

    
def fast_gpt_cg(cov,X_dom,X_test,rhs_with_der, sol_init,nugget,rho=4.0,k_neighbors=3,lamb=1.5,alpha=1.0,N_threads=1):
    N=np.shape(X_dom)[1]

    d=np.shape(X_dom)[0]

    meas_0=[PointMeasurement(X_dom[:, i]) for i in range(N)]
    meas_d=[dPointMeasurement(X_dom[:, i],j) for j in range(d) for i in range(N)]

    ind_2=[(0, 0), (0, 1), (1,1), (0,2), (1,2),(2,2)]
    meas_dd=[]
    for j in range(len(ind_2)):
        for i in range(N):
            meas_dd.append(ddPointMeasurement(X_dom[:, i],ind_2[j]))

    ind_3=[(0,0,0),(0,0,1), (0,1,1),(1,1,1), (0,0,2), (0,1,2) , (1,1,2),(0,2,2), (1,2,2) , (2,2,2)]
    meas_ddd=[]
    for j in range(len(ind_3)):
        for i in range(N):
            meas_ddd.append(dddPointMeasurement(X_dom[:, i],ind_3[j]))
   

    ind_4=[(0,0,0,0),(0,0,0,1),(0,0,1,1),(0,1,1,1),(1,1,1,1),(0,0,0,2),(0,0,1,2),(0,1,1,2),
           (1,1,1,2),(0,0,2,2),(0,1,2,2),(1,1,2,2),(0,2,2,2),(1,2,2,2),(2,2,2,2)]
    meas_dddd=[]
    for j in range(len(ind_4)):
        for i in range(N):
            meas_dddd.append(ddddPointMeasurement(X_dom[:, i],ind_4[j]))
    meas=[meas_0,meas_d,meas_dd,meas_ddd,meas_dddd]

    if args.type=="points":
        implicit_factor,_,_=ImplicitKLFactorization.implicit_kl_factorization_for_d(cov,meas,rho,k_neighbors)
    elif args.type=="meas":
        implicit_factor,_,_=ImplicitKLFactorization.implicit_kl_factorization_for_d_V2(cov,meas,rho,k_neighbors)
    else:
        print("Incorrect ordering type")
    explicit_factor=ExplicitKLFactorization.Explicit_from_implicit(implicit_factor,nugget=nugget,N_threads=N_threads)

    U=explicit_factor.U
    L = U.transpose().tocsc() # Transpose and convert to Compressed Sparse Column format
    P = explicit_factor.P


    rhs_now=rhs_with_der
    
    Θinv_rhs=sol_init
    Θinv_rhs[P]=U@(L @ rhs_now[P])


    # measurements for testing
    N_test=np.shape(X_test)[1]
    meas_test=[PointMeasurement(X_test[:, i]) for i in range(N_test)]

    # build theta (prediction, train)
    meas_train=meas_0+meas_d+meas_dd+meas_ddd+meas_dddd

    reorde_meas_trian=[meas_train[i] for i in P]

    reorde_meas_trian=convert_measurements_to_list_of_dicts(reorde_meas_trian)
    meas_test=convert_measurements_to_list_of_dicts(meas_test)

    cov2=g(cov.length_scale)

    Theta_test=cov2.build_test(meas_test,reorde_meas_trian)

    # Code for pure python
    # Theta_test=np.zeros((len(meas_test),len(reorde_meas_trian)))
    # for i in range(len(meas_test)):
    #     for j in range(len(reorde_meas_trian)):
    #         Theta_test[i,j]=cov(meas_test[i],reorde_meas_trian[j])


    

    y_predicted=np.zeros(len(meas_test))

    y_predicted=Theta_test @ Θinv_rhs[P]

    return y_predicted

if __name__=="__main__":

    args = parse_commandline()
    der_indices=[[0], [1],[2], [3], [1, 1], [1, 2], [2, 2], [1, 3], [2, 3], [3, 3], 
                [1, 1, 1], [1, 1, 2], [1, 2, 2], [2, 2, 2], [1, 1, 3], [1, 2, 3], [2, 2, 3], [1, 3, 3], [2, 3, 3],[3, 3, 3],
                [1, 1, 1, 1], [1, 1, 1, 2], [1, 1, 2, 2], [1, 2, 2, 2], [2, 2, 2, 2], [1, 1, 1, 3], [1, 1, 2, 3], [1, 2, 2, 3], 
                [2, 2, 2, 3], [1, 1, 3, 3], [1, 2, 3, 3], [2, 2, 3, 3], [1, 3, 3, 3], [2, 3, 3, 3], [3, 3, 3, 3]]

    # Reading initial training set
    n=int(args.h)
    file=pd.read_csv(f"./data/3D/train_G3_{n}.csv")
    data=np.zeros((3,n))
    data[0,:]=np.array(file['X0']).flatten()
    data[1,:]=np.array(file['X1']).flatten()
    data[2,:]=np.array(file['X2']).flatten()
    X_domain=data.T
    truth_with_der=np.zeros(n*len(der_indices))
    for i in range(len(der_indices)):
        truth_with_der[i*n:(i*n)+n]=np.array(file[str(der_indices[i])]).flatten()
    

    test_file=pd.read_csv("./data/3D/test_G3.csv")
    data_test=np.zeros((3,test_file.shape[0]))
    data_test[0,:]=np.array(test_file['X0']).flatten()
    data_test[1,:]=np.array(test_file['X1']).flatten()
    data_test[2,:]=np.array(test_file['X2']).flatten()
    X_test=data_test.T
    test_truth=np.zeros(test_file.shape[0])
    test_truth=np.array(test_file['[0]']).flatten()
    

    N_domain=np.shape(truth_with_der)

    lengthscale = args.sigma
    cov = GaussianCovariance_generic(lengthscale)

    sol_init=np.zeros(N_domain)
    nugget = args.nugget

    rho=args.rho
    k_neighbors = args.k_neighbors
    N_threads=args.threads

    sol_1=fast_gpt_cg(cov,data,data_test, truth_with_der,sol_init,nugget,rho=rho,k_neighbors=k_neighbors,lamb=1.5,alpha=1.0,N_threads=N_threads)

    print("test_mse",np.mean((test_truth - sol_1)**2))

    if args.compare_exact==True:
        sol_exact=iterGPR_exact(cov,data,data_test,truth_with_der,nugget)
        print("test_exact",np.mean((test_truth - sol_exact)**2))

