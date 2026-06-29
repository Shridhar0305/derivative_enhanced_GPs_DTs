import numpy as np
import pandas as pd
import sys
import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Cov import GaussianCovariance_generic
from Factors import ExplicitKLFactorization,ImplicitKLFactorization
from meas import PointMeasurement,dPointMeasurement,ddPointMeasurement,dddPointMeasurement,ddddPointMeasurement
from supernode_converter import convert_measurements_to_list_of_dicts
from Cov_g import gaussian_cov_generic as g
import scipy.sparse.linalg as lg
import argparse
from scipy.optimize import minimize
import copy
from scipy.spatial.distance import cdist



def parse_commandline():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma", help="lengthscale", type=float, default=4.0)
    parser.add_argument("--h", help="grid size", type=float, default=10)
    parser.add_argument("--nugget", type=float, default=1e-10)
    parser.add_argument("--rho", type=float, default=10.0)
    parser.add_argument("--k_neighbors", type=int, default=1)
    parser.add_argument("--threads",type=int,default=1) # threads for parallel processing
    parser.add_argument("--type",type=str,default="points") # used for specifying the way derivatives are ordered
    parser.add_argument("--compare_exact", type=bool, default=False)
    return parser.parse_args()

def compute_sparsity(matrix):
    total_elements = matrix.shape[0] * matrix.shape[1]
    nonzero_elements = matrix.nnz
    sparsity = 1.0 - (nonzero_elements / total_elements)
    return sparsity



def find_outliers_with_knn_dist(existing_points, new_points, k,pp):
    """
    Finds outliers using the k-NN distance method.
    """
    # Ensure inputs are NumPy arrays
    existing_points = np.asarray(existing_points)
    new_points = np.asarray(new_points)

    dist_matrix = cdist(existing_points, existing_points)
    
    np.sort(dist_matrix, axis=1)
    knn_distances_existing = np.sort(dist_matrix, axis=1)[:, k]

    outlier_threshold = np.percentile(knn_distances_existing, pp)
    

    results = []
    dist_new_to_existing = cdist(new_points, existing_points)
    
    for i, point_distances in enumerate(dist_new_to_existing):
        point_distances.sort()
        knn_dist_new = point_distances[k-1]
        
        # Compare to the threshold
        is_outlier = knn_dist_new > outlier_threshold
        
        results.append({
            'point': new_points[i].tolist(),
            'knn_dist': round(knn_dist_new, 2),
            'is_outlier': is_outlier
        })
        
    return results


def fast_gpt_cg(cov,X_dom,X_test,rhs_with_der, sol_init,nugget,rho=4.0,k_neighbors=3,N_threads=1,lamb=1.5,alpha=1.0):
    """
    Initial training
    """
    N=np.shape(X_dom)[1]

    d=np.shape(X_dom)[0]

    meas_0=[PointMeasurement(X_dom[:, i]) for i in range(N)]
    meas_d=[dPointMeasurement(X_dom[:, i],j) for j in range(d) for i in range(N)]

    ind_2=[(0, 0), (0, 1), (1,1)]
    meas_dd=[]
    for j in range(len(ind_2)):
        for i in range(N):
            meas_dd.append(ddPointMeasurement(X_dom[:, i],ind_2[j]))


    meas=[meas_0,meas_d,meas_dd]

    if args.type=="points":
        implicit_factor,SN_wot_der, P_wot_der,L_wot_der=ImplicitKLFactorization.implicit_kl_factorization_for_d_with_partial_split_approach_2(cov,meas,rho,k_neighbors,lambda_=lamb)

    else:
        print("Type not supported for dynamic update")

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
    meas_train=meas_0+meas_d+meas_dd

    reorde_meas_trian=[meas_train[i] for i in P]

    # If building the matrix using python
    # Theta_test=np.zeros((len(meas_test),len(reorde_meas_trian)))
    # for i in range(len(meas_test)):
    #     for j in range(len(reorde_meas_trian)):
    #         Theta_test[i,j]=cov(meas_test[i],reorde_meas_trian[j])


    # build the matrices using cython
    reorde_meas_trian=convert_measurements_to_list_of_dicts(reorde_meas_trian)
    meas_test=convert_measurements_to_list_of_dicts(meas_test)

    cov2=g(cov.length_scale)

    Theta_test=cov2.build_test(meas_test,reorde_meas_trian)

    y_predicted=np.zeros(len(meas_test))

    y_predicted=Theta_test @ Θinv_rhs[P]

    return y_predicted, explicit_factor, SN_wot_der,P_wot_der,L_wot_der

def new_fast_gpt(cov,EF_old,X_dom,old_pts,new_pts,loc_NP,SN_without_der,P_without_der,L_without_der,X_test,rhs_with_der, sol_init,nugget,rho=4.0,k_neighbors=1,N_threads=1,lamb=1.5,alpha=1.0):
    N=np.shape(X_dom)[1]

    d=np.shape(X_dom)[0]

    meas_0=[PointMeasurement(X_dom[:, i]) for i in range(N)]
    meas_d=[dPointMeasurement(X_dom[:, i],j) for j in range(d) for i in range(N)]

    ind_2=[(0, 0), (0, 1), (1,1)]
    meas_dd=[]
    for j in range(len(ind_2)):
        for i in range(N):
            meas_dd.append(ddPointMeasurement(X_dom[:, i],ind_2[j]))

    

    r_set=[node.column_indices for node in SN_without_der if not node.fixed]
    r_set=[item for sublist in r_set for item in sublist]
    r_set=P_without_der[r_set].flatten()
    r_set=np.hstack((r_set,np.array(loc_NP).flatten()))
    new_meas=[PointMeasurement(X_dom[:, i]) for i in r_set]

    if args.type=="points":
        meas=[meas_0,meas_d,meas_dd]
        implicit_factor,SN_wot_der, P_wot_der,L_wot_der=ImplicitKLFactorization.dynamic_implicit_kl_factorization_for_d_partial_split_approach_2(cov,old_pts,new_meas,SN_without_der,P_without_der,L_without_der,r_set,meas,loc_NP,rho,k_neighbors)


    explicit_factor=ExplicitKLFactorization.dynamic_Explicit_from_implicit(implicit_factor,EF_old.U,EF_old.SO,nugget=nugget,N_threads=1)


    U=explicit_factor.U
    L = U.transpose() # Transpose and convert to Compressed Sparse Column format
    P = explicit_factor.P


    rhs_now=rhs_with_der

    Θinv_rhs=sol_init
    Θinv_rhs[P]=U@(L @ rhs_now[P])

    # measurements for testing
    N_test=np.shape(X_test)[1]
    meas_test=[PointMeasurement(X_test[:, i]) for i in range(N_test)]

    # build theta (prediction, train)
    meas_train=meas_0+meas_d+meas_dd

    reorde_meas_trian=[meas_train[i] for i in P]

    # If python version is used
    # Theta_test=np.zeros((len(meas_test),len(reorde_meas_trian)))
    # for i in range(len(meas_test)):
    #     for j in range(len(reorde_meas_trian)):
    #         Theta_test[i,j]=cov(meas_test[i],reorde_meas_trian[j])


    reorde_meas_trian=convert_measurements_to_list_of_dicts(reorde_meas_trian)
    meas_test=convert_measurements_to_list_of_dicts(meas_test)

    cov2=g(cov.length_scale)

    Theta_test=cov2.build_test(meas_test,reorde_meas_trian)

    y_predicted=np.zeros(len(meas_test))
    y_predicted=Theta_test @ Θinv_rhs[P]


    return y_predicted, explicit_factor,SN_wot_der, P_wot_der,L_wot_der


if __name__=="__main__":

    args = parse_commandline()
    
    # Reading test file
    test_file=pd.read_csv("./data/Dynamic/dynamic_test_G2.csv")
    data_test=np.zeros((2,test_file.shape[0]))
    data_test[0,:]=np.array(test_file['X_0']).flatten()
    data_test[1,:]=np.array(test_file['X_1']).flatten()
    X_test=data_test.T
    test_truth=np.zeros(test_file.shape[0])
    test_truth=np.array(test_file['[0]']).flatten()

    der_indices=[[0], [1],[2],[1,1],[1,2],[2,2]] #Index for derivatives

    # Reading initial training set
    file=pd.read_csv(f"./data/Dynamic/train_G2_pt_25_2_der.csv")
    n=file.shape[0]
    data=np.zeros((2,n))
    data[0,:]=np.array(file['X0']).flatten()
    data[1,:]=np.array(file['X1']).flatten()
    X_domain=data.T

    # Rhs with derivative
    truth_with_der=np.zeros(n*len(der_indices))
    for i in range(len(der_indices)):
        truth_with_der[i*n:(i*n)+n]=np.array(file[str(der_indices[i])]).flatten()

    # Reading new data
    new_pt_file=pd.read_csv(f"./data/Dynamic/new_2D_pt_10_2_der.csv")
    n=new_pt_file.shape[0]
    new_points=np.zeros((n,2))
    new_points[:,0]=np.array(new_pt_file['X0']).flatten()
    new_points[:,1]=np.array(new_pt_file['X1']).flatten()

    truth_with_der_new=np.zeros(n*len(der_indices))
    for i in range(len(der_indices)):
        truth_with_der_new[i*n:(i*n)+n]=np.array(new_pt_file[str(der_indices[i])]).flatten()


    # Amound of data that can be stored without re-training
    data_cost_budget=int(0.2*np.shape(data)[1])
    pp_for_outlier=90

    N_domain=np.shape(truth_with_der)
    sol_init=np.zeros(N_domain)


    lengthscale = args.sigma
    cov = GaussianCovariance_generic(lengthscale) 
    nugget = args.nugget
    rho=args.rho
    k_neighbors = args.k_neighbors

    # using Grid search to find initial guess. This is done to avoid optimizer local minimum trap
    bounds = [(0.0001, 100), (1e-12, 1e-8)]
    num_len_sc_points = 100
    num_nugget_points = 3

    len_sc_values = np.linspace(1, 10, num_len_sc_points)
    nugget_values = np.logspace(np.log10(bounds[1][0]), np.log10(bounds[1][1]), num_nugget_points)

    # function for optimizer call
    def run_fast_cg(pa):
            len_sc,nugget=pa
            cov = GaussianCovariance_generic(len_sc)
            sol_1,temp_EF,temp_SN, temp_P,_=fast_gpt_cg(cov,data,data_test, truth_with_der,sol_init,nugget,rho=rho,k_neighbors=k_neighbors,N_threads=args.threads,lamb=1.5,alpha=1.0)
            ac=np.mean((sol_1-test_truth)**2)
            return ac
    
    best_ig_fcg = None
    best_fun_fcg_grid = np.inf

    for ls in len_sc_values:
        for n in nugget_values:
            fun = run_fast_cg([ls, n])
            if fun < best_fun_fcg_grid:
                best_fun_fcg_grid = fun
                best_ig_fcg = [ls, n]

    
    print("Running initial optimizer")
    opti_len = minimize(run_fast_cg, best_ig_fcg, bounds=bounds)
    lengthscale = opti_len.x[0]
    nugget=opti_len.x[1]
    
    cov = GaussianCovariance_generic(lengthscale)
    o_len_sc=lengthscale

    sol_1,EF,SN_without_der,P_without_der,L_without_der =fast_gpt_cg(cov,data,data_test, truth_with_der,sol_init,nugget,rho=rho,k_neighbors=k_neighbors,N_threads=args.threads,lamb=1.5,alpha=1.0)

    init_spar=compute_sparsity(EF.U)
    o_err=np.mean((test_truth - sol_1)**2)
    check_err=o_err
    d_set=[node.column_indices for node in SN_without_der if not node.fixed]

    d_set=[item for sublist in d_set for item in sublist]
    dp=X_domain[P_without_der[d_set],:][0]

    old_size=np.shape(X_domain)[0]
    new_size=np.shape(new_points)[0]
    update_size=old_size

    err_all=[]
    Sp_all=[]
    err=[]
    sp_=[]
    err.append(o_err)
    sp_.append(init_spar)
    err_ratio_hit=0
    budget_count=0

    # creating combined dataframe for reading in loop
    combined_df=file
    print("Starting dynamic update")
    # Loops through data in streaming set
    for i in range(new_size):
        loc_np=[update_size]
        
        #outlier detection
        report = find_outliers_with_knn_dist(X_domain, new_points[i:i+1,:], k=k_neighbors,pp=pp_for_outlier) #always only one point will be sent
        
        # retrain based on the criteria
        if report[0]['is_outlier']==True or err_ratio_hit>=3 or budget_count>=data_cost_budget:
            print(f"Retraining triggered, outlier={report[0]['is_outlier']}, err_ratio_hit={err_ratio_hit},budget_count={budget_count}")
            err_ratio_hit=0
            budget_count=0

            # Check to make sure the new point is not in training set
            existing_points_set = {tuple(row) for row in X_domain}
            points_to_add = [p for p in new_points[:i+1] if tuple(p) not in existing_points_set]


            if points_to_add:
                X_domain_new = np.vstack((X_domain, np.array(points_to_add)))
                pairs = set(map(tuple, np.array(points_to_add)))
                df_new = new_pt_file[new_pt_file[['X0', 'X1']].apply(tuple, axis=1).isin(pairs)]
                combined_df=pd.concat([combined_df,df_new])
            else:
                # This case can happen if the outlier point was already added in a previous retraining
                X_domain_new = X_domain.copy()
            
            data_new=X_domain_new.T
            n=np.shape(data_new)[1]

            truth_with_der=np.zeros(n*len(der_indices))
            for j in range(len(der_indices)):
                truth_with_der[j*n:(j*n)+n]=np.array(combined_df[str(der_indices[j])]).flatten()

            N_domain=np.shape(truth_with_der)
            sol_init=np.zeros(N_domain)

            def run_fast_cg_retrain(pa):
                len_sc,nugget=pa
                cov = GaussianCovariance_generic(len_sc)
                sol_1,temp_EF,temp_SN, temp_P,_=fast_gpt_cg(cov,data_new,data_test, truth_with_der,sol_init,nugget,rho=rho,k_neighbors=k_neighbors,N_threads=args.threads,lamb=1.5,alpha=1.0)
                ac=np.mean((sol_1-test_truth)**2)
                return ac

            # Function with rho as a parameter for optimzation
            def run_fast_cg_retrain_rho(pa):
                len_sc,nugget,rho=pa
                cov = GaussianCovariance_generic(len_sc)
                sol_1,temp_EF,temp_SN, temp_P,_=fast_gpt_cg(cov,data_new,data_test, truth_with_der,sol_init,nugget,rho=rho,k_neighbors=k_neighbors,N_threads=args.threads,lamb=1.5,alpha=1.0)
                ac=np.mean((sol_1-test_truth)**2)
                sp=compute_sparsity(temp_EF.U)
                return ac,sp

            ig=[o_len_sc,bounds[1][0]]
            opti_len=minimize(run_fast_cg_retrain,ig,bounds=bounds)
            opti_err=opti_len.fun
            lengthscale = opti_len.x[0]
            nugget = opti_len.x[1]
            coun=0
            if opti_err > check_err:
                break_flag=False
                best_ig_fcg = None
                best_fun_fcg_grid = np.inf

                for ls in len_sc_values:
                    for n in nugget_values:
                        fun = run_fast_cg_retrain([ls, n])
                        if fun < best_fun_fcg_grid:
                            best_fun_fcg_grid = fun
                            best_ig_fcg = [ls, n]
                if best_fun_fcg_grid < check_err:
                    lengthscale = best_ig_fcg[0]
                    nugget = best_ig_fcg[1]
                    break_flag=True
                
                # if the grid search did not result in a better error, increase the rho values
                if not break_flag:
                    coun=0
                    cov = GaussianCovariance_generic(lengthscale)
                    sol_1,EF,new_SN_wot_der,new_P_wot_der,new_L_wot_der =fast_gpt_cg(cov,data_new,data_test, truth_with_der,sol_init,nugget,rho=rho,k_neighbors=k_neighbors,N_threads=args.threads,lamb=1.5,alpha=1.0)
                    sp_rho=compute_sparsity(EF.U)
                    best_ig_fcg_rho = None
                    best_fun_fcg_grid_rho = np.inf
                    best_rho=rho
                    temp_rho=rho
                    if init_spar<0.5:
                        cond_spar=init_spar
                    else:
                        cond_spar=0.9*init_spar

                    limit_spar=np.round((N_domain[0]-1)/(2*N_domain[0]),8)
                    coun=0
                    while best_fun_fcg_grid_rho > check_err and sp_rho>cond_spar and coun<5:
                        # Count check to avoid infinite loop
                        if np.round(sp_rho,8)==limit_spar:
                            coun+=1
                        temp_rho=temp_rho+1

                        # Grid search for each rho
                        for ls in len_sc_values:
                            for n in nugget_values:
                                fun_rho,sp_rho  = run_fast_cg_retrain_rho([ls, n,temp_rho])
                                if fun_rho < best_fun_fcg_grid_rho:
                                    best_fun_fcg_grid_rho = fun_rho
                                    best_ig_fcg_rho = [ls, n]
                                    best_rho=temp_rho
                    if best_fun_fcg_grid_rho <best_fun_fcg_grid:
                        rho=best_rho
                        lengthscale = best_ig_fcg_rho[0]
                        nugget = best_ig_fcg_rho[1]

            ig=[lengthscale,nugget]
            opti_len=minimize(run_fast_cg_retrain,ig,bounds=bounds)
            opti_err=opti_len.fun
            lengthscale = opti_len.x[0]
            nugget = opti_len.x[1]

            update_size = np.shape(X_domain_new)[0]
            cov = GaussianCovariance_generic(lengthscale)
            sol_1,EF,new_SN_wot_der,new_P_wot_der,new_L_wot_der =fast_gpt_cg(cov,data_new,data_test, truth_with_der,sol_init,nugget,rho=rho,k_neighbors=k_neighbors,N_threads=args.threads,lamb=1.5,alpha=1.0)
            last_err = np.mean((test_truth - sol_1)**2)

            X_domain=X_domain_new
            d_set=[node.column_indices for node in SN_without_der if not node.fixed]
            d_set=[item for sublist in d_set for item in sublist]
            dp=X_domain[P_without_der[d_set],:][0]
            SN_without_der=new_SN_wot_der
            P_without_der=np.array(new_P_wot_der)
            L_without_der=np.array(new_L_wot_der)
            last_err=np.mean((test_truth - sol_1)**2)
            check_err=last_err
            err.append(np.mean((test_truth - sol_1)**2))
            sp_.append(compute_sparsity(EF.U))

        else:
            # Dynamic update without retraining
            X_domain_new=np.vstack((X_domain,new_points[i:i+1,:]))
            dp=np.vstack((dp,new_points[i:i+1,:]))
            data_new=X_domain_new.T
            # truth_with_der=rhs(data_new,der_indices)
            n=np.shape(data_new)[1]
            combined_df_new=pd.concat([combined_df,new_pt_file[i:i+1]])
            truth_with_der=np.zeros(n*len(der_indices))
            for j in range(len(der_indices)):
                truth_with_der[j*n:(j*n)+n]=np.array(combined_df_new[str(der_indices[j])]).flatten()

            N_domain=np.shape(truth_with_der)
            sol_init=np.zeros(N_domain)
            SN_without_der1=copy.deepcopy(SN_without_der)
            sol_1_new,EF_new,new_SN_wot_der,new_P_wot_der,new_L_wot_der=new_fast_gpt(cov,EF,data_new,X_domain.T,new_points[i:i+1,:].T,
                                                                                            loc_np,SN_without_der1,P_without_der,L_without_der,data_test, truth_with_der,
                                sol_init,nugget,rho=rho,k_neighbors=k_neighbors,lamb=1.5,alpha=1.0)


            last_err=np.mean((test_truth - sol_1_new)**2)
            
            if last_err< check_err:
                print("Dynamic update accepted")
                err.append(np.mean((test_truth - sol_1_new)**2))
                sp_.append(compute_sparsity(EF_new.U))
                X_domain=X_domain_new
                combined_df=combined_df_new
                EF=EF_new
                SN_without_der=copy.deepcopy(new_SN_wot_der)
                P_without_der=np.array(new_P_wot_der)
                L_without_der=np.array(new_L_wot_der)
                update_size+=1 
                check_err=np.mean((test_truth - sol_1_new)**2)
                err_ratio_hit=0
            else:
                print("dynamic update rejected")
                try:
                    if last_err/check_err>err_ratio:
                        err_ratio=last_err/check_err
                        err_ratio_hit+=1
                except:
                    if last_err/check_err>1:
                        err_ratio=last_err/check_err
                        err_ratio_hit+=1
                budget_count+=1
                err.append(check_err)
                sp_.append(compute_sparsity(EF_new.U))
                SN_without_der=copy.deepcopy(SN_without_der)
                P_without_der=np.array(P_without_der)
                L_without_der=np.array(L_without_der)


    err_all.append(err)
    Sp_all.append(sp_)
    np.savetxt(f"./results/dynamic_2D_2_approrach_2_err.csv",err_all,delimiter=",")



