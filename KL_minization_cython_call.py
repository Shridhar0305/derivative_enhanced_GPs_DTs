import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.sparse import csc_matrix
from joblib import Parallel, delayed
import time
from Cov_g import gaussian_cov_generic as g
from supernode_converter import convert_measurements_to_list_of_dicts


def _create_U_indices(supernodes):
    I = []
    J = []
    for node in supernodes:
        for i in node.row_indices_p():
            for j in node.column_indices_p():
                if i <= j:
                    I.append(i)  # 0-based
                    J.append(j)  # 0-based
    return I, J


def assemble_for_node(node, solve_triangular_output, U_data, U_indptr):
    """
    Worker function to assemble the results for a single supernode.
    This function modifies U_data in place.
    """
    for k, index in enumerate(node.column_indices_p()):
        range_of_nnzs = range(U_indptr[index], U_indptr[index+1])
        # This slice-based assignment is the core of the assembly
        U_data[range_of_nnzs] = solve_triangular_output[:len(range_of_nnzs), k]


def factorize_node(𝒢, node, meas, indptr, nugget):
    n_rows = node.size(0)
    n_columns = node.size(1)


    local_buffer_L = np.zeros((n_rows, n_rows))
    local_buffer_U = np.zeros((n_rows, n_columns))
    
    local_buffer_m = [meas[i] for i in node.row_indices_p()]


    for j, index_zero_based in enumerate(node.column_indices_p()):

        local_row_index = indptr[index_zero_based + 1]-indptr[index_zero_based]
        if local_row_index >= 1 and local_row_index <= local_buffer_U.shape[0]: 
            local_buffer_U[local_row_index-1, j] = 1
        
    st=time.time()
    ls=𝒢.length_scale


    meas_list=convert_measurements_to_list_of_dicts(local_buffer_m[:n_rows])



    cov2=g(ls)

    temp=None
    temp = cov2.build_symmetric(meas_list)
    local_buffer_L = np.asarray(temp)
    del temp, meas_list, cov2

    try:
        np.fill_diagonal(local_buffer_L, np.diag(local_buffer_L) + nugget)

        chol = cholesky(local_buffer_L, lower=False)

        solve_triangular_output=solve_triangular(chol, local_buffer_U, lower=False, overwrite_b=False,check_finite=False) # return to variable

        return solve_triangular_output

    except:

        print("Singular matrix at factorization")
        
        solve_triangular_output=np.zeros((n_rows,n_columns))

        return solve_triangular_output


def factorize(𝒢, supernodal_assignment, nugget=0.0, N_threads=1, N=None):
    """
    Computes the Cholesky factorization using supernodes and threads.
    """
    if N is None:
        N = len(supernodal_assignment.measurements)  # Number of original points

    I, J = _create_U_indices(supernodal_assignment.supernodes)


    U = csc_matrix((np.zeros(len(I)), (I, J)), shape=(N, N))
    solve_out=[]

    results = Parallel(n_jobs=N_threads)(
        delayed(factorize_node)(𝒢, node, supernodal_assignment.measurements, U.indptr, nugget)
        for node in supernodal_assignment.supernodes
    )
    if results:
        solve_out = results
    else:
        solve_out = [], []

    solve_out = list(solve_out)

    Parallel(n_jobs=N_threads, require='sharedmem')(
    delayed(assemble_for_node)(node, solve_out[i], U.data, U.indptr)
    for i, node in enumerate(supernodal_assignment.supernodes)
    )

    return U,solve_out



def dynamic_factorize(𝒢, supernodal_assignment,U,solve_U, nugget=0.0, N_threads=1, N=None):
    if N is None:
        N = len(supernodal_assignment.measurements)  # Number of original poin

    I, J = _create_U_indices(supernodal_assignment.supernodes)

    # 2. Initializing new U
    U = csc_matrix((np.zeros(len(I)), (I, J)), shape=(N, N))

    max_n_rows = max(node.size(0) for node in supernodal_assignment.supernodes)
    max_n_cols = max(node.size(1) for node in supernodal_assignment.supernodes)

    # 4. Allocate buffers for L and U (thread-local)
    buffer_L = [np.zeros(max_n_rows * max_n_rows, dtype=float) for _ in range(N_threads)]
    buffer_U = [np.zeros(max_n_rows * max_n_cols, dtype=float) for _ in range(N_threads)]

    # Creating a new list for chol_factor and solve_U and using the flags to copy the data from the old one

    chol_fac_new=[]
    solve_U_new=[]
    j=0
    count=0
    
    #looping through the supernodes and using flags to update/create supernodes
    for node in supernodal_assignment.supernodes:
        if node.update_flag==True and node.fixed==False:
            count+=1
            n_rows_new=node.size(0)
            n_cols_new=node.size(1)

            # list of measurements for this node
            local_buffer_m= [supernodal_assignment.measurements[i] for i in node.row_indices_p()]
            local_buffer_L = buffer_L[0][:n_rows_new * n_rows_new].reshape(n_rows_new, n_rows_new)
            local_buffer_U = buffer_U[0][:n_rows_new * n_cols_new].reshape(n_rows_new, n_cols_new)
            local_buffer_U[:] = 0 # MUST clear before use, prevent error from accumulation

            for j, index_zero_based in enumerate(node.column_indices_p()):
                local_row_index = U.indptr[index_zero_based + 1]-U.indptr[index_zero_based]
                if local_row_index >= 1 and local_row_index <= local_buffer_U.shape[0]: 
                    local_buffer_U[local_row_index-1, j] = 1


            meas_dict=convert_measurements_to_list_of_dicts(local_buffer_m[:n_rows_new])
            ls=𝒢.length_scale
            
            cov3=g(ls)

            temp = cov3.build_symmetric(meas_dict)
            local_buffer_L = np.asarray(temp)
            np.fill_diagonal(local_buffer_L, np.diag(local_buffer_L) + nugget)  

            temp = cholesky(local_buffer_L, lower=False)

            chol_fac_new.append(temp)
            solve_triangular_output=solve_triangular(temp, local_buffer_U, lower=False, overwrite_b=False)
            solve_U_new.append(solve_triangular_output)

            for k, index in enumerate(node.column_indices_p()):
                range_of_nnzs = range(U.indptr[index], U.indptr[index+1])
                U.data[range_of_nnzs]=solve_triangular_output[:len(range_of_nnzs), k]

        elif node.update_flag==False and node.fixed==True:
            solve_U_new.append(solve_U[count])

            for k, index in enumerate(node.column_indices_p()):
                range_of_nnzs = range(U.indptr[index], U.indptr[index+1])
                U.data[range_of_nnzs]=solve_U[count][:len(range_of_nnzs), k]
            count+=1

        elif node.update_flag==False and node.fixed==False:
            solve_U_new.append(solve_U[count])

            for k, index in enumerate(node.column_indices_p()):
                range_of_nnzs = range(U.indptr[index], U.indptr[index+1])
                U.data[range_of_nnzs]=solve_U[count][:len(range_of_nnzs), k]
            count+=1
        else:
            print("unsupported dynamic super node update")
        j+=1
    return U,solve_U_new