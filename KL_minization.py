import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.sparse import csc_matrix
import threading

from meas import (
    AbstractMeasurement,
    PointMeasurement,
    dPointMeasurement,
    ddPointMeasurement,
    dddPointMeasurement,
    ddddPointMeasurement,
)
from typing import List
from Cov import GaussianCovariance_generic



# Code used for calculationg Cholesky factors and U. 
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

# Function that is called to build the U matrix for each supernode. 
def _factorize_node(𝒢, node, supernodal_assignment, U, buffer_L, buffer_U, buffer_m, nugget, U_lock,chol_factors,solve_out):
    thread_id = threading.get_ident() % len(buffer_L)  # Safer indexing
    n_rows = node.size(0)
    n_columns = node.size(1)


    local_buffer_L = buffer_L[thread_id][:n_rows * n_rows].reshape(n_rows, n_rows)
    local_buffer_U = buffer_U[thread_id][:n_rows * n_columns].reshape(n_rows, n_columns)
    local_buffer_U[:] = 0 # MUST clear before use, prevent error from accumulation

    local_buffer_m = [supernodal_assignment.measurements[i] for i in node.row_indices_p()]


    for j, index_zero_based in enumerate(node.column_indices_p()):

        local_row_index = U.indptr[index_zero_based + 1]-U.indptr[index_zero_based]
        if local_row_index >= 1 and local_row_index <= local_buffer_U.shape[0]: 
            local_buffer_U[local_row_index-1, j] = 1
        
    for i in range(n_rows):
        for j in range(i, n_rows):
            local_buffer_L[i, j] = 𝒢(local_buffer_m[i], local_buffer_m[j])
            if i != j:
                local_buffer_L[j, i] = local_buffer_L[i, j]
    

    try:
        np.fill_diagonal(local_buffer_L, np.diag(local_buffer_L) + nugget)

        chol = cholesky(local_buffer_L, lower=False)

        solve_triangular_output=solve_triangular(chol, local_buffer_U, lower=False, overwrite_b=False) # return to variable
        chol_factors.append(chol)
        solve_out.append(solve_triangular_output)


    except:

        print("Singular matrix at factorization - Try increasing nugget values")
        
        solve_triangular_output=np.zeros((n_rows,n_columns))


    with U_lock:
        for k, index in enumerate(node.column_indices_p()):

            range_of_nnzs = range(U.indptr[index], U.indptr[index+1])

            U.data[range_of_nnzs]=solve_triangular_output[:len(range_of_nnzs), k]


# Receives the supernode information and factorizes each supernode. It returns the final upper triangular precision U matrix.
def factorize(𝒢, supernodal_assignment, nugget=0.0, N_threads=1, N=None):

    if N is None:
        N = len(supernodal_assignment.measurements)  # Number of original points
    

    # 1. Create the indices for the upper-triangular part of U
    I, J = _create_U_indices(supernodal_assignment.supernodes)

    # 2. Initialize U as a sparse matrix (CSC format)
    U = csc_matrix((np.zeros(len(I)), (I, J)), shape=(N, N))
    chol_factors=[]
    solve_out=[]

    
    # 3. Determine maximum buffer sizes needed
    max_n_rows = max(node.size(0) for node in supernodal_assignment.supernodes)
    max_n_cols = max(node.size(1) for node in supernodal_assignment.supernodes)

    # 4. Allocate buffers for L and U (thread-local)
    buffer_L = [np.zeros(max_n_rows * max_n_rows, dtype=float) for _ in range(N_threads)]
    buffer_U = [np.zeros(max_n_rows * max_n_cols, dtype=float) for _ in range(N_threads)]
    buffer_m = [[] for _ in range(N_threads)]

    # 5. Process each supernode (in parallel, if N_threads > 1)
    U_lock = threading.Lock()  # Lock for updating U

    if N_threads > 1:
        threads = []
        for node in supernodal_assignment.supernodes:
            thread = threading.Thread(target=_factorize_node, args=(𝒢, node, supernodal_assignment, U, buffer_L, buffer_U, buffer_m, nugget, U_lock))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()
    else:
        for node in supernodal_assignment.supernodes:
            _factorize_node(𝒢, node, supernodal_assignment, U, buffer_L, buffer_U, buffer_m, nugget, U_lock,chol_factors,solve_out)
    return U,chol_factors,solve_out
#     return U



# This function is called when the factorization occurs during dynamic update. The update is determined based on the update_flag and
# fixed_flag values associated with the supernode. 
# def dynamic_factorize(𝒢, supernodal_assignment,U, chol_fac,solve_U, nugget=0.0, N_threads=1, N=None):
    
#     if N is None:
#         N = len(supernodal_assignment.measurements)  # Number of original poin

#     I, J = _create_U_indices(supernodal_assignment.supernodes)

#     # Initializing new U
#     U = csc_matrix((np.zeros(len(I)), (I, J)), shape=(N, N))

#     max_n_rows = max(node.size(0) for node in supernodal_assignment.supernodes)
#     max_n_cols = max(node.size(1) for node in supernodal_assignment.supernodes)

#     # Allocate buffers for L and U (thread-local)
#     buffer_L = [np.zeros(max_n_rows * max_n_rows, dtype=float) for _ in range(N_threads)]
#     buffer_U = [np.zeros(max_n_rows * max_n_cols, dtype=float) for _ in range(N_threads)]
#     buffer_m = [[] for _ in range(N_threads)]

#     # Creating a new list for chol_factor and solve_U and using the flags to copy the data from the old one
#     chol_fac_new=[]
#     solve_U_new=[]
#     count=0
    
#     # Looping through the supernodes and using flags to update/create supernodes
#     for node in supernodal_assignment.supernodes:

#         # Updating dynamic supernode
#         if node.update_flag==True and node.fixed==False: 
       
#             count+=1
#             n_rows_new=node.size(0)
#             n_cols_new=node.size(1)

#             # list of measurements for this node
#             local_buffer_m= [supernodal_assignment.measurements[i] for i in node.row_indices_p()]
#             local_buffer_L = buffer_L[0][:n_rows_new * n_rows_new].reshape(n_rows_new, n_rows_new)
#             local_buffer_U = buffer_U[0][:n_rows_new * n_cols_new].reshape(n_rows_new, n_cols_new)
#             local_buffer_U[:] = 0 # MUST clear before use, prevent error from accumulation

#             for j, index_zero_based in enumerate(node.column_indices_p()):
#                 local_row_index = U.indptr[index_zero_based + 1]-U.indptr[index_zero_based]
#                 if local_row_index >= 1 and local_row_index <= local_buffer_U.shape[0]: 
#                     local_buffer_U[local_row_index-1, j] = 1


#             # calculating full covariance matrix for now.
#             for i in range(n_rows_new):
#                 for j in range(i, n_rows_new):
#                     local_buffer_L[i, j] = 𝒢(local_buffer_m[i], local_buffer_m[j])
#                     if i != j:
#                         local_buffer_L[j, i] = local_buffer_L[i, j]
#             np.fill_diagonal(local_buffer_L, np.diag(local_buffer_L) + nugget)  

#             temp = cholesky(local_buffer_L, lower=False)

#             chol_fac_new.append(temp)
#             solve_triangular_output=solve_triangular(temp, local_buffer_U, lower=False, overwrite_b=False)
#             solve_U_new.append(solve_triangular_output)


#             for k, index in enumerate(node.column_indices_p()):
#                 range_of_nnzs = range(U.indptr[index], U.indptr[index+1])
#                 U.data[range_of_nnzs]=solve_triangular_output[:len(range_of_nnzs), k]

#         # Reusing Cholesky factors for fixed supernodes
#         elif node.update_flag==False and node.fixed==True:
#             chol_fac_new.append(chol_fac[count])
#             solve_U_new.append(solve_U[count])

#             for k, index in enumerate(node.column_indices_p()):
#                 range_of_nnzs = range(U.indptr[index], U.indptr[index+1])
#                 U.data[range_of_nnzs]=solve_U[count][:len(range_of_nnzs), k]
#             count+=1

#         # In approach -2 of dynamic update, only one of the supernode is updates and other are neither fixed nor updated.
#         # This loop is for those supernodes.
#         elif node.update_flag==False and node.fixed==False:
#             chol_fac_new.append(chol_fac[count])
#             solve_U_new.append(solve_U[count])

#             for k, index in enumerate(node.column_indices_p()):
#                 range_of_nnzs = range(U.indptr[index], U.indptr[index+1])
#                 U.data[range_of_nnzs]=solve_U[count][:len(range_of_nnzs), k]
#             count+=1

#         else:
#             print("unsupported dynamic super node update")

#     return U,chol_fac_new,solve_U_new


def convert_measurements_to_list_of_dicts(measurements):
    """
    Converts a list of AbstractMeasurement objects into a list of dictionaries.

    Args:
        measurements (List[AbstractMeasurement]): A list of measurement objects.

    Returns:
        List[dict]: A list of dictionaries, where each dictionary
                    represents a measurement.
    """
    meas_list = []
    for m in measurements:
        if isinstance(m, PointMeasurement):
            entry = {"cord": m.get_coordinate(), "index": None, "len": 0}
        elif isinstance(m, dPointMeasurement):
            entry = {"cord": m.get_coordinate(), "index": m.derivative_index, "len": 1}
        else:
            entry = {
                "cord": m.get_coordinate(),
                "index": m.derivative_indices,
                "len": len(m.derivative_indices),
            }
        meas_list.append(entry)
    return meas_list

def build_symmetric(cov, M):
    N = len(M)
    output_matrix = np.zeros((N, N), dtype=np.float64)

    for i in range(N):
        for j in range(N):
            output_matrix[i, j] = cov(M[i], M[j])

    return output_matrix


def build_test(cov, te, tr):
    N = len(tr)
    M = len(te)
    output_matrix = np.zeros((M, N), dtype=np.float64)

    for i in range(M):
        for j in range(N):
            output_matrix[i, j] = cov(te[i], tr[j])

    return output_matrix


def dynamic_factorize(𝒢, supernodal_assignment,U,solve_U, nugget, N_threads=1, N=None):
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


#             meas_dict=convert_measurements_to_list_of_dicts(local_buffer_m[:n_rows_new])
            meas_dict=local_buffer_m[:n_rows_new]
            ls=𝒢.length_scale
            
            cov3=GaussianCovariance_generic(ls)

            temp = build_symmetric(cov3, meas_dict)
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