
import numpy as np
from scipy.spatial import KDTree,cKDTree
from He import MutableHeap
from Super_nodes import IndexSuperNode,IndirectSupernodalAssignment
from meas import AbstractMeasurement


def _update_distances(nearest_distances, id_, new_distance):
    if isinstance(id_, (int, np.integer)):
        id_ = [id_]  # Convert single integer to a list
    elif isinstance(id_, (float, np.floating)):
        id_ = [int(id_)] # Convert single float to int, and then a list.


    if nearest_distances.ndim == 1:  # Ordinary distance case
        for i in id_: # Iterate!
            nearest_distances[int(i)] = min(nearest_distances[int(i)], new_distance) # 0-based

        return new_distance # This is fine.
    else:  # k-nearest neighbors case
        k_neighbors = nearest_distances.shape[0]
        for i in id_: # Iterate!
            i = int(i)  # Ensure integer indexing.
            if new_distance >= nearest_distances[0, i]: # 0-based
                continue # use continue instead of return
            else:
                for k in range(1, k_neighbors):
                    if nearest_distances[k, i] <= new_distance: #0-based
                        nearest_distances[k-1, i] = new_distance #0-based
                        break  # Exit inner loop after update
                    else:
                        nearest_distances[k-1, i] = nearest_distances[k, i] #0-based
                else: # If loop completes without break
                    nearest_distances[k_neighbors-1, i] = new_distance #0-based
        return nearest_distances[0, int(id_[0])] #0-based


def _construct_initial_distance(tree, x, k_neighbors=None):
    if k_neighbors is None:
        dists, _ = tree.query(x.T, k=1)  # scipy returns distances and indices
        return dists.flatten() # query returns a (n_samples, k) array.
    else:
        if tree.n >= k_neighbors:
            dists, indices = tree.query(x.T, k=k_neighbors) #query returns a (n_samples, k) array.
            if indices.ndim == 2: # Expected 2D case
                indices = indices.copy() # Avoid modifying original array in-place if needed later
                indices.sort(axis=-1)
                indices = indices[:, ::-1]
                dists_sorted = np.sort(dists, axis=1)[:, ::-1]
            elif indices.ndim == 1: # 1D case - handle differently
                indices_sorted = np.sort(indices) # Sort the 1D array
                indices_sorted = indices_sorted[::-1] # Reverse 1D array
                dists_sorted = np.sort(dists) # Sort 1D dists
                dists_sorted = dists_sorted[::-1] # Reverse 1D dists
            else:
                raise ValueError(f"Unexpected dimensionality of indices array: {indices.ndim}. Expected 1D or 2D.")

            return dists_sorted

        else:
            k_max = tree.n
            out_distances = np.sort(tree.query(x.T, k=k_max)[0], axis=1)[:, ::-1] #query returns a (n_samples, k) array.
            padding = np.full((x.shape[1], k_neighbors - k_max), np.inf)
            return np.hstack((padding, out_distances))


def concatenate_ordering(P_vec):
    offset = 0
    P_out = []
    for P in P_vec:
        P_out.extend(P + offset) # fixed
        offset += len(P)
    return np.array(P_out)

# MMD ordering for mulitple set.
def maximin_ordering(x, k_neighbors=None, init_distances=None, Tree=KDTree):

    if isinstance(x, list):  # Handle list of matrices

        if init_distances is None:
                init_distances = [np.full((xi.shape[1]), np.inf) if k_neighbors is None else np.full((k_neighbors, xi.shape[1]), np.inf) for xi in x]

        for k, xk in enumerate(x):

            tree = Tree(xk.T)  # Transpose for scipy

            for l in range(k + 1, len(x)):
                
                if k_neighbors is None:

                    init_distances[l] = np.minimum(init_distances[l], _construct_initial_distance(tree, x[l])) # No k
                else:
                    init_distances[l] = np.minimum(init_distances[l], _construct_initial_distance(tree, x[l], k_neighbors).T)  # Transpose
        P = []
        ℓ = []
        for k, xk in enumerate(x):
            Pk, ℓk = maximin_ordering_single(xk, k_neighbors=k_neighbors, init_distances=init_distances[k], Tree=Tree)

            P.append(Pk)
            ℓ.append(ℓk)

        return concatenate_ordering(P), np.concatenate(ℓ)

    else:  # Single matrix case (already handled by previous functions)

        return maximin_ordering_single(x, k_neighbors, init_distances, Tree)
    
# orders the points using MMD ordering (majorly used)
def maximin_ordering_single(x, k_neighbors=None, init_distances=None, Tree=KDTree):
    if k_neighbors is None: # default case: 1-maximin
        # Constructing the tree
        N = x.shape[1]
        tree = Tree(x.T) # scipy expects data points as rows
        if init_distances is None:
            init_distances = np.full(N, np.inf)
        nearest_distances = init_distances.copy()
        assert len(nearest_distances) == N
        heap = MutableHeap(nearest_distances)
        ℓ = np.empty(N, dtype=init_distances.dtype)
        P = np.empty(N, dtype=int)

        for k in range(N):
            pivot = heap.pop_node() 
            ℓ[k] = pivot.getval()
            P[k] = pivot.getid() 

            # Use query_ball_point for range search
            number_in_range = len(tree.query_ball_point(x[:, P[k]], ℓ[k])) # Use P[k]
            # Use query for knn search
            dists,ids = tree.query(x[:, P[k]], k=number_in_range) #Use P[k]

            # Make ids and dists 1D arrays.
            ids, dists = np.atleast_1d(ids), np.atleast_1d(dists)

            for id_, dist in zip(ids, dists):
                if id_ != pivot.getid():  # Crucial check: don't update the pivot itself
                    new_dist = _update_distances(nearest_distances, id_, dist)  # Pass id_

                # if id_ in heap.loc: # check if still in heap
                    heap.update(id_, new_dist)  #Pass id_

        return P, ℓ

    else: # k-maximin
        # constructing the tree
        N = x.shape[1]
        tree = Tree(x.T)  # scipy expects data points as rows
        if init_distances is None:
            init_distances = np.full((k_neighbors, N), np.inf)

        nearest_distances = init_distances.copy()
        assert nearest_distances.shape == (k_neighbors, N)

        for k in range(N):
            nearest_distances[:,k] = np.sort(nearest_distances[:,k])[::-1]

        heap = MutableHeap(nearest_distances[0,:])
        ℓ = np.empty(N, dtype=init_distances.dtype)
        P = np.empty(N, dtype=int)
        for k in range(N):
            pivot = heap.pop_node() # pop instead of top
            ℓ[k] = pivot.getval()
            P[k] = pivot.getid() # 0-based indexing

            number_in_range = len(tree.query_ball_point(x[:, P[k]], ℓ[k]))# Use P[k]
            # ids, dists = tree.query(x[:, P[k]], k=number_in_range) # Use P[k]
            dists,ids=tree.query(x[:, P[k]], k=number_in_range)
            # Make ids and dists 1D arrays.
            ids, dists = np.atleast_1d(ids), np.atleast_1d(dists)


            
            for id_, dist in zip(ids, dists):
                if id_ != pivot.getid():
                    new_dist = _update_distances(nearest_distances, id_, dist) # Pass id_
                    heap.update(id_, new_dist) # Pass id_

        return P, ℓ

# Can be used to find the closest point in the existing set with respect to the new point while dynamically updating. 
# Not used in either of the approaches.
def closest_point_to_new_points(old,new):
    tree = cKDTree(old.T)
    n=np.shape(new)[1]
    dis=np.zeros(n)
    ind=np.zeros(n)
    # Query the tree to find the single closest point (k=1) for each new point
    distances, indices = tree.query(new.T, k=np.shape(old)[1])

    dis[:]=distances[0][:n]
    ind[:]=indices[0][:n].astype(np.int32)

    return dis,ind,distances,indices

def _split_into_supernodes(parent_list, ℓ, λ):
    out = []
    ℓ_max = np.inf
    node = []
    for id_ in parent_list:
        if ℓ[id_] > ℓ_max / λ:
            node.append(id_)
        elif not node:
            ℓ_max = ℓ[id_]
        else:
            ℓ_max = ℓ[id_]
            out.append(node)
            node = []
    if node:  # Add remaining node
        out.append(node)
    return out


def _gather_assignments(assignments, first_parent):
    perm = np.argsort(assignments)
    first_indices = np.unique(assignments[perm], return_index=True)[1]
    first_indices = np.append(first_indices, len(assignments))

    ranges = [slice(first_indices[k], first_indices[k + 1]) for k in range(len(first_indices) - 1)]

    return [perm[range_] + (first_parent-1) for range_ in ranges] 


def findnext(N, l_temp, alpha, rho, min_l, last_aggregation_point):

    threshold = alpha * rho * min_l
    for l in range(last_aggregation_point, N + 2):
        if l == N + 1:
            return l - 1 
        elif l - 1 < len(l_temp) and l - 1 >= 0 and l_temp[l - 1] < threshold:
            return l - 1 
    return -1 # Should not reach here in typical scenarios, but as a fallback.

def findnext_v2(N, l, min_l, first_parent):

    for current_l in range(first_parent, N + 1): # range in Python is exclusive of the end, so N+1 to include N, starting from first_parent
        if current_l == N: 
            return current_l
        elif current_l < len(l) and current_l >= 0 and l[current_l] < min_l: # Condition 2: l is within l bounds and l[l + 1] < min_ℓ, becomes l[l] < min_l in 0-indexed Python
            return current_l 
    return -1 # Should not reach here, but as a fallback.


# Create supernodes based on ordering and length values from ordering. 
def supernodal_reverse_maximin_sparsity_pattern(x, P, ℓ, ρ, lambda_=1.5, alpha=1.0, Tree=KDTree, reconstruct_ordering=True):
    λ = lambda_
    α = alpha
    assert λ > 1.0
    assert 0.0 <= α <= 1.0
    assert α * ρ > 1

    N = x.shape[1]
    assert N == len(P)
    x = x[:, P] 
    if reconstruct_ordering:
        P_temp, ℓ_temp = maximin_ordering(x, Tree=Tree)
        rev_P_temp = np.empty_like(P_temp)
        rev_P_temp[P_temp] = np.arange(N)
        P_temp = P_temp
        ℓ_temp = ℓ_temp

    else:
        P_temp = P.copy()
        ℓ_temp = ℓ.copy()
        rev_P_temp = np.empty_like(P_temp)
        rev_P_temp[P_temp] = np.arange(N)

    supernodes = []
    children_tree = Tree(x.T)
    min_ℓ = np.max(ℓ[~np.isinf(ℓ)])
    last_aggregation_point = 1
    last_parent = 0


    while last_parent < N:
        last_aggregation_point=findnext(N, ℓ_temp, alpha, ρ, min_l, last_aggregation_point)

        if last_aggregation_point > 0:
            aggregation_tree = Tree(x[:, P_temp[:last_aggregation_point]].T)
        else:
              aggregation_tree = Tree(x[:, P_temp[:last_aggregation_point+1]].T)

        first_parent = last_parent+1

        last_parent=findnext_v2(N, ℓ, min_ℓ, first_parent)
        if last_parent == 0:
            last_parent = N


        assignments = aggregation_tree.query(x[:, first_parent-1: last_parent].T)[1] 

        column_indices_list = _gather_assignments(assignments, first_parent)




        for column_indices in column_indices_list:
            row_indices = []
            for column_index in column_indices:
                new_row_indices = children_tree.query_ball_point(x[:, column_index], ρ * ℓ[column_index]) 
                new_row_indices = [idx for idx in new_row_indices if idx  <= column_index] 

                row_dists = children_tree.query(x[:, new_row_indices].T)[0].flatten() 
                new_row_indices_filtered = [idx for idx, dist in zip(new_row_indices, row_dists) if dist <= ρ*ℓ[idx]]
                row_indices.extend(new_row_indices_filtered)

            row_indices = sorted(list(set(row_indices))) 
            supernodes.append(IndexSuperNode(list(column_indices), row_indices))  

        min_ℓ = min_ℓ / λ



    return supernodes

# Called to create supernodes after ordering. 
def ordering_and_sparsity_pattern(measurements, k_neighbors=None, Tree=KDTree, lambda_=1.5, alpha=1.0, rho = 3.0): # added rho

    if isinstance(measurements, list) and all(isinstance(sub, list) for sub in measurements):

        # List of lists: each sub-list is a group of measurements
        x_grouped = [np.column_stack([m.get_coordinate() for m in group]) for group in measurements]

        num_measurements = [len(group) for group in measurements] # get the number of measurements
        init_distances = [np.full(num, np.inf) if k_neighbors is None else np.full((k_neighbors, num), np.inf) for num in num_measurements]
        P, ℓ = maximin_ordering(x_grouped, k_neighbors, init_distances = init_distances,Tree=Tree) # pass init_distances

        measurements_reordered = [m for sublist in measurements for m in sublist]
        measurements_reordered = [measurements_reordered[i] for i in P]
        supernodes = supernodal_reverse_maximin_sparsity_pattern(np.hstack(x_grouped), P, ℓ, ρ=rho, lambda_ = lambda_, alpha = alpha, Tree = Tree, reconstruct_ordering = False)
 
        supernodes_out = IndirectSupernodalAssignment(supernodes, measurements_reordered)



    elif isinstance(measurements, list) and all(isinstance(m, AbstractMeasurement) for m in measurements):
        # Single list of measurements
        x_grouped = [np.vstack([m.get_coordinate() for m in measurements]).T]
        # x_grouped =[np.column_stack([m.get_coordinate() for m in group]) for group in measurements]

        num_measurements = [len(measurements)]

        init_distances = [np.full(num_measurements[0], np.inf) if k_neighbors is None else np.full((k_neighbors, num_measurements[0]), np.inf)]

        P, ℓ = maximin_ordering(x_grouped, k_neighbors, init_distances = None, Tree=Tree) # pass init_distances

        measurements_reordered = [measurements[i] for i in P]
        supernodes = supernodal_reverse_maximin_sparsity_pattern(np.hstack(x_grouped), P, ℓ, ρ=rho, lambda_ = lambda_, alpha = alpha, Tree = Tree, reconstruct_ordering = False)
        supernodes_out = IndirectSupernodalAssignment(supernodes, measurements_reordered)


    else:
        raise TypeError("measurements must be a list of AbstractMeasurement objects, or a list of lists of AbstractMeasurement objects")

    return P, ℓ, supernodes # return supernodes_out

# This function was written to order point and derivative measurement for solving PDE. Used while verifying the code for PDE problem.
def ordering_and_sparsity_pattern_diracs_first_then_unif_scale(measurements, 𝒢, k_neighbors=None, lambda_=1.5, alpha=1.0, Tree=KDTree,rho = 1.0):
    # Construct x, correctly handling lists of lists
    if isinstance(measurements, list) and all(isinstance(sub, list) for sub in measurements):
        # List of lists: each sub-list is a group of measurements
        x = [np.column_stack([m.get_coordinate() for m in group]) for group in measurements]

    elif isinstance(measurements, list) and all(isinstance(m, AbstractMeasurement) for m in measurements):
        # Single list of measurements
        x = [np.column_stack([m.get_coordinate() for m in measurements])]
    else:
        raise TypeError("measurements must be a list of AbstractMeasurement objects, or a list of lists of AbstractMeasurement objects")

    lm = len(x)
    assert lm >= 3

    if k_neighbors is None:
        P, ℓ = maximin_ordering(x[0:2],  Tree=Tree)
    else:
        P, ℓ = maximin_ordering(x[0:2], k_neighbors, Tree=Tree)


    N_domain = x[1].shape[1]
    N_boundary = x[0].shape[1]


    P_all = np.zeros((lm - 1) * N_domain + N_boundary, dtype=int)
    P_all[:N_boundary] = P[:N_boundary]
    P_all[N_boundary:] = np.concatenate([P[N_boundary:N_boundary+N_domain] + i * N_domain for i in range(lm - 1)])
    ℓ_all = np.zeros((lm - 1) * N_domain + N_boundary)
    ℓ_all[:N_boundary + N_domain] = ℓ
    ℓ_all[N_boundary + N_domain:] = ℓ[-1]

    supernodes = supernodal_reverse_maximin_sparsity_pattern(np.hstack(x), P_all, ℓ_all, ρ=rho, lambda_ = lambda_, alpha = alpha, Tree = Tree, reconstruct_ordering = False) # what is ρ here
    measurements_reordered = [m for sublist in measurements for m in sublist]
    measurements_reordered = [measurements_reordered[i] for i in P_all]

    supernodes_out = IndirectSupernodalAssignment(supernodes, measurements_reordered)
    return P_all, ℓ_all, supernodes_out

