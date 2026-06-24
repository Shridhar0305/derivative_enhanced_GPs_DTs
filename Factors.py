
import numpy as np
from scipy.spatial import KDTree
from abc import ABC

from Max_NN import ordering_and_sparsity_pattern,ordering_and_sparsity_pattern_diracs_first_then_unif_scale
from Max_NN import closest_point_to_new_points
from Super_nodes import IndirectSupernodalAssignment,IndexSuperNode
try:
    from KL_minization_cython_call import factorize, dynamic_factorize # import this for cython version
except:
    print("Error importing cython version of factorization. Importing python version")
    from KL_minization import factorize, dynamic_factorize  
import copy


# Abstract base class for KL Factorization
class AbstractKLFactorization(ABC):  # Inherits from ABC
    pass

class ExplicitKLFactorization(AbstractKLFactorization):
    def __init__(self, P, measurements, 𝒢, U,SO):
        self.P = P
        self.measurements = measurements
        self.𝒢 = 𝒢
        self.U = U
        self.SO= SO


    @classmethod
    def Explicit_from_implicit(cls, implicit_kl_factorization, nugget=0.0,N_threads=1):
        N = len(implicit_kl_factorization.supernodes.measurements) # get the correct N
        U, SO, _ = factorize(implicit_kl_factorization.𝒢, implicit_kl_factorization.supernodes, nugget=nugget, N=N,N_threads=N_threads)
        return cls(implicit_kl_factorization.P, implicit_kl_factorization.supernodes.measurements, implicit_kl_factorization.𝒢, U, SO)

    @classmethod
    def dynamic_Explicit_from_implicit(cls, implicit_kl_factorization,U_old,SO_old,nugget=0.0,N_threads=1):
        N = len(implicit_kl_factorization.supernodes.measurements) # get the correct N
        
        U_new,SO_new = dynamic_factorize(implicit_kl_factorization.𝒢, implicit_kl_factorization.supernodes,U_old,SO_old, nugget=nugget, N=N,N_threads=N_threads)
        return cls(implicit_kl_factorization.P, implicit_kl_factorization.supernodes.measurements, implicit_kl_factorization.𝒢, U_new,SO_new)
    


class ImplicitKLFactorization(AbstractKLFactorization):
    def __init__(self, P, supernodes, 𝒢):
        self.P = P
        self.supernodes = supernodes
        self.𝒢 = 𝒢


    # Function used to factorization of point measuremnets without the need for dynamic update of the sparse GP
    def implicit_kl_factorization_list_k_maximin(𝒢, measurements_list, ρ, k_neighbors, lambda_=1.5, alpha=1.0, Tree=KDTree):
        P, ℓ, supernodes = ordering_and_sparsity_pattern(measurements_list, k_neighbors, lambda_=lambda_, alpha=alpha, Tree=Tree,rho=ρ)
        measurements = [m for sublist in measurements_list for m in sublist]
        measurements = [measurements[i] for i in P]
        supernodes = IndirectSupernodalAssignment(supernodes, measurements)
        return ImplicitKLFactorization(P, supernodes, 𝒢)    


    # Function used to factorization of point measuremnets with scope for future dynamic update. This function is called only during initial training.
    # This is only for appraoch - 1
    def implicit_kl_factorization_list_k_maximin_partial_approach_1(𝒢, measurements_list, ρ, k_neighbors, lambda_=1.5, alpha=1.0, Tree=KDTree):
        P, ℓ, supernodes = ordering_and_sparsity_pattern(measurements_list, k_neighbors, lambda_=lambda_, alpha=alpha, Tree=Tree,rho=ρ)

        P_wo_der=copy.deepcopy(P)
        fp=int(0.8*np.shape(P)[0]) # getting size for dynamic set
        m_set=list(P[fp:]) #indices of dynamic set

        # finding the supernodes with elemntes from m_set to flag it dynamic
        # Note: In this function, we only flag it as it is called during the initial training
        temp_row=[]
        temp_col=[]
        for node in supernodes:
            if all(item in m_set for item in P[node.column_indices]):
                node.fixed=False
                if not temp_row and not temp_col:
                    temp_col=node.column_indices
                    temp_row=node.row_indices
                else:
                    temp_row=list(set(temp_row+node.row_indices))
                    temp_col=list(set(temp_col+node.column_indices))
        temp_col.sort()
        temp_row.sort()
        supernodes_1=[node for node in supernodes if node.fixed]
        supernodes_1.append(IndexSuperNode(temp_col, temp_row,fixed=False))
        
        # This is an extra step to prevent issues during dynamic update. Sometimes, the dynamic supernode will not have all the elements in the range (N-M to N).
        # Find the missing elements in the dynamic set. 
        last_node = supernodes_1[-1]
        last_node_cols = last_node.column_indices
        missing_elements = set()
        if len(last_node_cols) > 1:
            min_val, max_val = min(last_node_cols), max(last_node_cols)
            full_range = set(range(min_val, max_val + 1))
            missing_elements = full_range - set(last_node_cols)

       # Find the node which has missing elements
        affected_nodes_p1 = {}
        if missing_elements:
            for i, node in enumerate(supernodes_1[:-1]):
                original_rows, original_cols = list(node.row_indices), list(node.column_indices)
                node.row_indices = [idx for idx in node.row_indices if idx not in missing_elements]
                node.column_indices = [idx for idx in node.column_indices if idx not in missing_elements]
                if node.row_indices != original_rows or node.column_indices != original_cols:
                    affected_nodes_p1[f"Node {i}"] = True

        # Removing the elements from other supernodes
        for i, node in enumerate(supernodes_1):
            node_key = f"Node {i}"
            if node_key in affected_nodes_p1:
                while node.row_indices and node.column_indices and max(node.row_indices) > max(node.column_indices):
                    max_row_val = max(node.row_indices)
                    node.row_indices.remove(max_row_val)

        #Adding the missed elements to the dynamic supernode
        if missing_elements:
            # Add the elements to both lists
            last_node.row_indices.extend(list(missing_elements))
            last_node.column_indices.extend(list(missing_elements))

            # Sort the lists to maintain order
            last_node.row_indices.sort()
            last_node.row_indices=list(set(last_node.row_indices))
            last_node.column_indices.sort()

        supernodes=copy.deepcopy(supernodes_1)

        supernodes_wo_der=copy.deepcopy(supernodes)

        measurements = [m for sublist in measurements_list for m in sublist]
        measurements = [measurements[i] for i in P]
        supernodes = IndirectSupernodalAssignment(supernodes, measurements)
        
        # Returning the implicit factors with ordering (P) and supernodes. I am also sending supernodes_wo_der and P_wo_der.
        # For none derivative case, P=P_wo_der, supernodes=supernodes_wo_der. There is redundancy in return but this is kept to keep consistency on return with derivative cases.
        return ImplicitKLFactorization(P, supernodes, 𝒢),supernodes_wo_der,P_wo_der
    
    # When there are no derivatives, this function is called for dynamic update using approach-1
    def dynamic_implicit_kl_factorization_list_k_maximin_partial_approach_1(𝒢, old_set,new_set,supernodes,P_old,r_set,measurements_list,loc_np, ρ, k_neighbors, lambda_=1.5, alpha=1.0, Tree=KDTree):
                
        # obtaining ordering on new_set
        P_new, ℓ_new, supernodes_new = ordering_and_sparsity_pattern(new_set,k_neighbors= k_neighbors, lambda_=lambda_, alpha=alpha, Tree=Tree,rho=ρ)

        new_order=r_set[P_new]

        remaining_elements = [item for item in P_old if item not in set(new_order)]

        P_update_1=remaining_elements+list(new_order) # updated order

        # Flagging supernodes for update (reuse or re-evaluation of Cholesky factors)
        for node in supernodes:
            if node.fixed==False:
                node.column_indices=np.array(node.column_indices+loc_np)
                node.column_indices=list(node.column_indices[P_new])
                node.row_indices=node.row_indices+loc_np
                node.update_flag=True
                node.column_indices.sort()
                node.row_indices.sort()

        supernodes_wo_der=copy.deepcopy(supernodes)
        P_wo_der=copy.deepcopy(P_update_1)
        measurements = [m for sublist in measurements_list for m in sublist]
        measurements = [measurements[i] for i in P_update_1]
        supernodes = IndirectSupernodalAssignment(supernodes, measurements)

        # Redudancy in return to maintain consistenty in outputs across derivative and non-derivative cases
        return ImplicitKLFactorization(P_update_1, supernodes, 𝒢),supernodes_wo_der,P_wo_der


    # Function used for factorization of point measuremnets with scope for future dynamic update. This function is called only during initial training.
    # This is only for appraoch - 2
    def implicit_kl_factorization_list_k_maximin_partial_approach_2(𝒢, measurements_list, ρ, k_neighbors, lambda_=1.5, alpha=1.0, Tree=KDTree):
        P, ℓ, supernodes = ordering_and_sparsity_pattern(measurements_list, k_neighbors, lambda_=lambda_, alpha=alpha, Tree=Tree,rho=ρ)


        P_wo_der=copy.deepcopy(P)
        L_wo_der=copy.deepcopy(ℓ)
        N=len(measurements_list[0])
        fp=int(0.8*np.shape(P)[0])
        m_set=list(P[fp:])
        L_wo_der=ℓ
        temp_row=[]
        temp_col=[]
        for node in supernodes:
            if all(item in m_set for item in P[node.column_indices]):
                node.fixed=False
        supernodes_wo_der=copy.deepcopy(supernodes)

        measurements = [m for sublist in measurements_list for m in sublist]
        measurements = [measurements[i] for i in P]
        supernodes = IndirectSupernodalAssignment(supernodes, measurements)

        return ImplicitKLFactorization(P, supernodes, 𝒢),supernodes_wo_der,P_wo_der,L_wo_der
    
    def dynamic_implicit_kl_factorization_list_k_maximin_partial_approach_2(𝒢, old_set,new_set,supernodes,P_old,L_old,r_set,measurements_list,loc_np, ρ, k_neighbors, lambda_=1.5, alpha=1.0, Tree=KDTree):
        
        #New point is always at the end of the measurement. Using that for calculating distance with other points in dynamic set
        
        for i,n in enumerate(new_set):
            if i==0:
                new_pt=n.coordinate
            else:
                new_pt=np.vstack((new_pt,n.coordinate))

        first_part = new_pt[:-1]
        last_part = new_pt[-1:]
            
        all_distances = np.linalg.norm(first_part - last_part, axis=1)


        rl=ρ*(L_old[-np.shape(first_part)[0]:])


        indices = np.where(all_distances < rl)[0]


        if len(indices) > 0:
            # If there are such locations, find the one with the maximum d value
            max_d_index = r_set[indices[np.argmax(all_distances[indices])]]
            # print(f"The location with the maximum d value where d < s is at index: {max_d_index}")
            # print(f"The value of d at this location is: {all_distances[max_d_index]}")
        else:
            max_d_index=-1
        
        # Checking if the dynamic supernode false before fixed. Sometimes a particular column can lie before 
        # a supernode with a group of column indices after first element of M set
        # print("supernode before",supernodes)
        
        max_col_index=-1
        for node in supernodes:
            if node.fixed:
                if node.column_indices:
                    # Find the maximum index in the current node's column_indices
                    max_in_node = max(node.column_indices)
                    # Update the overall maximum if the current one is larger
                    if max_in_node > max_col_index:
                        max_col_index = max_in_node

        if max_d_index<=max_col_index:
            max_d_index=-1
        if max_d_index!=-1:
            SN_loc=np.where(P_old==max_d_index)[0][0]
            P_update_1=np.insert(P_old,SN_loc + 1, loc_np)
            coun=0
            for node in supernodes:
                if node.fixed==False and max_d_index!=-1:
                    node.update_flag=True
                    if SN_loc in node.column_indices:
                        coun+=1
                        node.column_indices=[x + coun if x > SN_loc else x for x in node.column_indices]
                        node.row_indices=[x + coun if x > SN_loc else x for x in node.row_indices]
                        node.column_indices.insert(-1,SN_loc+1)
                        node.row_indices.insert(-1,SN_loc+1)
                        node.column_indices.sort()
                        node.row_indices.sort()

            for node in supernodes:
                if node.fixed==False and max_d_index!=-1:        
                        node.update_flag=True
                        if coun != 0 and SN_loc not in node.column_indices:
                            node.column_indices=[x + coun if x > SN_loc else x for x in node.column_indices]
                            node.row_indices=[x + coun if x > SN_loc else x for x in node.row_indices]
                            node.column_indices.sort()
                            node.row_indices.sort()
        else:
            SN_loc=loc_np
            P_update_1=np.hstack((P_old,loc_np))
            temp_col=list(loc_np)
            temp_row=[]
            for node in supernodes:
                if node.fixed==False:
                    temp_row.append(node.row_indices)
            temp_row.append(loc_np)
            temp_row=[x for sublist in temp_row for x in sublist]
            temp_row=list(set(temp_row))

            temp_row.sort()
            supernodes.append(IndexSuperNode(temp_col, temp_row,fixed=False,update_flag=True))

        supernodes_wo_der=copy.deepcopy(supernodes)
        P_wo_der=copy.deepcopy(P_update_1)
        measurements = [m for sublist in measurements_list for m in sublist]
        measurements = [measurements[i] for i in P_update_1]
        supernodes = IndirectSupernodalAssignment(supernodes, measurements)
        return ImplicitKLFactorization(P_update_1, supernodes, 𝒢),supernodes_wo_der,P_wo_der,L_old


    # ordering derivative measurment in a way P,d,dd,ddd,dddd,P,d,dd,...
    @staticmethod
    def implicit_kl_factorization_for_d(𝒢, measurements_list, ρ, k_neighbors, lambda_=1.5, alpha=1.0, Tree=KDTree):
        lm = len(measurements_list)
#         assert lm >= 2 #making sure derivative measurements are in it
        #sending only the point measurement 
        P, ℓ, supernodes = ordering_and_sparsity_pattern(measurements_list[0],k_neighbors= k_neighbors, lambda_=lambda_, alpha=alpha, Tree=Tree,rho=ρ)

        supernodes_wo_der=copy.deepcopy(supernodes)
        P_wo_der=copy.deepcopy(P)
        N=len(measurements_list[0])

        # Need to get length of each measurement list and then calculate Nd for multi-dimensional problems
        Nd=0
        for i in range(lm):
            Nd+=len(measurements_list[i])

        P_all=np.zeros(Nd,dtype=int)
        d=int(Nd/N)

        for i in range(N):
            P_all[i*d:(i+1)*d]=np.array([P[i]+N*j for j in range(d)])


        measurements = [m for sublist in measurements_list for m in sublist]

        measurements = [measurements[i] for i in P_all]

        for node in supernodes:
            m=len(node.row_indices)
            n=len(node.column_indices)

            for i in range(m):
                rowi = node.row_indices[i]
                # all the nodes will be rearranged
                node.row_indices[i] = (rowi+1)*d-1
                node.row_indices.extend(range(rowi*d, (rowi+1)*d-1))

            node.row_indices.sort()


            for j in range(n):
                columni = node.column_indices[j]
                node.column_indices[j] = (columni+1)*d-1
                node.column_indices.extend(range(columni*d, (columni+1)*d-1))

            node.column_indices.sort()

        supernodes = IndirectSupernodalAssignment(supernodes,measurements)

        return ImplicitKLFactorization(P_all, supernodes, 𝒢),supernodes_wo_der, P_wo_der
    

    def implicit_kl_factorization_for_d_with_partial_split_approach_1(𝒢, measurements_list, ρ, k_neighbors, lambda_=1.5, alpha=1.0, Tree=KDTree):
        lm = len(measurements_list)
        assert lm >= 2 #making sure derivative measurements are in it
        #sending only the point measurement 
        P, ℓ, supernodes = ordering_and_sparsity_pattern(measurements_list[0],k_neighbors= k_neighbors, lambda_=lambda_, alpha=alpha, Tree=Tree,rho=ρ)

        
        P_wo_der=copy.deepcopy(P)
        N=len(measurements_list[0])
        fp=int(0.8*np.shape(P)[0])
        m_set=list(P[fp:])


        # Need to get length of each measurement list and then calculate Nd for multi-dimensional problems
        Nd=0
        for i in range(lm):
            Nd+=len(measurements_list[i])

        P_all=np.zeros(Nd,dtype=int)
        d=int(Nd/N)

        for i in range(N):
            P_all[i*d:(i+1)*d]=np.array([P[i]+N*j for j in range(d)])
        
        # print(supernodes)
        temp_row=[]
        temp_col=[]
        for node in supernodes:
            if all(item in m_set for item in P[node.column_indices]):
                node.fixed=False
                if not temp_row and not temp_col:
                    temp_col=node.column_indices
                    temp_row=node.row_indices
                else:
                    temp_row=list(set(temp_row+node.row_indices))
                    temp_col=list(set(temp_col+node.column_indices))
        temp_col.sort()
        temp_row.sort()
        supernodes_1=[node for node in supernodes if node.fixed]
        supernodes_1.append(IndexSuperNode(temp_col, temp_row,fixed=False))
        
        last_node = supernodes_1[-1]
        last_node_cols = last_node.column_indices
        missing_elements = set()
        if len(last_node_cols) > 1:
            min_val, max_val = min(last_node_cols), max(last_node_cols)
            full_range = set(range(min_val, max_val + 1))
            missing_elements = full_range - set(last_node_cols)


        affected_nodes_p1 = {}
        if missing_elements:
            for i, node in enumerate(supernodes_1[:-1]):
                original_rows, original_cols = list(node.row_indices), list(node.column_indices)
                node.row_indices = [idx for idx in node.row_indices if idx not in missing_elements]
                node.column_indices = [idx for idx in node.column_indices if idx not in missing_elements]
                if node.row_indices != original_rows or node.column_indices != original_cols:
                    affected_nodes_p1[f"Node {i}"] = True

       
        for i, node in enumerate(supernodes_1):
            node_key = f"Node {i}"
            if node_key in affected_nodes_p1:

                while node.row_indices and node.column_indices and max(node.row_indices) > max(node.column_indices):
                    max_row_val = max(node.row_indices)
                    node.row_indices.remove(max_row_val)
                    
        if missing_elements:
            # Add the elements to both lists
            last_node.row_indices.extend(list(missing_elements))
            last_node.column_indices.extend(list(missing_elements))

            # Sort the lists to maintain order
            last_node.row_indices.sort()
            last_node.row_indices=list(set(last_node.row_indices))
            last_node.column_indices.sort()

        supernodes=copy.deepcopy(supernodes_1)

        supernodes_wo_der=copy.deepcopy(supernodes)


        measurements = [m for sublist in measurements_list for m in sublist]

        measurements = [measurements[i] for i in P_all]

        for node in supernodes:
            m=len(node.row_indices)
            n=len(node.column_indices)

            for i in range(m):
                rowi = node.row_indices[i]
                # all the nodes will be rearranged
                node.row_indices[i] = (rowi+1)*d-1
                node.row_indices.extend(range(rowi*d, (rowi+1)*d-1))

            node.row_indices.sort()


            for j in range(n):
                columni = node.column_indices[j]
                node.column_indices[j] = (columni+1)*d-1
                node.column_indices.extend(range(columni*d, (columni+1)*d-1))

            node.column_indices.sort()

        supernodes = IndirectSupernodalAssignment(supernodes,measurements)

        return ImplicitKLFactorization(P_all, supernodes, 𝒢),supernodes_wo_der, P_wo_der
    

    

    def dynamic_implicit_kl_factorization_for_d_partial_split_approach_1(𝒢, old_set,new_set,supernodes,P_old,r_set,measurements_list,loc_np, ρ, k_neighbors, lambda_=1.5, alpha=1.0, Tree=KDTree):
        lm = len(measurements_list)
        assert lm >= 2 #making sure derivative measurements are in it
        #sending only the point measurement 

        P_new, ℓ_new, supernodes_new = ordering_and_sparsity_pattern(new_set,k_neighbors= k_neighbors, lambda_=lambda_, alpha=alpha, Tree=Tree,rho=ρ)
  
        new_order=r_set[P_new]

        remaining_elements = [item for item in P_old if item not in set(new_order)]

        P_update_1=remaining_elements+list(new_order)


        for node in supernodes:
            if node.fixed==False:
                node.column_indices=np.array(node.column_indices+loc_np)
                node.column_indices=list(node.column_indices[P_new])
                node.row_indices=node.row_indices+loc_np
                node.update_flag=True
                node.column_indices.sort()
                node.row_indices.sort()


        supernodes_wo_der=copy.deepcopy(supernodes)
        P_wo_der=copy.deepcopy(P_update_1)

        # Need to get length of each measurement list and then calculate Nd for multi-dimensional problems
        N=len(measurements_list[0])
        Nd=0
        for i in range(lm):
            Nd+=len(measurements_list[i])

        P_all=np.zeros(Nd,dtype=int)
        
        d=int(Nd/N)

        for i in range(N):
            P_all[i*d:(i+1)*d]=np.array([P_update_1[i]+N*j for j in range(d)])

        measurements = [m for sublist in measurements_list for m in sublist]

        measurements = [measurements[i] for i in P_all]

        for node in supernodes:
            m=len(node.row_indices)
            n=len(node.column_indices)

            for i in range(m):
                rowi = node.row_indices[i]
                # all the nodes will be rearranged
                node.row_indices[i] = (rowi+1)*d-1
                node.row_indices.extend(range(rowi*d, (rowi+1)*d-1))

            node.row_indices.sort()


            for j in range(n):
                columni = node.column_indices[j]
                node.column_indices[j] = (columni+1)*d-1
                node.column_indices.extend(range(columni*d, (columni+1)*d-1))

            node.column_indices.sort()

        supernodes = IndirectSupernodalAssignment(supernodes,measurements)

        return ImplicitKLFactorization(P_all, supernodes, 𝒢),supernodes_wo_der,P_wo_der

    def implicit_kl_factorization_for_d_with_partial_split_approach_2(𝒢, measurements_list, ρ, k_neighbors, lambda_=1.5, alpha=1.0, Tree=KDTree):
        lm = len(measurements_list)
        assert lm >= 2 #making sure derivative measurements are in it
        #sending only the point measurement 
        P, ℓ, supernodes = ordering_and_sparsity_pattern(measurements_list[0],k_neighbors= k_neighbors, lambda_=lambda_, alpha=alpha, Tree=Tree,rho=ρ)

        
        P_wo_der=copy.deepcopy(P)
        L_wo_der=copy.deepcopy(ℓ)
        N=len(measurements_list[0])
        fp=int(0.8*np.shape(P)[0])
        m_set=list(P[fp:])




        # Need to get length of each measurement list and then calculate Nd for multi-dimensional problems
        Nd=0
        for i in range(lm):
            Nd+=len(measurements_list[i])

        P_all=np.zeros(Nd,dtype=int)
        d=int(Nd/N)

        for i in range(N):
            P_all[i*d:(i+1)*d]=np.array([P[i]+N*j for j in range(d)])
        

        for node in supernodes:
            if all(item in m_set for item in P[node.column_indices]):
                node.fixed=False
   
        supernodes_wo_der=copy.deepcopy(supernodes)

        measurements = [m for sublist in measurements_list for m in sublist]

        measurements = [measurements[i] for i in P_all]

        for node in supernodes:
            m=len(node.row_indices)
            n=len(node.column_indices)

            for i in range(m):
                rowi = node.row_indices[i]
                # all the nodes will be rearranged
                node.row_indices[i] = (rowi+1)*d-1
                node.row_indices.extend(range(rowi*d, (rowi+1)*d-1))

            node.row_indices.sort()


            for j in range(n):
                columni = node.column_indices[j]
                node.column_indices[j] = (columni+1)*d-1
                node.column_indices.extend(range(columni*d, (columni+1)*d-1))

            node.column_indices.sort()

        supernodes = IndirectSupernodalAssignment(supernodes,measurements)

        return ImplicitKLFactorization(P_all, supernodes, 𝒢),supernodes_wo_der, P_wo_der,L_wo_der
    

    def dynamic_implicit_kl_factorization_for_d_partial_split_approach_2(𝒢, old_set,new_set,supernodes,P_old,L_old,r_set,measurements_list,loc_np, ρ, k_neighbors, lambda_=1.5, alpha=1.0, Tree=KDTree):
        lm = len(measurements_list)
        assert lm >= 2 #making sure derivative measurements are in it
        #sending only the point measurement 
        
        # new_set is the one that has old reordering points and 
        for i,n in enumerate(new_set):
            if i==0:
                new_pt=n.coordinate
            else:
                new_pt=np.vstack((new_pt,n.coordinate))

        first_part = new_pt[:-1]
        last_part = new_pt[-1:]
            
        all_distances = np.linalg.norm(first_part - last_part, axis=1)

        rl=ρ*(L_old[-np.shape(first_part)[0]:])

        indices = np.where(all_distances < rl)[0]


        if len(indices) > 0:
            max_d_index = r_set[indices[np.argmax(all_distances[indices])]]
        else:
            max_d_index=-1
        
        # Checking if the dynamic supernode false before fixed. Sometimes a particular column can lie before 
        # a supernode with a group of column indices after first element of M set
        # print("supernode before",supernodes)
        
        max_col_index=-1
        for node in supernodes:
            if node.fixed:
                if node.column_indices:
                    # Find the maximum index in the current node's column_indices
                    max_in_node = max(node.column_indices)
                    # Update the overall maximum if the current one is larger
                    if max_in_node > max_col_index:
                        max_col_index = max_in_node

        if max_d_index<=max_col_index:
            max_d_index=-1

        if max_d_index!=-1:
            SN_loc=np.where(P_old==max_d_index)[0][0]
            P_update_1=np.insert(P_old,SN_loc + 1, loc_np)
            coun=0
            for node in supernodes:
                if node.fixed==False and max_d_index!=-1:
                    node.update_flag=True
                    if SN_loc in node.column_indices:
                        coun+=1
                        node.column_indices=[x + coun if x > SN_loc else x for x in node.column_indices]
                        node.row_indices=[x + coun if x > SN_loc else x for x in node.row_indices]
                        node.column_indices.insert(-1,SN_loc+1)
                        node.row_indices.insert(-1,SN_loc+1)
                        node.column_indices.sort()
                        node.row_indices.sort()

            for node in supernodes:
                if node.fixed==False and max_d_index!=-1:        
                        node.update_flag=True
                        if coun != 0 and SN_loc not in node.column_indices:
                            node.column_indices=[x + coun if x > SN_loc else x for x in node.column_indices]
                            node.row_indices=[x + coun if x > SN_loc else x for x in node.row_indices]
                            node.column_indices.sort()
                            node.row_indices.sort()
        else:
            SN_loc=loc_np
            P_update_1=np.hstack((P_old,loc_np))
            temp_col=list(loc_np)
            temp_row=[]
            for node in supernodes:
                if node.fixed==False:
                    temp_row.append(node.row_indices)
            temp_row.append(loc_np)
            temp_row=[x for sublist in temp_row for x in sublist]
            temp_row=list(set(temp_row))

            temp_row.sort()
            supernodes.append(IndexSuperNode(temp_col, temp_row,fixed=False,update_flag=True))

        supernodes_wo_der=copy.deepcopy(supernodes)
        P_wo_der=copy.deepcopy(P_update_1)

        # Need to get length of each measurement list and then calculate Nd for multi-dimensional problems
        N=len(measurements_list[0])
        Nd=0
        for i in range(lm):
            Nd+=len(measurements_list[i])

        P_all=np.zeros(Nd,dtype=int)
        
        d=int(Nd/N)

        for i in range(N):
            P_all[i*d:(i+1)*d]=np.array([P_update_1[i]+N*j for j in range(d)])

        measurements = [m for sublist in measurements_list for m in sublist]

        measurements = [measurements[i] for i in P_all]

        for node in supernodes:
            m=len(node.row_indices)
            n=len(node.column_indices)

            for i in range(m):
                rowi = node.row_indices[i]
                # all the nodes will be rearranged
                node.row_indices[i] = (rowi+1)*d-1
                node.row_indices.extend(range(rowi*d, (rowi+1)*d-1))

            node.row_indices.sort()


            for j in range(n):
                columni = node.column_indices[j]
                node.column_indices[j] = (columni+1)*d-1
                node.column_indices.extend(range(columni*d, (columni+1)*d-1))

            node.column_indices.sort()

        supernodes = IndirectSupernodalAssignment(supernodes,measurements)

        return ImplicitKLFactorization(P_all, supernodes, 𝒢),supernodes_wo_der,P_wo_der,L_old


   
    # ordering points first and then ordering each derivatives order in the same ordering as point measurement
    def implicit_kl_factorization_for_d_V2(𝒢, measurements_list, ρ, k_neighbors, lambda_=1.5, alpha=1.0, Tree=KDTree):
        lm = len(measurements_list)
        assert lm >= 2 #making sure derivative measurements are in it
        
        #sending only the point measurement 
        P, ℓ, supernodes = ordering_and_sparsity_pattern(measurements_list[0],k_neighbors= k_neighbors, lambda_=lambda_, alpha=alpha, Tree=Tree,rho=ρ)
        
        supernodes_wo_der=copy.deepcopy(supernodes)
        P_wo_der=copy.deepcopy(P)

        N=len(measurements_list[0])
        # Need to get length of each measurement list and then calculate Nd for multi-dimensional problems
        Nd=0
        for i in range(lm):
            Nd+=len(measurements_list[i])
        P_all=np.zeros(Nd,dtype=int)
        d=int(Nd/N)
        P_all[:N]=P
        for i in range(d-1):

            P_all[N*(i+1):N*(i+2)]=P+(N*(i+1))

        measurements = [m for sublist in measurements_list for m in sublist]
        measurements = [measurements[i] for i in P_all]

        der_supernodes=[]
        for k in range(1,d):
            for node in supernodes:
                m=len(node.row_indices)
                n=len(node.column_indices)

                temp_row=[]
                for i in range(m):
                    rowi = node.row_indices[i]
                    temp_row.append(rowi+(N*k))
                    temp_row.extend(range(rowi, rowi+(N*k)))
                temp_row.sort()
                temp_row=list(set(temp_row))


                temp_col=[]
                for j in range(n):
                    columni = node.column_indices[j]
                    temp_col.append(columni+(N*k))
                temp_col.sort()
                der_supernodes.append(IndexSuperNode(temp_col, temp_row))

        supernodes_f=supernodes+der_supernodes

        supernodes = IndirectSupernodalAssignment(supernodes_f,measurements)

        return ImplicitKLFactorization(P_all, supernodes, 𝒢),supernodes_wo_der,P_wo_der
    


    # used to test PDE problem
    def implicit_kl_factorization_follow_diracs(𝒢, measurements_list, ρ, k_neighbors, lambda_=1.5, alpha=1.0, Tree=KDTree):
        lm = len(measurements_list)
        assert lm >= 3
        P, ℓ, supernodes = ordering_and_sparsity_pattern(measurements_list[:2], k_neighbors, lambda_=lambda_, alpha=alpha, Tree=Tree,rho=ρ)

        N_boundary = len(measurements_list[0])
        N_domain = len(measurements_list[1])
        P_all = np.zeros(N_boundary + (lm - 1) * N_domain, dtype=int)

        P_all[:N_boundary] = P[:N_boundary]

        temp=[]
        for i in range(N_boundary,N_boundary+N_domain):
            for j in range(lm-1):
                temp.append(P[i]+j*N_domain)
        P_all[N_boundary:]=np.array(temp)


        measurements = [m for sublist in measurements_list for m in sublist]
        measurements = [measurements[i] for i in P_all]

        for node in supernodes:
            m = len(node.row_indices)
            n = len(node.column_indices)

            for i in range(m):
                rowi = node.row_indices[i]
                if rowi > N_boundary-1:
                    node.row_indices[i] = (rowi - N_boundary) * (lm - 1) + N_boundary
                    node.row_indices.extend(range((rowi - N_boundary) * (lm - 1) + N_boundary+1,(rowi - N_boundary + 1) * (lm - 1) + N_boundary))
            node.row_indices.sort()



            for j in range(n):
                columni = node.column_indices[j]

                if columni > N_boundary-1:
                    node.column_indices[j] = (columni - N_boundary ) * (lm - 1) + N_boundary
                    node.column_indices.extend(range((columni - N_boundary) * (lm - 1) + N_boundary +1, (columni - N_boundary+1) * (lm - 1) + N_boundary))
            node.column_indices.sort()

        supernodes = IndirectSupernodalAssignment(supernodes,measurements)

       
        return ImplicitKLFactorization(P_all, supernodes, 𝒢)

    def implicit_kl_factorization_diracs_first_then_unif_scale(𝒢, measurements_list, ρ, k_neighbors, lambda_=1.5, alpha=1.0, Tree=KDTree):
        x = [np.column_stack([m.get_coordinate() for m in measurements]) for measurements in measurements_list]
        P, ℓ, supernodes = ordering_and_sparsity_pattern_diracs_first_then_unif_scale(x, ρ, k_neighbors, lambda_=lambda_, alpha=alpha, Tree=Tree)
        measurements = [m for sublist in measurements_list for m in sublist]
        measurements = [measurements[i] for i in P]
        supernodes = IndirectSupernodalAssignment(supernodes, measurements)
        return ImplicitKLFactorization(P, supernodes, 𝒢)

