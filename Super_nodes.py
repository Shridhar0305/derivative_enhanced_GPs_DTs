
import numpy as np
# from numba import njit


def issorted_isn(node):
    return all(np.diff(node.row_indices) >= 0) and all(np.diff(node.column_indices) >= 0)

def issorted_isa(assignment):
    return all(issorted_isn(node) for node in assignment.supernodes) and all(node.column_indices[0] <= next_node.column_indices[0] for node, next_node in zip(assignment.supernodes[:-1], assignment.supernodes[1:]))

class IndexSuperNode:
    def __init__(self, column_indices, row_indices,update_flag=False,fixed=True):
        self.column_indices = [int(x) for x in column_indices] # convert to int
        self.row_indices = [int(x) for x in row_indices] # convert to int
        self.update_flag=update_flag # this flag is set to true for a dynamic supernode
        self.fixed=fixed # this flag is set to true if the new data creates a new supernode

    def column_indices_p(self):
        return self.column_indices

    def row_indices_p(self):
        return self.row_indices

    def size(self, dim=None):
        if dim is None:
            return (len(self.row_indices), len(self.column_indices))
        elif dim == 0 or dim == 1:
            return (len(self.row_indices), len(self.column_indices))[dim]
        else:
            raise ValueError("dim must be 0 or 1")
    def __str__(self):
        return f"IndexSuperNode(row_indices={self.row_indices}, column_indices={self.column_indices},update_flag={self.update_flag},fixed={self.fixed})"

    def __repr__(self):
        return self.__str__()


class IndirectSupernodalAssignment:
    def __init__(self, supernodes, measurements):
        self.supernodes = supernodes
        self.measurements = measurements

    def __str__(self):
        return f"IndirectSupernodalAssignment(supernodes={self.supernodes}, measurements_length={len(self.measurements)})"

    def __repr__(self):
        return self.__str__()
