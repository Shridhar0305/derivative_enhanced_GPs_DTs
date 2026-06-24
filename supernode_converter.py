import numpy as np


def convert_supernodes_to_structured_array(supernodal_assignment):

    supernode_dtype = np.dtype([
        ('size0', 'i4'),
        ('size1', 'i4'),
        ('update_flag', 'b1'),  # boolean
        ('fixed', 'b1'),        # boolean
        ('row_indices', object),
        ('col_indices', object),
    ])

    # Create and populate the structured array.
    supernodes = supernodal_assignment.supernodes
    num_supernodes = len(supernodes)

    supernodes_array = np.empty(num_supernodes, dtype=supernode_dtype)
    for i, node in enumerate(supernodes):
        
        supernodes_array[i] = (node.size(0), node.size(1), node.update_flag, node.fixed,
                               np.array(node.row_indices_p(), dtype=np.int32),
                               np.array(node.column_indices_p(), dtype=np.int32))

    return supernodes_array


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

def convert_supernodes_to_list(supernodal_assignment):
    """
    Converts the supernodes from a supernodal_assignment object into a dictionary of dictionaries.
    """
    supernodes_list = []
    for i, node in enumerate(supernodal_assignment.supernodes):
        node_dict = {
            'size0': node.size(0),
            'size1': node.size(1),
            'update_flag': node.update_flag,
            'fixed': node.fixed,
            'row_indices': node.row_indices_p(),
            'col_indices': node.column_indices_p(),
        }
        supernodes_list.append(node_dict)
    return supernodes_list


from meas import (
    AbstractMeasurement,
    PointMeasurement,
    dPointMeasurement,
    ddPointMeasurement,
    dddPointMeasurement,
    ddddPointMeasurement,
)
from typing import List


def convert_measurements_to_numpy(
    measurements: List[AbstractMeasurement],
) -> (np.ndarray, np.ndarray):
    """
    Converts a list of AbstractMeasurement objects into two NumPy arrays
    for efficient processing, potentially in Cython.

    Args:
        measurements (List[AbstractMeasurement]): A list of measurement objects.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - M_numpy_array: A structured NumPy array with supernode data.
            - input_cord: A NumPy array with measurement coordinates.
    """
    if not measurements:
        return np.array([]), np.array([])

    n_rows = len(measurements)
    # Assuming all coordinates have the same dimension
    dim = measurements[0].get_coordinate().shape[0]

    struct_dtype = np.dtype([("index", "i4", (4,)), ("lent", "i4")])
    M_numpy_array = np.zeros(n_rows, dtype=struct_dtype)
    input_cord = np.zeros((n_rows, dim))

    for i, m in enumerate(measurements):
        input_cord[i] = m.get_coordinate()
        if isinstance(m, PointMeasurement):
            M_numpy_array[i]["index"] = [0, 0, 0, 0]
            M_numpy_array[i]["lent"] = 0
        elif isinstance(m, dPointMeasurement):
            M_numpy_array[i]["index"] = [m.derivative_index, 0, 0, 0]
            M_numpy_array[i]["lent"] = 1
        elif isinstance(m, (ddPointMeasurement, dddPointMeasurement, ddddPointMeasurement)):
            indices = list(m.derivative_indices)
            lent = len(indices)
            # Pad with zeros to length 4
            padded_indices = indices + [0] * (4 - lent)
            M_numpy_array[i]["index"] = padded_indices
            M_numpy_array[i]["lent"] = lent
        else:
            raise TypeError(f"Unsupported measurement type: {type(m)}")

    return M_numpy_array, input_cord
