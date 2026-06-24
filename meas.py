import numpy as np
from abc import ABC, abstractmethod
# from numba import njit


# Abstract base class for measurements (like your PointMeasurement, etc.)

class AbstractMeasurement(ABC):
    @abstractmethod
    def get_coordinate(self):
        pass

class AbstractPointMeasurement(AbstractMeasurement):
    pass



class PointMeasurement(AbstractMeasurement):
    def __init__(self, coordinate):
        self.coordinate = np.array(coordinate)

    def get_coordinate(self):
        return self.coordinate

class dPointMeasurement(AbstractMeasurement):
    def __init__(self, coordinate, derivative_index):
        self.coordinate = np.array(coordinate)
        self.derivative_index = derivative_index

    def get_coordinate(self):
        return self.coordinate

class ddPointMeasurement(AbstractMeasurement):
    def __init__(self, coordinate, derivative_indices):
        self.coordinate = np.array(coordinate)
        self.derivative_indices = derivative_indices # Tuple of (index1, index2) for the second derivative

    def get_coordinate(self):
        return self.coordinate
    
class dddPointMeasurement(AbstractMeasurement):
    def __init__(self, coordinate, derivative_indices):
        self.coordinate = np.array(coordinate)
        self.derivative_indices = derivative_indices # Tuple of (index1, index2, index3) for the third derivative

    def get_coordinate(self):
        return self.coordinate

class ddddPointMeasurement(AbstractMeasurement):
    def __init__(self, coordinate, derivative_indices):
        self.coordinate = np.array(coordinate)
        self.derivative_indices = derivative_indices # Tuple of (index1, index2, index3, index4) for the fourth derivative

    def get_coordinate(self):
        return self.coordinate


class PointIndexMeasurement(AbstractMeasurement):
    def __init__(self, index):
        self.index = index

    def get_coordinate(self):
        return self.index

# below are the measurements used for PDE solving    
class ΔδPointMeasurement(AbstractPointMeasurement):
    def __init__(self, coordinate, weight_Δ=1.0, weight_δ=0.0):
        self.coordinate = np.array(coordinate)
        self.weight_Δ = weight_Δ
        self.weight_δ = weight_δ
        
    def get_coordinate(self):
        return self.coordinate

class ΔΔδPointMeasurement(AbstractPointMeasurement):
    def __init__(self, coordinate, weight_ΔΔ, weight_Δ=1.0, weight_δ=0.0):
        self.coordinate = np.array(coordinate)
        self.weight_ΔΔ = np.array(weight_ΔΔ)
        self.weight_Δ = weight_Δ
        self.weight_δ = weight_δ
    def get_coordinate(self):
        return self.coordinate

class ΔΔΔPointMeasurement(AbstractPointMeasurement):
    def __init__(self, coordinate, weight_Δ11=0.0, weight_Δ12=0.0, weight_Δ22=0.0):
        self.coordinate = np.array(coordinate)
        self.weight_Δ11 = weight_Δ11
        self.weight_Δ12 = weight_Δ12
        self.weight_Δ22 = weight_Δ22
    def get_coordinate(self):
        return self.coordinate