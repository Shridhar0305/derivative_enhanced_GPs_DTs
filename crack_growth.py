import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

from scipy.optimize import minimize
from scipy.integrate import cumulative_trapezoid
from scipy.spatial.distance import cdist

from Cov import GaussianCovariance_generic
from Factors import ExplicitKLFactorization
from Factors import ImplicitKLFactorization

from meas import (
    PointMeasurement,
    dPointMeasurement,
    ddPointMeasurement,
    dddPointMeasurement
)

from supernode_converter import convert_measurements_to_list_of_dicts
from scipy.special import ellipe

#helper function(s)
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

# def f_w(f_w_input_1, f_w_input_2):
#     return ((f_w_input_1 + np.cos(f_w_input_1))**0.25) / ((np.cos(f_w_input_2))**0.5) 


# def f_w(f_w_input_1, f_w_input_2, eps=1e-12):

#     raw1 = f_w_input_1 + np.cos(f_w_input_1)
#     raw2 = np.cos(f_w_input_2)

#     if np.any(raw1 <= 0):
#         print("WARNING: invalid term1 encountered")

#     if np.any(raw2 <= 0):
#         print("WARNING: invalid term2 encountered")

#     term1 = np.maximum(raw1, eps)
#     term2 = np.maximum(raw2, eps)

#     return np.power(term1, 0.25) / np.sqrt(term2)

def f_w(f_w_input_1, f_w_input_2, eps=1e-12):
    num = f_w_input_1 + np.cos(f_w_input_1)
    num = np.maximum(num, eps)

    cos_y = np.cos(f_w_input_2)
    cos_y = np.maximum(cos_y, eps)

    return np.power(num, 0.25) / np.sqrt(cos_y)

def M(f_w_input_1, M_input_2):
    return (1 + (0.06 * (f_w_input_1**2) * (M_input_2 - 1)) + ((1 - (1 / M_input_2)) * ((0.0069 * (1/M_input_2)) - (0.28 * f_w_input_1))))
    
def g(f_w_input_1, M_input_2, phi):
    return (1 + (f_w_input_1 * ((1-np.sqrt(np.sin(phi))) / (M_input_2 + 1))))
    
# def Q(a, c):
#     k = np.sqrt(1 - (a/c)**2)  
#     return ellipe(k)**2

# def Q(a, c):

#     ratio = a / c

#     k = np.where(
#         ratio <= 1,
#         np.sqrt(1 - ratio**2),
#         np.sqrt(1 - (1 / ratio**2))
#     )

#     return np.where(
#         ratio <= 1,
#         ellipe(k)**2,
#         (ratio**2) * ellipe(k)**2
#     )


# def Q(a, c):

#     ratio = a / c

#     k = np.where(
#         ratio <= 1,
#         np.sqrt(1 - ratio**2),
#         np.sqrt(1 - (1 / ratio**2))
#     )

#     return ellipe(k)**2


def Q(a, c):

    c_safe = np.sign(c) * np.maximum(np.abs(c), 1e-12)

    ratio = a / c_safe

    k = np.empty_like(ratio, dtype=np.float64)

    mask = ratio <= 1

    # a/c <= 1
    k[mask] = np.sqrt(1 - ratio[mask]**2)

    # a/c > 1
    k[~mask] = np.sqrt(1 - (1 / ratio[~mask]**2))

    return ellipe(k)**2


def F_base(l, a, c, t, b, phi, sigma):
    c_safe = np.sign(c) * np.maximum(np.abs(c), 1e-12)
    
    fw = f_w(a / t, c / b)
    Mv = M(a / t, a / c_safe)
    gv = g(a / t, a / c_safe, phi)

    return fw * Mv * gv
    
#need to find the l and Q to put here
def stress_intensity_factor_K(l, a, c, t, b, phi, sigma):
    c_safe = np.sign(c) * np.maximum(np.abs(c), 1e-12)
    
    f_w_input_1 = a / t
    f_w_input_2 = c / b
    M_input_2 = a / c_safe
    
    M_input_2 = np.clip(M_input_2, 0.2, 2)
    
    K = sigma * np.sqrt((np.pi * l) / Q(a, c)) * f_w(f_w_input_1, f_w_input_2) * M(f_w_input_1, M_input_2) * g(f_w_input_1, M_input_2, phi)
    
    return K

def paris_law_dc_dn(l, a, c, t, b, phi, sigma, C, m):
    a = np.clip(a, 0.001, 0.08)
    K = stress_intensity_factor_K(l, a, c, t, b, phi, sigma)
    
    K_safe = np.maximum(K, 1e-8)
    K_safe = np.clip(K, 1e-8, 1e3)
    dc_dN = C * (K_safe**m)
    return dc_dN
    
    
    
#     return C * (K**m)


#helper function
def dM_dc(l, a, c, t, b, phi, sigma):

    x = a / t
    y = np.clip(a / c, 0.2, 2.0)
    c_safe = np.sign(c) * np.maximum(np.abs(c), 1e-12)

    dy_dc = -a / (c_safe**2)

    term1 = 0.06 * (x**2) * dy_dc

    term2 = (
        (
            (1 / (y**2))
            *
            ((0.0069 / y) - (0.28 * x))
        )
        +
        (1 - (1 / y))
        *
        (-0.0069 / (y**2))
    ) * dy_dc

    return term1 + term2


def dg_dc(l, a, c, t, b, phi, sigma):

    x = a / t
    y = np.clip(a / c, 0.2, 2.0)
    c_safe = np.sign(c) * np.maximum(np.abs(c), 1e-12)


    dy_dc = -a / (c_safe**2)

    A = 1 - np.sqrt(np.sin(phi))

    return (
        -x * A / ((y + 1)**2)
    ) * dy_dc


def d2fw_dac(l, a, c, t, b, phi, sigma):

    x = a / t
    y = c / b

    A = x + np.cos(x)

    dA_da = (1 - np.sin(x)) / t

    B = (np.cos(y))**(-0.5)

    dB_dc = (
        0.5
        * np.sin(y)
        * (np.cos(y))**(-1.5)
        * (1 / b)
    )

    prefactor = 0.25 * A**(-0.75)

    return prefactor * dA_da * dB_dc

def d2M_dac(l, a, c, t, b, phi, sigma):

    x = a / t
    y = np.clip(a / c, 0.2, 2.0)
    c_safe = np.sign(c) * np.maximum(np.abs(c), 1e-12)

    dy_dc = -a / (c_safe**2)

    dy_da = 1 / c

    term1 = (
        0.12 * x / t
    ) * dy_dc

    A = (1 / (y**2))
    B = ((0.0069 / y) - (0.28 * x))

    dA_da = -2 / (y**3) * dy_da

    dB_da = (
        -0.0069 / (y**2)
    ) * dy_da - 0.28 / t

    term2a = dA_da * B + A * dB_da

    Cterm = (1 - (1 / y))

    dC_da = (1 / (y**2)) * dy_da

    Dterm = -0.0069 / (y**2)

    dD_da = 0.0138 / (y**3) * dy_da

    term2b = dC_da * Dterm + Cterm * dD_da

    return term1 + (term2a + term2b) * dy_dc


def d2g_dac(l, a, c, t, b, phi, sigma):

    x = a / t
    y = np.clip(a / c, 0.2, 2.0)
    c_safe = np.sign(c) * np.maximum(np.abs(c), 1e-12)

    dx_da = 1 / t
    dy_da = 1 / c
    dy_dc = -a / (c_safe**2)

    A = 1 - np.sqrt(np.sin(phi))

    base = -x * A / ((y + 1)**2)

    dbase_da = (
        -A * (
            dx_da / ((y + 1)**2)
            -
            (2 * x * dy_da) / ((y + 1)**3)
        )
    )

    return dbase_da * dy_dc


#helper function
def dfw_da(l, a, c, t, b, phi, sigma):
    x = a / t
    num = 1 + np.cos(x)
    return (0.25 * num**(-0.75) * (-np.sin(x)/t)) / np.sqrt(np.cos(c / b))


def d2fw_da2(l, a, c, t, b, phi, sigma):
    x = a / t
    num = 1 + np.cos(x)

    term1 = (-0.75) * (0.25) * num**(-1.75) * (np.sin(x)/t)**2
    term2 = (0.25) * num**(-0.75) * (-np.cos(x)/t**2)

    return (term1 + term2) / np.sqrt(np.cos(c / b))


def d3fw_da3(*args):
    return np.zeros_like(args[2]) 


def d4fw_da4(*args):
    return np.zeros_like(args[2])

def dM_da(l, a, c, t, b, phi, sigma):
    M2 = a / c
    dM2_da = 1 / c

    term1 = 0.12 * a * (M2 - 1) + 0.06 * a**2 * dM2_da
    term2 = (1 - 1/M2) * (0.0069 * (-1/M2**2) * dM2_da - 0.28 / c)

    return term1 + term2


def d2M_da2(*args):
    return np.zeros_like(args[2])

def d3M_da3(*args):
    return np.zeros_like(args[2])

def d4M_da4(*args):
    return np.zeros_like(args[2])

def dg_da(l, a, c, t, b, phi, sigma):
    M2 = a / c
    dM2_da = 1 / c

    return ((1 - np.sqrt(np.sin(phi))) / (M2 + 1)) + \
           a * (-(1 - np.sqrt(np.sin(phi))) / (M2 + 1)**2) * dM2_da


#helper functions
def dMv_da(l, a, c, t, b, phi, sigma):

    x = a / t
    y = np.clip(a / c, 0.2, 2.0)

    dx_da = 1 / t
    dy_da = 1 / c

    term1 = 0.12 * x * (y - 1) * dx_da

    term2 = 0.06 * x**2 * dy_da

    A = (1 - 1/y)
    B = (0.0069 / y) - 0.28 * x

    dA_da = (1 / y**2) * dy_da
    dB_da = -(0.0069 / y**2) * dy_da - 0.28 * dx_da

    term3 = dA_da * B + A * dB_da

    return term1 + term2 + term3


def d2Mv_da2(l, a, c, t, b, phi, sigma):

    y = np.clip(a / c, 0.2, 2.0)

    dx_da = 1 / t
    dy_da = 1 / c

    term1 = 0.12 * (y - 1) * dx_da**2

    term2 = 0.24 * (a / t) * dx_da * dy_da

    A = (1 - 1/y)
    B = (0.0069 / y) - 0.28 * (a / t)

    dA = (1 / y**2) * dy_da
    dB = -(0.0069 / y**2) * dy_da - 0.28 * dx_da

    d2A = -2 * dy_da**2 / y**3
    d2B = 0.0138 * dy_da**2 / y**3

    term3 = d2A * B + 2 * dA * dB + A * d2B

    return term1 + term2 + term3


def d3Mv_da3(l, a, c, t, b, phi, sigma):

    y = np.clip(a / c, 0.2, 2.0)

    dy = 1 / c

    A = (1 - 1/y)
    B = (0.0069 / y) - 0.28 * (a / t)

    dA = dy / y**2
    dB = -0.0069 * dy / y**2 - 0.28 / t

    d2A = -2 * dy**2 / y**3
    d2B = 0.0138 * dy**2 / y**3

    d3A = 6 * dy**3 / y**4
    d3B = -0.0414 * dy**3 / y**4

    return (
        d3A * B
        + 3 * d2A * dB
        + 3 * dA * d2B
        + A * d3B
    )


def d4Mv_da4(l, a, c, t, b, phi, sigma):

    y = np.clip(a / c, 0.2, 2.0)

    dy = 1 / c

    A = (1 - 1/y)
    B = (0.0069 / y) - 0.28 * (a / t)

    dA = dy / y**2
    dB = -0.0069 * dy / y**2 - 0.28 / t

    d2A = -2 * dy**2 / y**3
    d2B = 0.0138 * dy**2 / y**3

    d3A = 6 * dy**3 / y**4
    d3B = -0.0414 * dy**3 / y**4

    d4A = -24 * dy**4 / y**5
    d4B = 0.1656 * dy**4 / y**5

    return (
        d4A * B
        + 4 * d3A * dB
        + 6 * d2A * d2B
        + 4 * dA * d3B
        + A * d4B
    )

def dgv_da(l, a, c, t, b, phi, sigma):

    S = 1 - np.sqrt(np.sin(phi))

    y = np.clip(a / c, 0.2, 2.0)

    A = a / t
    B = y + 1

    dA = 1 / t
    dB = 1 / c

    return S * ((dA * B - A * dB) / B**2)


def d2gv_da2(l, a, c, t, b, phi, sigma):

    S = 1 - np.sqrt(np.sin(phi))

    y = np.clip(a / c, 0.2, 2.0)

    A = a / t
    B = y + 1

    dA = 1 / t
    dB = 1 / c

    numerator = -2 * (dA * B - A * dB) * dB

    return S * numerator / B**3


def d3gv_da3(l, a, c, t, b, phi, sigma):

    S = 1 - np.sqrt(np.sin(phi))

    y = np.clip(a / c, 0.2, 2.0)

    A = a / t
    B = y + 1

    dA = 1 / t
    dB = 1 / c

    P = dA * B - A * dB

    return 6 * S * P * dB**2 / B**4


def d4gv_da4(l, a, c, t, b, phi, sigma):

    S = 1 - np.sqrt(np.sin(phi))

    y = np.clip(a / c, 0.2, 2.0)

    A = a / t
    B = y + 1

    dA = 1 / t
    dB = 1 / c

    P = dA * B - A * dB

    return -24 * S * P * dB**3 / B**5


def d2g_da2(*args):
    return np.zeros_like(args[2])

def d3g_da3(*args):
    return np.zeros_like(args[2])

def d4g_da4(*args):
    return np.zeros_like(args[2])


def dfw_dc(l, a, c, t, b, phi, sigma):
    x = c / b

    fw_num = (a/t + np.cos(a/t))**0.25
    denom = (np.cos(x))**0.5

    ddenom_dx = -0.5 * (np.cos(x)**(-0.5)) * (-np.sin(x))
    dx_dc = 1/b

    return fw_num * ddenom_dx * dx_dc / (denom**2)


def d2fw_dc2(l, a, c, t, b, phi, sigma):
    x = c / b

    fw_num = (a/t + np.cos(a/t))**0.25

    cosx = np.cos(x)
    sinx = np.sin(x)

    dx = 1/b

    term1 = fw_num * (0.75 * cosx**(-2.5) * sinx**2)
    term2 = fw_num * (0.5 * cosx**(-1.5) * cosx)

    return (term1 + term2) * dx**2


def d3fw_dc3(*args):
    return np.zeros_like(args[2]) 


def d4fw_dc4(*args):
    return np.zeros_like(args[2])

def dMv_dc(l, a, c, t, b, phi, sigma):
    M2 = a / c
    dM2_dc = -a / (c**2)

    term1 = 0.06 * a**2 * dM2_dc

    term2 = (
        (1 - 1/M2) * (-0.0069 / (M2**2)) * dM2_dc
        + (1 / (M2**2)) * dM2_dc * (0.0069 / M2 - 0.28 * a/t)
    )

    return term1 + term2


def d2Mv_dc2(*args):
    return np.zeros_like(args[2])

def d3Mv_dc3(*args):
    return np.zeros_like(args[2])

def d4Mv_dc4(*args):
    return np.zeros_like(args[2])

def dgv_dc(l, a, c, t, b, phi, sigma):

    M2 = a / c
    dM2_dc = -a / (c**2)

    A = (1 - np.sqrt(np.sin(phi)))

    fw = f_w(a/t, c/b)

    term1 = fw * (-A / (M2 + 1)**2) * dM2_dc

    return term1


def d2gv_dc2(*args):
    return np.zeros_like(args[2])

def d3gv_dc3(*args):
    return np.zeros_like(args[2])

def d4gv_dc4(*args):
    return np.zeros_like(args[2])



# helper functions
def d3fw_da2dc(l, a, c, t, b, phi, sigma):

    x = a / t
    y = c / b

    A = x + np.cos(x)
    A1 = (1 - np.sin(x)) / t
    A2 = -(np.cos(x)) / (t**2)

    B1 = (
        0.5
        * np.sin(y)
        * (np.cos(y))**(-1.5)
        * (1 / b)
    )

    P1 = 0.25 * A**(-0.75)

    P2 = (
        -0.1875
        * A**(-1.75)
        * A1
    )

    return (P2 * A1 + P1 * A2) * B1


def d2M_dc2(l, a, c, t, b, phi, sigma):

    x = a / t
    y = np.clip(a / c, 0.2, 2.0)
    dy_dc = -a / (c**2)
    d2y_dc2 = 2 * a / (c**3)

    A1 = 0.06 * (x**2)

    d2A_dc2 = A1 * d2y_dc2
    A = (1 - 1/y)
    B = (0.0069/y - 0.28*x)

    # derivatives wrt y
    dA_dy = 1 / (y**2)
    d2A_dy2 = -2 / (y**3)

    dB_dy = -0.0069 / (y**2)
    d2B_dy2 = 2 * 0.0069 / (y**3)

    term1 = d2A_dy2 * (dy_dc**2) * B
    term2 = dA_dy * d2y_dc2 * B
    term3 = 2 * dA_dy * dy_dc * dB_dy * dy_dc
    term4 = A * d2B_dy2 * (dy_dc**2)
    term5 = A * dB_dy * d2y_dc2

    return d2A_dc2 + (term1 + term2 + term3 + term4 + term5)


def d3M_da2dc(l, a, c, t, b, phi, sigma):

    x = a / t
    y = np.clip(a / c, 0.2, 2.0)

    dy_da = 1 / c
    dy_dc = -a / (c**2)
    term1 = (0.12 / (t**2)) * dy_dc

    A = 1 / (y**2)
    B = (0.0069 / y) - (0.28 * x)

    A1 = -2 / (y**3) * dy_da

    B1 = (
        -0.0069 / (y**2)
    ) * dy_da - 0.28 / t

    A2 = (
        6 / (y**4)
    ) * (dy_da**2)

    B2 = (
        0.0138 / (y**3)
    ) * (dy_da**2)

    second1 = A2 * B + 2 * A1 * B1 + A * B2
    Cterm = 1 - (1 / y)

    C1 = (1 / (y**2)) * dy_da

    C2 = (
        -2 / (y**3)
    ) * (dy_da**2)

    Dterm = -0.0069 / (y**2)

    D1 = (
        0.0138 / (y**3)
    ) * dy_da

    D2 = (
        -0.0414 / (y**4)
    ) * (dy_da**2)

    second2 = (
        C2 * Dterm
        +
        2 * C1 * D1
        +
        Cterm * D2
    )

    return term1 + (second1 + second2) * dy_dc


def d3g_da2dc(l, a, c, t, b, phi, sigma):

    x = a / t
    y = np.clip(a / c, 0.2, 2.0)

    dx_da = 1 / t

    dy_da = 1 / c
    dy_dc = -a / (c**2)

    A = 1 - np.sqrt(np.sin(phi))
    T1 = dx_da / ((y + 1)**2)

    T2 = (
        2 * x * dy_da
    ) / ((y + 1)**3)

    dT1_da = (
        -2 * dx_da * dy_da
    ) / ((y + 1)**3)

    dT2_da = (
        (
            2 * dx_da * dy_da
        ) / ((y + 1)**3)
        -
        (
            6 * x * (dy_da**2)
        ) / ((y + 1)**4)
    )

    second_da2 = -A * (dT1_da - dT2_da)

    return second_da2 * dy_dc


# helper function
def d2g_dc2(l, a, c, t, b, phi, sigma):

    s = np.sqrt(np.sin(phi))

    M2 = np.clip(a / c, 0.2, 2.0)

    A = a * (1 - s)

    dM2_dc = -a / (c**2)

    d2M2_dc2 = 2 * a / (c**3)

    term1 = (
        2 * A * (dM2_dc**2)
        /
        ((M2 + 1)**3)
    )

    term2 = (
        -A * d2M2_dc2
        /
        ((M2 + 1)**2)
    )

    return term1 + term2


def d3fw_dadc2(l, a, c, t, b, phi, sigma):

    x = a / t
    y = c / b

    A = (x + np.cos(x))**0.25
    B = (np.cos(y))**(-0.5)

    u = x + np.cos(x)

    ux = (1 - np.sin(x)) / t

    uxx = (-np.cos(x)) / (t**2)

    A1 = 0.25 * (u**(-0.75)) * ux

    A2 = (
        0.25 * (-0.75) * (u**(-1.75)) * (ux**2)
        +
        0.25 * (u**(-0.75)) * uxx
    )

    sec_y = 1 / np.cos(y)
    tan_y = np.tan(y)

    By = (
        0.5
        *
        (sec_y**0.5)
        *
        tan_y
        /
        b
    )

    Byy = (
        0.5
        *
        (
            0.5 * (sec_y**0.5) * (tan_y**2)
            +
            (sec_y**2.5)
        )
        /
        (b**2)
    )

    return A1 * Byy


def d3M_dadc2(l, a, c, t, b, phi, sigma):

    x = a / t

    y = np.clip(a / c, 0.2, 2.0)

    dy_da = 1 / c

    dy_dc = -a / (c**2)

    d2y_dc2 = 2 * a / (c**3)

    d2y_dadc = -1 / (c**2)

    term1 = (
        0.12
        *
        x
        /
        t
        *
        d2y_dc2
    )

    A = (1 - 1/y)

    B = (0.0069/y - 0.28*x)

    # derivatives wrt a
    dA_da = (1 / y**2) * dy_da

    dB_da = -0.0069 * dy_da / (y**2) - 0.28 / t

    # derivatives wrt c
    dA_dc = (1 / y**2) * dy_dc

    dB_dc = -0.0069 * dy_dc / (y**2)

    # second wrt c
    d2A_dc2 = (
        (-2 / y**3) * (dy_dc**2)
        +
        (1 / y**2) * d2y_dc2
    )

    d2B_dc2 = (
        -0.0069
        *
        (
            d2y_dc2 / y**2
            -
            2 * (dy_dc**2) / y**3
        )
    )

    # mixed derivative
    d2A_dadc = (
        (-2 / y**3) * dy_da * dy_dc
        +
        (1 / y**2) * d2y_dadc
    )

    d2B_dadc = (
        -0.0069
        *
        (
            d2y_dadc / y**2
            -
            2 * dy_da * dy_dc / y**3
        )
    )

    term2 = (
        d2A_dadc * dB_dc
        +
        dA_da * d2B_dc2
        +
        d2A_dc2 * dB_da
        +
        dA_dc * d2B_dadc
    )

    return term1 + term2


def d3g_dadc2(l, a, c, t, b, phi, sigma):

    s = np.sqrt(np.sin(phi))

    y = np.clip(a / c, 0.2, 2.0)

    A = a * (1 - s)

    dy_da = 1 / c

    dy_dc = -a / (c**2)

    d2y_dc2 = 2 * a / (c**3)

    d2y_dadc = -1 / (c**2)

    term1 = (
        6
        *
        A
        *
        dy_da
        *
        (dy_dc**2)
        /
        ((y + 1)**4)
    )

    term2 = (
        -2
        *
        A
        *
        d2y_dadc
        *
        dy_dc
        /
        ((y + 1)**3)
    )

    term3 = (
        -2
        *
        A
        *
        dy_da
        *
        d2y_dc2
        /
        ((y + 1)**3)
    )

    return term1 + term2 + term3


# helper functions
def d4fw_dadc3(l, a, c, t, b, phi, sigma):

    x = a / t
    y = c / b

    A = x + np.cos(x)

    Ax   = (1 - np.sin(x)) / t
    Axx  = (-np.cos(x)) / (t**2)
    Axxx = (np.sin(x)) / (t**3)
    Axxxx = (np.cos(x)) / (t**4)

    By    = (0.5 * np.sin(y)) / (b * (np.cos(y)**1.5))

    Byy = (
        (0.5 / b**2)
        * (
            np.cos(y) / (np.cos(y)**1.5)
            +
            1.5 * (np.sin(y)**2) / (np.cos(y)**2.5)
        )
    )

    Byyy = (
        (1 / b**3)
        * (
            np.sin(y) / (np.cos(y)**1.5)
            +
            2.25 * np.sin(y) / (np.cos(y)**2.5)
            +
            3.75 * (np.sin(y)**3) / (np.cos(y)**3.5)
        )
    )

    Byyyy = (
        (1 / b**4)
        * (
            np.cos(y)**(-0.5)
            +
            7.5 * np.sin(y)**2 * np.cos(y)**(-2.5)
            +
            13.125 * np.sin(y)**2 * np.cos(y)**(-3.5)
            +
            13.125 * np.sin(y)**4 * np.cos(y)**(-4.5)
        )
    )

    alpha = 0.25

    f1 = alpha * A**(alpha - 1) * Ax

    f2 = (
        alpha * (alpha - 1) * A**(alpha - 2) * Ax * Axx
        +
        alpha * A**(alpha - 1) * Axxx
    )

    return f2 * Byyy + f1 * Byyyy


def d4M_dadc3(l, a, c, t, b, phi, sigma):

    x = a / t
    y = np.clip(a / c, 0.2, 2.0)

    dy  = -a / c**2
    d2y =  2*a / c**3
    d3y = -6*a / c**4

    dy_da  = 1 / c
    d2y_dadc = -1 / c**2
    d3y_dadc2 = 2 / c**3

    term1 = 0.12 * x * dy_da * d3y

    A = 1 - 1/y
    B = 0.0069/y - 0.28*x

    dA1 = 1 / y**2
    dA2 = -2 / y**3
    dA3 = 6 / y**4
    dA4 = -24 / y**5

    dB1 = -0.0069 / y**2
    dB2 = 2 * 0.0069 / y**3
    dB3 = -6 * 0.0069 / y**4
    dB4 = 24 * 0.0069 / y**5

    mixed = (
        dA4 * dy_da * dy**3 * B
        +
        3 * dA3 * dy_da * dy * d2y * B
        +
        dA2 * dy_da * d3y * B
        +
        dA1 * d3y_dadc2 * B

        +

        dA3 * dy_da * dy**2 * dB1
        +
        2 * dA2 * dy_da * d2y * dB1
        +
        dA2 * dy**2 * dB2 * dy_da
    )

    return term1 + mixed


def d4g_dadc3(l, a, c, t, b, phi, sigma):

    S = 1 - np.sqrt(np.sin(phi))

    y = np.clip(a / c, 0.2, 2.0)

    B = y + 1

    dy  = -a / c**2
    d2y =  2*a / c**3
    d3y = -6*a / c**4

    dy_da = 1 / c

    P = (1 / t) * B - (a / t) * dy_da

    term = (
        -24 * P * dy**3 / B**5
        +
        18 * dy * d2y / B**4
        -
        3 * d3y / B**3
    )

    return S * term

# helper function
def dK_dc(l, a, c, t, b, phi, sigma):

    a = np.clip(a, 0.001, 0.08)

    fw1 = a / t
    fw2 = c / b
    M2  = a / c

    M2 = np.clip(M2, 0.2, 2.0)

    base = sigma * np.sqrt((np.pi * l) / Q(a, c))

    A = f_w(fw1, fw2)
    B = M(fw1, M2)
    G = g(fw1, M2, phi)

    u = fw1 + np.cos(fw1)

    dA_dc = (
        (u ** 0.25)
        *
        (
            0.5
            * np.sin(fw2)
            * (np.cos(fw2) ** (-1.5))
            * (1 / b)
        )
    )

    dM2_dc = -a / (c ** 2)

    H1 = 0.06 * (fw1 ** 2) * (M2 - 1)

    dH1_dc = (
        0.06
        * (fw1 ** 2)
        * dM2_dc
    )

    H2a = (1 - (1 / M2))
    H2b = ((0.0069 / M2) - (0.28 * fw1))

    dH2a_dc = (
        (1 / (M2 ** 2))
        * dM2_dc
    )

    dH2b_dc = (
        (-0.0069 / (M2 ** 2))
        * dM2_dc
    )

    dH2_dc = (
        dH2a_dc * H2b
        +
        H2a * dH2b_dc
    )

    dB_dc = dH1_dc + dH2_dc

    S = (1 - np.sqrt(np.sin(phi)))

    dG_dc = (
        fw1
        * S
        *
        (
            -dM2_dc
            / ((M2 + 1) ** 2)
        )
    )

    return base * (
        dA_dc * B * G
        +
        A * dB_dc * G
        +
        A * B * dG_dc
    )


# helper function
def d2K_dc2(l, a, c, t, b, phi, sigma):
    
    f_w_input_1 = a / t
    f_w_input_2 = c / b
    M_input_2 = a / c

    base = sigma * np.sqrt((np.pi * l) / Q(a, c))

    A = f_w(f_w_input_1, f_w_input_2)
    B = M(f_w_input_1, M_input_2)
    G = g(f_w_input_1, M_input_2, phi)

    # d(f_w)/dc
    dfw_dc = (np.sin(f_w_input_2) / (2 * b * (np.cos(f_w_input_2) ** 0.5)))

    # d2(f_w)/dc2
    dfw_dc2 = (
        (np.cos(f_w_input_2) / (2 * b**2 * np.cos(f_w_input_2)**0.5))
        + (np.sin(f_w_input_2)**2 / (4 * b**2 * np.cos(f_w_input_2)**1.5))
    )

    # M derivatives
    dM2_dc = -a / (c**2)

    dM_dM2 = (
        0.06 * f_w_input_1**2
        + (1 / (M_input_2**2)) * ((0.0069 / M_input_2) - 0.28 * f_w_input_1)
        + (1 - 1 / M_input_2) * (-0.0069 / (M_input_2**2) + 0.28 * f_w_input_1 / (M_input_2**2))
    )

    d2M_dM22 = (
        (2 / (M_input_2**3)) * ((0.0069 / M_input_2) - 0.28 * f_w_input_1)
        + (2 / (M_input_2**3)) * (0.0069 / M_input_2 - 0.28 * f_w_input_1)
    )

    dM_dc = dM_dM2 * dM2_dc
    d2M_dc2 = d2M_dM22 * (dM2_dc**2) + dM_dM2 * (2 * a / (c**3))

    # g derivatives
    dg_dM2 = -f_w_input_1 * (1 - np.sqrt(np.sin(phi))) / ((M_input_2 + 1)**2)
    d2g_dM22 = 2 * f_w_input_1 * (1 - np.sqrt(np.sin(phi))) / ((M_input_2 + 1)**3)

    dg_dc = dg_dM2 * dM2_dc
    d2g_dc2 = d2g_dM22 * (dM2_dc**2) + dg_dM2 * (2 * a / (c**3))


    dK_dc = base * (
        dfw_dc * B * G
        + A * dM_dc * G
        + A * B * dg_dc
    )

    d2K_dc2 = base * (
        dfw_dc2 * B * G
        + 2 * dfw_dc * dM_dc * G
        + 2 * dfw_dc * B * dg_dc

        + A * d2M_dc2 * G
        + 2 * A * dM_dc * dg_dc

        + A * B * d2g_dc2
    )

    return d2K_dc2  



# helper function
def dK_da(l, a, c, t, b, phi, sigma):

    a = np.clip(a, 0.001, 0.08)

    fw1 = a / t
    fw2 = c / b
    M2  = a / c

    M2 = np.clip(M2, 0.2, 2.0)

    base = sigma * np.sqrt((np.pi * l) / Q(a, c))

    A = f_w(fw1, fw2)
    B = M(fw1, M2)
    G = g(fw1, M2, phi)


    u = fw1 + np.cos(fw1)

    du_da = (1.0 / t) - (np.sin(fw1) / t)

    dA_da = (
        0.25
        * (u ** (-0.75))
        * du_da
        * (np.cos(fw2) ** (-0.5))
    )

    dM2_da = 1.0 / c

    term1 = (
        0.12 * fw1 * (M2 - 1) / t
    )

    term2 = (
        0.06 * (fw1 ** 2) * dM2_da
    )

    H = (
        (1 - (1 / M2))
        * (
            (0.0069 / M2)
            - (0.28 * fw1)
        )
    )

    dH_da = (
        (1 / (M2 ** 2))
        * dM2_da
        * (
            (0.0069 / M2)
            - (0.28 * fw1)
        )
        +
        (1 - (1 / M2))
        * (
            (-0.0069 / (M2 ** 2))
            * dM2_da
            -
            (0.28 / t)
        )
    )

    dB_da = term1 + term2 + dH_da

    S = (1 - np.sqrt(np.sin(phi)))

    dG_da = (
        (fw1 / t)
        * (S / (M2 + 1))
        -
        fw1
        * S
        * (dM2_da / ((M2 + 1) ** 2))
        +
        (1 / t)
        * (S / (M2 + 1))
    )

    return base * (
        dA_da * B * G
        +
        A * dB_da * G
        +
        A * B * dG_da
    )


# helper function
def d2K_da2(l, a, c, t, b, phi, sigma):

    a = np.clip(a, 0.001, 0.08)

    fw1 = a / t
    fw2 = c / b
    M2  = a / c

    M2 = np.clip(M2, 0.2, 2.0)

    base = sigma * np.sqrt((np.pi * l) / Q(a, c))
    A = f_w(fw1, fw2)
    B = M(fw1, M2)
    G = g(fw1, M2, phi)
    u = fw1 + np.cos(fw1)

    du_da = (1 / t) - (np.sin(fw1) / t)

    d2u_da2 = -(np.cos(fw1) / (t**2))
    dA_da = (
        0.25
        * (u ** (-0.75))
        * du_da
        * (np.cos(fw2) ** (-0.5))
    )

    d2A_da2 = (

        0.25
        * (np.cos(fw2) ** (-0.5))
        * (

            (-0.75)
            * (u ** (-1.75))
            * (du_da ** 2)

            +

            (u ** (-0.75))
            * d2u_da2
        )
    )
    
    dM2_da = 1 / c

    H = (
        (1 - (1 / M2))
        * (
            (0.0069 / M2)
            - (0.28 * fw1)
        )
    )

    dH_da = (
        (1 / (M2 ** 2))
        * dM2_da
        * (
            (0.0069 / M2)
            - (0.28 * fw1)
        )
        +
        (1 - (1 / M2))
        * (
            (-0.0069 / (M2 ** 2))
            * dM2_da
            -
            (0.28 / t)
        )
    )

    dB_da = (
        0.12 * fw1 * (M2 - 1) / t
        +
        0.06 * (fw1 ** 2) * dM2_da
        +
        dH_da
    )

    d2B_da2 = (
        0.12 * (M2 - 1) / (t**2)
        +
        0.24 * fw1 * dM2_da / t
    )

    S = (1 - np.sqrt(np.sin(phi)))

    dG_da = (
        (fw1 / t)
        * (S / (M2 + 1))

        -

        fw1
        * S
        * (dM2_da / ((M2 + 1) ** 2))

        +

        (1 / t)
        * (S / (M2 + 1))
    )


    d2G_da2 = (
        2
        * S
        / (
            t
            * (M2 + 1)
        )
    )


    return base * (

        d2A_da2 * B * G

        +

        2 * dA_da * dB_da * G

        +

        2 * dA_da * B * dG_da

        +

        A * d2B_da2 * G

        +

        2 * A * dB_da * dG_da

        +

        A * B * d2G_da2
    )

#helper function
def d2K_dac(l, a, c, t, b, phi, sigma):

    fw1 = a / t
    fw2 = c / b
    M2  = np.clip(a / c, 0.2, 2.0)

    sqrt_term = np.sqrt((np.pi * l) / Q(a, c))

    F = f_w(fw1, fw2)
    Mv = M(fw1, M2)
    G = g(fw1, M2, phi)

    Fa = dfw_da(l, a, c, t, b, phi, sigma)
    Ma = dM_da(l, a, c, t, b, phi, sigma)
    Ga = dg_da(l, a, c, t, b, phi, sigma)

    Fc = dfw_dc(l, a, c, t, b, phi, sigma)
    Mc = dM_dc(l, a, c, t, b, phi, sigma)
    Gc = dg_dc(l, a, c, t, b, phi, sigma) 


    Fac = d2fw_dac(l, a, c, t, b, phi, sigma) 
    Mac = d2M_dac(l, a, c, t, b, phi, sigma) 
    Gac = d2g_dac(l, a, c, t, b, phi, sigma)

    return sigma * sqrt_term * (

        Fac * Mv * G
        +
        Fa * Mc * G
        +
        Fa * Mv * Gc

        +
        Fc * Ma * G
        +
        F * Mac * G
        +
        F * Ma * Gc

        +
        Fc * Mv * Ga
        +
        F * Mc * Ga
        +
        F * Mv * Gac
    )

# helper function
def d3K_da3(l, a, c, t, b, phi, sigma):

    fw1 = a / t
    fw2 = c / b
    M2 = a / c

    M2 = np.clip(M2, 0.2, 2.0)

    sqrt_term = np.sqrt((np.pi * l) / Q(a, c))

    fw = f_w(fw1, fw2)
    Mv = M(fw1, M2)
    gv = g(fw1, M2, phi)


    A = fw1 + np.cos(fw1)

    dA_da = (1 - np.sin(fw1)) / t
    d2A_da2 = -np.cos(fw1) / (t**2)
    d3A_da3 = np.sin(fw1) / (t**3)

    dfw_da = (
        0.25
        * (A ** (-3/4))
        * dA_da
        / ((np.cos(fw2))**0.5)
    )

    d2fw_da2 = (
        (
            -3/16 * (A ** (-7/4)) * (dA_da**2)
        )
        +
        (
            1/4 * (A ** (-3/4)) * d2A_da2
        )
    ) / ((np.cos(fw2))**0.5)

    d3fw_da3 = (
        (
            21/64 * (A ** (-11/4)) * (dA_da**3)
        )
        -
        (
            9/16 * (A ** (-7/4)) * dA_da * d2A_da2
        )
        +
        (
            1/4 * (A ** (-3/4)) * d3A_da3
        )
    ) / ((np.cos(fw2))**0.5)

  
    dM2_da = 1 / c
    d2M2_da2 = 0.0
    d3M2_da3 = 0.0

    dMv_da = (
        0.12 * fw1 * (M2 - 1) / t
        +
        0.06 * (fw1**2) * dM2_da
        +
        (
            (1 / (M2**2)) * dM2_da
        )
        *
        (
            (0.0069 / M2)
            -
            0.28 * fw1
        )
        +
        (
            1 - 1/M2
        )
        *
        (
            -0.0069 / (M2**2) * dM2_da
            -
            0.28 / t
        )
    )

    d2Mv_da2 = (

        0.12 * (M2 - 1) / (t**2)

        +

        0.24 * fw1 / (t * c)

        +

        (
            -2 / (M2**3)
        )
        *
        (
            1 / c
        )**2
        *
        (
            (0.0069 / M2)
            -
            0.28 * fw1
        )

        +

        2
        *
        (
            1 / (M2**2)
        )
        *
        (
            -0.0069 / (M2**2 * c)
            -
            0.28 / t
        )
        *
        (
            1 / c
        )

        +

        (
            1 - 1/M2
        )
        *
        (
            0.0138 / (M2**3 * c**2)
        )
    )

    d3Mv_da3 = (

        0.36 / (t**2 * c)

        +

        6
        *
        (
            1 / (M2**4)
        )
        *
        (
            1 / c
        )**3
        *
        (
            (0.0069 / M2)
            -
            0.28 * fw1
        )

        +

        3
        *
        (
            -2 / (M2**3)
        )
        *
        (
            1 / c
        )**2
        *
        (
            -0.0069 / (M2**2 * c)
            -
            0.28 / t
        )

        +

        3
        *
        (
            1 / (M2**2)
        )
        *
        (
            0.0138 / (M2**3 * c**2)
        )
        *
        (
            1 / c
        )

        +

        (
            1 - 1/M2
        )
        *
        (
            -0.0414 / (M2**4 * c**3)
        )
    )
    S = (1 - np.sqrt(np.sin(phi)))

    dgv_da = (
        S
        *
        (
            (1 / t) * (1 / (M2 + 1))
            -
            fw1 * dM2_da / ((M2 + 1)**2)
        )
    )

    d2gv_da2 = (
        S
        *
        (
            -2 * (1/t) * dM2_da / ((M2 + 1)**2)
            +
            2 * fw1 * (dM2_da**2) / ((M2 + 1)**3)
        )
    )

    d3gv_da3 = (
        S
        *
        (
            6 * (1/t) * (dM2_da**2) / ((M2 + 1)**3)
            -
            6 * fw1 * (dM2_da**3) / ((M2 + 1)**4)
        )
    )

    term1 = d3fw_da3 * Mv * gv

    term2 = 3 * d2fw_da2 * dMv_da * gv

    term3 = 3 * d2fw_da2 * Mv * dgv_da

    term4 = 3 * dfw_da * d2Mv_da2 * gv

    term5 = 6 * dfw_da * dMv_da * dgv_da

    term6 = 3 * dfw_da * Mv * d2gv_da2

    term7 = fw * d3Mv_da3 * gv

    term8 = 3 * fw * d2Mv_da2 * dgv_da

    term9 = 3 * fw * dMv_da * d2gv_da2

    term10 = fw * Mv * d3gv_da3

    return sigma * sqrt_term * (
        term1
        + term2
        + term3
        + term4
        + term5
        + term6
        + term7
        + term8
        + term9
        + term10
    )



# helper function
def d3K_dc3(l, a, c, t, b, phi, sigma):

    fw1 = a / t
    fw2 = c / b
    M2 = a / c

    M2 = np.clip(M2, 0.2, 2.0)

    sqrt_term = np.sqrt((np.pi * l) / Q(a, c))

    fw = f_w(fw1, fw2)
    Mv = M(fw1, M2)
    gv = g(fw1, M2, phi)

    A = fw1 + np.cos(fw1)

    B = np.cos(fw2)

    dB_dc = -np.sin(fw2) / b

    d2B_dc2 = -np.cos(fw2) / (b**2)

    d3B_dc3 = np.sin(fw2) / (b**3)

    A14 = A**0.25

    dfw_dc = (
        A14
        *
        (-0.5)
        *
        B**(-1.5)
        *
        dB_dc
    )

    d2fw_dc2 = (
        A14
        *
        (
            (3/4) * B**(-2.5) * (dB_dc**2)
            -
            0.5 * B**(-1.5) * d2B_dc2
        )
    )

    d3fw_dc3 = (
        A14
        *
        (
            (-15/8) * B**(-3.5) * (dB_dc**3)
            +
            (9/4) * B**(-2.5) * dB_dc * d2B_dc2
            -
            0.5 * B**(-1.5) * d3B_dc3
        )
    )

    dM2_dc = -a / (c**2)

    d2M2_dc2 = 2 * a / (c**3)

    d3M2_dc3 = -6 * a / (c**4)

    T = (
        (0.0069 / M2)
        -
        0.28 * fw1
    )

    dMv_dc = (

        0.06 * (fw1**2) * dM2_dc

        +

        (
            (1 / M2**2) * dM2_dc
        )
        *
        T

        +

        (
            1 - 1/M2
        )
        *
        (
            -0.0069 / (M2**2) * dM2_dc
        )
    )

    d2Mv_dc2 = (

        0.06 * (fw1**2) * d2M2_dc2

        +

        (
            -2 / M2**3
        )
        *
        (dM2_dc**2)
        *
        T

        +

        2
        *
        (
            1 / M2**2
        )
        *
        dM2_dc
        *
        (
            -0.0069 / (M2**2) * dM2_dc
        )

        +

        (
            1 / M2**2
        )
        *
        d2M2_dc2
        *
        T

        +

        (
            1 - 1/M2
        )
        *
        (
            0.0138 / (M2**3) * (dM2_dc**2)
            -
            0.0069 / (M2**2) * d2M2_dc2
        )
    )

    d3Mv_dc3 = (

        0.06 * (fw1**2) * d3M2_dc3

        +

        6 * (1 / M2**4) * (dM2_dc**3) * T

        +

        3 * (-2 / M2**3) * dM2_dc * d2M2_dc2 * T

        +

        3 * (-2 / M2**3) * (dM2_dc**2)
        *
        (
            -0.0069 / (M2**2) * dM2_dc
        )

        +

        3 * (1 / M2**2) * d2M2_dc2
        *
        (
            -0.0069 / (M2**2) * dM2_dc
        )

        +

        (1 / M2**2) * d3M2_dc3 * T

        +

        (
            1 - 1/M2
        )
        *
        (
            -0.0414 / (M2**4) * (dM2_dc**3)

            +

            0.0414 / (M2**3) * dM2_dc * d2M2_dc2

            -

            0.0069 / (M2**2) * d3M2_dc3
        )
    )

    S = (1 - np.sqrt(np.sin(phi)))

    dgv_dc = (
        fw1
        *
        S
        *
        (
            -1 / (M2 + 1)**2
        )
        *
        dM2_dc
    )

    d2gv_dc2 = (
        fw1
        *
        S
        *
        (
            2 * (dM2_dc**2) / (M2 + 1)**3
            -
            d2M2_dc2 / (M2 + 1)**2
        )
    )

    d3gv_dc3 = (
        fw1
        *
        S
        *
        (
            -6 * (dM2_dc**3) / (M2 + 1)**4

            +

            6 * dM2_dc * d2M2_dc2 / (M2 + 1)**3

            -

            d3M2_dc3 / (M2 + 1)**2
        )
    )

    term1 = d3fw_dc3 * Mv * gv

    term2 = 3 * d2fw_dc2 * dMv_dc * gv

    term3 = 3 * d2fw_dc2 * Mv * dgv_dc

    term4 = 3 * dfw_dc * d2Mv_dc2 * gv

    term5 = 6 * dfw_dc * dMv_dc * dgv_dc

    term6 = 3 * dfw_dc * Mv * d2gv_dc2

    term7 = fw * d3Mv_dc3 * gv

    term8 = 3 * fw * d2Mv_dc2 * dgv_dc

    term9 = 3 * fw * dMv_dc * d2gv_dc2

    term10 = fw * Mv * d3gv_dc3

    return sigma * sqrt_term * (
        term1
        + term2
        + term3
        + term4
        + term5
        + term6
        + term7
        + term8
        + term9
        + term10
    )
    
# helper function
def d3K_da2dc(l, a, c, t, b, phi, sigma):

    fw1 = a / t
    fw2 = c / b
    M2  = np.clip(a / c, 0.2, 2.0)

    sqrt_term = np.sqrt((np.pi * l) / Q(a, c))

    F = f_w(fw1, fw2)
    Mv = M(fw1, M2)
    G = g(fw1, M2, phi)

    Fa = dfw_da(l, a, c, t, b, phi, sigma)
    Ma = dM_da(l, a, c, t, b, phi, sigma)
    Ga = dg_da(l, a, c, t, b, phi, sigma)

    Fc = dfw_dc(l, a, c, t, b, phi, sigma)
    Mc = dM_dc(l, a, c, t, b, phi, sigma)
    Gc = dg_dc(l, a, c, t, b, phi, sigma)

    Faa = d2fw_da2(l, a, c, t, b, phi, sigma)
    Maa = d2M_da2(l, a, c, t, b, phi, sigma)
    Gaa = d2g_da2(l, a, c, t, b, phi, sigma)

    Fac = d2fw_dac(l, a, c, t, b, phi, sigma)
    Mac = d2M_dac(l, a, c, t, b, phi, sigma)
    Gac = d2g_dac(l, a, c, t, b, phi, sigma)

    Faac = d3fw_da2dc(l, a, c, t, b, phi, sigma) 
    Maac = d3M_da2dc(l, a, c, t, b, phi, sigma) 
    Gaac = d3g_da2dc(l, a, c, t, b, phi, sigma) 

    result = (

        Faac * Mv * G
        +
        2 * Fac * Ma * G
        +
        2 * Fac * Mv * Ga

        +
        Faa * Mc * G
        +
        Faa * Mv * Gc
        +
        2 * Fa * Mac * G

        +
        4 * Fa * Ma * Gc
        +
        4 * Fa * Mc * Ga
        +
        2 * Fa * Mv * Gac

        +
        Fc * Maa * G
        +
        2 * Fc * Ma * Ga
        +
        Fc * Mv * Gaa

        +
        F * Maac * G
        +
        2 * F * Mac * Ga
        +
        F * Maa * Gc

        +
        2 * F * Ma * Gac
        +
        F * Mc * Gaa
        +
        F * Mv * Gaac
    )

    return sigma * sqrt_term * result
    
    
def d3K_dadc2(l, a, c, t, b, phi, sigma):
    fw1 = a / t
    fw2 = c / b
    M2  = np.clip(a / c, 0.2, 2.0)

    sqrt_term = np.sqrt((np.pi * l) / Q(a, c))

    F0 = f_w(fw1, fw2)
    M0 = M(fw1, M2)
    G0 = g(fw1, M2, phi)

    Fa = dfw_da(l, a, c, t, b, phi, sigma)
    Fc = dfw_dc(l, a, c, t, b, phi, sigma)

    Ma = dM_da(l, a, c, t, b, phi, sigma)
    Mc = dM_dc(l, a, c, t, b, phi, sigma)

    Ga = dg_da(l, a, c, t, b, phi, sigma)
    Gc = dg_dc(l, a, c, t, b, phi, sigma)

    Fcc = d2fw_dc2(l, a, c, t, b, phi, sigma)
    Fac = d2fw_dac(l, a, c, t, b, phi, sigma)

    Mcc = d2M_dc2(l, a, c, t, b, phi, sigma)
    Mac = d2M_dac(l, a, c, t, b, phi, sigma)

    Gcc = d2g_dc2(l, a, c, t, b, phi, sigma) 
    Gac = d2g_dac(l, a, c, t, b, phi, sigma)


    Facc = d3fw_dadc2(l, a, c, t, b, phi, sigma) 
    Macc = d3M_dadc2(l, a, c, t, b, phi, sigma)
    Gacc = d3g_dadc2(l, a, c, t, b, phi, sigma) 


    K3 = (

        Facc * M0 * G0
        + F0 * Macc * G0
        + F0 * M0 * Gacc

        + 2 * Fac * Mc * G0
        + 2 * Fac * M0 * Gc

        + 2 * Fc * Mac * G0
        + 2 * F0 * Mac * Gc

        + 2 * Fc * M0 * Gac
        + 2 * F0 * Mc * Gac

        + Fa * Mcc * G0
        + Fa * M0 * Gcc

        + Fcc * Ma * G0
        + F0 * Ma * Gcc

        + Fcc * M0 * Ga
        + F0 * Mcc * Ga

        + 2 * Fa * Mc * Gc
        + 2 * Fc * Ma * Gc
        + 2 * Fc * Mc * Ga
    )

    return sigma * sqrt_term * K3    
    
def d4K_da4(l, a, c, t, b, phi, sigma):
    fw1 = a / t
    fw2 = c / b
    M2  = np.clip(a / c, 0.2, 2.0)
    sqrt_term = np.sqrt((np.pi * l) / Q(a, c))
    fw = f_w(fw1, fw2)
    Mv = M(fw1, M2)
    gv = g(fw1, M2, phi)


    fw0 = fw
    fw1_d = dfw_da(l, a, c, t, b, phi, sigma)          
    fw2_d = d2fw_da2(l, a, c, t, b, phi, sigma)
    fw3_d = d3fw_da3(l, a, c, t, b, phi, sigma)
    fw4_d = d4fw_da4(l, a, c, t, b, phi, sigma)       


    M0 = Mv
    M1 = dMv_da(l, a, c, t, b, phi, sigma) 
    M2_d = d2Mv_da2(l, a, c, t, b, phi, sigma) 
    M3_d = d3Mv_da3(l, a, c, t, b, phi, sigma) 
    M4_d = d4Mv_da4(l, a, c, t, b, phi, sigma) 

 
    g0 = gv
    g1 = dgv_da(l, a, c, t, b, phi, sigma) 
    g2 = d2gv_da2(l, a, c, t, b, phi, sigma) 
    g3 = d3gv_da3(l, a, c, t, b, phi, sigma) 
    g4 = d4gv_da4(l, a, c, t, b, phi, sigma)


    K4 = (
        fw4_d * M0 * g0
        +
        4 * fw3_d * M1 * g0
        +
        4 * fw3_d * M0 * g1

        +
        6 * fw2_d * M2_d * g0
        +
        12 * fw2_d * M1 * g1
        +
        6 * fw2_d * M0 * g2

        +
        4 * fw1_d * M3_d * g0
        +
        12 * fw1_d * M2_d * g1
        +
        12 * fw1_d * M1 * g2
        +
        4 * fw1_d * M0 * g3

        +
        fw0 * M4_d * g0
        +
        4 * fw0 * M3_d * g1
        +
        6 * fw0 * M2_d * g2
        +
        4 * fw0 * M1 * g3
        +
        fw0 * M0 * g4
    )

    return sigma * sqrt_term * K4


def d4K_dc4(l, a, c, t, b, phi, sigma):
    sqrt_term = np.sqrt((np.pi * l) / Q(a, c))
    A = sigma * sqrt_term

    fw1 = a / t
    fw2 = c / b
    M2  = np.clip(a / c, 0.2, 2.0)

    F0 = f_w(fw1, fw2)
    M0 = M(fw1, M2)
    G0 = g(fw1, M2, phi)

    # f_w derivatives wrt c
    F1 = dfw_dc(l, a, c, t, b, phi, sigma) 
    F2 = d2fw_dc2(l, a, c, t, b, phi, sigma) 
    F3 = d3fw_dc3(l, a, c, t, b, phi, sigma) 
    F4 = d4fw_dc4(l, a, c, t, b, phi, sigma) 

    # M derivatives wrt c
    M1 = dM_dc(l, a, c, t, b, phi, sigma) 
    M2_d = d2M_dc2(l, a, c, t, b, phi, sigma) 
    M3 = d3M_dc3(l, a, c, t, b, phi, sigma) 
    M4 = d4M_dc4(l, a, c, t, b, phi, sigma) 

    # g derivatives wrt c
    G1 = dg_dc(l, a, c, t, b, phi, sigma) 
    G2 = d2g_dc2(l, a, c, t, b, phi, sigma) 
    G3 = d3g_dc3(l, a, c, t, b, phi, sigma) 
    G4 = d4g_dc4(l, a, c, t, b, phi, sigma)

    K4 = (
        F4 * M0 * G0
        + 4 * F3 * M1 * G0
        + 4 * F3 * M0 * G1

        + 6 * F2 * M2_d * G0
        + 12 * F2 * M1 * G1
        + 6 * F2 * M0 * G2

        + 4 * F1 * M3 * G0
        + 12 * F1 * M2_d * G1
        + 12 * F1 * M1 * G2
        + 4 * F1 * M0 * G3

        + F0 * M4 * G0
        + 4 * F0 * M3 * G1
        + 6 * F0 * M2_d * G2
        + 4 * F0 * M1 * G3
        + F0 * M0 * G4
    )

    return A * K4


def d4K_da2dc2(l, a, c, t, b, phi, sigma):
    fw1 = a / t
    fw2 = c / b
    M2  = np.clip(a / c, 0.2, 2.0)

    prefactor = sigma * np.sqrt((np.pi * l) / Q(a, c))

    F = f_w(fw1, fw2)
    Mv = M(fw1, M2)
    G = g(fw1, M2, phi)


    Fa = dfw_da(l, a, c, t, b, phi, sigma)
    Fc = dfw_dc(l, a, c, t, b, phi, sigma)

    Ma = dM_da(l, a, c, t, b, phi, sigma)
    Mc = dM_dc(l, a, c, t, b, phi, sigma)

    Ga = dg_da(l, a, c, t, b, phi, sigma)
    Gc = dg_dc(l, a, c, t, b, phi, sigma)


    Faa = d2fw_da2(l, a, c, t, b, phi, sigma)
    Fcc = d2fw_dc2(l, a, c, t, b, phi, sigma)
    Fac = d2fw_dac(l, a, c, t, b, phi, sigma)

    Maa = d2M_da2(l, a, c, t, b, phi, sigma)
    Mcc = d2M_dc2(l, a, c, t, b, phi, sigma) 
    Mac = d2M_dac(l, a, c, t, b, phi, sigma)

    Gaa = d2g_da2(l, a, c, t, b, phi, sigma)
    Gcc = d2g_dc2(l, a, c, t, b, phi, sigma)
    Gac = d2g_dac(l, a, c, t, b, phi, sigma)


    Faac = d3fw_da2dc(l, a, c, t, b, phi, sigma)
    Facc = d3fw_dadc2(l, a, c, t, b, phi, sigma)

    Maac = d3M_da2dc(l, a, c, t, b, phi, sigma)
    Macc = d3M_dadc2(l, a, c, t, b, phi, sigma)

    Gaac = d3g_da2dc(l, a, c, t, b, phi, sigma)
    Gacc = d3g_dadc2(l, a, c, t, b, phi, sigma)


#     term_F = (
#         F * Mcc * Gaa
#         + Faa * Mcc * G
#         + Faa * M * Gcc
#         + Fcc * Maa * G
#         + 2 * Fac * Mac * G
#         + 2 * Fac * M * Gac
#         + 2 * F * Mac * Gac
#         + F * Maa * Gcc
#         + Fcc * M * Gaa
#     )

    
    term_F = (
        F * Mcc * Gaa
        + Faa * Mcc * G
        + Faa * Mv * Gcc
        + Fcc * Maa * G
        + 2 * Fac * Mac * G
        + 2 * Fac * Mv * Gac
        + 2 * F * Mac * Gac
        + F * Maa * Gcc
        + Fcc * Mv * Gaa
    )
    
    term_cross = (
        Faac * Mc * G
        + Facc * Ma * G
        + Fa * Macc * G
        + Fa * Mc * Gcc
        + Fc * Maac * G
        + Fc * Maa * Gc
        + F * Maac * Gc
        + F * Mc * Gacc
    )

    term_high = (
        Faa * Mc * Gcc
        + Fcc * Ma * Gaa
        + 2 * Fac * Mc * Gac
    )

    return prefactor * (term_F + term_cross + term_high)


def d3M_dc3(l, a, c, t, b, phi, sigma):

    x = a / t
    y = np.clip(a / c, 0.2, 2.0)

    dy_dc   = -a / (c**2)
    d2y_dc2 =  2*a / (c**3)
    d3y_dc3 = -6*a / (c**4)

    A1 = 0.06 * (x**2)

    d3A = A1 * d3y_dc3

    A = (1 - 1/y)
    B = (0.0069/y - 0.28*x)

    dA_dy   = 1 / (y**2)
    d2A_dy2 = -2 / (y**3)
    d3A_dy3 = 6 / (y**4)

    dB_dy   = -0.0069 / (y**2)
    d2B_dy2 = 2 * 0.0069 / (y**3)
    d3B_dy3 = -6 * 0.0069 / (y**4)

    termA = d3A_dy3 * (dy_dc**3)
    termB = 3 * d2A_dy2 * dy_dc * d2y_dc2
    termC = dA_dy * d3y_dc3

    termD = d3B_dy3 * (dy_dc**3) * A
    termE = 3 * d2B_dy2 * dy_dc * d2y_dc2 * A
    termF = dB_dy * d3y_dc3 * A

    cross1 = 3 * d2A_dy2 * (dy_dc**2) * dB_dy
    cross2 = 3 * dA_dy * dy_dc * d2B_dy2 * dy_dc
    cross3 = dA_dy * dB_dy * d3y_dc3

    return d3A + (termA + termB + termC + termD + termE + termF + cross1 + cross2 + cross3)


def d4M_dc4(l, a, c, t, b, phi, sigma):

    x = a / t
    y = np.clip(a / c, 0.2, 2.0)

    dy  = -a / (c**2)
    d2y =  2*a / (c**3)
    d3y = -6*a / (c**4)
    d4y = 24*a / (c**5)

    A1 = 0.06 * (x**2)

    d4A = A1 * d4y

    A = (1 - 1/y)
    B = (0.0069/y - 0.28*x)

    dA1 = 1/y**2
    dA2 = -2/y**3
    dA3 =  6/y**4
    dA4 = -24/y**5

    dB1 = -0.0069/y**2
    dB2 =  2*0.0069/y**3
    dB3 = -6*0.0069/y**4
    dB4 =  24*0.0069/y**5

    termA = dA4 * dy**4 + 6*dA3*dy**2*d2y + 3*dA2*d2y**2 + 4*dA2*dy*d3y + dA1*d4y
    termB = dB4 * dy**4 + 6*dB3*dy**2*d2y + 3*dB2*d2y**2 + 4*dB2*dy*d3y + dB1*d4y

    cross = (
        4*dA3*dy**3*dB1 +
        6*dA2*dy**2*dB2 +
        4*dA1*dy*dB3 +
        dA1*dB1*d4y
    )

    return d4A + termA + termB + cross


def d3g_dc3(l, a, c, t, b, phi, sigma):

    s = np.sqrt(np.sin(phi))
    A = a * (1 - s)

    y = np.clip(a / c, 0.2, 2.0)

    dy  = -a / (c**2)
    d2y =  2*a / (c**3)
    d3y = -6*a / (c**4)

    h1 = -1 / (y + 1)**2
    h2 =  2 / (y + 1)**3
    h3 = -6 / (y + 1)**4

    term1 = A * h3 * dy**3
    term2 = 3 * A * h2 * dy * d2y
    term3 = A * h1 * d3y

    return term1 + term2 + term3


def d4g_dc4(l, a, c, t, b, phi, sigma):

    s = np.sqrt(np.sin(phi))
    A = a * (1 - s)

    y = np.clip(a / c, 0.2, 2.0)

    dy  = -a / (c**2)
    d2y =  2*a / (c**3)
    d3y = -6*a / (c**4)
    d4y = 24*a / (c**5)

    h1 = -1 / (y + 1)**2
    h2 =  2 / (y + 1)**3
    h3 = -6 / (y + 1)**4
    h4 =  24 / (y + 1)**5

    term1 = A * h4 * dy**4
    term2 = 6 * A * h3 * dy**2 * d2y
    term3 = 3 * A * h2 * d2y**2
    term4 = 4 * A * h2 * dy * d3y
    term5 = A * h1 * d4y

    return term1 + term2 + term3 + term4 + term5


# helper functions
def dF_da(l, a, c, t, b, phi, sigma):

    fw = f_w(a / t, c / b)
    Mv = M(a / t, a / c)
    gv = g(a / t, a / c, phi)

    fw_a = dfw_da(l, a, c, t, b, phi, sigma)
    M_a  = dM_da(l, a, c, t, b, phi, sigma)
    g_a  = dg_da(l, a, c, t, b, phi, sigma)

    return fw_a * Mv * gv + fw * M_a * gv + fw * Mv * g_a

def dF_dc(l, a, c, t, b, phi, sigma):

    fw = f_w(a / t, c / b)
    Mv = M(a / t, a / c)
    gv = g(a / t, a / c, phi)

    fw_c = dfw_dc(l, a, c, t, b, phi, sigma)
    M_c  = dM_dc(l, a, c, t, b, phi, sigma)
    g_c  = dg_dc(l, a, c, t, b, phi, sigma)

    return fw_c * Mv * gv + fw * M_c * gv + fw * Mv * g_c


def d2F_dac(l, a, c, t, b, phi, sigma):

    fw = f_w(a / t, c / b)
    Mv = M(a / t, a / c)
    gv = g(a / t, a / c, phi)

    fw_a = dfw_da(l, a, c, t, b, phi, sigma)
    fw_c = dfw_dc(l, a, c, t, b, phi, sigma)

    M_a = dM_da(l, a, c, t, b, phi, sigma)
    M_c = dM_dc(l, a, c, t, b, phi, sigma)

    g_a = dg_da(l, a, c, t, b, phi, sigma)
    g_c = dg_dc(l, a, c, t, b, phi, sigma)

    fw_ac = d2fw_dac(l, a, c, t, b, phi, sigma)
    M_ac  = d2M_dac(l, a, c, t, b, phi, sigma)
    g_ac  = d2g_dac(l, a, c, t, b, phi, sigma)

    term = (
        fw_ac * Mv * gv
        + fw_a * M_c * gv
        + fw_a * Mv * g_c
        + fw_c * M_a * gv
        + fw * M_ac * gv
        + fw * M_a * g_c
        + fw_c * Mv * g_a
        + fw * M_c * g_a
        + fw * Mv * g_ac
    )

    return term

def d2F_dc2(l, a, c, t, b, phi, sigma):

    fw = f_w(a / t, c / b)
    Mv = M(a / t, a / c)
    gv = g(a / t, a / c, phi)

    fw_c = dfw_dc(l, a, c, t, b, phi, sigma)
    M_c  = dM_dc(l, a, c, t, b, phi, sigma)
    g_c  = dg_dc(l, a, c, t, b, phi, sigma)

    fw_cc = d2fw_dc2(l, a, c, t, b, phi, sigma)
    M_cc  = d2M_dc2(l, a, c, t, b, phi, sigma)
    g_cc  = d2g_dc2(l, a, c, t, b, phi, sigma)

    term = (
        fw_cc * Mv * gv
        + fw * M_cc * gv
        + fw * Mv * g_cc
        + 2 * fw_c * M_c * gv
        + 2 * fw_c * Mv * g_c
        + 2 * fw * M_c * g_c
    )

    return term


def d3F_dadc2(l, a, c, t, b, phi, sigma):

    fw = f_w(a / t, c / b)
    Mv = M(a / t, a / c)
    gv = g(a / t, a / c, phi)

    fw_a   = dfw_da(l, a, c, t, b, phi, sigma)
    fw_c   = dfw_dc(l, a, c, t, b, phi, sigma)
    fw_cc  = d2fw_dc2(l, a, c, t, b, phi, sigma)
    fw_acc = d3fw_dadc2(l, a, c, t, b, phi, sigma)

    M_a   = dM_da(l, a, c, t, b, phi, sigma)
    M_c   = dM_dc(l, a, c, t, b, phi, sigma)
    M_cc  = d2M_dc2(l, a, c, t, b, phi, sigma)
    M_acc = d3M_dadc2(l, a, c, t, b, phi, sigma)

    g_a   = dg_da(l, a, c, t, b, phi, sigma)
    g_c   = dg_dc(l, a, c, t, b, phi, sigma)
    g_cc  = d2g_dc2(l, a, c, t, b, phi, sigma)
    g_acc = d3g_dadc2(l, a, c, t, b, phi, sigma)

    term = (
        fw_acc * Mv * gv
        + fw_a * M_cc * gv
        + fw_a * Mv * g_cc
        + fw_cc * M_a * gv
        + fw * M_acc * gv
        + fw * M_a * g_cc
        + fw_cc * Mv * g_a
        + fw * M_cc * g_a
        + fw * Mv * g_acc
        + 2 * fw_c * M_c * g_c
        + 2 * fw_a * M_c * g_c
        + 2 * fw_c * M_a * g_c
    )

    return term


def d4F_dadc3(l, a, c, t, b, phi, sigma):

    fw = f_w(a / t, c / b)
    Mv = M(a / t, a / c)
    gv = g(a / t, a / c, phi)

    fw_a    = dfw_da(l, a, c, t, b, phi, sigma)
    fw_c    = dfw_dc(l, a, c, t, b, phi, sigma)
    fw_cc   = d2fw_dc2(l, a, c, t, b, phi, sigma)
    fw_ccc  = d3fw_dc3(l, a, c, t, b, phi, sigma)
    fw_accc = d4fw_dadc3(l, a, c, t, b, phi, sigma) # define this

    M_a    = dM_da(l, a, c, t, b, phi, sigma)
    M_c    = dM_dc(l, a, c, t, b, phi, sigma)
    M_cc   = d2M_dc2(l, a, c, t, b, phi, sigma)
    M_ccc  = d3M_dc3(l, a, c, t, b, phi, sigma)
    M_accc = d4M_dadc3(l, a, c, t, b, phi, sigma) # define this

    g_a    = dg_da(l, a, c, t, b, phi, sigma)
    g_c    = dg_dc(l, a, c, t, b, phi, sigma)
    g_cc   = d2g_dc2(l, a, c, t, b, phi, sigma)
    g_ccc  = d3g_dc3(l, a, c, t, b, phi, sigma)
    g_accc = d4g_dadc3(l, a, c, t, b, phi, sigma) # define this

    term = (
        fw_a * M_ccc * gv
        + fw_a * Mv * g_ccc
        + fw_ccc * M_a * gv
        + fw * M_accc * gv
        + fw * M_a * g_ccc
        + fw_ccc * Mv * g_a
        + fw * M_ccc * g_a
        + fw * Mv * g_accc

        + 3 * fw_cc * M_c * g_cc
        + 3 * fw_c * M_cc * g_c
        + 3 * fw_cc * M_cc * g_c
    )

    return term

# all derivatives of the paris law wrt a and c

def df_da(l, a, c, t, b, phi, sigma, C, m):
    a = np.clip(a, 0.001, 0.08)
    
    K = stress_intensity_factor_K(l, a, c, t, b, phi, sigma)
    dc_dn = paris_law_dc_dn(l, a, c, t, b, phi, sigma, C, m)
    
    
    f_w_input_1 = a / t
    f_w_input_2 = c / b
    M_input_2 = np.clip(a / c, 0.2, 2)

    A = sigma * np.sqrt((np.pi * l) / Q(a, c))

    fw = f_w(f_w_input_1, f_w_input_2)
    Mv = M(f_w_input_1, M_input_2)
    gv = g(f_w_input_1, M_input_2, phi)


    # df_w/da
    x = f_w_input_1
    denom = (np.cos(f_w_input_2))**0.5
    dfw_da = (1/4) * (x + np.cos(x))**(-3/4) * (1 - np.sin(x)) * (1/t) / denom

    # dM/da
    dM_da = (
        0.12 * a * (M_input_2 - 1)
        + (1 - 1/M_input_2) * (-0.28 / c)
    )

    # dg/da
    dg_da = (1/t) * (1 - np.sqrt(np.sin(phi))) / (M_input_2 + 1)


    dK_da = A * (
        dfw_da * Mv * gv
        + fw * dM_da * gv
        + fw * Mv * dg_da
    )


    return C * m * (A * fw * Mv * gv)**(m - 1) * dK_da
    

def df_dc(l, a, c, t, b, phi, sigma, C, m):
    a = np.clip(a, 0.001, 0.08)
    K = stress_intensity_factor_K(l, a, c, t, b, phi, sigma)

    f_w_input_1 = a / t
    f_w_input_2 = c / b
    M_input_2 = np.clip(a / c, 0.2, 2)

    A = sigma * np.sqrt((np.pi * l) / Q(a, c))

    fw = f_w(f_w_input_1, f_w_input_2)
    Mv = M(f_w_input_1, M_input_2)
    gv = g(f_w_input_1, M_input_2, phi)

    dfw_dc = fw * (1/2) * np.tan(c / b) * (1 / b)
    dM2_dc = -a / (c**2)

    dM_dc = (
        0.06 * a**2 * dM2_dc
        + (1 - 1/M_input_2) * (0.0069 * (-1 / M_input_2**2) * dM2_dc)
        + (0.28 * a / (c**2)) * (1 / M_input_2)
        - (0.28 * a / c) * (1 / M_input_2**2) * dM2_dc
    )

    dg_dc = (
        f_w_input_1
        * (-(1 - np.sqrt(np.sin(phi))) / (M_input_2 + 1)**2)
        * dM2_dc
    )

    dK_dc = A * (
        dfw_dc * Mv * gv
        + fw * dM_dc * gv
        + fw * Mv * dg_dc
    )

    return C * m * (K**(m - 1)) * dK_dc


def d2f_da2(l, a, c, t, b, phi, sigma, C, m):
    a = np.clip(a, 0.001, 0.08)
    K = stress_intensity_factor_K(l, a, c, t, b, phi, sigma)

    f_w_input_1 = a / t
    f_w_input_2 = c / b
    M_input_2 = np.clip(a / c, 0.2, 2)

    A = sigma * np.sqrt((np.pi * l) / Q(a, c))

    fw = f_w(f_w_input_1, f_w_input_2)
    Mv = M(f_w_input_1, M_input_2)
    gv = g(f_w_input_1, M_input_2, phi)

    # dfw/da
    x = f_w_input_1
    denom = (np.cos(f_w_input_2))**0.5
    dfw_da = (1/4) * (x + np.cos(x))**(-3/4) * (1 - np.sin(x)) * (1/t) / denom

    # dM/da
    dM_da = (
        0.12 * a * (M_input_2 - 1)
        + (1 - 1/M_input_2) * (-0.28 / c)
    )

    # dg/da
    dg_da = (1/t) * (1 - np.sqrt(np.sin(phi))) / (M_input_2 + 1)


    # d2fw/da2 
    d2fw_da2 = (
        (1/4) * (-3/4) * (x + np.cos(x))**(-7/4) * (1 - np.sin(x))**2 * (1/t)**2
        + (1/4) * (x + np.cos(x))**(-3/4) * (-np.cos(x)) * (1/t)**2
    ) / denom

    # d2M/da2
    d2M_da2 = 0.12 * (M_input_2 - 1)

    # d2g/da2 = 0 (linear in a)
    d2g_da2 = 0.0

    K_aa = A * (
        d2fw_da2 * Mv * gv
        + 2 * dfw_da * dM_da * gv
        + 2 * dfw_da * Mv * dg_da
        + fw * d2M_da2 * gv
        + 2 * fw * dM_da * dg_da
        + fw * Mv * d2g_da2
    )

    K_a = (
        sigma * np.sqrt((np.pi * l) / Q(a, c))
        * (
            dfw_da * Mv * gv
            + fw * dM_da * gv
            + fw * Mv * dg_da
        )
    )

    return (
        C * m * (m - 1) * (K**(m - 2)) * (K_a**2)
        + C * m * (K**(m - 1)) * K_aa
    )
    
def d2f_dc2(l, a, c, t, b, phi, sigma, C, m):
    K = stress_intensity_factor_K(l, a, c, t, b, phi, sigma)

    dKdc = dK_dc(l, a, c, t, b, phi, sigma)
    d2Kdc2 = d2K_dc2(l, a, c, t, b, phi, sigma)

    return (
        C * m * (m - 1) * (K ** (m - 2)) * (dKdc ** 2)
        + C * m * (K ** (m - 1)) * d2Kdc2
    )


#partial derivative for second derivative, i.e., d2f_dac
def d2f_dac(l, a, c, t, b, phi, sigma, C, m):

    K = stress_intensity_factor_K(
        l, a, c, t, b, phi, sigma
    )

    Ka = dK_da(
        l, a, c, t, b, phi, sigma
    )

    Kc = dK_dc(
        l, a, c, t, b, phi, sigma
    )

    Kac = d2K_dac(
        l, a, c, t, b, phi, sigma
    )

    return (
        C
        * (
            m * (m - 1) * (K ** (m - 2)) * Ka * Kc
            +
            m * (K ** (m - 1)) * Kac
        )
    )

def d3f_da3(l, a, c, t, b, phi, sigma, C, m):
    K = stress_intensity_factor_K(
        l, a, c, t, b, phi, sigma
    )

    #dk_da
    K_a = dK_da(
        l, a, c, t, b, phi, sigma
    )

    # d2k_da2
    K_aa = d2K_da2(
        l, a, c, t, b, phi, sigma
    )

    
    # d3k_da3
    K_aaa = d3K_da3(
        l, a, c, t, b, phi, sigma
    )


    term1 = (
        C
        * m
        * (m - 1)
        * (m - 2)
        * (K ** (m - 3))
        * (K_a ** 3)
    )

    term2 = (
        3
        * C
        * m
        * (m - 1)
        * (K ** (m - 2))
        * K_a
        * K_aa
    )

    term3 = (
        C
        * m
        * (K ** (m - 1))
        * K_aaa
    )

    return term1 + term2 + term3
  

def d3f_dc3(l, a, c, t, b, phi, sigma, C, m):
    K = stress_intensity_factor_K(
        l, a, c, t, b, phi, sigma
    )

    #dk_dc
    K_c = dK_dc(
        l, a, c, t, b, phi, sigma
    )

    
    # d2k_dc2
    K_cc = d2K_dc2(
        l, a, c, t, b, phi, sigma
    )

    #d3k_dc3
    K_ccc = d3K_dc3(
        l, a, c, t, b, phi, sigma
    )

    term1 = (
        C
        * m
        * (m - 1)
        * (m - 2)
        * (K ** (m - 3))
        * (K_c ** 3)
    )

    term2 = (
        3
        * C
        * m
        * (m - 1)
        * (K ** (m - 2))
        * K_c
        * K_cc
    )

    term3 = (
        C
        * m
        * (K ** (m - 1))
        * K_ccc
    )

    return term1 + term2 + term3


def d3f_da2dc(l, a, c, t, b, phi, sigma, C, m):
    K = stress_intensity_factor_K(
        l, a, c, t, b, phi, sigma
    )

    Ka = dK_da(
        l, a, c, t, b, phi, sigma
    )

    Kc = dK_dc(
        l, a, c, t, b, phi, sigma
    )

    Kaa = d2K_da2(
        l, a, c, t, b, phi, sigma
    )

    Kac = d2K_dac(
        l, a, c, t, b, phi, sigma
    )

    Kaac = d3K_da2dc(
        l, a, c, t, b, phi, sigma
    )

    term1 = (
        m * (m - 1) * (m - 2)
        * (K ** (m - 3))
        * (Ka ** 2)
        * Kc
    )

    term2 = (
        m * (m - 1)
        * (K ** (m - 2))
        * (
            2 * Ka * Kac
            +
            Kaa * Kc
        )
    )

    term3 = (
        m
        * (K ** (m - 1))
        * Kaac
    )

    return C * (term1 + term2 + term3)


def d3f_dadc2(l, a, c, t, b, phi, sigma, C, m):

    K = stress_intensity_factor_K(
        l, a, c, t, b, phi, sigma
    )

    Ka = dK_da(
        l, a, c, t, b, phi, sigma
    )

    Kc = dK_dc(
        l, a, c, t, b, phi, sigma
    )

    Kcc = d2K_dc2(
        l, a, c, t, b, phi, sigma
    )

    Kac = d2K_dac(
        l, a, c, t, b, phi, sigma
    )

    Kacc = d3K_dadc2(  
        l, a, c, t, b, phi, sigma
    )


    term1 = (
        m * (m - 1) * (m - 2)
        * (K ** (m - 3))
        * Ka
        * (Kc ** 2)
    )

    term2 = (
        m * (m - 1)
        * (K ** (m - 2))
        * (
            2 * Kc * Kac
            +
            Ka * Kcc
        )
    )

    term3 = (
        m
        * (K ** (m - 1))
        * Kacc
    )

    return C * (term1 + term2 + term3)

def d4f_da4(l, a, c, t, b, phi, sigma, C, m):
    K = stress_intensity_factor_K(
        l, a, c, t, b, phi, sigma
    )

    #dk_da
    dK1 = dK_da(
        l, a, c, t, b, phi, sigma
    )

    #d2k_da2
    dK2 = d2K_da2(
        l, a, c, t, b, phi, sigma
    )

    #d3k_da3
    dK3 = d3K_da3(
        l, a, c, t, b, phi, sigma
    )

    #d4k_da4
    dK4 = d4K_da4(
        l, a, c, t, b, phi, sigma
    )

    return (

        C

        *

        (
            m
            *
            (m - 1)
            *
            (m - 2)
            *
            (m - 3)
            *
            (K ** (m - 4))
            *
            (dK1 ** 4)

            +

            6
            *
            m
            *
            (m - 1)
            *
            (m - 2)
            *
            (K ** (m - 3))
            *
            (dK1 ** 2)
            *
            dK2

            +

            3
            *
            m
            *
            (m - 1)
            *
            (K ** (m - 2))
            *
            (dK2 ** 2)

            +

            4
            *
            m
            *
            (m - 1)
            *
            (K ** (m - 2))
            *
            dK1
            *
            dK3

            +

            m
            *
            (K ** (m - 1))
            *
            dK4

        )

    )
    
def d4f_dc4(l, a, c, t, b, phi, sigma, C, m):

    K = stress_intensity_factor_K(l, a, c, t, b, phi, sigma)

    K1 = dK_dc(l, a, c, t, b, phi, sigma)
    K2 = d2K_dc2(l, a, c, t, b, phi, sigma)
    K3 = d3K_dc3(l, a, c, t, b, phi, sigma)
    K4 = d4K_dc4(l, a, c, t, b, phi, sigma)

    term1 = m * (m-1) * (m-2) * (m-3) * (K**(m-4)) * (K1**4)

    term2 = 6 * m * (m-1) * (m-2) * (K**(m-3)) * (K1**2) * K2

    term3 = 3 * m * (m-1) * (K**(m-2)) * (K2**2)

    term4 = 4 * m * (m-1) * (K**(m-2)) * K1 * K3

    term5 = m * (K**(m-1)) * K4

    return C * (term1 + term2 + term3 + term4 + term5)


def d4f_da3dc(l, a, c, t, b, phi, sigma, C, m):
    K  = stress_intensity_factor_K(l, a, c, t, b, phi, sigma)

    Ka  = dK_da(l, a, c, t, b, phi, sigma)
    Kaa = d2K_da2(l, a, c, t, b, phi, sigma)
    Kaaa = d3K_da3(l, a, c, t, b, phi, sigma)

    Kc = dK_dc(l, a, c, t, b, phi, sigma)
    Kac = d2K_dac(l, a, c, t, b, phi, sigma)
    Kaac = d3K_da2dc(l, a, c, t, b, phi, sigma)

    m1 = m
    m2 = m * (m - 1)
    m3 = m * (m - 1) * (m - 2)

    f_aaa = C * (
        m1 * (K**(m - 1)) * Kaaa
        + 3 * m2 * (K**(m - 2)) * Ka * Kaa
        + m3 * (K**(m - 3)) * (Ka**3)
    )


    term1 = m1 * (
        (m - 1) * (K**(m - 2)) * Kc * Kaaa
        + (K**(m - 1)) * Kaac
    )

    term2 = 3 * m2 * (
        (m - 2) * (K**(m - 3)) * Kc * Ka * Kaa
        + (K**(m - 2)) * (Kac * Kaa + Ka * Kaac)
    )

    term3 = m3 * (
        (m - 3) * (K**(m - 4)) * Kc * (Ka**3)
        + (K**(m - 3)) * (3 * (Ka**2) * Kac)
    )

    return C * (term1 + term2 + term3)

def d4f_da2dc2(l, a, c, t, b, phi, sigma, C, m):
    
    K = stress_intensity_factor_K(l, a, c, t, b, phi, sigma)

    Ka = dK_da(l, a, c, t, b, phi, sigma)
    Kc = dK_dc(l, a, c, t, b, phi, sigma)

    Kaa = d2K_da2(l, a, c, t, b, phi, sigma)
    Kcc = d2K_dc2(l, a, c, t, b, phi, sigma)
    Kac = d2K_dac(l, a, c, t, b, phi, sigma)

    Kaaa = d3K_da3(l, a, c, t, b, phi, sigma)
    Kacc = d3K_dadc2(l, a, c, t, b, phi, sigma)

    Kcca = Kacc  

    Kaa_cc = d4K_da2dc2(l, a, c, t, b, phi, sigma) 

    m1 = m
    m2 = m * (m - 1)
    m3 = m * (m - 1) * (m - 2)
    m4 = m * (m - 1) * (m - 2) * (m - 3)

    term1 = m1 * (
        (m - 1) * (m - 2) * (K**(m - 3)) * (Ka**2) * (Kcc)
        + (m - 1) * (K**(m - 2)) * (Kaa * Kcc + 2 * Kac**2)
        + (K**(m - 1)) * Kaa_cc
    )

    term2 = 2 * m2 * (
        (m - 2) * (K**(m - 3)) * Ka * Kc * Kaa
        + (K**(m - 2)) * (Kac * Kaa + Ka * Kacc)
    )

    term3 = m3 * (
        (K**(m - 3)) * (Ka**2) * (Kc**2)
        + 2 * (m - 3) * (K**(m - 4)) * Ka * Kc * Kac
    )

    term4 = m4 * (
        (K**(m - 4)) * (Ka**2) * (Kc**2)
    )

    return C * (term1 + term2 + term3 + term4)


def d4f_dadc3(l, a, c, t, b, phi, sigma, C, m):
    a = np.clip(a, 0.001, 0.08)
    M_ratio = np.clip(a / c, 0.2, 2.0)

    S = sigma * np.sqrt((np.pi * l) / Q(a, c))

    fw = f_w(a / t, c / b)
    Mv = M(a / t, M_ratio)
    gv = g(a / t, M_ratio, phi)

    F = fw * Mv * gv
    K = S * F

    F_a = dF_da(l, a, c, t, b, phi, sigma) 
    F_c = dF_dc(l, a, c, t, b, phi, sigma) 

    K_a = S * F_a
    K_c = S * F_c

    K_ac = S * d2F_dac(l, a, c, t, b, phi, sigma) 
    K_cc = S * d2F_dc2(l, a, c, t, b, phi, sigma) 

    K_acc = S * d3F_dadc2(l, a, c, t, b, phi, sigma) 

    K_dadc3 = S * d4F_dadc3(l, a, c, t, b, phi, sigma) 

    term1 = m * K**(m - 1) * K_dadc3

    term2 = 3 * m * (m - 1) * K**(m - 2) * K_c * K_acc

    term3 = 3 * m * (m - 1) * K**(m - 2) * K_cc * K_ac

    term4 = m * (m - 1) * (m - 2) * K**(m - 3) * (K_c**3) * K_a

    term5 = 3 * m * (m - 1) * (m - 2) * K**(m - 3) * (K_c**2) * K_ac

    term6 = m * (m - 1) * (m - 2) * (m - 3) * K**(m - 4) * K_a * (K_c**3)

    return C * (term1 + term2 + term3 + term4 + term5 + term6)

# nominal DT training data
def generate_nominal_dataset(
    N_train = 10,
    N_test = 500,
    sigma = 8500,
    C = 5.52e-21,
    m = 4,
    seed = 0,
    t = 0.1,
    b = 0.72,
    phi = (np.pi/2) # 5 degrees (5 * np.pi / 180) and 90 degrees (np.pi/2)
):
    np.random.seed(seed)
    
    # training data
    a_train = np.random.uniform(0.001, 0.008, N_train)
    ac_ratio = np.random.uniform(0.2, 2, N_train)
    c_train = a_train / ac_ratio
    l_train = a_train
    
    # this is paris law. I am defining dc/dn but I will call it f_train
    f_train = paris_law_dc_dn(l_train, a_train, c_train, t, b, phi, sigma, C, m)
    
    # derivatives training data
    # first order
    df_da_train = df_da(l_train, a_train, c_train, t, b, phi, sigma, C, m)
    df_dc_train = df_dc(l_train, a_train, c_train, t, b, phi, sigma, C, m)
    
    # second order
    d2f_da2_train = d2f_da2(l_train, a_train, c_train, t, b, phi, sigma, C, m)
    d2f_dc2_train = d2f_dc2(l_train, a_train, c_train, t, b, phi, sigma, C, m)
    
    #partial derivatives for second order
    d2f_dac_train = d2f_dac(l_train, a_train, c_train, t, b, phi, sigma, C, m)
    
    #third order
    d3f_da3_train = d3f_da3(l_train, a_train, c_train, t, b, phi, sigma, C, m)
    d3f_dc3_train = d3f_dc3(l_train, a_train, c_train, t, b, phi, sigma, C, m)
    
    #partial derivatives for third order
    d3f_da2dc_train = d3f_da2dc(l_train, a_train, c_train, t, b, phi, sigma, C, m) 
    d3f_dadc2_train = d3f_dadc2(l_train, a_train, c_train, t, b, phi, sigma, C, m)
    
    #fourth order
    d4f_da4_train = d4f_da4(l_train, a_train, c_train, t, b, phi, sigma, C, m)
    d4f_dc4_train = d4f_dc4(l_train, a_train, c_train, t, b, phi, sigma, C, m)
    
    #partial derivatives for fourth order
    d4f_da3dc_train = d4f_da3dc(l_train, a_train, c_train, t, b, phi, sigma, C, m)
    d4f_da2dc2_train = d4f_da2dc2(l_train, a_train, c_train, t, b, phi, sigma, C, m) 
    d4f_dadc3_train = d4f_dadc3(l_train, a_train, c_train, t, b, phi, sigma, C, m) 
    
    # training data
    X_train = np.vstack((a_train, c_train)).T #(N_train, 2)
    
    
    # testing data
    a_test = np.random.uniform(0.001, 0.008, N_test)
    ac_ratio_test = np.random.uniform(0.2, 2, N_test)
    c_test = a_test / ac_ratio_test
    l_test = a_test
    
    # this is paris law. I am defining dc/dn but I will call it f_test
    f_test = paris_law_dc_dn(l_test, a_test, c_test, t, b, phi, sigma, C, m)
    
    # testing data
    X_test = np.vstack((a_test, c_test)).T #(N_test, 2)
    
    
    return {
        "X_train": X_train,
        "f_train": f_train,
        
        #first order
        "df_da_train": df_da_train,
        "df_dc_train": df_dc_train,
        
        #second order
        "d2f_da2_train": d2f_da2_train,
        "d2f_dc2_train": d2f_dc2_train,
        
        #partial second order
        "d2f_dac_train": d2f_dac_train,
        
        # third order
        "d3f_da3_train": d3f_da3_train,
        "d3f_dc3_train": d3f_dc3_train,
        
        #partial third order
        "d3f_da2dc_train": d3f_da2dc_train,
        "d3f_dadc2_train": d3f_dadc2_train,
        
        #fourth order
        "d4f_da4_train": d4f_da4_train,
        "d4f_dc4_train": d4f_dc4_train,
        
        #partial fourth order
        "d4f_da3dc_train": d4f_da3dc_train,
        "d4f_da2dc2_train": d4f_da2dc2_train,
        "d4f_dadc3_train": d4f_dadc3_train,
        
        "X_test": X_test,
        "f_test": f_test
    }


#code testing for bugs
data = generate_nominal_dataset()
X_train = data["X_train"]
print(X_train.shape)
X_test = data["X_test"]
print(X_test.shape)
f_test = data["f_test"]
print(f_test.shape)

# def aspect_growth_law(a, c, dc_dN):
#     return (a / c) * dc_dN

# def aspect_growth_law(a, c, dc_dN):
#     return 0.6 * dc_dN


# def aspect_growth_law(a, c, dc_dN):
#     ratio = a / c
#     return dc_dN * (0.3 + 0.7 * ratio**2)

# def aspect_growth_law(a, c, dc_dN):
#     ratio = a / c
#     target_ratio = 2.0

#     correction = 1.0 + 0.2 * (target_ratio - ratio)

#     correction = np.clip(correction, 0.5, 1.5)

#     return correction * dc_dN

def aspect_growth_law(a, c, dc_dN):
    ratio = a / c
    return dc_dN * (0.5 + 0.1 * ratio)

#PT simulation
def generate_physical_simulation(
    sigma = 8500,
    t = 0.1,
    b = 0.72,
    C = 5.25e-21,
    m = 3.97,
    seed = 0,
    a0=0.024,
    c0=0.012,
    dN=5e4,
    N_total=7.5e5,
    phi =  (np.pi/2)  # 5 degrees (5 * np.pi / 180) and 90 degrees (np.pi/2)
):
    N_steps = int(N_total / dN)
    N_hist = [0]
    a_hist = [a0]
    c_hist = [c0]
    dc_hist = []
    
    
#     for i in range(N_steps):
#         a_now = a_hist[-1]
#         c_now = c_hist[-1]
#         print("this is c_now: ", c_now)
#         print("this is a_now: ", a_now)
        
#         l = a_now
        
#         #aspect ratio calculation at the start of the interval
#         aspect_ratio = a_now / c_now
#         print("this is aspect_ratio: ", aspect_ratio)
        
#         dc_dN = paris_law_dc_dn(l, a_now, c_now, t, b, phi, sigma, C, m)
#         dc = dc_dN * dN
        
        
#         c_next = c_now + dc
#         a_next = c_next * aspect_ratio
#         a_hist.append(a_next)
#         c_hist.append(c_next)
#         dc_hist.append(dc_dN)
#         N_hist.append((i + 1) * dN)
        
        
    for i in range(N_steps):
        a_now = a_hist[-1]
        c_now = c_hist[-1]

        l = a_now

        # 1) compute dc/dN
        dc_dN = paris_law_dc_dn(l, a_now, c_now, t, b, phi, sigma, C, m)

        # 2) define da/dN 
        da_dN = aspect_growth_law(a_now, c_now, dc_dN)

        # 3) Euler update (start simple)
        c_next = c_now + dc_dN * dN
        a_next = a_now + da_dN * dN

        a_hist.append(a_next)
        c_hist.append(c_next)
        dc_hist.append(dc_dN)
        N_hist.append((i + 1) * dN)
        
        
        print(
            f"step={i}, "
            f"a={a_now:.6e}, "
            f"c={c_now:.6e}, "
            f"a/c={a_now/c_now:.6f}, "
            f"dc_dN={dc_dN:.6e}, "
            f"da_dN={da_dN:.6e}"
        )
        
    return {
        "N": np.array(N_hist),
        "a": np.array(a_hist),
        "c": np.array(c_hist),
        "dc_dN": np.array(dc_hist)
    }


#eta
def eta(pt, dt):
    return np.abs((pt - dt) / pt) * 100.0

def build_rhs(data, derivative_order):
    f = data["f_train"]
    blocks = [f]
    
    if derivative_order >= 1:
        blocks.append(data["df_da_train"])
        blocks.append(data["df_dc_train"])
    
    if derivative_order >= 2:
        blocks.append(data["d2f_da2_train"])
        blocks.append(data["d2f_dc2_train"])
        blocks.append(data["d2f_dac_train"])
        
    if derivative_order >= 3:
        blocks.append(data["d3f_da3_train"])
        blocks.append(data["d3f_dc3_train"])
        blocks.append(data["d3f_da2dc_train"])
        blocks.append(data["d3f_dadc2_train"])
        
    if derivative_order >= 4:
        blocks.append(data["d4f_da4_train"])
        blocks.append(data["d4f_dc4_train"])
        blocks.append(data["d4f_da3dc_train"])
        blocks.append(data["d4f_da2dc2_train"])
        blocks.append(data["d4f_dadc3_train"])
        
    return np.concatenate(blocks).astype(np.float64)


def build_measurement(X_train, derivative_order):
    N_train = X_train.shape[0]
#     print("this is N_train: ", N_train)   # should be 10
    
#     if derivative_order >= 0:
    # 0th order derivatives-function values
    meas_0 = [PointMeasurement(X_train[i]) for i in range(N_train)]
#     print("size of meas_0: ", len(meas_0))
#     print(meas_0)
    
    
    meas_d = []
    if derivative_order >= 1:
        #1st order derivatives
        meas_da = [dPointMeasurement(X_train[i], 0) for i in range(N_train)]
        meas_dc = [dPointMeasurement(X_train[i], 1) for i in range(N_train)]
        meas_d += (meas_da + meas_dc)
        
    #second order derivatives
    meas_dd = []
    if derivative_order >= 2:
        meas_ddaa = [ddPointMeasurement(X_train[i], (0, 0)) for i in range(N_train)]
        meas_ddcc = [ddPointMeasurement(X_train[i], (0, 1)) for i in range(N_train)]
        meas_ddac = [ddPointMeasurement(X_train[i], (1, 1)) for i in range(N_train)]
        meas_dd += (meas_ddaa + meas_ddcc + meas_ddac)
        
    # third order derivatives
    meas_ddd = []
    if derivative_order >= 3:
        meas_dddaaa = [dddPointMeasurement(X_train[i], (0, 0, 0)) for i in range(N_train)]
        meas_dddccc = [dddPointMeasurement(X_train[i], (1, 1, 1)) for i in range(N_train)]
        meas_dddaac = [dddPointMeasurement(X_train[i], (0, 0, 1)) for i in range(N_train)]
        meas_dddacc = [dddPointMeasurement(X_train[i], (0, 1, 1)) for i in range(N_train)]
        meas_ddd += (meas_dddaaa, meas_dddccc, meas_dddaac, meas_dddacc)
        
    # fourth order derivatives
    meas_dddd = []
    if derivative_order >= 4:
        meas_ddddaaaa = [ddddPointMeasurement(X_train[i], (0, 0, 0, 0)) for i in range(N_train)]
        meas_ddddcccc = [ddddPointMeasurement(X_train[i], (1, 1, 1, 1)) for i in range(N_train)]
        meas_ddddaaac = [ddddPointMeasurement(X_train[i], (0, 0, 0, 1)) for i in range(N_train)]
        meas_ddddaacc = [ddddPointMeasurement(X_train[i], (0, 0, 1, 1)) for i in range(N_train)]
        meas_ddddaccc = [ddddPointMeasurement(X_train[i], (0, 1, 1, 1)) for i in range(N_train)]
        meas_dddd += (meas_ddddaaaa, meas_ddddcccc, meas_ddddaaac, meas_ddddaacc, meas_ddddaccc)
        
    meas = [meas_0, meas_d, meas_dd, meas_ddd, meas_dddd]
    meas_train = meas_0 + meas_d + meas_dd + meas_ddd +meas_dddd
#     print("meas_train size in build: ", len(meas_train))
        
    return meas, meas_train


def fast_gpt_cg(
    cov, 
    X_train, 
    X_test, 
    derivative_order, 
    rhs_with_der, 
    sol_init, 
    nugget, 
    rho, 
    k_neighbors, 
    lengthscale,
    N_threads=1,
    lamb = 1.5
):
    N_train = X_train.shape[0]
    meas, meas_train = build_measurement(X_train, derivative_order)
    
    N_test = X_test.shape[0] # should be 500
    meas_test = [PointMeasurement(X_test[i]) for i in range(N_test)]
    
#     print("before implicit")
    implicit_factor,SN_wot_der, P_wot_der = ImplicitKLFactorization.implicit_kl_factorization_for_d_with_partial_split_approach_1(cov, meas, rho, k_neighbors, lambda_ = lamb)
#     print("after implicit")
    explicit_factor = ExplicitKLFactorization.Explicit_from_implicit(implicit_factor,nugget=nugget,N_threads=N_threads)
    
    U = explicit_factor.U
    L = U.transpose().tocsc()
    P = explicit_factor.P
    
    rhs_now = rhs_with_der
    
    Oinv_rhs = sol_init
#     print("size of 0inv_rhs: ", len(Oinv_rhs))
    Oinv_rhs[P] = U@(L @ rhs_now[P])
    
#     print("size of meas_train: ", len(meas_train))
#     print("size of P: ", len(P))
    
    
    reorder_meas_train = [meas_train[i] for i in P]
    
#     cov2 = GaussianCovariance_generic(lengthscale)
#     Theta_test = build_test(cov2, meas_test, reorder_meas_train)

    Theta_test=np.zeros((len(meas_test),len(reorder_meas_train)))
    for i in range(len(meas_test)):
        for j in range(len(reorder_meas_train)):
            Theta_test[i,j]=cov(meas_test[i],reorder_meas_train[j])
    
    
    y_predicted = np.zeros(len(meas_test))
    y_predicted = Theta_test @ Oinv_rhs[P]
    
#     print("shape of y predicted: ", y_predicted.shape)
    
    return y_predicted, explicit_factor, SN_wot_der, P_wot_der


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

def compute_sparsity(matrix):
    total_elements = matrix.shape[0] * matrix.shape[1]
    nonzero_elements = matrix.nnz
    sparsity = 1.0 - (nonzero_elements / total_elements)
    return sparsity


# function using the dynamic updates
def new_fast_gpt(
    cov, 
    EF_old,  #old explicit factor
    X_train, 
    old_pts,
    new_pts,
    loc_NP,
    SN_without_der,
    P_without_der,
    X_test, 
    rhs_with_der,
    sol_init,
    nugget,
    rho,
    k_neighbors,
    derivative_order, 
    lengthscale,
    N_threads=1
):
    N_train = X_train.shape[0]
    d = X_train.shape[1]   # dimension
    print("this is the dimension d: ", d)
    
    
    meas, meas_train = build_measurement(X_train, derivative_order)
    r_set = [node.column_indices for node in SN_without_der if not node.fixed]
    r_set = P_without_der[r_set].flatten()
    r_set = np.hstack((r_set, np.array(loc_NP).flatten()))
    new_meas = [PointMeasurement(X_train[i]) for i in r_set]
    
    implicit_factor, SN_wot_der, P_wot_der = ImplicitKLFactorization.dynamic_implicit_kl_factorization_for_d_partial_split_approach_1(cov, old_pts, new_meas, SN_without_der, P_without_der, r_set, meas, loc_NP, rho, k_neighbors)
    explicit_factor = ExplicitKLFactorization.dynamic_Explicit_from_implicit(implicit_factor, EF_old.U, EF_old.SO, nugget=nugget, N_threads=N_threads)
    
    U = explicit_factor.U
    L = U.transpose().tocsc() # transpose and convert to compressed sparse column format
    P = explicit_factor.P
    
    rhs_now = rhs_with_der
    Oinv_rhs = sol_init
    Oinv_rhs[P] = U @ (L @ rhs_now[P])
    
    N_test = X_test.shape[0]
    meas_test = [PointMeasurement(X_test[i]) for i in range(N_test)]
    reorder_meas_train = [meas_train[i] for i in P]
    
    Theta_test=np.zeros((len(meas_test),len(reorder_meas_train)))
    for i in range(len(meas_test)):
        for j in range(len(reorder_meas_train)):
            Theta_test[i,j]=cov(meas_test[i],reorder_meas_train[j])
            
    y_predicted=np.zeros(len(meas_test))
    y_predicted=Theta_test @ Oinv_rhs[P]
    
    return y_predicted, explicit_factor, SN_wot_der, P_wot_der

# dynamic DT experiments
def run_dynamic_dt_experiment(
    derivative_order,
    rho,
    lengthscale,
    der_indices,  # this will be a list/array of derivative indices 
    nugget,
    k_neighbors,
    n,   # number of training points. In the case of 2D, this will mean that there are n^(0.5) points in 1 axis
    dN=5e4
):
    data = generate_nominal_dataset()
    rhs = build_rhs(data, derivative_order=derivative_order)   
    
    #initial training
    cov = GaussianCovariance_generic(lengthscale)
    
#     truth_with_der=np.zeros(n*len(der_indices))
#     for i in range(len(der_indices)):
#         truth_with_der[i*n:(i*n)+n]=np.array(file[str(der_indices[i])]).flatten()
#     N_domain=np.shape(truth_with_der)
#     sol_init=np.zeros(N_domain)


    print("rhs length: ", len(rhs))
    sol_init = np.zeros(len(rhs), dtype=np.float64)
    
#     print("size of sol_init: ", len(sol_init))
    
#     y_predicted, explicit_factor, SN_without_der, P_without_der = fast_gpt_cg(
#         cov, 
#         data["X_train"], 
#         data["X_test"], 
#         derivative_order=derivative_order, 
#         rhs_with_der=rhs, 
#         sol_init=sol_init, 
#         nugget=nugget, 
#         rho=rho, 
#         k_neighbors=k_neighbors, 
#         lengthscale=lengthscale 
#     )
    
    
    #PT simulation
    pt = generate_physical_simulation()
    pt_queries = np.column_stack((pt["a"], pt["c"]))  # testing at inspection points
    test_truth_pt = pt["dc_dN"]
    print("shape of test_truth_pt: ", test_truth_pt.shape)
    
    
    eta_hist = []
    N_hist = []   # fatigue cycles
    eta_retrain = []
    X_train = data["X_train"].copy()

    # amount of data that can be stored without re-training
    data_cost_budget = int(0.2 * X_train.shape[0])
    pp_for_outlier = 99
    
    
    sol_1, EF, SN_without_der, P_without_der = fast_gpt_cg(
        cov,
        data["X_train"],
        pt_queries,
#         data["X_test"],
        derivative_order = derivative_order,
        rhs_with_der = rhs,
        sol_init = sol_init,
        nugget = nugget,
        rho = rho,
        k_neighbors = k_neighbors,
        lengthscale = lengthscale
    )

    init_spar = compute_sparsity(EF.U)
    
#     o_err = np.mean((test_truth_pt - sol_1) ** 2)
#     check_err = o_err

    # 1. error-1
#     o_err =  eta(test_truth_pt, sol_1)
#     check_err = o_err
    
    o_err = np.mean(eta(test_truth_pt, sol_1[1:]))
    check_err = o_err
    
    d_set = [node.column_indices for node in SN_without_der if not node.fixed]
    print("shape of P_without_der: ", P_without_der.shape)
    
    
    dp = X_train[P_without_der[d_set]][0]
    
    old_size = X_train.shape[0]
    update_size = old_size
    
#     err_all = []
    eta_all = []
    
    Sp_all = []
    
#     err = []
    
    sp_ = []
    
    N_hist.append(0)
    eta_hist.append(o_err)
    sp_.append(init_spar)
    err_ratio_hit = 0
    budget_count = 0
    new_points = []
    
    n_steps = min(
        len(pt["a"]),
        len(pt["c"]),
        len(pt["dc_dN"]),
        len(pt["N"])
    )

    for i in range(n_steps):
        a_pt = pt["a"][i]
        c_pt = pt["c"][i]
        dc_pt = pt["dc_dN"][i]
        
        # DT prediction at PT inspection state
        X_query = np.array([[a_pt, c_pt]])
        
        
        sol_init = np.zeros(len(rhs), dtype=np.float64)
        y_query, _, _, _ = fast_gpt_cg(
            cov, 
            X_train, 
            X_query, 
            derivative_order=derivative_order, 
            rhs_with_der=rhs, 
            sol_init=sol_init, 
            nugget=nugget, 
            rho=rho, 
            k_neighbors=k_neighbors, 
            lengthscale=lengthscale 
        )
        
        dc_dt = y_query[0]
        
        #relative error
        eta_now = eta(dc_pt, dc_dt)
        
        # eta_hist here
        eta_hist.append(eta_now)
        N_hist.append(pt["N"][i])
        
        
        print(
            f"Inspection {i+1:02d} | "
            f"N = {pt['N'][i]:0.2e} | "
            f"dc_PT = {dc_pt:.4e} | "
            f"dc_DT = {dc_dt:.4e} | "
            f"eta = {eta_now:.2f}%"
        )
        
        #dynamic update of the DT model
        loc_np = [update_size]
        
        #new point
        new_point = np.array([a_pt, c_pt])
        new_points.append(new_point)
        
        #outlier detection
#         report = find_outliers_with_knn_dist(X_train, new_points[i:i+1, :], k = k_neighbors, pp = pp_for_outlier)
        report = find_outliers_with_knn_dist(X_train, np.array(new_points[i:i+1]), k = k_neighbors, pp = pp_for_outlier)
        # retrain based on the criteria
        if report[0]['is_outlier'] == True or err_ratio_hit >=3 or budget_count >= data_cost_budget:
            print(f"Retraining triggered, outlier={report[0]['is_outlier']}, err_ratio_hit={err_ratio_hit},budget_count={budget_count}")
            err_ratio_hit = 0
            budget_count = 0
            
            #check to make sure the new point is not in the training set
            existing_points_set = {tuple(row) for row in X_train}
            points_to_add = [p for p in new_points[:i+1] if tuple(p) not in existing_points_set]
            
            if points_to_add:
                #append new PT observation
                X_train_new = np.vstack((X_train, np.array(points_to_add)))
            else:
                X_train_new = X_train.copy()
                
        
            #compute PT derivative information using PT materical properties
            sigma_pt = 8500
            C_pt = 5.25e-21
            m_pt = 3.97
            phi_pt =  (np.pi/2) # 5 degrees (5 * np.pi / 180) and 90 degrees (np.pi/2)
            t_pt = 0.1
            b_pt = 0.72
            l_pt = a_pt

            # derivatives data below
            # first order
            df_da_dynamic = df_da(l_pt, a_pt, c_pt, t_pt, b_pt, phi_pt, sigma_pt, C_pt, m_pt)
            df_dc_dynamic = df_dc(l_pt, a_pt, c_pt, t_pt, b_pt, phi_pt, sigma_pt, C_pt, m_pt)

            # second order
            d2f_da2_dynamic = d2f_da2(l_pt, a_pt, c_pt, t_pt, b_pt, phi_pt, sigma_pt, C_pt, m_pt)
            d2f_dc2_dynamic = d2f_dc2(l_pt, a_pt, c_pt, t_pt, b_pt, phi_pt, sigma_pt, C_pt, m_pt)

            #partial derivatives for second order
            d2f_dac_dynamic = d2f_dac(l_pt, a_pt, c_pt, t_pt, b_pt, phi_pt, sigma_pt, C_pt, m_pt)

            #third order
            d3f_da3_dynamic = d3f_da3(l_pt, a_pt, c_pt, t_pt, b_pt, phi_pt, sigma_pt, C_pt, m_pt)
            d3f_dc3_dynamic = d3f_dc3(l_pt, a_pt, c_pt, t_pt, b_pt, phi_pt, sigma_pt, C_pt, m_pt)

            #partial derivatives for third order
            d3f_da2dc_dynamic = d3f_da2dc(l_pt, a_pt, c_pt, t_pt, b_pt, phi_pt, sigma_pt, C_pt, m_pt)
            d3f_dadc2_dynamic = d3f_dadc2(l_pt, a_pt, c_pt, t_pt, b_pt, phi_pt, sigma_pt, C_pt, m_pt)

            #fourth order
            d4f_da4_dynamic = d4f_da4(l_pt, a_pt, c_pt, t_pt, b_pt, phi_pt, sigma_pt, C_pt, m_pt)
            d4f_dc4_dynamic = d4f_dc4(l_pt, a_pt, c_pt, t_pt, b_pt, phi_pt, sigma_pt, C_pt, m_pt)

            #partial derivatives for fourth order
            d4f_da3dc_dynamic = d4f_da3dc(l_pt, a_pt, c_pt, t_pt, b_pt, phi_pt, sigma_pt, C_pt, m_pt)
            d4f_da2dc2_dynamic = d4f_da2dc2(l_pt, a_pt, c_pt, t_pt, b_pt, phi_pt, sigma_pt, C_pt, m_pt)
            d4f_dadc3_dynamic = d4f_dadc3(l_pt, a_pt, c_pt, t_pt, b_pt, phi_pt, sigma_pt, C_pt, m_pt)

            #expanding RHS based on the order of derivatives
            if derivative_order >= 0:
                rhs = np.concatenate((
                    rhs,
                    np.array([float(dc_pt)])
                ), dtype=np.float64)

            if derivative_order >= 1:
                rhs = np.concatenate((rhs, np.array([float(df_da_dynamic), float(df_dc_dynamic)])), dtype=np.float64)

            if derivative_order >= 2:
                rhs = np.concatenate((rhs, np.array([float(d2f_da2_dynamic), float(d2f_dc2_dynamic), float(d2f_dac_dynamic)])), dtype=np.float64)

            if derivative_order >= 3:
                rhs = np.concatenate((rhs, np.array([float(d3f_da3_dynamic), float(d3f_dc3_dynamic), float(d3f_da2dc_dynamic), float(d3f_dadc2_dynamic)])), dtype=np.float64)

            if derivative_order == 4:
                rhs = np.concatenate((rhs, np.array([float(d4f_da4_dynamic), float(d4f_dc4_dynamic), float(d4f_da3dc_dynamic), float(d4f_da2dc2_dynamic), float(d4f_dadc3_dynamic)])), dtype=np.float64)

            update_size = X_train_new.shape[0]
            sol_init = np.zeros(len(rhs), dtype=np.float64)
            
            # retrain
            sol_1, EF, new_SN_wot_der, new_P_wot_der = fast_gpt_cg(
                cov,
                X_train_new, 
#                 data["X_test"],
                pt_queries,
                derivative_order = derivative_order,
                rhs_with_der = rhs,
                sol_init = sol_init,
                nugget = nugget,
                rho = rho,
                k_neighbors = k_neighbors,
                lengthscale = lengthscale
            )
            
#             last_err = np.mean((test_truth_pt - sol_1) ** 2)
#             last_err = eta(test_truth_pt, sol_1)
            last_err = np.mean(eta(test_truth_pt, sol_1[1:]))
            
            X_train = X_train_new
            d_set = [node.column_indices for node in SN_without_der if not node.fixed]
            dp = X_train[P_without_der[d_set], :][0]
            SN_without_der = new_SN_wot_der
            P_without_der = np.array(new_P_wot_der)
            
            #should be eta error
#             last_err = np.mean((test_truth_pt - sol_1) ** 2)
#             check_err = last_err
#             err.append(np.mean((test_truth_pt - sol_1) ** 2))
#             last_err = eta(test_truth_pt, sol_1)
            last_err = np.mean(eta(test_truth_pt, sol_1[1:]))
            check_err = last_err
#             eta_hist.append(last_err)
            sp_.append(compute_sparsity(EF.U))
                
        else:
            #dynamic update without retraining-check this again in the end
            X_train_new = np.vstack((X_train, np.array(new_points[i:i+1])))
            dp = np.vstack((dp, np.array(new_points[i:i+1])))

            # step 1 (to be filled): update the rhs to make rhs_with_der
            # derivatives data below
            # first order
            df_da_dynamic = df_da(l_pt, a_pt, c_pt, t_pt, b_pt, phi_pt, sigma_pt, C_pt, m_pt)
            df_dc_dynamic = df_dc(l_pt, a_pt, c_pt, t_pt, b_pt, phi_pt, sigma_pt, C_pt, m_pt)

            # second order
            d2f_da2_dynamic = d2f_da2(l_pt, a_pt, c_pt, t_pt, b_pt, phi_pt, sigma_pt, C_pt, m_pt)
            d2f_dc2_dynamic = d2f_dc2(l_pt, a_pt, c_pt, t_pt, b_pt, phi_pt, sigma_pt, C_pt, m_pt)

            #partial derivatives for second order
            d2f_dac_dynamic = d2f_dac(l_pt, a_pt, c_pt, t_pt, b_pt, phi_pt, sigma_pt, C_pt, m_pt)

            #third order
            d3f_da3_dynamic = d3f_da3(l_pt, a_pt, c_pt, t_pt, b_pt, phi_pt, sigma_pt, C_pt, m_pt)
            d3f_dc3_dynamic = d3f_dc3(l_pt, a_pt, c_pt, t_pt, b_pt, phi_pt, sigma_pt, C_pt, m_pt)

            #partial derivatives for third order
            d3f_da2dc_dynamic = d3f_da2dc(l_pt, a_pt, c_pt, t_pt, b_pt, phi_pt, sigma_pt, C_pt, m_pt)
            d3f_dadc2_dynamic = d3f_dadc2(l_pt, a_pt, c_pt, t_pt, b_pt, phi_pt, sigma_pt, C_pt, m_pt)

            #fourth order
            d4f_da4_dynamic = d4f_da4(l_pt, a_pt, c_pt, t_pt, b_pt, phi_pt, sigma_pt, C_pt, m_pt)
            d4f_dc4_dynamic = d4f_dc4(l_pt, a_pt, c_pt, t_pt, b_pt, phi_pt, sigma_pt, C_pt, m_pt)

            #partial derivatives for fourth order
            d4f_da3dc_dynamic = d4f_da3dc(l_pt, a_pt, c_pt, t_pt, b_pt, phi_pt, sigma_pt, C_pt, m_pt)
            d4f_da2dc2_dynamic = d4f_da2dc2(l_pt, a_pt, c_pt, t_pt, b_pt, phi_pt, sigma_pt, C_pt, m_pt)
            d4f_dadc3_dynamic = d4f_dadc3(l_pt, a_pt, c_pt, t_pt, b_pt, phi_pt, sigma_pt, C_pt, m_pt)

            #expanding RHS based on the order of derivatives
            if derivative_order >= 0:
                rhs = np.concatenate((
                    rhs,
                    np.array([float(dc_pt)])
                ), dtype=np.float64)

            if derivative_order >= 1:
                rhs = np.concatenate((rhs, np.array([float(df_da_dynamic), float(df_dc_dynamic)])), dtype=np.float64)

            if derivative_order >= 2:
                rhs = np.concatenate((rhs, np.array([float(d2f_da2_dynamic), float(d2f_dc2_dynamic), float(d2f_dac_dynamic)])), dtype=np.float64)

            if derivative_order >= 3:
                rhs = np.concatenate((rhs, np.array([float(d3f_da3_dynamic), float(d3f_dc3_dynamic), float(d3f_da2dc_dynamic), float(d3f_dadc2_dynamic)])), dtype=np.float64)

            if derivative_order == 4:
                rhs = np.concatenate((rhs, np.array([float(d4f_da4_dynamic), float(d4f_dc4_dynamic), float(d4f_da3dc_dynamic), float(d4f_da2dc2_dynamic), float(d4f_dadc3_dynamic)])), dtype=np.float64)

            # step 2 (to be filled): make sol_init 
            sol_init = np.zeros(len(rhs), dtype=np.float64)
            
            
            SN_without_der1=copy.deepcopy(SN_without_der)
            sol_1_new, EF_new, new_SN_wot_der, new_P_wot_der = new_fast_gpt(
                cov,
                EF,
                X_train_new,
                pt_queries,
                np.array(new_points[i:i+1]),
                loc_np,
                SN_without_der1,
                P_without_der,
                data["X_test"],
                rhs_with_der = rhs,
                sol_init = sol_init, 
                nugget = nugget,
                rho = rho,
                k_neighbors = k_neighbors,
                derivative_order = derivative_order,
                lengthscale = lengthscale
            )
            
            
            # should be eta most likely
#             last_err = np.mean((test_truth_pt - sol_1_new) ** 2)
            last_err = np.mean(eta(test_truth_pt, sol_1[1:]))
            
            if last_err < check_err:
                print("dynamic update accepted.")
#                 err.append(np.mean((test_truth_pt - sol_1_new) ** 2))
                eta_hist.append(last_err)
                sp_.append(compute_sparsity(EF_new.U))
                X_train = X_train_new
                EF = EF_new
                SN_without_der = copy.deepcopy(new_SN_wot_der)
                P_without_der = copy.deepcopy(new_P_wot_der)
                update_size += 1
#                 check_err = np.mean((test_truth_pt - sol_1_new) ** 2)
                check_err = eta(test_truth_pt, sol_1_new)
                err_ratio_hit = 0
            else:
                print("dynamic update rejected")
                try:
                    if last_err/check_err > err_ratio:
                        err_ratio = last_err/check_err
                        err_ratio_hit += 1
                except:
                    if last_err/check_err > 1:
                        err_ratio = last_err/check_err
                        err_ratio_hit += 1
                    budget_count += 1
#                     err.append(check_err)
#                     eta_hist.append(check_err)
                    sp_.append(compute_sparsity(EF_new.U))
                    SN_without_der = copy.deepcopy(SN_without_der)
                    P_without_der = np.array(P_without_der)
                
#     err_all.append(err)
    eta_all.append(eta_hist)
    Sp_all.append(sp_)
    np.savetxt(f"./dynamic_2D_4_approrach_1_err_application.csv",eta_all,delimiter=",")
            
    return {
        "N": np.array(N_hist),
        "eta": np.array(eta_hist)
    }

# run experiments with multiple order of derivatives
results = {}
orders = [0]   # orders = [0, 1, 2, 3, 4]

#     der_indices=[[0], [1],[2],[1,1],[1,2],[2,2],[1, 1, 1], [1, 1, 2], [1, 2, 2], [2, 2, 2],
#                 [1, 1, 1, 1], [1, 1, 1, 2], [1, 1, 2, 2], [1, 2, 2, 2], [2, 2, 2, 2]]
der_indices = [[0]]
n = 10 #number of training points. In the case of 2D, this will mean that there are n^(0.5) points in 1 axis
nugget = 1e-10

lengthscale = 0.7
k_neighbors = 3
rho = 20

for order in orders:
    print("\n" + "=" * 60)
    print(f"Running derivative order = {order}")
    print("=" * 60)

    results[order] = run_dynamic_dt_experiment(
        derivative_order=order,
        rho=rho,
        lengthscale=lengthscale,
        der_indices=der_indices,  # this will be a list/array of derivative indices 
        nugget=nugget,
        k_neighbors=k_neighbors,
        n=n,
        dN=5e4
    )
    
    
for order in orders:
    print("order =", order)
    print("len(N)   =", len(results[order]["N"]))
    print("len(eta) =", len(results[order]["eta"]))    
    

#save results
for order in orders:
    df = pd.DataFrame({
        "N": results[order]["N"],
        "eta": results[order]["eta"]
    })

    df.to_csv(
        f"results_order_{order}.csv",
        index=False
    )

print("\nSaved all experiment results.")