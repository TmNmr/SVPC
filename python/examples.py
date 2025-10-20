#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Example functions for 
    - Kohn-Strang-Dolzmann, 
    - Double well and
    - Saint Venant-Kirchhoff example,
for Signed Singular Value polyconvexification
    
author: Timo Neumeier, timo.neumeier@uni-a.de
date: created Oct 2025
"""

import numpy as np
from SVPC import *

# -----------------------------------------------------------------------------
# Kohn-Strang-Dolzmann
# -----------------------------------------------------------------------------
def W_KSD(F):
    return (2 * np.sqrt(2) * np.linalg.norm(F)
            if (np.linalg.norm(F) <= (np.sqrt(2) - 1))
            else (1 + np.linalg.norm(F)**2))

def Phi_KSD(v):
    return (2 * np.sqrt(2) * np.sqrt(np.sum(v**2, -1))) * (np.sqrt(np.sum(v**2, -1)) <= (np.sqrt(2) - 1)) \
        +  (1 + np.sum(v**2, -1)) * (np.sqrt(np.sum(v**2, -1)) > (np.sqrt(2) - 1))

def Wpc_KSD(F):
    rho = np.sqrt(np.linalg.norm(F) ** 2 + 2 * abs(np.linalg.det(F)))
    return (2 * (rho - abs(np.linalg.det(F)))) if (rho <= 1) else (1 + np.linalg.norm(F)**2)

def Phipc_KSD(v):
    return (2 * (np.sum(abs(v), -1) - abs(v[..., 0] * v[..., 1]))) * (np.sum(abs(v), -1) <= 1) \
        +  (1 + np.sum(v**2, -1)) * (np.sum(abs(v), -1) > 1)

def DWpc_KSD(F):
    rho = np.sqrt(np.linalg.norm(F) ** 2 + 2 * abs(np.linalg.det(F)))
    def Drho(F):
        rho = np.sqrt(np.linalg.norm(F) ** 2 + 2 * abs(np.linalg.det(F)))    
        return 1 / rho * (F + np.sign(np.linalg.det(F)) * np.linalg.det(F) * np.linalg.inv(F).T)
    return (2 * (Drho(F) - np.sign(np.linalg.det(F)) * np.linalg.det(F) * np.linalg.inv(F).T)) if (rho <= 1) else (2 * F)

def DPhi_KSD(v):
    return 2 * np.sqrt(2) / np.sqrt(sum(v**2)) * v if np.sqrt(sum(v**2)) <= (np.sqrt(2) - 1) else 2 * v

def DPhipc_KSD(v):
    return 2* (v / abs(v) - (v[0] * v[1]) / abs(v[0] * v[1]) * np.array([v[1], v[0]])) if sum(abs(v)) <= 1 else 2 * v

def W_KSD_vec(Fs):
    norms = np.linalg.norm(Fs, axis=1)
    return np.where(norms <= np.sqrt(2) - 1, 2 * np.sqrt(2) * norms, 1 + norms**2)

# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Double well
# -----------------------------------------------------------------------------
def W_DW(F):
    return (np.linalg.norm(F)**2 - 1)**2 
        
def DW_DW(F):
    return 4 * (np.linalg.norm(F)**2 - 1) * F

def Wpc_DW(F):
    return (np.linalg.norm(F)**2 - 1)**2  if (np.linalg.norm(F) >= 1) else 0

def DWpc_DW(F):
    return 4 * (np.linalg.norm(F)**2 - 1) * F  if (np.linalg.norm(F) >= 1) else np.zeros_like(F)

def Phi_DW(v):
    return (np.sum(v**2, -1) - 1)**2 

def DPhi_DW(v): 
    return 4 * (np.linalg.norm(v)**2 - 1) * v

def Phipc_DW(v):
    return (np.maximum(np.sum(v**2, -1) - 1, 0))**2 

def DPhipc_DW(v): 
    return 4 * (np.linalg.norm(v)**2 - 1) * v if (np.linalg.norm(v) >= 1) else np.zeros_like(v)

def W_DW_vec(Fs):
    return (np.linalg.norm(Fs, axis=1)**2 - 1)**2 

# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Saint Venant-Kirchhoff
# -----------------------------------------------------------------------------
# Material constants
E, nu = 1, 1 / 4

def W_STVK(F):
    if F.ndim == 1: d = int(np.sqrt(len(F))); F = F.reshape((d, d))
    return E / (8 * (1 + nu)) * np.linalg.norm(F.T @ F - np.eye(*F.shape))**2 + E * nu / (8 * (1 + nu) * (1 - 2*nu)) * (np.linalg.norm(F)**2 - np.trace(np.eye(*F.shape)))**2

def Phi_STVK(v):
    d = v.shape[-1]
    return E / (8 * (1 + nu)) * np.sum((v**2 - 1)**2, -1) + E * nu / (8 * (1 + nu) * (1 - 2*nu)) * (np.sum(v**2, -1) - d)**2

def Psi3D(v):
    # 0 <= v[0] <= v[1] <= v[2]
    v1, v2, v3 = v[..., 0], v[..., 1], v[..., 2]
    plus2 = lambda x: x**2 * (x >= 0) + 0.0
    return E / 8 * (plus2(v3**2 - 1) + 1 / (1 - nu**2) * plus2(v2**2 + nu*v3**2 - (1 + nu)) +
                    1 / ((1 - nu**2) * (1 - 2*nu)) * plus2((1 - nu) * v1**2 + nu * (v2**2 + v3**2) - (1 + nu)))

def Psi2D(v):
    # 0 <= v[0] <= v[1]
    v1, v2 = v[..., 0], v[..., 1]
    plus2 = lambda x: x**2 * (x >= 0) + 0.0
    return E / (8 * (1 - nu**2)) * (plus2(v2**2 - 1) + 1 / (1 - 2*nu) * plus2((1 - nu) * v1**2 + nu * v2**2 - 1))

                
def Wpc_STVK(F):
    if F.ndim == 1: d = int(np.sqrt(len(F))); F = F.reshape((d, d))
    d = F.shape[0]
    S = SV(signedSingularValues(F))
    if d == 2:
        return Psi2D(S)
    elif d == 3:
        return Psi3D(S)

def Phipc_STVK(v):
    d = v.shape[-1]    
    S = np.sort(abs(v), -1)
    if d == 2:
        return Psi2D(S)
    elif d == 3:
        return Psi3D(S)

def DPhipc_STVK(v):
    d = len(v)
    Dplus2 = lambda x: 2*x if x >= 0 else 0
    signsv = np.sign(v)
    idx = np.argsort(abs(v))    
    P = np.zeros((d, d))
    P[np.arange(d), idx] = 1
    sv = abs(v)[idx]
    
    if d == 2:
        sv1, sv2 = sv[0], sv[1]
        DPsi = E / (8 * (1 + nu) * (1 - nu)) * ( Dplus2(sv2**2 - 1) * np.array([0, 2*sv2]) 
                        + 1 / (1 - 2*nu) * Dplus2((1 - nu) * sv1**2 + nu * sv2**2 - 1) * np.array([2 * (1 - nu) * sv1, 2*nu*sv2]))
        
    elif d == 3:        
        sv1, sv2, sv3 = sv[0], sv[1], sv[2]
        DPsi = E / 8 * (Dplus2(sv3**2 - 1) * np.array([0, 0, 2*sv3]) 
                        + 1 / (1 - nu**2) * Dplus2(sv2**2 + nu*sv3**2 - (1 + nu)) * np.array([0, 2*sv2, 2*nu*sv3]) 
                        + 1 / ((1 - nu**2) * (1 - 2*nu)) * Dplus2((1 - nu) * sv1**2 + nu * (sv2**2 + sv3**2) - (1 + nu)) * np.array([2 * (1 - nu) * sv1, 2*nu*sv2, 2*nu*sv3]))

    return signsv * (P.T @ DPsi)

def W_STVK_vec(Fs):
    d = int(np.rint(Fs.shape[-1]**(0.5)))
    I = np.eye(d)
    Fs = np.reshape(Fs, (-1, d, d))
    return E / (8 * (1 + nu)) * np.sum((np.einsum('...ji,...jk->...ik', Fs, Fs) - I)**2, axis=(-2, -1)) + E * nu / (8 * (1 + nu) * (1 - 2*nu)) * (np.sum(Fs**2, axis=(-2, -1)) - d)**2

# -----------------------------------------------------------------------------
