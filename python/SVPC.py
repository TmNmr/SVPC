#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Signed singular value polyconvexification for isotropic energy densities based on
    [1] Timo Neumeier, Malte A. Peter, Daniel Peterseim, David Wiedemann.
        Computational polyconvexification of isotropic functions, MMS, 2024.


Implementation of the Quickhull and Linear Programming approach.


Signed Singular Value Polyconvexification (d dimensional setting) by
    - Quickhull (SVPC QH) 
    - Linear Programming (SVPC LP) 
and Polyconvexification (d x d dimensional setting) by
    - Quickhull (PC QH) 
    - Linear Programming (PC LP) 
    
author: Timo Neumeier, timo.neumeier@uni-a.de
date: created Oct 2025
"""

import numpy as np
import scipy.optimize
import scipy.spatial


# -----------------------------------------------------------------------------
# Singular Values and Signed Singular Values mapping for matrices
# -----------------------------------------------------------------------------
def singularValues(F):
    svs = scipy.linalg.svdvals(F)[::-1]
    return svs

def signedSingularValues(F):
    ssvs = scipy.linalg.svdvals(F)[::-1]
    ssvs[0] *= np.sign(np.linalg.det(F))
    return ssvs


# -----------------------------------------------------------------------------
# Singular Values and Signed Singular Values mapping for the vectors
#  -> canonicalisation
# -----------------------------------------------------------------------------
def SV(v):
    svs = np.sort(abs(v), -1)
    return svs

def SSV(v):
    ssvs = np.sort(abs(v), -1)
    ssvs[..., 0] *= np.sign(np.prod(v, -1))
    return ssvs


# -----------------------------------------------------------------------------
# Minors mappings
# -----------------------------------------------------------------------------
def minors(v):
    if v.shape[-1] == 2:
        return np.array([v[..., 0], v[..., 1], v[..., 0] * v[..., 1]]).T
    elif v.shape[-1] == 3:
        return np.array([v[..., 0], v[..., 1], v[..., 2],
                         v[..., 1] * v[..., 2], v[..., 0] * v[..., 2], v[..., 0] * v[..., 1],
                         v[..., 0] * v[..., 1] * v[..., 2]]).T


def Dminors(v):
    if v.shape[-1] == 2:
        return np.array([[1, 0, v[..., 1]],
                         [0, 1, v[..., 0]]]).T
    elif v.shape[-1] == 3:
        return np.array([[1, 0, 0, 0, v[..., 2], v[..., 1], v[..., 1] * v[..., 2]],
                         [0, 1, 0, v[..., 2], 0, v[..., 0], v[..., 0] * v[..., 2]],
                         [0, 0, 1, v[..., 1], v[..., 0], 0, v[..., 0] * v[..., 1]]]).T


def Minors(F):
    if F.shape[-1] == 4:
        return np.array([F[..., 0], F[..., 1], F[..., 2], F[..., 3], F[..., 0] * F[..., 3] - F[..., 1] * F[..., 2]]).T
    elif F.shape[-1] == 9:
        return np.array([F[..., 0], F[..., 1], F[..., 2], F[..., 3], F[..., 4], F[..., 5], F[..., 6], F[..., 7], F[..., 8],
                         F[..., 4] * F[..., 8] - F[..., 7] * F[..., 5],
                         - F[..., 1] * F[..., 8] + F[..., 7] * F[..., 2],
                         F[..., 1] * F[..., 5] - F[..., 4] * F[..., 2],
                         - F[..., 3] * F[..., 8] + F[..., 6] * F[..., 5],
                         F[..., 0] * F[..., 8] - F[..., 6] * F[..., 2], 
                         - F[..., 0] * F[..., 5] + F[..., 3] * F[..., 2],
                         F[..., 3] * F[..., 7] - F[..., 6] * F[..., 4],
                         - F[..., 0] * F[..., 7] + F[..., 6] * F[..., 1],
                         F[..., 0] * F[..., 4] - F[..., 3] * F[..., 1], 
                         np.prod(F[..., [0, 4, 8]], axis=-1) + np.prod(F[..., [1, 5, 6]], axis=-1) + np.prod(F[..., [2, 3, 7]], axis=-1) - (np.prod(F[..., [2, 4, 6]], axis=-1) + np.prod(F[..., [0, 5, 7]], axis=-1) + np.prod(F[..., [1, 3, 8]], axis=-1))
                         ]).T


def DMinors(F):
    if F.shape[-1] == 4:
        return np.array([[1, 0, 0, 0, F[..., 3]],
                         [0, 1, 0, 0, - F[..., 2]],
                         [0, 0, 1, 0, - F[..., 1]],
                         [0, 0, 0, 1, F[..., 0]],
                         ]).T
    elif F.shape[-1] == 9:
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, F[..., 8], -F[..., 5], 0, -F[..., 7], F[..., 4], F[..., 4] * F[..., 8] - F[..., 5] * F[..., 7]],
                         [0, 1, 0, 0, 0, 0, 0, 0, 0,  0, -F[..., 8], F[..., 5], 0, 0, 0, 0, F[..., 6], -F[..., 3], F[..., 5] * F[..., 6] - F[..., 3] * F[..., 8]],
                         [0, 0, 1, 0, 0, 0, 0, 0, 0,  0, F[..., 7], -F[..., 4], 0, -F[..., 6], F[..., 3], 0, 0, 0, F[..., 3] * F[..., 7] - F[..., 4] * F[..., 6]],
                         [0, 0, 0, 1, 0, 0, 0, 0, 0,  0, 0, 0, -F[..., 8], 0, F[..., 2], F[..., 7], 0, -F[..., 1], F[..., 2] * F[..., 7] - F[..., 1] * F[..., 8]],
                         [0, 0, 0, 0, 1, 0, 0, 0, 0,  F[..., 8], 0, -F[..., 2], 0, 0, 0, -F[..., 6], 0, F[..., 0], F[..., 0] * F[..., 8] - F[..., 2] * F[..., 6]],
                         [0, 0, 0, 0, 0, 1, 0, 0, 0,  -F[..., 7], 0, F[..., 1], F[..., 6], 0, -F[..., 0], 0, 0, 0, F[..., 1] * F[..., 6] - F[..., 0] * F[..., 7]],
                         [0, 0, 0, 0, 0, 0, 1, 0, 0,  0, 0, 0, F[..., 5], -F[..., 2], 0, -F[..., 4], F[..., 1], 0, F[..., 1] * F[..., 5] - F[..., 2] * F[..., 4]],
                         [0, 0, 0, 0, 0, 0, 0, 1, 0,  -F[..., 5], F[..., 2], 0, 0, 0, 0, F[..., 3], -F[..., 0], 0, F[..., 2] * F[..., 3] - F[..., 0] * F[..., 5]],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1,  F[..., 4], -F[..., 1], 0, -F[..., 3], F[..., 0], 0, 0, 0, 0, F[..., 0] * F[..., 4] - F[..., 1] * F[..., 3]],
                         ]).T


# -----------------------------------------------------------------------------
# Polyconvexification algortihms
# -----------------------------------------------------------------------------
def SVPC_LP(Phi, v, nref, r, return_grad=False):
    """
    Signed Singular Value Polyconvexification of function Phi evaluated at point v in R^d
    via the Linear Programming approach (SVPC LP)

    author: Timo Neumeier, timo.neumeier@uni-a.de

    Parameters
    ----------
    Phi : function, requires vectorized evaluation
    v : evaluation point
    nref : number of uniform lattice refinements
    r : discretisation radius for lattice
    return_grad : Boolean for returning derivaitive information
        
    Returns
    -------
    return :  Phi^{pc}_{delta,r}(v) (possibly DPhipc_{delta,r}(v))
    """

    # spatial dimension
    d = len(v)
    N = 2 ** nref + 1
    
    # discretization / grid
    Sigma = np.stack([x.flatten() for x in np.meshgrid(*d*[np.linspace(-r, r, N)])]).T
    
    # vector evaluation of the function
    Phi_Sigma = Phi(Sigma)
    
    # only keep finite valued points
    finitePhi = Phi_Sigma < np.inf
    Sigma, Phi_Sigma = Sigma[finitePhi, :], Phi_Sigma[finitePhi]

    # sigma is fixed, solve LP    
    result = SVPC_LP_core(Phi_Sigma, Sigma, v)
    
    # if no solution is found -> set the return value infinity
    if result.success:
        Phipc_v = result.fun
        if return_grad:
            DPhipc_v = result.eqlin.marginals[1:] @ Dminors(v)
    else:
        print(f"Problem SVPC LP for {v} infeasible, returning infinity")
        print(result)
        Phipc_v = np.inf
        if return_grad:
            DPhipc_v = np.zeros(d)
            return Phipc_v, DPhipc_v
        else:
            return Phipc_v
    
    if return_grad:
        return Phipc_v, DPhipc_v
    else:
        return Phipc_v


def SVPC_LP_core(Phi_Sigma, Sigma, v):
    """
    Core function for the SVPC LP
    Performs polyconvexification by LP solve
    """
    M = Sigma.shape[0]
    mSigma = minors(Sigma)
    # set up equality constraints of the optimization problem
    A_eq = np.hstack([np.ones((M, 1)), mSigma]).T
    b_eq = np.hstack([1, minors(v)])
    
    # Solve linear program
    result = scipy.optimize.linprog(Phi_Sigma, A_ub=None, b_ub=None,
                                    A_eq=A_eq,
                                    b_eq=b_eq,
                                    bounds=np.vstack((np.zeros(M), np.ones(M))).T,
                                    method='highs',
                                    callback=None, 
                                    options = {#'tol': 1e-10,
                                               'dual_feasibility_tolerance': 1e-9, 
                                               'primal_feasibility_tolerance': 1e-9, 
                                               'ipm_optimality_tolerance': 1e-9,
                                               }
                                    )
    
    return result 


def PC_LP(W, F, nref, r, N=0, nref_loc=0, return_grad=False, W_vec=None):
    """
    Polyconvexification of function W evaluated at point F in R^{d x d}
    via the Linear Programming approach (PC LP)
    
    author: Timo Neumeier, timo.neumeier@uni-a.de

    Parameters
    ----------
    W : function to polyconvexify, mapping matrices (vectorized) to scalars
    F : evaluation point F in R^{d x d}, ndarray, as a vector 
    nref : number of uniform lattice refinements, int
    r : discretisation radius, float
    return_grad : Boolean for returning derivaitive information, bool
                  The default is False. 
    W_vec : function, vectorized evaluation

    Returns
    -------
    return:  W^{pc}_{delta,r}(F) (possibly DWpc_{delta,r}(v))
    """
    d2 = F.shape[0]
    N = 2 ** nref + 1
    
    # discretization / grid
    X = np.stack([x.flatten() for x in np.meshgrid(*d2*[np.linspace(-r, r, N)])]).T
    
    if W_vec is None:
        W_X = np.array([W(X[i, :]) for i in range(X.shape[0])])
    else: 
        W_X = W_vec(X)
        
    # only keep finite valued points
    finiteW = W_X < np.inf
    X, W_X = X[finiteW, :], W_X[finiteW]
    
    result = PC_LP_core(W_X, X, F)
    
    # if no solution is found -> set the return value infinity
    if result.success:
        Wpc_F = result.fun
        if return_grad:
            DWpc_F = result.eqlin.marginals[1:] @ DMinors(F)
    else:
        Wpc_F = np.inf
        if return_grad:
            DWpc_F = np.full(d2, np.nan)
            
    if return_grad:
        return Wpc_F, DWpc_F
    else:
        return Wpc_F
    
    
def PC_LP_core(W_X, X, F):
    """
    Core function for the PC LP
    Performs polyconvexification by LP solve
    """
    M = X.shape[0]
    mX = Minors(X)
    
    # set up equality constraints of the optimization problem
    A_eq = np.hstack([np.ones((M, 1)), mX]).T
    b_eq = np.hstack([1, Minors(F)])

    # Solve linear program    
    result = scipy.optimize.linprog(W_X, A_ub=None, b_ub=None,
                                    A_eq=A_eq,
                                    b_eq=b_eq,
                                    bounds=np.vstack((np.zeros(M), np.ones(M))).T,
                                    method='highs',
                                    callback=None, 
                                    options = {#'tol': 1e-10,
                                               'dual_feasibility_tolerance': 1e-9, 
                                               'primal_feasibility_tolerance': 1e-9, 
                                               'ipm_optimality_tolerance': 1e-9,
                                               }
                                    )
    
    return result 


def SVPC_QH(Phi, v, nref, r):
    """
    Signed singular value polyconvexification of function Phi evaluated at point v in R^d
    based on the Quickhull approach

    author: Timo Neumeier, timo.neumeier@uni-a.de

    Parameters
    ----------
    Phi : function
    v : evaluation point
    nref : number of uniform lattice refinements
    r : discretisation radius

    Returns
    -------
    Phipc_v : Phi^{pc}_{delta,r}(v)
    """
    d = len(v)

    N = 2 ** nref + 1
    
    # discretization / grid
    Sigma = np.stack([x.flatten() for x in np.meshgrid(*d*[np.linspace(-r, r, N)])]).T

    # Phi_Sigma = np.array([Phi(Sigma[i, :]) for i in range(Sigma.shape[0])])
    Phi_Sigma = Phi(Sigma)
    # only keep finite valued points
    finitePhi = Phi_Sigma < np.inf
    X = np.vstack([minors(Sigma[finitePhi, :]).T, Phi_Sigma[finitePhi]]).T

    # computation of the convex hull of the lifted and evaluated point set
    hull = scipy.spatial.ConvexHull(X, incremental=False, qhull_options='QJ')
    
    lowerSimplices = hull.equations[:, -2] <= 0
    simplices = hull.simplices[lowerSimplices, :]

    # evaluation by interpolation: find the containing simplex and use barycentric coordinates
    idsimplex, b = findSimplex(simplices, hull.points[:, :-1], minors(v))
    
    # if the point is not contained in the mesh return infinity
    if idsimplex is not None:
        Phipc_v = b @ hull.points[simplices[idsimplex, :], -1]
        DPhipc_v = - hull.equations[idsimplex,:-2] @ Dminors(v) / hull.equations[idsimplex,-2] 
    else:
        Phipc_v = np.inf

    return Phipc_v
    

def PC_QH(W, F, nref, r, W_vec=None):
    """
    Polyconvexification of function W evaluated at point F in R^{d x d}
    based on the Quickhull approach
    
    author: Timo Neumeier, timo.neumeier@uni-a.de
    
    Parameters
    ----------
    W : function
    F : evaluation point
    nref : number of uniform lattice refinements
    r : discretisation radius
    W_vec : function, vectorized evaluation

    Returns
    -------
    Wpc_F :  W^{pc}_{delta,r}(F)

    """
    d2 = F.shape[0]
    N = 2 ** nref + 1
    
    # discretization / grid
    X = np.stack([x.flatten() for x in np.meshgrid(*d2*[np.linspace(-r, r, N)])]).T
    
    if W_vec is None:
        W_X = np.array([W(X[i, :]) for i in range(X.shape[0])])
    else: 
        W_X = W_vec(X)

    # only keep finite valued points
    finiteW = W_X < np.inf
    X = np.vstack([Minors(X[finiteW, :]).T, W_X[finiteW]]).T

    # computation of the convex hull of the lifted and evaluated point set
    hull = scipy.spatial.ConvexHull(X, incremental=False, qhull_options='QJ')
    lowerSimplices = hull.equations[:, -2] <= 0
    simplices = hull.simplices[lowerSimplices, :]

    # evaluation by interpolation: find the containing simplex and use barycentric coordinates
    idsimplex, b = findSimplex(simplices, hull.points[:, :-1], Minors(F))
    # if the point is not contained in the mesh return infinity
    if idsimplex is not None:
        Wpc_F = b @ hull.points[simplices[idsimplex, :], -1]
    else:
        Wpc_F = np.inf
    return Wpc_F


def findSimplex(simplices, points, x):
    """
    find index of simplex in simplices which contains the point x
    :param simplices: simplicial mesh
    :param points: coordinates of the mesh
    :param x: point
    :return: id of containing simplex, barycentric coordinates vector
    """
    n = len(x)
    idsimplex = None  # index of the containing simplex
    b = np.zeros(n + 1)  # barycentric coordinates
    for k in range(simplices.shape[0]):
        # solve linear system for the barycentric coordinates
        simplexvertices = points[simplices[k, :], :]
        A = np.hstack([np.ones((n + 1, 1)), simplexvertices]).T
        if np.linalg.cond(A) < 1e7:
            # computation of the barycentric coordinates
            # |1  1  1 |  |λ_1|   |1|
            # |x1 x2 x3|  |λ_2| = |x|
            # |y1 y2 y3|  |λ_3|   |y|
            b = np.linalg.solve(A, np.hstack([np.ones(1), x]))
            if abs(sum(abs(b)) - 1) < 1e-10:
                idsimplex = k
                return idsimplex, b
    return idsimplex, b



