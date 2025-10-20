#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Signed singular value polyconvexification for isotropic energy densities based on
    [1] Timo Neumeier, Malte A. Peter, Daniel Peterseim, David Wiedemann.
        Computational polyconvexification of isotropic functions, MMS, 2024.

Simple convergence and runtime study for the 
    - Kohn-Strang-Dolzmann, 
    - Double well and
    - Saint Venant-Kirchhoff example.

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
import matplotlib.pyplot as plt
import time
import SVPC
import examples


# -----------------------------------------------------------------------------
# Main file for convergence test 
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    """
    main: 
        
    Simple convergence and runtime study for the
        - Kohn-Strang-Dolzmann, 
        - Double well and
        - Saint Venant-Kirchhoff example.
        
    Signed Singular Value Polyconvexification by
        - Quickhull (SVPC QH) 
        - Linear Programming (SVPC LP) 
    and Polyconvexification by
        - Quickhull (PC QH) 
        - Linear Programming (PC LP) 

    """

    # -------------------------------------------------------------------------    
    # set up example
    example = 'KSD' # 2D only
    #example = 'DW'
    #example = 'STVK'
    
    # -------------------------------------------------------------------------    
    # spatial dimension 
    d = 2
    
    if example == 'KSD':
        W, W_vec, Wpc = examples.W_KSD, examples.W_KSD_vec, examples.Wpc_KSD
        Phi, Phipc = examples.Phi_KSD, examples.Phipc_KSD
        DPhipc = examples.DPhipc_KSD
                
    elif example == 'DW': 
        W, W_vec, Wpc = examples.W_DW, examples.W_DW_vec, examples.Wpc_DW
        Phi, Phipc = examples.Phi_DW, examples.Phipc_DW 
        DPhipc = examples.DPhipc_DW
    
    elif example == 'STVK':
        W, W_vec, Wpc = examples.W_STVK, examples.W_STVK_vec, examples.Wpc_STVK
        Phi, Phipc = examples.Phi_STVK, examples.Phipc_STVK
        DPhipc = examples.DPhipc_STVK

    # -------------------------------------------------------------------------    
    # evaluation points 
    if example == 'KSD':
        A = np.array([[0.2, 0.1], [0.1, 0.3]])        
    elif example == 'DW': 
        if d == 2:
            A = np.array([[0.2, 0.1], [0.1, 0.3]])
        elif d == 3: 
            A = np.diag(np.array([0.3, 0.3, 0.3]))  
    elif example == 'STVK':
        if d == 2:
            A = np.diag(np.array([0.4, 1.4]))
        elif d == 3:
            A = np.diag(np.array([0.4, 1.05, 1.4]))

    # -------------------------------------------------------------------------    
    # signed singular value evaluation point            
    v = SVPC.signedSingularValues(A)

    # -------------------------------------------------------------------------
    methods = ['SVPC LP', 
               'SVPC QH', 
               'PC LP', 
               'PC QH']

    # -------------------------------------------------------------------------
    # set discretization parameters
    # r         -> discretisation radius 
    # max_nref  -> maximum number of uniform lattice refinements
    if example == 'KSD':
        r = 1.1
    elif example == 'DW': 
        r = 2.1
    elif example == 'STVK':
        r = 2.1
        
    # number of runs for averaging
    nr_runs = 1
    
    # -------------------------------------------------------------------------    
    # maximum refinement bounds for individual methods
    if d == 2:
        max_nref_method = {'SVPC LP' : 11,  # max ~12
                           'SVPC QH' : 10,  # max ~11 
                           'PC LP'   : 5,  # max ~5
                           'PC QH'   : 3,  # max ~3
                           }
    elif d == 3:
        max_nref_method = {'SVPC LP' : 7, # max ~8
                           'SVPC QH' : 3, # max ~3
                           'PC LP'   : 2, # max ~2 
                           'PC QH'   : 0, # infeasible
                           }
    
    max_nref = max(max_nref_method.values())
    nr_refs = np.arange(1, max_nref + 1)

    times = {method: np.full((max_nref, nr_runs), np.nan, dtype=float) for method in methods}
    errors = {method: np.full(max_nref, np.nan, dtype=float) for method in methods}
    errorsGrad = {method: np.full(max_nref, np.nan, dtype=float) for method in methods}
    
    # print output formatting
    fmt = f".{6}f"

    print(f"Started polyconvexification convergence study for {example} in d = {d} spatial dimensions")    

    for (i, nref) in enumerate(nr_refs):
        print(f"nref = {nref} / {max_nref}")
        # ---------------------------------------------------------------------
        # SVPC LP
        # ---------------------------------------------------------------------
        if nref <= max_nref_method['SVPC LP']:
            print(f"\tSVPC LP  \tnref = {nref} \tmeshsize = {(2 * r / 2 ** nref):{fmt}}", end='')
            for j in range(nr_runs):
                tic = time.perf_counter()
                Phipc_v, DPhipc_v = SVPC.SVPC_LP(Phi, v, nref, r, return_grad=True)
                toc = time.perf_counter()
                times['SVPC LP'][i, j] = toc - tic

            errors['SVPC LP'][i] = Phipc_v - Phipc(v)
            errorsGrad['SVPC LP'][i] = np.linalg.norm(DPhipc_v - DPhipc(v))
            
            print(f"\t\terr = {errors['SVPC LP'][i]:{fmt}} \terrGrad = {errorsGrad['SVPC LP'][i]:{fmt}}")
        # ---------------------------------------------------------------------

        # ---------------------------------------------------------------------
        # SVPC QH
        # ---------------------------------------------------------------------
        if nref <= max_nref_method['SVPC QH']:
            print(f"\tSVPC QH  \tnref = {nref} \tmeshsize = {(2 * r / 2 ** nref):{fmt}}", end='')
            for j in range(nr_runs):
                tic = time.perf_counter()
                Phipc_v = SVPC.SVPC_QH(Phi, v, nref, r)
                toc = time.perf_counter()
                times['SVPC QH'][i, j] = toc - tic
            
            errors['SVPC QH'][i] = Phipc_v - Phipc(v)     
            print(f"\t\terr = {errors['SVPC QH'][i]:{fmt}}")
        # ---------------------------------------------------------------------

        # ---------------------------------------------------------------------
        # PC LP
        # ---------------------------------------------------------------------
        if nref <= max_nref_method['PC LP']: 
            print(f"\tPC LP    \tnref = {nref} \tmeshsize = {(2 * r / 2 ** nref):{fmt}}", end='')
            for j in range(nr_runs):
                tic = time.perf_counter()
                Wpc_F, DWpc_F = SVPC.PC_LP(W, A.flatten(), nref, r, return_grad=True, W_vec=W_vec)
                toc = time.perf_counter()
                times['PC LP'][i, j] = toc - tic

            errors['PC LP'][i] = Wpc_F - Wpc(A)
            print(f"\t\terr = {errors['PC LP'][i]:{fmt}}")
        # ---------------------------------------------------------------------

        # ---------------------------------------------------------------------
        # PC QH
        # ---------------------------------------------------------------------
        if nref <= max_nref_method['PC QH']:
            print(f"\tPC QH    \tnref = {nref} \tmeshsize = {(2 * r / 2 ** nref):{fmt}}", end='') 
            for j in range(nr_runs):
                tic = time.perf_counter()
                Wpc_F = SVPC.PC_QH(W, A.flatten(), nref, r, W_vec=W_vec)
                toc = time.perf_counter()
                times['PC QH'][i, j] = toc - tic

            errors['PC QH'][i] = Wpc_F - Wpc(A)
            print(f"\t\terr = {errors['PC QH'][i]:{fmt}}")
        # ---------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # Post processing and plotting
    # -------------------------------------------------------------------------

    # Lattice size for convergence plots
    deltas = 2 * r / (2 ** nr_refs)
    
    # -------------------------------------------------------------------------
    # error vs meshsize
    plt.figure(1)
    plt.title("Convergence SVPC")
    for method in methods:
        plt.loglog(deltas, errors[method], marker='x', label=method)
    
    # derivative error vs meshsize
    for method in ['SVPC LP']:
        plt.loglog(deltas, errorsGrad[method], marker='v', label=method + r" $\Phi_{\delta}^{pc}(\nu) - \Phi^{pc}(\nu)$")

    plt.loglog(deltas, deltas**1 / 1, ':', color='gray', marker='', label=r"$c \delta^{1}$")
    plt.loglog(deltas, deltas**2 / 1, '--', color='gray', marker='', label=r"$c \delta^{2}$")
    plt.loglog(deltas, deltas**3 / 1, '-.', color='gray', marker='', label=r"$c \delta^{3}$")
    if example == 'DW':
        plt.loglog(deltas, deltas**4 / 1, linestyle=(0, (3, 1, 1, 1)), color='gray', marker='', label=r"$c \delta^{4}$")
            
    plt.grid(which='minor', color='lightgray')
    plt.grid(which='major', color='dimgray')
    plt.xlabel(r"$\delta$")
    plt.ylabel(r"$\Phi_{\delta}^{pc}(\nu) - \Phi^{pc}(\nu)$ / $W_{\delta}^{pc}(\nu) - W^{pc}(\nu)$")
    plt.legend()
    plt.show()
    # -------------------------------------------------------------------------
        
    # -------------------------------------------------------------------------
    # average runntimes vs meshsize    
    avg_times = {method : np.sum(times[method], 1) / nr_runs for method in methods}
    plt.figure(2)
    plt.title("Runntime SVPC")
    for method in methods:
        plt.loglog(deltas, avg_times[method], marker='x', label=method)

    plt.loglog(deltas, deltas**(-1) / 1000, ':', color='gray', marker='', label=r"$c \delta^{-1}$")
    plt.loglog(deltas, deltas**(-2) / 1000, '--', color='gray', marker='', label=r"$c \delta^{-2}$")
    plt.loglog(deltas, deltas**(-3) / 1000, '-.', color='gray', marker='', label=r"$c \delta^{-3}$")
    plt.loglog(deltas, deltas**(-4) / 1000, linestyle=(0, (3, 1, 1, 1)), color='gray', marker='', label=r"$c \delta^{-4}$")

    plt.grid(which='minor', color='lightgray')
    plt.grid(which='major', color='dimgray')
    plt.xlabel(r"$\delta$")
    plt.ylabel(r"time ($s$)")
    plt.legend()
    plt.show()
    # -------------------------------------------------------------------------
    
# -----------------------------------------------------------------------------
