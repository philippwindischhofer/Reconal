import numpy as np
import matplotlib.pyplot as plt

A = 1.78
B = 1.326
C = 0.0202

def ior_exp1(z):
    return A - (A - B) * np.exp(C * z)

def grad_ior_exp1(z):
    return (B - A) * C * np.exp(C * z)      

def ior_exp3(z):
    
    def iorfunc_snow(z):
        return 1.52737 - 0.298415 * np.exp(0.107158 * z)

    def iorfunc_firn(z):
        return 1.89275 - 0.521529 * np.exp(0.0136059 * z)

    def iorfunc_bubbly(z):
        return 1.77943 - 1.576 * np.exp(0.0403732 * z)

    z1 = -14.9   # transition between snow and firn
    z2 = -80.5   # transition between firn and bubbly ice
    
    if isinstance(z, np.ndarray):
        snow_mask = np.argwhere(z > z1)
        firn_mask = np.argwhere(np.logical_and(z <= z1, z > z2))
        bubbly_mask = np.argwhere(z <= z2)

        iorvals = np.zeros_like(z)    
        iorvals[snow_mask] = iorfunc_snow(z[snow_mask])
        iorvals[firn_mask] = iorfunc_firn(z[firn_mask])
        iorvals[bubbly_mask] = iorfunc_bubbly(z[bubbly_mask])
        
        return iorvals
    
    else:
        if z > z1:
            return iorfunc_snow(z)
        if z <= z1 and z > z2:
            return iorfunc_firn(z)
        if z <= z2:
            return iorfunc_bubbly(z)

def grad_ior_exp3(z):

    def grad_iorfunc_snow(z):
        return - 0.298415 * 0.107158 * np.exp(0.107158 * z)

    def grad_iorfunc_firn(z):
        return - 0.521529 * 0.0136059 * np.exp(0.0136059 * z)

    def grad_iorfunc_bubbly(z):
        return - 1.576 * 0.0403732 * np.exp(0.0403732 * z)

    z1 = -14.9   # transition between snow and firn
    z2 = -80.5   # transition between firn and bubbly ice

    if isinstance(z, np.ndarray):
        snow_mask = np.argwhere(z > z1)
        firn_mask = np.argwhere(np.logical_and(z <= z1, z > z2))
        bubbly_mask = np.argwhere(z <= z2)

        grad_iorvals = np.zeros_like(z)    
        grad_iorvals[snow_mask] = grad_iorfunc_snow(z[snow_mask])
        grad_iorvals[firn_mask] = grad_iorfunc_firn(z[firn_mask])
        grad_iorvals[bubbly_mask] = grad_iorfunc_bubbly(z[bubbly_mask])
        
        return grad_iorvals
    
    else:
        if z > z1:
            return grad_iorfunc_snow(z)
        if z <= z1 and z > z2:
            return grad_iorfunc_firn(z)
        if z <= z2:
            return grad_iorfunc_bubbly(z)