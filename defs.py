import numpy as np

A = 1.78
B = 1.326
C = 0.0202

def ior_exp1(z):
    """
    Simple exponential ice model.
    """
    def iorfunc(z):
        return A - (A - B) * np.exp(C * z)

    if isinstance(z, np.ndarray):
        iorvals = iorfunc(z)
        iorvals[z > 0] = 1.0
        return iorvals
    
    else:
        if z > 0:
            return 1.0
        else:
            return iorfunc(z)

def grad_ior_exp1(z):
    
    def grad_iorfunc(z):
        return (B - A) * C * np.exp(C * z)
    
    if isinstance(z, np.ndarray):
        grad_iorvals = grad_iorfunc(z)
        grad_iorvals[z > 0] = 0
        return grad_iorvals
    
    else:
        if z > 0:
            return 0
        else:
            return grad_iorfunc(z)      

def ior_exp3(z):
    """
    3-layer piecewise exponential ice model.
    """
    def iorfunc_snow(z):
        return 1.51188 - 0.271579 * np.exp(0.114553 * z)

    def iorfunc_firn(z):
        return 1.89957 - 0.529715 * np.exp(0.0129175 * z)

    def iorfunc_bubbly(z):
        return 1.77468 - 1.41573 * np.exp(0.0387882 * z)

    z1 = -14.9   # transition between snow and firn
    z2 = -80.5   # transition between firn and bubbly ice
    
    if isinstance(z, np.ndarray):
        snow_mask = np.argwhere(np.logical_and(z <= 0, z > z1))
        firn_mask = np.argwhere(np.logical_and(z <= z1, z > z2))
        bubbly_mask = np.argwhere(z <= z2)

        iorvals = np.zeros_like(z)    
        iorvals[snow_mask] = iorfunc_snow(z[snow_mask])
        iorvals[firn_mask] = iorfunc_firn(z[firn_mask])
        iorvals[bubbly_mask] = iorfunc_bubbly(z[bubbly_mask])
        iorvals[z > 0] = 1.0

        return iorvals
    
    else:
        if z > 0:
            return 1.0
        if z <= 0 and z > z1:
            return iorfunc_snow(z)
        if z <= z1 and z > z2:
            return iorfunc_firn(z)
        if z <= z2:
            return iorfunc_bubbly(z)

def grad_ior_exp3(z):

    def grad_iorfunc_snow(z):
        return - 0.271579 * 0.114553 * np.exp(0.114553 * z)

    def grad_iorfunc_firn(z):
        return - 0.529715 * 0.0129175 * np.exp(0.0129175 * z)

    def grad_iorfunc_bubbly(z):
        return - 1.41573 * 0.0387882 * np.exp(0.0387882 * z)

    z1 = -14.9   # transition between snow and firn
    z2 = -80.5   # transition between firn and bubbly ice

    if isinstance(z, np.ndarray):
        snow_mask = np.argwhere(np.logical_and(z <= 0, z > z1))
        firn_mask = np.argwhere(np.logical_and(z <= z1, z > z2))
        bubbly_mask = np.argwhere(z <= z2)

        grad_iorvals = np.zeros_like(z)    
        grad_iorvals[snow_mask] = grad_iorfunc_snow(z[snow_mask])
        grad_iorvals[firn_mask] = grad_iorfunc_firn(z[firn_mask])
        grad_iorvals[bubbly_mask] = grad_iorfunc_bubbly(z[bubbly_mask])
        grad_iorvals[z > 0] = 0.0
        
        return grad_iorvals
    
    else:
        if z > 0:
            return 0.0
        if z <= 0 and z > z1:
            return grad_iorfunc_snow(z)
        if z <= z1 and z > z2:
            return grad_iorfunc_firn(z)
        if z <= z2:
            return grad_iorfunc_bubbly(z)