import numpy as np

A = 1.78
B = 1.326
C = 0.0202

def ior_exp1(z):
    """
    Simple exponential ice model.

    Parameters
    __________
    z : float or 1D numpy array (n,)

    Returns
    _______
    n : float or 1D numpy array (n,)
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
    """
    Depth derivative of simple exponential ice model.

    Parameters
    __________
    z : float or 1D numpy array (n,)

    Returns
    _______
    dn/dz : float or 1D numpy array (n,)
    """
    
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
    RNO-G 3-layer piecewise exponential ice model.

    Parameters
    __________
    z : float or 1D numpy array (n,)

    Returns
    _______
    n : float or 1D numpy array (n,)
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

        iorvals = np.zeros_like(z, dtype = float)    
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
    """
    Depth derivative of RNO-G 3-layer piecewise exponential ice model.

    Parameters
    __________
    z : float or 1D numpy array (n,)

    Returns
    _______
    dn/dz : float or 1D numpy array (n,)
    """

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

        grad_iorvals = np.zeros_like(z, dtype = float)    
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
        
def get_ior_from_nuradio(ice):
    """
    Wrapper for compatibility between NuRadioMC ice model and Reconal refractive index functions.

    Parameters
    __________
    ice : NuRadioMC.utilities.medium_base.IceModel
        NuRadioMC ice model with implemented get_index_of_refraction and get_gradient_of_index_of_refraction functions.

    Returns
    _______
    ior : function
        Returns index of refraction at z-value (not 3d coordinate).
    grad_ior : function
        Returns dn/dz at z-value (not 3d coordinate).
    """
    def iorfunc(z):
        if isinstance(z, np.ndarray):
            pts = np.array((np.full_like(z, 0), np.full_like(z, 0), z)).swapaxes(0, 1)
            return ice.get_index_of_refraction(pts)
        else:
            return ice.get_index_of_refraction([0, 0, z])
    
    def grad_iorfunc(z):
        if isinstance(z, np.ndarray):
            pts = np.array((np.full_like(z, 0), np.full_like(z, 0), z)).swapaxes(0, 1)
            return ice.get_gradient_of_index_of_refraction(pts)[:, 2]
        else:
            return ice.get_gradient_of_index_of_refraction([0, 0, z])[2]

    return iorfunc, grad_iorfunc
