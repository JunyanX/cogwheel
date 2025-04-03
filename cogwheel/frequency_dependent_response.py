import numpy as np
import matplotlib.pyplot as plt
import lal


DETECTORS = {'H': lal.CachedDetectors[lal.LHO_4K_DETECTOR],
             'L': lal.CachedDetectors[lal.LLO_4K_DETECTOR],
             'V': lal.CachedDetectors[lal.VIRGO_DETECTOR]}


DETECTOR_ARMS = {
    'H': (np.array([lal.LHO_4K_ARM_X_DIRECTION_X,
                    lal.LHO_4K_ARM_X_DIRECTION_Y,
                    lal.LHO_4K_ARM_X_DIRECTION_Z]),
          np.array([lal.LHO_4K_ARM_Y_DIRECTION_X,
                    lal.LHO_4K_ARM_Y_DIRECTION_Y,
                    lal.LHO_4K_ARM_Y_DIRECTION_Z])),
    'L': (np.array([lal.LLO_4K_ARM_X_DIRECTION_X,
                    lal.LLO_4K_ARM_X_DIRECTION_Y,
                    lal.LLO_4K_ARM_X_DIRECTION_Z]),
          np.array([lal.LLO_4K_ARM_Y_DIRECTION_X,
                    lal.LLO_4K_ARM_Y_DIRECTION_Y,
                    lal.LLO_4K_ARM_Y_DIRECTION_Z])),
    'V': (np.array([lal.VIRGO_ARM_X_DIRECTION_X,
                    lal.VIRGO_ARM_X_DIRECTION_Y,
                    lal.VIRGO_ARM_X_DIRECTION_Z]),
          np.array([lal.VIRGO_ARM_Y_DIRECTION_X,
                    lal.VIRGO_ARM_Y_DIRECTION_Y,
                    lal.VIRGO_ARM_Y_DIRECTION_Z]))}


def ra_dec_to_theta_phi(ra, dec, gmst):
    """ Convert from RA and DEC to polar coordinates on celestial sphere """
    phi = ra - gmst
    theta = np.pi / 2 - dec
    return theta, phi

def thetaphi_to_cart3d(theta, phi):
    """
    Return a unit vector (x, y, z) from spherical angles in radians.
    """
    return np.array([np.cos(phi) * np.sin(theta),
                     np.sin(phi) * np.sin(theta),
                     np.cos(theta)])

def get_polarization_tensor1(ra, dec, gmst, psi):
    """
    Calculate the polarization tensors epsilon_plus and epsilon_cross.
    
    Parameters
    ----------
    ra : float
        Right ascension in radians.
    dec : float
        Declination in radians.
    gmst : float
        Greenwich Mean Sidereal Time in radians.
    psi : float
        Polarization angle in radians.
    
    Returns
    -------
    epsilon_plus : np.ndarray
        The epsilon_plus polarization tensor (3x3 matrix).
    epsilon_cross : np.ndarray
        The epsilon_cross polarization tensor (3x3 matrix).
    """
    
    # Convert RA/DEC to theta/phi
    theta, phi = ra_dec_to_theta_phi(ra, dec, gmst)
    
    # Calculate the components of X and Y
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_psi = np.sin(psi)
    cos_psi = np.cos(psi)
    
    X = np.array([
        sin_phi * cos_psi - sin_psi * cos_phi * cos_theta,
        -cos_phi * cos_psi - sin_psi * sin_phi * cos_theta,
        sin_psi * sin_theta
    ])
    
    Y = np.array([
        -sin_phi * sin_psi - cos_psi * cos_phi * cos_theta,
        cos_phi * sin_psi - cos_psi * sin_phi * cos_theta,
        cos_psi * sin_theta
    ])
    
    # Calculate the polarization tensors
    epsilon_plus = np.outer(X, X) - np.outer(Y, Y)
    epsilon_cross = np.outer(X, Y) + np.outer(Y, X)
    
    return epsilon_plus, epsilon_cross

# def D_scalar(f, n_e, L):
#     """ Calculate the frequency-dependent detector scalar D(f, n_e) """
#     c = lal.C_SI
#     term1 = np.exp(-2j * np.pi * f * L / c) / (2 * (1 - n_e**2))
#     term2 = np.sinc(2 * np.pi * f * L / c) - n_e**2 * np.sinc(2 * np.pi * f * n_e * L / c)
#     term3 = -1j * n_e / (2 * np.pi * f * L / c) * (np.cos(2 * np.pi * f * L / c) - np.cos(2 * np.pi * f * n_e * L / c))
#     return term1 * (term2 + term3)


def D_scalar(f, n_e, L):
    """ Calculate the frequency-dependent detector scalar D(f, n_e) based on the new definition """
    if f == 0:
        return 0
        
    c = lal.C_SI
    term1 = (1 - np.exp(-2j * np.pi * f * (1 - n_e) * L / c)) / (1 - n_e)
    term2 = np.exp(-4j * np.pi * f * L / c) * (1 - np.exp(2j * np.pi * f * (1 + n_e) * L / c)) / (1 + n_e)
    
    return (c / (8 * np.pi * 1j * f * L)) * (term1 - term2)


def get_frequency_dependent_detector_tensor(detector_name, ra, dec, gmst, f, L=4000, detector_size=False):
    """
    Calculate the frequency-dependent detector tensor.
    
    Parameters
    ----------
    detector_name : str
        Name of the detector ('H', 'L', or 'V').
    ra : float
        Right ascension in radians.
    dec : float
        Declination in radians.
    gmst : float
        Greenwich Mean Sidereal Time in radians.
    f : float
        Frequency of the gravitational wave.
    L : float
        Arm length of the detector in meters.
    
    Returns
    -------
    D_f : np.ndarray
        The frequency-dependent detector tensor (3x3 matrix).
    """
    # Convert RA/DEC to theta/phi
    theta, phi = ra_dec_to_theta_phi(ra, dec, gmst)
    
    # Convert theta/phi to unit vector n
    n = thetaphi_to_cart3d(theta, phi)
    
    # Retrieve the arm directions for the given detector
    u, v = DETECTOR_ARMS[detector_name]

    
    # Calculate n_e as the dot product of n with arm directions
    n_e_u = np.dot(n, u)
    n_e_v = np.dot(n, v)

    if detector_size:
        # Calculate the frequency-dependent detector scalars
        D_u = D_scalar(f, n_e_u, L)
        D_v = D_scalar(f, n_e_v, L)
    else:
        D_u = 1/2
        D_v = 1/2
    
    # Calculate the detector tensor
    D_f = D_u * np.outer(u, u) - D_v * np.outer(v, v)
        
    
    return D_f



def get_antenna_response(f, ra, dec, gmst, psi, detector_name, L=4000, detector_size=False):
    """
    Calculate the frequency-dependent antenna response.
    
    Parameters
    ----------
    f : float
        Frequency of the gravitational wave.
    ra : float
        Right ascension in radians.
    dec : float
        Declination in radians.
    gmst : float
        Greenwich Mean Sidereal Time in radians.
    psi : float
        Polarization angle in radians.
    detector_name : str
        Name of the detector ('H', 'L', or 'V').
    L : float
        Arm length of the detector in meters (default is 4000).
    
    Returns
    -------
    F_plus : float
        The frequency-dependent antenna response for plus polarization.
    F_cross : float
        The frequency-dependent antenna response for cross polarization.
    """
    # Get the polarization tensors
    epsilon_plus, epsilon_cross = get_polarization_tensor1(ra, dec, gmst, psi)
    
    # Get the frequency-dependent detector tensor
    D_f = get_frequency_dependent_detector_tensor(detector_name, ra, dec, gmst, f, L, detector_size)
    
    # Calculate the antenna response for plus and cross polarizations
    F_plus = np.tensordot(epsilon_plus, D_f, axes=2)
    F_cross = np.tensordot(epsilon_cross, D_f, axes=2)
    
    return F_plus, F_cross