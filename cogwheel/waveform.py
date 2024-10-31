"""Generate strain waveforms and project them onto detectors."""
import itertools
from collections import defaultdict, namedtuple
import numpy as np

import lal
import lalsimulation

from cogwheel import gw_utils
from cogwheel import utils
from scipy.interpolate import interp1d
from cogwheel.frequency_dependent_response import get_antenna_response

from cogwheel.gw_utils import get_geocenter_delays, tgps_to_gmst
from cogwheel.skyloc_angles import ra_to_lon



ZERO_INPLANE_SPINS = {'s1x_n': 0.,
                      's1y_n': 0.,
                      's2x_n': 0.,
                      's2y_n': 0.}

DEFAULT_PARS = {**ZERO_INPLANE_SPINS,
                's1z': 0.,
                's2z': 0.,
                'l1': 0.,
                'l2': 0.}

FORCE_NNLO_ANGLES = (
    ('SimInspiralWaveformParamsInsertPhenomXPrecVersion', 102),)


def compute_hplus_hcross(f, par_dic, approximant: str,
                         harmonic_modes=None, lal_dic=None):
    """
    Generate frequency domain waveform using LAL.
    Return hplus, hcross evaluated at f.

    Parameters
    ----------
    f: 1d array of type float
        Frequency array in Hz

    par_dic: dict
        Source parameters. Needs to have these keys:
            * m1, m2: component masses (Msun)
            * d_luminosity: luminosity distance (Mpc)
            * iota: inclination (rad)
            * phi_ref: phase at reference frequency (rad)
            * f_ref: reference frequency (Hz)
        plus, optionally:
            * s1x_n, s1y_n, s1z, s2x_n, s2y_n, s2z: dimensionless spins
            * l1, l2: dimensionless tidal deformabilities

    approximant: str
        Approximant name.

    harmonic_modes: list of 2-tuples with (l, m) pairs, optional
        Which (co-precessing frame) higher-order modes to include.

    lal_dic: LALDict, optional
        Contains special approximant settings.
    """

    # Parameters ordered for lalsimulation.SimInspiralChooseFDWaveformSequence
    lal_params = [
        'phi_ref', 'm1_kg', 'm2_kg', 's1x', 's1y', 's1z', 's2x', 's2y', 's2z',
        'f_ref', 'd_luminosity_meters', 'iota', 'lal_dic', 'approximant', 'f']

    par_dic = DEFAULT_PARS | par_dic

    # Transform inplane spins to LAL's coordinate system.
    inplane_spins_xy_n_to_xy(par_dic)

    # SI unit conversions
    par_dic['d_luminosity_meters'] = par_dic['d_luminosity'] * 1e6 * lal.PC_SI
    par_dic['m1_kg'] = par_dic['m1'] * lal.MSUN_SI
    par_dic['m2_kg'] = par_dic['m2'] * lal.MSUN_SI

    par_dic['lal_dic'] = lal_dic or lal.CreateDict()
    # Tidal parameters
    lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(
        par_dic['lal_dic'], par_dic['l1'])
    lalsimulation.SimInspiralWaveformParamsInsertTidalLambda2(
        par_dic['lal_dic'], par_dic['l2'])
    # Higher-mode parameters
    if harmonic_modes is not None:
        mode_array = lalsimulation.SimInspiralCreateModeArray()
        for l, m in harmonic_modes:
            lalsimulation.SimInspiralModeArrayActivateMode(mode_array, l, m)
        lalsimulation.SimInspiralWaveformParamsInsertModeArray(
            par_dic['lal_dic'], mode_array)

    par_dic['approximant'] = lalsimulation.GetApproximantFromString(
        approximant)

    f0_is_0 = f[0] == 0  # In this case we will set h(f=0) = 0
    par_dic['f'] = lal.CreateREAL8Sequence(len(f))
    par_dic['f'].data = f
    if f0_is_0:
        par_dic['f'].data[0] = par_dic['f'].data[1]

    try:
        hplus, hcross = lalsimulation.SimInspiralChooseFDWaveformSequence(
            *[par_dic[par] for par in lal_params])
    except Exception:
        print('Error while calling LAL at these parameters:', par_dic)
        raise
    hplus_hcross = np.stack([hplus.data.data, hcross.data.data])
    if f0_is_0:
        hplus_hcross[:, 0] = 0

    return hplus_hcross


def compute_hplus_hcross_by_mode(f, par_dic, approximant: str,
                                 harmonic_modes, lal_dic=None):
    """
    Return dictionary of the form {(l, m): h_lm} with the contribution
    of each harmonic mode to hplus, hcross.

    Parameters
    ----------
    f: 1d array of type float
        Frequency array in Hz

    par_dic: dict
        Source parameters. Needs to have these keys:
            * m1, m2: component masses (Msun)
            * d_luminosity: luminosity distance (Mpc)
            * iota: inclination (rad)
            * phi_ref: phase at reference frequency (rad)
            * f_ref: reference frequency (Hz)
        plus, optionally:
            * s1x_n, s1y_n, s1z, s2x_n, s2y_n, s2z: dimensionless spins
            * l1, l2: dimensionless tidal deformabilities

    approximant: str
        Approximant name.

    harmonic_modes: list of 2-tuples with (l, m) pairs
        Which (co-precessing frame) higher-order modes to include.

    lal_dic: LALDict, optional
        Contains special approximant settings.
    """
    return {mode: compute_hplus_hcross(f, par_dic, approximant,
                                       harmonic_modes=[mode], lal_dic=lal_dic)
            for mode in harmonic_modes}


Approximant = namedtuple(
    'Approximant',
    ('harmonic_modes', 'aligned_spins', 'tides', 'hplus_hcross_by_mode_func'),
    defaults=([(2, 2)], True, False, compute_hplus_hcross_by_mode))

APPROXIMANTS = {
    'IMRPhenomD_NRTidalv2': Approximant(tides=True),
    'IMRPhenomD': Approximant(),
    'IMRPhenomXPHM': Approximant(harmonic_modes=[(2, 2), (2, 1), (3, 3),
                                                 (3, 2), (4, 4)],
                                 aligned_spins=False),
    'IMRPhenomXAS': Approximant(),
    }


def inplane_spins_xy_n_to_xy(par_dic):
    """
    Rotate inplane spins (s1x_n, s1y_n) and (s2x_n, s2y_n) by an angle
    `-phi_ref` to get (s1x, s1y), (s2x, s2y).
    `par_dic` needs to have keys 's1x_n', 's1y_n', 's2x_n', 's2y_n'.
    Entries for 's1x', 's1y', 's2x', 's2y' will be added.

    `x_n`, `y_n` are axes perpendicular to the orbital angular momentum
    `L`, so that the line of sight `N` lies in the y-z plane, i.e.
        N = (0, sin(iota), cos(iota))
    in the (x_n, y_n, z) system.
    `x`, `y` are axes perpendicular to the orbital angular momentum `L`,
    so that the orbital separation is the x direction.
    The two systems coincide when `phi_ref=0`.
    """
    sin_phi_ref = np.sin(par_dic['phi_ref'])
    cos_phi_ref = np.cos(par_dic['phi_ref'])
    rotation = np.array([[cos_phi_ref, sin_phi_ref],
                         [-sin_phi_ref, cos_phi_ref]])

    ((par_dic['s1x'], par_dic['s2x']),
     (par_dic['s1y'], par_dic['s2y'])
        ) = rotation.dot(((par_dic['s1x_n'], par_dic['s2x_n']),
                          (par_dic['s1y_n'], par_dic['s2y_n'])))


def inplane_spins_xy_to_xy_n(par_dic):
    """
    Rotate inplane spins (s1x, s1y) and (s2x, s2y) by an angle
    `phi_ref` to get (s1x_n, s1y_n), (s2x_n, s2y_n).
    `par_dic` needs to have keys 's1x', 's1y', 's2x', 's2y'.
    Entries for 's1x_n', 's1y_n', 's2x_n', 's2y_n' will be added.

    `x_n`, `y_n` are axes perpendicular to the orbital angular momentum
    `L`, so that the line of sight `N` lies in the y-z plane, i.e.
        N = (0, sin(iota), cos(iota))
    in the (x_n, y_n, z) system.
    `x`, `y` are axes perpendicular to the orbital angular momentum `L`,
    so that the orbital separation is the x direction.
    The two systems coincide when `phi_ref=0`.
    """
    sin_phi_ref = np.sin(par_dic['phi_ref'])
    cos_phi_ref = np.cos(par_dic['phi_ref'])
    rotation = np.array([[cos_phi_ref, -sin_phi_ref],
                         [sin_phi_ref, cos_phi_ref]])

    ((par_dic['s1x_n'], par_dic['s2x_n']),
     (par_dic['s1y_n'], par_dic['s2y_n'])
        ) = rotation.dot(((par_dic['s1x'], par_dic['s2x']),
                          (par_dic['s1y'], par_dic['s2y'])))


def within_bounds(par_dic: dict) -> bool:
    """
    Return whether parameters in `par_dic` are within physical bounds.
    """
    return (all(par_dic[positive] >= 0
                for positive in {'m1', 'm2', 'd_luminosity', 'l1', 'l2', 'iota'
                                }.intersection(par_dic))
            and np.all(np.linalg.norm(
                [(par_dic['s1x_n'], par_dic['s1y_n'], par_dic['s1z']),
                 (par_dic['s2x_n'], par_dic['s2y_n'], par_dic['s2z'])],
                axis=1) <= 1)
            and par_dic.get('iota', 0) <= np.pi
            and np.abs(par_dic.get('dec', 0)) <= np.pi/2
           )

def get_phase_from_frequency_domain_waveform(hp):
    """
    Calculates the phase from the frequency-domain gravitational wave polarization h+.

    Parameters:
    - hp (numpy array): Array containing the plus polarization (h+) in the frequency domain.

    Returns:
    - phase_unwrapped_hp (numpy array): Unwrapped phase of the plus polarization (h+).
    """

    # Calculate the phase for h+
    phase_hp = np.angle(hp)

    # Unwrap the phase to ensure continuity
    phase_unwrapped_hp = np.unwrap(phase_hp)

    return phase_unwrapped_hp



def generate_time_frequency_mapping(hp, frequencies):
    """
    Generates the time-frequency mapping for a gravitational wave signal using the frequency domain waveform.

    Parameters:
    - hp (numpy array): Array containing the plus polarization (h+) in the frequency domain.
    - frequencies (numpy array): Array of frequency values.

    Returns:
    - time_hp (numpy array): Array of time values corresponding to the instantaneous frequency for h+.
    """
    
    # Get the unwrapped phase for h+ from the frequency domain waveform
    phase_unwrapped_hp = get_phase_from_frequency_domain_waveform(hp)

    # Calculate the instantaneous time as the derivative of the phase with respect to frequency for h+
    time_hp = -np.gradient(phase_unwrapped_hp) / (2.0 * np.pi * np.gradient(frequencies))

    # Return the time array for h+
    return time_hp


def extract_monotonic_increasing_segment(time, frequency, min_freq_step=0.1):
    """
    Extracts the longest monotonic increasing part of the time-frequency mapping based on the frequency span,
    after discarding the part where time is greater than 0, and using frequency steps to check monotonicity.

    Parameters:
    - time (numpy array): Array of time values.
    - frequency (numpy array): Array of frequency values.
    - min_freq_step (float): Minimum allowed frequency step over which to check monotonicity (default is 0.1 Hz).

    Returns:
    - time_mono (numpy array): Time array corresponding to the longest monotonic increasing segment.
    - frequency_mono (numpy array): Frequency array corresponding to the longest monotonic increasing segment.
    """

    # Discard any part where time > 0
    mask = time <= 0
    time = time[mask]
    frequency = frequency[mask]

    max_freq_span = 0
    max_start = 0
    max_end = 0
    current_start = 0
    current_end = 0

    n = len(frequency)  # Length of frequency array
    i = 0

    # Step through the frequency array, using np.searchsorted to find the next index
    while i < n - 1:
        # Find the next index where frequency is at least min_freq_step larger than frequency[i]
        next_idx = np.searchsorted(frequency, frequency[i] + min_freq_step, side='right')

        # Ensure next_idx doesn't exceed the array bounds
        if next_idx >= n:
            break

        # Check if the time is still increasing
        if time[next_idx] > time[i]:
            current_end = next_idx
        else:
            # End the current segment and check its frequency span
            current_freq_span = frequency[current_end] - frequency[current_start]
            if current_freq_span > max_freq_span:
                max_freq_span = current_freq_span
                max_start = current_start
                max_end = current_end

            # Start a new segment
            current_start = next_idx
            current_end = next_idx

        # Move the index forward
        i = next_idx

    # Final check for the last segment
    current_freq_span = frequency[current_end] - frequency[current_start]
    if current_freq_span > max_freq_span:
        max_start = current_start
        max_end = current_end

    # Slice the time and frequency arrays to keep only the longest frequency span segment
    time_mono = time[max_start:max_end + 1]
    frequency_mono = frequency[max_start:max_end + 1]

    return time_mono, frequency_mono



def generate_interpolation_function(hp, frequencies):
    """
    Generates an interpolation function for the time-frequency mapping of the h+ polarization.

    Parameters:
    - hp (numpy array): Array containing the plus polarization (h+) in the frequency domain.
    - frequencies (numpy array): Array of frequency values.

    Returns:
    - interp_func_hp (scipy.interpolate.interp1d): Interpolation function for h+ mapping frequency to time.
    """
    
    # Generate the time-frequency mapping for h+
    time_hp = generate_time_frequency_mapping(hp, frequencies)

    # Extract the longest monotonic increasing segment for h+
    time_mono_hp, frequency_mono_hp = extract_monotonic_increasing_segment(time_hp, frequencies)

    # Generate interpolation function for h+
    interp_func_hp = interp1d(frequency_mono_hp, time_mono_hp, kind='linear', fill_value="extrapolate")

    return interp_func_hp


class WaveformGenerator(utils.JSONMixin):
    """
    Class that provides methods for generating frequency domain
    waveforms, in terms of `hplus, hcross` or projected onto detectors.
    "Fast" and "slow" parameters are distinguished: the last waveform
    calls are cached and can be computed fast when only fast parameters
    are changed.
    The attribute `n_cached_waveforms` can be used to control how many
    waveform calls to save in the cache.
    The boolean attribute `disable_precession` can be set to ignore
    inplane spins.
    """
    params = sorted(['d_luminosity', 'dec', 'f_ref', 'iota', 'l1', 'l2',
                     'm1', 'm2', 'psi', 'ra', 's1x_n', 's1y_n', 's1z',
                     's2x_n', 's2y_n', 's2z', 't_geocenter', 'phi_ref'])

    fast_params = sorted(['d_luminosity', 'dec', 'psi', 'ra', 't_geocenter',
                          'phi_ref'])
    slow_params = sorted(set(params) - set(fast_params))

    _projection_params = sorted(['dec', 'psi', 'ra', 't_geocenter'])
    _waveform_params = sorted(set(params) - set(_projection_params))
    polarization_params = sorted(set(params) - {'psi'})

    def __init__(self, detector_names, tgps, tcoarse, approximant,
                 harmonic_modes=None, disable_precession=False,
                 n_cached_waveforms=1, lalsimulation_commands=()):
        super().__init__()

        if approximant == 'IMRPhenomXODE':
            from cogwheel.waveform_models import xode as _  # TODO more elegant

        self.detector_names = tuple(detector_names)
        self.tgps = tgps
        self.tcoarse = tcoarse
        self._approximant = approximant
        self.harmonic_modes = harmonic_modes
        self.disable_precession = disable_precession
        self.lalsimulation_commands = lalsimulation_commands
        self.n_cached_waveforms = n_cached_waveforms

        self.n_slow_evaluations = 0
        self.n_fast_evaluations = 0

        self._cached_f = None
        self._cached_t = None
        self._cached_fp_fc = None

    @classmethod
    def from_event_data(cls, event_data, approximant,
                        harmonic_modes=None, disable_precession=False,
                        n_cached_waveforms=1, lalsimulation_commands=()):
        """
        Constructor that takes `detector_names`, `tgps` and `tcoarse`
        from an instance of `data.EventData`.
        """
        return cls(event_data.detector_names, event_data.tgps,
                   event_data.tcoarse, approximant, harmonic_modes,
                   disable_precession, n_cached_waveforms,
                   lalsimulation_commands)

    @property
    def approximant(self):
        """String with waveform approximant name."""
        return self._approximant

    @approximant.setter
    def approximant(self, approximant: str):
        """
        Set `approximant` and reset `harmonic_modes` per
        `APPROXIMANTS[approximant].harmonic_modes`; print a warning that
        this was done.
        Raise `ValueError` if `APPROXIMANTS` does not contain the
        requested approximant.
        """
        if approximant not in APPROXIMANTS:
            raise ValueError(f'Add {approximant} to `waveform.APPROXIMANTS`.')
        self._approximant = approximant

        old_harmonic_modes = self.harmonic_modes
        self.harmonic_modes = None
        if self.harmonic_modes != old_harmonic_modes:
            print(f'`approximant` changed to {approximant!r}, setting'
                  f'`harmonic_modes` to {self.harmonic_modes}.')
        utils.clear_caches()

    @property
    def harmonic_modes(self):
        """List of `(l, m)` pairs."""
        return self._harmonic_modes

    @harmonic_modes.setter
    def harmonic_modes(self, harmonic_modes):
        """
        Set `self._harmonic_modes` implementing defaults based on the
        approximant, this requires hardcoding which modes are
        implemented by each approximant.
        Also set `self._harmonic_modes_by_m` with a dictionary whose
        keys are `m` and whose values are a list of `(l, m)` tuples with
        that `m`.
        """
        if harmonic_modes is None:
            harmonic_modes = APPROXIMANTS[self.approximant].harmonic_modes
        else:
            harmonic_modes = [tuple(mode) for mode in harmonic_modes]
        self._harmonic_modes = harmonic_modes

        self._harmonic_modes_by_m = defaultdict(list)
        for l, m in self._harmonic_modes:
            self._harmonic_modes_by_m[m].append((l, m))
        utils.clear_caches()

    @property
    def m_arr(self):
        """Int array of m harmonic mode numbers."""
        return np.fromiter(self._harmonic_modes_by_m, int)

    @property
    def n_cached_waveforms(self):
        """Nonnegative integer, number of cached waveforms."""
        return self._n_cached_waveforms

    @n_cached_waveforms.setter
    def n_cached_waveforms(self, n_cached_waveforms):
        self.cache = [{'slow_par_vals': np.array(np.nan),
                       'approximant': None,
                       'f': None,
                       'harmonic_modes_by_m': {},
                       'hplus_hcross_0': None,
                       'lalsimulation_commands': ()}
                      for _ in range(n_cached_waveforms)]
        self._n_cached_waveforms = n_cached_waveforms

    @property
    def lalsimulation_commands(self):
        """
        Tuple of `(key, value)` where `key` is the name of a
        `lalsimulation` function and `value` is its second argument,\
        after `lal_dic`.
        """
        return self._lalsimulation_commands

    @lalsimulation_commands.setter
    def lalsimulation_commands(self, lalsimulation_commands):
        self._lalsimulation_commands = lalsimulation_commands
        utils.clear_caches()

    def get_m_mprime_inds(self):
        """
        Return two lists of integers, these zipped are pairs (i, j) of
        indices with j >= i that run through the number of m modes.
        """
        return map(list, zip(*itertools.combinations_with_replacement(
            range(len(self._harmonic_modes_by_m)), 2)))

    def get_strain_at_detectors(self, f, par_dic, by_m=False, vary_polarization = False, doppler = False, f_lower = 2, f_higher = 10, use_cached = False):
        """
        Get strain measurable at detectors with time-dependent antenna response.
    
        Parameters
        ----------
        f: 1d array of frequencies [Hz]
        par_dic: parameter dictionary per `WaveformGenerator.params`.
        by_m: bool, whether to return waveform separated by `m`
              harmonic mode (summed over `l`), or already summed.
        time_varying: bool, whether to account for time-varying antenna response.
    
        Return
        ------
        Array of shape (n_m?, n_detectors, n_frequencies) with strain at
        detector, `n_m` is there only if `by_m=True`.
        """
    
        # shape: (n_m?, 2, n_detectors, n_frequencies)
        hplus_hcross_at_detectors = self.get_hplus_hcross_at_detectors(f, par_dic, by_m, doppler)
        n_detectors = len(self.detector_names)

        if vary_polarization or doppler:
            if self._cached_t is None or use_cached:
                self._cached_t = self.time_series(f, par_dic, by_m, f_lower, f_higher)

    
        if vary_polarization:

            ts = self._cached_t

            
            if self._cached_fp_fc is None or use_cached:
                n_m = hplus_hcross_at_detectors.shape[0]
                fplus_fcross = self.compute_fplus_fcross(f, par_dic['ra'], par_dic['dec'], par_dic['psi'])
                self._cached_fp_fc = fplus_fcross

            else:
                fplus_fcross = self._cached_fp_fc

                
            strain = np.einsum('...pdf, ...pdf -> ...df', fplus_fcross, hplus_hcross_at_detectors)
    
    
            return strain
    
    
        # fplus_fcross shape: (2, n_detectors)
        fplus_fcross = gw_utils.fplus_fcross(
            self.detector_names, par_dic['ra'], par_dic['dec'], par_dic['psi'],
            self.tgps)

        # Detector strain (n_m?, n_detectors, n_frequencies)
        return np.einsum('pd, ...pdf -> ...df',
                         fplus_fcross, hplus_hcross_at_detectors)

    def compute_fplus_fcross(self, f, ra, dec, psi):

        n_detectors = len(self.detector_names)
        fplus_fcross = np.zeros((2, n_detectors, len(f)), dtype = complex)
        
        for d in range(n_detectors):
        
            for i, frequency in enumerate(f):
                delta_t = self._cached_t[i]
                gmst = tgps_to_gmst(self.tgps + delta_t)
        
                # Use the new get_antenna_response function
                fplus, fcross = get_antenna_response(frequency, ra, dec, gmst, psi, self.detector_names[d])
        
                # Assign the antenna responses
                fplus_fcross[0, d, i] = fplus # F+ for hp
                fplus_fcross[1, d, i] = fcross # Fx for hc
    
        return fplus_fcross



    def get_hplus_hcross_at_detectors(self, f, par_dic, by_m=False, doppler = False):
        """
        Return plus and cross polarizations with time shifts applied, including Doppler shift.
        
        Parameters
        ----------
        f : 1d array of frequencies [Hz]
        par_dic : dict
            Parameter dictionary per `WaveformGenerator.params`.
        by_m : bool
            Whether to return waveform separated by `m` harmonic mode (summed over `l`), or already summed.
        time_varying : bool
            Whether to apply time-dependent Doppler shift.
    
        Returns
        -------
        Array of shape (n_m?, 2, n_detectors, n_frequencies) with hplus, hcross at detector, `n_m` is there only if `by_m=True`.
        """
        
        waveform_par_dic = {par: par_dic[par] for par in self._waveform_params}
    
        # hplus_hcross shape: (n_m?, 2, n_frequencies)
        hplus_hcross = self.get_hplus_hcross(f, waveform_par_dic, by_m)
    
        # shifts shape: (n_detectors, n_frequencies)
        if not np.array_equal(f, self._cached_f):
            self._get_shifts.cache_clear()
            self._cached_f = f

            
        shifts = self._get_shifts(par_dic['ra'], par_dic['dec'], par_dic['t_geocenter'], doppler)
    
        # hplus, hcross (n_m?, 2, n_detectors, n_frequencies)
        return np.einsum('...pf, df -> ...pdf', hplus_hcross, shifts)



    @utils.lru_cache(maxsize=16)
    def _get_shifts(self, ra, dec, t_geocenter, doppler):
        """Return (n_detectors, n_frequencies) array with e^(-2 i f t_det)."""
        
        if not doppler:
            gmst = tgps_to_gmst(t_geocenter)
            lon = ra_to_lon(ra, gmst)
            lat = dec
    
            time_delays = gw_utils.get_geocenter_delays(
                self.detector_names, lat, lon)
            
            return np.exp(-2j*np.pi * self._cached_f
                      * (self.tcoarse
                         + t_geocenter
                         + time_delays[:,np.newaxis]))

        lat = dec
        time_delays = []
        
        for t in self._cached_t:
            gmst = tgps_to_gmst(t_geocenter + t)
            lon = ra_to_lon(ra, gmst)
 
            time_delays.append(gw_utils.get_geocenter_delays(self.detector_names, lat, lon).squeeze())

        time_delays = np.array(time_delays)
            
            
        return np.exp(-2j*np.pi * self._cached_f
                      * (self.tcoarse
                         + t_geocenter
                         + time_delays)).reshape(1, -1)

    def get_hplus_hcross(self, f, waveform_par_dic, by_m=False):
        """
        Return hplus, hcross waveform strain.
        Note: inplane spins will be zeroized if `self.disable_precession`
              is `True`.

        Parameters
        ----------
        f: 1d array of frequencies [Hz]
        waveform_par_dic: dictionary per
                          `WaveformGenerator._waveform_params`.
        by_m: bool, whether to return harmonic modes separately by m (l
              summed over) or all modes already summed over.

        Return
        ------
        array with (hplus, hcross), of shape `(2, len(f))` if `by_m` is
        `False`, or `(n_m, 2, len(f))` if `by_m` is `True`, where `n_m`
        is the number of harmonic modes with different `m`.
        """
        if self.disable_precession:
            waveform_par_dic.update(ZERO_INPLANE_SPINS)

        slow_par_vals = np.array([waveform_par_dic[par]
                                  for par in self.slow_params])

        # Attempt to use cached waveform for fast evaluation:
        if matching_cache := self._matching_cache(slow_par_vals, f):
            hplus_hcross_0 = matching_cache['hplus_hcross_0']
            self.n_fast_evaluations += 1
        else:
            # Compute the waveform mode by mode and update cache.
            lal_dic = self.create_lal_dict()

            waveform_par_dic_0 = dict(zip(self.slow_params, slow_par_vals),
                                      d_luminosity=1., phi_ref=0.)

            # hplus_hcross_0 is a (n_m x 2 x n_frequencies) array with
            # sum_l (hlm+, hlmx), at phi_ref=0, d_luminosity=1Mpc.
            hplus_hcross_modes \
                = APPROXIMANTS[self.approximant].hplus_hcross_by_mode_func(
                    f,
                    waveform_par_dic_0,
                    self.approximant,
                    self.harmonic_modes,
                    lal_dic)

            hplus_hcross_0 = np.array(
                [np.sum([hplus_hcross_modes[mode] for mode in m_modes], axis=0)
                 for m_modes in self._harmonic_modes_by_m.values()])

            cache_dic = {'approximant': self.approximant,
                         'f': f,
                         'slow_par_vals': slow_par_vals,
                         'harmonic_modes_by_m': self._harmonic_modes_by_m,
                         'hplus_hcross_0': hplus_hcross_0,
                         'lalsimulation_commands': self.lalsimulation_commands}

            # Append new cached waveform and delete oldest
            self.cache.append(cache_dic)
            self.cache.pop(0)

            self.n_slow_evaluations += 1

        # hplus_hcross is a (n_m x 2 x n_frequencies) array.
        m_arr = self.m_arr.reshape(-1, 1, 1)
        hplus_hcross = (np.exp(1j * waveform_par_dic['phi_ref'] * m_arr)
                        / waveform_par_dic['d_luminosity'] * hplus_hcross_0)
        if by_m:
            return hplus_hcross
        return np.sum(hplus_hcross, axis=0)

    def create_lal_dict(self):
        """Return a LAL dict object per ``self.lalsimulation_commands``."""
        lal_dic = lal.CreateDict()
        for function_name, value in self.lalsimulation_commands:
            getattr(lalsimulation, function_name)(lal_dic, value)
        return lal_dic

    def _matching_cache(self, slow_par_vals, f, eps=1e-6):
        """
        Return entry of the cache that matches the requested waveform, or
        `False` if none of the cached waveforms matches that requested.
        """
        for cache_dic in self.cache[ : : -1]:
            if (np.linalg.norm(slow_par_vals-cache_dic['slow_par_vals']) < eps
                    and cache_dic['approximant'] == self.approximant
                    and np.array_equal(cache_dic['f'], f)
                    and cache_dic['harmonic_modes_by_m']
                        == self._harmonic_modes_by_m
                    and cache_dic['lalsimulation_commands']
                        == self.lalsimulation_commands):
                return cache_dic

        return False

    def get_f_finer(self, f, delta_f_finer=1/2**18, delta_f_coarse=1/2**10):
        """
        Generate finer and coarser frequency grids based on the input frequency range.
    
        Parameters
        ----------
        f : array-like
            Input frequency array.
        delta_f_finer : float, optional
            Frequency step for finer grid, by default 1/2**18.
        delta_f_coarse : float, optional
            Frequency step for coarse grid, by default 1/2**10.
    
        Returns
        -------
        f_finer : numpy array
            Finer frequency grid array.
        f_coarse : numpy array
            Coarser frequency grid array.
        """
        f_min = np.min(f)
        f_max = np.max(f)
        mean_delta_f = np.mean(np.diff(f))
    
        # Determine f_finer and f_coarse based on mean_delta_f
        if mean_delta_f <= delta_f_finer:
            f_finer = f
            f_coarse = f
            
        elif mean_delta_f <= delta_f_coarse:
            f_finer = np.arange(f_min, f_max, delta_f_finer)
            f_coarse = f
            
        else:
            f_finer = np.arange(f_min, f_max, delta_f_finer)
            f_coarse = np.arange(f_min, f_max, delta_f_coarse)
    
        return f_finer, f_coarse



    def time_series(self, f, par_dic, by_m=False, f_lower=2, f_higher=10):
        """
        Generate the full time series based on the given frequency array, using 
        finer frequencies at lower frequencies and coarse frequencies at higher frequencies.
        
        Parameters
        ----------
        f : array-like
            Frequency array for generating the time series.
        par_dic : dict
            Dictionary of parameters.
        by_m : bool, optional
            Whether to return waveform separated by m harmonic mode, by default False.
        f_lower : float, optional
            Lower threshold of frequency, by default 2.
        f_higher : float, optional
            Higher threshold of frequency, by default 10.
    
        Returns
        -------
        t_full : array-like
            Full time series array.
        """
        # Generate finer frequency grid within [f_lower, f_higher]
        f_finer, f_coarse = self.get_f_finer(f)
        f_finer = f_finer[(f_finer >= f_lower) & (f_finer <= f_higher)]
        f_coarse = f_coarse[f_coarse >= f_higher]
        
    
        # Compute hp for finer and coarse frequency segments
        hp_finer = self.get_hplus_hcross(f_finer, par_dic, by_m)[0, :]
        hp_coarse = self.get_hplus_hcross(f_coarse, par_dic, by_m)[0, :]
    
        # Create interpolation functions for both segments
        t_interp_finer = generate_interpolation_function(hp_finer, f_finer)
        t_interp_coarse = generate_interpolation_function(hp_coarse, f_coarse)
    
        # Initialize the time series array
        time_series = np.zeros_like(f)
    
        # Populate time_series using the two interpolation functions
        for i, freq in enumerate(f):
            if f_lower <= freq <= f_higher:
                time_series[i] = t_interp_finer(freq)
            elif freq > f_higher:
                time_series[i] = t_interp_coarse(freq)
    
        return time_series

    
        


        
