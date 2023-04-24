import numpy as np
import matplotlib.pylab as plt
import healpy as hp
import rubin_sim.scheduler.basis_functions as bf
from rubin_sim.scheduler.surveys import (Long_gap_survey, Scripted_survey,
                                         Blob_survey)
import rubin_sim.scheduler.detailers as detailers
from rubin_sim.scheduler.basis_functions import Base_basis_function


class Delay_start_basis_function(Base_basis_function):
    """Force things to not run before a given night
    """
    def __init__(self, nights_delay=365.25*5):
        super().__init__()
        self.nights_delay = nights_delay

    def check_feasibility(self, conditions):
        result = True
        if conditions.night < self.nights_delay:
            result = False
        return result


def blob_for_long(nside, nexp=2, exptime=30., filter1s=['g'],
                  filter2s=['i'], pair_time=33.,
                  camera_rot_limits=[-80., 80.], n_obs_template=3,
                  season=300., season_start_hour=-4., season_end_hour=2.,
                  shadow_minutes=60., max_alt=76., moon_distance=30., ignore_obs=['DD', 'twilight_neo'],
                  m5_weight=6., footprint_weight=1.5, slewtime_weight=3.,
                  stayfilter_weight=3., template_weight=12., u_template_weight=24., footprints=None, u_nexp1=True,
                  night_pattern=[True, True], time_after_twi=30., HA_min=12, HA_max=24-3.5,
                  nights_delayed=-1):
    """
    Generate surveys that take observations in blobs.

    Parameters
    ----------
    nside : int (32)
        The HEALpix nside to use
    nexp : int (1)
        The number of exposures to use in a visit.
    exptime : float (30.)
        The exposure time to use per visit (seconds)
    filter1s : list of str
        The filternames for the first set
    filter2s : list of str
        The filter names for the second in the pair (None if unpaired)
    pair_time : float (33)
        The ideal time between pairs (minutes)
    camera_rot_limits : list of float ([-80., 80.])
        The limits to impose when rotationally dithering the camera (degrees).
    n_obs_template : int (3)
        The number of observations to take every season in each filter
    season : float (300)
        The length of season (i.e., how long before templates expire) (days)
    season_start_hour : float (-4.)
        For weighting how strongly a template image needs to be observed (hours)
    sesason_end_hour : float (2.)
        For weighting how strongly a template image needs to be observed (hours)
    shadow_minutes : float (60.)
        Used to mask regions around zenith (minutes)
    max_alt : float (76.
        The maximium altitude to use when masking zenith (degrees)
    moon_distance : float (30.)
        The mask radius to apply around the moon (degrees)
    ignore_obs : str or list of str ('DD')
        Ignore observations by surveys that include the given substring(s).
    m5_weight : float (3.)
        The weight for the 5-sigma depth difference basis function
    footprint_weight : float (0.3)
        The weight on the survey footprint basis function.
    slewtime_weight : float (3.)
        The weight on the slewtime basis function
    stayfilter_weight : float (3.)
        The weight on basis function that tries to stay avoid filter changes.
    template_weight : float (12.)
        The weight to place on getting image templates every season
    u_template_weight : float (24.)
        The weight to place on getting image templates in u-band. Since there
        are so few u-visits, it can be helpful to turn this up a little higher than
        the standard template_weight kwarg.
    u_nexp1 : bool (True)
        Add a detailer to make sure the number of expossures in a visit is always 1 for u observations.
    """

    blob_survey_params = {'slew_approx': 7.5, 'filter_change_approx': 140.,
                          'read_approx': 2., 'min_pair_time': 15., 'search_radius': 30.,
                          'alt_max': 85., 'az_range': 90., 'flush_time': 30.,
                          'smoothing_kernel': None, 'nside': nside, 'seed': 42, 'dither': True,
                          'twilight_scale': True}

    surveys = []

    times_needed = [pair_time, pair_time*2]
    for filtername, filtername2 in zip(filter1s, filter2s):
        detailer_list = []
        detailer_list.append(detailers.Camera_rot_detailer(min_rot=np.min(camera_rot_limits),
                                                           max_rot=np.max(camera_rot_limits)))
        detailer_list.append(detailers.Close_alt_detailer())
        # List to hold tuples of (basis_function_object, weight)
        bfs = []

        if filtername2 is not None:
            bfs.append((bf.M5_diff_basis_function(filtername=filtername, nside=nside), m5_weight/2.))
            bfs.append((bf.M5_diff_basis_function(filtername=filtername2, nside=nside), m5_weight/2.))

        else:
            bfs.append((bf.M5_diff_basis_function(filtername=filtername, nside=nside), m5_weight))

        if filtername2 is not None:
            bfs.append((bf.Footprint_basis_function(filtername=filtername,
                                                    footprint=footprints,
                                                    out_of_bounds_val=np.nan, nside=nside), footprint_weight/2.))
            bfs.append((bf.Footprint_basis_function(filtername=filtername2,
                                                    footprint=footprints,
                                                    out_of_bounds_val=np.nan, nside=nside), footprint_weight/2.))
        else:
            bfs.append((bf.Footprint_basis_function(filtername=filtername,
                                                    footprint=footprints,
                                                    out_of_bounds_val=np.nan, nside=nside), footprint_weight))

        bfs.append((bf.Slewtime_basis_function(filtername=filtername, nside=nside), slewtime_weight))
        bfs.append((bf.Strict_filter_basis_function(filtername=filtername), stayfilter_weight))

        if filtername2 is not None:
            bfs.append((bf.N_obs_per_year_basis_function(filtername=filtername, nside=nside,
                                                         footprint=footprints.get_footprint(filtername),
                                                         n_obs=n_obs_template, season=season,
                                                         season_start_hour=season_start_hour,
                                                         season_end_hour=season_end_hour), template_weight/2.))
            bfs.append((bf.N_obs_per_year_basis_function(filtername=filtername2, nside=nside,
                                                         footprint=footprints.get_footprint(filtername2),
                                                         n_obs=n_obs_template, season=season,
                                                         season_start_hour=season_start_hour,
                                                         season_end_hour=season_end_hour), template_weight/2.))
        else:
            bfs.append((bf.N_obs_per_year_basis_function(filtername=filtername, nside=nside,
                                                         footprint=footprints.get_footprint(filtername),
                                                         n_obs=n_obs_template, season=season,
                                                         season_start_hour=season_start_hour,
                                                         season_end_hour=season_end_hour), template_weight))
        # Masks, give these 0 weight
        bfs.append((bf.Zenith_shadow_mask_basis_function(nside=nside, shadow_minutes=shadow_minutes, max_alt=max_alt,
                                                         penalty=np.nan, site='LSST'), 0.))
        bfs.append((bf.Moon_avoidance_basis_function(nside=nside, moon_distance=moon_distance), 0.))
        filternames = [fn for fn in [filtername, filtername2] if fn is not None]
        bfs.append((bf.Filter_loaded_basis_function(filternames=filternames), 0))
        if filtername2 is None:
            time_needed = times_needed[0]
        else:
            time_needed = times_needed[1]
        bfs.append((bf.Time_to_twilight_basis_function(time_needed=time_needed), 0.))
        bfs.append((bf.Not_twilight_basis_function(), 0.))
        bfs.append((bf.Planet_mask_basis_function(nside=nside), 0.))
        bfs.append((bf.After_evening_twi_basis_function(time_after=time_after_twi), 0.))
        # XXX--move kwargs up
        bfs.append((bf.HA_mask_basis_function(HA_min=HA_min, HA_max=HA_max), 0.))
        # don't execute every night
        bfs.append((bf.Night_modulo_basis_function(night_pattern), 0.))
        # possibly force things to delay
        bfs.append((Delay_start_basis_function(nights_delay=nights_delayed), 0.))

        # unpack the basis functions and weights
        weights = [val[1] for val in bfs]
        basis_functions = [val[0] for val in bfs]
        if filtername2 is None:
            survey_name = 'blob_long, %s' % filtername
        else:
            survey_name = 'blob_long, %s%s' % (filtername, filtername2)
        if filtername2 is not None:
            detailer_list.append(detailers.Take_as_pairs_detailer(filtername=filtername2))

        if u_nexp1:
            detailer_list.append(detailers.Filter_nexp(filtername='u', nexp=1))
        surveys.append(Blob_survey(basis_functions, weights, filtername1=filtername, filtername2=filtername2,
                                   exptime=exptime,
                                   ideal_pair_time=pair_time,
                                   survey_note=survey_name, ignore_obs=ignore_obs,
                                   nexp=nexp, detailers=detailer_list, **blob_survey_params))

    return surveys


def gen_long_gaps_survey(footprints, nside=32, night_pattern=[True, True],
                         gap_range=[2, 7], HA_min=12, HA_max=24-3.5,
                         time_after_twi=120, nights_delayed=-1):
    """
    Paramterers
    -----------
    HA_min(_max) : float
        The hour angle limits passed to the initial blob scheduler.
    """

    surveys = []
    f1 = ['g', 'r', 'i']
    f2 = ['r', 'i', 'z']
    # Maybe force scripted to not go in twilight?

    for filtername1, filtername2 in zip(f1, f2):
        blob = blob_for_long(footprints=footprints, nside=nside, filter1s=[filtername1],
                             filter2s=[filtername2], night_pattern=night_pattern, time_after_twi=time_after_twi,
                             HA_min=HA_min, HA_max=HA_max, nights_delayed=nights_delayed)
        scripted = Scripted_survey([], nside=nside, ignore_obs=['blob', 'DDF', 'twi'])
        surveys.append(Long_gap_survey(blob[0], scripted,
                                       gap_range=gap_range, avoid_zenith=True))

    return surveys
