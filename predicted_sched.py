#!/usr/bin/env python

import numpy as np
import matplotlib.pylab as plt
import healpy as hp
from rubin_sim.scheduler.modelObservatory import Model_observatory
from rubin_sim.scheduler.schedulers import Core_scheduler, simple_filter_sched
from rubin_sim.scheduler import sim_runner
from rubin_sim.scheduler.utils import schema_converter
from rubin_sim.utils import survey_start_mjd
import argparse
from baseline import create_scheduler
from rubin_sim.site_models import Almanac
# So things don't fail on hyak
from astropy.utils import iers
iers.conf.auto_download = False


# grabbing this from rubin_sim and making some mods
def restore_scheduler(mjd_set, night_max, scheduler, observatory,
                      filename, filter_sched=None, delta=0.1):
    """Put the scheduler and observatory in the state they were in. Handy for checking reward fucnction
    Parameters
    ----------
    night_max : int
        The night to restore up to (and including)
    scheduler : rubin_sim.scheduler.scheduler object
        Scheduler object.
    observatory : rubin_sim.scheduler.observatory.Model_observatory
        The observaotry object
    filename : str
        The output sqlite dayabase to use
    filter_sched : rubin_sim.scheduler.scheduler object
        The filter scheduler. Note that we don't look up the official end of the previous night,
        so there is potential for the loaded filters to not match.
    """
    sc = schema_converter()
    # load up the observations
    observations = sc.opsim2obs(filename)
    good_obs = np.where(observations["night"] <= night_max)[0]
    observations = observations[good_obs]

    # replay the observations back into the scheduler
    for obs in observations:
        scheduler.add_observation(obs)
        if filter_sched is not None:
            filter_sched.add_observation(obs)

    if filter_sched is not None:
        # Make sure we have mounted the right filters for the night
        # XXX--note, this might not be exact, but should work most of the time.
        
        observatory.mjd = mjd_set
        conditions = observatory.return_conditions()
        filters_needed = filter_sched(conditions)
    else:
        filters_needed = ["u", "g", "r", "i", "y"]

    # update the observatory
    observatory.mjd = mjd_set
    observatory.observatory.park()
    observatory.observatory.mounted_filters = filters_needed
    # Note that we haven't updated last_az_rad, etc, but those values should be ignored.

    return scheduler, observatory


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", dest='verbose', action='store_true')
    parser.set_defaults(verbose=False)
    parser.add_argument("--night_start", type=float, default=0)
    parser.add_argument("--truth_file", type=str, default='baseline_1yrs.db')
    parser.add_argument("--survey_length", type=float, default=15.)
   
    nside = 32
    illum_limit = 40.

    args = parser.parse_args()
    # I like to unpack the args
    verbose = args.verbose
    night_start = args.night_start
    truth_file = args.truth_file
    survey_length = args.survey_length

    mjd_start = survey_start_mjd()
    almanac = Almanac(mjd_start)

    indx = np.where(almanac.sunsets['night'] == night_start)[0]
    mjd_skip_to = almanac.sunsets['sun_n12_setting'][indx]

    # Let's build the baseline scheduler object.
    # Would probably be faster to do this once and pickle/restore it.
    scheduler = create_scheduler()

    # An observatory model that has no weather downtime
    observatory = Model_observatory(mjd_start=mjd_start, nside=nside, ideal_conditions=True)
    filter_sched = simple_filter_sched(illum_limit=illum_limit)

    # 
    scheduler, observatory = restore_scheduler(mjd_skip_to, night_start, scheduler,
                                               observatory, truth_file,
                                               filter_sched=filter_sched)

    observatory, scheduler, observations = sim_runner(observatory, scheduler,
                                                      survey_length=survey_length,
                                                      filename='night%i_%idays.db' % (night_start, survey_length),
                                                      delete_past=True, n_visit_limit=None,
                                                      verbose=verbose, extra_info=None,
                                                      filter_scheduler=filter_sched)
