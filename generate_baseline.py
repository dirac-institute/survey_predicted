from baseline import create_scheduler
from rubin_sim.scheduler.schedulers import simple_filter_sched
from rubin_sim.scheduler import sim_runner
from rubin_sim.scheduler.modelObservatory import Model_observatory
import numpy as np
mjd_start = 59638

if __name__ == '__main__':
    nside = 32
    survey_length = 365.25
    fileroot = 'baseline_'
    extra_info = None
    illum_limit = 40.
    years = np.round(survey_length/365.25)
    verbose = False

    scheduler = create_scheduler()
    n_visit_limit = None
    filter_sched = simple_filter_sched(illum_limit=illum_limit)
    observatory = Model_observatory(nside=nside, mjd_start=mjd_start)
    observatory, scheduler, observations = sim_runner(observatory, scheduler,
                                                      survey_length=survey_length,
                                                      filename=fileroot+'%iyrs.db' % years,
                                                      delete_past=True, n_visit_limit=n_visit_limit,
                                                      verbose=verbose, extra_info=extra_info,
                                                      filter_scheduler=filter_sched)