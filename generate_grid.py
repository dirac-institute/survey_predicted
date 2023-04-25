import numpy as np
import healpy as hp
from rubin_sim.utils import ddf_locations
import rubin_sim.skybrightness as sb
from rubin_sim.utils import m5_flat_sed, survey_start_mjd
from rubin_sim.site_models import SeeingModel
import sys
from astroplan import Observer
import astropy.units as u
from astropy.time import Time
from rubin_sim.utils import Site

if __name__ == "__main__":

    verbose = True
    dds = ddf_locations()
    mjd0 = 60218.0
    delta_t = 15./60./24.  # to days
    survey_length = 12.*365.25  # Let's just go 12 years for good measure
    sun_limit = np.radians(-12.)  # degrees
    nominal_seeing = 0.7  # arcsec
    filtername = 'g'

    site = Site("LSST")
    observer = Observer(
        longitude=site.longitude * u.deg,
        latitude=site.latitude * u.deg,
        elevation=site.height * u.m,
        name="LSST",
    )

    seeing_model = SeeingModel()

    seeing_indx = 1  # 0=u, 1=g, 2=r, etc.

    mjds = np.arange(mjd0, mjd0+survey_length, delta_t)

    names = ['mjd', 'sun_alt', 'sun_n18_rising_next']
    for survey_name in dds.keys():
        names.append(survey_name+'_airmass')
        names.append(survey_name+'_sky_g')
        names.append(survey_name+'_m5_g')

    types = [float]*len(names)
    result = np.zeros(mjds.size, dtype=list(zip(names, types)))
    result['mjd'] = mjds

    # pretty sure these are radians
    ras = np.radians(np.array([dds[survey][0] for survey in dds]))
    decs = np.radians(np.array([dds[survey][1] for survey in dds]))

    sm = sb.SkyModel(mags=True)
    mags = []
    airmasses = []
    sun_alts = []

    maxi = mjds.size
    for i, mjd in enumerate(mjds):
        if verbose:
            progress = i/maxi*100
            text = "\rprogress = %0.1f%%" % progress
            sys.stdout.write(text)
            sys.stdout.flush()

        sm.set_ra_dec_mjd(ras, decs, mjd, degrees=False)
        if sm.sun_alt > sun_limit:
            mags.append(sm.return_mags()['g']*0)
            airmasses.append(sm.airmass*0)
        else:
            mags.append(sm.return_mags()['g'])
            airmasses.append(sm.airmass)
        sun_alts.append(sm.sun_alt)
        result["sun_n18_rising_next"][i] = observer.twilight_morning_astronomical(Time(mjd, format='mjd'), which="next").mjd

    mags = np.array(mags)
    airmasses = np.array(airmasses)
    result['sun_alt'] = sun_alts

    for i, survey_name in enumerate(dds.keys()):
        result[survey_name+'_airmass'] = airmasses[:, i]
        result[survey_name+'_sky_g'] = mags[:, i]

        # now to compute the expected seeing if the zenith is nominal
        FWHMeff = seeing_model(nominal_seeing, airmasses[:, i])['fwhmEff'][seeing_indx, :]
        result[survey_name+'_m5_g'] = m5_flat_sed('g', mags[:, i], FWHMeff, 30.,
                                                  airmasses[:, i], nexp=1)

    np.savez('ddf_grid.npz', ddf_grid=result)
