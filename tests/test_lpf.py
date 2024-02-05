import pytest
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from tfop_analysis import LPF  # Replace 'your_module' with the actual module name
from matplotlib.figure import Figure

target = "TOI-6449"
ticid = "168936945"
phot_dir = "../data"
date = "240128"
ra, dec = 125.93045, -5.18949

star_params = {
    "rstar": (0.454183, 0.0155273),
    "mstar": (0.454183, 0.0155273),
    "rhostar": (4.813903712185822, 0.5460704755765515),
    "teff": (3356.0, 157.0),
    "logg": (4.7778, 0.008642),
    "feh": (0.0, 0.1),
}


planet_params = {
    "t0": (2460338.0465, 3 / 60 / 24),
    "period": (2.4404127, 0.000413),
    "tdur": (81 / 60 / 24, 12 / 60 / 24),
    "imp": (0.0, 0.1),
    "rprs": (0.25979992, 0.07893168),
    "a_Rs": (12.879247193048114, 0.48699053548942756),
}

# @pytest.mark.exclude
def read_phot_data(phot_dir, date, bands=["g", "r", "i", "z"]):
    data = {}
    phot_files = list(Path(phot_dir).glob(f"*{date}*.csv"))
    errmsg = f"No photometry files found in {phot_dir}"
    assert len(phot_files) > 0, errmsg
    for fp in phot_files:
        band = fp.name.split("_")[2]
        if band in bands:
            data[band] = pd.read_csv(fp)
    return data


def test_read_phot_data():
    data = read_phot_data(phot_dir, date)
    assert isinstance(data, dict)


@pytest.fixture
def sample_inputs():
    # Provide sample data for testing
    return {
        "name": target,
        "ticid": ticid,
        "date": date,
        "data": read_phot_data(phot_dir, date),
        "star_params": star_params,
        "planet_params": planet_params,
    }


def test_lpf(sample_inputs):
    lpf_instance = LPF(**sample_inputs)

    # Check that instance is created
    assert isinstance(lpf_instance, LPF)

    # assert isinstance(lpf_instance.plot_raw_data(), Figure)
    p0 = [v[0] for k, v in lpf_instance.model_params.items()]
    assert isinstance(lpf_instance.get_chi2_chromatic_transit(p0), float)
    while True:
        lpf_instance.optimize_chromatic_transit(p0)
        if lpf_instance.opt_result.success:
            break
    pv = lpf_instance.optimum_params
    fig = lpf_instance.plot_lightcurves(pv, figsize=(10, 5))
    # lpf_instance.sample_mcmc(nsteps=3_000)
    # fig = lpf_instance.plot_chain()
    # assert isinstance(fig, Figure)
    # df = lpf_instance.get_mcmc_samples()
    # fig = lpf_instance.plot_corner(discard=1, thin=10, start=0, end=7)
    # assert isinstance(fig, Figure)
    # fig = lpf_instance.plot_posteriors(nsigma=3)
    # assert isinstance(fig, Figure)
    # fig = lpf_instance.plot_final_fit(ylims_top=(0.9,1.08),
    #                      ylims_bottom=(0.87,1.05),
    #                      fsize=25)
    # assert isinstance(fig, Figure)
    # pv = lpf_instance.best_fit_params
    df = pd.read_csv(f"{phot_dir}/chromatic_samples.csv.gz")
    pv = df.median().values
    fig = lpf_instance.plot_detrended_data_and_transit(
        pv,
        figsize=(10, 10),
        # xlims=(-1.3,1.3),
        ylims=(0.85, 1.05),
        title_height=1,
        font_size=18,
    )
    assert isinstance(fig, Figure)
    ref_fits_file_path = f"{phot_dir}/ref-coj2m002-ep09-20240128-0075-e91.fits"
    ref_obj_file_path = f"{phot_dir}/ref-coj2m002-ep09-20240128-0075-e91.objects"
    fig = lpf_instance.plot_fov(
        ra,
        dec,
        ref_fits_file_path,
        ref_obj_file_path,
        phot_aper_pix=30,
        cmap="gray",
        contrast=0.1,
        marker_color="g",
        text_offset=(0.005, -0.007),
        scale_color="w",
        font_size=20,
        title_height=0.99,
        show_target=True,
    )
    assert isinstance(fig, Figure)
    fig = lpf_instance.plot_fov_zoom(
        ra,
        dec,
        ref_fits_file_path,
        ref_obj_file_path,
        zoom_rad_arcsec=60,
        bar_arcsec=30,
        phot_aper_pix=16,
        cmap="gray",
        contrast=0.1,
        marker_color="g",
        text_offset=(-0.001, -0.002),
        scale_color="w",
        font_size=20,
        title_height=0.99,
    )
    assert isinstance(fig, Figure)
    # fig = lpf_instance.plot_radii(*star_params['rstar'])
    # assert isinstance(fig, Figure)
    # gaia_sources = star.get_gaia_sources(rad_arcsec=60)
    # fig = lpf_instance.plot_gaia_sources(ref_fits_file_path,
    #                             gaia_sources,
    #                             phot_aper_pix=20,
    #                             text_offset=(0.001, 0),
    #                             fov_padding=1.2,
    #                             cmap='gray_r',
    #                             contrast=0.1,
    #                             bar_arcsec=20,
    #                             marker_color='r',
    #                             scale_color='k',
    #                             font_size=20,
    #                             title_height=0.99,
    #                         )
    # assert isinstance(fig, Figure)
