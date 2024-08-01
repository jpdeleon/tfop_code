#!/usr/bin/env python
"""
Written by J.P. de Leon based on
TFOP analysis code by A. Fukui.

User-friedly features include:
* easy to fetch TFOP parameters
* handles any number of bands
* easy to switch between models
  - achromatic & chromatic
  - specifying covariates (default=Airmass)
  - polynomial order of linear model (default=1)
* shows useful plots: quicklook, raw data, posteriors, FOV, FOV with gaia sources, etc
* implements new parameterization for efficient sampling of impact parameter and Rp/Rs
  (Espinosa+2018: https://iopscience.iop.org/article/10.3847/2515-5172/aaef38)

Note:
* To re-define the priors, manually edit
  `get_chi2_transit` method within `LPF` class

TODO
* use other optimizers e.g. PSO
    - https://github.com/ljvmiranda921/pyswarms
* parallelize for faster chi2 evaluation within mcmc
"""
import os
import json
from typing import Dict, Tuple, List
from urllib.request import urlopen
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import pandas as pd
from scipy.optimize import minimize
from scipy import stats
from astropy.io import fits
from astropy.visualization import ZScaleInterval, ImageNormalize
from astropy.wcs import WCS
import astropy.units as u
from astropy.visualization.wcsaxes import SphericalCircle  # , add_scalebar
from astropy.coordinates import SkyCoord
from astroquery.mast import Catalogs
from aesthetic.plot import savefig
import emcee
import corner
import seaborn as sb
from pytransit import QuadraticModel
from ldtk import LDPSetCreator, BoxcarFilter

# from aesthetic.plot import set_style
# set_style("science")
# try:
#     sys.path.insert(0, "/ut3/muscat/src/AFPy")
#     import LC_funcs as lc
# except Exception as e:
#     print(e)
#     print("/ut3/muscat/src/AFPy/LC_funcs.py not found!")
#     pass

import seaborn as sb

sb.set(
    context="paper",
    style="ticks",
    palette="deep",
    font="sans-serif",
    font_scale=1.5,
    color_codes=True,
)
sb.set_style({"xtick.direction": "in", "ytick.direction": "in"})
sb.set_context(rc={"lines.markeredgewidth": 1})
plt.rcParams["font.size"] = 26

os.environ["OMP_NUM_THREADS"] = "1"
os.nice(19)

PRIOR_K_MIN = 0.001
PRIOR_K_MAX = 1.0

# bandpasses in increasing wavelength
filter_widths = {
    "g": (430, 540),
    "r": (560, 700),
    "i": (700, 820),
    "z": (830, 910),
    "g-narrow": (489.5, 540.5), #515 / 51
    "Na_D": (586, 592.2), #589.1 / 6.2
    "i-narrow": (776, 808), #792 / 32
    "z-narrow": (846.5, 889.5), #868 / 43
} 
bands = list(filter_widths.keys())
colors = {
    "g": "blue",
    "r": "green",
    "i": "darkorange",
    "z": "red",
    "g-narrow": "blue",
    "Na_D": "green",
    "i-narrow": "darkorange",
    "z-narrow": "red",
}

fovs = {
    'sinistro_full': 13*60, #CONFMODE= 'full_frame'
    'sinistro_2x2': 26.5*60, #CONFMODE= 'central_2k_2x2'
    'muscat': 6.1, 
    'muscat2': 7.4, #7.4 × 7.4
    'muscat3': 9.1,
    'muscat4': 9.1
    
}
pixscales = {
    'sinistro_full': 0.389, #CONFMODE= 'full_frame'
    'sinistro_2x2': 0.778, #CONFMODE= 'central_2k_2x2'
    'muscat': 0.358,
    'muscat2': 0.44,
    'muscat3': 0.267,
    'muscat4': 0.267,
}


@dataclass
class LPF:
    name: str
    date: str
    toi_name: str
    ticid: str
    data: Dict = field(repr=False)
    star_params: Dict[str, Tuple[float, float]] = field(repr=False)
    planet_params: Dict[str, Tuple[float, float]] = field(repr=False)
    inst: str = "MuSCAT"
    alias: str = ".01"
    bands: List[str] = None
    model: str = "chromatic"
    # time_offset: float = None
    covariate: str = "Airmass"
    lm_order: int = 1
    mask_start: float = None
    mask_end: float = None
    outdir: str = "results"
    use_r1r2: bool = False
    DEBUG: bool = field(repr=False, default=False)

    def __post_init__(self):
        self.date = str(self.date)
        self._validate_inputs()
        self._init_data()
        self._init_params()
        self._init_ldc()
        self.date4plot = datetime.strptime(self.date, "%y%m%d").strftime(
            "%Y-%b-%d UT"
        )
        if not Path(self.outdir).exists():
            Path(self.outdir).mkdir()
        self.outfile_prefix = f"{self.ticid}{self.alias}_20{self.date}"
        self.outfile_prefix += f"_{self.inst}_{''.join(self.bands)}_{self.model}"
        self._mcmc_samples = None

    def _validate_inputs(self):
        """Make sure inputs are correct"""
        assert (PRIOR_K_MIN>=0) & (PRIOR_K_MAX<=1)
        assert isinstance(self.data, dict), "data must be a dict"
        b = list(self.data.keys())[0]
        errmsg = "each item of data must be a pd.DataFrame"
        assert isinstance(self.data[b], pd.DataFrame), errmsg

        # sort data in griz order
        self.data = dict(sorted(self.data.items(), key=lambda x: bands.index(x[0])))
        self.obs_start = min([self.data[b]["BJD_TDB"].min() for b in self.data])
        self.obs_end = max([self.data[b]["BJD_TDB"].max() for b in self.data])
        self.bands = list(self.data.keys()) if self.bands is None else self.bands
        errmsg = f"`bands` can only use the given keys in `data`: {self.data.keys()}"
        assert len(self.bands) <= len(self.data.keys()), errmsg
        for b in self.bands:
            errmsg = f"{b} not in `data` keys: {self.data.keys()}"
            assert b in self.data.keys(), errmsg
        self.nband = len(self.bands)

        assert isinstance(self.star_params, dict)
        assert isinstance(self.planet_params, dict)
        assert (
            np.array(self.planet_params["tdur"]) < 1
        ).all(), "`tdur` must be in days"
        assert (np.array(self.planet_params["rprs"]) < 1).all(), "Check `rprs`"
        assert self.planet_params["a_Rs"][0] > 1, "`a/Rs` is <1?"

        errmsg = f"{self.covariate} not in {self.data[b].columns}"
        assert self.covariate in self.data[b].columns, errmsg

        if np.all([self.mask_start, self.mask_end]):
            errmsg = f"{self.mask_start} is too early. Try {self.obs_start}"
            assert self.mask_start >= self.obs_start, errmsg
            errmsg = f"{self.mask_end} is too late. Try {self.obs_end}"
            assert self.mask_end <= self.obs_end, errmsg
            tdiff = (self.mask_end - self.mask_start) * 60 * 24
            print(f"Masking {tdiff:.1f} min of data.")

        self.time_offset = np.floor(self.obs_start)
        errmsg = f"`time_offset` is too big. Observation ended at {self.obs_end}."
        assert self.time_offset < self.obs_end, errmsg

        # update times
        self.obs_start -= self.time_offset
        self.obs_end -= self.time_offset

        self.teff = self.star_params["teff"]
        self.logg = self.star_params["logg"]
        self.feh = self.star_params["feh"]

        models = ["achromatic", "chromatic"]
        errmsg = f"model is either {' or '.join(models)}"
        assert self.model in models, errmsg

        if self.DEBUG:
            print("==========")
            print("DEBUG MODE")
            print("==========")
            print("obs_start", self.obs_start)
            print("obs_end", self.obs_end)
            print("bands", self.bands)
            print("nband", self.nband)
            print(f"time_offset {self.time_offset:,}")

    def _init_data(self):
        """initialize user-given photometry data"""
        self.masks = {}
        self.times_raw = {}
        self.fluxes_raw = {}
        self.flux_errs_raw = {}
        self.covariates_raw = {}
        self.times = {}
        self.fluxes = {}
        self.flux_errs = {}
        self.covariates = {}
        self.transit_models = {}
        self.lin_model_offsets = {b: 0 for b in self.bands}

        for b in self.bands:
            df = self.data[b]
            t = df["BJD_TDB"].values - self.time_offset
            f = df["Flux"].values
            e = df["Err"].values
            z = df[self.covariate].values
            self.times_raw[b] = t + self.time_offset
            self.fluxes_raw[b] = f
            self.flux_errs_raw[b] = e
            self.covariates_raw[b] = z
            m1 = np.zeros_like(t, dtype=bool)
            m2 = np.zeros_like(t, dtype=bool)
            if self.mask_start:
                m1 = t >= self.mask_start - self.time_offset
            if self.mask_end:
                m2 = t <= self.mask_end - self.time_offset
            mask = m1 & m2
            if sum(mask) > 0:
                print(f"Masked {sum(mask)} data points in {b}-band.")
            self.masks[b] = m1 & m2
            self.times[b] = t[~mask]
            self.fluxes[b] = f[~mask]
            self.flux_errs[b] = e[~mask]
            self.covariates[b] = z[~mask]
            self.transit_models[b] = QuadraticModel()
            self.transit_models[b].set_data(t[~mask])
        self.ndata = sum([len(self.times[b]) for b in self.bands])

    def _init_params(self, return_dict=False):
        """initialize parameters of the model"""
        self.period = self.planet_params["period"]
        self.tdur = self.planet_params["tdur"]
        tc0 = self.planet_params["t0"]
        tc, tc_err = tc0[0] - self.time_offset, tc0[1]

        # obs_mid = self.obs_end - self.obs_start
        tc0 = self.planet_params["t0"]
        tc, tc_err = tc0[0] - self.time_offset, tc0[1]
        tmed = np.median(np.concatenate([self.times[b] for b in self.data]))
        if abs(tc - tmed) > self.period[0]:
            print(f"Input t0: {tc0}")
            norbit = round((tmed - tc) / self.period[0])
            norbits = np.array([norbit-1,norbit,norbit+1])
            diffs = abs((tc+self.period[0]*norbits)-tmed)
            i = np.argmin(diffs)
            self.norbits = norbits[i]
            tc = tc + self.period[0] * self.norbits
            tc_err = np.sqrt(tc_err**2 + (self.period[1] * self.norbits) ** 2)
            print(f"Shifted t0: ({tc+self.time_offset}, {tc_err})")
            print(f"Shifted by {self.norbits} periods.")
        errmsg = f"obs start={self.obs_start:.6f} < tc={tc:.6f} < obs end={self.obs_end:.6f}"
        assert tc > self.obs_start or tc < self.obs_end, errmsg
        self.tc = (tc, tc_err)
        self.ingress = np.array(self.tc) - self.tdur[0] / 2
        self.egress = np.array(self.tc) + self.tdur[0] / 2

        if self.use_r1r2:
            imp = self.planet_params.get("imp", (0, 0.1))
            k = self.planet_params["rprs"]
            # FIXME: imp_k_to_r1r2 is still experimental
            # r1, r2 = imp_k_to_r1r2(imp[0], k[0], k_lo=PRIOR_K_MIN, k_up=PRIOR_K_MAX)
            # r1_err, r2_err = imp_k_to_r1r2(
            #     imp[1], k[1], k_lo=PRIOR_K_MIN, k_up=PRIOR_K_MAX
            # )
            r1, r2 = 0.5, 0.5
            r1_err, r2_err = 0.5, 0.5
            params = {
                "tc": self.tc,
                "a_Rs": self.planet_params["a_Rs"],
                "r1": (r1, r1_err),  # optional
            }
            self.k_idx = len(params)
            if self.model == "chromatic":
                params.update({"r2_" + b: (r2, r2_err) for b in self.bands})
            else:
                params.update({"r2": (r2, r2_err)})
        else:
            params = {
                "tc": self.tc,
                "a_Rs": self.planet_params["a_Rs"],
                "imp": self.planet_params.get("imp", (0, 0.1)),  # optional
            }
            self.k_idx = len(params)
            if self.model == "chromatic":
                params.update(
                    {"k_" + b: self.planet_params["rprs"] for b in self.bands}
                )
            elif self.model == "achromatic":
                params.update({"k": self.planet_params["rprs"]})
        self.d_idx = len(params)
        params.update({"d_" + b: (self.lin_model_offsets[b], 0) for b in self.bands})
        self.model_params = params
        vals = [v[0] for k, v in params.items()]
        self.model_params_names = list(params.keys())
        self.model_params_vals = vals
        self.ndim = len(params)
        return params if return_dict else vals

    def _init_ldc(self):
        """initialize quadratic limb-darkening coefficients"""
        self.ldc = {}
        self.ldtk_filters = [BoxcarFilter(b, *filter_widths[b]) for b in self.bands]
        sc = LDPSetCreator(
            teff=self.teff,
            logg=self.logg,
            z=self.feh,
            filters=self.ldtk_filters,
        )
        # Create the limb darkening profiles
        ps = sc.create_profiles()
        # Estimate quadratic law coefficients
        cq, eq = ps.coeffs_qd(do_mc=False)
        qc, qe = ps.coeffs_qd()
        for i, b in enumerate(ps._filters):
            if self.DEBUG:
                print(f"{ps._filters[i]}: q1,q2=({qc[i][0]:.2f}, {qc[i][1]:.2f})")
            self.ldc[b] = (qc[i][0], qc[i][1])

    def get_chi2_linear_baseline(self, p0):
        """
        p0 : list
            parameter vector

        chi2 of linear baseline model
        """
        assert len(p0)==self.nband
        int_t0 = self.obs_start
        chi2 = 0.0
        for i, b in enumerate(self.bands):
            flux_time = p0[i] * (self.times[b] - int_t0)
            c = np.polyfit(
                self.covariates[b], self.fluxes[b] - flux_time, self.lm_order
            )
            linear_model = np.polyval(c, self.covariates[b])
            chi2 += np.sum(
                (self.fluxes[b] - linear_model + flux_time) ** 2
                / self.flux_errs[b] ** 2
            )
        return chi2

    def optimize_chi2_linear_baseline(self, p0=None, repeat=1):
        """
        p0 : list
            parameter vector
        """
        p0 = list(self.lin_model_offsets.values()) if p0 is None else p0
        assert len(p0) == self.nband
        for i in range(repeat):
            p = p0 if i == 0 else res_lin.x
            res_lin = minimize(
                self.get_chi2_linear_baseline, p, method="Nelder-Mead"
            )
            print(res_lin.fun, res_lin.success, res_lin.x)

        npar_lin = len(res_lin.x)
        # print('npar(linear) = ', npar_lin)
        self.bic_lin = res_lin.fun + npar_lin * np.log(self.ndata)
        # print('BIC(linear) = ', self.bic_lin)
        self.lin_model_offsets = {b: res_lin.x[i] for i, b in enumerate(self.bands)}
        return res_lin.x

    def unpack_parameters(self, pv):
        """
        pv : list
            parameter vector from MCMC or optimization

        Unpack commonly used parameters for transit and systematics models
        """
        assert len(pv) == self.ndim
        tc, a_Rs, imp = pv[: self.k_idx]
        if self.model == "chromatic":
            k = np.array([pv[self.k_idx + i] for i in range(self.nband)])
        elif self.model == "achromatic":
            k = np.zeros(self.nband) + pv[self.k_idx]
        d = np.array(pv[self.d_idx : self.d_idx + self.nband])
        return tc, a_Rs, imp, k, d

    def get_chi2_transit(self, pv):
        """
        fixed parameters
        ----------------
        period : fixed

        free parameters
        ---------------
        tc : mid-transit
        imp : impact parameter
        a_Rs : scaled semi-major axis
        k : radius ratio = Rp/Rs
        d : linear model coefficients

        TODO: add argument to set (normal) priors on Tc, Tdur, and a_Rs
        """
        # unpack fixed parameters
        per = self.period[0]
        # unpack free parameters
        if self.use_r1r2:
            tc, a_Rs, r1, r2, d = self.unpack_parameters(pv)
            imp, k = r1r2_to_imp_k(r1, r2, k_lo=PRIOR_K_MIN, k_up=PRIOR_K_MAX)
            
            # uniform priors
            if (r1 < 0.0) or (r1 > 1.0):
                if self.DEBUG:
                    print(f"Error (r1): 0<{r1:.2f}<1")
                return np.inf
            if np.any(r2 < 0.0) or np.any(r2 > 1.0):
                if self.DEBUG:
                    print(f"Error (r2): 0<{r2}<1")
                return np.inf
        else:
            # use imp and k
            tc, a_Rs, imp, k, d = self.unpack_parameters(pv)

            # uniform priors
            if np.any(k < PRIOR_K_MIN):
                if self.DEBUG:
                    print(f"Error (k_min): {k:.2f}<{PRIOR_K_MIN}")
                    print(f"You may need to decrease PRIOR_K_MIN={PRIOR_K_MIN}")
                return np.inf
            if np.any(k > PRIOR_K_MAX):
                if self.DEBUG:
                    print(f"Error (k_max): {k:.2f}>{PRIOR_K_MAX}")
                    print(f"You may need to increase PRIOR_K_MAX={PRIOR_K_MAX}")
                return np.inf
            if (imp < 0.0) or (imp > 1.0):
                if self.DEBUG:
                    print(f"Error (imp): 0<{imp:.2f}<1")
                return np.inf

        # derived
        inc = np.arccos(imp / a_Rs)
        tdur = tdur_from_per_imp_aRs_k(per, imp, a_Rs, np.mean(k))

        if a_Rs <= 0.0:
            if self.DEBUG:
                print(f"Error (a_Rs): {a_Rs:.2f}<0")
            return np.inf
        
        if imp / a_Rs >= 1.0:
            if self.DEBUG:
                print(f"Error (imp/a_Rs): {imp:.2f}/{a_Rs:.2f}>1")
            return np.inf
        # assumes transit occurs within window
        if (tc < self.obs_start) or (tc > self.obs_end):
            if self.DEBUG:
                errmsg = "Error (tc outside data): "
                errmsg += f"{self.obs_start:.4f}>{tc:.4f}>{self.obs_start:.4f}"
                print(errmsg)
            return np.inf
        # tc shouldn't be more or less than half a period
        if (tc > 0.0) and (abs(tc - self.obs_start) > per / 2.0):
            if self.DEBUG:
                errmsg = f"Error (tc more than half of period): "
                errmsg += f"{tc}-{self.obs_start:.4f} > {per/2:.4f}"
                print(errmsg)
            return np.inf
        chi2 = 0.0
        for i, b in enumerate(self.bands):
            t = self.times[b]
            f = self.fluxes[b]
            e = self.flux_errs[b]
            z = self.covariates[b]
            flux_tr = self.transit_models[b].evaluate_ps(
                k[i], self.ldc[b], tc, per, a_Rs, inc, e=0, w=0
            )
            flux_tr_time = d[i] * (t - tc) * flux_tr
            c = np.polyfit(z, (f - flux_tr_time) / flux_tr, self.lm_order)
            trend = np.polyval(c, z) + d[i] * (t - tc)
            model = trend * flux_tr
            chi2 = chi2 + np.sum((f - model) ** 2 / e**2)
            if self.DEBUG:
                print(f"k={k[i]:.2f}, ldc={self.ldc[b]}, tc={tc:.4f}, per={per:.4f}, a_Rs={a_Rs:.2f}, inc={inc:.2f}")
                print(f"flux_tr={flux_tr}")
                print(f"flux_tr_time={flux_tr_time}")
                print(f"c={c}")
                print(f"trend={trend}")
                print(f"model={model}")
                print(f"chi2 ({b}): {chi2}")
        # add normal priors
        if tc > 0.:
            chi2 += ((tc - self.tc[0])/self.tc[1])**2
        a_Rs0 = self.planet_params['a_Rs']
        if a_Rs > 0.:
            chi2 += ((a_Rs - a_Rs0[0])/a_Rs0[1])**2
        if tdur > 0.:
            chi2 += ((tdur - self.tdur[0]) / self.tdur[1]) ** 2
        return chi2

    def neg_likelihood(self, pv):
        return -self.get_chi2_transit(pv)

    def neg_loglikelihood(self, pv):
        raise NotImplementedError("unstable")
        return -np.log(self.get_chi2_transit(pv))

    def optimize_chi2_transit(self, p0, method="Nelder-Mead"):
        """
        Optimize parameters using `scipy.minimize`
        Uses previous optimized parameters if run again
        """
        if hasattr(self, "opt_result"):
            pv = self.opt_result.x
        else:
            assert len(p0) == self.ndim
            pv = p0

        self.opt_result = minimize(self.get_chi2_transit, pv, method=method)
        if self.opt_result.success:
            print("Optimization successful!")
            print("---------------------")
            print("Optimized parameters:")
            for n, i in zip(self.model_params_names, self.opt_result.x):
                print(f"{n}: {i:.2f}")
            self.optimum_params = self.opt_result.x
        else:
            print("Caution: Optimization **NOT** successful!")
        return self.opt_result.x

    def sample_mcmc(self, pv=None, nsteps=1_000, nwalkers=None):
        """
        pv : list
            parameter vector (uses optimized values if None)
        """
        if self.DEBUG:
            print("Setting DEBUG=False.")
            self.DEBUG = False
        self.nwalkers = 10 * self.ndim if nwalkers is None else nwalkers
        # if hasattr(self, 'sampler'):
        #     params = self.sampler
        self.nsteps = nsteps
        if pv:
            params = pv
        else:
            if hasattr(self, 'optimum_params'):
                params = self.optimum_params 
            else:
                errmsg = "Run `optimize_chi2_transit()` first or provide pv"
                raise ValueError(errmsg)
        assert len(params) == self.ndim
        pos = [
            params + 1e-5 * np.random.randn(self.ndim) for i in range(self.nwalkers)
        ]
        with Pool(self.ndim) as pool:
            self.sampler = emcee.EnsembleSampler(
                self.nwalkers, self.ndim, self.neg_likelihood, pool=pool
            )
            state = self.sampler.run_mcmc(pos, self.nsteps // 2, progress=True)
            # if reset:
            self.sampler.reset()
            self.sampler.run_mcmc(state, self.nsteps, progress=True)

        # Extract and analyze the results
        self._analyze_mcmc_results()
        
    def _analyze_mcmc_results(self):
        log_prob = self.sampler.get_log_prob()
        argmax = np.argmax(log_prob)
        self.best_fit_params = self.sampler.flatchain[argmax]
        # compute bic
        j = int(argmax / self.nwalkers)
        i = argmax - self.nwalkers * j
        self.chi2_best = -log_prob[j, i]
        # print(chi2_best)
        npar_tr = len(self.best_fit_params)
        # print('ndata = ', self.ndata)
        # print('npar(transit+linear) = ', npar_tr)
        self.bic_transit = self.chi2_best + npar_tr * np.log(self.ndata)
        # print('BIC(transit+linear) = ', bic_tr)
        if not hasattr(self, "bic_lin"):
            _ = self.optimize_chi2_linear_baseline()
        self.bic_delta = self.bic_lin - self.bic_transit
        # print('delta_BIC = ', delta_bic)

    def get_delta_bic(self):
        p0 = self.optimize_chi2_linear_baseline()
        ll_no_transit = self.get_chi2_linear_baseline(p0)
        pv = self.best_fit_params
        ll_with_transit = self.get_chi2_transit(pv)
        ndim = len(self.best_fit_params)

        def delta_bic(ll1, ll2, d1, d2, n):
            return ll2 - ll1 + 0.5 * (d1 - d2) * np.log(n)
        
        return delta_bic(ll_no_transit, ll_with_transit, 4, ndim, self.ndata)

    def get_mcmc_samples(self, discard=1, thin=1):
        """
        samples are converted from r1,r2 to imp and k
        """
        # FIXME: using get_chain() overwrites the chain somehow!
        # fc = self.sampler.get_chain(flat=True, discard=discard, thin=thin).copy()
        if self.mcmc_samples is None:
            fc = self.sampler.flatchain.copy()
            param_names = self.model_params_names.copy()
            fc = fc.reshape(self.nsteps, self.nwalkers, -1)
            fc = fc[discard::thin].reshape(-1, self.ndim)
            df = pd.DataFrame(fc, columns=self.model_params_names)
            df["tc"] = df["tc"] + self.time_offset
            if self.use_r1r2:
                print("Converting r1,r2 --> imp,k")
                r1s = df["r1"].values
                if self.model == "chromatic":
                    for b in self.bands:
                        col = f"r2_{b}"
                        r2s = df[col].values
                        imps = np.zeros_like(r1s)
                        ks = np.zeros_like(r1s)
                        for i, (r1, r2) in enumerate(zip(r1s, r2s)):
                            imps[i], ks[i] = r1r2_to_imp_k(
                                r1, r2, k_lo=PRIOR_K_MIN, k_up=PRIOR_K_MAX
                            )
                        df[f"k_{b}"] = ks
                        df = df.drop(labels=col, axis=1)
                else:
                    col = "r2"
                    r2s = df[col].values
                    imps = np.zeros_like(r1s)
                    ks = np.zeros_like(r1s)
                    for i, (r1, r2) in enumerate(zip(r1s, r2s)):
                        imps[i], ks[i] = r1r2_to_imp_k(
                            r1, r2, k_lo=PRIOR_K_MIN, k_up=PRIOR_K_MAX
                        )
                    df["k"] = ks
                    df = df.drop(labels=col, axis=1)
                df["imp"] = imps
                param_names = [s.replace("r1", "imp") for s in param_names]
                param_names = [s.replace("r2", "k") for s in param_names]
            self._mcmc_samples = df[param_names]
            return df[param_names]
        else:
            return self.mcmc_samples[discard::thin]

    @property
    def mcmc_samples(self):
        return self._mcmc_samples

    def get_upsampled_transit_models(self, pv, npoints=200):
        """
        pv : list
            parameter vector from MCMC or optimization

        returns a dict = {band: (time,flux_transit)}
        """
        assert len(pv) == self.ndim
        # unpack fixed parameters
        per = self.period[0]
        # unpack free parameters
        if self.use_r1r2:
            tc, a_Rs, r1, r2, d = self.unpack_parameters(pv)
            imp, k = r1r2_to_imp_k(r1, r2, k_lo=PRIOR_K_MIN, k_up=PRIOR_K_MAX)
        else:
            tc, a_Rs, imp, k, d = self.unpack_parameters(pv)
        # derived
        inc = np.arccos(imp / a_Rs)

        models = {}
        for i, b in enumerate(self.bands):
            t = self.times[b]
            xmodel = np.linspace(np.min(t), np.max(t), npoints)
            tmodel = QuadraticModel()
            tmodel.set_data(xmodel)
            ymodel = tmodel.evaluate_ps(
                k[i], self.ldc[b], tc, per, a_Rs, inc, e=0, w=0
            )
            models[b] = (xmodel, ymodel)
        return models

    def get_trend_models(self, pv):
        """
        pv : list
            parameter vector from MCMC or optimization

        returns a dict {band: (time,flux_transit)}
        """
        assert len(pv) == self.ndim
        # unpack fixed parameters
        per = self.period[0]
        # unpack free parameters
        if self.use_r1r2:
            tc, a_Rs, r1, r2, d = self.unpack_parameters(pv)
            imp, k = r1r2_to_imp_k(r1, r2, k_lo=PRIOR_K_MIN, k_up=PRIOR_K_MAX)
        else:
            tc, a_Rs, imp, k, d = self.unpack_parameters(pv)
        # derived
        inc = np.arccos(imp / a_Rs)

        models = {}
        for i, b in enumerate(self.bands):
            t = self.times[b]
            f = self.fluxes[b]
            z = self.covariates[b]

            flux_tr = self.transit_models[b].evaluate_ps(
                k[i], self.ldc[b], tc, per, a_Rs, inc, e=0, w=0
            )
            flux_tr_time = d[i] * (t - tc) * flux_tr
            c = np.polyfit(z, (f - flux_tr_time) / flux_tr, self.lm_order)
            trend = np.polyval(c, z) + d[i] * (t - tc)
            models[b] = trend
        return models

    def plot_raw_data(self, binsize=600 / 86400, figsize=(10, 10), ylims=None):
        fig = plt.figure(figsize=figsize)
        ncol = 2 if self.nband > 1 else 1
        nrow = 2 if self.nband > 2 else 1
        for i, b in enumerate(self.bands):
            t0 = np.min(self.fluxes[b])
            ax = fig.add_subplot(ncol, nrow, i + 1)
            ax.plot(
                self.times_raw[b] - self.time_offset,
                self.fluxes_raw[b],
                "k.",
                alpha=0.1,
                label="raw data",
            )
            tbin, ybin, yebin = binning_equal_interval(
                self.times[b], self.fluxes[b], self.flux_errs[b], binsize, t0
            )
            ax.errorbar(tbin, ybin, yerr=yebin, marker="o", c=colors[b], ls="")
            if self.mask_start or self.mask_end:
                mask = self.masks[b]
                ax.plot(
                    self.times_raw[b][mask] - self.time_offset,
                    self.fluxes_raw[b][mask],
                    "k.",
                    label="masked data",
                )
            ax.set_title(f"{b}-band")
            if ylims:
                ax.set_ylim(*ylims)
        ax.legend()
        return fig

    def plot_lightcurves(self, pv, binsize=600 / 86400, ylims=None, figsize=(8, 5)):
        """
        pv : list
            parameter vector from MCMC or optimization

        Raw and detrended lightcurves using `pv`

        See also `plot_detrended_data_and_transit()`
        """
        assert len(pv) == self.ndim
        # unpack fixed parameters
        per = self.period[0]
        # unpack free parameters
        if self.use_r1r2:
            tc, a_Rs, r1, r2, d = self.unpack_parameters(pv)
            imp, k = r1r2_to_imp_k(r1, r2, k_lo=PRIOR_K_MIN, k_up=PRIOR_K_MAX)
        else:
            tc, a_Rs, imp, k, d = self.unpack_parameters(pv)
        # derived
        inc = np.arccos(imp / a_Rs)
        depth = self.planet_params["rprs"][0] ** 2

        trends = self.get_trend_models(pv)
        transits = self.get_upsampled_transit_models(pv)
        for i, b in enumerate(self.bands):
            fig, ax = plt.subplots(1, 2, sharey=True, figsize=figsize)
            t = self.times[b]
            f = self.fluxes[b]
            e = self.flux_errs[b]
            # z = self.covariates[b]
            t0 = np.min(t)
            # transit
            flux_tr = self.transit_models[b].evaluate_ps(
                k[i], self.ldc[b], tc, per, a_Rs, inc, e=0, w=0
            )

            # raw data and binned
            tbin, ybin, _ = binning_equal_interval(t, f, e, binsize, t0)

            ax[0].plot(t, f, ".k", alpha=0.1)
            ax[0].plot(tbin, ybin, "o", color=colors[b], alpha=0.5)
            # flux with trend
            ax[0].plot(t, flux_tr * trends[b], lw=3, c=colors[b])
            ax[0].set_xlabel(f"BJD-{self.time_offset}")
            ax[0].set_ylabel("Normalized Flux")
            if ylims:
                ax[0].set_ylim(*ylims)

            # detrended and binned
            tbin, ybin, _ = binning_equal_interval(t, f / trends[b], e, binsize, t0)
            ax[1].plot(t, f / trends[b], ".k", alpha=0.1)
            ax[1].plot(tbin, ybin, "o", color=colors[b], alpha=0.5)
            # upsampled transit
            xmodel, ymodel = transits[b]
            ax[1].plot(xmodel, ymodel, lw=3, c=colors[b])
            _ = self.plot_ing_egr(ax=ax[1], ymin=0.9, ymax=1.0, color="C0")
            ax[1].axhline(
                1 - depth,
                color="blue",
                linestyle="dashed",
                label="TESS",
                alpha=0.5,
            )
            if ylims:
                ax[1].set_ylim(*ylims)
            ax[1].set_xlabel(f"BJD-{self.time_offset}")
            ax[1].legend(loc="best")
            fig.suptitle(f"{b}-band")
        return fig

    def plot_chain(self, start=0, end=None, figsize=(10, 10)):
        """
        visualize MCMC walkers

        start : int
            parameter id (0 means first)
        end : int
            parameter id
        """
        end = self.ndim if end is None else end
        fig, axes = plt.subplots(end - start, figsize=figsize, sharex=True)
        samples = self.sampler.get_chain()
        for i in np.arange(start, end):
            ax = axes[i]
            ax.plot(samples[:, :, start + i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(self.model_params_names[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
        axes[-1].set_xlabel("step number")
        return fig

    def plot_corner(self, transform=True, truths=None, discard=1, thin=1, start=0, end=None):
        """
        corner plot of MCMC chain

        start : int
            parameter id (0 means first)
        end : int
            parameter id
        """
        end = self.ndim if end is None else end
        if self.use_r1r2 and transform:
            print("Converting r1,r2 --> imp,k")
            # get a copy, otherwise chain is overwritten
            df = self.get_mcmc_samples(discard=discard, thin=thin).copy()
            df["tc"] = df["tc"].values - self.time_offset
            labels = df.columns.values.copy()
            labels[0] = f"tc-{self.time_offset:,}"
            fig = corner.corner(
                df.iloc[:, start:end], 
                labels=labels[start:end], 
                show_titles=True, 
                truths=truths
            )
        else:
            labels = self.model_params_names.copy()
            labels[0] = f"tc-{self.time_offset:,}"
            # fc = self.sampler.get_chain(flat=True, discard=discard, thin=thin)
            fc = self.sampler.flatchain.copy()
            if discard > 1:
                fc = fc.reshape(self.nsteps, self.nwalkers, -1)
                fc = fc[discard::thin].reshape(-1, self.ndim)
            fig = corner.corner(
                fc[:, start:end], 
                labels=labels[start:end], 
                show_titles=True, 
                truths=truths
            )
        return fig

    def plot_detrended_data_and_transit(
        self,
        pv: list,
        title: str = None,
        xlims: tuple = None,
        ylims: tuple = None,
        binsize: float = 600 / 86400,
        msize: int = 5,
        font_size: int = 20,
        title_height: float = 0.95,
        figsize: tuple = (10, 10),
    ):
        """
        pv : list
            parameter vector (uses optimized values if None)

        - 2x2 plot of detrended data with transit model
        - time is in units of hours

        See also `plot_detrended_data_and_transit()`
        """
        title = (
            f"{self.name}{self.alias} (TIC{self.ticid}{self.alias})"
            if title is None
            else title
        )
        ncols = 2 if self.nband > 1 else 1
        nrows = 2 if self.nband > 2 else 1
        fig, axs = plt.subplots(
            ncols,
            nrows,
            figsize=figsize,
            sharey="row",
            sharex="col",
            tight_layout=True,
        )
        ax = axs.flatten() if nrows>1 else axs
        depth = self.planet_params["rprs"][0] ** 2
        # unpack fixed parameters
        # per = self.period[0]
        # unpack free parameters
        tc, _, _, _, _ = self.unpack_parameters(pv)

        for i, b in enumerate(self.bands):
            t = self.times[b]
            f = self.fluxes[b]
            e = self.flux_errs[b]
            detrended_flux = f / self.get_trend_models(pv)[b]
            ax[i].plot((t - tc) * 24, detrended_flux, "k.", alpha=0.2)
            # raw data and binned
            tbin, ybin, yebin = binning_equal_interval(
                t, detrended_flux, e, binsize, tc
            )
            ax[i].errorbar(
                (tbin - tc) * 24, ybin, yerr=yebin, fmt="ok", markersize=msize
            )
            xmodel, ymodel = self.get_upsampled_transit_models(pv, npoints=500)[b]
            ax[i].plot((xmodel - tc) * 24, ymodel, "-", lw=3, color=colors[b])
            ax[i].axhline(
                1 - depth,
                color="blue",
                linestyle="dashed",
                label="TESS",
                alpha=0.5,
            )
            if self.nband % 2 == 1:
                ax[i].set_ylabel("Relative Flux", fontsize=font_size * 0.8)
            ax[i].set_xlabel(
                "Time from transit center (hours)", fontsize=font_size * 0.8
            )
            if (self.nband > 2) and (i < 2):
                ax[i].set_xlabel("")

            if xlims is None:
                xmin, xmax = ax[i].get_xlim()
            else:
                ax[i].set_xlim(*xlims)
                xmin, xmax = xlims

            if ylims is None:
                ymin, ymax = ax[i].get_ylim()
            else:
                ax[i].set_ylim(*ylims)
                ymin, ymax = ylims
            ax[i].xaxis.set_minor_locator(AutoMinorLocator(5))
            ax[i].yaxis.set_minor_locator(AutoMinorLocator(5))
            ax[i].tick_params(labelsize=font_size * 0.8)
            tx = np.min(t) + (np.max(t) - np.min(t)) * 0.02
            # tx = xmin + (xmax - xmin) * 0.75
            ty = ymin + (ymax - ymin) * 0.9
            ax[i].text(
                tx, ty, f"{b}-band", color=colors[b], fontsize=font_size * 0.8
            )
        ax[1].set_title(
            f"{self.date4plot}, {self.inst}", loc="right", fontsize=font_size
        )
        ax[i].legend(loc="best")
        fig.suptitle(title, fontsize=font_size, y=title_height)
        return fig

    def plot_posteriors(
        self,
        title: str = None,
        figsize: tuple = (12, 5),
        font_size: float = 12,
        nsigma: float = 3,
        save: bool = False,
        suffix: str = ".pdf",
    ):
        """
        plot Rp/Rs, Tc and impact parameter posteriors
        """
        errmsg = "Valid only for chromatic model"
        if not self.model == "chromatic":
            raise ValueError(errmsg)
        title = (
            f"{self.name}{self.alias} (TIC{self.ticid}{self.alias})"
            if title is None
            else title
        )

        df = self.get_mcmc_samples()

        fig, axs = plt.subplots(1, 3, figsize=figsize)
        ax = axs.flatten()

        ############# Rp/Rs
        for i, b in enumerate(self.bands):
            k = df["k_" + b].values
            k_med = df["k_" + b].median()
            k_percs = get_percentile(k)
            # med, low1, hig1, low2, hig2, low3, hig3
            k_err1 = k_percs[0] - k_percs[1], k_percs[2] - k_percs[0]
            k_err2 = k_percs[0] - k_percs[3], k_percs[4] - k_percs[0]
            k_err3 = k_percs[0] - k_percs[5], k_percs[6] - k_percs[0]
            ax[0].errorbar(
                i,
                k_med,
                yerr=np.c_[k_err1].T,
                elinewidth=40,
                fmt="none",
                alpha=0.5,
                zorder=1,
                color=colors[b],
            )
            ax[0].errorbar(
                i,
                k_med,
                yerr=np.c_[k_err2].T,
                elinewidth=40,
                fmt="none",
                alpha=0.3,
                zorder=2,
                color=colors[b],
            )
            ax[0].errorbar(
                i,
                k_med,
                yerr=np.c_[k_err3].T,
                elinewidth=40,
                fmt="none",
                alpha=0.1,
                zorder=3,
                color=colors[b],
            )
            print(f"Rp/Rs({b})^2 = {1e3*k_med**2:.2f} ppt")

        k0 = self.planet_params["rprs"][0]
        ax[0].axhline(k0, linestyle="dashed", color="black", label="TESS")
        ax[0].legend(loc="best")
        ax[0].set_xlim(-0.5, 3.5)
        ax[0].set_xticks(range(self.nband))
        ax[0].set_xticklabels(self.bands)
        ax[0].set_xlabel("Band", fontsize=font_size * 1.5)
        ax[0].set_ylabel("Radius ratio", fontsize=font_size * 1.5)

        ax[0].text(
            0.0,
            1.12,
            title,
            horizontalalignment="left",
            verticalalignment="center",
            transform=ax[0].transAxes,
            fontsize=font_size * 1.5,
        )
        text = f"{self.model.title()} transit fit, "
        text += f"$\Delta$BIC (non-transit - transit) = {self.bic_delta:.1f}"
        ax[0].text(
            0.0,
            1.05,
            text,
            horizontalalignment="left",
            verticalalignment="center",
            transform=ax[0].transAxes,
            fontsize=font_size,
        )
        ############# Mid-transit
        # posterior
        tc = df["tc"].values - self.time_offset
        n = self.plot_kde(tc, ax=ax[1], label="Posterior")
        # prediction
        tc0, tc_err0 = self.tc
        xmodel = np.linspace(tc0 - nsigma * tc_err0, tc0 + nsigma * tc_err0, 200)
        ymodel = n * np.exp(-((xmodel - tc0) ** 2) / tc_err0**2)
        ax[1].plot(xmodel, ymodel, label="Prediction", color="C1", lw=3, zorder=0)
        ax[1].set_xlabel(
            f"Tc (BJD-{self.time_offset:.0f})",
            fontsize=font_size * 1.5,
        )
        ax[1].legend(loc="best")

        imp = df["imp"].values
        _ = self.plot_kde(imp, ax=ax[2], color="C0")
        ax[2].set_xlabel("Impact parameter", fontsize=font_size * 1.5)
        ax[2].set_title(
            f"{self.date4plot}, {self.inst}", loc="right", fontsize=font_size * 1.5
        )
        if save:
            outfile = f"{self.outdir}/{self.outfile_prefix}_posteriors"
            outfile += suffix if self.mask_start is None else f"_mask{suffix}"
            savefig(fig, outfile, dpi=300, writepdf=False)
        return fig

    def plot_kde(self, vals, ax=None, color="C0", label="", fill=True, alpha=0.5):
        if ax is None:
            _, ax = plt.subplots()
        kde = stats.gaussian_kde(vals)
        xmodel = np.linspace(min(vals), max(vals), 200)
        ax.plot(xmodel, kde(xmodel), lw=3, color=color, zorder=5, label=label)
        if fill:
            ax.fill_between(xmodel, 0, kde(xmodel), color=color, alpha=alpha)
        return max(kde(xmodel))

    def plot_final_fit(
            self,
            discard: int = 1,
            thin: int = 1,
            nsamples: int = 100,
            ylims_top: tuple = (0.9, 1.02),
            ylims_bottom: tuple = (0.9, 1.02),
            msize: int = 5,
            font_size: int = 25,
            title: str = None,
            figsize: tuple = (16, 12),
            binsize: float = 600 / 86400,
            save: bool = False,
            suffix: str = ".pdf",
        ):
            ymin1, ymax1 = ylims_top
            ymin2, ymax2 = ylims_bottom

            fig, axs = plt.subplots(
                2, self.nband, figsize=figsize, sharey="row", sharex="col"
            )
            plt.subplots_adjust(hspace=0.1, wspace=0)

            # unpack fixed parameters
            per = self.period[0]
            # unpack free parameters
            if not hasattr(self, "best_fit_params"):
                raise ValueError("Run `sample_mcmc()` first.")

            pv = self.best_fit_params
            if self.use_r1r2:
                tc_best, a_Rs_best, r1, r2, d_best = self.unpack_parameters(pv)
                imp_best, k_best = r1r2_to_imp_k(
                    r1, r2, k_lo=PRIOR_K_MIN, k_up=PRIOR_K_MAX
                )
            else:
                tc_best, a_Rs_best, imp_best, k_best, d_best = self.unpack_parameters(pv)

            # derived
            inc_best = np.arccos(imp_best / a_Rs_best)

            trends_best = self.get_trend_models(self.best_fit_params)
            transits_best = self.get_upsampled_transit_models(self.best_fit_params)

            # fc = self.sampler.get_chain(flat=True, discard=discard, thin=thin)
            fc = self.sampler.flatchain.copy()
            if discard > 1:
                fc = fc.reshape(self.nsteps, self.nwalkers, -1)
                fc = fc[discard::thin].reshape(-1, self.ndim)

            for i, b in enumerate(self.bands):
                ax1 = axs[0, i] if self.nband>1 else axs[0] 
                ax2 = axs[1, i] if self.nband>1 else axs[1]

                t = self.times[b]
                f = self.fluxes[b]
                e = self.flux_errs[b]
                z = self.covariates[b]
                t0 = np.min(t)

                # raw and binned data
                tbin, ybin, yebin = binning_equal_interval(t, f, e, binsize, t0)
                ax1.plot(t, f, ".k", alpha=0.1)
                ax1.errorbar(tbin, ybin, yerr=yebin, fmt="ok", markersize=msize)

                # plot each random mcmc samples
                rand = np.random.randint(len(fc), size=nsamples)
                for j in range(len(rand)):
                    idx = rand[j]
                    # unpack free parameters
                    if self.use_r1r2:
                        tc, a_Rs, r1 = fc[idx, : self.k_idx]
                        if self.model == "chromatic":
                            r2 = fc[idx, self.k_idx : self.d_idx]
                        elif self.model == "achromatic":
                            r2 = np.zeros(self.nband) + fc[idx, self.k_idx]
                        imp, k = r1r2_to_imp_k(
                            r1, r2, k_lo=PRIOR_K_MIN, k_up=PRIOR_K_MAX
                        )
                    else:
                        tc, a_Rs, imp = fc[idx, : self.k_idx]
                        if self.model == "chromatic":
                            k = fc[idx, self.k_idx : self.d_idx]
                        elif self.model == "achromatic":
                            k = np.zeros(self.nband) + fc[idx, self.k_idx]
                    d = fc[idx, self.d_idx : self.d_idx + self.nband]
                    # derived parameters
                    inc = np.arccos(imp / a_Rs)
                    # transit
                    flux_tr = self.transit_models[b].evaluate_ps(
                        k[i], self.ldc[b], tc, per, a_Rs, inc, e=0, w=0
                    )
                    flux_tr_time = d[i] * (t - tc) * flux_tr
                    c = np.polyfit(z, (f - flux_tr_time) / flux_tr, self.lm_order)
                    # transit with trend
                    ax1.plot(
                        t,
                        flux_tr * (np.polyval(c, z) + d[i] * (t - tc)),
                        alpha=0.05,
                        color=colors[b],
                    )

                # best-fit transit model
                flux_tr = self.transit_models[b].evaluate_ps(
                    k_best[i],
                    self.ldc[b],
                    tc_best,
                    per,
                    a_Rs_best,
                    inc_best,
                    e=0,
                    w=0,
                )

                tbin, ybin, yebin = binning_equal_interval(
                    t, f / trends_best[b], e, binsize, t0
                )
                # detrended flux
                ax2.plot(t, f / trends_best[b], ".k", alpha=0.1)
                ax2.errorbar(tbin, ybin, yerr=yebin, fmt="ok", markersize=msize)
                # super sampled best-fit transit model
                xmodel, ymodel = transits_best[b]
                ax2.plot(xmodel, ymodel, color=colors[b], linewidth=3)
                ax1.yaxis.set_minor_locator(AutoMinorLocator(5))
                ax2.yaxis.set_minor_locator(AutoMinorLocator(5))
                ax1.xaxis.set_minor_locator(AutoMinorLocator(5))
                ax2.xaxis.set_minor_locator(AutoMinorLocator(5))
                ax1.set_ylim(ymin1, ymax1)
                ax2.set_ylim(ymin2, ymax2)

                # tx = np.min(t) + (np.max(t) - np.min(t)) * 0.75
                tx = np.min(t) + (np.max(t) - np.min(t)) * 0.02
                ty = ymin1 + (ymax1 - ymin1) * 0.9
                ax1.text(
                    tx, ty, f"{b}-band", color=colors[b], fontsize=font_size * 0.6
                )
                tx = np.min(t) + (np.max(t) - np.min(t)) * 0.02
                ty = ymin2 + (ymax2 - ymin2) * 0.8
                ax2.text(tx, ty, "Detrended", fontsize=font_size * 0.6)

                rms = np.std(
                    f - flux_tr * (np.polyval(c, z) + d_best[i] * (t - tc_best))
                )
                rms_text = f"rms = {rms:.4f}"
                ty = ymin2 + (ymax2 - ymin2) * 0.1
                ax2.text(tx, ty, rms_text, fontsize=font_size * 0.6)
                depth = self.planet_params["rprs"][0] ** 2
                ax2.axhline(
                    1 - depth,
                    color="blue",
                    linestyle="dashed",
                    label="TESS",
                    alpha=0.5,
                )
                _ = self.plot_ing_egr(ax=ax2, ymin=0.9, ymax=1.0, color="C0")
                if i == 0:
                    ax1.set_ylabel("Flux ratio", fontsize=font_size)
                    ax2.set_ylabel("Flux ratio", fontsize=font_size)
                    ax1.tick_params(labelsize=16)
                    ax2.tick_params(labelsize=16)
                    target_name = (
                        f"{self.name}{self.alias} (TIC{self.ticid}{self.alias})"
                        if title is None
                        else title
                    )
                    ax1.text(
                        0.0,
                        1.14,
                        target_name,
                        horizontalalignment="left",
                        verticalalignment="center",
                        transform=ax1.transAxes,
                        fontsize=font_size,
                    )
                    text = f"{self.model.title()} transit fit,  "
                    text += f"$\Delta$BIC (non-transit - transit) = {self.bic_delta:.1f}"
                    ax1.text(
                        0.0,
                        1.05,
                        text,
                        horizontalalignment="left",
                        verticalalignment="center",
                        transform=ax1.transAxes,
                        fontsize=font_size * 0.6,
                    )
                if (i > 0) and (i == self.nband - 1):
                    ax1.set_title(
                        f"{self.date4plot}, {self.inst}",
                        loc="right",
                        fontsize=font_size * 0.8,
                    )
                if i > 0:
                    ax1.tick_params(
                        labelleft=False,
                        labelright=False,
                        labeltop=False,
                        labelsize=16,
                    )
                    ax2.tick_params(
                        labelleft=False,
                        labelright=False,
                        labeltop=False,
                        labelsize=16,
                    )
                ax2.set_xlabel(
                    f"BJD - {self.time_offset:.0f}",
                    labelpad=30,
                    fontsize=font_size,
                )
            ax2.legend(loc="lower right", fontsize=12)
            if save:
                outfile = f"{self.outdir}/{self.outfile_prefix}_transit_fit"
                outfile += suffix if self.mask_start is None else f"_mask{suffix}"
                savefig(fig, outfile, dpi=300, writepdf=False)
            return fig

    def plot_radii(
        self, rstar, rstar_err, fill=False, unit=u.Rjup, figsize=(5, 5), alpha=0.5
    ):
        fig, ax = plt.subplots(figsize=figsize)
        df = self.get_mcmc_samples()
        rstar_samples = np.random.normal(rstar, rstar_err, size=len(df))
        if self.model == "chromatic":
            cols = [f"k_{b}" for b in self.bands]
            d = df[cols].apply(lambda x: x * rstar_samples * u.Rsun.to(unit))
            d.columns = self.bands
            for b in self.bands:
                _ = self.plot_kde(
                    d[b].values,
                    ax=ax,
                    color=colors[b],
                    label=b,
                    alpha=alpha,
                    fill=fill,
                )
        else:
            d = df["k"] * rstar_samples * u.Rsun.to(unit)
            d.name = "achromatic Rp"
            b = self.bands[0]
            _ = self.plot_kde(
                d.values, ax=ax, color=colors[b], alpha=alpha, fill=fill
            )

        ax.set_xlabel(f"Companion radius ({unit._format['latex']})")
        ax.set_title(
            f"(assuming Rs={rstar:.2f}+/-{rstar_err:.2f}" + r" $R_{\odot}$)"
        )
        Rp_tfop = self.planet_params["rprs"][0] * rstar * u.Rsun.to(unit)
        ax.axvline(
            Rp_tfop, 0, 1, c="k", ls="--", lw=2, label=f"Rp={Rp_tfop:.1f}\n(TFOP)"
        )
        ax.set_ylabel("Probability")
        ax.legend()
        return fig

    def plot_ing_egr(self, ax, ymin=0.9, ymax=1.0, color="C0"):
        """
        plot ingress and egress timings over detrended light curve plot
        """
        tdur, tdure = self.tdur
        ing, egr = self.ingress[0], self.egress[0]
        ax.axvspan(
            ing - tdure / 2,
            ing + tdure / 2,
            alpha=1,
            ymin=ymin,
            ymax=ymax,
            color=color,
        )
        ax.axvspan(
            ing - 3 * tdure / 2,
            ing + 3 * tdure / 2,
            alpha=0.5,
            ymin=ymin,
            ymax=ymax,
            color=color,
        )
        ax.axvspan(
            egr - tdure / 2,
            egr + tdure / 2,
            alpha=1,
            ymin=ymin,
            ymax=ymax,
            color=color,
        )
        ax.axvspan(
            egr - 3 * tdure / 2,
            egr + 3 * tdure / 2,
            alpha=0.5,
            ymin=ymin,
            ymax=ymax,
            color=color,
        )
        return ax

    def plot_fov(
        self,
        ra: float,
        dec: float,
        ref_fits_file_path: str,
        ref_obj_file_path: str,
        target_ID : int = None,
        target_color: str = "red",
        cIDs: list = None,
        cIDs_color: str = "blue",
        cmap: str = "gray",
        contrast: float = 0.5,
        text_offset: tuple = (0, 0),
        phot_aper_pix: int = 10,
        title: str = None,
        title_height: float = 1.0,
        font_size: float = 20,
        show_scale_bar: bool = True,
        marker_color: str = "yellow",
        scale_color: str = "white",
        figsize: tuple = (10, 10),
        show_grid: bool = True,
        save: bool = False,
        suffix: str = ".pdf",
    ):
        """
        Field of View given reference image produced by AFPHOT pipeline
        """
        dr, dd = text_offset

        header = fits.getheader(ref_fits_file_path)
        data = fits.getdata(ref_fits_file_path)
        wcs = WCS(header)

        columns = "id x y xpin ypix flux bkg".split()
        df = pd.read_csv(
            ref_obj_file_path,
            comment="#",
            names=columns,
            delim_whitespace=True,
        )
        star_ids = df["id"].values
        refxy = df[["x", "y"]].values
        coords = wcs.all_pix2world(refxy, 0)
        pixscale = header["PIXSCALE"]
        band = header["FILTER"]
        title = (
            f"{self.name}\n{self.inst} {band[0]}-band" if title is None else title
        )

        fig = plt.figure(figsize=figsize)
        fig.subplots_adjust(
            top=title_height
        )  # Adjusted subplots_adjust to give more space to the title
        ax = fig.add_subplot(111, projection=wcs)
        norm = ImageNormalize(data, interval=ZScaleInterval(contrast=contrast))
        ax.imshow(data, norm=norm, origin="lower", cmap=cmap)

        # target
        rad_marker = phot_aper_pix * pixscale
        if target_ID:
            c = SphericalCircle(
                (ra * u.deg, dec * u.deg),
                rad_marker * u.arcsec,
                edgecolor=target_color,
                facecolor="none",
                lw=3,
                zorder=10,
                transform=ax.get_transform("fk5"),
            )
            ax.add_patch(c)
            # ax.text(ra+dr, dec+dd, 'target', fontsize=20,
            #         color='red', transform=ax.get_transform('fk5'))
        for i, (r, d) in enumerate(coords):
            j = int(star_ids[i])
            mcolor = marker_color
            if cIDs:
                cIDs = [int(c) for c in cIDs]
                if j in cIDs:
                    mcolor = cIDs_color
            if target_ID:
                if j==int(target_ID):
                    mcolor = target_color
            c = SphericalCircle(
                (r * u.deg, d * u.deg),
                rad_marker * u.arcsec,
                edgecolor=mcolor,
                facecolor="none",
                lw=2,
                transform=ax.get_transform("fk5"),
            )
            ax.add_patch(c)
            ax.text(
                r + dr,
                d + dd,
                str(j),
                fontsize=20,
                color=mcolor,
                transform=ax.get_transform("fk5"),
            )
        ax.set_xlim(0, data.shape[1])
        ax.set_ylim(0, data.shape[0])
        if show_scale_bar:
            sx, sy = 400, 50
            ax.hlines(sy, sx - 60 / pixscale, sx, color=scale_color, lw=3)
            ax.annotate(
                text="1'",
                xy=(sx - 60 / pixscale, sy + 10),
                fontsize=font_size * 1.5,
                color=scale_color,
                xycoords="data",
            )
        fig.suptitle(title, y=title_height, fontsize=font_size)
        # coord_format="dd:mm:ss"
        # ax.coords[1].set_major_formatter(coord_format)
        # ax.coords[0].set_major_formatter(coord_format)
        ax.set_ylabel("Dec")
        ax.set_xlabel("RA")
        if show_grid:
            ax.grid()
        fig.tight_layout()
        if save:
            outfile = f"{self.outdir}/{self.outfile_prefix}_FOV{suffix}"
            savefig(fig, outfile, dpi=300, writepdf=False)
        return fig

    def plot_fov_zoom(
        self,
        ra: float,
        dec: float,
        ref_fits_file_path: str,
        ref_obj_file_path: str,
        zoom_rad_arcsec: float,
        show_target: bool = False,
        cmap: str = "gray",
        contrast: float = 0.5,
        text_offset: tuple = (0, 0),
        phot_aper_pix: int = 10,
        title: str = None,
        title_height: float = 1.0,
        font_size: float = 20,
        show_scale_bar: bool = True,
        bar_arcsec=None,
        marker_color: str = "yellow",
        scale_color: str = "w",
        figsize: tuple = (10, 10),
        show_grid: bool = True,
        save: bool = False,
        suffix: str = ".pdf",
    ):
        """
        Zoomed-in FOV
        """
        dr, dd = text_offset

        header = fits.getheader(ref_fits_file_path)
        data = fits.getdata(ref_fits_file_path)
        wcs = WCS(header)

        pixscale = header["PIXSCALE"]
        band = header["FILTER"]
        title = (
            f"{self.name}\n{self.inst} {band[0]}-band (zoomed-in)"
            if title is None
            else title
        )

        columns = "id x y xpin ypix flux bkg".split()
        df = pd.read_csv(
            ref_obj_file_path,
            comment="#",
            names=columns,
            delim_whitespace=True,
        )
        star_ids = df["id"].values
        refxy = df[["x", "y"]].values
        coords = wcs.all_pix2world(refxy, 0)
        # filter those within zoom_rad_arcsec
        sep = SkyCoord(coords * u.deg).separation(SkyCoord(ra, dec, unit="deg"))
        idx = sep < zoom_rad_arcsec * u.arcsec
        coords = coords[idx]
        star_ids = star_ids[idx]

        xy = wcs.all_world2pix(np.c_[ra, dec], 0)
        xpix, ypix = int(xy[0][0]), int(xy[0][1])
        dx = dy = round(zoom_rad_arcsec / pixscale)
        dcrop = data[ypix - dy : ypix + dy, xpix - dx : xpix + dx]
        wcscrop = wcs[ypix - dy : ypix + dy, xpix - dx : xpix + dx]

        fig = plt.figure(figsize=figsize)
        fig.subplots_adjust(top=title_height)
        ax = fig.add_subplot(111, projection=wcscrop)
        norm = ImageNormalize(dcrop, interval=ZScaleInterval(contrast=contrast))
        ax.imshow(dcrop, norm=norm, origin="lower", cmap=cmap)

        # target
        rad_marker = phot_aper_pix * pixscale
        if show_target:
            c = SphericalCircle(
                (ra * u.deg, dec * u.deg),
                rad_marker * u.arcsec,
                edgecolor="red",
                facecolor="none",
                lw=3,
                zorder=10,
                transform=ax.get_transform("fk5"),
            )
            ax.add_patch(c)
            ax.text(
                ra + dr,
                dec + dd,
                "target",
                fontsize=20,
                color="red",
                transform=ax.get_transform("fk5"),
            )
        for i, (r, d) in enumerate(coords):
            j = int(star_ids[i])
            c = SphericalCircle(
                (r * u.deg, d * u.deg),
                rad_marker * u.arcsec,
                edgecolor=marker_color,
                facecolor="none",
                lw=2,
                transform=ax.get_transform("fk5"),
            )
            ax.add_patch(c)
            ax.text(
                r + dr,
                d + dd,
                str(j),
                fontsize=20,
                color=marker_color,
                transform=ax.get_transform("fk5"),
            )
        ax.set_xlim(0, dcrop.shape[1])
        ax.set_ylim(0, dcrop.shape[0])
        if show_scale_bar:
            bar_arcsec = bar_arcsec if bar_arcsec else zoom_rad_arcsec // 2
            imsize = dcrop.shape
            sx, sy = int(imsize[0] * 0.3), int(imsize[1] * 0.1)
            ax.hlines(sy, sx - bar_arcsec / pixscale, sx, color=scale_color, lw=3)
            ax.annotate(
                text=f'{bar_arcsec}"',
                xy=(sx - bar_arcsec / pixscale, sy + 10),
                fontsize=font_size * 2,
                color=scale_color,
                xycoords="data",
            )
        fig.suptitle(title, y=title_height, fontsize=font_size)
        # coord_format="dd:mm:ss"
        # ax.coords[1].set_major_formatter(coord_format)
        # ax.coords[0].set_major_formatter(coord_format)
        ax.set_ylabel("Dec")
        ax.set_xlabel("RA")
        if show_grid:
            ax.grid()
        fig.tight_layout()
        if save:
            outfile = f"{self.outdir}/{self.outfile_prefix}_FOV_zoom{suffix}"
            savefig(fig, outfile, dpi=300, writepdf=False)
        return fig

    def plot_gaia_sources(
        self,
        fits_file_path: str,
        gaia_sources: pd.DataFrame,
        phot_aper_pix: int = 10,
        show_scale_bar: bool = True,
        bar_arcsec: float = 30,
        text_offset: tuple = (0, 0),
        fov_padding: float = 1.1,
        title_height: float = 1,
        figsize: tuple = (10, 10),
        title: str = None,
        marker_color: str = "yellow",
        scale_color: str = "w",
        cmap: str = "gray",
        contrast: float = 0.5,
        font_size: float = 20,
        show_grid: bool = True,
        save: bool = False,
        suffix: str = ".pdf",
    ):
        """
        plots gaia sources on the given fits image and zoomed up
        to the furthest gaia_source separation from target
        based on `distance` (in arcsec) column

        See also `Star.get_gaia_sources()`
        """
        dr, dd = text_offset
        header = fits.getheader(fits_file_path)
        data = fits.getdata(fits_file_path)
        wcs = WCS(header)
        pixscale = header["PIXSCALE"]
        band = header["FILTER"]
        title = (
            f"{self.name}\n{self.inst} {band[0]}-band" if title is None else title
        )

        # zoomed-in image
        ra, dec = gaia_sources.loc[0, ["ra", "dec"]].values
        xy = wcs.all_world2pix(np.c_[ra, dec], 0)
        xpix, ypix = int(xy[0][0]), int(xy[0][1])
        rad_arcsec = fov_padding * gaia_sources["distance_arcsec"].max()
        dx = dy = round(rad_arcsec / pixscale)
        dcrop = data[ypix - dy : ypix + dy, xpix - dx : xpix + dx]
        wcscrop = wcs[ypix - dy : ypix + dy, xpix - dx : xpix + dx]

        fig = plt.figure(figsize=figsize)
        fig.subplots_adjust(
            top=title_height
        )  # Adjusted subplots_adjust to give more space to the title
        ax = fig.add_subplot(111, projection=wcscrop)
        norm = ImageNormalize(dcrop, interval=ZScaleInterval(contrast=contrast))
        ax.imshow(dcrop, norm=norm, origin="lower", cmap=cmap)

        coords = gaia_sources[["ra", "dec"]].values
        rad_marker = phot_aper_pix * pixscale
        for i, (r, d) in enumerate(coords):
            if i == 0:
                # target
                lw = 3
                mc = "r"
            else:
                lw = 2
                mc = marker_color
            c = SphericalCircle(
                (r * u.deg, d * u.deg),
                rad_marker * u.arcsec,
                edgecolor=mc,
                facecolor="none",
                lw=lw,
                transform=ax.get_transform("fk5"),
            )
            ax.add_patch(c)
            ax.text(
                r + dr,
                d + dd,
                str(i + 1),
                fontsize=20,
                color=mc,
                transform=ax.get_transform("fk5"),
            )
        # ax.set_xlim(0, dcrop.shape[1])
        # ax.set_ylim(0, dcrop.shape[0])
        if show_scale_bar:
            imsize = dcrop.shape
            sx, sy = int(imsize[0] * 0.3), int(imsize[1] * 0.1)
            ax.hlines(sy, sx - bar_arcsec / pixscale, sx, color=scale_color, lw=3)
            ax.annotate(
                text=f'{bar_arcsec}"',
                xy=(sx - bar_arcsec / pixscale, sy + 10),
                fontsize=font_size * 2,
                color=scale_color,
                xycoords="data",
            )
        fig.suptitle(title, y=title_height, fontsize=font_size)
        # coord_format="dd:mm:ss"
        # ax.coords[1].set_major_formatter(coord_format)
        # ax.coords[0].set_major_formatter(coord_format)
        ax.set_ylabel("Dec")
        ax.set_xlabel("RA")
        if show_grid:
            ax.grid()
        fig.tight_layout()
        if save:
            outfile = f"{self.outdir}/{self.outfile_prefix}_gaia_sources{suffix}"
            savefig(fig, outfile, dpi=300, writepdf=False)
        return fig

    def get_report(self, mcmc_samples_fp=None) -> str:
        inst = self.inst.lower()
        txt = f"Title: TIC {self.ticid}{self.alias} ({self.toi_name}) on UT 20{self.date} "
        if inst=='sinistro':
            txt += "from LCO-1m-"
        elif inst=='muscat2':
            txt += "from TCS-1.52m-"
        elif inst=='muscat3':
            txt += "from FTN-2m-"
        elif inst=='muscat4':
            txt += "from FTS-2m-"
        txt += f"{self.inst} in {','.join(self.bands)}\n\n"
        txt += f"We observed a full/ingress/egress on 20{self.date} UT in {','.join(self.bands)} "

        if mcmc_samples_fp:
            df = pd.read_csv(mcmc_samples_fp)
        else:
            df = self.get_mcmc_samples()
        tc = df["tc"].median()
        tc_sig = df["tc"].std()
        tc0 = self.tc
        tdiff = tc - tc0[0] - self.time_offset
        tdiff_mins = abs(tdiff) * 60 * 24
        timing = "late" if tdiff > 0 else "early"
        tsigma = tdiff / tc0[1]

        depths = []
        for b in self.bands:
            m = df["k_" + b].median()
            depth = 1e3 * m**2
            # print(f'Rp/Rs({b})^2 = {depth:.2f} ppt')
            depths.append(round(depth, 1))
        # print(depths)
        txt += f"and detected a {tdiff_mins:.1f}-min {timing} ({abs(tsigma):.1f} sigma), [Rp/Rs]^2: "
        txt += f"{','.join(list(map(str, depths)))} ppt event using XXX\" uncontaminated/contaminated aperture.\n\n"
        
        try:
            dt = 2_450_000
            txt += "Typical FWHM: NA\n"
            txt += f"Predicted Tc: {tc0[0]+self.time_offset-dt:.6f} ± {tc0[1]:.6f} (+{dt}) BJD_TDB\n"
            txt += f"Measured Tc: {tc-dt:.6f} ± {tc_sig:.6f} (+{dt}) BJD_TDB\n"
            txt += "NEBcheck stars NOT cleared: NEBs not checked"
        except Exception as e:
            print(e)
        return txt


@dataclass
class Star:
    name: str
    star_params: Dict[str, Tuple[float, float]] = None
    source: str = "tic"

    def __post_init__(self):
        if self.star_params is None:
            self.get_star_params()

        sources = set(
            [
                p.get("prov")
                for i, p in enumerate(self.data_json["stellar_parameters"])
            ]
        )
        errmsg = f"{self.source} must be in {sources}"
        assert self.source in sources, errmsg

    def get_tfop_data(self) -> None:
        base_url = "https://exofop.ipac.caltech.edu/tess"
        self.exofop_url = (
            f"{base_url}/target.php?id={self.name.replace(' ','')}&json"
        )
        response = urlopen(self.exofop_url)
        assert response.code == 200, "Failed to get data from ExoFOP-TESS"
        try:
            data_json = json.loads(response.read())
            return data_json
        except Exception:
            raise ValueError(f"No TIC data found for {self.name}")

    def get_star_params(self) -> None:
        if not hasattr(self, "data_json"):
            self.data_json = self.get_tfop_data()

        self.ra = float(self.data_json["coordinates"].get("ra"))
        self.dec = float(self.data_json["coordinates"].get("dec"))
        self.ticid = self.data_json["basic_info"].get("tic_id")
        self.toi_name = self.data_json["tois"][0].get("toi")
        self.notes = self.data_json["tois"][0].get("notes")
        if self.notes is not None:
            print(f"TFOP notes: {self.notes}")

        idx = 1
        for i, p in enumerate(self.data_json["stellar_parameters"]):
            if p.get("prov") == self.source:
                idx = i + 1
                break
        star_params = self.data_json["stellar_parameters"][idx]

        try:
            self.rstar = tuple(
                map(
                    float,
                    (
                        star_params.get("srad", np.nan),
                        star_params.get("srad_e", np.nan),
                    ),
                )
            )
            self.mstar = tuple(
                map(
                    float,
                    (
                        star_params.get("mass", np.nan),
                        star_params.get("mass_e", np.nan),
                    ),
                )
            )
            # stellar density in rho_sun
            self.rhostar = (
                self.mstar[0] / self.rstar[0] ** 3,
                np.sqrt(
                    (1 / self.rstar[0] ** 3) ** 2 * self.mstar[1] ** 2
                    + (3 * self.mstar[0] / self.rstar[0] ** 4) ** 2
                    * self.rstar[1] ** 2
                ),
            )
            print(f"Mstar=({self.mstar[0]:.2f},{self.mstar[1]:.2f}) Msun")
            print(f"Rstar=({self.rstar[0]:.2f},{self.rstar[1]:.2f}) Rsun")
            print(f"Rhostar=({self.rhostar[0]:.2f},{self.rhostar[1]:.2f}) rhosun")
            self.teff = tuple(
                map(
                    float,
                    (
                        star_params.get("teff", np.nan),
                        star_params.get("teff_e", 500),
                    ),
                )
            )
            self.logg = tuple(
                map(
                    float,
                    (
                        star_params.get("logg", np.nan),
                        star_params.get("logg_e", 0.1),
                    ),
                )
            )
            val = star_params.get("feh", 0)
            val = 0 if (val is None) or (val == "") else val
            val_err = star_params.get("feh_e", 0.1)
            val_err = 0.1 if (val is None) or (val_err == "") else val_err
            self.feh = tuple(map(float, (val, val_err)))
            print(f"teff=({self.teff[0]:.0f},{self.teff[1]:.0f}) K")
            print(f"logg=({self.logg[0]:.2f},{self.logg[1]:.2f}) cgs")
            print(f"feh=({self.feh[0]:.2f},{self.feh[1]:.2f}) dex")
        except Exception as e:
            print(e)
            raise ValueError(f"Check exofop: {self.exofop_url}")

    def get_gaia_sources(self, rad_arcsec=30) -> pd.DataFrame:
        target_coord = SkyCoord(ra=self.ra * u.deg, dec=self.dec * u.deg)
        if (not hasattr(self, "gaia_sources")) or (rad_arcsec > 30):
            msg = f'Querying Gaia sources {rad_arcsec}" around {self.name}: '
            msg += f"({self.ra:.4f}, {self.dec:.4f}) deg."
            print(msg)
            self.gaia_sources = Catalogs.query_region(
                target_coord,
                radius=rad_arcsec * u.arcsec,
                catalog="Gaia",
                version=3,
            ).to_pandas()
            self.gaia_sources["distance_arcsec"] = self.gaia_sources[
                "distance"
            ] * u.arcmin.to(u.arcsec)
        self.gaia_sources = self.gaia_sources[
            self.gaia_sources["distance_arcsec"] <= rad_arcsec
        ]
        #self.gaia_sources["distance_pix"] = self.gaia_sources["distance_arcsec"]/pixscales[self.inst]
        assert len(self.gaia_sources) > 1, "gaia_sources contains single entry"
        return self.gaia_sources

    def params_to_dict(self) -> dict:
        return {
            "rstar": self.rstar,
            "mstar": self.rstar,
            "rhostar": self.rhostar,
            "teff": self.teff,
            "logg": self.logg,
            "feh": self.feh,
        }


@dataclass
class Planet(Star):
    name: str
    alias: str = ".01"
    star_params: Dict[str, Tuple[float, float]]
    planet_params: Dict[str, Tuple[float, float]] = None
    source: str = "toi"

    def __post_init__(self):
        self.get_planet_params()

    def get_planet_params(self):
        if not hasattr(self, "data_json"):
            data_json = self.get_tfop_data()

        sources = set(
            [p.get("prov") for i, p in enumerate(data_json["planet_parameters"])]
        )
        errmsg = f"{self.source} must be in {sources}"
        assert self.source in sources, errmsg

        # try:
        #     idx = int(self.alias.replace('.', ''))
        # except:
        #     idx = 1
        idx = 1
        for i, p in enumerate(data_json["planet_parameters"]):
            if p.get("prov") == self.source:
                idx = i + 1
                break
        planet_params = data_json["planet_parameters"][idx]

        try:
            self.t0 = tuple(
                map(
                    float,
                    (
                        planet_params.get("epoch", np.nan),
                        planet_params.get("epoch_e", 0.1),
                    ),
                )
            )
            self.period = tuple(
                map(
                    float,
                    (
                        planet_params.get("per", np.nan),
                        planet_params.get("per_e", 0.1),
                    ),
                )
            )
            self.tdur = (
                np.array(
                    tuple(
                        map(
                            float,
                            (
                                planet_params.get("dur", 0),
                                planet_params.get("dur_e", 0),
                            ),
                        )
                    )
                )
                / 24
            )
            self.rprs = np.sqrt(
                np.array(
                    tuple(
                        map(
                            float,
                            (
                                planet_params.get("dep_p", 0),
                                planet_params.get("dep_p_e", 0),
                            ),
                        )
                    )
                )
                / 1e6
            )
            self.imp = tuple(
                map(
                    float,
                    (
                        (
                            0
                            if planet_params.get("imp", 0) == ""
                            else planet_params.get("imp", 0)
                        ),
                        (
                            0.1
                            if planet_params.get("imp_e", 0.1) == ""
                            else planet_params.get("imp_e", 0.1)
                        ),
                    ),
                )
            )
            print(f"t0={self.t0} BJD\nP={self.period} d\nRp/Rs={self.rprs}")
            rhostar = self.star_params["rhostar"]
            self.a_Rs = (
                (rhostar[0] / 0.01342 * self.period[0] ** 2) ** (1 / 3),
                1
                / 3
                * (1 / 0.01342 * self.period[0] ** 2) ** (1 / 3)
                * rhostar[0] ** (-2 / 3)
                * rhostar[1],
            )
        except Exception as e:
            print(e)
            raise ValueError(f"Check exofop: {self.exofop_url}")

    def params_to_dict(self) -> dict:
        return {
            "t0": self.t0,
            "period": self.period,
            "tdur": self.tdur,
            "imp": self.imp,
            "rprs": self.rprs,
            "a_Rs": self.a_Rs,
        }


def get_tfop_data(target_name: str) -> dict:
    base_url = "https://exofop.ipac.caltech.edu/tess"
    url = f"{base_url}/target.php?id={target_name.replace(' ','')}&json"
    response = urlopen(url)
    assert response.code == 200, "Failed to get data from ExoFOP-TESS"
    try:
        data_json = json.loads(response.read())
        return data_json
    except Exception:
        raise ValueError(f"No TIC data found for {target_name}")


def get_params_from_tfop(data_json, name="planet_parameters", idx=None) -> dict:
    params_dict = data_json.get(name)
    if idx is None:
        key = "pdate" if name == "planet_parameters" else "sdate"
        # get the latest parameter based on upload date
        dates = []
        for d in params_dict:
            t = d.get(key)
            dates.append(t)
        df = pd.DataFrame({"date": dates})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        idx = df["date"].idxmax()
    return params_dict[idx]


def get_tic_id(target_name: str) -> int:
    return int(get_tfop_data(target_name)["basic_info"]["tic_id"])


def get_toi_ephem(
    target_name: str, idx: int = 1, params=["epoch", "per", "dur"]
) -> list:
    print(f"Querying ephemeris for {target_name}:")
    r = get_tfop_data(target_name)
    planet_params = r["planet_parameters"][idx]
    vals = []
    for p in params:
        val = planet_params.get(p)
        val = float(val) if val else 0.1
        err = planet_params.get(p + "_e")
        err = float(err) if err else 0.1
        print(f"     {p}: {val}, {err}")
        vals.append((val, err))
    return vals


def tdur_from_per_imp_aRs_k(per, imp, a_Rs, k) -> float:
    # inc = np.arccos(imp / a_Rs)
    cosi = imp / a_Rs
    sini = np.sqrt(1.0 - cosi**2)
    return (
        per
        / np.pi
        * np.arcsin(1.0 / a_Rs * np.sqrt((1.0 + k) ** 2 - imp**2) / sini)
    )


def r1r2_to_imp_k(r1, r2, k_lo=0, k_up=1) -> tuple:
    """
    Efficient Joint Sampling of Impact Parameters and
    Transit Depths in Transiting Exoplanet Light Curves
    Espinosa+2018: RNAAS, 2 209
    https://iopscience.iop.org/article/10.3847/2515-5172/aaef38
    """
    A_r = (k_up-k_lo) / (2.0+k_lo+k_up)
    if r1 > A_r:
        imp = (1.0+k_lo) * (1.0+(r1-1.0)/(1.0-A_r))
        k = (1.0-r2)*k_lo + r2*k_up
    else:
        #FIXME: added mean if r2 is a vector so imp is not a vector
        imp = (1.0+k_lo) + np.sqrt(r1/A_r)*np.mean(r2) * (k_up-k_lo)
        k = k_up + (k_lo-k_up) * np.sqrt(r1/A_r)*(1.0-r2)
    assert isinstance(imp, float)
    return imp, k


def imp_k_to_r1r2(imp, k, k_lo=0.01, k_up=0.5) -> tuple:
    """
    Inverse function for r1r2_to_imp_k function.
    """
    raise NotImplemented
    
    A_r = (k_up-k_lo) / (2.0+k_lo+k_up)
    imp_threshold = (1.0 + k_lo)
    if imp > imp_threshold:
        r1 = (-A_r*i + A_r*k_lo + A_r + i)/(k_lo + 1.0)
        r2 = (k_lo - k)/(k_lo - k_up)
    else:
        r1 = k_lo + 1.0
        r2 = k_lo*np.sqrt(r1/A_r) - k_up*np.sqrt(r1/A_r) + k_up
    return r1, r2

def tdur_from_per_aRs_r1_r2(per, a_Rs, r1, r2) -> float:
    imp, k = r1r2_to_imp_k(r1, r2, k_lo=PRIOR_K_MIN, k_up=PRIOR_K_MAX)
    # inc = np.arccos(imp / a_Rs)
    cosi = imp / a_Rs
    sini = np.sqrt(1.0 - cosi**2)
    return (
        per
        / np.pi
        * np.arcsin(1.0 / a_Rs * np.sqrt((1.0 + k) ** 2 - imp**2) / sini)
    )


def binning_equal_interval(t, y, ye, binsize, t0) -> tuple:
    intt = np.floor((t - t0) / binsize)
    intt_unique = np.unique(intt)
    n_unique = len(intt_unique)
    tbin = np.zeros(n_unique)
    ybin = np.zeros(n_unique)
    yebin = np.zeros(n_unique)

    for i in range(n_unique):
        index = np.where(intt == intt_unique[i])
        tbin[i] = t0 + float(intt_unique[i]) * binsize + 0.5 * binsize
        w = 1 / ye[index] / ye[index]
        ybin[i] = np.sum(y[index] * w) / np.sum(w)
        yebin[i] = np.sqrt(1 / np.sum(w))

    return tbin, ybin, yebin


def get_percentile(array: list) -> float:
    med = np.median(array)
    low1 = np.percentile(array, 15.85)
    hig1 = np.percentile(array, 84.15)
    low2 = np.percentile(array, 2.275)
    hig2 = np.percentile(array, 97.725)
    low3 = np.percentile(array, 0.135)
    hig3 = np.percentile(array, 99.865)
    return med, low1, hig1, low2, hig2, low3, hig3


def plot_ql(
    df: pd.DataFrame,
    exptime: float = 10,
    title: str = None,
    mcolor: str = None,
    binsize: float = 600 / 86400,
    toffset: float = None,
    figsize: tuple = (8, 8),
    font_size: float = 16,
    title_height=1,
    zlims: tuple = (1, 2.5),
    debug: bool = False,
):
    fig, axs = plt.subplots(6, 1, figsize=figsize, tight_layout=True, sharex=True)
    axs = axs.flatten()
    if zlims:
        idx = (df["Airmass"] >= zlims[0]) & (df["Airmass"] <= zlims[1])
    else:
        idx = np.ones_like(df["Airmass"])

    if debug:
        print(sum(idx))

    if toffset is None:
        toffset = int(df["BJD_TDB"].min())
    t, f, e = (
        df.loc[idx, "BJD_TDB"] - toffset,
        df.loc[idx, "Flux"],
        df.loc[idx, "Err"],
    )

    binsize_mins = binsize * u.day.to(u.minute)
    t2, f2, e2 = binning_equal_interval(
        t.values, f.values, e.values, binsize, t0=toffset
    )

    ax = axs[0]
    p = "Flux"
    ax.plot(t, df.loc[idx, p], "k.", alpha=0.1)
    ax.errorbar(
        t2,
        f2,
        yerr=e2,
        fmt="o",
        markersize=5,
        color=mcolor,
        label=f"{binsize_mins}-min bin",
    )
    rms = np.std(np.diff(f2)) * 1e3
    text = f"rms={rms:.2f} ppt/{binsize_mins:.1f} min"

    ax.set_title(text)
    ax.set_ylabel("Normalized flux")

    p = "Airmass"
    ax = axs[1]
    ax.plot(t, df.loc[idx, p], "k-")
    ax.set_ylabel(p)

    p = "DX(pix)"
    ax = axs[2]
    ax.plot(t, df.loc[idx, p], "k-")
    ax.set_ylabel(p)

    p = "DY(pix)"
    ax = axs[3]
    ax.plot(t, df.loc[idx, p], "k-")
    ax.set_ylabel(p)

    p = "FWHM(pix)"
    ax = axs[4]
    ax.plot(t, df.loc[idx, p], "k-")
    ax.set_ylabel(p)

    p = "Peak(ADU)"
    ax = axs[5]
    ax.plot(t, df.loc[idx, p], "k-")
    ax.set_xlabel(f"BJD_TDB - {toffset}")
    ax.set_ylabel(p)

    if title:
        fig.suptitle(title, fontsize=font_size, y=title_height)
    return fig