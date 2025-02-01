import matplotlib ; matplotlib.use('Agg') #Set so that a windowless server does not crash # specifically for cluster usage
from matplotlib.colors import to_rgba as rgba
import lightkurve as lk
import numpy as np
import exoplanet as xo
import matplotlib.pyplot as plt
import pandas as pd
import pymc3 as pm
import pymc3_ext as pmx
import aesara_theano_fallback.tensor as tt
from celerite2.theano import terms, GaussianProcess
import starry
import pdb
import datetime
from astropy import constants as const
from astropy import units as u
import corner
import arviz as az
import pickle
import string
import glob
from functools import partial
from collections import OrderedDict
from scipy.stats import norm
from matplotlib.lines import Line2D

az.rcParams["plot.max_subplots"] = 100 # set to 100 to avoid error when generating trace plots
# plt.style.use('ggplot')
# matplotlib.use( 'tkagg' )

################### READING IN DATA #################### READING IN DATA #################### READING IN DATA ####################
################### READING IN DATA #################### READING IN DATA #################### READING IN DATA ####################
################### READING IN DATA #################### READING IN DATA #################### READING IN DATA ####################
tess_time_offset = 2457000
kepler_time_offset = 2454833
# Assuming that TESS data is in BTJD but rest of data is in BJD so thats why we are subtracting tess offset from data
### PHOTOMETRY DATA ###
### PHOTOMETRY DATA ###
### PHOTOMETRY DATA ###
# tess_data = pd.read_csv('TOI-5349_stitched_data.csv')
tess_data = pd.read_csv('TOI-5349_stitched_data_tess_sectors.csv')


tess_time = tess_data['time'].values

tess_flux = tess_data['flux'].values
tess_flux_error = tess_data['flux_err'].values
sector = tess_data['sector'].values

tess_exp = tess_data['exp'].values

mask2 = ((tess_time < 2600) & (tess_flux < 1.02)) | ((tess_time > 2600) & (tess_flux < 1.04))
# Reassigning tess data variables to only include flux values less than 1.05
#make second mask for left side data of 1.02
adj_flux = tess_flux[mask2]
adj_time_lc = tess_time[mask2]
adj_flux_error = tess_flux_error[mask2]
adj_sector = sector[mask2]
adj_tess_exp = tess_exp[mask2]


# RBO DATA #
rbo_data1 = pd.read_csv('TOI-5349_20230104_RBO_measurements.csv')
rbo_data2 = pd.read_csv('TOI-5349_20230114_RBO_measurements.csv')

time_rbo1 = rbo_data1['BJD_TDB'].values - tess_time_offset
rbo_flux1 = rbo_data1['rel_flux_T1'].values
rbo_flux_err1 = rbo_data1['rel_flux_err_T1'].values
rbo_exp = rbo_data1['EXPTIME'].values

time_rbo2 = rbo_data2['BJD_TDB'].values - tess_time_offset
rbo_flux2 = rbo_data2['rel_flux_T1'].values
rbo_flux_err2 = rbo_data2['rel_flux_err_T1'].values

norm_rbo_flux1 = rbo_flux1/np.median(rbo_flux1)
norm_rbo_flux_err1 = rbo_flux_err1/np.median(rbo_flux1)

norm_rbo_flux2 = rbo_flux2/np.median(rbo_flux2)
norm_rbo_flux_err2 = rbo_flux_err2/np.median(rbo_flux2)

# POMONA DATA #
pom_data1 = pd.read_csv('Processed_fdb_TESS_3_20250103_233518_00295.fits_measurements.xls',sep='\s+')
pom_data2 = pd.read_csv('Processed_fdb_TESS_5349_3_20250113_222523_009_affineremap_measurements.xls',sep='\s+')
pom_data3 = pd.read_csv('Processed_fdb_TESS_5349_3_20250123_210320_00360_affineremap_measurements.xls',sep='\s+')

time_pom1 = pom_data1['BJD_TDB'].values - tess_time_offset
pom_flux1 = pom_data1['rel_flux_T1'].values
pom_flux_err1 = pom_data1['rel_flux_err_T1'].values
pom_exp = pom_data1['EXPTIME'].values

time_pom2 = pom_data2['BJD_TDB'].values - tess_time_offset
pom_flux2 = pom_data2['rel_flux_T1'].values
pom_flux_err2 = pom_data2['rel_flux_err_T1'].values

time_pom3 = pom_data3['BJD_TDB'].values - tess_time_offset
pom_flux3 = pom_data3['rel_flux_T1_dfn'].values
pom_flux_err3 = pom_data3['rel_flux_err_T1_dfn'].values

norm_pom_flux1 = pom_flux1/np.median(pom_flux1)
norm_pom_flux_err1 = pom_flux_err1/np.median(pom_flux1)

norm_pom_flux2 = pom_flux2/np.median(pom_flux2)
norm_pom_flux_err2 = pom_flux_err2/np.median(pom_flux2)

norm_pom_flux3 = pom_flux3/np.median(pom_flux3)
norm_pom_flux_err3 = pom_flux_err3/np.median(pom_flux3)

# Creating individual dictionaries for each photometry dataset based on TESS sector/instrument

tess_sector_datasets = {}

for i in np.unique(adj_sector):
    sector_mask = adj_sector == i
    temp_datasets = OrderedDict(
    [
        ("TESS Sector {}".format(i), [adj_time_lc[sector_mask], adj_flux[sector_mask], adj_flux_error[sector_mask], adj_tess_exp[sector_mask]]),
    ]
    )
    tess_sector_datasets.update(temp_datasets)

rbo_dataset = OrderedDict(
    [

        ("RBO (04-Jan-2023)", [time_rbo1, norm_rbo_flux1, norm_rbo_flux_err1, rbo_exp]), 
        ("RBO (14-Jan-2023)", [time_rbo2, norm_rbo_flux2, norm_rbo_flux_err2, rbo_exp])

    ]
    )
pomona_dataset = OrderedDict(
    [

        ("Pomona (04-Jan-2025)", [time_pom1, norm_pom_flux1, norm_pom_flux_err1, pom_exp]), 
        ("Pomona (14-Jan-2025)", [time_pom2, norm_pom_flux2, norm_pom_flux_err2, pom_exp]),
        ("Pomona (24-Jan-2025)", [time_pom3, norm_pom_flux3, norm_pom_flux_err3, pom_exp])

    ]
    )


# Creating one dictionary containing all photometric datasets by merging dicts from above
all_datasets = {**tess_sector_datasets, **rbo_dataset, **pomona_dataset}

# Orbital Period
periods = [3.3176675]
# Orbital Period Error 
period_error = [0.1] #in days, must be in same unit as periods variable
# Time of mid-transit
t0s = [2459521.813826 - tess_time_offset]
# Transit Error
t0_error = [0.1]

def phaseup(t, t0, period):
    return (((t-t0) - period/2.) % period - period/2.)
phase = phaseup(tess_time, 2521.81748925, 3.31793068)

# plt.plot(phase, tess_flux, linestyle = 'none', marker = '.')
# plt.xlim(-0.1, 0.1)

### RADIAL VELOCITY DATA ###
### RADIAL VELOCITY DATA ###
### RADIAL VELOCITY DATA ###

# MAROON-X DATA
mx_red_data = pd.read_csv('TOI-5349_rv_bin_MAROONX_red.csv', header = 0,
                 names = ['time', 'radial_velocity', 'rv_error'])

mx_red_data['rv_instrument'] = 'maroon_x_red'
mx_blue_data = pd.read_csv('TOI-5349_rv_bin_MAROONX_blue.csv', header = 0,
                 names = ['time', 'radial_velocity', 'rv_error'])
mx_blue_data['rv_instrument'] = 'maroon_x_blue'

hpf_data = pd.read_csv('TOI-5349_rv_bin_HPF_013024.csv', header = 0,
                 names = ['time', 'radial_velocity', 'rv_error'])
hpf_data['rv_instrument'] = 'HPF'

mx_red_time = mx_red_data['time'].values #
mx_red_rv = mx_red_data['radial_velocity'].values
mx_red_rv_err = mx_red_data['rv_error'].values

mx_blue_time = (mx_blue_data['time']).values 
mx_blue_rv = mx_blue_data['radial_velocity'].values
mx_blue_rv_err = mx_blue_data['rv_error'].values

# Merging all the datasets together #
merged_data = pd.concat([mx_blue_data, mx_red_data, hpf_data], axis = 0).reset_index(drop=True)
time_rv = (merged_data['time'] - tess_time_offset).values 
rv = merged_data['radial_velocity'].values
rv_err = merged_data['rv_error'].values
rv_instrument = merged_data['rv_instrument'].values

# Computing a reference time that will be used to normalize the trends model #
time_ref = 0.5 * (time_rv.min() + time_rv.max())

# Making a fine grid that spans the observation window for plotting purposes #
t = np.linspace(time_rv.min() - 5, time_rv.max() + 5, 5000)

################## THE MODEL #################### THE MODEL #################### THE MODEL ################## THE MODEL ##################
################## THE MODEL #################### THE MODEL #################### THE MODEL ################## THE MODEL ##################
################## THE MODEL #################### THE MODEL #################### THE MODEL ################## THE MODEL ##################

# Orbital Period
periods = [3.3176675]
# Orbital Period Error 
period_error = [0.1] #in days, must be in same units as periods variable
# Transit Duration
t0s = [2459521.813826 - tess_time_offset]
# Transit Error
t0_error = [0.1]
#Number of Planets
nplanets = len(periods)
# Stellar Radius
R_star = [0.582, 0.014] #old values [0.578, 0.017] #in solar radii
# Stellar Mass
M_star = [0.608,0.022] #in solar mass #[0.610, 0.025] old values
Mjup2Mearth = (const.M_jup / const.M_earth).value
# Effective Temperature 
Teff = [3751, 59] # [3751, 88] #in Kelvin and uncertainty is +/- 88
Expected_msini = np.ones(nplanets) 
Rsun2Rearth = u.Rsun.to('Rearth')
RsunPerDay = ((const.R_sun/u.d).to(u.m/u.s)).value
Rsun2AU = u.Rsun.to('au')
Rsun2Rjup = u.Rsun.to('Rjup')
MJup2MSun = u.Mjup.to('Msun')
MSun2MJup = u.Msun.to('Mjup')
MSun2MEarth = u.Msun.to('Mearth')
# Decide if we are using dilution
tessdilution = True
# Decide if we are loading posterior pkl file
onlyplot = False

if not onlyplot:
    with pm.Model() as model:

        # Stellar Parameters
        BoundedNormal = pm.Bound(pm.Normal, lower=0, upper=1.5)
        m_star = BoundedNormal("m_star", mu=M_star[0], sd=M_star[1], shape = 1) # Stellar Mass
        r_star = BoundedNormal("r_star", mu=R_star[0], sd=R_star[1], shape = 1) # Stellar Radius
        teff = pm.Bound(pm.Normal, lower=2000, upper=7000)("teff", mu=Teff[0], sd=Teff[1], shape = 1) # Effective Temperature
        st_lum = pm.Deterministic("st_lum", (r_star**2) * ((teff/5777)**4)) # Stellar Luminosity
        # star_params = [mean, ustar] 
        
        # Planet Parameters (note: Deterministic means values that were derived from the model)
        ror = pm.Uniform("ror", lower=0.0, upper = 1.0, shape=nplanets) # Radius ratio
        r_pl = pm.Deterministic("r_pl", ror*r_star) #in physical units aka stellar radii #not a free parameter but you can calculate at every step
        pm.Deterministic("r_jup", r_pl*Rsun2Rjup)
        m_pl = pm.Uniform("m_pl", lower = 1e-3, upper = 1e3, testval=Expected_msini, shape=nplanets)
        # pm.Deterministic("m_jup", m_pl*MSun2MJup)
        pm.Deterministic("density_pl", m_pl*Mjup2Mearth/((r_pl*Rsun2Rearth)**3) * 5.514) # Convert from rho_earth to g/cm3
        pm.Deterministic("logg_pl", tt.log10( const.G.to('cm3 Mjup-1 s-2').value * m_pl / (r_pl * u.Rsun.to('cm') )**2 )  ) #Calculate surface gravity
        
        # Orbital Parameters
        period = pm.Normal("period", mu = np.array(periods), sigma = np.array(period_error), shape = nplanets)
        t0 = pm.Normal("t0", mu = t0s, sigma = np.array(t0_error), shape = nplanets)
        # For some reason the uniform + dilution gives odd results and vice versa
        if tessdilution:
            b = xo.distributions.ImpactParameter("b", ror=ror, shape = nplanets)
        else:
            b = pm.Uniform("b", lower = 0, upper = 1, shape=nplanets)        

        #Here are defining our eccentric model
        #Look at section 17 for more information on sampling eccentricity vs. omega on a UnitDisk (https://arxiv.org/pdf/1907.09480.pdf)
        ecs = pmx.UnitDisk("ecs", testval = np.array([[0.1, 0.1]] * nplanets).T, shape = (2, nplanets))
        ecc = pm.Deterministic("ecc", tt.sum(ecs ** 2, axis = 0))
        omega = pm.Deterministic("omega", tt.arctan2(ecs[1], ecs[0]))
        pm.Deterministic("omegadeg", omega * u.rad.to('deg'))
        
        # Set up the Orbit Model   
        orbit = xo.orbits.KeplerianOrbit(r_star = r_star, m_star = m_star, 
                                        period = period, t0 = t0, b = b, 
                                        ecc = ecc, omega = omega, m_planet = m_pl,
                                        m_planet_units = u.M_jup)

        # Stellar Density recovered from the transit
        pm.Deterministic("rho_star", orbit.rho_star)

        # Ratio of semi-major axis 
        aor = pm.Deterministic("aor", orbit.a / r_star)

        # Add semi-major axis in AU
        pm.Deterministic("semimajoraxis", orbit.a * Rsun2AU)

        # Add orbital inclination
        pm.Deterministic("inclination", orbit.incl * u.rad.to('deg'))

        # Add time of periastron
        tp = pm.Deterministic("t_p", orbit.t_periastron)

        # Add time of secondary eclipse
        phase1 = ( t0 - tp ) / period
        trueanom = 3.0*np.pi/2.0 - omega
        eccenanom = 2.0*tt.arctan(tt.sqrt((1.0-ecc)/(1.0+ecc))*tt.tan((trueanom)/2.0))
        Meananom = eccenanom - ecc*tt.sin(eccenanom)
        phase2 = tt.mod(Meananom/(2.0*np.pi),1)
        pm.Deterministic("t_s", t0 - period*(phase1-phase2))

        # Add transit duration, see Eq 14 & 16 here: https://ui.adsabs.harvard.edu/abs/2010exop.book...55W/abstract, Shubham wrote these for the below for his code
        sin_incl = tt.sqrt(1 - orbit.cos_incl**2)
        EccentricityMultiplicativeFactor = tt.sqrt(1-ecc**2)/(1+ecc*tt.sin(omega)) # Eq 16
        pm.Deterministic("transit_duration", (period/np.pi) * tt.arcsin( tt.sqrt( (1+ror)**2 - b**2 ) / (aor*sin_incl) ) * EccentricityMultiplicativeFactor *u.d.to('hr') )

        # Add impact parameter of secondary (Eq 8)
        pm.Deterministic('bsec', aor * orbit.cos_incl * (1-ecc**2) / (1 - ecc * tt.sin( omega )) )

        # Add equilibrium temperature
        pm.Deterministic("equilibrium_temp", teff * tt.sqrt(1/(2*aor)))

        # Add temporally the avergaed insolation flux in units of W/m2, see Eq 14 https://arxiv.org/pdf/1702.07314, 
        # luminsity is a black body or L = 4*pi*rstar^2*sigmab*teff^4
        pm.Deterministic("insolation_flux", 4 * np.pi * const.sigma_sb.to('W m-2 K-4').value * teff**4 / (aor**2 * tt.sqrt(1e0 - ecc**2)))

        #################### RV MODEL #################### RV MODEL #################### RV MODEL #####################
        #################### RV MODEL #################### RV MODEL #################### RV MODEL #####################
        #################### RV MODEL #################### RV MODEL #################### RV MODEL #####################

        # Prior for Semi-Amplitude
        K = pm.Deterministic("K", orbit.K0 * orbit.m_planet * RsunPerDay)
        # K = pm.Uniform(
        #     "K", lower = 1, upper = 1000, shape = nplanets)
        
        #Jitter & a baseline for now RV trend each instrument will have its own rv offset aka trend
        RVOffset = pm.Normal("RVOffset",
                            mu = np.array([0]*len(merged_data.rv_instrument.unique())), 
                            sigma = 20000, 
                            shape = len(merged_data.rv_instrument.unique()),) #offset relative to each instrument
        
        RVJitter = pm.Uniform("RVJitter", 1e-3, 1e3, shape = len(merged_data.rv_instrument.unique())) #adding additional noise
        
        RVMean = tt.zeros(merged_data.shape[0])
        RVError = tt.zeros(merged_data.shape[0])
        
        for i, inst in enumerate(merged_data.rv_instrument.unique()):
            RVMean += RVOffset[i] * (merged_data.rv_instrument == inst)
            RVError += tt.sqrt(rv_err**2 + RVJitter[i]**2) * (merged_data.rv_instrument == inst)
            
        pm.Deterministic("RVMean", RVMean)
        pm.Deterministic("RVError", RVError)

        # A function for computing the full RV model
        def get_rv_model(t, name = ""):
            
            # First the RVs induced by the planets
            vrad = orbit.get_radial_velocity(t, K = K) #the individual planet rv curves
            pm.Deterministic("vrad" + name, vrad)
            
            if nplanets > 1:
                # Sum over planets and add the background to get the full model
                return pm.Deterministic("rv_model" + name, tt.sum(vrad, axis = -1)) # depending on shape of v rad but should change....
            else:
                return pm.Deterministic("rv_model" + name, vrad)
        
        # Define the RVs at the observed times
        rv_model = get_rv_model(time_rv) # For all 3 instruments #rv model was the sum of multiple planets

        # Define the model on a fine grid as computed above (for plotting purposes)
        rv_model_pred = get_rv_model(t, name = "_pred")

        # Saving the time for RV model
        # pm.Deterministic("RVmodeltime", t)

        # Finally add in the observation model to the likelikehood. 
        # This next line adds a new contribution to the log probability of the PyMC3 model
        pm.Normal("rv_obs", mu = rv_model, sd = RVError, observed = rv - RVMean, shape = len(rv))
        
        #################### TRANSIT MODEL #################### TRANSIT MODEL #################### TRANSIT MODEL #####################
        #################### TRANSIT MODEL #################### TRANSIT MODEL #################### TRANSIT MODEL #####################
        #################### TRANSIT MODEL #################### TRANSIT MODEL #################### TRANSIT MODEL #####################

        # Loop over the instruments
        parameters = dict()
        # gp_preds = dict()
        # gp_preds_with_mean = dict()

        utess = xo.QuadLimbDark("utess")
        urbo = xo.QuadLimbDark("urbo")
        upomona = xo.QuadLimbDark("upomona")

        hi_cad_time = {}

        for n, (name, (time, flux, flux_error, texp)) in enumerate(all_datasets.items()):
            parameters[name] = []
            t_lc = np.linspace(time.min() - 5, time.max() + 5, 5000)
            hi_cad_time[name] = t_lc

            # We define the per-instrument parameters in a submodel so that we don't have to prefix the names manually
            # with pm.Model(name=name, model=model):
            # The flux zero point
            mean = pm.Normal(f"{name}_mean", mu = 1.0, sigma = 10.0, shape = 1)

            if bool(name.find('TESS')+1):
                ustar = utess
                # DILUTE = 1 - F2/(F1+F2)
                # The fraction of to basline flux that is due to your target.
                # Where F1 is the flux from the host star, and F2 is the flux from all other sources in the aperture.
                if tessdilution:
                    dilution = pm.Uniform(f"{name}_dilution", 0.0, 2.0, shape = 1)
                    parameters[name] += [dilution]
                else:
                    dilution = 1
            elif bool(name.find('RBO')+1):
                ustar = urbo
                dilution = 1
            else:
                ustar = upomona
                dilution = 1
            star = xo.LimbDarkLightCurve(ustar)

            # Calculates light curve for each planet at its time vector
            light_curves = star.get_light_curve(orbit = orbit, r = r_pl, t = time, texp = texp[0]*u.s.to('d')) #this will also change # Change texp to match your data set
            # Define GP Here

            # Saves the individual lightcurves 
            pm.Deterministic(f"{name}_light_curves", light_curves) 

            hi_cad_light_curves = star.get_light_curve(orbit = orbit, r = r_pl, t = t_lc, texp = texp[0]*u.s.to('d'))

            # Saves the individual lightcurves 
            pm.Deterministic(f"{name}_hi_cad_light_curves", hi_cad_light_curves)

            #Lightcurve Jitter
            Ln_Jitter = pm.Uniform(f"{name}_Ln_Jitter", np.log(1e-10), np.log(1e3), shape = 1)

            #Saving Log_Jitter as value in 
            LC_Jitter = pm.Deterministic(f"{name}_Jitter", tt.exp(Ln_Jitter))

            # LC_Jitter = pm.Deterministic(f"{name}_Jitter", tt.power(10, Log_Jitter))
            # LC_Jitter = pm.Uniform(f"{name}_Jitter", 0, 1e3)
            
            # Full photometric model, see Eq 5 https://ui.adsabs.harvard.edu/abs/2019MNRAS.490.2262E/abstract       
            lc_model = ( (1 + tt.sum(light_curves, axis=-1)) * dilution + (1 - dilution) ) * (1. / (1 + dilution * mean) )

            lc_error = tt.sqrt(LC_Jitter**2 + flux_error ** 2)

            # The likelihood function assuming known Gaussian uncertainty
            pm.Normal(f"{name}_transit_obs", mu = lc_model, sd = lc_error, observed = flux, shape = len(flux))

            parameters[name] += [mean, ustar]
            parameters[f"{name}_noise"] = [Ln_Jitter]

        ################## OPTIMIZING ################ OPTIMIZING ########################### OPTIMIZING ##################
        ################## OPTIMIZING ################ OPTIMIZING ########################### OPTIMIZING ##################
        ################## OPTIMIZING ################ OPTIMIZING ########################### OPTIMIZING ##################
        
        # In this portion, we will pptimize the Maximum a Posteriori AKA MAP Solution

        # Defining a random starting point based on our priors defined 
        map_soln = model.test_point

        # Optimizing system parameters for each lightcurve
        for name in all_datasets:
            map_soln = pmx.optimize(map_soln, parameters[name] + [ror, b, r_star])

        # Optimizing noise parameter for each instrument
        for name in all_datasets:
            map_soln = pmx.optimize(map_soln, parameters[f"{name}_noise"])

        # Optimizing Radial Velocity Model
        map_soln = pmx.optimize(map_soln, [RVJitter, RVOffset, K])
        map_soln = pmx.optimize(map_soln, vars = ecs)

        # # Optimizing Gaussian Process 
        # map_soln = pmx.optimize(map_soln, vars=[loggpamp, loggptimescale, loggpfactor, prot])
            
        # Refining all parameters simultaneously
        map_soln = pmx.optimize(map_soln)


        #################### GP MODEL #################### GP MODEL #################### GP MODEL #####################
        #################### GP MODEL #################### GP MODEL #################### GP MODEL #####################
        #################### GP MODEL #################### GP MODEL #################### GP MODEL #####################

        # #We are using a quasi-periodic kernel following Equation 56 from Foreman-Mackey et al. 2017 (https://arxiv.org/abs/1703.09710)
        
        # # A jitter term describing excess white noise which can be instrumental, astrophysical etc.
        # logjitter = pm.Uniform("logjitter",lower=-6,upper=0)
        # jitter = pm.Deterministic("jitter", tt.exp(logjitter))

        # # The Parameters of the RotationTerm Kernel (Eqn. 56)
        # loggpamp = pm.Uniform("loggpamp", lower = -20, upper = 0)
        # gpamp = pm.Deterministic("gpamp", tt.exp(loggpamp))
        # loggptimescale = pm.Uniform("loggptimescale", lower = 1.5, upper = 5)
        # gptimescale = pm.Deterministic("gptimescale", tt.exp(loggptimescale))
        # loggpfactor = pm.Uniform("loggpfactor", lower = -5, upper = 5)
        # gpfactor = pm.Deterministic("gpfactor", tt.exp(loggpfactor))
        # logprot = pm.Uniform("logprot", lower = -3, upper = 5)
        # prot = pm.Deterministic("prot", tt.exp(logprot))
        
        # # Finally defining the Kernel AKA Eqn. 56
        # kernel = terms.RealTerm(a = gpamp * (1+gpfactor)/(2+gpfactor), c = 1/gptimescale) + terms.ComplexTerm(a=gpamp / (2.0 + gpfactor), b = 0, c=1/gptimescale,d =2*np.pi/prot) 

        # # Defining our Gaussian Process Object
        # gp = GaussianProcess(
        #     kernel,
        #     t = time_lc,
        #     diag = flux_error**2 + jitter**2, 
        #     mean = 0,
        #     quiet = True,
        # )
        
        # # Compute the Gaussian Process likelihood and add it into the the PyMC3 model as a "potential"
        # gp.marginal("no_transit_lc", observed = flux - lc_model)

        # # Compute the GP model prediction for plotting purposes
        # pm.Deterministic("gp_pred", gp.predict(flux - lc_model))
        

    ################ THE MAP FIT PARAMETERS ################## THE MAP FIT PARAMETERS ################## THE MAP FIT PARAMETERS ######
    ################ THE MAP FIT PARAMETERS ################## THE MAP FIT PARAMETERS ################## THE MAP FIT PARAMETERS ######
    ################ THE MAP FIT PARAMETERS ################## THE MAP FIT PARAMETERS ################## THE MAP FIT PARAMETERS ######

    print('*** MAP Fit Parameters ***')
    for thiskey in list(map_soln.keys())[:-1]:
    # for thiskey in ['period', 't0', 'u', 'ror']:
        if bool(thiskey.find('__')+1):
            continue
        print('{}: {}'.format(thiskey, map_soln[thiskey]))

    ################ TRANSIT AND RV INITIAL BEST FIT PLOTS ################## TRANSIT AND RV INITIAL BEST FIT PLOTS ##################
    ################ TRANSIT AND RV INITIAL BEST FIT PLOTS ################## TRANSIT AND RV INITIAL BEST FIT PLOTS ##################
    ################ TRANSIT AND RV INITIAL BEST FIT PLOTS ################## TRANSIT AND RV INITIAL BEST FIT PLOTS ################## 

    t0 = map_soln["t0"]
    period = map_soln["period"] 
    rv_model = map_soln["rv_model"]
    rv_model_pred = map_soln["rv_model_pred"]
    rv_mean = map_soln['RVMean']

    vrad = map_soln["vrad"]
    vrad_pred = map_soln["vrad_pred"]

    ################ RV RESIDUALS PLOT ######### RV RESIDUALS PLOT ######### RV RESIDUALS PLOT ######### RV RESIDUALS PLOT #############
    ################ RV RESIDUALS PLOT ######### RV RESIDUALS PLOT ######### RV RESIDUALS PLOT ######### RV RESIDUALS PLOT #############
    ################ RV RESIDUALS PLOT ######### RV RESIDUALS PLOT ######### RV RESIDUALS PLOT ######### RV RESIDUALS PLOT #############

    datelabel = "{:%m-%d-%Y}".format(datetime.datetime.now())
    with model:
        fig, axes = plt.subplots(2, 1, figsize = (10, 5), sharex = True)
        ax = axes[0]
        ax.errorbar(time_rv, rv - rv_mean, yerr = rv_err, fmt = ".k")
        ax.plot(t, vrad_pred, "--k", alpha = 0.5) 
        # axes.plot(t, pmx.eval_in_model(model.vrad_pred), "--k", alpha= 0.5) 
        # axes.plot(t, pmx.eval_in_model(model.vrad), ":k", alpha= 0.5) 

        ax.set_title("initial model")
        ax.set_ylabel("radial velocity [m/s]")
        #plt.legend()
        # # fig.savefig('rv-best-initial-best-fit')
        # # plt.close(fig)
        # plt.plot(t, pmx.eval_in_model(model.bkg_pred), ":k", alpha=0.5, zorder = 100)


        ax = axes[1]
        ax.errorbar(time_rv, rv - rv_mean - rv_model, yerr = rv_err, fmt = ".k")
        ax.axhline(0, color = "k", lw = 1)
        ax.set_title("residuals")
        ax.set_ylabel("residuals [m/s]")
        # ax.set_xlim(3230, 3250)
        ax.set_xlabel("time [days]")
        ax.figure.savefig(f'TOI-5349-b_residuals_plot_{datelabel}.pdf', bbox_inches = 'tight', pad_inches = 0.0)


    ################ TRANSIT RESIDUALS PLOT ######### TRANSIT RESIDUALS PLOT ######### TRANSIT RESIDUALS PLOT ######### TRANSIT RESIDUALS PLOT #############
    ################ TRANSIT RESIDUALS PLOT ######### TRANSIT RESIDUALS PLOT ######### TRANSIT RESIDUALS PLOT ######### TRANSIT RESIDUALS PLOT #############
    ################ TRANSIT RESIDUALS PLOT ######### TRANSIT RESIDUALS PLOT ######### TRANSIT RESIDUALS PLOT ######### TRANSIT RESIDUALS PLOT #############

    ###### PRELIM TRANSIT PHASE PLOT ###### PRELIM TRANSIT PHASE PLOT ######
    ###### PRELIM TRANSIT PHASE PLOT ###### PRELIM TRANSIT PHASE PLOT ######
    ###### PRELIM TRANSIT PHASE PLOT ###### PRELIM TRANSIT PHASE PLOT ######
    for n, (name, (time, flux, flux_error, texp)) in enumerate(all_datasets.items()):

        fig, ax = plt.subplots(figsize = (10, 5))

        x_fold = (time - t0 + 0.5 * period) % period - 0.5 * period #in days
        m = np.abs(x_fold) < 0.5 # plot will only show phases between -0.5 to 0.5

        ax.scatter(x_fold[m], 1e3 * (flux[m]), #*******
            c = "k",
            marker = ".",
            alpha = 0.2,
            linewidths = 0,
        )

        lc_mod = map_soln[f'{name}_light_curves']
        lc_modx = np.sort(x_fold)

        if f'{name}_dilution' in map_soln:
            dilution = map_soln[f'{name}_dilution']
        else:
            dilution = 1
        lc_mody = ( (1 + lc_mod[np.argsort(x_fold)]) * dilution + (1 - dilution) ) * (1. / (1 + dilution * map_soln[f'{name}_mean']) )

        ax.plot(lc_modx, 1e3 * lc_mody, c = "blue", zorder = 1) #*******

        # Overplot the phase binned light curve
        lkobj = lk.LightCurve(time = x_fold,
                                    flux = flux * 1e3,
                                    flux_err = flux_error * 1e3)


        binned = lkobj.bin(time_bin_size = 8*u.min)
        # bins = np.linspace(-0.51, 0.51, 100)
        # denom, _ = np.histogram(x_fold, bins)
        # num, _ = np.histogram(x_fold, bins, weights = flux)
        # denom[num == 0] = 1.0

        ax.errorbar(binned.time.value,binned.flux.value,
            yerr=binned.flux_err.value, 
            color = "C1", 
            linestyle = 'none')

        # ax.scatter(0.5 * (bins[1:] + bins[:-1]), 1e3 * num / denom, #*******
        #     color = "C1",
        #     zorder = 2,
        #     linewidths = 0,
        # )
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylabel("Normalized Flux (ppt)")
        _ = ax.set_xlabel("Days from Mid-Transit")
        # plt.show()
        ax.figure.savefig(f'{name}_TOI-5349-b_LC_phase_plot_{datelabel}.pdf', bbox_inches = 'tight', pad_inches = 0.0)


    ###### PRELIM RV PHASE PLOT ###### PRELIM RV PHASE PLOT ######
    ###### PRELIM RV PHASE PLOT ###### PRELIM RV PHASE PLOT ######
    ###### PRELIM RV PHASE PLOT ###### PRELIM RV PHASE PLOT ######

    fig, ax = plt.subplots(figsize = (10, 5))

    rv_xfold = (time_rv - t0 + 0.5 * period) % period - 0.5 * period # stacking everything by 1 period and then shifting reference point to 0
    plt.errorbar(rv_xfold, rv - map_soln["RVMean"], yerr = rv_err, fmt = ".k", label = "data")

    ax.scatter(rv_xfold, rv - map_soln["RVMean"],
        c = "k",
        marker = ".",
        alpha = 0.2,
        linewidths = 0,
    )

    t_fold = (t - t0 + 0.5 * period) % period - 0.5 * period
    rv_modx = np.sort(t_fold)
    rv_mody = vrad_pred[np.argsort(t_fold)]

    plt.plot(rv_modx, rv_mody, c = "blue", zorder = 1, label = 'model') 
    plt.xlim(-0.5 * period, 0.5 * period)
    plt.title("TOI-5349b")
    plt.ylabel("RV - Systematic Velocity (ms/s)")
    plt.xlabel("Phase (days)")
    plt.legend()
    plt.savefig('TOI-5349-b_RV_phase_plot_{}.pdf'.format(datelabel), bbox_inches = 'tight', pad_inches = 0.0)

    ############ SAMPLING THE DATA ############ SAMPLING THE DATA ############ SAMPLING THE DATA ############ SAMPLING THE DATA ################### 
    ############ SAMPLING THE DATA ############ SAMPLING THE DATA ############ SAMPLING THE DATA ############ SAMPLING THE DATA ################### 
    ############ SAMPLING THE DATA ############ SAMPLING THE DATA ############ SAMPLING THE DATA ############ SAMPLING THE DATA ################### 

    NSteps = 10000
    Nchains = 24
    Ncores = 24
    with model:

        trace = pmx.sample(
            tune = NSteps,
            draws = int(NSteps/2),
            start = map_soln,
            cores = Ncores,
            chains = Nchains,
            target_accept = 0.95,
            return_inferencedata = True,
        )


    ################ SAVING THE MCMC OUTPUT ################## SAVING THE MCMC OUTPUT ################## SAVING THE MCMC OUTPUT #######################
    ################ SAVING THE MCMC OUTPUT ################## SAVING THE MCMC OUTPUT ################## SAVING THE MCMC OUTPUT #######################
    ################ SAVING THE MCMC OUTPUT ################## SAVING THE MCMC OUTPUT ################## SAVING THE MCMC OUTPUT #######################

    flat_samps = trace.posterior.stack(sample = ("chain", "draw"))

    var_names = ["period", "t0", 'ecc', 'omegadeg', 'K', 'RVOffset', #  Traditional RV Paramters (period, tc, eccentricity, omega, semi-major amp, gamma)
                'utess', 'urbo', 'upomona', "ror", 'aor', 'b', 'inclination', 'transit_duration', # The transit parameters (limb dark, rp/r*, a/r*, impact param, inc)
                'semimajoraxis', 'bsec', 't_s','t_p', 'ecs', #Derived parameters (a in au, impact param of sec, time of sec, time of peri)
                'teff', 'r_star', 'm_star', 'st_lum', 'rho_star', # The Physical Stellar Parameters
                'm_pl', 'r_jup', 'logg_pl', 'density_pl',  # The Planetary Parameters (mass, density, log g_p, rho_p)
                'RVJitter', # RV systematics
                ]
    #Transit systematics
    var_names += (np.char.array(list(all_datasets.keys()))+'_Jitter').tolist()
    var_names += (np.char.array(list(all_datasets.keys()))+'_mean').tolist()
    for thiskey in (np.char.array(list(all_datasets.keys()))+'_dilution').tolist():
        if thiskey in map_soln:
            var_names += [thiskey]
            
    output_dict = {'all_datasets' : all_datasets,
                'hi_cad_time' : hi_cad_time,
                'time_rv' : time_rv,
                'rv' : rv,      #Save RV
                'rv_err' : rv_err,
                'rv_instrument' : rv_instrument,
                "map_solution" : map_soln, #Save solution
                "trace" : trace,
                "var_names" : var_names,
                "flat_samples" : flat_samps}

    with open('TOI-5349_{}.pkl'.format(datelabel), 'wb') as f:
        pickle.dump(output_dict, f)
else:
    with open(np.sort(glob.glob('*pkl'))[-1], 'rb') as f:
        all_datasets, hi_cad_time, time_rv, rv, rv_err, rv_instrument, map_soln, trace, var_names, flat_samps = pickle.load(f).values()
    datelabel = np.sort(glob.glob('*pkl'))[-1].split('_')[-1].replace('.pkl','')
    model = pm.Model()

############# CHECKING STATUS OF CONVERGENCE ############# CHECKING STATUS OF CONVERGENCE ############# CHECKING STATUS OF CONVERGENCE #############
############# CHECKING STATUS OF CONVERGENCE ############# CHECKING STATUS OF CONVERGENCE ############# CHECKING STATUS OF CONVERGENCE #############
############# CHECKING STATUS OF CONVERGENCE ############# CHECKING STATUS OF CONVERGENCE ############# CHECKING STATUS OF CONVERGENCE #############
from functools import partial

posteriors  = az.summary(
    trace, 
    var_names = var_names, 
    round_to = "None", 
    stat_funcs = {'median':np.nanmedian, 'lowerpercentile':partial(np.nanpercentile,q=16), 'upperpercentile':partial(np.nanpercentile,q=84)}
    )
posteriors_df = pd.DataFrame(posteriors)
posteriors_df['neg1sigma'] = posteriors_df['median'] - posteriors_df.lowerpercentile
posteriors_df['pos1sigma'] = posteriors_df.upperpercentile - posteriors_df['median']
print('***Saving the posteriors***')
posteriors_df.to_csv('TOI-5349_posteriors_{}.csv'.format(datelabel))

################ GENERATING CORNER + TRACE PLOTS ################# GENERATING CORNER + TRACE PLOTS ################## GENERATING CORNER + TRACE PLOTS #######
################ GENERATING CORNER + TRACE PLOTS ################# GENERATING CORNER + TRACE PLOTS ################## GENERATING CORNER + TRACE PLOTS #######
################ GENERATING CORNER + TRACE PLOTS ################# GENERATING CORNER + TRACE PLOTS ################## GENERATING CORNER + TRACE PLOTS #######

### TRACE PLOT ###
### TRACE PLOT ###
### TRACE PLOT ###
print('***Saving the trace plot***')
_ = az.plot_trace(trace, var_names = var_names) 

plt.savefig('TOI-5349_trace_plot_{}.pdf'.format(datelabel),bbox_inches = 'tight', pad_inches = 0.0) 

########### GENERATING FINAL PHASE FOLDED PLOTS ########### GENERATING FINAL PHASE FOLDED PLOTS ########### GENERATING FINAL PHASE FOLDED PLOTS ###########
########### GENERATING FINAL PHASE FOLDED PLOTS ########### GENERATING FINAL PHASE FOLDED PLOTS ########### GENERATING FINAL PHASE FOLDED PLOTS ###########
########### GENERATING FINAL PHASE FOLDED PLOTS ########### GENERATING FINAL PHASE FOLDED PLOTS ########### GENERATING FINAL PHASE FOLDED PLOTS ###########

######## TRANSIT FOLDED PHASE PLOTS ####### TRANSIT FOLDED PHASE PLOTS #######
######## TRANSIT FOLDED PHASE PLOTS ####### TRANSIT FOLDED PHASE PLOTS #######
######## TRANSIT FOLDED PHASE PLOTS ####### TRANSIT FOLDED PHASE PLOTS #######

#Quantiles, assuming Gaussian distribution
quantiles = [[1-norm.sf(-1),1-norm.sf(1)], #1sigma
             [1-norm.sf(-2),1-norm.sf(2)], #2sigma
             [1-norm.sf(-3),1-norm.sf(3)], #3sigma
             ][::-1]
print('***Saving the photometry plots***')
for n, (name, (time, flux, flux_error, texp)) in enumerate(all_datasets.items()):

    for k in range(nplanets):
        # ------------------------------- Plot the data ------------------------------ #
        fig = plt.figure()
        ax = plt.subplot2grid((3, 1), (0, 0),rowspan=2)
        ax.tick_params(direction = "in", which = 'both',bottom = True, top = True, left = True, right = True)

        # Get the posterior median orbital parameters
        p = np.median(flat_samps["period"][k])
        t0 = np.median(flat_samps["t0"][k])
        
        # Plot the folded data
        x_fold = (time - t0 + 0.5 * p) % p - 0.5 * p
        gp_mod=np.zeros(len(time)) #To be modified when using a GP
        m = (np.abs(x_fold) < 0.3) &(flux < 1.05)

        # --------------------------- Plot the folded model -------------------------- #
        for j, thisquantile in enumerate(quantiles):
            pred = np.percentile(flat_samps[f"{name}_light_curves"][:, k, :], [thisquantile[0]*100, 50, thisquantile[-1]*100], axis = -1) # finding the scatter between the 16th through 84th percentile (its the +/- 1 sigma of a gaussian distribution)
            if f'{name}_dilution' in map_soln:
                dilution = np.median(flat_samps[f"{name}_dilution"])
            else:
                dilution = 1
            pred = ( (1 + pred) * dilution + (1 - dilution) ) * (1. / (1 + dilution * np.median(flat_samps[f"{name}_mean"])) )
            sort = np.argsort(x_fold)
            baseline = np.max(pred[1])
            if j == 0:
                ax.errorbar(x_fold[m], (flux[m] - gp_mod[m])/baseline, yerr = flux_error[m]/baseline, linestyle='none',
                            color="k", marker='o', label = "data", zorder = -1000, alpha = 0.5, rasterized = True)                
                ax.plot(x_fold[sort], pred[1][sort]/baseline, color = "C1", zorder=10)
            art = ax.fill_between(x_fold[sort], 
                                  pred[0][sort]/baseline, 
                                  pred[2][sort]/baseline, 
                                  color = "C1", 
                                  alpha = 0.7/len(quantiles), 
                                  zorder = 1000
                                  )
            art.set_edgecolor("none")
        # ax.legend(fontsize = 10, loc = 4)
        ax.set_ylabel("Normalized Flux")
        plt.setp(ax.get_xticklabels(), visible=False)
        letter = string.ascii_lowercase[n]
        ax.set_title(f"({letter}) {name}",fontdict={'weight': 'heavy'})
        ax.set_xlim(-0.3, 0.3)

        # ---------------------------- Plot the residuals ---------------------------- #
        bx = plt.subplot2grid((3, 1), (2, 0),rowspan=1)
        bx.tick_params(direction = "in", which = 'both',bottom = True, top = True, left = True, right = True)
        bx.errorbar(x_fold[m], (flux[m] - gp_mod[m] - pred[1][m])/baseline, yerr = flux_error[m]/baseline, linestyle='none',
                    color="k", marker='o', label = "data", alpha = 0.5, rasterized = True)
        bx.set_xlabel("Days from mid-transit")
        bx.set_ylabel("Residuals")
        rmse = np.sqrt(np.nanmean( ((flux - gp_mod - pred[1])/baseline) ** 2 )) * 1e6
        leg = bx.legend(handles=[Line2D([], [], color='C0', label='RMSE$='+'{:.0f}$ ppm'.format(rmse),linestyle='-')],
                        loc='best',frameon=True,handlelength=0, handletextpad=0)
        for item in leg.legend_handles:
            item.set_visible(False)
        bx.set_xlim(ax.get_xlim())
        bx.axhline(0,linestyle='--',zorder=-1, color='k')

        # --------------------------------- Save plot -------------------------------- #
        fig.tight_layout()
        plt.subplots_adjust(hspace=0)
        plt.savefig(f'TOI-5349_transit_folded_phase_plot_{datelabel}_{name}.pdf', bbox_inches = 'tight', pad_inches = 0.0)
        plt.close()


######## RV FOLDED PHASE PLOTS ####### RV FOLDED PHASE PLOTS #######
######## RV FOLDED PHASE PLOTS ####### RV FOLDED PHASE PLOTS #######
######## RV FOLDED PHASE PLOTS ####### RV FOLDED PHASE PLOTS #######
print('***Saving the RV plot***')
for n, letter in enumerate("b"):
    fig = plt.figure()

    # --------------------------- Plot phase-folded RVs -------------------------- #
    ax = plt.subplot2grid((3, 1), (0, 0),rowspan=2)
    ax.tick_params(direction = "in", which='both',bottom = True, top = True, left = True, right = True)

    # Get the posterior median orbital parameters
    p = np.median(flat_samps["period"][n])
    t0 = np.median(flat_samps["t0"][n])

    # Compute the median of posterior estimate of the background RV
    # and the contribution from the other planet. Then we can remove
    # this from the data to plot just the planet we care about.
    bkg = np.median(flat_samps["RVMean"],axis=1)
    
    # Plot the folded data
    x_fold = (time_rv - t0 + 0.5 * p) % p - 0.5 * p
    for thisinstrument in pd.Series(rv_instrument).unique():
        mask = rv_instrument == thisinstrument
        ax.errorbar(x_fold[mask], (rv - bkg)[mask], 
                    yerr = rv_err[mask], 
                    marker = 'o', 
                    linestyle = 'none', 
                    ecolor=rgba('black',0.2),
                    markeredgecolor=rgba('black',0.2),
                    label="{}".format(thisinstrument.replace('maroon_x_blue','MAROON-X (Blue)').replace('maroon_x_red','MAROON-X (Red)'))
                    )
    ax.tick_params(axis = 'both',which = 'major', width = 1.00, length = 5)
    ax.tick_params(axis = 'both', which ='minor', direction ='in', length = 4, width = 1)
    
    # ------------------------- Plot the model posteriors ------------------------ #
    # Compute the posterior prediction for the folded RV model for this planet
    t_rv = np.linspace(time_rv.min() - 5, time_rv.max() + 5, 5000)
    t_fold = (t_rv - t0 + 0.5 * p) % p - 0.5 * p
    inds = np.argsort(t_fold)
    for j, thisquantile in enumerate(quantiles):
        pred = np.percentile(flat_samps["rv_model_pred"][inds,:], [thisquantile[0]*100, 50, thisquantile[-1]*100], axis=-1)
        if j == 0:
            ax.plot(t_fold[inds], pred[1], color="C1", label="model", zorder = 10)
        art = ax.fill_between( t_fold[inds], pred[0], pred[2], color="C1", alpha= 0.7/len(quantiles) )
        art.set_edgecolor("none")

    plt.legend(fontsize=10)
    plt.xlim(-0.5 * p, 0.5 * p)
    plt.xlabel("Phase")
    plt.ylabel("RV - Systematic Velocity")
    
    # --------------------- Create a secondary axis in phase --------------------- #
    def phasetodays(x):
        return x*p
    def daystophase(x):
        return x / p
    secax = plt.gca().secondary_xaxis('top', functions=(daystophase, phasetodays))
    secax.set_xlabel('Phase')

    ax.legend(fontsize=10)
    ax.set_ylabel(r"RV - Systematic Velocity ($\mathrm{m~s^{-1}}$)")
    plt.setp(ax.get_xticklabels(), visible=False)

    # ------------------------------ Plot residuals ------------------------------ #
    bx = plt.subplot2grid((3, 1), (2, 0),rowspan=1)
    bkg = np.median(flat_samps['RVMean'],axis=1)
    pred = np.median(flat_samps["rv_model"],axis=1)
    for thisinstrument in pd.Series(rv_instrument).unique():
        mask = rv_instrument == thisinstrument
        bx.errorbar(x_fold[mask], (rv - bkg - pred)[mask], 
                    yerr = rv_err[mask], 
                    marker = 'o', 
                    linestyle = 'none', 
                    ecolor=rgba('black',0.2),
                    markeredgecolor=rgba('black',0.2),
                    label="{}".format(thisinstrument.replace('maroon_x_blue','MAROON-X (Blue)').replace('maroon_x_red','MAROON-X (Red)'))
                    )
    bx.set_xlabel("Days from mid-transit")
    bx.set_ylabel("Residuals")
    bx.set_xlim(ax.get_xlim())
    bx.axhline(0,linestyle='--',zorder=-1,color='k')
    bx.tick_params(direction = "in", which='both',bottom = True, top = True, left = True, right = True)

    # --------------------------------- Save plot -------------------------------- #
    fig.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.savefig('TOI-5349_RV_folded_phase_plot_{}.pdf'.format(datelabel), bbox_inches = 'tight', pad_inches = 0.0)
    plt.close()

### CORNER PLOT ###
### CORNER PLOT ###
### CORNER PLOT ###
print('***Saving the corner plot***')
with model:
    # _ = corner.corner(trace, var_names = var_names)
    _ = corner.corner(
        trace, 
        var_names = var_names, 
        quantiles=[0.16, 0.5, 0.84], 
        show_titles=True, 
        title_kwargs={"fontsize": 12}, 
        use_math_text=True
        )

plt.savefig('TOI-5349_corner_plot_{}.pdf'.format(datelabel),bbox_inches = 'tight', pad_inches = 0.0)
