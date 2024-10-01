# import matplotlib ; matplotlib.use('Agg') #Set so that a windowless server does not crash # specifically for cluster usage
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
from functools import partial
from collections import OrderedDict

az.rcParams["plot.max_subplots"] = 100 # set to 100 to avoid error when generating trace plots
# plt.style.use('ggplot')
# matplotlib.use( 'tkagg' )

################### READING IN DATA #################### READING IN DATA #################### READING IN DATA ####################
################### READING IN DATA #################### READING IN DATA #################### READING IN DATA ####################
################### READING IN DATA #################### READING IN DATA #################### READING IN DATA ####################

### PHOTOMETRY DATA ###
### PHOTOMETRY DATA ###
### PHOTOMETRY DATA ###
# tess_data = pd.read_csv('TOI-5349_stitched_data.csv')
tess_data = pd.read_csv('TOI-5349_stitched_data_tess_sectors.csv')


tess_time = tess_data['time'].values
t_lc = np.linspace(tess_time.min() - 5, tess_time.max() + 5, 5000)

tess_flux = tess_data['flux'].values
tess_flux_error = tess_data['flux_err'].values
sector = tess_data['sector'].values

tess_exp = tess_data['exp'].values

mask2 = ((tess_time < 2600) & (tess_flux < 1.02)) | ((tess_time > 2600) & (tess_flux < 1.04))
# Reassigning tess data variables to only include flux values less than 1.05
# mask2 = flux < 1.04
#make second mask for left side data of 1.02
adj_flux = tess_flux[mask2]
adj_time_lc = tess_time[mask2]
adj_flux_error = tess_flux_error[mask2]
adj_sector = sector[mask2]

# plt.figure()
# plt.plot(time_lc[mask], flux[mask], linestyle = 'none', color = 'k', marker = '.', ms = 1)
# plt.ylabel("relative flux")
# plt.xlabel("time [days]")
# plt.title('TESS Photometry of TOI-5349')
# plt.show()
# plt.savefig('tess_lc1.png',bbox_inches='tight', pad_inches=0.0)
# plt.close()

# plt.figure()
# plt.plot(time_lc[mask2], flux[mask2], color = 'k', marker = ".", ms = 1, linestyle = 'none')
# # plt.plot(time_lc, flux, color = 'k', marker = ".", ms = 1, linestyle = 'none')
# # plt.xlim(3205, 3235)
# # plt.ylim(0.6, 1.5)
# plt.ylabel("relative flux")
# plt.xlabel("time [days]")
# plt.title('TESS Photometry of TOI-5349')
# plt.show()
# # plt.savefig('tess_lc2.png',bbox_inches='tight', pad_inches=0.0)
# # plt.close()


# plt.figure()
# # plt.plot(time_lc[mask2], flux[mask2], linestyle = 'none',marker = 'o')
# for n_sector in np.unique(adj_sector):
#     cull = adj_sector == n_sector
#     plt.plot(adj_time_lc[cull], adj_flux[cull], linestyle = 'none', marker = 'o', label = n_sector)
# plt.ylabel("relative flux")
# plt.xlabel("time [days]")
# plt.title('TESS Photometry of TOI-5349')
# plt.legend()
# plt.show()


# RBO DATA #
# RBO DATA #
# RBO DATA #

rbo_data1 = pd.read_csv('TOI-5349_20230104_RBO_measurements.csv')
rbo_data2 = pd.read_csv('TOI-5349_20230114_RBO_measurements.csv')

time_rbo1 = rbo_data1['BJD_TDB'].values
rbo_flux1 = rbo_data1['rel_flux_T1'].values
rbo_flux_err1 = rbo_data1['rel_flux_err_T1'].values
rbo_exp = rbo_data1['EXPTIME'].values

time_rbo2 = rbo_data2['BJD_TDB'].values
rbo_flux2 = rbo_data2['rel_flux_T1'].values
rbo_flux_err2 = rbo_data2['rel_flux_err_T1'].values

norm_rbo_flux1 = rbo_flux1/np.median(rbo_flux1)
norm_rbo_flux_err1 = rbo_flux_err1/np.median(rbo_flux1)

norm_rbo_flux2 = rbo_flux2/np.median(rbo_flux2)
norm_rbo_flux_err2 = rbo_flux_err2/np.median(rbo_flux2)


#need to add exposure time


# Creating individual dictionaries for each photometry dataset based on TESS sector/instrument

tess_sector_datasets = {}

for i in np.unique(adj_sector):
    temp_datasets = OrderedDict(
    [
        ("TESS Sector {}".format(i), [adj_time_lc, adj_flux, adj_flux_error, tess_exp]),
    ]
    )
    tess_sector_datasets.update(temp_datasets)
# print(tess_sector_datasets['TESS Sector 42'])

rbo_dataset = OrderedDict(
    [

        ("RBO Data 1", [time_rbo1, norm_rbo_flux1, norm_rbo_flux_err1, rbo_exp]),
        ("RBO Data 2", [time_rbo2, norm_rbo_flux2, norm_rbo_flux_err2, rbo_exp]),

    ]
    )


# Creating one dictionary containing all photometric datasets by merging dicts from above
all_datasets = {**tess_sector_datasets, **rbo_dataset}


# pdb.set_trace()


tess_time_offset = 2457000
kepler_time_offset = 2454833

# Orbital Period
periods = [3.3176675]
# Orbital Period Error 
period_error = [0.1] #in days, must be in same unit as periods variable
# Transit Duration
t0s = [2459521.813826 - tess_time_offset]
# Transit Error
t0_error = [0.1]

def phaseup(t, t0, period):
    return (((t-t0) - period/2.) % period - period/2.)
phase = phaseup(tess_time, 2521.81748925, 3.31793068)

plt.plot(phase, tess_flux, linestyle = 'none', marker = '.')
plt.xlim(-0.1, 0.1)

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

mx_red_time = (mx_red_data['time'] - tess_time_offset).values #
mx_red_rv = mx_red_data['radial_velocity'].values
mx_red_rv_err = mx_red_data['rv_error'].values

mx_blue_time = (mx_blue_data['time'] - tess_time_offset).values 
mx_blue_rv = mx_blue_data['radial_velocity'].values
mx_blue_rv_err = mx_blue_data['rv_error'].values

# Merging all the datasets together #
merged_data = pd.concat([mx_red_data, mx_blue_data, hpf_data], axis = 0).reset_index(drop=True)
time_rv = (merged_data['time'] - tess_time_offset).values 
rv = merged_data['radial_velocity'].values
rv_err = merged_data['rv_error'].values
rv_instrument = merged_data['rv_instrument'].values

# Computing a reference time that will be used to normalize the trends model #
time_ref = 0.5 * (time_rv.min() + time_rv.max())

# Making a fine grid that spans the observation window for plotting purposes #
t = np.linspace(time_rv.min() - 5, time_rv.max() + 5, 5000)

# print(merged_data)
plt.figure()
plt.plot(time_rv, rv, linestyle = 'none',marker = 'o')
for n_instrument in np.unique(rv_instrument):
    cull = rv_instrument == n_instrument
    plt.plot(time_rv[cull], rv[cull], linestyle = 'none', marker = 'o', 
        label = n_instrument.replace('maroon_x_blue','MAROON-X (Blue)').replace('maroon_x_red','MAROON-X (Red)'))
plt.title('Radial Velocity Data')
plt.xlabel("time [days]")
plt.ylabel("radial velocity [m/s]")
plt.legend()
# plt.savefig('radial_velocities.png',bbox_inches='tight', pad_inches=0.0)
# plt.close()
# print('all plots are done')

# phase = phaseup(mx_blue_time, 2521.81748925, 3.31793068)
# plt.plot(phase, mx_blue_rv, linestyle = 'none', marker = '.')


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

# For the Radial Velocity Model
Ks = xo.estimate_semi_amplitude(periods, time_rv, rv, rv_err, t0s = t0s)


with pm.Model() as model:

    mean = pm.Normal("mean", mu = 0, sigma = 1.0) # The mean of the lightcurve (which is approximately = 1)
    ustar = xo.distributions.QuadLimbDark("u")  # This  parameter is also being fit but is being sampled from Kipping 2013: 
                                                # (https://ui.adsabs.harvard.edu/abs/2013MNRAS.435.2152K/abstract)

    
    # Stellar Parameters
    BoundedNormal = pm.Bound(pm.Normal, lower=0, upper=1.5)
    m_star = BoundedNormal("m_star", mu=M_star[0], sd=M_star[1]) # Stellar Mass
    r_star = BoundedNormal("r_star", mu=R_star[0], sd=R_star[1]) # Stellar Radius
    teff = pm.Bound(pm.Normal, lower=2000, upper=7000)("teff", mu=Teff[0], sd=Teff[1]) # Effective Temperature
    st_lum = pm.Deterministic("st_lum", (r_star**2) * ((teff/5777)**4)) # Stellar Luminosity
    star_params = [mean, ustar] 
    
    # Planet Parameters (note: Deterministic means values that were derived from the model)
    ror = pm.Uniform("ror", lower=0.01, upper = 0.99, shape=nplanets) # Radius ratio
    r_pl = pm.Deterministic("r_pl", ror*r_star) #in physical units aka stellar radii #not a free parameter but you can calculate at every step
    pm.Deterministic("r_jup", r_pl*Rsun2Rjup)
    m_pl = pm.Uniform("m_pl", lower = 0.01, upper = 1000, testval=Expected_msini, shape=nplanets)
    # pm.Deterministic("m_jup", m_pl*MSun2MJup)
    density_pl = pm.Deterministic("density_pl", m_pl*Mjup2Mearth/((r_pl*Rsun2Rearth)**3) * 5.514) # Convert from rho_earth to g/cm3
    
    # Orbital Parameters
    period = pm.Normal("period", mu = np.array(periods), sigma = np.array(period_error), shape = nplanets)
    t0 = pm.Normal("t0", mu = t0s, sigma = np.array(t0_error), shape = nplanets)
    b = pm.Uniform("b", lower = 0, upper = 1, shape = nplanets, testval = np.array([0.9]))

    #Here are definining our eccentric model
    #Look at section 17 for more information on sampling eccentricity vs. omega on a UnitDisk (https://arxiv.org/pdf/1907.09480.pdf)
    ecs = pmx.UnitDisk("ecs", testval = np.array([[0.1, 0.1]] * nplanets).T, shape = (2, nplanets))
    ecc = pm.Deterministic("ecc", tt.sum(ecs ** 2, axis = 0))
    omega = pm.Deterministic("omega", tt.arctan2(ecs[1], ecs[0]))
    
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
    pm.Deterministic("inclination", orbit.incl)

    # Add time of periastron
    pm.Deterministic("t_p", orbit.t_periastron)

    # Add transit duration, see Eq 14 & 16 here: https://ui.adsabs.harvard.edu/abs/2010exop.book...55W/abstract, Shubham wrote these for the below for his code
    sin_incl = pm.Deterministic("sin_incl", tt.sqrt(1 - orbit.cos_incl**2))

    EccentricityMultiplicativeFactor = pm.Deterministic("X", tt.sqrt(1-(ecc**2))/(1+ecc*tt.sin(omega))) # Eq 16

    pm.Deterministic("transit_duration", (period/np.pi) * tt.arcsin( tt.sqrt( (1+ror)**2 - b**2 ) / (aor*sin_incl) ) * EccentricityMultiplicativeFactor )

    # Add equilibrium temperature
    pm.Deterministic("equilibrium_temp", teff * tt.sqrt(1/(2*aor)))

    # Add temporally the avergaed insolation flux in units of W/m2, see Eq 14 https://arxiv.org/pdf/1702.07314, 
    # luminsity is a black body or L = 4*pi*rstar^2*sigmab*teff^4
    pm.Deterministic("insolation_flux", 4 * np.pi * const.sigma_sb.to('W m-2 K-4').value * teff**4 / (aor**2 * tt.sqrt(1e0 - ecc**2)))

    #################### RV MODEL #################### RV MODEL #################### RV MODEL #####################
    #################### RV MODEL #################### RV MODEL #################### RV MODEL #####################
    #################### RV MODEL #################### RV MODEL #################### RV MODEL #####################
    #pdb.set_trace()

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
    pm.Normal("rv_obs", mu = rv_model, sd = RVError, observed = rv - RVMean)
       
    #################### TRANSIT MODEL #################### TRANSIT MODEL #################### TRANSIT MODEL #####################
    #################### TRANSIT MODEL #################### TRANSIT MODEL #################### TRANSIT MODEL #####################
    #################### TRANSIT MODEL #################### TRANSIT MODEL #################### TRANSIT MODEL #####################

    # Loop over the instruments
    parameters = dict()
    lc_models = dict()
    # gp_preds = dict()
    # gp_preds_with_mean = dict()

    utess = xo.QuadLimbDark("utess")
    tessstar = xo.LimbDarkLightCurve(utess)
    urbo = xo.QuadLimbDark("urbo")
    rbostar = xo.LimbDarkLightCurve(urbo) 

    for n, (name, (time, flux, flux_error, texp)) in enumerate(all_datasets.items()):

        # We define the per-instrument parameters in a submodel so that we don't have to prefix the names manually
        with pm.Model(name=name, model=model):
            # The flux zero point
            mean = pm.Normal("mean", mu=0.0, sigma=10.0)

            if bool(name.find('TESS')+1):
                star = tessstar
            else:
                star = rbostar

            # Calculates light curve for each planet at its time vector
            light_curves = star.get_light_curve(orbit = orbit, r = r_pl, t = time, texp = texp[0]*u.s.to('d')) #this will also change # Change texp to match your data set
    
            # Saves the individual lightcurves 
            pm.Deterministic("light_curves_{}".format(name), light_curves) 

            hi_cad_light_curves = star.get_light_curve(orbit = orbit, r = r_pl, t = t_lc, texp = texp[0]*u.s.to('d'))
    
            # Saves the individual lightcurves 
            pm.Deterministic("hi_cad_light_curves_{}".format(name), hi_cad_light_curves)

            # Save time for lightcurves
            # pm.Deterministic("LCmodeltime", t_lc)

            # pdb.set_trace()

            #Lightcurve Jitter
            Jitter = pm.Uniform(f"{name}_Jitter", 0, 1e3, shape = len(all_datasets[name][0]))
            LC_error =

            parameters[name] = [mean]
            parameters[f"{name}_noise"] = [Jitter]
            
            # The light curve model
        def lc_model(mean, star, ror, texp, t):
            return mean + 1e3 * tt.sum(light_curves(orbit=orbit, r=ror, t=time, texp = texp[0]*u.s.to('d')),
                axis=-1,
            )

        lc_model = partial(lc_model, mean, star, ror, texp)
        lc_models[name] = lc_model

    # Full photometric model, the sum of all transits + the baseline (mean)
    lc_model = mean + tt.sum(light_curves, axis=-1)
    
    # The likelihood function assuming known Gaussian uncertainty
    pm.Normal("transit_obs", mu = lc_model, sd = flux_error, observed = flux)

    ################## OPTIMIZING ################ OPTIMIZING ########################### OPTIMIZING ##################
    ################## OPTIMIZING ################ OPTIMIZING ########################### OPTIMIZING ##################
    ################## OPTIMIZING ################ OPTIMIZING ########################### OPTIMIZING ##################
    
    # In this portion, we will pptimize the Maximum a Posteriori AKA MAP Solution

    # Defining a random starting point based on our priors defined 
    map_soln = model.test_point
    # pdb.set_trace()

    #optimizing system parameters for each lightcurve
    for name in all_datasets:
        if bool(name.find('TESS')+1):
            map_soln = pmx.optimize(map_soln, parameters[name] + [ror, b, r_star, utess])
        else:
            map_soln = pmx.optimize(map_soln, parameters[name] + [ror, b, r_star, urbo])

    #optimizing noise for each instrument
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


pdb.set_trace()
################ TRANSIT AND RV INITIAL BEST FIT PLOTS ################## TRANSIT AND RV INITIAL BEST FIT PLOTS ##################
################ TRANSIT AND RV INITIAL BEST FIT PLOTS ################## TRANSIT AND RV INITIAL BEST FIT PLOTS ##################
################ TRANSIT AND RV INITIAL BEST FIT PLOTS ################## TRANSIT AND RV INITIAL BEST FIT PLOTS ################## 

t0 = map_soln["t0"]
period = map_soln["period"] 
rv_model = map_soln["rv_model"]
rv_model_pred = map_soln["rv_model_pred"]

vrad = map_soln["vrad"]
vrad_pred = map_soln["vrad_pred"]

################ RV RESIDUALS PLOT ######### RV RESIDUALS PLOT ######### RV RESIDUALS PLOT ######### RV RESIDUALS PLOT #############
################ RV RESIDUALS PLOT ######### RV RESIDUALS PLOT ######### RV RESIDUALS PLOT ######### RV RESIDUALS PLOT #############
################ RV RESIDUALS PLOT ######### RV RESIDUALS PLOT ######### RV RESIDUALS PLOT ######### RV RESIDUALS PLOT #############

datelabel = "{:%m-%d-%Y}".format(datetime.datetime.now())
with model:
    fig, axes = plt.subplots(2, 1, figsize = (10, 5), sharex = True)
    ax = axes[0]
    ax.errorbar(time_rv, rv - map_soln["RVMean"], yerr = rv_err, fmt = ".k")
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
    ax.errorbar(time_rv, rv - map_soln['RVMean'] - map_soln["rv_model"], yerr = rv_err, fmt = ".k")
    ax.axhline(0, color = "k", lw = 1)
    ax.set_title("residuals")
    ax.set_ylabel("residuals [m/s]")
    # ax.set_xlim(3230, 3250)
    ax.set_xlabel("time [days]")
    ax.figure.savefig('TOI-5349-b_residuals_plot_{}.pdf'.format(datelabel), bbox_inches = 'tight', pad_inches = 0.0)
# print(np.unique(map_soln['RVMean'])) 
# print(map_soln['RVOffset'])


###### PRELIM TRANSIT PHASE PLOT ###### PRELIM TRANSIT PHASE PLOT ######
###### PRELIM TRANSIT PHASE PLOT ###### PRELIM TRANSIT PHASE PLOT ######
###### PRELIM TRANSIT PHASE PLOT ###### PRELIM TRANSIT PHASE PLOT ######

fig, ax = plt.subplots(figsize = (10, 5))

x_fold = (time_lc - t0 + 0.5 * period) % period - 0.5 * period
m = np.abs(x_fold) < 0.5 # plot will only show phases between -0.5 to 0.5

ax.scatter(x_fold[m], 1e3 * (flux[m]), #*******
    c = "k",
    marker = ".",
    alpha = 0.2,
    linewidths = 0,
)

lc_mod = map_soln['light_curves']
lc_modx = np.sort(x_fold)
lc_mody = lc_mod[np.argsort(x_fold)]

ax.plot(lc_modx, 1e3 * (lc_mody + map_soln["mean"]), c = "purple", zorder = 1) #*******

# Overplot the phase binned light curve
bins = np.linspace(-0.51, 0.51, 100)
denom, _ = np.histogram(x_fold, bins)
num, _ = np.histogram(x_fold, bins, weights = flux)
denom[num == 0] = 1.0

ax.scatter(0.5 * (bins[1:] + bins[:-1]), 1e3 * num / denom, #*******
    color = "C1",
    zorder = 2,
    linewidths = 0,
)

ax.set_xlim(-0.5, 0.5)
ax.set_ylabel("de-trended flux [ppt]")
_ = ax.set_xlabel("time since transit")

ax.figure.savefig('TOI-5349-b_LC_phase_plot_{}.pdf'.format(datelabel), bbox_inches = 'tight', pad_inches = 0.0)


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

plt.plot(rv_modx, rv_mody, c = "purple", zorder = 1, label = 'model') 
plt.xlim(-0.5 * period, 0.5 * period)
plt.title("TOI-5349b")
plt.ylabel("radial velocity [ms/s]")
plt.xlabel("phase [days]")
plt.legend()
plt.savefig('TOI-5349-b_RV_phase_plot_{}.pdf'.format(datelabel),bbox_inches = 'tight', pad_inches = 0.0)

#pdb.set_trace()

############ SAMPLING THE DATA ############ SAMPLING THE DATA ############ SAMPLING THE DATA ############ SAMPLING THE DATA ################### 
############ SAMPLING THE DATA ############ SAMPLING THE DATA ############ SAMPLING THE DATA ############ SAMPLING THE DATA ################### 
############ SAMPLING THE DATA ############ SAMPLING THE DATA ############ SAMPLING THE DATA ############ SAMPLING THE DATA ################### 

NSteps = 1000
Nchains = 2
Ncores = 1
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

var_names = ["period", "t0", 'ecc', 'omega', 'K', 'RVOffset', 'RVJitter', #  Traditional RV Paramters
             'u', "ror", 'aor', 'b', # The transit parameters
             'teff', 'r_star', 'm_star', 'st_lum', 'rho_star', # The Physical Stellar Parameters
             'm_pl', 'r_jup',  'density_pl'] # The Planetary Parameters 

output_dict = {'time_lc' : time_lc, #Save photometry
               'lc' : flux,
               'lc_err' : flux_error,
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

posteriors_df.to_csv('TOI-5349_posteriors_{}.csv'.format(datelabel))

# pdb.set_trace()

################ GENERATING CORNER + TRACE PLOTS ################# GENERATING CORNER + TRACE PLOTS ################## GENERATING CORNER + TRACE PLOTS #######
################ GENERATING CORNER + TRACE PLOTS ################# GENERATING CORNER + TRACE PLOTS ################## GENERATING CORNER + TRACE PLOTS #######
################ GENERATING CORNER + TRACE PLOTS ################# GENERATING CORNER + TRACE PLOTS ################## GENERATING CORNER + TRACE PLOTS #######

### TRACE PLOT ###
### TRACE PLOT ###
### TRACE PLOT ###

_ = az.plot_trace(trace, var_names = var_names) 

plt.savefig('TOI-5349_trace_plot_{}.pdf'.format(datelabel),bbox_inches = 'tight', pad_inches = 0.0)

### CORNER PLOT ###
### CORNER PLOT ###
### CORNER PLOT ###

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

# pdb.set_trace()

### STEPS TO READ PKL FILE AND VIEW FIT RESULTS ###

# To read in pkl file:

# import pickle
# import glob

# with open( glob.glob('TOI*pkl')[0], 'rb') as f:
#     output_dict = pickle.load(f)
#     map_soln = output_dict['map_solution']
#     flat_samps = output_dict['flat_samples']
#     time_lc = output_dict['time_lc']
#     flux = output_dict['lc']
#     flux_error = output_dict['lc_err']
#     time_rv = output_dict['time_rv']
#     rv = output_dict['rv']
#     rv_error = output_dict['rv_err']
#     rv_instrument = output_dict['rv_instrument']

# with open( glob.glob('TOI*pkl')[0], 'rb') as f:
#     output_dict = pickle.load(f)
#     map_soln = output_dict['map_solution']
#     flat_samps = output_dict['flat_samples']
#     time_lc = output_dict['time_lc']
#     flux = output_dict['lc']
#     flux_error = output_dict['lc_err']
#     rv = output_dict['time_rv']
#     y_rv = output_dict['rv']
#     rv_err = output_dict['rv_err']
#     rv_instrument = output_dict['rv_instrument']
#     map_soln =   output_dict['map_solution']

########### GENERATING FINAL PHASE FOLDED PLOTS ########### GENERATING FINAL PHASE FOLDED PLOTS ########### GENERATING FINAL PHASE FOLDED PLOTS ###########
########### GENERATING FINAL PHASE FOLDED PLOTS ########### GENERATING FINAL PHASE FOLDED PLOTS ########### GENERATING FINAL PHASE FOLDED PLOTS ###########
########### GENERATING FINAL PHASE FOLDED PLOTS ########### GENERATING FINAL PHASE FOLDED PLOTS ########### GENERATING FINAL PHASE FOLDED PLOTS ###########

gp_mod=np.zeros(len(time_lc))

######## TRANSIT FOLDED PHASE PLOTS ####### TRANSIT FOLDED PHASE PLOTS #######
######## TRANSIT FOLDED PHASE PLOTS ####### TRANSIT FOLDED PHASE PLOTS #######
######## TRANSIT FOLDED PHASE PLOTS ####### TRANSIT FOLDED PHASE PLOTS #######

for n, letter in enumerate("b"):

    plt.figure()

    plt.gca().tick_params(direction = "in", which = 'both',bottom = True, top = False, left = True, right = True)
    # Get the posterior median orbital parameters
    p = np.median(flat_samps["period"][n])
    t0 = np.median(flat_samps["t0"][n])
    

    # Plot the folded data
    x_fold = (time_lc - t0 + 0.5 * p) % p - 0.5 * p
    m = (np.abs(x_fold) < 0.3) &(flux < 1.05)
    plt.plot(
        x_fold[m], flux[m] - gp_mod[m], ".k", label = "data", zorder = -1000)

    # Plot the folded model
    pred = np.percentile(flat_samps["light_curves"][:, n, :], [16, 50, 84], axis = -1) # finding the scatter between the 16th through 84th percentile (its the +/- 1 sigma of a gaussian distribution)
    pred +=1
    sort=np.argsort(x_fold)
    plt.plot(x_fold[sort], pred[1][sort], color = "C1", label = "model")
    art = plt.fill_between(
        x_fold[sort], 
        pred[0][sort], 
        pred[2][sort], 
        color = "C1", 
        alpha = 0.5, 
        zorder = 1000
    )
    art.set_edgecolor("none")

    # Annotate the plot with the planet's period
    txt = "period = {0:.4f} +/- {1:.4f} d".format(
        np.mean(flat_samps["period"][n].values),
        np.std(flat_samps["period"][n].values),
    )
    plt.annotate(
        txt,
        (0, 0),
        xycoords = "axes fraction",
        xytext = (5, 5),
        textcoords = "offset points",
        ha = "left",
        va = "bottom",
        fontsize = 12,
    )

    plt.legend(fontsize = 10, loc = 4)
    plt.xlabel("time since transit [days]")
    plt.ylabel("de-trended flux")
    plt.title("TOI-5349{0}".format(letter))
    plt.xlim(-0.3, 0.3)
    plt.show()

######## RV FOLDED PHASE PLOTS ####### RV FOLDED PHASE PLOTS #######
######## RV FOLDED PHASE PLOTS ####### RV FOLDED PHASE PLOTS #######
######## RV FOLDED PHASE PLOTS ####### RV FOLDED PHASE PLOTS #######

for n, letter in enumerate("b"):
    plt.figure()
    plt.gca().tick_params(direction = "in", which='both',bottom = True, top = False, left = True, right = True)

    # Get the posterior median orbital parameters
    p = np.median(flat_samps["period"][n])
    t0 = np.median(flat_samps["t0"][n])

    # Compute the median of posterior estimate of the background RV
    # and the contribution from the other planet. Then we can remove
    # this from the data to plot just the planet we care about.
    bkg = map_soln["RVMean"]
    
    # Plot the folded data
    x_fold = (rv - t0 + 0.5 * p) % p - 0.5 * p
    # plt.errorbar(x_fold, y_rv -bkg, yerr=rv_err, fmt=".k", label="data")
    for thisinstrument in pd.Series(rv_instrument).unique():
        mask = rv_instrument == thisinstrument
        plt.errorbar(
            x_fold[mask], (rv - bkg)[mask], 
            yerr = rv_err[mask], 
            marker = 'o', 
            linestyle = 'none', 
            ecolor=rgba('black',0.2),
            markeredgecolor=rgba('black',0.2),
            label="{}".format(thisinstrument.replace('maroon_x_blue','MAROON-X (Blue)').replace('maroon_x_red','MAROON-X (Red)'))
            )
        plt.tick_params(axis = 'both',which = 'major', width = 1.00, length = 5)
        plt.tick_params(axis = 'both', which ='minor', direction ='in', length = 4, width = 1)
        
    # Compute the posterior prediction for the folded RV model for this planet
    t_rv = np.linspace(rv.min() - 5, rv.max() + 5, 5000)
    t_fold = (t_rv - t0 + 0.5 * p) % p - 0.5 * p
    inds = np.argsort(t_fold)
    bkg = np.median(flat_samps['RVMean'],axis=1)
    pred = np.percentile(flat_samps["rv_model_pred"][inds,:], [16, 50, 84], axis=-1)
    plt.plot(t_fold[inds], pred[1], color="C1", label="model")
    art = plt.fill_between(
        t_fold[inds], pred[0], pred[2], color="C1", alpha=0.3
    )
    art.set_edgecolor("none")

    plt.legend(fontsize=10)
    plt.xlim(-0.5 * p, 0.5 * p)
    plt.xlabel("phase [days]")
    plt.ylabel("radial velocity [m/s]")
    
    def phasetodays(x):
        return x*p
    def daystophase(x):
        return x / p
    secax = plt.gca().secondary_xaxis('top', functions=(daystophase, phasetodays))
    secax.set_xlabel('phase')

    plt.legend(fontsize=10)
    plt.xlabel("phase [days]")
    plt.ylabel("radial velocity [m/s]")
    plt.title("TOI-5349 {}".format(letter))
    plt.show()

# trace.to_dataframe(trace, var_names=var_names)
# dir(trace) # to check what commands are available for trace
# samples = trace.to_dataframe(trace, varnames=var_names)