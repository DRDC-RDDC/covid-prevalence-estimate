'''
    COVID-Prevalence  Copyright (c) Her Majesty the Queen in Right of Canada, 
    as represented by the Minister of National Defence, 2020.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

import logging
import numpy as np
import pymc3 as pm
import datetime
import theano.tensor as tt
import theano
from dateutil.parser import isoparse
from covid19_inference.model import Cov19Model
import covid19_inference as cov19

log = logging.getLogger(__name__)

def dynamicChangePoints(pop):
    bd = isoparse(pop['date_start']) # This is the first day of data
    change_points_d2 = []
    daystep = pop['daystep_lambda']
    delta = datetime.datetime.utcnow() - bd
    for dd in np.arange(daystep,delta.days-daystep,daystep,dtype="int"):
        change_points_d2 += [
            dict( # Fit the end
                pr_mean_date_transient=bd+datetime.timedelta(days=int(dd)),
                pr_median_transient_len=daystep/2,    # how fast is this transition?  
                pr_sigma_transient_len=0.5,   # uncertainty how long to apply
                pr_sigma_date_transient=2,    # uncertainty when applied
                relative_to_previous=True,    
                pr_factor_to_previous=1,      # mean moves log this -> i.e. log(1) = 0+
                pr_median_lambda=0,           # normal offset rel to prev
                pr_sigma_lambda=0.2,
                )
        ]

    change_points = change_points_d2 
    return change_points

def SEIRa(
    lambda_t_log,             # spreading rate over time
    gamma,                    # latent
    mu,                       # time for asymptomatic recovery
    mus,                      # time for symptomatic to show symptoms
    pa,                       # asymptomatic probability distribution
    pu=0,                     # under-reporting probability
    asym_ratio=1.0,           # relative infectiousness of asymptomatic cases
    name_new_I_t="new_I_t",
    name_I_begin="I_begin",
    name_I_t="I_t",
    name_S_t="S_t",
    pr_E_begin=20,
    pr_Ia_begin=20,
    pr_Is_begin=20,
    pr_mean_median_symptomatic=5,
    pr_sigma_median_symptomatic=1,
    sigma_isolate=0.3,
    model=None,
    return_all=False):
    # Total number of people in population
    N = model.N_population

    # Prior distributions of starting populations (infectious, exposed, susceptibles)
    Ia_begin =   pm.HalfCauchy(name="Ia_begin", beta=pr_Ia_begin) #pr_I_begin
    E_begin =    pm.HalfCauchy(name="E_begin", beta=pr_E_begin)
    Ecum_begin = pm.HalfCauchy(name="Ecum_begin", beta=1)
    R_begin =    pm.HalfCauchy(name="R_begin", beta=1)
    Is_begin =   pm.HalfCauchy(name="Is_begin", beta=pr_Is_begin) #pr_I_begin

    # Prior for susceptible
    S_begin = N  - Ia_begin - Is_begin

    lambda_t = tt.exp(lambda_t_log)
    new_I_0 = tt.zeros_like(Ia_begin)
    new_E_0 = tt.zeros_like(Ia_begin)

    # Runs SIRa model:
    def next_day(lambda_t, # infection rate
                 S_t,      # number of susceptible
                 E_t,      # number of exposed
                 E_cum_t,  # cumulative number exposed
                 R_t,      # number of resolved
                 Ia_t,     # number of asymptomatic-infected 
                 Is_t,     # number of pre-symptomatic infected
                 _,        # not used
                 ne,       # not used - for saving new exposures
                 mu,       # recovery rate (asymptomatic)
                 mus,
                 gamma,    # incubation rate
                 pa,       # probability asymptomatic
                 pu,
                 N):       # population size
      
        # New infections:

        # New Exposed from asymptomatic + presymptomatic
        new_E_t = lambda_t / N * (asym_ratio*Ia_t + Is_t) * S_t

        S_t = S_t - new_E_t

        # Exposed become infectious
        new_I_t = E_t * gamma

        E_t = E_t + new_E_t - new_I_t
        E_cum_t = E_cum_t + new_E_t
        
        pnodet = 1.0-(1.0-pa)*(1.0-pu)

        new_Is_t = (1 - pnodet) * new_I_t    # assume only symptomatic cases are reported (observed) - this is the variable used for Bayesian updates
        new_Ia_t = pnodet * new_I_t

        detected = mus * Is_t

        # distribute the new infections to be asymptomatic or symptomatic infectors
        Ia_t = Ia_t + new_Ia_t - mu * Ia_t  # recover as mu
        Is_t = Is_t + new_Is_t - detected   # isolate as lognorm
        R_t = mu * Ia_t + detected          # resolved either recovered or isolated
        new_I_t = tt.clip(new_I_t, 0, N/3)  # for stability
        Ia_t = tt.clip(Ia_t, -1, N-1)       # for stability
        Is_t = tt.clip(Is_t, -1, N-1)       # for stability
        E_t = tt.clip(E_t, -1, N-1)         # for stability
        S_t  = tt.clip(S_t,  -1, N)         # bound to population
        R_t  = tt.clip(R_t,  -1, N)         # bound to population

        return S_t, E_t, E_cum_t, R_t, Ia_t, Is_t, detected, new_E_t

    # theano scan returns two tuples, first one containing a time series of
    # what we give in outputs_info : S, I, new_I
    outputs, _ = theano.scan(
        fn=next_day,
        sequences=[lambda_t],
        outputs_info=[
            S_begin, 
            E_begin,
            Ecum_begin,
            R_begin,
            Ia_begin,
            Is_begin,
            new_I_0,
            new_E_0
          ],
        non_sequences=[mu, mus, gamma, pa, pu, N],
    )
    S_t, E_t, Ecum_t, R_t, Ia_t, Is_t, new_detections, new_E_t = outputs

    # This is the population susceptible
    if name_S_t is not None:
        pm.Deterministic(name_S_t, S_t)

    # This is important for prevalence - the infected asymptomatic = presymptomatic
    if name_I_t is not None:
        pm.Deterministic(name_I_t, Ia_t + Is_t)

    pm.Deterministic("new_E_t", new_E_t)
    pm.Deterministic("E_t", E_t)
    pm.Deterministic("Ecum_t", Ecum_t)
    pm.Deterministic("R_t", R_t)
    pm.Deterministic("Is_t", Is_t)
    pm.Deterministic("Ia_t", Ia_t)
    pm.Deterministic("new_detections", new_detections)

    # Return the new symptomatic cases as the element to fit to data
    if return_all:
        return new_detections, E_t, Ia_t, Is_t, S_t
    else:
        return new_detections

class PrevModel(Cov19Model):
    '''
    '''
    def __init__(
            self,
            new_cases_obs,
            data_begin,
            fcast_len,
            diff_data_sim,
            N_population,
            pop = None,
            settings = None,
            name="",
            model=None):

        # Initialize the base model (from covid_inference)
        super().__init__(
            new_cases_obs=new_cases_obs, 
            data_begin=data_begin, 
            fcast_len=fcast_len,
            diff_data_sim=diff_data_sim,
            N_population=N_population,
            name=name, 
            model=model)

        # apply change points, lambda is in log scale
        lambda_t_log = cov19.model.lambda_t_with_sigmoids(
            pr_median_lambda_0 = pop['median_lambda_0'],
            pr_sigma_lambda_0 = pop['sigma_lambda_0'],  # The initial spreading rate
            change_points_list = dynamicChangePoints(pop),  # these change points are periodic over time and inferred
        )

        # Probability of asymptomatic case
        if settings['model']['pa'] == 'Beta':
          pa = pm.Beta(name="pa", alpha=settings['model']['pa_a'], beta=settings['model']['pa_b'])
        elif settings['model']['pa'] == 'BoundedNormal':
          BoundedNormal = pm.Bound(pm.Normal, lower=settings['model']['pa_lower'], upper=settings['model']['pa_upper'])
          pa = BoundedNormal(name="pa", mu=settings['model']['pa_mu'], sigma=settings['model']['pa_sigma'])
        else:
          pa = pm.Uniform(name="pa", lower=0.15, upper=0.5)

        # Probability of undetected case
        if settings['model']['pu'] == 'BoundedNormal':
          BoundedNormal_pu = pm.Bound(pm.Normal, lower=settings['model']['pu_a'], upper=settings['model']['pu_b'])
          pu = BoundedNormal_pu(name="pu", mu=settings['model']['pu_b']-settings['model']['pu_a'], sigma=0.2)
        else:
          pu = pm.Uniform(name="pu", upper=settings['model']['pu_b'], lower=settings['model']['pu_a'])

        mu = pm.Lognormal(name="mu", mu=np.log(1 / settings['model']['asym_recover_mu_days']), sigma=settings['model']['asym_recover_mu_sigma'])    # Asymptomatic infectious period until recovered
        mus = pm.Lognormal(name="mus", mu=np.log(1 / settings['model']['sym_recover_mu_days']), sigma=settings['model']['sym_recover_mu_sigma'])   # Pre-Symptomatic infectious period until showing symptoms -> isolated
        gamma = pm.Lognormal(name="gamma", mu=np.log(1 / settings['model']['gamma_mu_days']), sigma=settings['model']['gamma_mu_sigma'])

        new_Is_t = SEIRa(lambda_t_log, gamma, mu, mus, pa, pu,
                        asym_ratio = settings['model']['asym_ratio'],  # 0.5 asymptomatic people are less infectious? - source CDC
                        pr_Ia_begin=pop['pr_Ia_begin'],
                        pr_Is_begin=pop['pr_Is_begin'],
                        model=self)
        
        new_cases_inferred_raw = cov19.model.delay_cases(
            cases=new_Is_t,
            pr_mean_of_median=pop['pr_delay_mean_of_median'],
        )

        # apply a weekly modulation, fewer reports during weekends
        if 'noweekmod' in pop and pop['noweekmod']:
          log.info('Not using weekly modulation')
          pm.Deterministic("new_cases", new_cases_inferred_raw)
          new_cases_inferred = new_cases_inferred_raw
        else:
          new_cases_inferred = cov19.model.week_modulation(
              new_cases_inferred_raw,
              pr_mean_weekend_factor=pop['pr_mean_weekend_factor'],
              pr_sigma_weekend_factor=pop['pr_sigma_weekend_factor'],
              name_cases="new_cases")

        # set the likeliehood
        cov19.model.student_t_likelihood(new_cases_inferred)