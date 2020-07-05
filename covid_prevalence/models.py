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

    The author can be contacted at steven.horn@forces.gc.ca
'''

import logging
import pymc3 as pm
import theano.tensor as tt
import theano

log = logging.getLogger(__name__)

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
    return_all=False,
):
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

    # Runs SIRa model:
    def next_day(lambda_t, # infection rate
                 S_t,      # number of susceptible
                 E_t,      # number of exposed
                 E_cum_t,  # cumulative number exposed
                 R_t,      # number of resolved
                 Ia_t,     # number of asymptomatic-infected 
                 Is_t,     # number of pre-symptomatic infected
                 _,        # not used
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

        return S_t, E_t, E_cum_t, R_t, Ia_t, Is_t, detected

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
            new_I_0
          ],
        non_sequences=[mu, mus, gamma, pa, pu, N],
    )
    S_t, E_t, Ecum_t, R_t, Ia_t, Is_t, new_detections = outputs

    # This is the population susceptible
    if name_S_t is not None:
        pm.Deterministic(name_S_t, S_t)

    # This is important for prevalence - the infected asymptomatic = presymptomatic
    if name_I_t is not None:
        pm.Deterministic(name_I_t, Ia_t + Is_t)

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