
'''

This is the main file for the worker process.  There will normally be
many worker processes.


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

import os

# Here, we add an environment variable to help theano locate the BLAS libraries.
# found in ./usr/lib/x86_64-linux-gnu/libblas.so
# More info at http://deeplearning.net/software/theano/troubleshooting.html
os.environ["THEANO_FLAGS"] = 'blas.ldflags="-L/usr/lib/x86_64-linux-gnu/ -lblas"'

import time
import logging
import json
import git
from covid_prevalence.rediswq import RedisWQ 
import datetime
import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as tt
import pandas as pd
import covid19_inference as cov19
from covid19_inference.model import *
from datetime import timezone
from dateutil.parser import parse, isoparse
from pathlib import Path
import pymc3.stats as pms

import covid_prevalence as covprev
from covid_prevalence.models import SEIRa     # Our model
from covid_prevalence.models import dynamicChangePoints # Dynamic spreading rate
from covid_prevalence.plots import plot_data, plot_fit, plot_IFR, plot_posteriors, plot_prevalence
from covid_prevalence.repository import gitpush
from covid_prevalence.utility import get_folders

# this is used for json serialization of dates
def converters(o):
    if isinstance(o, datetime.datetime):
        return o.isoformat()

# These plots just look nicer!
plt.style.use('seaborn-darkgrid')

# Configure logging format
logging.basicConfig(level=logging.INFO, 
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S')

pymc3logger = logging.getLogger('pymc3')
pymc3logger.setLevel(logging.DEBUG)

log = logging.getLogger(__name__)

# These are (optionally) configurable values via environment variables
host = os.getenv("REDIS_SERVICE_HOST", default="redis")
queuename = os.getenv('redis_worker_queue', default='covidprev')
oauth = os.getenv('gitlab_oauth', default='53gZwasv6UmExxrohqPm')
timelimit = 60*60*1 # these hours should be enough per region - make env?
repo_path = "/data/covid-prevalence"

# push the results at end.  If false, they mush be manually pushed
always_push = False

if __name__=='__main__':
  q = RedisWQ(name=queuename, host=host)

  # priority queue
  pq = RedisWQ(name=queuename + 'p1', host=host)

  log.info("Worker with sessionID: " +  q.sessionID())
  log.info("Initial queue state: empty=" + str(q.empty()))

  repo = git.Repo(repo_path) # repo = git.Repo('/content/covid-prevalence')

  # run the processing loop - if the queue is empty, then this program will end
  # This sets up a loop to get data from the queue.  Continues if 
  # 30 seconds has elapsed - logging a waiting message.
  timed_out = False
  num_wait_loops = 0
  while not timed_out and not q.empty():
    # if we have something in the prioiry queue, then that gets processed next
    ispq = False
    if not pq.empty():
      item = pq.lease(lease_secs=timelimit, block=True, timeout=60)
      if item is not None:
        ispq = True
        log.info("Processing from queue.")
      else:
        # in case another worker got the priority
        item = q.lease(lease_secs=timelimit, block=True, timeout=60)
    else:
      item = q.lease(lease_secs=timelimit, block=True, timeout=60)

    if item is not None:
      start_time = datetime.datetime.utcnow()

      num_wait_loops = 0
      itemstr = item.decode("utf-8")
      log.debug("Recieved task " + itemstr)
      task = json.loads(itemstr)
      model = task['model']
      settings = task['settings']
      pop = task['pop']
      log.info("Working on " + pop['name'])

      savefolder, folder = get_folders(pop, rootpath='/data')

      # First we read the json into a dataframe
      nc_df = pd.read_json(task['new_cases'], orient='table')

      # There should only be one column, so we will have only that data selected as a series type
      new_cases = nc_df[nc_df.columns[0]]

      # Same for cumulative deaths, as for new cases
      cd_df = pd.read_json(task['cum_deaths'], orient='table')
      cum_deaths = cd_df[cd_df.columns[0]]

      bd = isoparse(pop['date_start']) # This is the first day of data

      # Model parameters
      params_model = dict(
        new_cases_obs=new_cases,
        data_begin=bd,
        fcast_len=pop['fcast_len'],             # forecast model
        diff_data_sim=pop['diff_data_sim'],     # number of days for burn-in
        N_population=pop['N'],
      )

      with cov19.model.Cov19Model(**params_model) as this_model:
        # apply change points, lambda is in log scale
        lambda_t_log = cov19.model.lambda_t_with_sigmoids(
            pr_median_lambda_0 = pop['median_lambda_0'],
            pr_sigma_lambda_0 = pop['sigma_lambda_0'],  # The initial spreading rate
            change_points_list = dynamicChangePoints(pop),  # these change points are periodic over time and inferred
        )

        if model['pa'] == 'Beta':
          pa = pm.Beta(name="pa", alpha=model['pa_a'], beta=model['pa_b'])
        elif model['pa'] == 'BoundedNormal':
          BoundedNormal = pm.Bound(pm.Normal, lower=model['pa_lower'], upper=model['pa_upper'])
          pa = BoundedNormal(name="pa", mu=model['pa_mu'], sigma=model['pa_sigma'])
        else:
          pa = pm.Uniform(name="pa", lower=0.15, upper=0.5)

        # Probability of undetected case
        if model['pu'] == 'BoundedNormal':
          BoundedNormal_pu = pm.Bound(pm.Normal, lower=model['pu_a'], upper=model['pu_b'])
          pu = BoundedNormal_pu(name="pu", mu=model['pu_b']-model['pu_a'], sigma=0.2)
        else:
          pu = pm.Uniform(name="pu", upper=model['pu_b'], lower=model['pu_a'])

        mu = pm.Lognormal(name="mu", mu=np.log(1 / model['asym_recover_mu_days']), sigma=model['asym_recover_mu_sigma'])    # Asymptomatic infectious period until recovered
        mus = pm.Lognormal(name="mus", mu=np.log(1 / model['sym_recover_mu_days']), sigma=model['sym_recover_mu_sigma'])   # Pre-Symptomatic infectious period until showing symptoms -> isolated
        gamma = pm.Lognormal(name="gamma", mu=np.log(1 / model['gamma_mu_days']), sigma=model['gamma_mu_sigma'])

        new_Is_t = SEIRa(lambda_t_log, gamma, mu, mus, pa, pu,
                        asym_ratio = model['asym_ratio'],  # 0.5 asymptomatic people are less infectious? - source CDC
                        pr_Ia_begin=pop['pr_Ia_begin'],
                        pr_Is_begin=pop['pr_Is_begin'],
                        model=this_model)
        
        new_cases_inferred_raw = cov19.model.delay_cases(
            cases=new_Is_t,
            pr_mean_of_median=pop['pr_delay_mean_of_median'],
        )

        # apply a weekly modulation, fewer reports during weekends
        if 'noweekmod' in pop and pop['noweekmod']:
          log.info('Not using weekly modulation')
          new_cases_inferred_tr = pm.Deterministic("new_cases", new_cases_inferred_raw)
          new_cases_inferred = new_cases_inferred_raw
        else:
          new_cases_inferred = cov19.model.week_modulation(
              new_cases_inferred_raw,
              pr_mean_weekend_factor=pop['pr_mean_weekend_factor'],  # 1.1
              pr_sigma_weekend_factor=pop['pr_sigma_weekend_factor'],   # 1.2 0.5 default
              name_cases="new_cases")

        # set the likeliehood
        cov19.model.student_t_likelihood(new_cases_inferred)

        # sampling settings
        numsims = settings['numsims']
        numtune = settings['numtune']

        # overrides for populations
        if 'numsims' in pop:
          log.info('numsims override in population')
          numsims = pop['numsims']
        if 'numtune' in pop:
          log.info('numtune override in population')
          numsims = pop['numtune']

        cores = 4
        if pop['run'] == False:
          plot_data(this_model, new_cases, pop, settings)
        else:
          log.info('Starting sampling')
          trace = pm.sample(
            model=this_model, 
            chains=settings['chains'],
            tune=numtune, 
            draws=numsims, 
            n_init=50000,     # we really should have converged by 50k
            init="advi+adapt_diag", cores=cores)

        # TODO: check if advi did not converge (model mismatch)

        # Here, we record how long it took to run the model
        stop_time = datetime.datetime.utcnow()
        elapsed_time = stop_time-start_time
        pop['compute_time'] = str(elapsed_time)
        pop['divs'] = -1 # This will record the number of divergences (-1 default)
        pop['draws'] = numsims # This will record the number of runs (-1 default)
        pop['tunes'] = numtune # This will record the number of tuning samples

        if pop['run'] == True:
          log.info('Generating plots')
          try:
            plot_fit(this_model, trace, new_cases, pop, settings, rootpath='/data')
            plot_prevalence(this_model, trace, pop, settings, rootpath='/data')
          except Exception as e:
            log.error(str(e))

          log.info('Saving statistics')
          try:
            divs = trace.get_sampler_stats('diverging')
            pop['divs'] = np.sum(divs)
            llostat = pms.loo(trace,pointwise=True, scale="log")
            llostat_str = str(llostat)

            summary = pm.summary(trace, var_names=["pa", "pu","mu","mus", "Is_begin","Ia_begin","E_begin"])
            summary_str = str(summary)
            savepath = savefolder + '/'+folder+'_stats.txt'
            with open(savepath, 'w') as f:
              f.write('%d Divergences \n' % np.sum(divs))
              f.write(llostat_str)
              f.write('\n')
              f.write(summary_str)
          except Exception as e:
            log.error(str(e))

          log.info('Updating CSV files')
          try:
            _, _ = covprev.data.savecsv(this_model, trace, pop, rootpath='/data')
          except Exception as e:
            log.error(str(e))

      # We try to push the results to git
      regionid = pop["source_country"] + pop["source_state"] + ("" if pop["source_region"] == None else pop["source_region"])
      regionid = regionid.replace(' ','')  # regionid = 'USColoradoElPaso'
      message = "Updates for " + pop['name'] # message = "Updates for " + regionid
      try:
        log.info('Local commit prior to pulling')
        #repo.git.checkout('-b', worker_branch + '_' + regionid)
        # we need to commit our changes before pulling
        repo.git.add('--all')
        repo.git.commit('-m', message, author='Steven Horn')
      except git.GitCommandError as e:
        log.error(str(e))
      
      if always_push:
        numpushattempts = 0
        success = False
        while not success and numpushattempts < 3:
          numpushattempts = numpushattempts + 1
          try:
            log.info('Pushing branch %s to origin' % repo.active_branch.name)
            repo.remotes.origin.push(repo.active_branch.name)
            success = True
          except Exception as e:
            log.error("Error pushing. " + str(e))
            # Wait before trying again to avoid hammering git
            time.sleep(30)

            # this probably happened since the remote was updated before we pushed,
            # we will try again.
            # force reset the branch (removing the failed merge result)
            #repo.git.merge('--abort')
            #repo.git.checkout('-f', repo.active_branch.name)
        if not success:
          log.error('Unable to commit %s' % pop['name'])

      # Mark as completed and remove from work queue.
      if ispq:
        pq.complete(item)
      else:
        q.complete(item)

      log.info("Completed " + pop['name'])
    else:
      num_wait_loops = num_wait_loops + 1
      log.info("Waiting for work")

      if num_wait_loops > 10:  # minutes
        # this will break the loop and exit the program
        timed_out = True

      # this is to poke the queue for failed workers
      # currently disabled - not tested for redis db concurrency
      # q.check_expired_leases()

  log.info("Worker queue empty, exiting")
