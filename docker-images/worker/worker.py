
'''

This is the main file for the worker process.  There will normally be
many worker processes.

TODO: Document


LICENSE

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
# THEANO_FLAGS=blas.ldflags="-L/usr/lib/x86_64-linux-gnu/ -lblas"
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
from covid_prevalence.models import SEIRa     # Our model
import covid_prevalence as covprev
from covid_prevalence.plots import plot_data, plot_fit, plot_IFR, plot_posteriors, plot_prevalence
from covid_prevalence._repository import gitpush

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
repo_path = "/content/covid-prevalence"

if __name__=='__main__':
  q = RedisWQ(name=queuename, host=host)
  log.info("Worker with sessionID: " +  q.sessionID())
  log.info("Initial queue state: empty=" + str(q.empty()))

  # Clone this for the hr_map.csv needed to save csv file
  #  todo - make this more seamless so that the whole repo isn't needed.
  repo_ishaberry = git.Repo.clone_from("https://github.com/ishaberry/Covid19Canada.git", 
        "/content/Covid19Canada", 
        depth=1,
        branch="master")

  # Clone the repository which will recieve the output
  repo_origin_path = "https://oauth2:"+ oauth +"@gitlab.com/stevenhorn/covid-prevalence.git"
  log.info("cloning repository from %s" % repo_origin_path)

  repo = git.Repo.clone_from(repo_origin_path, 
      repo_path,
      depth=1,
      branch="master")

  # Configure git user
  repo.config_writer().set_value("user", "name", "Steven Horn").release()
  repo.config_writer().set_value("user", "email", "steven@horn.work").release()

  # we will work on a different branch
  worker_branch = 'latest-' + datetime.datetime.utcnow().strftime('%y%m%d')
  repo.git.checkout('-b', worker_branch)

  # work on the most recent version of the branch if it exists on origin.
  try:
    res = repo.remotes.origin.pull(worker_branch)
  except git.GitCommandError as e:
    log.error(str(e))

  # It's possible that this is a new branch.  This ensures it on the remote.
  try:
    repo.remotes.origin.push(worker_branch)
  except git.GitCommandError as e:
    log.error(str(e))

  # run the processing loop - if the queue is empty, then this program will end
  timed_out = False
  num_wait_loops = 0
  while not timed_out and not q.empty():

    # This sets up a loop to get data from the queue.  Continues if 
    # 30 seconds has elapsed - logging a waiting message.
    item = q.lease(lease_secs=timelimit, block=True, timeout=60)

    if item is not None:
      num_wait_loops = 0
      itemstr = item.decode("utf-8")

      log.debug("Recieved task " + itemstr)

      task = json.loads(itemstr)

      pop = task['pop']
      log.info("Working on " + pop['name'])

      model = task['model']
      settings = task['settings']

      # First we read the json into a dataframe
      nc_df = pd.read_json(task['new_cases'], orient='table')

      # There should only be one column, so we will have only that data selected as a series type
      new_cases = nc_df[nc_df.columns[0]]

      # Same for cumulative deaths, as for new cases
      cd_df = pd.read_json(task['cum_deaths'], orient='table')
      cum_deaths = cd_df[cd_df.columns[0]]

      bd = isoparse(pop['date_start']) # This is the first day of data
      pa_a = model['pa_a']
      pa_b = model['pa_b']
      pu_a = model['pu_a']
      pu_b = model['pu_b']
      pr_gamma_mu_days = model['gamma_mu_days']
      pr_gamma_mu_sigma = model['gamma_mu_sigma']
      pr_asym_recover_mu_days = model['asym_recover_mu_days']
      pr_asym_recover_mu_sigma = model['asym_recover_mu_sigma']
      pr_sym_recover_mu_days = model['sym_recover_mu_days']
      pr_sym_recover_mu_sigma = model['sym_recover_mu_sigma']

      # Model parameters
      params_model = dict(
        new_cases_obs=new_cases,
        data_begin=bd,
        fcast_len=pop['fcast_len'],             # forecast model
        diff_data_sim=pop['diff_data_sim'],     # number of days for burn-in
        N_population=pop['N'],
      )

      # Set up inferrence of infection rate
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

      change_points = change_points_d2  # dynamic

      with cov19.model.Cov19Model(**params_model) as this_model:
        # apply change points, lambda is in log scale
        lambda_t_log = cov19.model.lambda_t_with_sigmoids(
            pr_median_lambda_0=pop['median_lambda_0'],
            pr_sigma_lambda_0=pop['sigma_lambda_0'],
            change_points_list=change_points,
        )
        pa = pm.Beta(name="pa", alpha=model['pa_a'], beta=model['pa_b'])
        pu = pm.Uniform(name="pu", lower=model['pu_a'], upper=model['pu_b'])
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
          tune=numtune, 
          draws=numsims, 
          n_init=50000,     # we really should have converged by 50k
          init="advi+adapt_diag", cores=cores)

      # TODO: check if advi did not converge (bad model fit)

      if pop['run'] == True:
        log.info('Generating plots')
        plot_fit(this_model, trace, new_cases, pop, settings)
        plot_posteriors(this_model, trace, pop, settings)
        plot_prevalence(this_model, trace, pop, settings)
        plot_IFR(this_model, trace, pop, settings, cum_deaths)
        #dft, dfn = savecsv(this_model, trace, pop)
        log.info('Updating CSV files')
        try:
          _, _ = covprev.data.savecsv(this_model, trace, pop)
        except Exception as e:
          log.error(str(e))

      # We try to push the results to git
      regionid = pop["source_country"] + pop["source_state"] + ("" if pop["source_region"] == None else pop["source_region"])
      regionid = regionid.replace(' ','')  # regionid = 'USColoradoElPaso'

      message = "Updates for " + pop['name']
      try:
        log.info('Local commit prior to pulling')
        repo.git.checkout('-b', worker_branch + '_' + regionid)
        # we need to commit our changes before pulling
        repo.git.add('--all')
        repo.git.commit('-m', message, author='Steven Horn')
        # switch back to branch
        repo.git.checkout(worker_branch)
      except git.GitCommandError as e:
        log.error(str(e))
      
      numpushattempts = 0
      success = False
      while not success and numpushattempts < 6:
        numpushattempts = numpushattempts + 1
        try:
          log.info('Trying to commit.  Attempt %d' % numpushattempts)
          try:
            log.info('Pulling from git prior to pushing')
            # the merge should be seamless
            res = repo.remotes.origin.pull(worker_branch)
          except git.GitCommandError as e:
            log.error(str(e))

          try:
            log.info('Merging...')
            # we should now be on the newest branch - so we merge our result in
            repo.git.merge('-s','recursive','-X','theirs', worker_branch + '_' + regionid)
          except Exception as e:
            log.error(str(e))

          log.info('Pushing branch %s to origin' % worker_branch)
          repo.remotes.origin.push(worker_branch)
          success = True
        except Exception as e:
          log.error("Error pushing. " + str(e))
          # Wait before trying again to avoid hammering git
          time.sleep(30)

          # this probably happened since the remote was updated before we pushed,
          # we will try again.
          # force reset the branch (removing the failed merge result)
          repo.git.checkout('-f', worker_branch)
      
      if not success:
        log.error('Unable to commit %s' % pop['name'])

      # Mark as completed and remove from work queue.
      q.complete(item)
      log.info("Completed " + pop['name'])
    else:
      num_wait_loops = num_wait_loops + 1
      log.info("Waiting for work")

      if num_wait_loops > 20:  # minutes
        # this will break the loop and exit the program
        timed_out = True

      # q.check_expired_leases()

  log.info("Worker queue empty, exiting")
