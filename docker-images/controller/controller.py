''' Controller pod for loading analysis jobs

How this program works

1 - Download the data from most recent repository
2 - Check what areas need to be processed
3 - Send the configuration and data to worker nodes via a redis message queue


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
import json
import git
import os
import numpy as np
import pandas as pd
import datetime
import covid_prevalence as covprev
from covid_prevalence.rediswq import RedisWQ 

## DEBUGGING IMPORTS
from random import randint
from time import sleep
##

logging.basicConfig(level=logging.INFO, 
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger(__name__)

can_data_path = os.getenv("CAN_DATA_PATH", default="/data/Covid19Canada")
jhu_data_path = os.getenv("JHU_DATA_PATH", default="/data/COVID-19")
cov_repo_path = os.getenv("COV_REPO_PATH", default="/data/covid-prevalence-estimate")
result_repo_path = os.getenv("RESULT_REPO_PATH", default="/data/covid-prevalence")
redis_host = os.getenv("REDIS_SERVICE_HOST", default="redis")
oauth = os.getenv('GITLAB_OAUTH', default='53gZwasv6UmExxrohqPm')

if __name__=='__main__':
    log.info("Started controller")

    log.info('pulling latest Canadian data...')
    repo_ishaberry = git.Git(can_data_path)
    repo_ishaberry.pull('origin','master')
    log.info('success')

    log.info('pulling latest USA data...')
    repo_jhu = git.Git(jhu_data_path)
    repo_jhu.pull('origin','master')
    log.info('success')

    log.info('pulling latest configuration data...')
    repo_covprev = git.Git(cov_repo_path)
    repo_covprev.pull('origin','master')
    log.info('success')

    log.info('pulling latest covid-estimate repo...')
    repo_out = git.Git(result_repo_path)
    repo_out.fetch('--all')
    repo_out.fetch('--prune') # remove branches that don't exist on the remotes
    repo_out.checkout('master')
    repo_out.pull('origin','master')
    repo_out = git.Repo(result_repo_path)
    repo_out.config_writer().set_value("user", "name", "Steven Horn").release()
    repo_out.config_writer().set_value("user", "email", "steven@horn.work").release()
    # we should now be set up with our repo in the right state to start work
    log.info('success')

    log.info('try to checkout worker branch')
    worker_branch = 'latest-' + datetime.datetime.utcnow().strftime('%y%m%d')
    repo_branches = repo_out.branches
    repo_branch_names = [h.name for h in repo_branches]
    if worker_branch in repo_branch_names:
        log.info("branch exists - switching")
        try:
            repo_out.git.checkout(worker_branch)
            log.info('success')
        except Exception as e:
            log.error(str(e))
    else:
        log.info("branch doesn't exist - creating")
        try:
            repo_out.git.checkout('-b', worker_branch)
            log.info('success')
        except Exception as e:
            log.error(str(e))
            # this is a fatal error
            sleep(60*5) # a timeout to give time to see logs before k8s eats it
            os._exit(1)
    
    # debug
    #sleep(60*60*12)
    #os._exit(0)

    # This is the worker queue.
    queuename = os.getenv('redis_worker_queue', default='covidprev')
    log.info("Connecting to redis queue (%s)" % queuename)
    q = RedisWQ(name=queuename, host=redis_host)

    # Load the config file from the repo
    log.info("Loading configuration")
    with open('/data/covid-prevalence-estimate/config/config.json','r') as f:
        config = json.load(f)

    model = config['model']
    settings = config['settings']

    # create worker jobs for each population
    log.info("Processing populations")
    for pop in config['populations']: # pop = config['populations'][3000]
        if pop['run'] == True:
            # Check when it was last run
            folder = pop["source_country"] + pop["source_state"] + ("" if pop["source_region"] == None else pop["source_region"])
            folder = folder.replace(' ','')  # folder = 'USMichiganMidland'
            try:
                savefolder = '/data/covid-prevalence/results/latest/' + folder
                rfilepath = savefolder + '/' + folder + '_latest.csv'
                dfr = pd.read_csv(rfilepath, parse_dates=['analysisTime'])
                lastrun = dfr[dfr['nameid'] == folder]['analysisTime']
                dt = datetime.datetime.utcnow() - lastrun
                dt_hours = dt.to_list()[0].total_seconds()/60/60
            except Exception as e:
                log.error('error checking last run time, assume 200 hours. ' + str(e))
                dt_hours = 200

            # This is the frequency with which to run the model for this region
            max_frequency = config['settings']['frequency']
            
            if 'frequency' in pop:
                # This region has an override on the frequency
                log.info("Frequency override: %d hours" % pop['frequency'] )
                max_frequency = pop['frequency']

            if dt_hours < max_frequency:
                log.info("Job Skipped: %s, last run %d hours ago" % (pop['name'], dt_hours ))
                continue

            # Fetch Data for processing
            log.debug('Fetching region data.')
            new_cases, cum_deaths, bd = covprev.data.get_data(pop, rootpath = '/data')
            
            # fix up negative case values - reduces model quality
            new_cases[new_cases < 0] = 0

            if np.sum(new_cases > 0) == 0:
                # no cases in this region
                log.info("Job Skipped: %s, no cases" % pop['name'])
                continue

            cases_per_day = np.sum(new_cases)/len(new_cases)

            if cases_per_day < 0.2:
                log.info("Job Skipped: %s, few cases reported" % pop['name'])
                continue

            # filter for frequency by level of activity in region
            cases_prev10days = np.sum(new_cases[-10:])
            if cases_prev10days == 0 and dt_hours < 24*5:
                log.info("Job Skipped: %s, no cases past 10 days and 5 days not passed" % pop['name'])
                continue

            if cases_prev10days > 0 and cases_prev10days < 10 and dt_hours < 24*3:
                log.info("Job Skipped: %s, < 10 cases past 10 days and 3 days not passed" % pop['name'])
                continue

            if cases_prev10days >= 10 and cases_prev10days < 30 and dt_hours < 24*2:
                log.info("Job Skipped: %s, 10-30 cases past 10 days and 2 days not passed" % pop['name'])
                continue

            # don't run more frequently than this
            if dt_hours < 24*2:
                log.info("Job Skipped: %s, 2 days not passed" % pop['name'])
                continue
            # > 30 past 10 days, run daily

            log.info("Job Queuing: %s, last run %d hours ago" % (pop['name'], dt_hours ))

            # Structure work item for worker
            work_item = dict(
                model=model,
                settings=settings,
                pop=pop,
                new_cases=new_cases.to_json(orient='table'),        # This serializes the pandas data
                cum_deaths=cum_deaths.to_json(orient='table'),      # 
                submitted=str(datetime.datetime.utcnow())
            )
            
            # send the work-item to the worker queue
            try:
                q.enqueue(json.dumps(work_item).encode('utf-8'))
                log.info("Job Queued: %s" % pop['name'] )
            except Exception as e:
                log.info(str(work_item))
                log.error(str(e))
        else:
            log.info("Job Disabled: %s" % pop['name'] )

    log.info("Controller completed")

    # The program will now end.  
    # The pod running it will close if restart=Never

    # Keep the pod active for the next 12 hours.  This is useful for debugging by
    # connecting into the pod shell
    sleep(60*60*12)