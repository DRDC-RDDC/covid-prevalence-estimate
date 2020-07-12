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

''' How this program works

1 - Download the data from most recent repository
2 - Check what areas need to be processed
3 - Send the configuration and data to worker nodes via a redis message queue
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

if __name__=='__main__':
    log.info("Starting controller")

    log.info("cloning ishaberry/Covid19Canada Data")
    # ! git clone https://github.com/ishaberry/Covid19Canada.git --depth 1 --branch master --single-branch /content/Covid19Canada
    repo_ishaberry = git.Repo.clone_from("https://github.com/ishaberry/Covid19Canada.git", 
        "/content/Covid19Canada", 
        depth=1,
        branch="master")

    log.info("cloning CSSEGISandData/COVID-19 Data")
    #! git clone https://github.com/CSSEGISandData/COVID-19.git --depth 1 --branch master --single-branch /content/COVID-19
    repo_jhu = git.Repo.clone_from("https://github.com/CSSEGISandData/COVID-19.git", 
        "/content/COVID-19", 
        depth=1,
        branch="master")

    oauth = os.getenv('gitlab_oauth', default='53gZwasv6UmExxrohqPm')

    repopath = "https://oauth2:"+ oauth +"@gitlab.com/stevenhorn/covid-prevalence-estimate.git"

    log.info("cloning program from %s" % repopath)

    repo = git.Repo.clone_from(repopath, 
        "/content/covid-prevalence-estimate",
        depth=1,
        branch="master")

    # This is the worker queue.
    queuename = os.getenv('redis_worker_queue', default='covidprev')
    log.info("Connecting to redis queue (%s)" % queuename)
    q = RedisWQ(name=queuename, host="redis")

    # Load the config file from the repo
    log.info("Loading configuration")
    with open('/content/covid-prevalence-estimate/config/config.json','r') as f:
        config = json.load(f)

    model = config['model']
    settings = config['settings']

    try:
        dataurl = "https://stevenhorn.gitlab.io/covid-prevalence/results/latest_results.csv"
        df = pd.read_csv(dataurl, parse_dates=['analysisTime'])
    except:
        # results file not there - create dummy
        df = None

    # create worker jobs for each population
    log.info("Processing populations")
    for pop in config['populations']: # pop = config['populations'][1]
        if pop['run'] == True:
            # Check when it was last run
            try:
                folder = pop["source_country"] + pop["source_state"] + ("" if pop["source_region"] == None else pop["source_region"])
                folder = folder.replace(' ','')  # folder = 'USColoradoElPaso'
                lastrun = df[df['nameid'] == folder]['analysisTime']
                dt = datetime.datetime.utcnow() - lastrun
                dt_hours = dt.to_list()[0].total_seconds()/60/60
            except Exception as e:
                log.error('error checking last run time, assume 96. ' + str(e))
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
            new_cases, cum_deaths, bd = covprev.data.get_data(pop)
            
            # fix up negative case values - reduces model quality
            new_cases[new_cases < 0] = 0

            if np.sum(new_cases > 0) == 0:
                # no cases in this region
                log.info("Job Skipped: %s, no cases" % pop['name'])
                continue

            # filter for frequency by level of activity in region
            cases_prev10days = np.sum(new_cases[-10:])
            if cases_prev10days == 0 and dt_hours > 24*5:
                log.info("Job Skipped: %s, no cases past 10 days and 5 days not passed" % pop['name'])
                continue

            if cases_prev10days > 0 and cases_prev10days < 10 and dt_hours > 24*3:
                log.info("Job Skipped: %s, < 10 cases past 10 days and 3 days not passed" % pop['name'])
                continue

            if cases_prev10days >= 10 and cases_prev10days < 30 and dt_hours > 24*2:
                log.info("Job Skipped: %s, 10-30 cases past 10 days and 2 days not passed" % pop['name'])
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

    #sleep(60*60*12)