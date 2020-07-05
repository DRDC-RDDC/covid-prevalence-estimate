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
3 - Send the configuration and data to worker nodes
'''

import logging
import json
import git
import os
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

    repopath = "https://oauth2:"+ oauth +"@gitlab.com/steven.horn/covid-prevalence-estimate.git"

    log.info("cloning program from %s" % repopath)

    repo = git.Repo.clone_from(repopath, 
        "/content/covid-prevalence-estimate",
        depth=1,
        branch="master")

    # This is the worker queue.
    queuename = os.getenv('redis_worker_queue', default='covidprev')
    log.info("Connecting to redis queue (%s)" % queuename)
    #q = rediswq.RedisWQ(name=queuename, host="redis")

    # Load the config file from the repo
    log.info("Loading configuration")
    with open('/content/covid-prevalence-estimate/config/config.json','r') as f:
        config = json.load(f)

    model = config['model']
    settings = config['settings']

    # create worker jobs for each population
    log.info("Processing populations")
    for pop in config['populations']:
        # Fetch Data for processing
        new_cases, cum_deaths, bd = covprev.data.get_data(pop)

        # Structure work item for worker
        work_item = dict(
            model=model,
            settings=settings,
            pop=pop,
            new_cases=new_cases.to_json(orient='table'),        # This serializes the pandas data
            cum_deaths=cum_deaths.to_json(orient='table'),      # 
            submitted=str(datetime.datetime.utcnow()),
            rval = randint(1,1000)
        )
        
        # send the work-item to the worker queue
        log.info("Job Queued: %s" % pop['name'] )
        #q.enqueue(json.dumps(work_item))
        #sleep(randint(10,50))

    log.info("Controller completed")
    sleep(60*60*12)