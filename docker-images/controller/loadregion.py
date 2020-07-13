

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

import sys

logging.basicConfig(level=logging.INFO, 
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger(__name__)

opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]

if __name__=='__main__':
    log.info("Starting controller")

    clone = "-c" in opts

    priority1 = "-P1" in opts
    
    if clone:
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

    country = args[0]
    state = args[1]
    region = args[2]

    if region == 'None':
        region = None
    
    queuename = os.getenv('redis_worker_queue', default='covidprev')

    if priority1:
        queuename = queuename + 'p1'

    log.info("Connecting to redis queue (%s)" % queuename)
    q = RedisWQ(name=queuename, host="redis")

    # Load the config file from the repo
    log.info("Loading configuration")
    with open('/content/covid-prevalence-estimate/config/config.json','r') as f:
        config = json.load(f)

    model = config['model']
    settings = config['settings']

    pops = config['populations']
    reqkey = (state, region, country)

    for pop in pops:

        # This is how we handle wildcards 
        # so you can call loadregion -P1 Canada * *    which loads all CAN
        ss = pop['source_state']
        if state == '*':
            ss = '*'
        sr = pop['source_region']
        if region == '*':
            sr = '*'
        sc = pop['source_country']
        if country == '*':
            sc = '*'

        popkey = (ss, sr, sc)
        if popkey != reqkey:
            continue

        # Fetch Data for processing
        log.debug('Fetching region data.')
        new_cases, cum_deaths, bd = covprev.data.get_data(pop)
        # fix up negative case values - reduces model quality
        new_cases[new_cases < 0] = 0
        log.info("Job Queuing: %s" % pop['name'])

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

