import logging
import json
import git
import os
import sys
import numpy as np
import pandas as pd
import datetime
import covid_prevalence as covprev
from covid_prevalence.utility import get_folders
from covid_prevalence.rediswq import RedisWQ 

## DEBUGGING IMPORTS
from random import randint
from time import sleep

logging.basicConfig(level=logging.INFO, 
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger(__name__)

opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]

rootpath= "/data/covid-prev"

if __name__=='__main__':
    log.info("Starting controller")

    clone = "-c" in opts
    priority1 = "-P1" in opts
    timefilter = "-tf" in opts
    checkbad = "-b" in opts
    
    if clone:
        log.info("cloning ishaberry/Covid19Canada Data")
        repo_ishaberry = git.Repo.clone_from("https://github.com/ishaberry/Covid19Canada.git", 
            "/content/Covid19Canada", 
            depth=1,
            branch="master")

        log.info("cloning CSSEGISandData/COVID-19 Data")
        repo_jhu = git.Repo.clone_from("https://github.com/CSSEGISandData/COVID-19.git", 
            "/content/COVID-19", 
            depth=1,
            branch="master")

        # If the repository is set to use an oauth token, apply it
        oauth = os.getenv('gitlab_oauth', default='')
        repo_url = os.getenv('gitlab_oauth', default='gitlab.com/stevenhorn/covid-prevalence-estimate.git')
        if oauth == '':
            repopath = "https://" + repo_url
        else:
            log.info('using Oauth token for repo ' + repo_url)
            repopath = "https://oauth2:"+ oauth + "@" + repo_url

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
    with open(rootpath + '/covid-prevalence-estimate/config/config.json','r') as f:
        config = json.load(f)

    model = config['settings']['model']
    settings = config['settings']

    pops = config['populations']

    reqkey = (state, region, country)

    for pop in pops:
        # This is how we handle wildcards 
        # so you can call loadregion -P1 Canada * *    which loads all CAN
        ss = pop['source_state']
        if state == '*':
            ss = '*'
        if state == 'None':
            ss = 'None'
        sr = pop['source_region']
        if region == '*':
            sr = '*'
        if region == 'None':
            sr = 'None'
        sc = pop['source_country']
        if country == '*':
            sc = '*'

        popkey = (ss, sr, sc)
        if popkey != reqkey:
            continue
        
        if checkbad:
            _, folder = get_folders(pop, rootpath)
            try:
                folderpath = rootpath + '/covid-prevalence/results/latest/' + folder
                stats_file = folderpath + "/" + folder + "_stats.txt"
                if os.path.exists(stats_file):
                    with open(stats_file) as f:
                        divs = f.readline()
                        divergences = int(divs.split(' ')[0])
                        validation = f.readline()
                        if validation.startswith("Passed"):
                            continue
            except:
                log.error('unable to check for bad run' + str(e))
                continue

        if timefilter:
            # Check when it was last run
            _, folder = get_folders(pop, rootpath)
            try:
                savefolder = rootpath + '/covid-prevalence/results/latest/' + folder
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
        try:
            new_cases, cum_deaths, bd = covprev.data.get_data(pop, rootpath=rootpath)
        except Exception as e:
                log.error('Error fetching region data for ' + pop['name'])
                log.error(str(e))
                continue

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

