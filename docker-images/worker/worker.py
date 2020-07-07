import time
import logging
import json
import git
from covid_prevalence.rediswq import RedisWQ 
import os
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
log = logging.getLogger(__name__)

# These are (optionally) configurable values via environment variables
host = os.getenv("REDIS_SERVICE_HOST", default="redis")
queuename = os.getenv('redis_worker_queue', default='covidprev')
oauth = os.getenv('gitlab_oauth', default='53gZwasv6UmExxrohqPm')
timelimit = 60*60*1 # these hours should be enough per region - make env?

if __name__=='__main__':
  q = RedisWQ(name=queuename, host=host)
  log.info("Worker with sessionID: " +  q.sessionID())
  log.info("Initial queue state: empty=" + str(q.empty()))

  # Clone the repository for output


  # run the processing loop - if the queue is empty, then this program will end
  while not q.empty():

    # This sets up a loop to get data from the queue.  Continues if 
    # 30 seconds has elapsed - logging a waiting message.
    item = q.lease(lease_secs=timelimit, block=True, timeout=30)

    if item is not None:
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

      bd = isoparse(pop['date_start'])

      time.sleep(50) # Put your actual work here instead of sleep.

      # Mark as completed and remove from work queue.
      q.complete(item)
      log.info("Completed " + pop['name'])
    else:
      log.info("Waiting for work")
      # q.check_expired_leases()

  log.info("Worker queue empty, exiting")
