import datetime
import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as tt
import pandas as pd
import covid19_inference as cov19
import logging
import seaborn as sns
import json
import git
import os
from scipy.stats import beta, lognorm, norm, uniform
from covid19_inference.model import *
from datetime import timezone
from dateutil.parser import parse, isoparse
from pathlib import Path

# this is used for json serialization
def converters(o):
    if isinstance(o, datetime.datetime):
        return o.isoformat()

plt.style.use('seaborn-darkgrid')

if __name__=='__main__':
    print("Starting script")

    oauth = os.getenv('gitlab_oauth', default='53gZwasv6UmExxrohqPm')

    repopath = "https://oauth2:"+ oauth +"@gitlab.com/steven.horn/covid-prevalence.git"

    print("cloning program from %s" % repopath)

    repo = git.Repo.clone_from(repopath, 
        "/content/covid-prevalence",
        branch="master")

    print("Completed script")


'''
# for saving results to git
# Configure git
! git config --global user.email "steven@horn.work"
! git config --global user.name "Steven Horn"
! git clone https://gitlab.com/steven.horn/covid-prevalence.git
% cd /content/covid-prevalence

# we use an oauth token (with limited privilages) to access the repo
! git remote add gitlab https://oauth2:53gZwasv6UmExxrohqPm@gitlab.com/steven.horn/covid-prevalence.git

! git checkout -b latest
! git checkout latest
! git pull gitlab latest

# go back to normal colab root
% cd /content
'''