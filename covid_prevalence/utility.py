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
'''

import datetime
import numpy as np
from pathlib import Path

def converters(o):
    if isinstance(o, datetime.datetime):
        return o.isoformat()

def get_folders(pop, rootpath='/content'):
  '''
      Returns: savefolder, nameid
  '''
  folder = pop["source_country"] + ("" if pop["source_state"] == None else pop["source_state"]) + ("" if pop["source_region"] == None else pop["source_region"])
  folder = folder.replace(' ','')
  folder = folder.replace('*','')
  savefolder = rootpath + '/covid-prevalence/results/latest/' + folder
  Path(savefolder).mkdir(parents=True, exist_ok=True)
  return savefolder,folder

def get_percentile_timeseries(X_t, islambda=False, degen=None):
  ''' Get the 2.5%, 50%, and 97.5% percentiles from the time series.

  TODO: add other percentiles: c(0.01, 0.025, seq(0.05, 0.95, by = 0.05), 0.975, 0.99)

  Parameters
  ----------
  X_t : :array: float
      This is the time series data which is X by Y dimensional.  The
      dimensions are time and samples.

  islambda : bool
      If true, this is for the spreading rate (lambda)

  degen : :array: bool
      If false, these runs are masked out.  Degenerate cases are when
      the solution is valid for the model as defined, but not feasible.
  '''
  X_t_025 = []
  X_t_50 = []
  X_t_975 = []
  ix = 2
  if islambda:
    ix = 1
  tx = np.arange(0,X_t.shape[ix])

  # Handle degenerate filter, if none, consider all as non-degenerate
  if degen is None:
    degen = np.ones(len(X_t[:,0])) > 0

  for t in tx:
    if islambda:
      a,b,c = np.percentile(X_t[:,t][degen==False],[2.5,50,97.5])
    else:
      a,b,c = np.percentile(X_t[:,0,t][degen==False],[2.5,50,97.5])
    X_t_025.append(a)
    X_t_50.append(b)
    X_t_975.append(c)
  
  return np.array(X_t_025), np.array(X_t_50), np.array(X_t_975)