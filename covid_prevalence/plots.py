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

import covid19_inference as cov19
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from . import utility as ut
import logging

log = logging.getLogger(__name__)

def plot_data(this_model, new_cases, pop, settings, closeplot=True, rootpath='/content'):
  ''' Plots just the case data
  '''
  log.info('Plotting Data')
  savefolder, folder = ut.get_folders(pop, rootpath=rootpath)
  x_data = pd.date_range(start=this_model.data_begin, 
                         end=this_model.data_begin + datetime.timedelta(days=new_cases.shape[0]-1) )

  fig, ax1 = plt.subplots()
  plt.ylabel("Number of new cases reported")
  ax1.plot(x_data[:len(new_cases.values)], new_cases.values, 'b--', label="new cases data")
  plt.legend()
  plt.title(f"Model comparison to data {pop['name']}")
  plt.xticks(rotation=45)
  plt.xlabel("Day")

  if settings is not None and 'ShowPreliminary' in settings and settings['ShowPreliminary']:
    fig.text(0.75, 0.25, 'PRELIMINARY',
            fontsize=30, color='gray',
            ha='right', va='bottom', alpha=0.5, rotation='30')

  plt.tight_layout()
  plt.savefig(savefolder + '/'+folder+'_fit.png')
  if closeplot:
    plt.close()

def plot_fit(this_model, trace, new_cases, pop, settings, closeplot=True,rootpath='/content'):
  ''' Plots the data and the model trace fits
  '''
  log.info('Plotting Fit')
  N = this_model.N_population

  popname = pop['name']
  savefolder, folder = ut.get_folders(pop, rootpath=rootpath)
    # inspect the chain realizations
  trace.varnames
  I_t = trace["new_detections"][:, None]
  I_t = trace["new_cases"][:, None]
  numpts = I_t.shape[0]

  I_t_05 = []
  I_t_50 = []
  I_t_95 = []
  tx = np.arange(0,I_t.shape[2])
  for t in tx:
    a,b,c = np.percentile(I_t[:,0,t],[2.5,50,97.5])
    I_t_05.append(a)
    I_t_50.append(b)
    I_t_95.append(c)

  x_data_in = pd.date_range(start=this_model.data_begin, end=this_model.data_end)
  x_sim = pd.date_range(start=this_model.sim_begin, end=this_model.sim_end )

  fig, ax1 = plt.subplots()
  for i in range(numpts):
    ax1.plot(x_sim,I_t[i,:][0],alpha=0.3)
    pass

  plt.ylabel("Number of new cases reported")
  ax1.plot(x_data_in, new_cases.values, 'b--', label="new cases data")

  ax1.plot(x_sim,I_t_50, label="new cases model")
  plt.legend()
  plt.title("Model comparison to data \n %s, pop. = %s" % (popname, N))
  plt.xticks(rotation=45)
  plt.xlabel("Day")

  if settings is not None and 'ShowPreliminary' in settings and settings['ShowPreliminary']:
    fig.text(0.75, 0.25, 'PRELIMINARY',
            fontsize=30, color='gray',
            ha='right', va='bottom', alpha=0.5, rotation='30')

  plt.tight_layout()
  savepath = savefolder + '/'+folder+'_fit.png'
  plt.savefig(savepath)
  if closeplot:
    plt.close()

  log.info('Fit plot saved to %s' % savepath)

def plot_posteriors(this_model, trace, pop, settings,rootpath='/content'):
  ''' Plot the posterior spreading rate
  '''
  N = this_model.N_population
  savefolder, folder = ut.get_folders(pop, rootpath=rootpath)

  lambda_t, x = cov19.plot._get_array_from_trace_via_date(this_model, trace, "lambda_t")

  y = lambda_t[:, :]

  L_t_025, L_t_50, L_t_975 = ut.get_percentile_timeseries(y,islambda=True)

  plt.figure()
  plt.plot(x,L_t_50, label="lambda")
  plt.fill_between(x,L_t_025,L_t_975,lw=0,alpha=0.1, label="95CI")

  plt.xticks(rotation=45)
  plt.xlabel("Day")
  plt.title("Spreading rate ($\lambda$) \n %s, pop. = %s" % (pop['name'], N))

  plt.tight_layout()
  plt.savefig(savefolder + r'/' + folder + '_lambda.png')
  plt.close()

def plot_introduction(this_model, trace, pop, settings, closeplot=True):
  '''
  '''
  ShowPreliminary = settings['ShowPreliminary']
  savefolder, folder = ut.get_folders(pop)
  Ein_t, x = cov19.plot._get_array_from_trace_via_date(this_model, trace, "Ein_t")
  y = Ein_t[:, :]

  l_t_05 = []
  l_t_50 = []
  l_t_95 = []
  tx = np.arange(0,y.shape[1])
  for t in tx:
    a,b,c = np.percentile(y[:,t],[2.5,50,97.5])
    l_t_05.append(a)
    l_t_50.append(b)
    l_t_95.append(c)

  plt.figure()
  plt.plot(x,l_t_50, label="Introduced")
  plt.fill_between(x,l_t_05,l_t_95,lw=0,alpha=0.1, label="95CI")

  plt.xticks(rotation=45)
  plt.xlabel("Day")
  plt.title("Imported infections ($E_{in}$)")

  plt.tight_layout()
  plt.savefig(savefolder + '/'+folder+'_ein.png')
  if closeplot:
    plt.close()


# This code needs major cleanup
def plot_prevalence(this_model, trace, pop, settings, closeplot=True, rootpath='/content'): #closeplot=False
  # this is the infected asymptomatic AND symptimatic
  ShowPreliminary = settings['ShowPreliminary']
  popname = pop['name']
  savefolder, folder = ut.get_folders(pop, rootpath=rootpath)
  trimend = -1#-25
  trimstart=0

  N = this_model.N_population
  E_t = trace["E_t"][:, None]
  R_t = trace["R_t"][:, None]
  I_t = trace["I_t"][:, None]
  new_E_t= trace["new_E_t"][:, None] 
  lambda_t, _ = cov19.plot._get_array_from_trace_via_date(this_model, trace, "lambda_t")
  p0 = 100.0*I_t/N * lambda_t
  Ip_t = I_t + E_t + R_t

  # we check for degenerate/oscillating solutions if nne < 0 in the trace
  nne = new_E_t/E_t
  degens = nne < 0
  degen = np.array([np.sum(d[0])>0 for d in degens])

  if np.sum(degen) == len(degen):
    log.warn("All traces degenerate")
    degens = np.ones(len(degen)) == 0 # all false

  I_t_025 = []
  I_t_50 = []
  I_t_975 = []
  tx = np.arange(0,I_t.shape[2])

  for t in tx:
    a,b,c = np.percentile(100*I_t[:,0,t][degen==False]/N,[2.5,50,97.5])
    if a < 0:
      a = 0
    if b < 0:
      b = 0
    if c < 0:
      c = 0
    I_t_025.append(a)
    I_t_50.append(b)
    I_t_975.append(c)

  Ip_t_05 = []
  Ip_t_50 = []
  Ip_t_95 = []
  tx = np.arange(0,Ip_t.shape[2])

  for t in tx:
    a,b,c = np.percentile(100*Ip_t[:,0,t][degen==False]/N,[2.5,50,97.5])
    if a < 0:
      a = 0
    if b < 0:
      b = 0
    if c < 0:
      c = 0
    Ip_t_05.append(a)
    Ip_t_50.append(b)
    Ip_t_95.append(c)

  p_t_05 = []
  p_t_50 = []
  p_t_95 = []
  tx = np.arange(0,p0.shape[2])
  N = this_model.N_population
  for t in tx:
    a,b,c = np.percentile(p0[:,0,t][degen==False],[2.5,50,97.5])
    if a < 0:
      a = 0
    if b < 0:
      b = 0
    if c < 0:
      c = 0
    p_t_05.append(a)
    p_t_50.append(b)
    p_t_95.append(c)

  Ia_t = trace["Ia_t"][:, None]
  Ia_t_05 = []
  Ia_t_50 = []
  Ia_t_95 = []
  tx = np.arange(0,Ia_t.shape[2])
  for t in tx:
    #a,b,c = np.percentile(100*Ia_t[:,0,t]/N,[5,50,95])
    a,b,c = np.percentile(Ia_t[:,0,t][degen==False],[2.5,50,97.5])
    if a < 0:
      a = 0
    if b < 0:
      b = 0
    if c < 0:
      c = 0
    Ia_t_05.append(a)
    Ia_t_50.append(b)
    Ia_t_95.append(c)

  Is_t = trace["Is_t"][:, None]
  Is_t_05 = []
  Is_t_50 = []
  Is_t_95 = []
  tx = np.arange(0,Is_t.shape[2])
  for t in tx:
    #a,b,c = np.percentile(100*Is_t[:,0,t]/N,[5,50,95])
    a,b,c = np.percentile(Is_t[:,0,t][degen==False],[2.5,50,97.5])
    if a < 0:
      a = 0
    if b < 0:
      b = 0
    if c < 0:
      c = 0
    Is_t_05.append(a)
    Is_t_50.append(b)
    Is_t_95.append(c)

  x_sim = pd.date_range(start=this_model.sim_begin, end=this_model.sim_end )

  fig, ax1 = plt.subplots()
  plt.plot(x_sim,Ip_t_50, label="Prevalence")
  plt.fill_between(x_sim,Ip_t_05,Ip_t_95,lw=0,alpha=0.1, label="95CI")
  maxy = np.max(Ip_t_95)
  if maxy < 0.01:
    plt.ylim(0,0.01)
    maxy = 0.01

  plt.plot([datetime.datetime.today(), datetime.datetime.today()], [0,maxy],'r-', label="Today")
  plt.ylabel("%")
  plt.xticks(rotation=45)
  plt.title("Prevalence of COVID-19 \n %s, pop. = %s" % (popname, N))
  plt.xlabel("Date")
  start, end = ax1.get_ylim()
  locs, _ = plt.yticks()

  ax2 = ax1.twinx()
  ax2.set_ylim(start,end)
  labs = ["%d" % (N*l/100.0) for l in locs]
  ax2.set_yticklabels(labs)
  ax2.grid(False)
  plt.ylabel("# of infections")
  ax1.legend()

  if ShowPreliminary:
    fig.text(0.75, 0.25, 'PRELIMINARY',
            fontsize=30, color='gray',
            ha='right', va='bottom', alpha=0.5, rotation='30')

  plt.tight_layout()
  plt.savefig(savefolder + '/'+folder+'_prev.png')
  if closeplot:
    plt.close()

  '''
  fig, ax1 = plt.subplots()

  # Cap the upper bound to 100%
  I_t_50 = np.array(I_t_50)
  I_t_50[I_t_50 > 100] = 100

  plt.plot(x_sim,I_t_50, label="Prevalence")

  # Cap the upper bound to 100%
  I_t_975 = np.array(I_t_975)
  I_t_975[I_t_975 > 100] = 100

  plt.fill_between(x_sim,I_t_025,I_t_975,lw=0,alpha=0.1, label="95CI")
  plt.title("Infectious prevalence ($I$)\n %s, pop. = %s" % (popname, N))
  plt.ylabel("%")
  plt.xlabel("Date")
  plt.xticks(rotation=45)
  start, end = ax1.get_ylim()
  locs, _ = plt.yticks()
  ax2 = ax1.twinx()
  ax2.set_ylim(start,end)
  labs = ["%d" % (N*l/100.0) for l in locs]
  ax2.set_yticklabels(labs)
  ax2.grid(False)
  plt.ylabel("# of infections")
  ax1.legend()

  maxy = np.max(I_t_975)
  plt.plot([datetime.datetime.today(), datetime.datetime.today()], [0,maxy],'r-', label="Today")

  if ShowPreliminary:
    fig.text(0.75, 0.25, 'PRELIMINARY',
            fontsize=30, color='gray',
            ha='right', va='bottom', alpha=0.5, rotation='30')

  plt.tight_layout()
  plt.savefig(savefolder + '/'+folder+'_prev_i.png')
  if closeplot:
    plt.close()

  fig, ax1 = plt.subplots()
  plt.plot(x_sim,p_t_50, label="Prevalence")
  plt.fill_between(x_sim,p_t_05,p_t_95,lw=0,alpha=0.1, label="95CI")
  plt.title("Infectious prevalence times probability of infecting ($p_0$)\n %s, pop. = %s" % (popname, N))
  plt.ylabel("$p_0$ (%)")
  plt.xlabel("Date")
  plt.xticks(rotation=45)

  maxy = np.max(p_t_95)
  if maxy < 0.01:
    plt.ylim(0,0.01)
    maxy = 0.01
  plt.plot([datetime.datetime.today(), datetime.datetime.today()], [0,maxy],'r-', label="Today")

  if ShowPreliminary:
    fig.text(0.75, 0.25, 'PRELIMINARY',
            fontsize=30, color='gray',
            ha='right', va='bottom', alpha=0.5, rotation='30')

  plt.tight_layout()
  plt.savefig(savefolder + '/'+folder+'_p0.png')
  if closeplot:
    plt.close()
'''
  # Sym and asym
  ''' Removed this plot for now
  fig, ax1 = plt.subplots()

  plt.plot(x_sim,Ia_t_50, label="Asymptomatic or undetected")
  plt.fill_between(x_sim,Ia_t_05,Ia_t_95,lw=0,alpha=0.1, label="95CI")
  plt.plot(x_sim,Is_t_50, label="Presymptomatic")
  plt.fill_between(x_sim,Is_t_05,Is_t_95,lw=0,alpha=0.1, label="95CI")
  plt.ylabel("# of infections")
  plt.xticks(rotation=45)
  plt.title("Prevalence of COVID-19 \n %s, pop. = %s" % (popname, N))
  plt.legend()
  plt.xlabel("Date")

  if ShowPreliminary:
    fig.text(0.75, 0.25, 'PRELIMINARY',
            fontsize=30, color='gray',
            ha='right', va='bottom', alpha=0.5, rotation='30')

  plt.tight_layout()
  plt.savefig(savefolder + '/'+folder+'_a_s.png')
  if closeplot:
    plt.close()
  '''

  I_t_025 = []
  I_t_50 = []
  I_t_975 = []
  trimend = -1
  trimstart = -45#60
  tx = np.arange(0,I_t.shape[2])
  #N = 36e6
  for t in tx:
    a,b,c = np.percentile(I_t[:,0,t],[2.5,50,97.5])
    if a < 0:
      a = 0
    if b < 0:
      b = 0
    if c < 0:
      c = 0
    I_t_025.append(a)
    I_t_50.append(b)
    I_t_975.append(c)

  fig = plt.figure()
  plt.plot(x_sim[trimstart:trimend],I_t_50[trimstart:trimend], label="Prevalence")
  plt.fill_between(x_sim[trimstart:trimend],I_t_025[trimstart:trimend],I_t_975[trimstart:trimend],lw=0,alpha=0.1, label="95CI")
  maxy = np.max(I_t_975[trimstart:trimend])
  if maxy < 1:
    plt.ylim(0,1)
    maxy = 1
  plt.plot([datetime.datetime.today(), datetime.datetime.today()], [0,maxy],'r-', label="Today")
  #plt.yscale('log')
  #plt.ylim((0,10))
  plt.xticks(rotation=45)
  plt.title("Prevalence of COVID-19 \n %s, pop. = %s" % (popname, N))
  plt.legend()
  plt.xlabel("Date")
  plt.ylabel("# undetected infections")

  if ShowPreliminary:
    fig.text(0.75, 0.25, 'PRELIMINARY',
            fontsize=30, color='gray',
            ha='right', va='bottom', alpha=0.5, rotation='30')

  plt.tight_layout()
  plt.savefig(savefolder + '/'+folder+'_predict.png')
  if closeplot:
    plt.close()

def plot_IFR(this_model, trace, pop, settings, cum_deaths, rootpath='/content'):
  # this is the infected asymptomatic AND symptimatic
  ShowPreliminary = settings['ShowPreliminary']
  popname = pop['name']
  savefolder, folder = ut.get_folders(pop, rootpath=rootpath)
  # IFR estimate
  death_t = np.array(cum_deaths)
  #print(death_t)
  trimend = -1  
  shift = 0
  Ecum_t = trace["Ecum_t"][:, None]

  #print(cum_deaths.index[0])
  #print(cum_deaths.index.shape)

  deathdelay = 0#13   # https://www.medrxiv.org/content/10.1101/2020.03.05.20031773v2.full.pdf

  Ecum_t = Ecum_t[:,:,deathdelay:cum_deaths.index.shape[0]+deathdelay]

  Ecum_t_05 = []
  Ecum_t_50 = []
  Ecum_t_95 = []
  tx = np.arange(0,Ecum_t.shape[2])
  for t in tx:
    a,b,c = np.percentile(100*death_t[t]/Ecum_t[:,0,t],[2.5,50,97.5])
    if a < 0:
      a = 0
    Ecum_t_05.append(a)
    Ecum_t_50.append(b)
    Ecum_t_95.append(c)

  x_data2 = pd.date_range(start=this_model.data_begin, end=this_model.data_begin + datetime.timedelta(days=Ecum_t.shape[2]-1) )
  #print(Ecum_t.shape[2]-1)
  fig, ax1 = plt.subplots()
  trimstart = 14
  plt.plot(x_data2[trimstart:trimend-shift],Ecum_t_50[trimstart:trimend], label="Infected")
  plt.fill_between(x_data2[trimstart:trimend-shift],Ecum_t_05[trimstart:trimend],Ecum_t_95[trimstart:trimend],lw=0,alpha=0.1, label="95CI")

  x_data3 = pd.date_range(start=cum_deaths.index[0],end=cum_deaths.index[-1])

  plt.title("Running Estimate of IFR: %s" % popname)
  plt.xlabel("Date")
  plt.xticks(rotation=45)
  plt.ylabel("IFR (%)")

  if ShowPreliminary:
    fig.text(0.75, 0.25, 'PRELIMINARY',
            fontsize=30, color='gray',
            ha='right', va='bottom', alpha=0.5, rotation='30')


  plt.tight_layout()
  plt.savefig(savefolder + '/'+folder+'_running_IFR.png')
  plt.close()

  plt.figure()
  values = 100*death_t[-1]/Ecum_t[:,0,-1]
  #print (values)
  sns.distplot(values, hist = False, kde = True,
                bins=30,
                kde_kws = {'shade': True, 'linewidth': 1,'cumulative': False},
                label = 'IFR')#.set(xlim=(0,1))
  plt.title("Estimated IFR")
  plt.xlabel("%")
  plt.ylabel("prob. density")

  plt.tight_layout()
  plt.savefig(savefolder + '/'+folder+'_IFR.png')

  plt.close()