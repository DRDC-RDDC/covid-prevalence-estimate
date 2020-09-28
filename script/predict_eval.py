
import numpy as np
import pandas as pd
import metriculous
import datetime

import matplotlib.pyplot as plt

plt.style.use('seaborn-darkgrid')
'''
# Mock the ground truth, a one-dimensional array of floats
ground_truth = np.random.random(300)

# Mock the output of a few models
perfect_model = ground_truth
noisy_model = ground_truth + 0.1 * np.random.randn(*ground_truth.shape)
random_model = np.random.randn(*ground_truth.shape)
zero_model = np.zeros_like(ground_truth)

metriculous.compare_regressors(
    ground_truth=ground_truth,
    model_predictions=[perfect_model, noisy_model, random_model, zero_model],
    model_names=["Perfect Model", "Noisy Model", "Random Model", "Zero Model"],
).display()
'''

offsets = [7,14,21,28,35]
offsets = list(range(1,60))

#nameid = 'UnitedKingdom'
nameid = 'CanadaOntario'
savepath = 'D:/GoogleDrive/eval/'+nameid + '/'

truthfile = savepath + 'truth.csv'

#df_t = pd.read_csv(truthfile, parse_dates=["date"])

df_t = pd.read_csv(savepath + 'pred_0.csv', parse_dates=["date"])

days = 35#35#45
last_day = df_t.date[len(df_t.date)-1]

dfs = {}
plt.figure()
for offset in offsets: # offset = 14
    filepath = savepath + 'pred_' + str(offset) + '.csv'

    df = pd.read_csv(filepath, parse_dates=["date"])
    
    # When the predictions begin
    start_day = last_day - datetime.timedelta(days=offset)

    plt.plot(df.date[df.date >= start_day],df.prev[df.date >= start_day],
        alpha=0.5)
    plt.ylim(0,0.4)
    #plt.plot(df.date,df.prev,alpha=0.1)

    for d in np.arange(1,offset): # d = 1
        if d not in dfs:
            dfs[d] = pd.DataFrame()

        preddate = start_day+datetime.timedelta(days=int(d))

        if True:#preddate in df.date:
            truevals = df_t[df_t.date == preddate]
            predvals = df[df.date == preddate]
            df_row = pd.DataFrame(dict(
                date=start_day+datetime.timedelta(days=int(d)),
                true=truevals.prev.to_numpy()[0],
                pred=predvals.prev.to_numpy()[0],
                offset=offset,
                predlen=d
                ), index=[0])
            dfs[d] = dfs[d].append(df_row,ignore_index=True)
plt.plot(df_t.date, df_t.prev, LineWidth=3, label="Median prevalence (full data)")

plt.xticks(rotation=45)
plt.xlabel("Day")
plt.ylabel("% p.p.")

RMSEs = []
nRMSEs = []
MAEs = []
MGT = []
MPred = []
R2s = []
d=[]
for predlen in dfs.keys(): # predlen = 1
    perf = metriculous.compare_regressors(
        ground_truth=dfs[predlen].true,
        model_predictions=[dfs[predlen].pred],
        model_names=["7"],
    )#.display()
    R2s = R2s + [(perf.evaluations[0].quantities[0].value)]
    RMSEs = RMSEs + [(perf.evaluations[0].quantities[3].value)]
    MAEs = MAEs + [(perf.evaluations[0].quantities[10].value)]
    MGT = MGT + [(perf.evaluations[0].quantities[11].value)]
    MPred = MPred + [(perf.evaluations[0].quantities[12].value)]
    nRMSEs = nRMSEs + [(perf.evaluations[0].quantities[3].value/perf.evaluations[0].quantities[8].value)]
    d = d + [predlen]



plt.figure()
plt.plot(range(1,days),RMSEs[:days-1], LineWidth=4, label="Root Mean Square Error")
#plt.plot(RMSEs)
plt.title("")
plt.xlabel("Days predicted")
plt.ylabel("RMSE (% p.p.)")
plt.legend()

plt.figure()
#plt.plot(range(1,30),RMSEs[:29])
plt.plot(range(1,days), MAEs[:days-1], LineWidth=4, label="Median Absolute Error")
plt.title("")
plt.xlabel("Days predicted")
plt.ylabel("MAE (% p.p.)")
plt.legend()

plt.figure()
plt.plot(range(1,days),MGT[:days-1], LineWidth=4, label="Median true prevalence")
plt.title("")
plt.plot(range(1,days),MPred[:days-1], 'r--', LineWidth=3, label="Median predicted prevalence")
plt.title("")
plt.xlabel("Days predicted")
plt.ylabel("% p.p.")
plt.legend()

plt.figure()
plt.plot(range(1,days),RMSEs[:days-1])
plt.title("")
plt.xlabel("Days")
plt.ylabel("%")

plt.figure()
plt.plot(R2s)
plt.title("$R^2$")
plt.ylim(-10,1)