"""
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
"""

import covid19_inference as cov19
import datetime
from dateutil.parser import parse, isoparse
import numpy as np
import os
import pandas as pd
import requests
import re
from . import utility as ut


def savecsv(this_model, trace, pop, rootpath="/content"):
    """Export results as csv files"""
    analysis_date = datetime.datetime.utcnow()
    savefolder, folder = ut.get_folders(pop, rootpath=rootpath)
    hr_uid = 0
    FIPS = 0

    if pop["source"] == "codwg":
        # infodf = pd.read_csv(rootpath + '/Covid19Canada/other/hr_map.csv')
        # popinfo = infodf.loc[lambda df: (df['province'] == pop["source_state"]) & (df['health_region'] == pop["source_region"])]
        # hr_uid = popinfo["HR_UID"].to_numpy()[0]

        # the health region id is now saved in the pop dictionary
        hr_uid = int(pop["geoid"])

    if pop["source"] == "codwg-plus-sk":
        hr_uid = int(pop["geoid"])

    if pop["source"] == "jhu-us":
        infodf = pd.read_csv(
            rootpath + "/covid-prevalence/data/UID_ISO_FIPS_LookUp_Table.csv"
        )
        if pop["source_region"] is None:
            popinfo = infodf.loc[
                lambda df: (df["Province_State"] == pop["source_state"])
                & (df["iso2"] == pop["source_country"])
            ]
        else:
            popinfo = infodf.loc[
                lambda df: (df["Province_State"] == pop["source_state"])
                & (df["Admin2"] == pop["source_region"])
            ]
        FIPS = popinfo["FIPS"].to_numpy()[0]

    # Make sure it's an integer, otherwise appears as float in CSV
    FIPS = int(FIPS)

    # Timeseries
    N = pop["N"]
    E_t = trace["E_t"][:, None]
    R_t = trace["R_t"][:, None]
    I_t = trace["I_t"][:, None]
    new_E_t = trace["new_E_t"][:, None]
    lambda_t, x = cov19.plot._get_array_from_trace_via_date(
        this_model, trace, "lambda_t"
    )
    Ip_t = I_t + E_t + R_t

    # we check for degenerate/oscillating solutions if nne < 0 in the trace (This shouldn't happen!)
    nne = new_E_t / E_t
    degens = nne < 0
    degen = np.array([np.sum(d[0]) > 0 for d in degens])

    Prev_t_025, Prev_t_50, Prev_t_975 = ut.get_percentile_timeseries(Ip_t, degen=degen)
    I_t_025, I_t_50, I_t_975 = ut.get_percentile_timeseries(I_t, degen=degen)
    L_t_025, L_t_50, L_t_975 = ut.get_percentile_timeseries(
        lambda_t, islambda=True, degen=degen
    )

    # These will be the columns in the CSV file
    data = dict(
        FIPS=FIPS,
        HR_UID=list(np.repeat(hr_uid, x.shape[0])),
        nameid=list(np.repeat(folder, x.shape[0])),
        province=list(np.repeat(pop["source_state"], x.shape[0])),
        region=list(np.repeat(pop["source_region"], x.shape[0])),
        analysisTime=list(np.repeat(analysis_date, x.shape[0])),
        dates=list(x.to_numpy()),
        pointprevalence_025=list(
            map(lambda x: 100 * x / N if x >= 0 else 0.0, Prev_t_025)
        ),
        pointprevalence_50=list(
            map(lambda x: 100 * x / N if x >= 0 else 0.0, Prev_t_50)
        ),
        pointprevalence_975=list(
            map(lambda x: 100 * x / N if x >= 0 else 0.0, Prev_t_975)
        ),
        pointinfections_025=list(
            map(lambda x: int(np.floor(x)) if np.floor(x) >= 0 else 0, Prev_t_025)
        ),
        pointinfections_50=list(
            map(lambda x: int(np.floor(x)) if np.floor(x) >= 0 else 0, Prev_t_50)
        ),
        pointinfections_975=list(
            map(lambda x: int(np.floor(x)) if np.floor(x) >= 0 else 0, Prev_t_975)
        ),
        pointinfectious_025=list(
            map(lambda x: int(np.floor(x)) if np.floor(x) >= 0 else 0, I_t_025)
        ),
        pointinfectious_50=list(
            map(lambda x: int(np.floor(x)) if np.floor(x) >= 0 else 0, I_t_50)
        ),
        pointinfectious_975=list(
            map(lambda x: int(np.floor(x)) if np.floor(x) >= 0 else 0, I_t_975)
        ),
        pointinfectiousprevalence_05=list(
            map(lambda x: 100 * x / N if x >= 0 else 0.0, I_t_025)
        ),
        pointinfectiousprevalence_50=list(
            map(lambda x: 100 * x / N if x >= 0 else 0.0, I_t_50)
        ),
        pointinfectiousprevalence_95=list(
            map(lambda x: 100 * x / N if x >= 0 else 0.0, I_t_975)
        ),
        infectrate_025=L_t_025,
        infectrate_50=L_t_50,
        infectrate_975=L_t_975,
    )

    dfr = pd.DataFrame(data)
    rfilepath = savefolder + "/" + folder + "_timeseries.csv"
    dfr = dfr.sort_values(by=["dates"])
    dfr["FIPS"] = dfr["FIPS"].astype(int)
    dfr.to_csv(rfilepath, index=False, float_format="%.8f")
    dfu = dfr  # update is only the region
    todayix = np.where(x > analysis_date)[0][0] - 1

    # point result csv file (time of analysis)
    data_now = dict(
        FIPS=FIPS,
        HR_UID=hr_uid,
        nameid=folder,
        province=pop["source_state"],
        region=pop["source_region"],
        analysisTime=analysis_date,
        dates=x[todayix],
        compute_time=pop["compute_time"],
        divergences=pop["divs"],
        draws=pop["draws"],
        tunes=pop["tunes"],
        pointprevalence_025=100 * Prev_t_025[todayix] / N
        if Prev_t_025[todayix] >= 0
        else 0.0,
        pointprevalence_50=100 * Prev_t_50[todayix] / N
        if Prev_t_50[todayix] >= 0
        else 0.0,
        pointprevalence_975=100 * Prev_t_975[todayix] / N
        if Prev_t_975[todayix] >= 0
        else 0.0,
        pointinfections_025=list(
            map(
                lambda x: int(np.floor(x)) if np.floor(x) >= 0 else 0,
                [Prev_t_025[todayix]],
            )
        ),
        pointinfections_50=list(
            map(
                lambda x: int(np.floor(x)) if np.floor(x) >= 0 else 0,
                [Prev_t_50[todayix]],
            )
        ),
        pointinfections_975=list(
            map(
                lambda x: int(np.floor(x)) if np.floor(x) >= 0 else 0,
                [Prev_t_975[todayix]],
            )
        ),
        pointinfectious_025=list(
            map(
                lambda x: int(np.floor(x)) if np.floor(x) >= 0 else 0,
                [I_t_025[todayix]],
            )
        ),
        pointinfectious_50=list(
            map(
                lambda x: int(np.floor(x)) if np.floor(x) >= 0 else 0, [I_t_50[todayix]]
            )
        ),
        pointinfectious_975=list(
            map(
                lambda x: int(np.floor(x)) if np.floor(x) >= 0 else 0,
                [I_t_975[todayix]],
            )
        ),
        pointinfectiousprevalence_025=list(
            map(lambda x: 100 * x / N if x >= 0 else 0.0, [I_t_025[todayix]])
        ),
        pointinfectiousprevalence_50=list(
            map(lambda x: 100 * x / N if x >= 0 else 0.0, [I_t_50[todayix]])
        ),
        pointinfectiousprevalence_975=list(
            map(lambda x: 100 * x / N if x >= 0 else 0.0, [I_t_975[todayix]])
        ),
        infectrate_025=L_t_025[todayix],
        infectrate_50=L_t_50[todayix],
        infectrate_975=L_t_975[todayix],
    )

    dfnow = pd.DataFrame(data_now)
    rfilepath = savefolder + "/" + folder + "_latest.csv"
    dfnow["FIPS"] = dfnow["FIPS"].astype(int)  # make sure FIPS is saved as an integer
    dfnow.to_csv(
        rfilepath, index=False, float_format="%.8f"
    )  # save with 8 decimal places
    return dfu, dfnow


def get_data(pop, rootpath="/content"):
    bd = isoparse(pop["date_start"])
    if pop["source"] == "jhu":
        jhu = cov19.data_retrieval.JHU(auto_download=True)
        country = pop["source_country"]
        state = pop["source_state"]
        new_cases = jhu.get_new(country=country, state=state, data_begin=bd)
        cum_deaths = jhu.get_total(
            value="deaths", country=country, state=state, data_begin=bd
        )

    elif pop["source"] == "jhu-us":
        dataurl = (
            rootpath
            + r"/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv"
        )
        df = pd.read_csv(dataurl)
        df = df.drop(
            columns=[
                "Lat",
                "Long_",
                "UID",
                "iso2",
                "iso3",
                "code3",
                "FIPS",
                "Combined_Key",
                "Population",
            ]
        ).rename(
            columns={
                "Province_State": "state",
                "Country_Region": "country",
                "Admin2": "county",
            }
        )
        if pop["source_region"] is None:
            df = df.drop(columns=["county"])
            df = df.set_index(["country", "state"])
        else:
            df = df.set_index(["country", "state", "county"])

        df.columns = pd.to_datetime(df.columns)
        if pop["source_region"] is None:
            df = df.loc[(pop["source_country"], pop["source_state"])]
        else:
            df = df.loc[
                (pop["source_country"], pop["source_state"], pop["source_region"])
            ]
        df = df.T
        df.index.name = "date"
        dfddata = pd.DataFrame(columns=["date", "deaths"]).set_index("date")
        # popname = "Colorado - El Paso"
        if pop["source_region"] is None:
            dfddata["deaths"] = df[(pop["source_country"], pop["source_state"])]
        else:
            dfddata["deaths"] = df

        end_date = dfddata.index[-1]
        dfdcumdata = dfddata[bd:end_date]
        cum_deaths = dfdcumdata["deaths"]

        dataurl = (
            rootpath
            + r"/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv"
        )

        # cases
        df = pd.read_csv(dataurl)
        df = df.drop(
            columns=[
                "Lat",
                "Long_",
                "UID",
                "iso2",
                "iso3",
                "code3",
                "FIPS",
                "Combined_Key",
            ]
        ).rename(
            columns={
                "Province_State": "state",
                "Country_Region": "country",
                "Admin2": "county",
            }
        )

        if pop["source_region"] is None:
            df = df.drop(columns=["county"])
            df = df.set_index(["country", "state"])
        else:
            df = df.set_index(["country", "state", "county"])

        df.columns = pd.to_datetime(df.columns)
        if pop["source_region"] is None:
            df = df.loc[(pop["source_country"], pop["source_state"])]
        else:
            df = df.loc[
                (pop["source_country"], pop["source_state"], pop["source_region"])
            ]

        df = df.T
        df.index.name = "date"
        dfdata = pd.DataFrame(columns=["date", "confirmed"]).set_index("date")

        if pop["source_region"] is None:
            dfdata["confirmed"] = df[(pop["source_country"], pop["source_state"])]
        else:
            dfdata["confirmed"] = df

        end_date = dfdata.index[-1]
        dfdatafilter = dfdata[bd:end_date]
        dfdatafilter = dfdatafilter.diff().drop(dfdatafilter.index[0]).astype(int)
        new_cases = dfdatafilter["confirmed"]

    elif pop["source"] == "codwg-plus-sk":
        pageurl = "https://dashboard.saskatchewan.ca/health-wellness/covid-19/cases"
        page = requests.get(pageurl)
        htmltext = page.text
        filename = re.search(r"\d+.csv", htmltext).group()
        dataurl = "https://dashboard.saskatchewan.ca/export/cases/" + filename
        df = pd.read_csv(dataurl)

        df = df[["Date", "New Cases", "Region"]].rename(
            columns={
                "New Cases": "confirmed",
                "Date": "date",
                "Region": "health_region",
            }
        )
        df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y/%m/%d")
        df = df.set_index(["health_region"])
        dfi = df.loc[(pop["source_region"])]

        dfi = dfi.groupby(["date"]).sum()  # fix reporting by sk multiple times per day

        dfdata = dfi
        end_date = dfdata.index[-1]

        if "bd" not in locals():
            bd = dfdata.index[0]

        if bd < dfdata.index[0]:
            bd = dfdata.index[0]

        dfdatafilter = dfdata[bd:end_date]
        new_cases = dfdatafilter["confirmed"]

        try:
            # dataurl = "https://dashboard.saskatchewan.ca/export/cases/1688.csv"

            df = pd.read_csv(dataurl)

            df = df[["Date", "Deaths", "Region"]].rename(
                columns={"Deaths": "deaths", "Date": "date", "Region": "health_region"}
            )
            df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y/%m/%d")
            df = df.set_index(["health_region"])
            dfi = df.loc[(pop["source_region"])]
            dfi = dfi.groupby(
                ["date"]
            ).sum()  # fix reporting by sk multiple times per day
            dfdata = dfi
            end_date = dfdata.index[-1]

            if "bd" not in locals():
                bd = dfdata.index[0]

            if bd < dfdata.index[0]:
                bd = dfdata.index[0]

            dfdatafilter = dfdata[bd:end_date]
            cum_deaths = dfdatafilter["deaths"]
        except:
            # IFR estimation will not work without deaths, but we need a value here. TODO: better error handling.
            cum_deaths = new_cases

    elif pop["source"] == "codwg":
        # Check if we've downloaded the latest
        dataurl = rootpath + r"/Covid19Canada/timeseries_hr/cases_timeseries_hr.csv"
        df = pd.read_csv(dataurl)
        df = df.drop(columns=["cumulative_cases"]).rename(
            columns={"cases": "confirmed", "date_report": "date"}
        )
        df["date"] = pd.to_datetime(df["date"].astype(str), format="%d-%m-%Y")
        df = df.set_index(["province", "health_region"])
        dfi = df.loc[(pop["source_state"], pop["source_region"])]
        dfi = dfi.set_index("date")
        dfdata = dfi
        end_date = dfdata.index[-1]
        dfdatafilter = dfdata[bd:end_date]
        new_cases = dfdatafilter["confirmed"]

        try:
            dataurl = (
                rootpath + r"/Covid19Canada/timeseries_hr/mortality_timeseries_hr.csv"
            )
            df = pd.read_csv(dataurl)
            df = df.drop(columns=["deaths"]).rename(
                columns={"cumulative_deaths": "deaths", "date_death_report": "date"}
            )
            df["date"] = pd.to_datetime(df["date"].astype(str), format="%d-%m-%Y")
            df = df.set_index(["province", "health_region"])
            dfi = df.loc[(pop["source_state"], pop["source_region"])]
            dfdata = dfi.set_index("date")
            end_date = dfdata.index[-1]
            dfdatafilter = dfdata[bd:end_date]
            cum_deaths = dfdatafilter["deaths"]
        except:
            # IFR estimation will not work without deaths, but we need a value here. TODO: better error handling.
            cum_deaths = new_cases

    return new_cases, cum_deaths, bd
