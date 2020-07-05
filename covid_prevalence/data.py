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

import pandas as pd
import covid19_inference as cov19
from dateutil.parser import parse, isoparse

def get_data(pop):
  bd = isoparse(pop['date_start'])
  if pop['source'] == 'jhu':
    jhu = cov19.data_retrieval.JHU(auto_download=True)
    country = pop['source_country']
    state = pop['source_state']
    new_cases = jhu.get_new(country=country, state=state, data_begin=bd)
    cum_deaths = jhu.get_total(value='deaths', country=country, state=state, data_begin=bd)

  elif pop['source'] == 'jhu-us':
    dataurl = r"/content/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv"
    df = pd.read_csv(dataurl)
    df = df.drop(columns=["Lat", "Long_","UID","iso2","iso3","code3","FIPS","Combined_Key","Population"]).rename(
                        columns={"Province_State": "state", "Country_Region": "country", "Admin2": "county"}
                    )
    df = df.set_index(["country", "state", "county"])
    df.columns = pd.to_datetime(df.columns)
    df = df.T
    df.index.name = "date"
    dfddata = pd.DataFrame(columns=["date", "deaths"]).set_index("date")
    #popname = "Colorado - El Paso"
    dfddata["deaths"] = df[("US","Colorado", "El Paso")]
    end_date = dfddata.index[-1]
    #dfddatafilter = dfddata[bd:end_date]
    dfdcumdata = dfddata[bd:end_date]
    #dfddata.plot()
    print("Last data: " + str(end_date))
    cum_deaths = dfdcumdata["deaths"]
    
    dataurl = r"/content/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv"

    # cases
    df = pd.read_csv(dataurl)
    df = df.drop(columns=["Lat", "Long_","UID","iso2","iso3","code3","FIPS","Combined_Key"]).rename(
                        columns={"Province_State": "state", "Country_Region": "country", "Admin2": "county"}
                    )
    df = df.set_index(["country", "state", "county"])
    df.columns = pd.to_datetime(df.columns)
    df = df.T
    df.index.name = "date"
    dfdata = pd.DataFrame(columns=["date", "confirmed"]).set_index("date")
    dfdata["confirmed"] = df[("US","Colorado", "El Paso")]
    end_date = dfdata.index[-1]
    dfdatafilter = dfdata[bd:end_date]
    dfdatafilter = (dfdatafilter.diff().drop(dfdatafilter.index[0]).astype(int))
    new_cases = dfdatafilter["confirmed"]
    
  elif pop['source'] == 'codwg':
    # Check if we've downloaded the latest
    
    #dataurl = r"https://raw.githubusercontent.com/ishaberry/Covid19Canada/master/timeseries_hr/cases_timeseries_hr.csv"
    dataurl = "/content/Covid19Canada/timeseries_hr/cases_timeseries_hr.csv"
    df = pd.read_csv(dataurl)
    df = df.drop(columns=["cumulative_cases"]).rename(
                    columns={"cases": "confirmed", "date_report":"date"}
                )
    df['date']= pd.to_datetime(df['date'].astype(str), format='%d-%m-%Y')
    df = df.set_index(["province", "health_region"])
    dfi = df.loc[(pop['source_state'],pop['source_region'])]#df[df['health_region']==pop['source_region']]#'Montréal']
    dfi = dfi.set_index("date")
    dfdata = dfi
    #dfdata = dfi.drop(columns=["province","health_region"])
    #dfdata = dfdata.set_index("date")
    end_date = dfdata.index[-1]
    dfdatafilter = dfdata[bd:end_date]
    #dfdata.plot()
    #dfdatafilter.plot()
    new_cases = dfdatafilter["confirmed"]

    try:
      #dataurl = r"https://raw.githubusercontent.com/ishaberry/Covid19Canada/master/timeseries_hr/mortality_timeseries_hr.csv"
      dataurl = "/content/Covid19Canada/timeseries_hr/mortality_timeseries_hr.csv"
      df = pd.read_csv(dataurl)
      df = df.drop(columns=["deaths"]).rename(
                      columns={"cumulative_deaths": "deaths", "date_death_report":"date"}
                  )
      df['date']= pd.to_datetime(df['date'].astype(str), format='%d-%m-%Y')
      df = df.set_index(["province", "health_region"])
      dfi = df.loc[(pop['source_state'],pop['source_region'])]
      dfdata = dfi.set_index("date")
      end_date = dfdata.index[-1]
      dfdatafilter = dfdata[bd:end_date]
      cum_deaths = dfdatafilter["deaths"]
    except:
      # IFR will not work
      cum_deaths = new_cases

  return new_cases, cum_deaths, bd