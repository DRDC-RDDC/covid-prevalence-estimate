![](img/DRDC.png)

# COVID-Prevalence-Estimate

An implementation of Bayesian inference and prediction of COVID-19 point-prevalence.

Definition of point-prevalence:  The portion of infected individuals in a population at a given time.

## Model Description

The model is an SEIR-like model, which treats case detections as biased for symptomatic onset with a variable delay to test report.
Consideration is given for the likelihood of asymptomatic cases and undetected cases.
Since the focus of this model is on active (infectious) cases in the population, recovery in quarantine and deaths are not considered.

Once the model is tuned to the case detections of the population, a prediction is provided to provide an indication of the expected future prevalence and cases.

## Acknowledgements

The following individuals have provided support or contribution to this project:

* Dr. Ramzi Mirshak
* Dr. David Waller
* Dr. Steven Schofield
* Mr. Michael A. Salciccoli
* Dr. Steve Guillouzic
* Mr. Andrew Sirjoosingh
* Mr. Alasdair Grant

## Citation

Horn, S., COVID-19 prevalence estimation tool, DRDC CORA Software, 2020

## References

1. Mirshak and Horn (2020), The probability of a non-symptomatic individual bringing a coronavirus disease 2019 (COVID-19) infection into a group, DRDC Scientific Letter, DRDC-RDDC-2020-L116
2. Berry I, Soucy J-PR, Tuite A, Fisman D. Open access epidemiologic data and an interactive dashboard to monitor the COVID-19 outbreak in Canada. CMAJ. 2020 Apr 14;192(15):E420. doi: https://doi.org/10.1503/cmaj.75262
3. Dehning, J., Zierenberg, J., Spitzner, F. P., Wibral, M., Neto, J. P., Wilczek, M., & Priesemann, V. (2020). Inferring change points in the spread of COVID-19 reveals the effectiveness of interventions. Science.
4. COVID-19 Data Repository by the Center for Systems Science and Engineering (CSSE) at Johns Hopkins University

## Licenses

GPL-3.0

Copyright (c) Her Majesty the Queen in Right of Canada, as represented by the Minister of National Defence, 2020.

This software is a derived work and uses, under GPL-3.0, parts of Bayesian inference and forecast of COVID-19, code repository: https://github.com/Priesemann-Group/covid19_inference
(C) Copyright 2020, Jonas Dehning, Johannes Zierenberg, F. Paul Spitzner, Michael Wibral, Joao Pinheiro Neto, Michael Wilczek, Viola Priesemann

![](img/Canada.png)