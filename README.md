# COVID-Prevalence-Estimate

An implementation of Bayesian inference of COVID-19 point-prevalence.

Definition of point-prevalence:  The portion of infected individuals at any given time.

## Architecture

This code is configured to be run on a compute cluster using a 
Kubernetes + Docker framework - such as on Amazon Elastic Kubernetes Service.

The program is structured into 3 primary components.

1. Controller Pod.  This loads the data and configuration and creates work units.
2. Redis Pod.  This is a processing and messaging bus which holds the work queues.
3. Worker Jobs. This is the processing unit, which can scale out and consumes work units.

When the workers are completed, they submit their results to a pre-configured
git repository.

## Acknowledgements

The following individuals have provided significant support and contribution to this project:

* Dr. Ramzi Mirshak
* Dr. David Waller
* Dr. Steve Schofield
* Mr. Michael Salciccoli
* Mr. Alasdair Grant

Open data is provided by 

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