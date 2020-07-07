# COVID-Prevalence-Estimate

This is the implementation of the Bayesian inference of COVID-19 
point-prevalence.

## Architecture

This code is intended to be run on a compute cluster using a 
Kubernetes + Docker framework - such as on Amazon Elastic Kubernetes Service.

The program is structured into 3 primary components.

1. Controller Pod.  This loads the data and configuration and creates work units.
2. Redis Pod.  This is a processing and messaging bus which holds the work queues.
3. Worker Jobs. This is the processing unit, which can scale out and consumes work units.

When the workers are completed, they submit their results to a pre-configured
git repository.

## Licenses

Copyright (c) Her Majesty the Queen in Right of Canada, as represented by the Minister of National Defence, 2020.