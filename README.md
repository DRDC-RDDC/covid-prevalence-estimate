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

## Installation

Start: Assume a cluster has been created, and the user is authenticated with kubectl cli 
configured.

### Metrics server

https://docs.aws.amazon.com/eks/latest/userguide/metrics-server.html

`kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/download/v0.3.6/components.yaml`

### Autoscale

The autoscale allows kubernetes to launch new compute nodes when the requested CPU is higher than available in the cluster.

https://docs.aws.amazon.com/eks/latest/userguide/cluster-autoscaler.html

The node group requires these tags:

k8s.io/cluster-autoscaler/<cluster-name>    owned
k8s.io/cluster-autoscaler/enabled   true



### Install Redis service

`kubectl apply -f ./docker-images/controller/redis-service.yaml`

Check if installed by typing `kubectl get services`

### Install Redis persistent storage

Create the storage class.  Here we use EBS storage.

Install the ebs-csi driver:

`kubectl apply -k "github.com/kubernetes-sigs/aws-ebs-csi-driver/deploy/kubernetes/overlays/stable/?ref=master"`

Create the storage class:

`kubectl apply -f ./docker-images/controller/gp2-storage-class.yaml`

Create the persistent volume claim:

`kubectl apply -f ./redis-volume-data.yaml`

Check if configured by typing `kubectl get pvc`

### Deploy the redis configuration map

`kubectl create configmap redis-config --from-file=./redis.conf`

Check if deployed ok: `kubectl get configmap`

### Deploy the redis service master

Deploy the master node:

`kubectl apply -f redis-deployment.yaml`

Check if creted by typing `kubectl get pods`.  You should see redis-master-*

### Configure EFS connections

See https://aws.amazon.com/premiumsupport/knowledge-center/eks-persistent-storage/ for an example walkthrough.

`kubectl apply -k "github.com/kubernetes-sigs/aws-efs-csi-driver/deploy/kubernetes/overlays/stable/?ref=master"`

update `./specs/pv.yaml'

Set the last line to the EFS file system name to use.

`kubectl apply -f specs/`

This will create the efs claims.

To check if the claim is there, type `kubectl get pvc` and note efs-claim should be there.

## Build and deploy docker containers

Assume that docker is running on machine building the containers.

### log into aws ecr

`aws ecr get-login-password --region <region> --profile <profile> | docker login --username AWS --password-stdin <Account>.dkr.ecr.<region>.amazonaws.com`

Replace <profile>, <account>, and <region> for example:

`aws ecr get-login-password --region ca-central-1 --profile horn-eks | docker login --username AWS --password-stdin 979276261708.dkr.ecr.ca-central-1.amazonaws.com`

### Controller

From the /docker-images/controller folder:

`docker build -t covprev-ctl .`

`docker tag covprev-ctl:latest 979276261708.dkr.ecr.ca-central-1.amazonaws.com/covprev-ctl:latest`

`docker push 979276261708.dkr.ecr.ca-central-1.amazonaws.com/covprev-ctl:latest`

To launch:

`kubectl apply -f ./cov-prev-ctl.yaml`

To view logs:

``

## Acknowledgements

The following individuals have provided significant support and contribution to this project:

* Dr. Ramzi Mirshak
* Dr. David Waller
* Dr. Steven Schofield
* Mr. Michael A. Salciccoli
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