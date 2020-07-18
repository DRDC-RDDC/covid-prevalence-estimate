# Work controller

This is the parallel processing work controller, which will create jobs.

## Installation

1. Install Kubernetes

1. Configure EFS persistent storage.  see https://aws.amazon.com/premiumsupport/knowledge-center/eks-persistent-storage/ for a great walkthrough

1. Add gp2 storage class for persistent EBS storage for redis

1. Add efs storage class for shared storage for workers

1. Add redis service:  `kubectl apply -f ./redis-service.yaml`

1. Add redis volume claim: `kubectl apply -f ./redis-volume-data.yaml`

1. Add redis master: `kubectl apply -f ./redis-deployment.yaml`

1. EFS will mount to /data.  Clone the 4 repositories there which must be accessible by the account simuser.
