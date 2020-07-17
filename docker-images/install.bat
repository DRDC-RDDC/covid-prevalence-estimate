REM REQUIREMENTS
REM   - installed kubectl, aws
REM   - aws logged in (aws sso login)

REM install the persistent volume for the redis controller
kubectl apply -f ./controller/redis-volume-data.yaml

REM install the redis service - networking
kubectl apply -f ./controller/redis-service.yaml

REM launch the redis database
kubectl apply -f ./controller/redis-deployment.yaml

REM log into docker.  replace horn-eks with the profile used, and region with desired region
aws ecr get-login-password --region ca-central-1 --profile horn-eks | docker login --username AWS --password-stdin 979276261708.dkr.ecr.ca-central-1.amazonaws.com

REM Controller
docker build --no-cache -t covprev-ctl ./controller

REM this line should have the container registry used for docker
docker tag covprev-ctl:latest 979276261708.dkr.ecr.ca-central-1.amazonaws.com/covprev-ctl:latest
docker push 979276261708.dkr.ecr.ca-central-1.amazonaws.com/covprev-ctl:latest

