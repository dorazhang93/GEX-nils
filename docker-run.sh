# 1. build docker image 
docker build -f Dockerfile_py3.8 -t gex-nils-app .

#2. run docker by starting a interactive shell
docker run --gpus all -it gex-nils-app bash

