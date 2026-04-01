# GEX-nils
Deep learning models for gene expression analysis to predict lymph node metastasis in early breast cancer

## Quick installation using Docker
### 1. Build Docker image 


```shell
cd GEX-nils
docker build -f Dockerfile_py3.8 -t gex-nils-app .
```

### 2. Run Docker by starting an interactive shell
```shell
docker run --gpus all -it gex-nils-app bash
```
