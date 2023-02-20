az group create --name vhacdworkers --location westus
az container create --resource-group vhacdworkers --name scheduler --cpu 4 --memory 16 --image ghcr.io/dask/dask --dns-name-label scheduler --ports 8786 8787

For ($i=0; $i -le 5; $i++) {
    az container create --resource-group vhacdworkers --name worker --image docker.io/igibson/vhacd-worker --cpu 4 --memory 16 --command-line "dask-worker scheduler:8786"
}