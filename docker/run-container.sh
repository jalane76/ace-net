#/bin/bash

GPU_MODE="--runtime=nvidia"
WORKSPACE_PATH=
DATA_PATH=

usage()
{
    echo "Usage: $0 [-c|--cpu] path/to/workspace path/to/data"
}

if [ $# -le 1 ]
then
    usage
    exit 1
fi

if [[ ($1 == "-c") || $1 == "--cpu" ]]
then
    GPU_MODE=""
    WORKSPACE_PATH=$2
    DATA_PATH=$3
else
    WORKSPACE_PATH=$1
    DATA_PATH=$2
fi

docker run -d $GPU_MODE --name=ace-net --volume $WORKSPACE_PATH:/workspace --volume $DATA_PATH:/data --entrypoint= jalane76/ace-net:latest tail -f /dev/null