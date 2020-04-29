#/bin/bash

docker run -d --runtime=nvidia --name=ace-net --volume /home/jesse/git/ace-ml:/workspace --entrypoint= ace-net:latest tail -f /dev/null