#/bin/bash

# TODO: I should probably modify this to take GPU support as an argument

# Run with GPU support
docker run -d --runtime=nvidia --name=ace-net --volume $ACE_NET_HOME:/workspace --entrypoint= jalane76/ace-net:latest tail -f /dev/null

# Uncomment to run without GPU support
# docker run -d --name=ace-net --volume /home/jesse/git/ace-net:/workspace --entrypoint= jalane76/ace-net:latest tail -f /dev/null