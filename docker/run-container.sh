#/bin/bash

# Run with GPU support  TODO: I should change the mapped directory to an env var
docker run -d --runtime=nvidia --name=ace-net --volume /home/jesse/git/ace-net:/workspace --entrypoint= jalane76/ace-net:latest tail -f /dev/null

# Uncomment to run without GPU support
# docker run -d --name=ace-net --volume /home/jesse/git/ace-net:/workspace --entrypoint= jalane76/ace-net:latest tail -f /dev/null