## Build the image

    docker build . -t jbaptista99/hgr:0.0

    docker push jbaptista99/hgr:0.0

    docker pull jbaptista99/hgr:0.0

## Spawn the container - interactive

    docker run -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --network=host -v /home/jbaptista/Datasets/ASL/train:/root/Datasets/ASL/train -v /home/jbaptista/Datasets/ASL/val:/root/Datasets/ASL/val -v /home/jbaptista/network/ptorch:/root/network/ptorch  jbaptista99/hgr:0.0 bash

## Attach to a running container

    docker attach <container_id>

## Dettach from running container

    press CRTL-p & CRTL-q