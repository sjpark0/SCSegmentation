xhost +local:docker
docker run --shm-size=64g --memory=90g --memory-swap=90g --gpus all -it --rm --env DISPLAY=$DISPLAY --env QT_X11_NO_MITSHM=1 --env SCSAM3_IMAGE_ID="$(docker images --no-trunc -q scsam3:latest 2>/dev/null | head -n1)" --volume /tmp/.X11-unix:/tmp/.X11-unix --volume /:/host --workdir /host$PWD scsam3
