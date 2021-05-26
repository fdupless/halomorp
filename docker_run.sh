docker build -t halomorp .

docker run -v $(pwd):/halomorp  \
            -e DISPLAY=$DISPLAY  \
            -v /tmp/.X11-unix:/tmp/.X11-unix:rw  \
            -p 8888:8888  \
            -it --rm -i halomorp /bin/bash