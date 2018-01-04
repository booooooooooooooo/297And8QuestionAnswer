## My Memo
Always run code in 297And8QuestionAnswer!

docker run -it --rm -p 8888:8888 -v \`pwd\`:/297And8QuestionAnswer  masterproject_2

docker exec -i -t 5b0bb9b8547d  /bin/bash

docker build -t friendlyname .  # Create image using this directory's Dockerfile

docker container ls                                # List all running containers

docker container ls -a             # List all containers, even those not running

docker container stop <hash>           # Gracefully stop the specified container

docker container kill <hash>         # Force shutdown of the specified container

docker container rm <hash>        # Remove specified container from this machine

docker container rm $(docker container ls -a -q)         # Remove all containers

docker image ls -a                             # List all images on this machine

docker image rm <image id>            # Remove specified image from this machine

docker image rm $(docker image ls -a -q)   # Remove all images from this machine



git rm -r --cached .

# Question Answerer

### Project Description
```
TODO
```

### How To Run

```
TODO
```
