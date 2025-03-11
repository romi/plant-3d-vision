# ROMI P3DV WebTerm

A simple web terminal to run reconstructions in docker containers.


## Start with th docker compose service

```shell
docker-compose up --build
```

## Start container individually

```shell
ROMI_DB="/data/ROMI/test_owner/"

group_name=$(stat -c "%G" ${ROMI_DB})                              # get the name of the group for the 'host database path'
gid=$(getent group ${group_name} | cut --delimiter ':' --fields 3) # get the 'gid' of this group
```

### PlantDB
To start the `plantdb` container:
```shell
docker run --rm \
 --name plantdb \
 --user romi:${gid} \
 -p 5000:5000 \
 -v ${ROMI_DB}:/myapp/db \
 roboticsmicrofarms/plantdb:latest \
 "fsdb_rest_api --port 5000"
```

### Plant 3D Vision
To start the `plant3dvision` container:
```shell
docker run --rm \
 --name plant3dvision \
 --user romi:${gid} \
 --gpus all \
 --env PYOPENCL_CTX='0' \
 -it \
 roboticsmicrofarms/plant-3d-vision:latest \
 "bash"
```

### WebTerm
To build the `webterm` container:
```shell
docker build -t webterm -f Dockerfile .
```

To start the `webterm` container:
```shell
docker run --rm \
 --name webterm \
 -p 5001:5001 \
 -v /var/run/docker.sock:/var/run/docker.sock \
 --network="host" \
 webterm
```

Important notes:
1. The webapp container needs access to the Docker socket (`/var/run/docker.sock`) to interact with other containers. This is provided through the volume mount in the docker-compose file.
2. Make sure your target container (`plant3dvision`) is running in the same Docker network or is accessible to the webapp container.
3. To connect the terminal to your target container, you might need to add it to the same network:
``` bash
docker network connect project_app_network plant3dvision
```

For security in production:

- Consider using Docker API authentication
- Implement proper security measures for the Docker socket access
- Use environment variables for sensitive configuration
- Set up proper CORS policies
- Use HTTPS
