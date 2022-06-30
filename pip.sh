#!/bin/sh

docker-compose exec python pip $*
docker-compose exec python pip freeze > requirements.txt
