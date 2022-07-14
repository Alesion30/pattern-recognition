MAKEFILE_DIR:=$(dir $(abspath $(lastword $(MAKEFILE_LIST))))

bash: # use bash
	docker-compose exec python bash

recreate: # delete and restart docker
	docker-compose down -v --rmi all --remove-orphans
	docker-compose up -d

restart: # restart docker
	docker-compose restart

freeze: # show list of installed pip packages
	docker-compose exec python pip freeze

reinstall: # uninstall all package and reinstall pip package
	./pip.sh uninstall -r requirements.txt -y
	./pip.sh install -r requirements.txt

sshkey: # create ssh key
	ssh-keygen -t rsa -f $(MAKEFILE_DIR)key/id_rsa
