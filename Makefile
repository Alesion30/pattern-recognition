bash:
	docker-compose exec python bash

recreate:
	docker-compose down -v --rmi all --remove-orphans
	docker-compose up -d

restart:
	docker-compose restart

freeze:
	docker-compose exec python pip freeze
