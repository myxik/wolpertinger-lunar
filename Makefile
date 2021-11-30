NAME?=wolpertinger
GPUS?=all
PORT?=3333
ifeq ($(GPUS), none)
	GPUS_OPTION=
else
	GPUS_OPTION=$(GPUS)
endif

PROJ_ROOT?=/raid/maxim_mametkulov/rl/wolp/
CODE_ROOT?=

.PHONY: build stop attach exec run-dev

build:
	docker build \
	-t $(NAME) .

stop:
	-docker stop $(NAME)
	-docker rm $(NAME)

exec:
	docker exec -it $(NAME) bash

attach:
	docker attach $(NAME)

run-dev:
	docker run --rm -it -p $(PORT):5901 --runtime=nvidia \
	-e NVIDIA_VISIBLE_DEVICES=$(GPUS_OPTION) \
	-v $(PROJ_ROOT):/workspace \
	--name=$(NAME) \
	--shm-size=32g \
	$(NAME) 

