
ifeq ($(shell uname),Linux)
CFLAGS=-std=c++11 -O3 -g
endif

ifeq ($(shell uname),Darwin)
CFLAGS=--std=c++0x -stdlib=libc++ -O3 -g -framework Accelerate -DEIGEN_USE_BLAS
endif

default:
	g++ $(CFLAGS) -I./eigen node.cpp main.cpp mnist_loader.cpp

