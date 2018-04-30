
ifeq ($(shell uname),Linux)
CFLAGS=-std=c++11 -O3
endif

ifeq ($(shell uname),Darwin)
CFLAGS=--std=c++0x -stdlib=libc++
endif

default:
	g++ $(CFLAGS) -I./eigen node.cpp main.cpp mnist_loader.cpp

