CXX = g++

SRCS = src/sim.cpp $(shell find includes -name '*.cpp')

BREW = $(shell brew --prefix)

.PHONY: sim sim-mac train clean

#build and run sim
sim:
	mkdir -p bin
	$(CXX) -Wall -g -I include $(SRCS) -o bin/sim -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
	./bin/sim

#build and run sim on mac
sim-mac:
	mkdir -p bin
	$(CXX) -std=c++17 -w -g -I include -I$(BREW)/include $(SRCS) -o bin/sim -L$(BREW)/lib -lraylib -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo
	./bin/sim

clean:
	rm -rf bin
