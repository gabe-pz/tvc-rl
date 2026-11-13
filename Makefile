CXX = g++

.PHONY: sim train clean

#Build & run sim
sim:
	mkdir -p bin
	$(CXX) -Wall -g -I include src/sim.cpp $(shell find includes -name '*.cpp') -o bin/sim -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
	./bin/sim

clean:
	rm -rf bin
