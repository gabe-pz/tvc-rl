CXX = g++

#all cpp files in includes
LIB_SRCS = $(shell find includes -name '*.cpp')

#train skips graphics
TRAIN_SRCS = src/train.cpp $(filter-out includes/graphics/%, $(LIB_SRCS))

#sim uses everything
SIM_SRCS = src/sim.cpp $(LIB_SRCS)

BREW = $(shell brew --prefix)

.PHONY: sim sim-mac train clean

#build and run sim
sim:
	mkdir -p bin
	$(CXX) -std=c++17 -Wall -g -I includes $(SIM_SRCS) -o bin/sim -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
	./bin/sim

#build and run train
train:
	mkdir -p bin
	$(CXX) -std=c++17 -Wall -O2 -I includes $(TRAIN_SRCS) -o bin/train
	./bin/train

#build and run sim on mac
sim-mac:
	mkdir -p bin
	$(CXX) -std=c++17 -Wall -g -I includes -I$(BREW)/include $(SIM_SRCS) -o bin/sim -L$(BREW)/lib -lraylib -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo
	./bin/sim

#delete build files
clean:
	rm -rf bin
