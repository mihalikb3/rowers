CXX = g++
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra -march=native -ffast-math -I./eigen-3.4.0 -I.
LDFLAGS = 

TARGET = rowers
SOURCES = main.cpp Simulation.cpp PhysicsEngine.cpp
OBJECTS = $(SOURCES:.cpp=.o)

EIGEN_URL = https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz

all: $(TARGET)

$(TARGET): eigen-3.4.0 $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

eigen-3.4.0:
	@echo "Downloading Eigen 3.4.0..."
	curl -L $(EIGEN_URL) -o eigen.tar.gz
	tar -xzf eigen.tar.gz
	rm eigen.tar.gz
	@echo "Eigen downloaded and extracted."

clean:
	rm -f $(TARGET) $(OBJECTS)

distclean: clean
	rm -rf eigen-3.4.0

.PHONY: all clean distclean help

help:
	@echo "Rowers Simulation Makefile"
	@echo "Usage:"
	@echo "  make            - Build the simulation (automatically downloads Eigen)"
	@echo "  make clean      - Remove objects and executable"
	@echo "  make distclean  - Remove objects, executable, and Eigen directory"
	@echo ""
	@echo "If you get 'Eigen/Dense: No such file' errors on the server:"
	@echo "  1. Run 'make distclean'"
	@echo "  2. Run 'make'"
	@echo "  3. This will re-download a fresh, correctly structured Eigen library."
