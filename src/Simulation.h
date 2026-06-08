#pragma once

#include "Types.h"
#include "PhysicsEngine.h"
#include <fstream>
#include <memory>

namespace modern {

class Simulation {
public:
    explicit Simulation(const std::string& input_file, GeometryType geometry, bool use_rotne_prager);
    
    void run();

private:
    void parseInput(const std::string& input_file);
    void initializeState();
    void step(double time);
    void activeTrapping();
    void saveData(std::ostream& out, int frame);
    void saveTraps(std::ostream& out, int frame);
    void saveForces(std::ostream& out, int frame);
    
    std::string generateTitle() const;

    std::string input_filename_;
    SimulationConfig config_;
    State state_;
    std::unique_ptr<PhysicsEngine> engine_;
    
    // cached buffers for physics
    MatrixXd tensor_;
    MatrixXd diffusion_;
    VectorXd forces_;
    VectorXd noise_;
    
    GeometryType geometry_arg_;
};

} // namespace modern
