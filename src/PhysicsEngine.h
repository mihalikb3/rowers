#pragma once

#include "Types.h"
#include <random>

namespace modern {

class PhysicsEngine {
public:
    explicit PhysicsEngine(const SimulationConfig& config);
    
    // assemble hydrodynamic tensor
    void computeHydrodynamicTensor(const State& state, MatrixXd& tensor) const;
    
    // Cholesky decomposition
    void computeCholesky(const MatrixXd& tensor, MatrixXd& diffusion) const;
    
    // calculate trap forces
    void computeForces(const State& state, double time, VectorXd& forces) const;
    
    // rng
    double getNormalRandom();

private:
    SimulationConfig config_;
    std::mt19937_64 rng_;
    std::normal_distribution<double> dist_;
};

} // namespace modern
