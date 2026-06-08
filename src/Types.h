#pragma once

#include <Eigen/Dense>
#include <vector>
#include <string>

namespace modern {

using Vector2d = Eigen::Vector2d;
using Matrix2d = Eigen::Matrix2d;
using Vector3d = Eigen::Vector3d;
using Matrix3d = Eigen::Matrix3d;
using VectorXd = Eigen::VectorXd;
using MatrixXd = Eigen::MatrixXd;

enum class GeometryType {
    RING,
    CHAIN,
    ARBITRARY
};

struct SimulationConfig {
    int num_beads;
    double amplitude;
    double base_distance;
    double bead_radius;
    double dt;
    double total_time;
    double viscosity;
    double alpha;
    double kx;
    double beta;
    double ky;
    double temperature;
    int sampling_feedback;
    int sampling_write;
    double epsilon;
    double velox;
    double ks;
    double fl0;
    double veloy;
    double f0_signal;
    double t_signal;
    double default_lambda;
    
    std::vector<double> inter_bead_distances; // For CHAIN: N-1 values
    std::vector<double> lambdas;              // N values
    std::vector<Vector2d> initial_positions;  // For ARBITRARY: N values
    
    GeometryType geometry;
    bool use_rotne_prager = true;
};

struct State {
    VectorXd positions;       // [x0, y0, x1, y1, ...]
    VectorXd trap_offsets;    // offsets from reference positions
    VectorXd reference_pos;   // fixed reference positions (eg Temp_d in CHAIN)
    Vector2d stiffness;       // [kx, ky]
    
    explicit State(int n) 
        : positions(2 * n), trap_offsets(2 * n), reference_pos(2 * n) {
        positions.setZero();
        trap_offsets.setZero();
        reference_pos.setZero();
    }
};

} // namespace modern
