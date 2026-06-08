#include "PhysicsEngine.h"
#include <cmath>
#include <iostream>

namespace modern {

PhysicsEngine::PhysicsEngine(const SimulationConfig& config)
    : config_(config), rng_(std::random_device{}()), dist_(0.0, 1.0) {}

void PhysicsEngine::computeHydrodynamicTensor(const State& state, MatrixXd& tensor) const {
    int N = config_.num_beads;
    double a = config_.bead_radius;
    double zeta = 6.0 * M_PI * config_.viscosity * a;
    double inv_zeta = 1.0 / zeta;
    
    tensor.setZero();
    
    for (int p = 0; p < N; ++p) {
        for (int q = 0; q < N; ++q) {
            if (p == q) {
                tensor.block<2, 2>(2 * p, 2 * q) = Matrix2d::Identity() * inv_zeta;
            } else {
                Vector2d dp = state.positions.segment<2>(2 * q) - state.positions.segment<2>(2 * p);
                double r2 = dp.squaredNorm();
                double r = std::sqrt(r2);
                
                Matrix2d I = Matrix2d::Identity();
                Matrix2d unit_rr = (dp * dp.transpose()) / r2;
                Matrix2d D;
                
                if (config_.use_rotne_prager) {
                    // Rotne-Prager
                    if (r >= 2 * a) {
                        D = (I + unit_rr + (I - 3.0 * unit_rr) * (2.0 * a * a / (3.0 * r2))) * (3.0 * a / (4.0 * zeta * r));
                    } else {
                        D = (I * (1.0 - 9.0 * r / (32.0 * a)) + unit_rr * (3.0 / (32.0 * a * r))) * inv_zeta;
                    }
                } else {
                    // Oseen
                    D = (I + unit_rr) * (1.0 / (8.0 * M_PI * config_.viscosity * r));
                }
                tensor.block<2, 2>(2 * p, 2 * q) = D;
            }
        }
    }
}

void PhysicsEngine::computeCholesky(const MatrixXd& tensor, MatrixXd& diffusion) const {
    Eigen::LLT<MatrixXd> llt(tensor);
    diffusion = llt.matrixL();
}

void PhysicsEngine::computeForces(const State& state, double time, VectorXd& forces) const {
    int N = config_.num_beads;
    forces.setZero();
    
    double F0 = config_.f0_signal;
    double Tsignal = config_.t_signal;
    double signal = (time > 5.0) ? F0 * (std::cos(2.0 * M_PI * time / Tsignal) > 0 ? 1.0 : -1.0) : 0.0;

    double i0 = (N - 1) / 2.0;

    for (int i = 0; i < N; ++i) {
        // current position relative to reference + offset
        double dx = state.positions(2 * i) - (state.reference_pos(2 * i) + state.trap_offsets(2 * i));
        double dy = state.positions(2 * i + 1) - (state.reference_pos(2 * i + 1) + state.trap_offsets(2 * i + 1));
        
        // calculate H factor
        double distance_ratio = 1.0;
        if (config_.geometry == GeometryType::CHAIN && N > 1) {
            double current_dist = config_.base_distance;
            if (i == 0) current_dist = config_.inter_bead_distances[0];
            else if (i == N - 1) current_dist = config_.inter_bead_distances[N - 2];
            else current_dist = 0.5 * (config_.inter_bead_distances[i-1] + config_.inter_bead_distances[i]);
            
            distance_ratio = std::pow(current_dist / config_.base_distance, 1.0/3.0);
        }

        double lambda = config_.lambdas[i];
        double C = 1.0 - 0.25 * lambda;
        double H = distance_ratio * (lambda * (i - i0) * (i - i0) + C);
        
        double kx = -state.stiffness(0) * H; 
        double ky = -state.stiffness(1) * H;
        
        double fx = 0, fy = 0;
        if (std::abs(dx) > 1e-18) fx = kx * dx * config_.alpha * std::pow(std::abs(dx), config_.alpha - 2);
        if (std::abs(dy) > 1e-18) fy = ky * dy * config_.beta * std::pow(std::abs(dy), config_.beta - 2);

        if (i == 0) fx += signal;
        
        forces(2 * i) = fx;
        forces(2 * i + 1) = fy;
    }
}

double PhysicsEngine::getNormalRandom() {
    return dist_(rng_);
}

} // namespace modern
