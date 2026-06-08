#include "Simulation.h"
#include <iostream>
#include <sstream>
#include <filesystem>
#include <cmath>
#include <iomanip>

namespace modern {

static constexpr double KB = 1.3806504E-5;

Simulation::Simulation(const std::string& input_file, GeometryType geometry, bool use_rotne_prager)
    : input_filename_(input_file), state_(0), geometry_arg_(geometry) {
    parseInput(input_file);
    config_.use_rotne_prager = use_rotne_prager;
    initializeState();
    engine_ = std::make_unique<PhysicsEngine>(config_);
    
    int size = 2 * config_.num_beads;
    tensor_.resize(size, size);
    diffusion_.resize(size, size);
    forces_.resize(size);
    noise_.resize(size);
}

void Simulation::parseInput(const std::string& input_file) {
    std::ifstream is(input_file);
    if (!is) throw std::runtime_error("Could not open input file: " + input_file);
    
    std::string line;
    std::getline(is, line);
    std::istringstream iss(line);
    std::vector<double> p;
    double val;
    while (iss >> val) p.push_back(val);
    
    if (p.size() < 21) throw std::runtime_error("Not enough parameters in input file");
    
    config_.num_beads = static_cast<int>(p[0]);
    config_.amplitude = p[1];
    config_.base_distance = p[2];
    config_.bead_radius = p[3];
    config_.dt = p[4];
    config_.total_time = p[5];
    config_.viscosity = p[6];
    config_.alpha = p[7];
    config_.kx = p[8];
    config_.beta = p[9];
    config_.ky = p[10];
    config_.temperature = p[11];
    config_.sampling_feedback = static_cast<int>(p[12]);
    config_.sampling_write = static_cast<int>(p[13]);
    config_.epsilon = p[14];
    config_.velox = p[15];
    config_.ks = p[16];
    config_.fl0 = p[17];
    config_.veloy = p[18];
    config_.f0_signal = p[19];
    config_.t_signal = p[20];
    config_.default_lambda = p[21];
    config_.geometry = geometry_arg_;
    
    int N = config_.num_beads;
    int current = 22;
    if (config_.geometry == GeometryType::CHAIN) {
        for (int i = 0; i < N - 1; ++i) config_.inter_bead_distances.push_back(p[current++]);
    }
    
    // Lambdas
    for (int i = 0; i < N; ++i) {
        if (current < (int)p.size()) config_.lambdas.push_back(p[current++]);
        else config_.lambdas.push_back(config_.default_lambda);
    }
    
    if (config_.geometry == GeometryType::ARBITRARY) {
        for (int i = 0; i < N; ++i) {
            double x = p[current++];
            double y = p[current++];
            config_.initial_positions.push_back({x, y});
        }
    }
}

void Simulation::initializeState() {
    int N = config_.num_beads;
    state_ = State(N);
    state_.stiffness = {config_.kx, config_.ky};
    
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(-0.5, 0.5);

    if (config_.geometry == GeometryType::CHAIN) {
        double current_x = 0;
        for (int i = 0; i < N; ++i) {
            state_.reference_pos(2 * i) = current_x;
            state_.reference_pos(2 * i + 1) = 0;
            
            double noise = dist(rng) * config_.amplitude;
            state_.positions(2 * i) = current_x + noise;
            state_.positions(2 * i + 1) = 0;
            
            double side = (dist(rng) > 0) ? 1.0 : -1.0;
            state_.trap_offsets(2 * i) = side * (config_.amplitude / 2.0 + config_.epsilon);
            state_.trap_offsets(2 * i + 1) = 0;
            
            if (i < N - 1) current_x += config_.inter_bead_distances[i];
        }
    } else if (config_.geometry == GeometryType::ARBITRARY) {
        for (int i = 0; i < N; ++i) {
            Vector2d pos = config_.initial_positions[i];
            state_.reference_pos(2 * i) = pos.x();
            state_.reference_pos(2 * i + 1) = pos.y();
            
            double noise = dist(rng) * config_.amplitude;
            state_.positions(2 * i) = pos.x() + noise;
            state_.positions(2 * i + 1) = pos.y();
            
            double side = (dist(rng) > 0) ? 1.0 : -1.0;
            state_.trap_offsets(2 * i) = side * (config_.amplitude / 2.0 + config_.epsilon);
            state_.trap_offsets(2 * i + 1) = 0;
        }
    } else if (config_.geometry == GeometryType::RING) {
        double R = config_.base_distance / (2.0 * std::sin(M_PI / N));
        for (int i = 0; i < N; ++i) {
            double angle = 2.0 * M_PI * i / N;
            double rx = R * std::cos(angle);
            double ry = R * std::sin(angle);
            state_.reference_pos(2 * i) = rx;
            state_.reference_pos(2 * i + 1) = ry;
            
            double noise = dist(rng) * config_.amplitude;
            state_.positions(2 * i) = rx + noise;
            state_.positions(2 * i + 1) = ry;
            
            double side = (dist(rng) > 0) ? 1.0 : -1.0;
            state_.trap_offsets(2 * i) = side * (config_.amplitude / 2.0 + config_.epsilon);
            state_.trap_offsets(2 * i + 1) = 0;
        }
    }
}

void Simulation::run() {
    std::string title = generateTitle();
    std::filesystem::path p(input_filename_);
    std::string dir_name = p.stem().string(); // Strips extension
    if (dir_name.empty()) dir_name = "results";
    
    std::filesystem::create_directories(dir_name);
    
    std::ofstream out_dat(dir_name + "/" + title + ".dat");
    std::ofstream out_trap(dir_name + "/" + title + ".trap");
    std::ofstream out_force(dir_name + "/" + title + ".forces");
    
    double time = 0;
    int step_count = 0;
    int frame = 0;
    
    std::cout << "Starting Simulation: " << title << " in directory: " << dir_name << std::endl;
    
    while (time < config_.total_time) {
        if (step_count % config_.sampling_feedback == 0) {
            activeTrapping();
            frame++;
        }
        
        if (step_count % config_.sampling_write == 0) {
            saveData(out_dat, frame);
            saveTraps(out_trap, frame);
            saveForces(out_force, frame);
        }
        
        step(time);
        
        time += config_.dt;
        step_count++;
        
        if (step_count % 100000 == 0) {
            std::cout << std::fixed << std::setprecision(1) 
                      << 100.0 * time / config_.total_time << "% done" << std::endl;
        }
    }
}

void Simulation::step(double time) {
    engine_->computeHydrodynamicTensor(state_, tensor_);
    engine_->computeForces(state_, time, forces_);
    
    if (config_.temperature > 0) {
        MatrixXd diff_tensor = tensor_ * (KB * config_.temperature);
        engine_->computeCholesky(diff_tensor, diffusion_);
        
        for (int i = 0; i < noise_.size(); ++i) {
            noise_(i) = engine_->getNormalRandom() * std::sqrt(2.0 * config_.dt);
        }
        noise_ = diffusion_ * noise_;
    } else {
        noise_.setZero();
    }
    
    VectorXd flow(2 * config_.num_beads);
    for (int i = 0; i < config_.num_beads; ++i) {
        flow(2 * i) = config_.velox;
        flow(2 * i + 1) = config_.veloy;
    }
    
    state_.positions += (flow * config_.dt) + (tensor_ * forces_ * config_.dt) + noise_;
}

void Simulation::activeTrapping() {
    int N = config_.num_beads;
    for (int i = 0; i < N; ++i) {
        // displacement relative to the reference position
        double dx = state_.positions(2 * i) - state_.reference_pos(2 * i);
        
        if (dx > (config_.amplitude / 2.0)) {
            state_.trap_offsets(2 * i) = -(config_.amplitude / 2.0 + config_.epsilon);
        } else if (dx < -(config_.amplitude / 2.0)) {
            state_.trap_offsets(2 * i) = (config_.amplitude / 2.0 + config_.epsilon);
        }
    }
}

void Simulation::saveData(std::ostream& out, int frame) {
    out.precision(9);
    out << frame << " " << (double)frame * config_.dt * config_.sampling_feedback << " ";
    out.precision(6);
    for (int i = 0; i < state_.positions.size(); ++i) {
        out << state_.positions(i) << " ";
    }
    out << "\n";
}

void Simulation::saveTraps(std::ostream& out, int frame) {
    out << frame << " ";
    for (int i = 0; i < state_.trap_offsets.size(); ++i) {
        out << state_.reference_pos(i) + state_.trap_offsets(i) << " ";
    }
    out << "\n";
}

void Simulation::saveForces(std::ostream& out, int frame) {
    out.precision(9);
    out << frame << " " << (double)frame * config_.dt * config_.sampling_feedback << " ";
    out.precision(6);
    for (int i = 0; i < forces_.size(); ++i) {
        out << forces_(i) << " ";
    }
    out << "\n";
}

std::string Simulation::generateTitle() const {
    std::filesystem::path p(input_filename_);
    return p.stem().string();
}

} // namespace modern
