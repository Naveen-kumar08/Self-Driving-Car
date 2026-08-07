#ifndef PARTICLE_FILTER_H
#define PARTICLE_FILTER_H

#include <vector>

using namespace std;

// Particle structure
struct Particle {
    int id;
    double x;
    double y;
    double theta;
    double weight;
};

class ParticleFilter {
public:
    // Number of particles
    int num_particles;

    // List of particles
    vector<Particle> particles;

    // Constructor
    ParticleFilter();

    // Initialize particles
    void init(double x, double y, double theta, double std[]);

    // Predict particle positions
    void prediction(double delta_t,
                    double std_pos[],
                    double velocity,
                    double yaw_rate);

    // Update particle weights
    void updateWeights(double landmark_x,
                       double landmark_y,
                       double sensor_distance);

    // Resample particles
    void resample();

    // Get best particle
    void getBestParticle(double &x,
                         double &y,
                         double &theta);
};

#endif