#ifndef PARTICLE_FILTER_H
#define PARTICLE_FILTER_H

#include <vector>

struct Particle
{
    int id;
    double x;
    double y;
    double theta;
    double weight;
};

class ParticleFilter
{
public:
    int num_particles;
    std::vector<Particle> particles;

    ParticleFilter();

    void init(double x,double y,double theta,double std[]);
    void prediction(double delta_t,double std_pos[],
                    double velocity,double yaw_rate);
    void updateWeights(double landmark_x,double landmark_y,double sensor_dist);
    void resample();
};

#endif