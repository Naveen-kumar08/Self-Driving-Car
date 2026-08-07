#include "particle_filter.h"

#include <random>
#include <cmath>

using namespace std;

default_random_engine gen(42);

ParticleFilter::ParticleFilter()
{
    num_particles = 100;
}

void ParticleFilter::init(double x,double y,double theta,double std[])
{
    normal_distribution<double> dx(x,std[0]);
    normal_distribution<double> dy(y,std[1]);
    normal_distribution<double> dt(theta,std[2]);

    particles.clear();

    for(int i=0;i<num_particles;i++)
    {
        Particle p;

        p.id=i;
        p.x=dx(gen);
        p.y=dy(gen);
        p.theta=dt(gen);
        p.weight=1.0;

        particles.push_back(p);
    }
}

void ParticleFilter::prediction(double delta_t,double std_pos[],
                                double velocity,double yaw_rate)
{
    for(auto &p:particles)
    {
        p.x += velocity*delta_t*cos(p.theta);
        p.y += velocity*delta_t*sin(p.theta);
        p.theta += yaw_rate*delta_t;
    }
}

void ParticleFilter::updateWeights(double landmark_x,
                                   double landmark_y,
                                   double sensor_dist)
{
    for(auto &p:particles)
    {
        double d=sqrt((landmark_x-p.x)*(landmark_x-p.x)+
                      (landmark_y-p.y)*(landmark_y-p.y));

        p.weight=1.0/(1.0+fabs(d-sensor_dist));
    }
}

void ParticleFilter::resample()
{
    vector<double> weights;

    for(auto &p:particles)
        weights.push_back(p.weight);

    discrete_distribution<int> dist(weights.begin(),weights.end());

    vector<Particle> newParticles;

    for(int i=0;i<num_particles;i++)
        newParticles.push_back(particles[dist(gen)]);

    particles=newParticles;
}