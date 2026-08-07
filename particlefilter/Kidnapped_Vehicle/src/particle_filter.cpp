#include "particle_filter.h"
#include <iostream>
#include <algorithm>

using namespace std;

// Constructor
ParticleFilter::ParticleFilter() {
    num_particles = 100;
    is_initialized = false;
}

// Destructor
ParticleFilter::~ParticleFilter() {}

// Initialize particles
void ParticleFilter::init(double x, double y, double theta, double std[]) {
    std_x = std[0];
    std_y = std[1];
    std_theta = std[2];

    normal_distribution<double> dist_x(x, std_x);
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> dist_theta(theta, std_theta);

    particles.clear();

    for (int i = 0; i < num_particles; i++) {
        Particle p;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1.0;
        particles.push_back(p);
    }

    is_initialized = true;
}

// Prediction step
void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {

    normal_distribution<double> noise_x(0, std_pos[0]);
    normal_distribution<double> noise_y(0, std_pos[1]);
    normal_distribution<double> noise_theta(0, std_pos[2]);

    for (int i = 0; i < num_particles; i++) {

        if (fabs(yaw_rate) > 0.0001) {
            particles[i].x += velocity / yaw_rate *
                (sin(particles[i].theta + yaw_rate * delta_t) -
                 sin(particles[i].theta));

            particles[i].y += velocity / yaw_rate *
                (-cos(particles[i].theta + yaw_rate * delta_t) +
                  cos(particles[i].theta));

            particles[i].theta += yaw_rate * delta_t;
        }
        else {
            particles[i].x += velocity * delta_t * cos(particles[i].theta);
            particles[i].y += velocity * delta_t * sin(particles[i].theta);
        }

        particles[i].x += noise_x(gen);
        particles[i].y += noise_y(gen);
        particles[i].theta += noise_theta(gen);
    }
}

// Update weights (simple version)
void ParticleFilter::updateWeights(double sensor_range,
                                   double std_landmark[],
                                   const vector<int>& observations,
                                   const Map& map_landmarks) {

    for (int i = 0; i < num_particles; i++) {
        particles[i].weight = 1.0;
    }
}

// Resample
void ParticleFilter::resample() {

    vector<double> weights;

    for (auto &p : particles)
        weights.push_back(p.weight);

    discrete_distribution<int> dist(weights.begin(), weights.end());

    vector<Particle> new_particles;

    for (int i = 0; i < num_particles; i++)
        new_particles.push_back(particles[dist(gen)]);

    particles = new_particles;
}

// Best particle
void ParticleFilter::getBestParticle(double &x,
                                     double &y,
                                     double &theta) {

    auto best = max_element(
        particles.begin(),
        particles.end(),
        [](Particle a, Particle b) {
            return a.weight < b.weight;
        });

    x = best->x;
    y = best->y;
    theta = best->theta;
}

// Write particle data
void ParticleFilter::write(string filename) {
    cout << "Writing particle data to " << filename << endl;
}

// Distance
double ParticleFilter::distance(double x1,
                                double y1,
                                double x2,
                                double y2) {

    return sqrt((x1-x2)*(x1-x2) +
                (y1-y2)*(y1-y2));
}

// Nearest landmark
int ParticleFilter::nearestLandmark(double x,
                                    double y,
                                    const Map& map_landmarks) {

    double minDist = 1e9;
    int index = -1;

    for (int i = 0; i < map_landmarks.landmark_x.size(); i++) {

        double d = distance(x, y,
                            map_landmarks.landmark_x[i],
                            map_landmarks.landmark_y[i]);

        if (d < minDist) {
            minDist = d;
            index = i;
        }
    }

    return index;
}