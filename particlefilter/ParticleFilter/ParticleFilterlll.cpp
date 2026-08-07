#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

using namespace std;

struct Particle {
    int id;
    double x, y, theta, weight;
};

int main() {
    // --- STAGE 1: INITIALIZATION ---
    cout << "--- Particle Filter: Stage 1 (Initialization) ---" << endl;

    double gps_x = 1.2, gps_y = 0.5, gps_theta = 0.0;
    int num_particles = 100;
    vector<Particle> particles;

    // We use a fixed seed (42) to ensure you get consistent "random" numbers
    default_random_engine gen(42);
    normal_distribution<double> dist_x(gps_x, 0.3);
    normal_distribution<double> dist_y(gps_y, 0.3);
    normal_distribution<double> dist_theta(gps_theta, 0.01);

    for (int i = 0; i < num_particles; ++i) {
        particles.push_back({ i, dist_x(gen), dist_y(gen), dist_theta(gen), 1.0 });
    }

    cout << "SUCCESS: Created a cloud of " << particles.size() << " particles." << endl;
    cout << "Particle 0 -> X: " << particles[0].x << ", Y: " << particles[0].y << endl;
    cout << "Particle 1 -> X: " << particles[1].x << ", Y: " << particles[1].y << endl;
    cout << "Particle 2 -> X: " << particles[2].x << ", Y: " << particles[2].y << endl;


    // --- STAGE 2: PREDICTION ---
    cout << "\n--- Particle Filter: Stage 2 (Prediction) ---" << endl;
    double delta_t = 1.0, velocity = 5.0, yaw_rate = 0.1;

    cout << "Car is moving... updating particles." << endl;
    for (auto& p : particles) {
        p.x += velocity * delta_t * cos(p.theta);
        p.y += velocity * delta_t * sin(p.theta);
        p.theta += yaw_rate * delta_t;
    }

    cout << "SUCCESS: All particles moved to new predicted positions." << endl;
    cout << "New Particle 0 Position -> X: " << particles[0].x << ", Y: " << particles[0].y << endl;


    // --- STAGE 3: WEIGHTS ---
    cout << "\n--- Particle Filter: Stage 3 (Weights) ---" << endl;
    double sensor_dist = 8.0, landmark_x = 10.0, landmark_y = 10.0;

    cout << "Comparing particles to landmarks..." << endl;
    for (auto& p : particles) {
        double dist = sqrt(pow(landmark_x - p.x, 2) + pow(landmark_y - p.y, 2));
        p.weight = 1.0 / (1.0 + abs(dist - sensor_dist));
    }

    cout << "SUCCESS: All particles have been weighted." << endl;
    cout << "Particle 0 Weight (Score): " << particles[0].weight << endl;


    // --- STAGE 4: RESAMPLING ---
    cout << "\n--- Particle Filter: Stage 4 (Resampling) ---" << endl;
    vector<double> weights;
    for (const auto& p : particles) weights.push_back(p.weight);

    discrete_distribution<int> d(weights.begin(), weights.end());
    vector<Particle> resampled;
    for (int i = 0; i < num_particles; ++i) {
        resampled.push_back(particles[d(gen)]);
    }
    particles = resampled;

    cout << "SUCCESS: Resampling complete." << endl;
    cout << "The cloud has converged! All 100 particles are now high-probability guesses." << endl;
    cout << "Final Particle 0 Position -> X: " << particles[0].x << ", Y: " << particles[0].y << endl;

    return 0;
}