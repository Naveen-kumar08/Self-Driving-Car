#include <iostream>
#include <vector>
#include "particle_filter.h"

using namespace std;

int main() {
    cout << "=== Kidnapped Vehicle Particle Filter ===" << endl;
    
    // Initialize particle filter
    ParticleFilter pf;
    
    // Initial GPS measurement
    double gps_x = 100.0;
    double gps_y = 100.0;
    double gps_theta = 0.0;
    double gps_std[] = {3.0, 3.0, 0.03};
    
    pf.init(gps_x, gps_y, gps_theta, gps_std);
    cout << "Initialized with " << pf.num_particles << " particles" << endl;
    
    // Simulate motion
    double delta_t = 0.1;
    double velocity = 10.0;
    double yaw_rate = 0.1;
    double std_pos[] = {0.3, 0.3, 0.01};
    
    cout << "\nRunning 10 prediction steps..." << endl;
    for (int i = 0; i < 10; i++) {
        pf.prediction(delta_t, std_pos, velocity, yaw_rate);
    }
    
    // Get best estimate
    double best_x, best_y, best_theta;
    pf.getBestParticle(best_x, best_y, best_theta);
    
    cout << "\nBest particle estimate:" << endl;
    cout << "X: " << best_x << endl;
    cout << "Y: " << best_y << endl;
    cout << "Theta: " << best_theta << endl;
    
    cout << "\n✅ Particle filter test completed!" << endl;
    
    return 0;
}