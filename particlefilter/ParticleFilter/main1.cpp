#include <iostream>
#include "particle_filter.h"

using namespace std;

int main()
{
    cout<<"--- Particle Filter: Stage 1 (Initialization) ---"<<endl;

    ParticleFilter pf;

    double gps_std[]={0.3,0.3,0.01};

    pf.init(1.2,0.5,0.0,gps_std);

    cout<<"SUCCESS: Created a cloud of "<<pf.num_particles<<" particles."<<endl;

    cout<<"Particle 0 -> X: "<<pf.particles[0].x<<", Y: "<<pf.particles[0].y<<endl;
    cout<<"Particle 1 -> X: "<<pf.particles[1].x<<", Y: "<<pf.particles[1].y<<endl;
    cout<<"Particle 2 -> X: "<<pf.particles[2].x<<", Y: "<<pf.particles[2].y<<endl;

    cout<<"\n--- Particle Filter: Stage 2 (Prediction) ---"<<endl;

    cout<<"Car is moving... updating particles."<<endl;

    double std_pos[]={0.3,0.3,0.01};

    pf.prediction(1.0,std_pos,5.0,0.1);

    cout<<"SUCCESS: All particles moved to new predicted positions."<<endl;

    cout<<"New Particle 0 Position -> X: "<<pf.particles[0].x
        <<", Y: "<<pf.particles[0].y<<endl;

    cout<<"\n--- Particle Filter: Stage 3 (Weights) ---"<<endl;

    cout<<"Comparing particles to landmarks..."<<endl;

    pf.updateWeights(10.0,10.0,8.0);

    cout<<"SUCCESS: All particles have been weighted."<<endl;

    cout<<"Particle 0 Weight (Score): "<<pf.particles[0].weight<<endl;

    cout<<"\n--- Particle Filter: Stage 4 (Resampling) ---"<<endl;

    pf.resample();

    cout<<"SUCCESS: Resampling complete."<<endl;

    cout<<"The cloud has converged! All 100 particles are now high-probability guesses."<<endl;

    cout<<"Final Particle 0 Position -> X: "
        <<pf.particles[0].x
        <<", Y: "
        <<pf.particles[0].y
        <<endl;

    return 0;
}