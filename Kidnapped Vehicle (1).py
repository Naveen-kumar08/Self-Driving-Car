import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ==============================================================================
# PART 1: THE PID CONTROLLER CLASS
# Built to be more robust, as you'd use in a real system.
# ==============================================================================
class PID:
    def __init__(self, Kp, Ki, Kd, output_limits=(-1, 1)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        
        self.min_output, self.max_output = output_limits
        
        self.p_error = 0.0
        self.i_error = 0.0
        self.d_error = 0.0
        
        self.last_cte = 0.0
        self.last_time = None

    def update(self, cte, current_time):
        """
        Calculates PID error terms based on the time difference between updates.
        This is crucial for a real-world (or realistic simulation) scenario.
        """
        if self.last_time is None:
            self.last_time = current_time
            delta_t = 0.0
        else:
            delta_t = current_time - self.last_time
        
        # Avoid division by zero if time hasn't passed
        if delta_t == 0:
            return 0.0

        # Proportional error
        self.p_error = cte

        # Integral error (with anti-windup)
        self.i_error += cte * delta_t
        # Clamp integral term to prevent it from growing out of control
        self.i_error = max(min(self.i_error, self.max_output), self.min_output)

        # Derivative error
        self.d_error = (cte - self.last_cte) / delta_t
        
        # Save current state for next iteration
        self.last_cte = cte
        self.last_time = current_time

        # Calculate the PID output
        output = (self.Kp * self.p_error) + \
                 (self.Ki * self.i_error) + \
                 (self.Kd * self.d_error)
        
        # Clamp the final output to steering limits
        return max(min(output, self.max_output), self.min_output)


# ==============================================================================
# PART 2: A MORE REALISTIC CAR SIMULATOR
# This class simulates a car with position, speed, and turning physics.
# ==============================================================================
class CarSimulator:
    def __init__(self, initial_x=0, initial_y=5, initial_angle=0, speed=10):
        self.x = initial_x             # Car's position (x-axis)
        self.y = initial_y             # Car's position (y-axis, this is our CTE)
        self.angle = initial_angle     # Car's heading
        self.speed = speed             # Car's speed in m/s
        self.max_steer_angle = np.pi / 4  # Max steering angle (45 degrees)

    def step(self, steering_angle, delta_t):
        """
        Updates the car's state based on the steering angle and time step.
        Uses a simple bicycle model.
        """
        # Clamp steering angle
        steering_angle = max(min(steering_angle, self.max_steer_angle), -self.max_steer_angle)

        # Update angle
        self.angle += steering_angle * delta_t
        
        # Update position
        self.x += self.speed * np.cos(self.angle) * delta_t
        self.y += self.speed * np.sin(self.angle) * delta_t
        
        # In this simulation, the Cross-Track Error (CTE) is simply the y-position.
        return self.y


# ==============================================================================
# PART 3: THE MAIN SIMULATION LOOP AND VISUALIZATION
# This orchestrates the simulation and creates an animated plot.
# ==============================================================================
if __name__ == '__main__':
    # --- TUNING PARAMETERS ---
    # These are the ONLY values you should need to change for your experiments.
    # A good starting point for this more realistic simulator.
    KP = -0.1   # Proportional gain
    KI = -0.002 # Integral gain
    KD = -1.5   # Derivative gain

    # Note: Gains are negative because a positive CTE (car is "above" the line)
    # requires a negative angle to steer back down.

    # --- SIMULATION SETUP ---
    SIMULATION_TIME = 20  # seconds
    DELTA_T = 0.1         # time step in seconds
    INITIAL_CTE = 5.0     # Start 5 meters off the center line
    CAR_SPEED = 20        # m/s

    # Initialize our PID controller and car simulator
    pid = PID(Kp=KP, Ki=KI, Kd=KD, output_limits=(-np.pi / 4, np.pi / 4))
    car = CarSimulator(initial_y=INITIAL_CTE, speed=CAR_SPEED)

    # Store data for plotting
    time_points = []
    cte_history = []
    car_positions = []

    # --- RUN THE SIMULATION ---
    print("Running simulation...")
    current_time = 0.0
    while current_time < SIMULATION_TIME:
        # Get the current CTE (car's y-position)
        current_cte = car.y
        
        # Get the steering angle from the PID controller
        steer_angle = pid.update(current_cte, current_time)
        
        # Move the car one step
        car.step(steer_angle, DELTA_T)
        
        # Record data
        time_points.append(current_time)
        cte_history.append(current_cte)
        car_positions.append((car.x, car.y))
        
        current_time += DELTA_T

    print("✅ Simulation complete. Generating visualization.")

    # --- VISUALIZATION ---
    fig = plt.figure(figsize=(15, 10))
    
    # Subplot 1: Cross-Track Error vs. Time
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(time_points, cte_history, label='Cross-Track Error (CTE)')
    ax1.axhline(0, color='r', linestyle='--', label='Center Line')
    ax1.set_title(f'PID Controller Performance\nKp={KP}, Ki={KI}, Kd={KD}')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('CTE (meters)')
    ax1.legend()
    ax1.grid(True)

    # Subplot 2: Animated Car Path
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.set_title('Vehicle Path')
    ax2.set_xlabel('X Position (meters)')
    ax2.set_ylabel('Y Position (CTE, meters)')
    ax2.axhline(0, color='r', linestyle='--', label='Center of Road')
    ax2.grid(True)

    x_coords, y_coords = zip(*car_positions)
    ax2.set_xlim(min(x_coords) - 5, max(x_coords) + 5)
    ax2.set_ylim(min(y_coords) - 5, max(y_coords) + 5)

    path_line, = ax2.plot([], [], 'b-', label='Car Path')
    car_marker, = ax2.plot([], [], 'go', markersize=10, label='Car')
    
    def init():
        path_line.set_data([], [])
        car_marker.set_data([], [])
        return path_line, car_marker,

    def animate(i):
        path_line.set_data(x_coords[:i], y_coords[:i])
        # This is the corrected line:
        car_marker.set_data([x_coords[i]], [y_coords[i]])
        return path_line, car_marker,

    ani = FuncAnimation(fig, animate, frames=len(car_positions),
                        init_func=init, blit=True, interval=DELTA_T * 1000)

    ax2.legend()
    plt.tight_layout()
    plt.show()
