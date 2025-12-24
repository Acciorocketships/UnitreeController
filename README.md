## Installation
 1. Install Unitree SDK2py package with these [setup instructions](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/doc/setup_en.md).
 2. Clone this repo and install its requirements (torch, numpy, opencv, ...)

## Running
1. Put the robot in READY mode (this is for high-level control, for low-level it needs to be in debug mode)
2. Connect your to your robot via wifi or ethernet. Note the network interface (e.g. en8)
3. Update `net = "en8"` at the bottom of `run_highlevel_policy.py` to your network interface
4. Change the policy at the bottom of `run_highlevel_policy.py` to your desired policy. To debug and make sure the robot works without a policy, change `controller.run()` (which runs the given policy) to `controller.run_debug_forward()`, which gives a short movement command.
5. run `python run_highlevel_policy.py`

## Custom Policies
- To implement your own policy, create a class that extends BasePolicy, and override the forward() method. Look at ManualPolicy for reference.
- The `forward()` method takes in an observation, and outputs an action.
- By default, the observation is both the image from the robot's camera and the low-level state (imu roll pitch yaw, imu gyroscope, motor positions and velocities). However, these can be turned off in the config in `run_highlevel_policy` if they are not needed for extra speed.
- The output is [vx, vy, vyaw], which is the velocity in the forward direction (vx), the velocity in the left/right direction (vy), and the angular velocity for rotating (vyaw).


