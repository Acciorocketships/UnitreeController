from typing import Optional
import argparse
import os
import time
import numpy as np
import torch
import numpy as np
import cv2 as cv2
from dataclasses import dataclass

from unitree_sdk2py.go2.video.video_client import VideoClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG

from policies.BasePolicy import BasePolicy
from policies.ManualPolicy import ManualPolicy
 

# Config parameters for the robot/controller
@dataclass
class SimpleConfig:
	# Frequency of the run() control loop (s)
	control_dt: float = 0.02
	# Velocity caps [vx, vy, vyaw]
	max_cmd: np.ndarray = np.array([0.6, 0.4, 1.2], dtype=np.float32)
	# Observation selection
	obs_image: bool = True
	obs_state: bool = True
	# Low-level state subscription
	lowstate_topic: str = "rt/lowstate"


class HighLevelController:
	def __init__(self, config: SimpleConfig, net: str, policy: BasePolicy) -> None:
		# Local variables
		self.config = config
		self.policy = policy
		self._lowstate_sub = None
		self._low_state_msg = None
		# high-level sport state not used; rely on lowstate topic

		# Init network
		try:
			ChannelFactoryInitialize(0, net)
		except Exception as err:
			print(f"Could not find robot's network interface on {net}\n{err}")
			exit(1)

		# Init locomotion
		self.client = LocoClient()
		self.client.SetTimeout(5.0)
		self.client.Init()
		self.client.Start()

		# Init video feed
		if self.config.obs_image:
			self.vc = VideoClient()
			self.vc.SetTimeout(2.0)
			self.vc.Init()
		
		# Init lowstate subscriber if requested
		if self.config.obs_state:
			self._lowstate_sub = ChannelSubscriber(self.config.lowstate_topic, LowStateHG)
			self._lowstate_sub.Init(self._on_lowstate, 10)

	def run(self):
		# The main control loop.
		# - Gets the observation
		# - Passes the observation to the policy, which returns an action
		# - Sends that action to the robot
		time_last = time.time()
		while True:
			image = self.get_image() if self.config.obs_image else None
			state = self.get_state() if self.config.obs_state else None
			with torch.no_grad():
				vel = self.policy(image, state).cpu().numpy().reshape(-1)
			vx, vy, vyaw = float(vel[0]), float(vel[1]), float(vel[2])
			# Clip to configured limits
			vx = float(np.clip(vx, -self.config.max_cmd[0], self.config.max_cmd[0]))
			vy = float(np.clip(vy, -self.config.max_cmd[1], self.config.max_cmd[1]))
			vyaw = float(np.clip(vyaw, -self.config.max_cmd[2], self.config.max_cmd[2]))

			# Set the velocity to the 
			self.set_velocity(vx, vy, vyaw)

			# Sleep so the loop time is exactly control_dt
			dt = time.time() - time_last
			time.sleep(max(0, self.config.control_dt - dt))
			time_last = time.time()

	def run_debug_forward(self, vx: float = 0.3, vy: float = 0.0, vyaw: float = 0.0, seconds: float = 0.0):
		# For debugging, use controller.run_debug_forward(vx, vy, vyaw, seconds)
		# to check if movement works without a policy
		print(f"Debug forward: vx={vx:.3f} m/s, vy={vy:.3f} m/s, vyaw={vyaw:.3f} rad/s")
		start_t = time.time()
		while True:
			self.set_velocity(vx, vy, vyaw)
			if seconds > 0.0 and (time.time() - start_t) >= seconds:
				break
			time.sleep(self.config.control_dt)
		# Stop command
		self.client.Move(0.0, 0.0, 0.0, continous_move=False)

	def set_velocity(self, vx: float, vy: float, vyaw: float):
		# Sends the velocity command to the robot
		# vx: forward/backward velocity
		# vy: left/right velocity
		# vyaw: yaw velocity
		try:
			code = self.client.SetVelocity(vx, vy, vyaw, duration=1.0)
			if code not in (0, None):
				print(f"[loco] SetVelocity({vx:.3f},{vy:.3f},{vyaw:.3f}) -> code={code}")
		except Exception as e:
			print(f"[loco] SetVelocity error: {e}")

	def get_image(self) -> Optional[np.ndarray]:
		# Gets the current image from the robot's camera
		code, data = self.vc.GetImageSample()
		if code == 0 and data is not None:
			arr = np.frombuffer(bytes(data), dtype=np.uint8)
			img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
			return img
		return None

	def _on_lowstate(self, msg):
		self._low_state_msg = msg

	def get_state(self) -> Optional[dict]:
		# Returns the low-level state of the robot
		msg = self._low_state_msg
		if msg is None:
			return {}
		imu = msg.imu_state
		state = {
			"imu_gyroscope": [float(v) for v in imu.gyroscope],
			"imu_rpy": [float(v) for v in imu.rpy],
		}
		# Motor joint positions and velocities
		motor_q = []
		motor_dq = []
		for m in msg.motor_state:
			motor_q.append(float(m.q))
			motor_dq.append(float(m.dq))
		state["motor_q"] = motor_q
		state["motor_dq"] = motor_dq
		return state


def main():
	# Change this to the network interface of the robot. It defaults to en8 (ethernet interface 8)
	net = "en8"

	# Change this to the policy you want to use. It defaults to ManualPolicy (keyboard control with video feed display)
	policy = ManualPolicy()

	# Config parameters for the robot/controller
	config = SimpleConfig()

	# Initialize the controller
	controller = HighLevelController(config, net, policy)

	# Run the main control loop of thecontroller
	controller.run()


if __name__ == "__main__":
	main()


