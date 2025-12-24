from typing import Optional
import torch
from torch import nn


class BasePolicy(nn.Module):
 
	def __init__(self):
		# All of the setup code for the policy goes here
		super().__init__()
 
	def forward(self, image: Optional["np.ndarray"], state: Optional[dict]) -> torch.Tensor:
		# This is the function that defines the policy, which is called in the main control loop
		# The inputs are the current observations from the robot
		# - image: camera frame (BGR uint8 HxWx3) or None
		# - state: dict with keys like fsm_id, balance_mode, stand_height, swing_height
		# The output is the action for the robot to take in the next timestep.
		# The output is a tensor of shape [3] with [vx, vy, vyaw].
		# vx: forward/backward velocity
		# vy: left/right velocity
		# vyaw: yaw velocity
		return torch.zeros(3, dtype=torch.float32)