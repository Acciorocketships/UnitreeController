from policies.BasePolicy import BasePolicy
import torch
from typing import Optional, Set
import numpy as np
import cv2
from pynput import keyboard


class ManualPolicy(BasePolicy):
	"""
	Manual teleop policy with two input paths:
	- Preferred: pynput keyboard listener (non-blocking, works without OpenCV windows)
	- Fallback: cv2.waitKey(1) if pynput is unavailable
	Also displays the camera stream if OpenCV can create a window.
	"""

	def __init__(self, max_vx: float = 0.6, max_vy: float = 0.4, max_vyaw: float = 1.2, use_camera: bool = True):
		super().__init__()
		self._current = torch.zeros(3, dtype=torch.float32)
		with torch.no_grad():
			self.max_vel = torch.tensor([max_vx, max_vy, max_vyaw], dtype=torch.float32)
		self._use_camera = use_camera
		# Keyboard listener state
		self._pynput_ok = False
		self._pressed: Set[str] = set()
		try:

			def on_press(key):
				try:
					self._pressed.add(key.char.lower())
				except Exception:
					if hasattr(key, "name") and key.name:
						self._pressed.add(key.name.lower())

			def on_release(key):
				try:
					self._pressed.discard(key.char.lower())
				except Exception:
					if hasattr(key, "name") and key.name:
						self._pressed.discard(key.name.lower())

			self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
			self._listener.daemon = True
			self._listener.start()
			self._pynput_ok = True
		except Exception as e:
			print(f"Error initializing pynput: {e}")
			self._pynput_ok = False
		# OpenCV window flag
		self._cv_inited = False

	def get_control_from_keyboard(self):
		if not self._pynput_ok:
			return torch.zeros(3, dtype=torch.float32)
		vx, vy, vyaw = float(self._current[0]), float(self._current[1]), float(self._current[2])
		if "w" in self._pressed:
			vx = float(self.max_vel[0])
		if "s" in self._pressed:
			vx = -float(self.max_vel[0])
		if "a" in self._pressed:
			vy = float(self.max_vel[1])
		if "d" in self._pressed:
			vy = -float(self.max_vel[1])
		if "q" in self._pressed:
			vyaw = float(self.max_vel[2])
		if "e" in self._pressed:
			vyaw = -float(self.max_vel[2])
		if "x" in self._pressed:
			vx, vy, vyaw = 0.0, 0.0, 0.0
		return torch.tensor([vx, vy, vyaw], dtype=torch.float32)

	def forward(self, image: Optional["np.ndarray"], state: Optional[dict]) -> torch.Tensor:
		# Get control from keyboard
		control = self.get_control_from_keyboard()

		# Show camera image
		if image is not None and self._use_camera:
		    if not self._cv_inited:
		        cv2.namedWindow("unitree_camera", cv2.WINDOW_AUTOSIZE)
		        self._cv_inited = True
		    cv2.imshow("unitree_camera", image)
		key = cv2.waitKey(1)

		# Return the control vector as the action
		return control

