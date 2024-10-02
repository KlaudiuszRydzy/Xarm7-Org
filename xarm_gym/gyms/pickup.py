import gymnasium as gym
from base import LUCBase
from gymnasium import env


@env.register("luc-pickup-v0.0")
class Pickup(LUCBase):
    def __init__(self):
        super().__init__()

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(64, 64, 3), dtype=np.uint8
        )
        self._state = np.zeros((64, 64, 3), dtype=np.uint8)

    def reset(self):
        self._state = np.zeros((64, 64, 3), dtype=np.uint8)

        return self._state

    def _step(self, action):
        self._state = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        return self._state, 0, False, False, {}

    def render(self, mode="human"):
        pass
