from sheeprl.envs.turtlebot_rooms_room_reward import Env
import time
#['image', 'lidar', 'world_info'] 


from typing import Any, Dict, List, Optional, Sequence, SupportsFloat, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import RenderFrame


class WebotsWrapper(gym.Wrapper):
    def __init__(self) -> None:
        
        seed = 42
        # image size: (270, 480, 3) 
        # ['image', 'lidar', 'world_info'
        observation_space = ['rgb', "world_info", "lidar"]  # 'lidar' , 'actions' , 'world_info', 'image'
        env = Env(seed=3, action_space='discrete', observation_space=observation_space, kb_control=False)
        super().__init__(env)
        self.observation_space = self.env.observation_space
        self.action_space = spaces.Discrete(self.env.action_space.n)
        self.reward_range = self.env.reward_range or (-np.inf, np.inf)
        self.observation_space.seed(seed)
        self.action_space.seed(seed)
        # render
        self._render_mode: str = "rgb_array"
        # metadata
        self._metadata = {"render_fps": 30}

    @property
    def render_mode(self) -> str | None:
        return self._render_mode

    def _convert_obs(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
        return {"rgb": obs["rgb"], "world_info": obs["world_info"], "lidar": obs["lidar"]}

    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, reward, done, truncated, info = self.env.step(action)

        #print(obs["rgb"].shape)
        return self._convert_obs(obs), reward, done, truncated, info

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        obs, _ = self.env.reset()
        return self._convert_obs(obs), {}

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        return self.env.render()

    def close(self) -> None:
        return
    

# env = WebotsWrapper("webots", (270, 480, 3))
# env.reset()

# for i in range(100):
#     env.step(0)
#     time.sleep(2)