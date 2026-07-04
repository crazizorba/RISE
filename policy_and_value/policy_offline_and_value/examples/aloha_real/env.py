from typing import List, Optional  # noqa: UP035

import einops
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override

#from examples.aloha_real import real_env as _real_env


class AlohaRealEnvironment(_environment.Environment):
    """An environment for an Aloha robot on real hardware."""

    def __init__(
        self,
        reset_position: Optional[List[float]] = None,  # noqa: UP006,UP007
        render_height: int = 224,
        render_width: int = 224,
    ) -> None:
        #self._env = _real_env.make_real_env(init_node=True, reset_position=reset_position)
        self._render_height = render_height
        self._render_width = render_width

        self._ts = None

    @override
    def reset(self): #-> None:
        #self._ts = self._env.reset()
        import collections
        import dm_env
        import numpy as np
        import einops
        
        # 1. Tạo ảnh đen giả lập chuẩn kích thước (Height, Width, Channels)
        fake_image_hwc = np.zeros((self._render_height, self._render_width, 3), dtype=np.uint8)
        # Chuyển đổi sang (Channels, Height, Width) chuẩn OpenPI
        fake_image_chw = einops.rearrange(fake_image_hwc, "h w c -> c h w")
        
        fake_qpos = np.zeros(14, dtype=np.float32)

        # 2. Cấu trúc observation đồng bộ tuyệt đối
        obs_data = {
            "state": fake_qpos,
            "qpos": fake_qpos,
            "images": {
                "hand_left": fake_image_chw,
                "hand_right": fake_image_chw,
                "top_head": fake_image_chw
            },
        }

        TimeStep = collections.namedtuple('TimeStep', ['step_type', 'reward', 'discount', 'observation'])
        self._ts = TimeStep(step_type=0, reward=0.0, discount=1.0, observation=obs_data)
        return self._ts

    @override
    def step(self, action):# -> dm_env.TimeStep:
        return self._ts

    @override
    def is_episode_complete(self) -> bool:
        return False

    @override
    def get_observation(self) -> dict:
    #     if self._ts is None:
    #         raise RuntimeError("Timestep is not set. Call reset() first.")

    #     obs = self._ts.observation
    #     for k in list(obs["images"].keys()):
    #         if "_depth" in k:
    #             del obs["images"][k]

    #     for cam_name in obs["images"]:
    #         img = image_tools.convert_to_uint8(
    #             image_tools.resize_with_pad(obs["images"][cam_name], self._render_height, self._render_width)
    #         )
    #         obs["images"][cam_name] = einops.rearrange(img, "h w c -> c h w")

    #     return {
    #         "state": obs["qpos"],
    #         "images": obs["images"],
    #     }
        import numpy as np
        import einops
        
        fake_image_hwc = np.zeros((self._render_height, self._render_width, 3), dtype=np.uint8)
        fake_image_chw = einops.rearrange(fake_image_hwc, "h w c -> c h w")
        fake_qpos = np.zeros(14, dtype=np.float32)
        
        # Trả về y hệt như hàm reset ở trên
        return {
            "state": fake_qpos,
            "qpos": fake_qpos,
            "images": {
                "hand_left": fake_image_chw,
                "hand_right": fake_image_chw,
                "top_head": fake_image_chw
            },
        }

    @override
    def apply_action(self, action: dict) -> None:
        #self._ts = self._env.step(action["actions"])
        self._ts = self.step(action)
