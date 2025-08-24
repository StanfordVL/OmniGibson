import logging
import torch as th
from omnigibson.learning.utils.array_tensor_utils import torch_to_numpy
from omnigibson.learning.utils.network_utils import WebsocketClientPolicy
from typing import Optional


class LocalPolicy:
    """
    Local policy that directly queries action from policy,
        outputs zero delta action if policy is None.
    """

    def __init__(self, *args, action_dim: Optional[int] = None, **kwargs) -> None:
        self.policy = None  # To be set later
        self.action_dim = action_dim

    def forward(self, obs: dict, *args, **kwargs) -> th.Tensor:
        """
        Directly return a zero action tensor of the specified action dimension.
        """
        if self.policy is not None:
            return self.policy.act(obs).detach().cpu()
        else:
            assert self.action_dim is not None
            return th.zeros(self.action_dim, dtype=th.float32)

    def reset(self) -> None:
        if self.policy is not None:
            self.policy.reset()


class WebsocketPolicy:
    """
    Websocket policy for controlling the robot over a websocket connection.
    """

    def __init__(
        self,
        *args,
        host: Optional[str] = None,
        port: Optional[int] = None,
        **kwargs,
    ) -> None:
        logging.info(f"Creating websocket client policy with host: {host}, port: {port}")
        self.policy = WebsocketClientPolicy(host=host, port=port)

    def forward(self, obs: dict, *args, **kwargs) -> th.Tensor:
        # convert observation to numpy
        obs = torch_to_numpy(obs)
        return self.policy.act(obs).detach().cpu()

    def reset(self) -> None:
        self.policy.reset()
