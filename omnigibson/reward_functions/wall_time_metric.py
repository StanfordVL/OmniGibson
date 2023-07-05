from omnigibson.reward_functions.reward_function_base import BaseRewardFunction


class WallTimeMetric(BaseRewardFunction):
    """
    WallTimeMetric
    Metric for wall time accumulated in policy steps
    """

    def __init__(self):
        # Run super
        super().__init__()
        self._reward = 0

    def _step(self, task, env, action):
        self._reward += env.wall_time_step
        return self._reward, {}
