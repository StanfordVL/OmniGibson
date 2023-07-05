from omnigibson.reward_functions.reward_function_base import BaseRewardFunction


class StepMetric(BaseRewardFunction):
    """
    Step Metric
    Metric for each simulator step
    """

    def __init__(self):
        # Run super
        super().__init__()
        self._reward = 0

    def _step(self, task, env, action):
        self._reward += 1
        return self._reward, {}
