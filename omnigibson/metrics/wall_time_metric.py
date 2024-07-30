from omnigibson.metrics.metrics_base import BaseMetric


class WallTimeMetric(BaseMetric):
    """
    WallTimeMetric
    Metric for wall time accumulated in policy steps
    """

    def __init__(self):
        # Run super
        super().__init__()

    def _step(self, task, env, action):
        self._metric += env.last_step_wall_time
        return self._metric

    def reset(self, task, env):
        super().reset(task, env)
