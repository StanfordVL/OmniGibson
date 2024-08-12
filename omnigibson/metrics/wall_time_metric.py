from omnigibson.metrics.metrics_base import BaseMetric


class WallTimeMetric(BaseMetric):
    """
    WallTimeMetric
    Metric for wall time accumulated in policy steps
    """

    def __init__(self):
        # initialize wall time metric
        self._metric = 0

    def _step(self, task, env, action):
        self._metric += env.last_step_wall_time
        return {"wall_time": self._metric}

    def reset(self, task, env):
        self._metric = 0
