from omnigibson.metrics.metrics_base import BaseMetric


class StepMetric(BaseMetric):
    """
    Step Metric
    Metric for each simulator step
    """

    def __init__(self):
        # initialize step
        self._metric = 0

    def _step(self, task, env, action):
        self._metric += 1
        return {"step": self._metric}

    def reset(self, task, env):
        self._metric = 0
