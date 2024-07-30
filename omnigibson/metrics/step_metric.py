from omnigibson.metrics.metrics_base import BaseMetric


class StepMetric(BaseMetric):
    """
    Step Metric
    Metric for each simulator step
    """

    def __init__(self):
        # Run super
        super().__init__()

    def _step(self, task, env, action):
        self._metric += 1
        return self._metric
    
    def reset(self, task, env):
        super().reset(task, env)
