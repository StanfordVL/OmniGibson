from omnigibson.metrics.metrics_base import BaseMetric


class TaskSuccessMetric(BaseMetric):
    """
    TaskSuccessMetric
    Metric for partial or full task success
    """

    def __init__(self):
        # Run super
        super().__init__()

    def _step(self, task, env, action):
        successes = []
        partial_success = 0

        # Evaluate termination conditions
        for termination_condition in task._termination_conditions.values():

            # Check if partial success is supported, and if so, store the score (e.g. Behavior Task)
            if termination_condition.partial_success:
                partial_success = task.success_score
            
            done, success = termination_condition.step(task, env, action)
            successes.append(success)

        # Calculate metric
        if any(successes):
            self._metric = 1.0
        elif partial_success > 0:
            self._metric = partial_success
        else:
            self._metric = 0.0

        return self._metric

    def reset(self, task, env):
        super().reset(task, env)
