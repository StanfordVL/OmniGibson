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
        partial_successes = []
        for termination_condition in task._termination_conditions.values():
            if termination_condition.partial_success >= 0.0:
                partial_successes.append(termination_condition.partial_success)
            done, success = termination_condition.step(task, env, action)
            # success <=> done and non failure
            successes.append(success)
        if sum(successes) > 0:
            self._metric = 1.0
        elif partial_successes:
            self._metric = sum(partial_successes) / len(partial_successes)
        else:
            self._metric = 0.0
        # Populate info
        return self._metric

    def reset(self, task, env):
        super().reset(task, env)
