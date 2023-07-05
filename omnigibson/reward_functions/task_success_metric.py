from omnigibson.reward_functions.reward_function_base import BaseRewardFunction

class TaskSuccessMetric(BaseRewardFunction):
    """
    TaskSuccessMetric
    Metric for partial or full task success 
    """

    def __init__(self):
        # Run super
        super().__init__()
        self._reward = 0

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
            self._reward = 1.0
        elif partial_successes:
            self._reward = sum(partial_successes) / len(partial_successes)
        else:
            self._reward = 0.0
        # Populate info
        return self._reward, {}
