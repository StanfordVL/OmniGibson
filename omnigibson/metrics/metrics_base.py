from abc import ABCMeta, abstractmethod
from copy import deepcopy

# from omnigibson.utils.python_utils import Registerable, classproperty
class BaseMetric():
    """
    Base Metric class
    Metric-specific reset and step methods are implemented in subclasses
    """

    def __init__(self):
        # Store internal vars that will be filled in at runtime
        self._metric = 0

    @abstractmethod
    def _step(self, task, env, action):
        """
        Step the metric function and compute the metric at the current timestep. Overwritten by subclasses.

        Args:
            task (BaseTask): Task instance
            env (Environment): Environment instance
            action (n-array): 1D flattened array of actions executed by all agents in the environment

        Returns:
            2-tuple:
                - bool: computed metric
                - dict: any metric-related information for this specific metric
        """
        raise NotImplementedError()

    def step(self, task, env, action):
        """
        Step the metricfunction and compute the metric at the current timestep. Overwritten by subclasses

        Args:
            task (BaseTask): Task instance
            env (Environment): Environment instance
            action (n-array): 1D flattened array of actions executed by all agents in the environment

        Returns:
            2-tuple:
                - bool: computed metric
                - dict: any metric-related information for this specific metric
        """
        # Step internally and store output
        self._metric = self._step(task=task, env=env, action=action)

        # Return metric
        return self._metric

    def reset(self, task, env):
        """
        General metrics reset
        """

        # Reset internal vars
        self._metric = 0
