from omnigibson.reward_functions.reward_function_base import BaseRewardFunction
import numpy as np


class EnergyMetric(BaseRewardFunction):
    """
    Energy Metric

    Measures displacement * mass for every link

    Args:
        measure_work: If true, measure beginning and end delta rather than step by step delta
    """

    def __init__(self, measure_work=False):
        # Run super
        super().__init__()
        self._reward = 0
        self.initialized = False
        self.state_cache = {}
        self.link_masses = {}
        self.measure_work = measure_work 

    def calculate_displacement(self, posrot, posrot2):
        return np.linalg.norm(posrot[0] - posrot2[0])

    def _step(self, task, env, action):
        new_state_cache = {}
        for obj in env.scene.objects:
            for link_name, link in obj._links.items():
                pos, rot = link.get_position_orientation()
                new_state_cache[link_name] = (pos, rot)

        if not self.initialized:
            self.initialized = True
            self.state_cache = new_state_cache

            for obj in env.scene.objects:
                for link_name, link in obj._links.items():
                    self.link_masses[link_name] = link.mass
            return 0.0, {}

        work_metric = 0.0
        for linkname, posrot in new_state_cache.items():
            work_metric += self.calculate_displacement(posrot, self.state_cache[linkname]) * self.link_masses[linkname]

        if self.measure_work:
            self._reward = 0
        if not self.measure_work:
            self.state_cache = new_state_cache

        self._reward += work_metric
        return self._reward, {}

    def reset(self, task, env):
        super().reset(task, env)
        self.state_cache = {}
        self.initialized = False
