import numpy as np

from omnigibson.metrics.metrics_base import BaseMetric


class WorkEnergyMetric(BaseMetric):
    """
    Work and Energy Metric

    Measures displacement * mass for every link

    """

    def __init__(self):

        # Run work and energy metric initialization
        self._work_metric = 0
        self._energy_metric = 0
        self.state_cache = None
        self.prev_state_cache = None
        self.link_masses = {}

    def _step(self, task, env, action):

        # calculate the current pose of all object links
        new_state_cache = {}
        for obj in env.scene.objects:
            for link_name, link in obj._links.items():
                pos, rot = link.get_position_orientation()
                new_state_cache[link_name] = (pos, rot)

        # if the state cache is empty, set it to the current state and return 0
        if not self.state_cache:
            self.state_cache = new_state_cache

            for obj in env.scene.objects:
                for link_name, link in obj._links.items():
                    self.link_masses[link_name] = link.mass

            return {"work": 0, "energy": 0}

        # calculate the energy spent from the previous state to the current state
        work_metric = 0.0
        energy_metric = 0.0
        for linkname, posrot in new_state_cache.items():

            # TODO: this computation is very slow, consider using a more efficient method
            # TODO: this method needs to be updated to account for object addition and removal
            init_posrot = self.state_cache[linkname]
            work_metric += np.linalg.norm(posrot[0] - init_posrot[0]) * self.link_masses[linkname]

            if self.prev_state_cache is not None:
                prev_posrot = self.prev_state_cache[linkname]
                energy_metric += np.linalg.norm(posrot[0] - prev_posrot[0]) * self.link_masses[linkname]

        # update the prev_state cache for energy measurement
        self.prev_state_cache = new_state_cache

        # update the metric accordingly, set the work done (measuring work) and add energy spent this step to previous energy spent (measuring energy)
        self._work_metric = work_metric
        self._energy_metric += energy_metric
        return {"work": self._work_metric, "energy": self._energy_metric}

    def reset(self, task, env):

        # reset both _work_metric and _energy_metric, and the state cache
        self._work_metric = 0
        self._energy_metric = 0
        self.state_cache = None
        self.prev_state_cache = None
