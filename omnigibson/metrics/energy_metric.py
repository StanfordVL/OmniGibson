import numpy as np

from omnigibson.metrics.metrics_base import BaseMetric


class EnergyMetric(BaseMetric):
    """
    Energy Metric

    Measures displacement * mass for every link

    Args:
        measure_work: If true, measure beginning and end delta rather than step by step delta
    """

    def __init__(self, measure_work=False):

        # Run super
        super().__init__()
        self.state_cache = None
        self.link_masses = {}

        # parameter for checking if this is a work or energy metric. If true, measure work done, else measure energy
        self.measure_work = measure_work

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

            return 0.0

        # calculate the energy spent from the previous state to the current state
        work_metric = 0.0
        for linkname, posrot in new_state_cache.items():

            # TODO: this computation is very slow, consider using a more efficient method
            # TODO: this method needs to be updated to account for object addition and removal
            posrot2 = self.state_cache[linkname]
            work_metric += np.linalg.norm(posrot[0], posrot2[0]) * self.link_masses[linkname]

        # if measuring energy, update the state cache
        if not self.measure_work:
            self.state_cache = new_state_cache

        # update the metric accordingly, either set the work done (measuring work) or add it to previous energy spent (measuring energy)
        self._metric = work_metric if self.measure_work else (self._metric + work_metric)
        return self._metric

    def reset(self, task, env):
        super().reset(task, env)
        self.state_cache = None
