import numpy as np

from omnigibson.metrics.metrics_base import BaseMetric

<<<<<<< HEAD
import omnigibson.utils.transform_utils as T

=======


>>>>>>> b6fc0eac3180abd5cd47cc1c0b9b9928e50492ca
class WorkEnergyMetric(BaseMetric):
    """
    Work and Energy Metric

    Measures displacement * mass for every link

    """

    def __init__(self, metric_config=None):

        # Run work and energy metric initialization
        self._work_metric = 0
        self._energy_metric = 0
        self.state_cache = None
        self.prev_state_cache = None
        self.link_masses = {}
        self.metric_config = metric_config

    def _step(self, task, env, action):

        # calculate the current pose of all object links
        new_state_cache = {}
        for obj in env.scene.objects:
            for link_name, link in obj._links.items():
                pos, rot = link.get_position_orientation()
                new_state_cache[link_name] = (pos, rot)

                # compute this every step to account for object addition and removal
                self.link_masses[link_name] = link.mass

        # if the state cache is empty, set it to the current state and return 0
        if not self.state_cache:
            self.state_cache = new_state_cache
            self.prev_state_cache = new_state_cache
            return {"work": 0, "energy": 0}

        # calculate the energy spent from the previous state to the current state
        work_metric = 0.0
        energy_metric = 0.0

        for linkname, posrot in new_state_cache.items():

            # TODO: this computation is very slow, consider using a more efficient method

            # check if the link is originally in the state cache, if not, skip it to account for object addition and removal
            if linkname not in self.state_cache:
                continue

            init_posrot = self.state_cache[linkname]
            position, orientation = T.relative_pose_transform(posrot[0], posrot[1], init_posrot[0], init_posrot[1])
            work_metric += np.linalg.norm(position) * self.link_masses[linkname] * self.metric_config["translation"]

            # calculate the energy spent in rotation. TODO: this is a very rough approximation, consider using a more accurate method
            work_metric += np.linalg.norm(orientation) * self.link_masses[linkname] * self.metric_config["rotation"]

            # check if the link is in the prev_state_cache, if not, skip it to account for object addition and removal
            if linkname not in self.prev_state_cache:
                continue

            if self.prev_state_cache is not None:
                prev_posrot = self.prev_state_cache[linkname]
                position, orientation = T.relative_pose_transform(posrot[0], posrot[1], prev_posrot[0], prev_posrot[1])
                energy_metric += np.linalg.norm(position) * self.link_masses[linkname] * self.metric_config["translation"]

                # calculate the energy spent in rotation. TODO: this is a very rough approximation, consider using a more accurate method
                energy_metric += np.linalg.norm(orientation) * self.link_masses[linkname] * self.metric_config["rotation"]

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
