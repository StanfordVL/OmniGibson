import torch as th

import omnigibson.utils.transform_utils as T
from omnigibson.metrics.metrics_base import BaseMetric


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

        # stores object mass and rotational inertia
        self.link_info = {}
        self.metric_config = metric_config

    def _step(self, task, env, action):

        # calculate the current pose of all object links
        new_state_cache = {}
        self.link_info = {}
        for obj in env.scene.objects:
            for link_name, link in obj._links.items():
                pos, rot = link.get_position_orientation()
                new_state_cache[link_name] = (pos, rot)

                # compute this every step to account for object addition and removal
                # store link mass and inertia for energy calculation
                mass = link.mass
                inertia = link.inertia
                self.link_info[link_name] = (mass, inertia)

        # if the state cache is empty, set it to the current state and return 0
        if not self.state_cache:
            self.state_cache = new_state_cache
            self.prev_state_cache = new_state_cache
            return {"work": 0, "energy": 0}

        # calculate the energy spent from the previous state to the current state
        work_metric = 0.0
        energy_metric = 0.0

        for linkname, new_posrot in new_state_cache.items():

            # TODO: this computation is very slow, consider using a more efficient method

            # check if the link is originally in the state cache, if not, skip it to account for object addition and removal
            if linkname not in self.state_cache:
                continue

            init_posrot = self.state_cache[linkname]
            mass, inertia = self.link_info[linkname]
            work_metric = self.calculate_work_energy(init_posrot, new_posrot, mass, inertia)

            # check if the link is in the prev_state_cache, if not, skip it to account for object addition and removal
            if linkname not in self.prev_state_cache:
                continue

            if self.prev_state_cache is not None:
                prev_posrot = self.prev_state_cache[linkname]
                energy_metric = self.calculate_work_energy(prev_posrot, new_posrot, mass, inertia)

        # update the prev_state cache for energy measurement
        self.prev_state_cache = new_state_cache

        # update the metric accordingly, set the work done (measuring work) and add energy spent this step to previous energy spent (measuring energy)
        self._work_metric = work_metric
        self._energy_metric += energy_metric
        return {"work": self._work_metric, "energy": self._energy_metric}
    
    def calculate_work_energy(self, new_posrot, curr_posrot, mass, inertia):

        """
        Calculate the work done and energy spent in a single step

        Args:
            new_posrot (tuple): new position and orientation of the link  
            curr_posrot (tuple): current position and orientation of the link. 
                                 use initial posrot for work metric, and previous posrot for energy metric
            mass (float): mass of the link
            inertia (th.Tensor): rotational inertia of the link
        
        Returns:
            float: work/energy spent in a single step between two states
        """

        position, orientation = T.relative_pose_transform(new_posrot[0], new_posrot[1], curr_posrot[0], curr_posrot[1])
        orientation = T.quat2axisangle(orientation)

        # formula for translational work metric: displacement * mass * translation weight coefficient
        work_metric = th.norm(position) * mass * self.metric_config["translation"]

        # formula for rotational work metric: rotation * moment of inertia * rotation weight coefficient
        work_metric += sum(th.matmul(orientation, inertia)) * self.metric_config["rotation"]

        return work_metric

    def reset(self, task, env):

        # reset both _work_metric and _energy_metric, and the state cache
        self._work_metric = 0
        self._energy_metric = 0
        self.state_cache = None
        self.prev_state_cache = None
        self.link_info = {}