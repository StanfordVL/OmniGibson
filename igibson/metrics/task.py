import numpy as np

from igibson.metrics.metric_base import MetricBase
from bddl.activity import evaluate_goal_conditions


class TaskMetric(MetricBase):
    def __init__(self):
        self.satisfied_predicates = []
        self.timesteps = 0

    def start_callback(self, env, _):
        self.render_timestep = env.simulator.get_rendering_dt()           

    def step_callback(self, env, _):
        self.timesteps += 1
        _, satisfied_predicates = evaluate_goal_conditions(env.task.ground_goal_state_options[0])
        self.satisfied_predicates.append(satisfied_predicates)

    def end_callback(self, env, _):
        candidate_q_score = []
        for option in env.task.ground_goal_state_options:
            predicate_truth_values = []
            for predicate in option:
                predicate_truth_values.append(predicate.evaluate())
            candidate_q_score.append(np.mean(predicate_truth_values))
        self.final_q_score = np.max(candidate_q_score)

    def gather_results(self):
        return {
            "satisfied_predicates": {
                "timestep": self.satisfied_predicates,
            },
            "q_score": {"final": self.final_q_score},
            "time": {
                "simulator_steps": self.timesteps,
                "simulator_time": self.timesteps * self.render_timestep,
            },
        }
