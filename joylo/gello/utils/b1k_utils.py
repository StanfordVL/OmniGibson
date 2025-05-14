from gello.utils.qa_utils import *


# Maps QA metric class to dict with metric name, initializer, and kwargs to pass into the metric's respective
# @validate_episode calls
ALL_QA_METRICS = {
    "motion": {
        "cls": MotionMetric,
        "init": None,
        "validate_kwargs": dict(
            vel_max_limit=None,
            acc_max_limit=None,
            jerk_max_limit=10000.0,
        ),
    },
    "collision": {
        "cls": CollisionMetric,
        "init": lambda: create_collision_metric(include_robot_self_collision=True, include_robot_nonarm_nonkinematic_collision=True),
        "validate_kwargs": dict(
            collision_limits=dict(
                robot_self=0,
                robot_nonarm_nonstructure=10,
            ),
        ),
    },
    "task_success": {
        "cls": TaskSuccessMetric,
        "init": None,
        "validate_kwargs": dict(),
    },
    "ghost_hand_appearance": {
        "cls": GhostHandAppearanceMetric,
        "init": None,
        "validate_kwargs": dict(
            gh_appearance_limit=50,
            gh_appearance_limit_while_ungrasping=10,
        ),
    },
    "prolonged_pause": {
        "cls": ProlongedPauseMetric,
        "init": None,
        "validate_kwargs": dict(
            pause_steps_limit=50,
        ),
    },
    "failed_grasp": {
        "cls": FailedGraspMetric,
        "init": None,
        "validate_kwargs": dict(
            failed_grasp_limit=5,
        ),
    },
    "task_relevant_obj_vel": {
        "cls": TaskRelevantObjectVelocityMetric,
        "init": None,
        "validate_kwargs": dict(
            vel_max_limit=4.0,
        ),
    },
    "gripper_in_fov": {
        "cls": FieldOfViewMetric,
        "init": None,
        "validate_kwargs": dict(
            gripper_changes_outside_fov_limit=0,
        ),
    },
}

# Set of common metric names to use
COMMON_QA_METRICS = {
    "motion",
    "collision",
    "task_success",
    "ghost_hand_appearance",
    "prolonged_pause",
    "failed_grasp",
    "task_relevant_obj_vel",
    "gripper_in_fov",
}

# Task specific qa metrics
TASK_QA_METRICS = {

}


def aggregate_episode_validation(all_episode_metrics):
    """
    Validates the given @all_episode_metrics

    Args:
        all_episode_metrics (dict): Keyword-mapped aggregated episode metrics

    Returns:
        2-tuple:
        - bool: Whether the validation succeeded or not (requires all metric validation checks to pass)
        - dict: Per-metric information
    """
    results = dict()
    sorted_metrics = dict()
    for name, val in all_episode_metrics.items():
        metric_name = name.split("::")[0]
        metric_val_name = "::".join(name.split("::")[1:])
        if metric_name not in sorted_metrics:
            sorted_metrics[metric_name] = dict()
        sorted_metrics[metric_name][metric_val_name] = val
    for metric_name, episode_metrics in sorted_metrics.items():
        metric_info = ALL_QA_METRICS[metric_name]
        results.update(metric_info["cls"].validate_episode(
            episode_metrics=episode_metrics,
            **metric_info["validate_kwargs"],
        ))

    # Passes if all metric validations pass
    success = all(v["success"] for v in results.values())
    return success, results
