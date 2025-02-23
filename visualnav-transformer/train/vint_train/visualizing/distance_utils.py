import os
import wandb
import numpy as np
from typing import List, Optional, Tuple
from vint_train.visualizing.visualize_utils import numpy_to_img
import matplotlib.pyplot as plt


def visualize_dist_pred(
    batch_obs_images: np.ndarray,
    batch_goal_images: np.ndarray,
    batch_dist_preds: np.ndarray,
    batch_dist_labels: np.ndarray,
    eval_type: str,
    save_folder: str,
    epoch: int,
    num_images_preds: int = 8,
    use_wandb: bool = True,
    display: bool = False,
    rounding: int = 4,
    dist_error_threshold: float = 3.0,
):
    """
    Visualize the distance classification predictions and labels for an observation-goal image pair.

    Args:
        batch_obs_images (np.ndarray): batch of observation images [batch_size, height, width, channels]
        batch_goal_images (np.ndarray): batch of goal images [batch_size, height, width, channels]
        batch_dist_preds (np.ndarray): batch of distance predictions [batch_size]
        batch_dist_labels (np.ndarray): batch of distance labels [batch_size]
        eval_type (string): {data_type}_{eval_type} (e.g. recon_train, gs_test, etc.)
        epoch (int): current epoch number
        num_images_preds (int): number of images to visualize
        use_wandb (bool): whether to use wandb to log the images
        save_folder (str): folder to save the images. If None, will not save the images
        display (bool): whether to display the images
        rounding (int): number of decimal places to round the distance predictions and labels
        dist_error_threshold (float): distance error threshold for classifying the distance prediction as correct or incorrect (only used for visualization purposes)
    """
    visualize_path = os.path.join(
        save_folder,
        "visualize",
        eval_type,
        f"epoch{epoch}",
        "dist_classification",
    )
    if not os.path.isdir(visualize_path):
        os.makedirs(visualize_path)
    assert (
        len(batch_obs_images)
        == len(batch_goal_images)
        == len(batch_dist_preds)
        == len(batch_dist_labels)
    )
    batch_size = batch_obs_images.shape[0]
    wandb_list = []
    for i in range(min(batch_size, num_images_preds)):
        dist_pred = np.round(batch_dist_preds[i], rounding)
        dist_label = np.round(batch_dist_labels[i], rounding)
        obs_image = numpy_to_img(batch_obs_images[i])
        goal_image = numpy_to_img(batch_goal_images[i])

        save_path = None
        if save_folder is not None:
            save_path = os.path.join(visualize_path, f"{i}.png")
        text_color = "black"
        if abs(dist_pred - dist_label) > dist_error_threshold:
            text_color = "red"

        display_distance_pred(
            [obs_image, goal_image],
            ["Observation", "Goal"],
            dist_pred,
            dist_label,
            text_color,
            save_path,
            display,
        )
        if use_wandb:
            wandb_list.append(wandb.Image(save_path))
    if use_wandb:
        wandb.log({f"{eval_type}_dist_prediction": wandb_list}, commit=False)


def visualize_dist_pairwise_pred(
    batch_obs_images: np.ndarray,
    batch_close_images: np.ndarray,
    batch_far_images: np.ndarray,
    batch_close_preds: np.ndarray,
    batch_far_preds: np.ndarray,
    batch_close_labels: np.ndarray,
    batch_far_labels: np.ndarray,
    eval_type: str,
    save_folder: str,
    epoch: int,
    num_images_preds: int = 8,
    use_wandb: bool = True,
    display: bool = False,
    rounding: int = 4,
):
    """
    Visualize the distance classification predictions and labels for an observation-goal image pair.

    Args:
        batch_obs_images (np.ndarray): batch of observation images [batch_size, height, width, channels]
        batch_close_images (np.ndarray): batch of close goal images [batch_size, height, width, channels]
        batch_far_images (np.ndarray): batch of far goal images [batch_size, height, width, channels]
        batch_close_preds (np.ndarray): batch of close predictions [batch_size]
        batch_far_preds (np.ndarray): batch of far predictions [batch_size]
        batch_close_labels (np.ndarray): batch of close labels [batch_size]
        batch_far_labels (np.ndarray): batch of far labels [batch_size]
        eval_type (string): {data_type}_{eval_type} (e.g. recon_train, gs_test, etc.)
        save_folder (str): folder to save the images. If None, will not save the images
        epoch (int): current epoch number
        num_images_preds (int): number of images to visualize
        use_wandb (bool): whether to use wandb to log the images
        display (bool): whether to display the images
        rounding (int): number of decimal places to round the distance predictions and labels
    """
    visualize_path = os.path.join(
        save_folder,
        "visualize",
        eval_type,
        f"epoch{epoch}",
        "pairwise_dist_classification",
    )
    if not os.path.isdir(visualize_path):
        os.makedirs(visualize_path)
    assert (
        len(batch_obs_images)
        == len(batch_close_images)
        == len(batch_far_images)
        == len(batch_close_preds)
        == len(batch_far_preds)
        == len(batch_close_labels)
        == len(batch_far_labels)
    )
    batch_size = batch_obs_images.shape[0]
    wandb_list = []
    for i in range(min(batch_size, num_images_preds)):
        close_dist_pred = np.round(batch_close_preds[i], rounding)
        far_dist_pred = np.round(batch_far_preds[i], rounding)
        close_dist_label = np.round(batch_close_labels[i], rounding)
        far_dist_label = np.round(batch_far_labels[i], rounding)
        obs_image = numpy_to_img(batch_obs_images[i])
        close_image = numpy_to_img(batch_close_images[i])
        far_image = numpy_to_img(batch_far_images[i])

        save_path = None
        if save_folder is not None:
            save_path = os.path.join(visualize_path, f"{i}.png")

        if close_dist_pred < far_dist_pred:
            text_color = "black"
        else:
            text_color = "red"

        display_distance_pred(
            [obs_image, close_image, far_image],
            ["Observation", "Close Goal", "Far Goal"],
            f"close_pred = {close_dist_pred}, far_pred = {far_dist_pred}",
            f"close_label = {close_dist_label}, far_label = {far_dist_label}",
            text_color,
            save_path,
            display,
        )
        if use_wandb:
            wandb_list.append(wandb.Image(save_path))
    if use_wandb:
        wandb.log({f"{eval_type}_pairwise_classification": wandb_list}, commit=False)


def display_distance_pred(
    imgs: list,
    titles: list,
    dist_pred: float,
    dist_label: float,
    text_color: str = "black",
    save_path: Optional[str] = None,
    display: bool = False,
):
    plt.figure()
    fig, ax = plt.subplots(1, len(imgs))

    plt.suptitle(f"prediction: {dist_pred}\nlabel: {dist_label}", color=text_color)

    for axis, img, title in zip(ax, imgs, titles):
        axis.imshow(img)
        axis.set_title(title)
        axis.xaxis.set_visible(False)
        axis.yaxis.set_visible(False)

    # make the plot large
    fig.set_size_inches((18.5 / 3) * len(imgs), 10.5)

    if save_path is not None:
        fig.savefig(
            save_path,
            bbox_inches="tight",
        )
    if not display:
        plt.close(fig)
