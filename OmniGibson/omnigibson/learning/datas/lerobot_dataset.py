import argparse
import av
import json
import logging
import os
import packaging.version
import numpy as np
import torch as th
import torchvision
from collections.abc import Callable
from omnigibson.learning.utils.eval_utils import TASK_NAMES_TO_INDICES, ROBOT_CAMERA_NAMES
from omnigibson.learning.utils.obs_utils import dequantize_depth, MIN_DEPTH, MAX_DEPTH, DEPTH_SHIFT
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.io import VideoReader
from typing import Any, Dict, Iterable, Tuple
from tqdm import tqdm

from lerobot.constants import HF_LEROBOT_HOME
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata, CODEBASE_VERSION
from lerobot.datasets.utils import (
    EPISODES_PATH,
    check_delta_timestamps,
    check_timestamps_sync,
    get_delta_indices,
    get_episode_data_index,
    get_safe_version,
    load_jsonlines,
)
from lerobot.datasets.video_utils import get_safe_default_codec, decode_video_frames as _decode_video_frames


class BehaviorLerobotDatasetMetadata(LeRobotDatasetMetadata):
    """
    LerobotDatasetMetadata with the following customizations:
        1. Custom task names mapping to indices.
    """

    def __init__(self, *args, modalities: Iterable[str] = None, cameras: Iterable[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.modalities = set(modalities)
        self.camera_names = set(cameras)
        assert self.modalities.issubset(
            {"rgb", "depth", "seg_instance_id"}
        ), f"Modalities must be a subset of ['rgb', 'depth', 'seg_instance_id'], but got {self.modalities}"
        assert self.camera_names.issubset(
            ROBOT_CAMERA_NAMES["R1Pro"]
        ), f"Camera names must be a subset of {ROBOT_CAMERA_NAMES['R1Pro']}, but got {self.camera_names}"

    @property
    def features(self) -> dict[str, dict]:
        """All features contained in the dataset."""
        features = dict()
        # pop not required features
        for name in self.info["features"].keys():
            if (
                name.startswith("observation.images.")
                and name.split(".")[-1] in self.camera_names
                and name.split(".")[-2] in self.modalities
            ):
                features[name] = self.info["features"][name]
        return features


class BehaviorLeRobotDataset(LeRobotDataset):
    """
    LeRobotDataset with the following customizations:
        1. Custom chunking logic based on
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        tasks: Iterable[str] = None,
        modalities: Iterable[str] = None,
        cameras: Iterable[str] = None,
        video_backend: str | None = "pyav",
        local_only: bool = False,
    ):
        """
        Custom args:
            tasks (List[str]): list of task names to load. If None, all tasks will be loaded.
            modalities (List[str]): list of modality names to load. If None, all modalities will be loaded.
                must be a subset of ["rgb", "depth", "seg_instance_id"]
            cameras (List[str]): list of camera names to load. If None, all cameras will be loaded.
                must be a subset of ["left_wrist", "right_wrist", "head"]
            video_backend: video backend to use for decoding videos.
            local_only: whether to only use local data (not download from HuggingFace).
        """
        Dataset.__init__(self)
        self.repo_id = repo_id
        self.root = Path(root) if root else HF_LEROBOT_HOME / repo_id
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else CODEBASE_VERSION
        self.video_backend = video_backend if video_backend else get_safe_default_codec()
        if "depth" in modalities:
            assert self.video_backend == "pyav", (
                "Depth videos can only be decoded with the 'pyav' backend. "
                "Please set video_backend='pyav' when initializing the dataset."
            )
        self.delta_indices = None

        # Unused attributes
        self.image_writer = None
        self.episode_buffer = None

        self.root.mkdir(exist_ok=True, parents=True)

        # ========== Customizations ==========
        # Load metadata
        self.meta = BehaviorLerobotDatasetMetadata(
            repo_id=self.repo_id,
            root=self.root,
            revision=self.revision,
            force_cache_sync=force_cache_sync,
            modalities=modalities,
            cameras=cameras,
        )
        self.tasks_names = set(tasks) if tasks is not None else set(TASK_NAMES_TO_INDICES.keys())
        # overwrite episode based on task
        if episodes is None:
            episodes = load_jsonlines(self.root / EPISODES_PATH)
            self.episodes = sorted([item["episode_index"] for item in episodes if item["tasks"][0] in self.tasks_names])
        # ====================================

        if self.episodes is not None and self.meta._version >= packaging.version.parse("v2.1"):
            episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
            self.stats = aggregate_stats(episodes_stats)

        # Load actual data
        try:
            if force_cache_sync:
                raise FileNotFoundError
            assert all((self.root / fpath).is_file() for fpath in self.get_episodes_file_paths())
            self.hf_dataset = self.load_hf_dataset()
        except (AssertionError, FileNotFoundError, NotADirectoryError):
            if local_only:
                raise FileNotFoundError
            self.revision = get_safe_version(self.repo_id, self.revision)
            self.download_episodes(download_videos)
            self.hf_dataset = self.load_hf_dataset()

        self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)

        # Check timestamps
        timestamps = th.stack(self.hf_dataset["timestamp"]).numpy()
        episode_indices = th.stack(self.hf_dataset["episode_index"]).numpy()
        ep_data_index_np = {k: t.numpy() for k, t in self.episode_data_index.items()}
        check_timestamps_sync(timestamps, episode_indices, ep_data_index_np, self.fps, self.tolerance_s)

        # Setup delta_indices
        if self.delta_timestamps is not None:
            check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

    def get_episodes_file_paths(self) -> list[str]:
        """
        Overwrite the original method to use the episodes indices instead of range(self.meta.total_episodes)
        """
        episodes = self.episodes if self.episodes is not None else list(self.meta.episodes.keys())
        fpaths = [str(self.meta.get_data_file_path(ep_idx)) for ep_idx in episodes]
        if len(self.meta.video_keys) > 0:
            video_files = [
                str(self.meta.get_video_file_path(ep_idx, vid_key))
                for vid_key in self.meta.video_keys
                for ep_idx in episodes
            ]
            fpaths += video_files

        return fpaths

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, th.Tensor]:
        """Note: When using data workers (e.g. DataLoader with num_workers>0), do not call this function
        in the main process (e.g. by using a second Dataloader with num_workers=0). It will result in a
        Segmentation Fault. This probably happens because a memory reference to the video loader is created in
        the main process and a subprocess fails to access it.
        """
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
            frames = decode_video_frames(video_path, query_ts, self.tolerance_s, self.video_backend)
            # post-process seg instance id:
            if "seg_instance_id" in vid_key:
                N, H, W, C = frames.shape
                rgb_flat = frames.reshape(N, -1, C)  # (H*W, 3)
                # For each rgb pixel, find the index of the nearest color in the equidistant bins
                distances = th.cdist(rgb_flat, self.palette.unsqueeze(0).expand(N, -1, -1), p=2)
                ids = th.argmin(distances, dim=-1)  # (N, H*W)
                frames = self.id_list[ids].reshape(N, H, W)  # (N, H, W)
            item[vid_key] = frames.squeeze(0)

        return item


def decode_video_frames(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str | None = None,
) -> th.Tensor:
    if "depth" in video_path.name:
        return decode_video_frames_depth(video_path, timestamps, tolerance_s)
    else:
        return _decode_video_frames(
            video_path=video_path, timestamps=timestamps, tolerance_s=tolerance_s, backend=backend
        )


def decode_video_frames_depth(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    log_loaded_timestamps: bool = False,
) -> th.Tensor:
    """
    Adapted from decode_video_frames_vision to handle depth decoding
    """
    video_path = str(video_path)

    # set backend
    torchvision.set_video_backend("pyav")
    keyframes_only = True  # pyav doesn't support accurate seek

    # set a video stream reader
    # TODO(rcadene): also load audio stream at the same time
    reader = DepthVideoReader(video_path, "video")

    # set the first and last requested timestamps
    # Note: previous timestamps are usually loaded, since we need to access the previous key frame
    first_ts = min(timestamps)
    last_ts = max(timestamps)

    # access closest key frame of the first requested frame
    # Note: closest key frame timestamp is usually smaller than `first_ts` (e.g. key frame can be the first frame of the video)
    # for details on what `seek` is doing see: https://pyav.basswood-io.com/docs/stable/api/container.html?highlight=inputcontainer#av.container.InputContainer.seek
    reader.seek(first_ts, keyframes_only=keyframes_only)

    # load all frames until last requested frame
    loaded_frames = []
    loaded_ts = []
    for frame in reader:
        current_ts = frame["pts"]
        if log_loaded_timestamps:
            logging.info(f"frame loaded at timestamp={current_ts:.4f}")
        loaded_frames.append(frame["data"])
        loaded_ts.append(current_ts)
        if current_ts >= last_ts:
            break

    reader.container.close()

    reader = None

    query_ts = th.tensor(timestamps)
    loaded_ts = th.tensor(loaded_ts)

    # compute distances between each query timestamp and timestamps of all loaded frames
    dist = th.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    assert is_within_tol.all(), (
        f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."
        "It means that the closest frame that can be loaded from the video is too far away in time."
        "This might be due to synchronization issues with timestamps during data collection."
        "To be safe, we advise to ignore this item during training."
        f"\nqueried timestamps: {query_ts}"
        f"\nloaded timestamps: {loaded_ts}"
        f"\nvideo: {video_path}"
        f"\nbackend: pyav"
    )

    # get closest frames to the query timestamps
    closest_frames = th.stack([loaded_frames[idx] for idx in argmin_])
    closest_ts = loaded_ts[argmin_]

    if log_loaded_timestamps:
        logging.info(f"{closest_ts=}")

    # convert to the pytorch format which is float32 in [0,1] range (and channel first)
    closest_frames = closest_frames.type(th.float32) / 255

    assert len(timestamps) == len(closest_frames)
    return closest_frames


class DepthVideoReader(VideoReader):
    """
    Adapted from torchvision.io.VideoReader to support gray16le decoding for depth
    """

    def __next__(self) -> Dict[str, Any]:
        """Decodes and returns the next frame of the current stream.
        Frames are encoded as a dict with mandatory
        data and pts fields, where data is a tensor, and pts is a
        presentation timestamp of the frame expressed in seconds
        as a float.

        Returns:
            (dict): a dictionary and containing decoded frame (``data``)
            and corresponding timestamp (``pts``) in seconds

        """
        try:
            frame = next(self._c)
            pts = float(frame.pts * frame.time_base)
            if "video" in self.pyav_stream:
                frame = th.as_tensor(
                    dequantize_depth(
                        frame.reformat(format="gray16le").to_ndarray(),
                        min_depth=MIN_DEPTH,
                        max_depth=MAX_DEPTH,
                        shift=DEPTH_SHIFT,
                    )
                )
            elif "audio" in self.pyav_stream:
                frame = th.as_tensor(frame.to_ndarray()).permute(1, 0)
            else:
                frame = None
        except av.error.EOFError:
            raise StopIteration

        if frame.numel() == 0:
            raise StopIteration

        return {"data": frame, "pts": pts}


def generate_task_json(data_dir: str) -> int:
    num_tasks = len(TASK_NAMES_TO_INDICES)

    with open(f"{data_dir}/meta/tasks.jsonl", "w") as f:
        for task_name, task_index in tqdm(TASK_NAMES_TO_INDICES.items()):
            json.dump({"task_index": task_index, "task": task_name}, f)
            f.write("\n")
    return num_tasks


def generate_episode_json(data_dir: str) -> Tuple[int, int]:
    assert os.path.exists(f"{data_dir}/meta/tasks.jsonl"), "Task JSON does not exist!"
    assert os.path.exists(f"{data_dir}/meta/episodes"), "Episode metadata directory does not exist!"
    with open(f"{data_dir}/meta/tasks.jsonl", "r") as f:
        task_json = [json.loads(line) for line in f]
    num_frames = 0
    num_episodes = 0
    with open(f"{data_dir}/meta/episodes.jsonl", "w") as out_f:
        with open(f"{data_dir}/meta/episodes_stats.jsonl", "w") as out_stats_f:
            for task_info in tqdm(task_json):
                task_index = task_info["task_index"]
                task_name = task_info["task"]
                if not os.path.exists(f"{data_dir}/meta/episodes/task-{task_index:04d}"):
                    continue
                for episode_name in sorted(os.listdir(f"{data_dir}/meta/episodes/task-{task_index:04d}")):
                    with open(f"{data_dir}/meta/episodes/task-{task_index:04d}/{episode_name}", "r") as f:
                        episode_info = json.load(f)
                        episode_index = int(episode_name.split(".")[0].split("_")[-1])
                        episode_json = {
                            "episode_index": episode_index,
                            "tasks": [task_name],
                            "length": episode_info["num_samples"],
                        }
                        episode_stats_json = {
                            "episode_index": episode_index,
                            "tasks": [task_name],
                            "stats": {
                                "obs": {
                                    "min": np.array([episode_info["num_samples"]]).tolist(),
                                    "max": np.array([episode_info["num_samples"]]).tolist(),
                                    "mean": np.array([episode_info["num_samples"]]).tolist(),
                                    "std": np.array([episode_info["num_samples"]]).tolist(),
                                    "count": np.array([episode_info["num_samples"]]).tolist(),
                                }
                            },
                        }
                        num_episodes += 1
                        num_frames += episode_info["num_samples"]
                    json.dump(episode_json, out_f)
                    out_f.write("\n")
                    json.dump(episode_stats_json, out_stats_f)
                    out_stats_f.write("\n")
    return num_episodes, num_frames


def generate_info_json(
    data_dir: str,
    fps: int = 30,
    total_episodes: int = 50,
    total_tasks: int = 50,
    total_frames: int = 50,
):
    info = {
        "codebase_version": "v2.1",
        "robot_type": "R1Pro",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_episodes * 6,
        "chunks_size": 10000,
        "fps": fps,
        "splits": {
            "train": "0:" + str(total_episodes),
        },
        "data_path": "data/task-{episode_chunk:04d}/episode_{episode_index:08d}.parquet",
        "video_path": "videos/task-{episode_chunk:04d}/{video_key}/episode_{episode_index:08d}.mp4",
        "features": {
            "observation.images.rgb.left_wrist": {
                "dtype": "video",
                "shape": [480, 480, 3],
                "names": ["height", "width", "rgb"],
                "info": {
                    "video.fps": 30.0,
                    "video.height": 480,
                    "video.width": 480,
                    "video.channels": 3,
                    "video.codec": "libx265",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "observation.images.rgb.right_wrist": {
                "dtype": "video",
                "shape": [480, 480, 3],
                "names": ["height", "width", "rgb"],
                "info": {
                    "video.fps": 30.0,
                    "video.height": 480,
                    "video.width": 480,
                    "video.channels": 3,
                    "video.codec": "libx265",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "observation.images.rgb.head": {
                "dtype": "video",
                "shape": [720, 720, 3],
                "names": ["height", "width", "rgb"],
                "info": {
                    "video.fps": 30.0,
                    "video.height": 720,
                    "video.width": 720,
                    "video.channels": 3,
                    "video.codec": "libx265",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "observation.images.depth.left_wrist": {
                "dtype": "video",
                "shape": [480, 480, 3],
                "names": ["height", "width", "depth"],
                "info": {
                    "video.fps": 30.0,
                    "video.height": 480,
                    "video.width": 480,
                    "video.channels": 3,
                    "video.codec": "libx265",
                    "video.pix_fmt": "yuv420p16le",
                    "video.is_depth_map": True,
                    "has_audio": False,
                },
            },
            "observation.images.depth.right_wrist": {
                "dtype": "video",
                "shape": [480, 480, 3],
                "names": ["height", "width", "depth"],
                "info": {
                    "video.fps": 30.0,
                    "video.height": 480,
                    "video.width": 480,
                    "video.channels": 3,
                    "video.codec": "libx265",
                    "video.pix_fmt": "yuv420p16le",
                    "video.is_depth_map": True,
                    "has_audio": False,
                },
            },
            "observation.images.depth.head": {
                "dtype": "video",
                "shape": [720, 720, 3],
                "names": ["height", "width", "depth"],
                "info": {
                    "video.fps": 30.0,
                    "video.height": 720,
                    "video.width": 720,
                    "video.channels": 3,
                    "video.codec": "libx265",
                    "video.pix_fmt": "yuv420p16le",
                    "video.is_depth_map": True,
                    "has_audio": False,
                },
            },
            "observation.images.seg_instance_id.left_wrist": {
                "dtype": "video",
                "shape": [480, 480, 3],
                "names": ["height", "width", "seg_instance_id"],
                "info": {
                    "video.fps": 30.0,
                    "video.height": 480,
                    "video.width": 480,
                    "video.channels": 3,
                    "video.codec": "libx265",
                    "video.pix_fmt": "yuv420p",
                    "has_audio": False,
                },
            },
            "observation.images.seg_instance_id.right_wrist": {
                "dtype": "video",
                "shape": [480, 480, 3],
                "names": ["height", "width", "seg_instance_id"],
                "info": {
                    "video.fps": 30.0,
                    "video.height": 480,
                    "video.width": 480,
                    "video.channels": 3,
                    "video.codec": "libx265",
                    "video.pix_fmt": "yuv420p",
                    "has_audio": False,
                },
            },
            "observation.images.seg_instance_id.head": {
                "dtype": "video",
                "shape": [720, 720, 3],
                "names": ["height", "width", "seg_instance_id"],
                "info": {
                    "video.fps": 30.0,
                    "video.height": 720,
                    "video.width": 720,
                    "video.channels": 3,
                    "video.codec": "libx265",
                    "video.pix_fmt": "yuv420p",
                    "has_audio": False,
                },
            },
            "action": {"dtype": "float32", "shape": [23], "names": None},
            "timestamp": {"dtype": "float64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "observation.cam_rel_poses": {"dtype": "float32", "shape": [21], "names": None},
            "observation.state": {"dtype": "float32", "shape": [259], "names": None},
            "observation.task_info": {"dtype": "float32", "shape": [None], "names": None},
        },
    }

    with open(f"{data_dir}/meta/info.json", "w") as f:
        json.dump(info, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, default="/home/svl/behavior")
    args = parser.parse_args()

    num_tasks = generate_task_json(args.data_dir)
    num_episodes, num_frames = generate_episode_json(args.data_dir)
    print(num_tasks, num_episodes, num_frames)

    generate_info_json(
        args.data_dir, fps=30, total_episodes=num_episodes, total_tasks=num_tasks, total_frames=num_frames
    )
