"""

"""

import os
import json
import sys
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Union
from dataclasses import dataclass
from pathlib import Path

# Add the parent directory to sys.path to import scene_graph_utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'OmniGibson'))

from omnigibson.utils.scene_graph_utils import SceneGraphReader


@dataclass
class TaskData:
    """
    Data structure containing all information needed to process a single task.
    
    Attributes:
        task_name (str): Name of the task
        scene_graph_reader (SceneGraphReader): Reader for accessing scene graph data
        key_frame_ids (List[str]): List of key frame IDs in chronological order
        image_paths (Dict[str, Dict[str, str]]): Mapping from frame_id to {sensor_name: image_path}
        task_dir (str): Path to the task directory
    """
    task_name: str
    scene_graph_reader: SceneGraphReader
    key_frame_ids: List[str]
    image_paths: Dict[str, Dict[str, str]]
    task_dir: str


@dataclass
class QAPair:
    """
    Standard structure for a Q&A pair.
    
    Attributes:
        id (str): Unique identifier for this Q&A pair
        images (List[str]): List of image paths involved in the question
        question (str): The question text (prompt for VLM)
        gt_answer (Any): Ground truth answer
    """
    id: str
    images: List[str]
    question: str
    gt_answer: Any

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for serialization."""
        return {
            'id': self.id,
            'type': self.gt_answer['type'],
            'images': self.images,
            'question': self.question,
            'options': self.gt_answer['options'],
            'gt_answer': self.gt_answer['correct_option']
        }


class AbstractQAGenerator(ABC):
    """
    Abstract base class for all Q&A generators.
    
    This class defines the standard interface that all Q&A generators must implement,
    ensuring consistency across different generation strategies.
    """

    @abstractmethod
    def generate(self, task_data: TaskData) -> List[QAPair]:
        """
        Generate Q&A pairs for a given task.
        
        Args:
            task_data (TaskData): All data needed to process the task
            
        Returns:
            List[QAPair]: List of generated Q&A pairs
        """
        pass

    @property
    @abstractmethod
    def qa_type(self) -> str:
        """
        Return the type of Q&A this generator produces.
        
        Returns:
            str: The Q&A type identifier (e.g., "forward_dynamics", "inverse_dynamics")
        """
        pass


