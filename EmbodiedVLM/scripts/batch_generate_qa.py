#!/usr/bin/env python3
"""
Batch QA Generation Script

Processes multiple task directories from segmented trajectories and generates
Q&A pairs using the QAGenerationManager class, saving results to JSONL format.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add parent directories to path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'OmniGibson'))

try:
    from EmbodiedVLM.heuristics.qa_generation import QAGenerationManager
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


class BatchQAGenerator:
    """
    Processes multiple task directories to generate Q&A pairs using QAGenerationManager.
    """
    
    def __init__(self, input_root: str, output_file: str, raw_data_dir: str):
        """
        Initialize the batch QA generator.
        
        Args:
            input_root: Root directory containing segmented task directories
            output_file: Output JSONL file path
            raw_data_dir: Root directory containing raw data
        """
        self.input_root = Path(input_root)
        self.output_file = Path(output_file)
        self.raw_data_dir = Path(raw_data_dir)
        
        if not self.input_root.exists():
            raise FileNotFoundError(f"Input root directory not found: {input_root}")
        
        # Create output directory if it doesn't exist
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
    
    def get_task_directories(self):
        """
        Get all valid task directories from the input root.
        
        Returns:
            list: List of task directory paths
        """
        task_dirs = []
        for item in self.input_root.iterdir():
            if item.is_dir():
                # Check if directory contains segmented_scene_graph_0.json
                scene_graph_file = item / "segmented_scene_graph_0.json"
                if scene_graph_file.exists():
                    task_dirs.append(item)
        
        task_dirs.sort()  # Sort for consistent processing order
        return task_dirs
    
    def run(self):
        """
        Run the complete batch QA generation process using QAGenerationManager.
        """
        print("=== BATCH QA GENERATION ===")
        print(f"Input directory: {self.input_root}")
        print(f"Output file: {self.output_file}")
        print()
        
        try:
            # Initialize QA generation manager with the input root
            # The manager will automatically load all valid tasks
            manager = QAGenerationManager(str(self.input_root), str(self.raw_data_dir))
            
            print(f"📚 Loaded {manager.num_tasks} tasks from {self.input_root}")
            
            if manager.num_tasks == 0:
                print("❌ No valid tasks found for QA generation")
                return
            
            multi_forward_qa_pairs = []
            multi_inv_qa_pairs = []
            step_length = 7
            # Generate multi-step forward dynamics Q&A pairs for all tasks
            print("\n⏭️ Generating Multi-Step Forward Dynamics Q&A pairs...")
            multi_forward_qa_pairs = manager.generate("multi_forward_dynamics", step_length=step_length)
            print(f"✅ Generated {len(multi_forward_qa_pairs)} multi-step forward dynamics Q&A pairs")

            # Generate multi-step inverse dynamics Q&A pairs for all tasks
            print("\n⏪ Generating Multi-Step Inverse Dynamics Q&A pairs...")
            multi_inv_qa_pairs = manager.generate("multi_inverse_dynamics", step_length=step_length)
            print(f"✅ Generated {len(multi_inv_qa_pairs)} multi-step inverse dynamics Q&A pairs")
            
            forward_qa_pairs = []
            inverse_qa_pairs = []
            
            # Generate forward dynamics Q&A pairs for all tasks
            # print("\n⏭️ Generating Forward Dynamics Q&A pairs...")
            # forward_qa_pairs = manager.generate("forward_dynamics")
            # print(f"✅ Generated {len(forward_qa_pairs)} forward dynamics Q&A pairs")
            
            # Generate inverse dynamics Q&A pairs for all tasks
            # print("\n⏪ Generating Inverse Dynamics Q&A pairs...")
            # manager.clear_qa_pairs()
            # inverse_qa_pairs = manager.generate("inverse_dynamics")
            # print(f"✅ Generated {len(inverse_qa_pairs)} inverse dynamics Q&A pairs")
            
            # Combine all QA pairs
            all_qa_pairs = forward_qa_pairs + inverse_qa_pairs + multi_forward_qa_pairs + multi_inv_qa_pairs
            manager.qa_pairs = all_qa_pairs
            
            # Save results to JSONL
            manager.save_to_jsonl(str(self.output_file))
            
            print(f"\n📈 BATCH QA GENERATION SUMMARY:")
            print("=" * 40)
            print(f"⏭️ Forward Dynamics Q&A pairs: {len(forward_qa_pairs)}")
            print(f"⏪ Inverse Dynamics Q&A pairs: {len(inverse_qa_pairs)}")
            print(f"⏭️ Multi-Step Forward Dynamics Q&A pairs: {len(multi_forward_qa_pairs)}")
            print(f"⏪ Multi-Step Inverse Dynamics Q&A pairs: {len(multi_inv_qa_pairs)}")
            print(f"📊 Total Q&A pairs: {len(all_qa_pairs)}")
            print(f"💾 Output saved to: {self.output_file}")
            
        except Exception as e:
            print(f"❌ Error during QA generation: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Batch generate QA pairs from segmented trajectories",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'input_root',
        nargs='?',
        default='/home/mll-laptop-1/01_projects/03_behavior_challenge/segmented_replayed_trajecotries',
        help='Root directory containing segmented task directories'
    )
    
    parser.add_argument(
        'raw_data_dir',
        nargs='?',
        default='/home/mll-laptop-1/01_projects/03_behavior_challenge/replayed_trajectories',
        help='Root directory containing raw data'
    )
    
    parser.add_argument(
        'output_file',
        nargs='?',
        default='/home/mll-laptop-1/01_projects/BEHAVIOR-1K/EmbodiedVLM/tmp/gen_qa_step_7.jsonl',
        help='Output JSONL file path'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be processed without actually generating QA pairs'
    )
    
    args = parser.parse_args()
    
    print(f"Input root: {args.input_root}")
    print(f"Raw data dir: {args.raw_data_dir}")
    print(f"Output file: {args.output_file}")
    
    if args.dry_run:
        print("\n[DRY RUN MODE] - No QA pairs will be generated")
        input_root = Path(args.input_root)
        if input_root.exists():
            task_dirs = [d for d in input_root.iterdir() 
                        if d.is_dir() and (d / "segmented_scene_graph_0.json").exists()]
            print(f"Would process {len(task_dirs)} task directories:")
            for task_dir in sorted(task_dirs):
                print(f"  - {task_dir.name}")
        else:
            print(f"Input directory does not exist: {args.input_root}")
        return
    
    try:
        generator = BatchQAGenerator(args.input_root, args.output_file, args.raw_data_dir)
        generator.run()
        print("\n🎉 Batch QA generation complete!")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
