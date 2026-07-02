#!/usr/bin/env python
"""
Download RLBench datasets for Chain-of-Action.

This script downloads RLBench demonstration datasets from HuggingFace Hub
and organizes them in the local filesystem for training and evaluation.
"""
import os
import sys
import argparse
from huggingface_hub import hf_hub_download
from typing import List


all_tasks = [
    "phone_on_base",
    "push_button",
    "pick_up_cup",
    "meat_off_grill",
    "open_door",
    "put_money_in_safe",
    "take_lid_off_saucepan",
    "open_washing_machine",
    "open_drawer",
    "take_umbrella_out_of_umbrella_stand",
    "open_box",
    "put_bottle_in_fridge",
    "put_knife_on_chopping_board",
    "reach_and_drag",
    "get_ice_from_fridge",
    "take_off_weighing_scales",
    "beat_the_buzz",
    "stack_wine",
    "turn_tap",
    "put_plate_in_colored_dish_rack",
    "take_frame_off_hanger",
    "slide_block_to_target",
    "move_hanger",
    "take_toilet_roll_off_stand",
    "open_microwave",
    "change_channel",
    "change_clock",
    "take_usb_out_of_computer",
    "insert_usb_in_computer",
    "close_fridge",
    "close_grill",
    "take_shoes_out_of_box",
    "hit_ball_with_queue",
    "lift_numbered_block",
    "hang_frame_on_hanger",
    "toilet_seat_up",
    "water_plants",
    "open_wine_bottle",
    "toilet_seat_down",
    "close_drawer",
    "close_box",
    "basketball_in_hoop",
    "put_groceries_in_cupboard",
    "hockey",
    "setup_checkers",
    "lamp_on",
    "open_grill",
    "turn_oven_on",
    "unplug_charger",
    "lamp_off",
    "take_plate_off_colored_dish_rack",
    "play_jenga",
    "place_hanger_on_rack",
    "push_buttons",
    "screw_nail",
    "straighten_rope",
    "take_money_out_safe",
    "reach_target",
    "sweep_to_dustpan",
    "press_switch"
]

subset_tasks = [
    "reach_target",
    "press_switch",
    "pick_up_cup",
    "open_drawer",
    "stack_wine",
    "open_box",
    "sweep_to_dustpan",
    "turn_tap",
    "push_button",
    "take_lid_off_saucepan"
]

rlbench_18_tasks = [
    "open_drawer",
    "close_drawer",
    "open_jar",
    "close_jar",
    "insert_onto_square_peg",
    "stack_blocks",
    "place_cups",
    "place_shape_in_shape_sorter",
    "put_groceries_in_cupboard",
    "put_item_in_drawer",
    "put_money_in_safe",
    "push_buttons",
    "reach_and_drag",
    "slide_block_to_target",
    "sweep_to_dustpan",
    "turn_tap",
    "light_bulb_in",
    "meat_off_grill",
]

def download_task_files(task_name: str, save_dir: str, train_episodes: int, eval_episodes: int,
                       repo_id: str = "Solomonz/chain-of-action") -> bool:
    """
    Download files for a specific task.
    """
    print(f"Downloading task: {task_name}")
    print(f"  Train episodes: {train_episodes}")
    print(f"  Eval episodes: {eval_episodes}")
    
    # Create directories
    train_dir = os.path.join(save_dir, "train", task_name, "variation0", "episodes")
    eval_dir = os.path.join(save_dir, "eval", task_name, "variation0", "episodes")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    
    success = True
    
    # Download train episodes
    if train_episodes > 0:
        # Download variation_descriptions.pkl for train
        variation_file = os.path.join(save_dir, "train", task_name, "variation0", "variation_descriptions.pkl")
        if os.path.exists(variation_file):
            print(f"  Skipping existing train variation_descriptions.pkl")
        else:
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename=f"train/{task_name}/variation0/variation_descriptions.pkl",
                    local_dir=save_dir,
                    local_dir_use_symlinks=False,
                    resume_download=True
                )
                print(f"  Downloaded train variation_descriptions.pkl")
            except Exception as e:
                print(f"  Warning: Failed to download train variation_descriptions.pkl: {e}")
                success = False
        # Download train episodes
        print(f"Downloading train episodes for {task_name}...")
        for episode in range(train_episodes):
            local_file = os.path.join(train_dir, f"episode{episode}", "low_dim_obs.pkl")
            if os.path.exists(local_file):
                print(f"  Skipping existing train episode {episode}")
                continue
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename=f"train/{task_name}/variation0/episodes/episode{episode}/low_dim_obs.pkl",
                    local_dir=save_dir,
                    local_dir_use_symlinks=False,
                    resume_download=True
                )
            except Exception as e:
                print(f"  Warning: Failed to download train episode {episode}: {e}")
                success = False

    
    # Download eval episodes
    if eval_episodes > 0:
        # Download variation_descriptions.pkl for eval
        variation_file = os.path.join(save_dir, "eval", task_name, "variation0", "variation_descriptions.pkl")
        if os.path.exists(variation_file):
            print(f"  Skipping existing eval variation_descriptions.pkl")
        else:
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename=f"eval/{task_name}/variation0/variation_descriptions.pkl",
                    local_dir=save_dir,
                    local_dir_use_symlinks=False,
                    resume_download=True
                )
                print(f"  Downloaded eval variation_descriptions.pkl")
            except Exception as e:
                print(f"  Warning: Failed to download eval variation_descriptions.pkl: {e}")
                success = False

        # Download eval episodes
        print(f"Downloading eval episodes for {task_name}...")
        for episode in range(eval_episodes):
            local_file = os.path.join(eval_dir, f"episode{episode}", "low_dim_obs.pkl")
            if os.path.exists(local_file):
                print(f"  Skipping existing eval episode {episode}")
                continue
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename=f"eval/{task_name}/variation0/episodes/episode{episode}/low_dim_obs.pkl",
                    local_dir=save_dir,
                    local_dir_use_symlinks=False,
                    resume_download=True
                )
            except Exception as e:
                print(f"  Warning: Failed to download eval episode {episode}: {e}")
                success = False

    
    print(f"Completed downloading task: {task_name}")
    return success

def main():
    parser = argparse.ArgumentParser(description="Download RLBench datasets for Chain-of-Action")
    parser.add_argument("--task", type=str, help="Task name to download dataset for")
    parser.add_argument("--train-episodes", type=int, default=100, help="Number of train episodes (default: 100)")
    parser.add_argument("--eval-episodes", type=int, default=25, help="Number of eval episodes (default: 25)")
    parser.add_argument("--subset", action="store_true", help="Download only subset tasks")
    parser.add_argument("--rlbench18", action="store_true", help="Download the 18-task RLBench benchmark")
    parser.add_argument("--save-dir", type=str, default="data/rlbench", help="Directory to save datasets")
    parser.add_argument("--repo-id", type=str, default="Solomonz/chain-of-action", 
                       help="HuggingFace repository ID")
    parser.add_argument("--list-tasks", action="store_true", help="List all available tasks")
    parser.add_argument("--return-path", action="store_true", help="Return dataset path only (for scripting)")
    
    args = parser.parse_args()
    
    if args.list_tasks:
        for i, task in enumerate(all_tasks, 1):
            print(f"{i:2d}. {task}")
        return
    
    # 选择任务列表
    if args.task:
        tasks_to_download = [args.task]
        print(f"Downloading single task: {args.task}")
    elif args.subset:
        tasks_to_download = subset_tasks
        print(f"Downloading subset tasks ({len(tasks_to_download)} tasks)")
    elif args.rlbench18:
        tasks_to_download = rlbench_18_tasks
        print(f"Downloading RLBench-18 benchmark tasks ({len(tasks_to_download)} tasks)")
    else:
        tasks_to_download = all_tasks
        print(f"Downloading all tasks ({len(tasks_to_download)} tasks)")
    
    print(f"Train episodes per task: {args.train_episodes}")
    print(f"Eval episodes per task: {args.eval_episodes}")
    print(f"Save directory: {args.save_dir}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    success_count = 0
    for task in tasks_to_download:
        if download_task_files(task, args.save_dir, args.train_episodes, args.eval_episodes, args.repo_id):
            success_count += 1
    
    print(f"Download completed!")
    print(f"Successfully downloaded: {success_count}/{len(tasks_to_download)} tasks")
    print(f"Data saved to: {args.save_dir}")

if __name__ == "__main__":
    main() 
