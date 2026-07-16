import os
# Disable symlinks to prevent WinError 1314 on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

import json
import shutil
from pathlib import Path
import pandas as pd
from huggingface_hub import snapshot_download

base_dst_dir = Path(r"C:\TONGHOPTRENLOP\HK6\ML\Project\RISE\policy_and_value\policy_offline_and_value\datasets")

# The dummy stats from our local svla dataset, to satisfy mini_lerobot parser
dummy_stats_str = '{"episode_index": 0, "stats": {"observation.images.top_head": {"min": [[[0.0]], [[0.0]], [[0.0]]], "max": [[[1.0]], [[1.0]], [[1.0]]], "mean": [[[0.5]], [[0.5]], [[0.5]]], "std": [[[0.2]], [[0.2]], [[0.2]]], "count": [100]}, "observation.state": {"min": [-2.0], "max": [2.0], "mean": [0.0], "std": [0.5], "count": [100]}, "action": {"min": [-2.0], "max": [2.0], "mean": [0.0], "std": [0.5], "count": [100]}, "timestamp": {"min": [0.0], "max": [10.0], "mean": [5.0], "std": [2.0], "count": [100]}, "frame_index": {"min": [0], "max": [100], "mean": [50.0], "std": [25.0], "count": [100]}, "episode_index": {"min": [0], "max": [0], "mean": [0.0], "std": [0.0], "count": [100]}, "index": {"min": [0], "max": [100], "mean": [50.0], "std": [25.0], "count": [100]}, "task_index": {"min": [0], "max": [0], "mean": [0.0], "std": [0.0], "count": [100]}}}'
dummy_stats = json.loads(dummy_stats_str)

datasets = {
    "aloha_cabinet": "lerobot/aloha_mobile_cabinet",
    "aloha_ziploc": "lerobot/aloha_static_ziploc_slide"
}

# Clean up old unused datasets
for old_ds in ["aloha_transfer", "aloha_insertion", "svla_2", "svla_3"]:
    old_dir = base_dst_dir / old_ds
    if old_dir.exists():
        print(f"Removing old dataset: {old_ds}")
        shutil.rmtree(old_dir, ignore_errors=True)

for local_name, repo_id in datasets.items():
    print(f"Downloading subset of {repo_id}...")
    
    # Download only what we need: info, stats, tasks, first chunk of episodes, first chunk of data, first chunk of videos
    allow_patterns = [
        "meta/info.json",
        "meta/stats.json",
        "meta/tasks.parquet",
        "meta/episodes/chunk-000/file-000.parquet",
        "data/chunk-000/file-000.parquet",
        "videos/*/chunk-000/file-000.mp4"
    ]
    
    # We download to a temporary cache then copy out
    local_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=allow_patterns,
    )
    
    src_dir = Path(local_path)
    dst_dir = base_dst_dir / local_name
    os.makedirs(dst_dir / "meta", exist_ok=True)
    os.makedirs(dst_dir / "data" / "chunk-000", exist_ok=True)
    
    # 1. info.json
    with open(src_dir / "meta" / "info.json", "r") as f:
        info = json.load(f)
    
    # 2. Convert tasks.parquet -> tasks.jsonl
    tasks_df = pd.read_parquet(src_dir / "meta" / "tasks.parquet")
    tasks_records = tasks_df.to_dict(orient="records")
    with open(dst_dir / "meta" / "tasks.jsonl", "w") as f:
        for r in tasks_records:
            f.write(json.dumps(r) + "\n")
            
    # 3. Convert episodes parquet -> episodes.jsonl (only episode 0)
    eps_df = pd.read_parquet(src_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    # Convert safely to native python types
    ep0_record = json.loads(eps_df.iloc[0].to_json())
    with open(dst_dir / "meta" / "episodes.jsonl", "w") as f:
        f.write(json.dumps(ep0_record) + "\n")
        
    # 4. Dummy episodes_stats.jsonl
    dummy_stats["episode_index"] = 0
    with open(dst_dir / "meta" / "episodes_stats.jsonl", "w") as f:
        f.write(json.dumps(dummy_stats) + "\n")
        
    # Modify info.json for single episode and camera renaming
    info["total_episodes"] = 1
    info["total_frames"] = ep0_record["length"]
    if "splits" in info and "train" in info["splits"]:
        info["splits"]["train"] = "0:1"
        
    features = info.get("features", {})
    # Rename cameras in info.json
    cam_mapping = {
        "observation.images.cam_high": "observation.images.top_head",
        "observation.images.cam_left_wrist": "observation.images.hand_left",
        "observation.images.cam_right_wrist": "observation.images.hand_right",
        # Fallback for ziploc if it uses 'top' instead of 'cam_high'
        "observation.images.top": "observation.images.top_head"
    }
    
    new_features = {}
    for k, v in features.items():
        if k in cam_mapping:
            new_features[cam_mapping[k]] = v
        else:
            new_features[k] = v
    info["features"] = new_features

    with open(dst_dir / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=4)
        
    # 5. Copy stats.json (needed for normalization if used)
    shutil.copy2(src_dir / "meta" / "stats.json", dst_dir / "meta" / "stats.json")
        
    # 6. Copy data parquet
    shutil.copy2(src_dir / "data" / "chunk-000" / "file-000.parquet", dst_dir / "data" / "chunk-000" / "episode_000000.parquet")
    
    # 7. Copy videos
    video_src = src_dir / "videos"
    video_dst = dst_dir / "videos" / "chunk-000"
    for cam_dir in os.listdir(video_src):
        cam_src = video_src / cam_dir / "chunk-000" / "file-000.mp4"
        if cam_src.exists():
            # Rename camera directory to match our Pi0 requirement
            new_cam_dir = cam_dir
            for old_cam, new_cam in cam_mapping.items():
                if cam_dir == old_cam:
                    new_cam_dir = new_cam
            
            os.makedirs(video_dst / new_cam_dir, exist_ok=True)
            shutil.copy2(cam_src, video_dst / new_cam_dir / "episode_000000.mp4")
            
    print(f"Successfully created subset for {repo_id} at {dst_dir}")
