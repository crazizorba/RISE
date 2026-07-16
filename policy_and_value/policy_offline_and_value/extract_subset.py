import os
import shutil
import json
from pathlib import Path

src_dir = Path(r"C:\TONGHOPTRENLOP\HK6\ML\Project\lerobot_output_root\svla_subset")
base_dst_dir = Path(r"C:\TONGHOPTRENLOP\HK6\ML\Project\RISE\policy_and_value\policy_offline_and_value\datasets")

# Lấy 3 tập dữ liệu (episode 0, 1, 2)
episodes_to_extract = [0, 1, 2]
dataset_names = ["svla_1", "svla_2", "svla_3"]

with open(src_dir / "meta" / "episodes.jsonl", "r") as f:
    all_episodes = [json.loads(line) for line in f]
with open(src_dir / "meta" / "episodes_stats.jsonl", "r") as f:
    all_episodes_stats = [json.loads(line) for line in f]
with open(src_dir / "meta" / "tasks.jsonl", "r") as f:
    all_tasks = [json.loads(line) for line in f]
with open(src_dir / "meta" / "info.json", "r") as f:
    info = json.load(f)

for ep_idx, ds_name in zip(episodes_to_extract, dataset_names):
    dst_dir = base_dst_dir / ds_name
    os.makedirs(dst_dir / "meta", exist_ok=True)
    os.makedirs(dst_dir / "data" / "chunk-000", exist_ok=True)

    # Chỉ lấy đúng 1 episode
    subset_episodes = [all_episodes[ep_idx]]
    # Cập nhật lại index = 0 cho đúng chuẩn format
    subset_episodes[0]["episode_index"] = 0
    
    subset_stats = [all_episodes_stats[ep_idx]]
    subset_stats[0]["episode_index"] = 0

    info["total_episodes"] = 1
    info["total_frames"] = subset_episodes[0]["length"]
    if "splits" in info and "train" in info["splits"]:
        info["splits"]["train"] = "0:1"

    with open(dst_dir / "meta" / "episodes.jsonl", "w") as f:
        f.write(json.dumps(subset_episodes[0]) + "\n")
    with open(dst_dir / "meta" / "episodes_stats.jsonl", "w") as f:
        f.write(json.dumps(subset_stats[0]) + "\n")
    with open(dst_dir / "meta" / "tasks.jsonl", "w") as f:
        for t in all_tasks:
            f.write(json.dumps(t) + "\n")
    with open(dst_dir / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    # Parquet file
    filename = f"episode_{ep_idx:06d}.parquet"
    src_file = src_dir / "data" / "chunk-000" / filename
    dst_file = dst_dir / "data" / "chunk-000" / "episode_000000.parquet"
    if src_file.exists():
        shutil.copy2(src_file, dst_file)

    # Video files
    video_src = src_dir / "videos" / "chunk-000"
    video_dst = dst_dir / "videos" / "chunk-000"
    if video_src.exists():
        for cam_dir in os.listdir(video_src):
            os.makedirs(video_dst / cam_dir, exist_ok=True)
            src_vid = video_src / cam_dir / f"episode_{ep_idx:06d}.mp4"
            dst_vid = video_dst / cam_dir / "episode_000000.mp4"
            if src_vid.exists():
                shutil.copy2(src_vid, dst_vid)

    print(f"Created new dataset: {ds_name}")
