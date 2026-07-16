import os
import json
import shutil
from pathlib import Path

base_dst_dir = Path(r"C:\TONGHOPTRENLOP\HK6\ML\Project\RISE\policy_and_value\policy_offline_and_value\datasets")

for ds_name in ["aloha_transfer", "aloha_insertion"]:
    dst_dir = base_dst_dir / ds_name
    
    # Fix info.json
    info_path = dst_dir / "meta" / "info.json"
    with open(info_path, "r") as f:
        info = json.load(f)
        
    features = info.get("features", {})
    if "observation.images.top" in features:
        top_feat = features.pop("observation.images.top")
        features["observation.images.top_head"] = top_feat
        features["observation.images.hand_left"] = top_feat
        features["observation.images.hand_right"] = top_feat
        
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)
        
    # Duplicate videos
    video_dir = dst_dir / "videos" / "chunk-000"
    top_dir = video_dir / "observation.images.top"
    
    if top_dir.exists():
        # Rename top -> top_head
        top_head_dir = video_dir / "observation.images.top_head"
        os.rename(top_dir, top_head_dir)
        
        # Copy to hand_left
        hand_left_dir = video_dir / "observation.images.hand_left"
        if not hand_left_dir.exists():
            shutil.copytree(top_head_dir, hand_left_dir)
            
        # Copy to hand_right
        hand_right_dir = video_dir / "observation.images.hand_right"
        if not hand_right_dir.exists():
            shutil.copytree(top_head_dir, hand_right_dir)
            
    print(f"Fixed schema for {ds_name}")
