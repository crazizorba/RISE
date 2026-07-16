import os
import json
import pandas as pd
from pathlib import Path

base = Path("datasets")
for d in ["aloha_cabinet", "aloha_ziploc"]:
    dst = base / d
    parquet_path = dst / "data" / "episode_000000.parquet"
    
    # Read parquet
    df = pd.read_parquet(parquet_path)
    
    # Filter only episode 0
    df = df[df["episode_index"] == 0]
    num_frames = len(df)
    
    # Save back
    df.to_parquet(parquet_path)
    
    # Update info.json
    info_path = dst / "meta" / "info.json"
    with open(info_path, "r") as f:
        info = json.load(f)
    info["total_frames"] = num_frames
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)
        
    # Update episodes.jsonl
    ep_path = dst / "meta" / "episodes.jsonl"
    with open(ep_path, "w") as f:
        json.dump({
            "episode_index": 0,
            "tasks": [d],
            "length": num_frames
        }, f)
        f.write("\n")
        
    print(f"Truncated {d} to {num_frames} frames.")
