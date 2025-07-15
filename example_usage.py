#!/usr/bin/env python3
"""
Example usage of the crosscoder dashboard generator.
"""

from pathlib import Path
import subprocess
import sys

def main():
    """Example of how to use the dashboard generator."""
    
    # Path to a saved crosscoder checkpoint
    saves_dir = Path("./crosscoder/saves")
    
    # Find the most recent checkpoint
    latest_version = None
    latest_checkpoint = None
    
    for version_dir in saves_dir.glob("version_*"):
        version_num = int(version_dir.name.split("_")[1])
        if latest_version is None or version_num > latest_version:
            latest_version = version_num
            # Find the latest checkpoint in this version
            checkpoints = list(version_dir.glob("*.pt"))
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda p: int(p.stem))
    
    if latest_checkpoint is None:
        print("No checkpoints found in crosscoder/saves/")
        return 1
    
    print(f"Using checkpoint: {latest_checkpoint}")
    
    # Generate dashboard
    output_path = Path("./crosscoder_dashboard.html")
    
    cmd = [
        sys.executable,
        "generate_dashboard.py",
        str(latest_checkpoint),
        "-o", str(output_path),
        "-n", "20",  # visualize 20 features
        "-s", "50",  # use 50 text samples
        "--hook-point", "blocks.6.hook_resid_post"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print(f"\n✓ Dashboard generated successfully: {output_path}")
        print(f"Open {output_path} in your browser to view the visualization")
    else:
        print(f"\n✗ Dashboard generation failed with code {result.returncode}")
        return result.returncode
    
    return 0

if __name__ == "__main__":
    exit(main())