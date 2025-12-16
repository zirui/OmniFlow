"""Encode vidoe to H.264 format
"""

import os
import subprocess
import argparse
from multiprocessing import Pool, cpu_count

def reencode_task(args):
    in_path, out_path = args
    cmd = [
        "ffmpeg", "-y",
        "-i", in_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-c:a", "aac",
        out_path
    ]

    if os.path.exists(out_path):
        return f"Skip: {out_path}"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return f"OK: {out_path}"
    except Exception as e:
        return f"FAIL: {in_path}, error={str(e)}"


def batch_reencode_parallel(input_dir, output_dir):
    tasks = []

    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith((".mp4", ".mov", ".mkv")):
                in_path = os.path.join(root, f)
                rel = os.path.relpath(in_path, input_dir)
                out_path = os.path.join(output_dir, rel)
                tasks.append((in_path, out_path))

    print(f"Total videos: {len(tasks)}")

    with Pool(cpu_count()) as p:
        for result in p.imap_unordered(reencode_task, tasks):
            print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch re-encode videos to clean H.264 format")
    parser.add_argument("-i", "--input", required=True, help="Input directory path")
    parser.add_argument("-o", "--output", required=True, help="Output directory path")
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output

    os.makedirs(output_dir, exist_ok=True)
    # batch_reencode(input_dir, output_dir)
    batch_reencode_parallel(input_dir, output_dir)
