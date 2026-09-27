"""
Batch conversion script: convert existing MediaPipe avatar-data JSON files
to SMPL-X parameter format.

Reads from public/avatar-data/<label>.json files (each containing 345-float
frames), fits SMPL-X parameters using the IK fallback, and writes the results
to public/avatar-data-smplx/<label>.smplx.json.

Usage:
    python convert_avatar_data.py [--input-dir public/avatar-data] [--output-dir public/avatar-data-smplx]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from fit_smplx import SmplxFitter, TOTAL_FLOATS_PER_FRAME


def convert_frame(raw_frame: list[float], fitter: SmplxFitter) -> dict:
    """Convert a single landmark frame to SMPL-X parameters."""
    landmarks = np.asarray(raw_frame, dtype=np.float64)
    if landmarks.size != TOTAL_FLOATS_PER_FRAME:
        raise ValueError(f"Expected {TOTAL_FLOATS_PER_FRAME} floats, got {landmarks.size}")

    result = fitter.fit_frame(landmarks)

    return {
        "pose": result.pose,
        "betas": result.betas,
        "transl": result.transl,
        "global_orient": result.global_orient,
        "smplx_joints": result.smplx_joints,
        "confidence": result.confidence,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert avatar data to SMPL-X format")
    parser.add_argument("--input-dir", default="public/avatar-data", help="Input directory with .json avatar data")
    parser.add_argument("--output-dir", default="public/avatar-data-smplx", help="Output directory for SMPL-X data")
    parser.add_argument("--model-dir", default=None, help="Directory containing SMPL-X model files (optional)")
    parser.add_argument("--gender", default="NEUTRAL", help="SMPL-X gender (NEUTRAL, MALE, FEMALE)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the fitter
    fitter = SmplxFitter(model_dir=args.model_dir, gender=args.gender)

    # Get all JSON files except manifest.json
    json_files = sorted([f for f in input_dir.glob("*.json") if f.name != "manifest.json"])
    print(f"Found {len(json_files)} avatar data files to convert")

    for json_file in json_files:
        label = json_file.stem
        output_file = output_dir / f"{label}.smplx.json"

        # Skip if already converted
        if output_file.exists():
            print(f"  Skipping {label} (already converted)")
            continue

        print(f"  Converting {label}...")

        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            frames_data = data.get("frames", [])
            if not frames_data:
                print(f"    WARNING: {label} has no frames, skipping")
                continue

            frame_count = data.get("frameCount", len(frames_data) if isinstance(frames_data[0], list) else len(frames_data) // TOTAL_FLOATS_PER_FRAME)

            # Handle both flat and nested formats
            if isinstance(frames_data[0], list):
                flat_frames = [val for frame in frames_data for val in frame]
            else:
                flat_frames = frames_data

            total_floats = len(flat_frames)
            n_frames = total_floats // TOTAL_FLOATS_PER_FRAME

            smplx_frames = []
            for i in range(n_frames):
                frame_start = i * TOTAL_FLOATS_PER_FRAME
                frame_data = flat_frames[frame_start:frame_start + TOTAL_FLOATS_PER_FRAME]
                smplx_frame = convert_frame(frame_data, fitter)
                smplx_frames.append(smplx_frame)

            output = {
                "label": label,
                "frameRate": 30,
                "frameCount": n_frames,
                "frames": smplx_frames,
            }

            with open(output_file, "w") as f:
                json.dump(output, f, indent=2)

            print(f"    Done: {n_frames} frames written to {output_file}")

        except Exception as e:
            print(f"    ERROR converting {label}: {e}", file=sys.stderr)

    print(f"\nConversion complete. Output in {output_dir}")


if __name__ == "__main__":
    main()
