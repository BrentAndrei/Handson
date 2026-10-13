#!/usr/bin/env python3
"""
Download FSL (Filipino Sign Language) videos from YouTube.

Usage:
    pip install yt-dlp
    python download_youtube.py --label "KNOW" --max-videos 3
    python download_youtube.py --queue capture_queue.json --top 10 --max-videos 2

The script searches YouTube for each sign label and downloads the top
matching videos. Videos are saved to a structured directory:
    youtube_raw/<LABEL>/<video_id>.mp4

You can then drag these files into the Capture Studio "Import" tab,
mark in/out points, and extract landmarks.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def sanitize_filename(s: str) -> str:
    """Make a string safe for use as a filename."""
    s = s.replace("/", "_").replace("\\", "_")
    s = re.sub(r'[<>:|"?*]', "_", s)
    return s.strip()


def safe_print(s: str):
    """Print safely handling encoding issues on Windows."""
    try:
        print(s)
    except UnicodeEncodeError:
        # Replace unencodable chars
        print(s.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding))


def search_youtube(query: str, max_results: int) -> List[dict]:
    """Search YouTube and return list of {id, title, url, duration}."""
    try:
        import yt_dlp
    except ImportError:
        safe_print("ERROR: yt-dlp not installed. Run: pip install yt-dlp")
        sys.exit(1)

    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": "in_playlist",
        "default_search": f"ytsearch{max_results}:",
        "js_runtimes": {"node": {}},
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(query, download=False)

    entries = result.get("entries", []) if result else []
    videos = []
    for entry in entries:
        if entry and entry.get("id"):
            videos.append({
                "id": entry["id"],
                "title": entry.get("title", "Unknown"),
                "url": f"https://www.youtube.com/watch?v={entry['id']}",
                "duration": entry.get("duration"),
            })
    return videos


def download_video(video_id: str, output_path: Path, max_duration: int = 300) -> bool:
    """Download a single video to output_path. Returns True on success."""
    try:
        import yt_dlp
    except ImportError:
        safe_print("ERROR: yt-dlp not installed. Run: pip install yt-dlp")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # More flexible format selection - try multiple fallbacks
    ydl_opts = {
        "format": "bv*[height<=720]+ba/b[height<=720]/bv*+ba/b",
        "outtmpl": str(output_path.with_suffix(".%(ext)s")),
        "quiet": False,
        "no_warnings": True,
        "match_filter": yt_dlp.utils.match_filter_func(f"duration <= {max_duration}"),
        "merge_output_format": "mp4",
        "js_runtimes": {"node": {}},
    }

    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return True
    except Exception as e:
        safe_print(f"  Failed to download {video_id}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", help="Single sign label to search for (e.g. KNOW)")
    ap.add_argument("--queue", type=Path, help="capture_queue.json to process top N labels")
    ap.add_argument("--top", type=int, default=10, help="Number of labels from queue to process")
    ap.add_argument("--max-videos", type=int, default=3, help="Max videos per label")
    ap.add_argument("--max-duration", type=int, default=300, help="Max video duration in seconds (default 5 min)")
    ap.add_argument("--output-dir", type=Path, default=Path("youtube_raw"),
                    help="Output directory (default: youtube_raw/)")
    ap.add_argument("--dry-run", action="store_true", help="Show what would be downloaded without downloading")
    args = ap.parse_args()

    if not args.label and not args.queue:
        ap.error("Need either --label or --queue")

    labels = []
    if args.label:
        labels = [args.label]
    else:
        with open(args.queue, "r") as f:
            data = json.load(f)
        for item in data.get("queue", [])[:args.top]:
            labels.append(item["label"])

    safe_print(f"Processing {len(labels)} label(s): {', '.join(labels)}")
    safe_print(f"Max {args.max_videos} videos per label, max duration {args.max_duration}s")
    safe_print(f"Output dir: {args.output_dir}")
    if args.dry_run:
        safe_print("DRY RUN - no downloads will occur")
    safe_print("")

    total_downloaded = 0
    for label in labels:
        safe_label = sanitize_filename(label)
        query = f"Filipino Sign Language {label}"
        safe_print(f"\n{'='*60}")
        safe_print(f"Label: {label}")
        safe_print(f"Search: {query}")

        videos = search_youtube(query, args.max_videos)
        if not videos:
            safe_print("  No results found")
            continue

        for i, v in enumerate(videos, 1):
            dur = v["duration"]
            dur_str = f"{dur//60}:{dur%60:02d}" if dur else "?"
            safe_print(f"  {i}. {v['title'][:60]} ({dur_str}) - {v['url']}")

            if args.dry_run:
                continue

            out_path = args.output_dir / safe_label / v["id"]
            # Check for any existing file with this video ID
            existing = list(out_path.parent.glob(f"{out_path.name}.*"))
            if existing:
                safe_print(f"    Already exists, skipping")
                continue

            safe_print(f"    Downloading...")
            if download_video(v["id"], out_path, args.max_duration):
                safe_print(f"    OK -> {out_path}.mp4")
                total_downloaded += 1
            else:
                safe_print(f"    FAILED")

    safe_print(f"\n{'='*60}")
    safe_print(f"Done. Downloaded {total_downloaded} new videos.")
    safe_print(f"Next steps:")
    safe_print(f"  1. Open Capture Studio -> Import tab")
    safe_print(f"  2. Drag videos from {args.output_dir}/<LABEL>/")
    safe_print(f"  3. Mark in/out points, extract landmarks")
    safe_print(f"  4. Test clips against classifier")
    safe_print(f"  5. Export fsl_dataset.json from Train tab")


if __name__ == "__main__":
    main()