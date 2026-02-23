"""
Task 1.1 — Video Utilities
Download, load, and iterate over video frames.
"""

import os
import cv2
import numpy as np
from typing import Generator, List, Tuple


def download_video(url: str, output_path: str, quality: str = "720") -> str:
    """
    Download a YouTube video using yt-dlp.

    Args:
        url: YouTube URL.
        output_path: Local file path to save the video (e.g. 'data/video.mp4').
        quality: Max height in pixels (e.g. '720', '480').

    Returns:
        Absolute path to the downloaded file.
    """
    try:
        import yt_dlp
    except ImportError:
        raise ImportError("yt-dlp is required. Install with: pip install yt-dlp")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Video-only — audio is irrelevant for CV inference and avoids
    # the ffmpeg merge requirement that bestvideo+bestaudio triggers.
    ydl_opts = {
        "format": (
            f"bestvideo[height<={quality}][ext=mp4]"
            f"/bestvideo[height<={quality}]"
            f"/bestvideo[ext=mp4]"
            f"/bestvideo"
        ),
        "outtmpl": output_path,
        "quiet": False,
        "no_warnings": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return os.path.abspath(output_path)


def get_video_metadata(video_path: str) -> dict:
    """
    Extract basic metadata from a video file.

    Returns dict with keys: fps, frame_count, width, height, duration_sec.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = frame_count / fps if fps > 0 else 0.0
    cap.release()

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": duration_sec,
    }


def sample_frames(
    video_path: str, n: int = 5
) -> List[Tuple[int, np.ndarray]]:
    """
    Read n evenly spaced frames from the video.

    Returns:
        List of (frame_index, BGR frame as numpy array).
    """
    meta = get_video_metadata(video_path)
    total = meta["frame_count"]
    indices = [int(i * (total - 1) / (n - 1)) for i in range(n)]

    cap = cv2.VideoCapture(video_path)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append((idx, frame))
    cap.release()
    return frames


def frame_generator(
    video_path: str, stride: int = 2
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Generator that yields (frame_index, BGR frame) at the given stride.

    Args:
        video_path: Path to the video file.
        stride: Process every `stride`-th frame (1 = all frames, 2 = every other, etc.).

    Yields:
        (frame_index, frame) tuples.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride == 0:
            yield frame_idx, frame
        frame_idx += 1

    cap.release()


def display_sample_frames(video_path: str, n: int = 5) -> None:
    """
    Display n evenly spaced frames from the video in a matplotlib grid.
    Requires matplotlib to be installed.
    """
    import matplotlib.pyplot as plt

    frames = sample_frames(video_path, n=n)
    cols = min(n, 5)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]

    meta = get_video_metadata(video_path)
    fps = meta["fps"]

    for i, (idx, frame) in enumerate(frames):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        ts = idx / fps if fps > 0 else 0
        ax.set_title(f"Frame {idx} — {ts:.1f}s")
        ax.axis("off")

    # Hide unused axes
    for i in range(len(frames), rows * cols):
        r, c = divmod(i, cols)
        axes[r][c].axis("off")

    plt.suptitle(f"Sample frames from: {os.path.basename(video_path)}", fontsize=14)
    plt.tight_layout()
    plt.show()
