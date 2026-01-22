"""
Frame Interpolation Project
Video Frame Interpolation Pipeline using Deep Learning
"""

__version__ = "0.1.0"
__author__ = "SENAC UC14 Team"

from .video_processor import VideoProcessor
from .interpolator import FrameInterpolator
from . pipeline import InterpolationPipeline

__all__ = [
    "VideoProcessor",
    "FrameInterpolator",
    "InterpolationPipeline",
]
