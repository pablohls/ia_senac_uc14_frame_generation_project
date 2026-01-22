"""
Video Processing Module
Handles video I/O operations, frame extraction and assembly
"""

import cv2
import os
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Handles video processing operations"""
    
    def __init__(self, video_path: str):
        """Initialize VideoProcessor"""
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        self.fps = self. cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video loaded: {self.width}x{self.height} @ {self.fps}fps")
        
    def extract_frames(self, output_dir: str, max_frames: Optional[int] = None) -> List[np.ndarray]:
        """Extract frames from video"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        frames = []
        frame_idx = 0
        
        logger.info(f"Extracting frames to {output_dir}")
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or (max_frames and frame_idx >= max_frames):
                break
                
            frames.append(frame)
            frame_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.png")
            cv2.imwrite(frame_path, frame)
            frame_idx += 1
            
            if frame_idx % 100 == 0:
                logger.debug(f"Extracted {frame_idx} frames")
        
        self.cap.release()
        logger.info(f"Extracted {len(frames)} frames")
        return frames
    
    @staticmethod
    def frames_to_video(frames_dir: str, output_path: str, fps: int = 60, codec: str = 'mp4v') -> None:
        """Convert frames back to video"""
        frames = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.png', '.jpg'))])
        
        if not frames:
            raise ValueError(f"No frames found in {frames_dir}")
        
        logger.info(f"Assembling {len(frames)} frames into video @ {fps}fps")
        
        first_frame = cv2.imread(os.path. join(frames_dir, frames[0]))
        height, width, _ = first_frame.shape
        
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for i, frame_name in enumerate(frames):
            frame_path = os.path.join(frames_dir, frame_name)
            frame = cv2.imread(frame_path)
            out.write(frame)
            
            if (i + 1) % 100 == 0:
                logger.debug(f"Wrote {i + 1}/{len(frames)} frames")
        
        out.release()
        logger.info(f"Video saved to {output_path}")
    
    @staticmethod
    def resize_frame(frame: np.ndarray, target_size: Tuple[int, int], keep_aspect: bool = True) -> np.ndarray:
        """Resize frame to target size"""
        if keep_aspect:
            h, w = frame.shape[:2]
            target_w, target_h = target_size
            
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            if new_w < target_w or new_h < target_h:
                padded = np.zeros((target_h, target_w, 3), dtype=frame.dtype)
                y_offset = (target_h - new_h) // 2
                x_offset = (target_w - new_w) // 2
                padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
                return padded
            
            return resized
        else: 
            return cv2.resize(frame, target_size, interpolation=cv2.INTER_LANCZOS4)
    
    def get_video_info(self) -> dict:
        """Get video metadata"""
        return {
            "path": self.video_path,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "duration": self.frame_count / self.fps if self.fps > 0 else 0
        }
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap. release()
