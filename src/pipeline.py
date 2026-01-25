"""
Frame Interpolation Pipeline
Complete end-to-end pipeline for video frame interpolation
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import time

from .video_processor import VideoProcessor
from .interpolator import FrameInterpolator
from .device_utils import validate_device, get_optimal_num_workers

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class InterpolationPipeline:
    """Complete pipeline for video frame interpolation"""
    
    def __init__(self, model_name: str = "film", config_path: Optional[str] = "config.yaml"):
        """Initialize pipeline"""
        self.model_name = model_name
        self.config = self._load_config(config_path) if config_path else {}
        self.interpolator = None
        
        logger.info(f"Pipeline initialized with model: {model_name}")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {}
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def initialize(self) -> None:
        """Initialize the interpolation model"""
        logger.info("Initializing interpolation model...")
        
        # Use device from config, or auto-detect
        device_config = self.config.get('model', {}).get('device', 'auto')
        device = validate_device(device_config)
        
        self.interpolator = FrameInterpolator(model_name=self.model_name, device=device)
        self.interpolator.load_model()
        
        logger.info("Pipeline ready!")
    
    def process_video(self, input_path: str, output_path: str, target_fps: Optional[int] = None,
                     multiplier: Optional[int] = None, max_resolution: Optional[tuple] = None,
                     save_frames: bool = False) -> Dict[str, Any]:
        """Process video with frame interpolation"""
        start_time = time.time()
        
        logger.info(f"Processing video: {input_path}")
        
        if self.interpolator is None:
            self.initialize()
        
        processor = VideoProcessor(input_path)
        video_info = processor.get_video_info()
        
        logger.info(f"Video info: {video_info}")
        
        original_fps = video_info['fps']
        
        if multiplier is None:
            if target_fps is None:
                target_fps = self.config.get('processing', {}).get('target_fps', 60)
            multiplier = max(int(target_fps / original_fps), 1)
        
        if target_fps is None:
            target_fps = int(original_fps * multiplier)
        
        logger.info(f"Original FPS: {original_fps}, Target FPS: {target_fps}, Multiplier: {multiplier}")
        
        temp_dir = Path("temp") / f"frames_{Path(input_path).stem}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Extracting frames...")
        frames = processor.extract_frames(str(temp_dir / "original"))
        
        if max_resolution:
            logger.info(f"Resizing frames to max resolution: {max_resolution}")
            frames = [processor.resize_frame(f, max_resolution, keep_aspect=True) for f in frames]
        
        logger.info(f"Interpolating frames...")
        interpolated_frames = self.interpolator.interpolate_sequence(frames, multiplier=multiplier)
        
        interpolated_dir = temp_dir / "interpolated"
        interpolated_dir.mkdir(exist_ok=True)
        
        import cv2
        for i, frame in enumerate(interpolated_frames):
            frame_path = interpolated_dir / f"frame_{i: 06d}.png"
            cv2.imwrite(str(frame_path), frame)
        
        logger.info("Assembling output video...")
        VideoProcessor.frames_to_video(str(interpolated_dir), output_path, fps=target_fps)
        
        if not save_frames:
            import shutil
            shutil.rmtree(temp_dir)
            logger.info("Temporary frames cleaned up")
        else:
            logger.info(f"Frames saved to:  {temp_dir}")
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        stats = {
            "input_path": input_path,
            "output_path": output_path,
            "original_fps": original_fps,
            "target_fps": target_fps,
            "multiplier":  multiplier,
            "original_frames": len(frames),
            "interpolated_frames": len(interpolated_frames),
            "processing_time_seconds": round(processing_time, 2),
        }
        
        logger.info(f"Processing complete!  Stats: {stats}")
        
        return stats


if __name__ == "__main__": 
    import argparse
    
    parser = argparse.ArgumentParser(description="Video Frame Interpolation Pipeline")
    parser.add_argument("input", help="Input video file")
    parser.add_argument("output", help="Output video file")
    parser.add_argument("--fps", type=int, default=60, help="Target frame rate")
    parser.add_argument("--model", default="film", choices=["film", "rife", "cain"])
    parser.add_argument("--save-frames", action="store_true")
    
    args = parser. parse_args()
    
    pipeline = InterpolationPipeline(model_name=args.model)
    pipeline.initialize()
    pipeline.process_video(args.input, args.output, target_fps=args.fps, save_frames=args.save_frames)
