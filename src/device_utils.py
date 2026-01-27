"""
Device detection utilities - Auto-detects GPU/accelerator based on platform
Works on: Windows (CUDA/CPU), Mac (MPS/CPU), Linux (CUDA/CPU)
"""

import torch
import platform
import logging

logger = logging.getLogger(__name__)


def get_optimal_device():
    """
    Auto-detect the best available device for the current platform.
    
    Priority order:
    1. CUDA (NVIDIA GPU) - Windows, Linux
    2. MPS (Apple Silicon) - Mac
    3. CPU - Fallback for all platforms
    
    Returns:
        str: Device name ('cuda', 'mps', or 'cpu')
    """
    system = platform.system()
    
    # Check CUDA first (Windows, Linux)
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"✓ CUDA available on {system}: {gpu_name}")
        return device
    
    # Check MPS (Mac only)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Check if MPS is actually working (can have build issues)
        try:
            test_tensor = torch.zeros(1, device='mps')
            device = "mps"
            logger.info(f"✓ MPS available on {system} (Apple Silicon)")
            return device
        except Exception as e:
            logger.warning(f"MPS detected but not functional: {e}. Falling back to CPU")
            return "cpu"
    
    # Fallback to CPU
    logger.info(f"No GPU acceleration available on {system}. Using CPU")
    return "cpu"


def get_optimal_num_workers():
    """
    Get optimal num_workers for DataLoader based on platform.
    
    Returns:
        int: Recommended num_workers
             - 0 for Mac (avoids multiprocessing issues)
             - 4 for Linux/Windows (can use multiprocessing)
    """
    system = platform.system()
    
    if system == "Darwin":  # macOS
        workers = 0
        logger.info(f"macOS detected: num_workers={workers} (multiprocessing disabled)")
    else:
        workers = 0
        logger.info(f"{system} detected: num_workers={workers}")
    
    return workers


def validate_device(device_str: str) -> str:
    """
    Validate if requested device is available, fallback if not.
    
    Args:
        device_str: Requested device ('cuda', 'mps', 'cpu', or 'auto')
    
    Returns:
        str: Valid device name
    """
    if device_str == "auto":
        return get_optimal_device()
    
    if device_str == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        logger.warning("CUDA requested but not available. Falling back to CPU")
        return "cpu"
    
    if device_str == "mps":
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                test_tensor = torch.zeros(1, device='mps')
                return "mps"
            except Exception:
                logger.warning("MPS requested but not functional. Falling back to CPU")
                return "cpu"
        logger.warning("MPS requested but not available (Mac only). Falling back to CPU")
        return "cpu"
    
    if device_str == "cpu":
        return "cpu"
    
    logger.warning(f"Unknown device '{device_str}'. Using CPU")
    return "cpu"


def print_device_info():
    """Print detailed device information for debugging"""
    system = platform.system()
    print(f"\n{'='*50}")
    print(f"System: {system} {platform.release()}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"MPS Available: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
    print(f"Optimal Device: {get_optimal_device()}")
    print(f"Optimal num_workers: {get_optimal_num_workers()}")
    print(f"{'='*50}\n")
