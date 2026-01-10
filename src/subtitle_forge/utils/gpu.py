"""GPU detection utilities."""

import logging

logger = logging.getLogger(__name__)


def get_available_vram() -> int:
    """
    Get available GPU VRAM in MB.

    Returns:
        Available VRAM in MB, or 0 if unable to detect.
    """
    try:
        import torch

        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            total = torch.cuda.get_device_properties(device).total_memory
            reserved = torch.cuda.memory_reserved(device)
            available = (total - reserved) // (1024 * 1024)
            return available
    except ImportError:
        logger.warning("PyTorch not installed, cannot detect GPU VRAM")
    except Exception as e:
        logger.warning(f"GPU VRAM detection failed: {e}")

    return 0


def get_optimal_compute_type(device: str) -> str:
    """
    Select optimal compute type based on device.

    Args:
        device: Device type (cuda/cpu)

    Returns:
        Compute type string.
    """
    if device == "cpu":
        return "int8"

    try:
        import torch

        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            # Ampere and above (SM 8.0+) have better int8 support
            if capability[0] >= 8:
                return "int8_float16"
            else:
                return "float16"
    except Exception:
        pass

    return "float16"


def check_cuda_available() -> bool:
    """Check if CUDA is available."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def get_gpu_info() -> dict:
    """Get GPU information."""
    info = {
        "cuda_available": False,
        "device_name": None,
        "total_vram_mb": 0,
        "available_vram_mb": 0,
    }

    try:
        import torch

        if torch.cuda.is_available():
            info["cuda_available"] = True
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            info["device_name"] = props.name
            info["total_vram_mb"] = props.total_memory // (1024 * 1024)
            info["available_vram_mb"] = get_available_vram()
    except ImportError:
        pass

    return info
