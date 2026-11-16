"""Hardware awareness and local embedding utility for zero-cost offline execution."""

import logging
import os
import platform
import subprocess
from dataclasses import asdict, dataclass

import psutil

logger = logging.getLogger("sellbird.hardware")


@dataclass
class HardwareProfile:
    os: str
    cpu_model: str
    physical_cores: int
    logical_threads: int
    total_ram_gb: float
    available_ram_gb: float
    gpu_detected: bool
    gpu_name: str
    gpu_vram_gb: float
    hardware_tier: str
    recommended_model: str
    fallback_model: str
    standard_laptop_config: dict
    high_performance_config: dict

    def to_dict(self) -> dict:
        return asdict(self)


def _detect_nvidia_gpu() -> tuple[bool, str, float]:
    """Detect NVIDIA GPU name and VRAM via nvidia-smi without external calls."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        ).strip()
        if out:
            first_line = out.splitlines()[0]
            parts = [p.strip() for p in first_line.split(",")]
            name = parts[0] if parts else "NVIDIA GPU"
            vram_mb = float(parts[1]) if len(parts) > 1 else 0.0
            return True, name, round(vram_mb / 1024.0, 2)
    except Exception:
        pass
    return False, "CPU / Integrated Graphics", 0.0


def detect_hardware() -> HardwareProfile:
    """Inspect system RAM, CPU, and GPU and recommend a suitable local Qwen model."""
    vm = psutil.virtual_memory()
    total_ram_gb = round(vm.total / (1024 ** 3), 2)
    available_ram_gb = round(vm.available / (1024 ** 3), 2)
    physical_cores = psutil.cpu_count(logical=False) or os.cpu_count() or 2
    logical_threads = psutil.cpu_count(logical=True) or physical_cores
    cpu_model = platform.processor() or platform.machine()
    gpu_detected, gpu_name, gpu_vram_gb = _detect_nvidia_gpu()

    standard_cfg = {
        "target_hardware": "Ordinary Laptop (8-16 GB RAM, CPU-only or 4 GB VRAM GPU)",
        "recommended_model": "qwen3:8b (or qwen3:4b if <12 GB RAM)",
        "embedding_model": "all-MiniLM-L6-v2 (CPU/GPU, ~90 MB)",
        "num_ctx": 4096,
    }
    high_perf_cfg = {
        "target_hardware": "High-Performance Workstation (32+ GB RAM or >=8 GB VRAM GPU)",
        "recommended_model": "qwen3:14b or qwen3:32b",
        "embedding_model": "all-MiniLM-L6-v2",
        "num_ctx": 8192,
    }

    if total_ram_gb >= 28 and (not gpu_detected or gpu_vram_gb >= 8.0):
        tier = "high_performance"
        rec_model = "qwen3:14b"
        fb_model = "qwen3:8b"
    elif total_ram_gb >= 12:
        tier = "standard_gpu" if gpu_detected else "standard_cpu"
        rec_model = "qwen3:8b"
        fb_model = "qwen3:4b"
    else:
        tier = "lightweight_laptop"
        rec_model = "qwen3:4b"
        fb_model = "qwen3:1.7b"

    return HardwareProfile(
        os=f"{platform.system()} {platform.release()}",
        cpu_model=cpu_model,
        physical_cores=physical_cores,
        logical_threads=logical_threads,
        total_ram_gb=total_ram_gb,
        available_ram_gb=available_ram_gb,
        gpu_detected=gpu_detected,
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram_gb,
        hardware_tier=tier,
        recommended_model=rec_model,
        fallback_model=fb_model,
        standard_laptop_config=standard_cfg,
        high_performance_config=high_perf_cfg,
    )


_EMBEDDER = None


def get_local_embedder():
    """Load Apache-2.0 SentenceTransformer model locally (cached after first download)."""
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDER


def semantic_similarity(text_a: str, text_b: str) -> float:
    """Compute local cosine similarity (0.0 to 100.0) using Sentence Transformers."""
    if not text_a.strip() or not text_b.strip():
        return 0.0
    model = get_local_embedder()
    vecs = model.encode([text_a, text_b], normalize_embeddings=True)
    score = float((vecs[0] @ vecs[1]) * 100.0)
    return round(max(0.0, min(100.0, score)), 2)


if __name__ == "__main__":
    import json
    profile = detect_hardware()
    print(json.dumps(profile.to_dict(), indent=2))
    print("\nPre-caching local SentenceTransformer (all-MiniLM-L6-v2)...")
    sim = semantic_similarity("AI clinical workflow automation for hospitals", "Healthcare hospital EMR software")
    print(f"Local embedding test similarity score: {sim}/100")