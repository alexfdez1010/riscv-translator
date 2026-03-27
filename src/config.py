import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
REFERENCE_FILE = PROJECT_DIR / "docs" / "riscv-reference" / "reference.md"

# ---------------------------------------------------------------------------
# SSH
# ---------------------------------------------------------------------------

SSH_HOST = os.getenv("SSH_HOST", "final")
SSH_JUMP_HOST = os.getenv("SSH_JUMP_HOST", "")
REMOTE_DIR = os.getenv("REMOTE_DIR", "/tmp/sse2rvv")
SSH_CC = os.getenv("SSH_CC", "clang")
SSH_CXX = os.getenv("SSH_CXX", "clang++")

# ---------------------------------------------------------------------------
# LLM (OpenRouter)
# ---------------------------------------------------------------------------

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "minimax/minimax-m2.7")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
LLM_MAX_COMPLETION_TOKENS = int(os.getenv("LLM_MAX_COMPLETION_TOKENS", "5000"))

# ---------------------------------------------------------------------------
# Translation pipeline
# ---------------------------------------------------------------------------

LLM_VALIDATION_RETRIES = int(os.getenv("LLM_VALIDATION_RETRIES", "2"))
REACT_MAX_STEPS = int(os.getenv("REACT_MAX_STEPS", "15"))
VALIDATION_TIMEOUT_SECONDS = int(os.getenv("VALIDATION_TIMEOUT_SECONDS", "120"))

# ---------------------------------------------------------------------------
# Docker / RISC-V toolchain
# ---------------------------------------------------------------------------

DOCKER_IMAGE = os.getenv("DOCKER_IMAGE", "luispimo/riscv-toolchain:arm64-2025-10-20")
RISCVCC = os.getenv("RISCVCC", "riscv64-unknown-elf-gcc")
RISCVCXX = os.getenv("RISCVCXX", "riscv64-unknown-elf-g++")
DATASETS_DIR = Path(os.getenv("DATASETS_DIR", str(PROJECT_DIR / "datasets")))
VLEN = int(os.getenv("VLEN", "128"))
SIMULATOR = os.getenv(
    "SIMULATOR",
    f"spike --isa=rv64gcv pk64",
)

# ---------------------------------------------------------------------------
# RVV Reference context (loaded once for LLM prompts)
# ---------------------------------------------------------------------------

RVV_REFERENCE = REFERENCE_FILE.read_text() if REFERENCE_FILE.exists() else ""
