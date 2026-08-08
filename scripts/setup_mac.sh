#!/usr/bin/env bash
# setup_mac.sh — auto-detect Apple Silicon, RAM, disk; install dermlat-audit + backend.
set -Eeuo pipefail

RED=$'\033[31m'; YEL=$'\033[33m'; GRN=$'\033[32m'; RST=$'\033[0m'

die() { echo "${RED}error${RST}: $*" >&2; exit 1; }
say() { echo "${GRN}==>${RST} $*"; }
warn(){ echo "${YEL}warn${RST}: $*"; }

# --- 1. hardware ---------------------------------------------------------
uname_s=$(uname -s)
arch=$(uname -m)
os_ver=$(sw_vers -productVersion 2>/dev/null || echo "unknown")
is_apple_silicon=false
chip="unknown"
if [[ "$uname_s" == "Darwin" && "$arch" == "arm64" ]]; then
  is_apple_silicon=true
  chip=$(/usr/sbin/sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Apple Silicon")
fi

ram_bytes=$(/usr/sbin/sysctl -n hw.memsize 2>/dev/null || echo 0)
ram_gb=$(( ram_bytes / 1024 / 1024 / 1024 ))
free_disk_kb=$(df -k . | awk 'NR==2 {print $4}')
free_disk_gb=$(( free_disk_kb / 1024 / 1024 ))

say "OS         : $uname_s $os_ver ($arch)"
say "Chip       : $chip"
say "RAM        : ${ram_gb} GB"
say "Free disk  : ${free_disk_gb} GB (in current directory)"

# --- 2. python -----------------------------------------------------------
if ! command -v python3.11 >/dev/null 2>&1 && ! command -v python3.12 >/dev/null 2>&1 && ! command -v python3 >/dev/null 2>&1; then
  die "no python3 found. Install via https://www.python.org/downloads/mac-osx/ or brew install python@3.11"
fi
py=$(command -v python3.12 || command -v python3.11 || command -v python3)
py_ver=$($py -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
say "Python     : $py ($py_ver)"

major=${py_ver%%.*}
minor=${py_ver##*.}
if [[ "$major" -lt 3 || ( "$major" -eq 3 && "$minor" -lt 11 ) ]]; then
  die "need Python 3.11+; found $py_ver"
fi

# --- 3. venv -------------------------------------------------------------
if [[ ! -d .venv ]]; then
  say "creating .venv"
  "$py" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
say "venv       : $(command -v python)"

python -m pip install --upgrade pip wheel setuptools

# --- 4. deps -------------------------------------------------------------
say "installing base package"
pip install -e ".[review,ollama,stats,dev]"

# Optional: matplotlib for figures
pip install "matplotlib>=3.8" || warn "matplotlib install failed — figures will be skipped"

# Try MLX only on Apple Silicon.
mlx_installed=false
if $is_apple_silicon; then
  if [[ "$ram_gb" -lt 8 ]]; then
    warn "less than 8 GB RAM — skipping MLX-VLM install."
  else
    say "installing MLX-VLM (Apple Silicon)"
    if pip install "mlx>=0.11" "mlx-vlm>=0.0.10"; then
      mlx_installed=true
    else
      warn "MLX-VLM install failed — you can still use the ollama or mock backends."
    fi
  fi
else
  warn "not Apple Silicon — MLX not installed. Use --backend ollama or --backend mock."
fi

# --- 5. ollama detection -------------------------------------------------
if command -v ollama >/dev/null 2>&1; then
  say "ollama     : $(command -v ollama) ($(ollama --version 2>/dev/null | head -1))"
else
  warn "ollama not installed. If you want the ollama backend, install: https://ollama.com"
fi

# --- 6. batch-size recommendation ---------------------------------------
if [[ "$ram_gb" -lt 12 ]]; then
  batch=16
elif [[ "$ram_gb" -lt 24 ]]; then
  batch=32
else
  batch=64
fi
say "recommended batch size = $batch"

# --- 7. summary ----------------------------------------------------------
cat <<EOF

${GRN}Setup complete.${RST}
- Apple Silicon detected : $is_apple_silicon ($chip)
- MLX-VLM installed      : $mlx_installed
- Recommended batch size : $batch

Next:
  source .venv/bin/activate
  dermlat-audit doctor --images /path/to/images --output /path/to/private-results
  ./scripts/run_audit.sh --images /path/to/images --annotations /path/to/annotations.csv --output /path/to/private-results
EOF
