#!/usr/bin/env bash
# Direction 2: set up a python venv WITHOUT Docker.
#   bash deploy/setup_venv.sh                # create .venv, choose the torch CUDA wheel from the driver shown by nvidia-smi
#   bash deploy/setup_venv.sh --dry-run      # only show what would be installed
#   bash deploy/setup_venv.sh --cuda cu121   # force a wheel (cu118 | cu121 | cu124 | cu126)
#   bash deploy/setup_venv.sh --dir /path/venv --python python3.11
# Afterwards:  source .venv/bin/activate  and run  bash scripts/d2_target_check.sh
set -euo pipefail

VENV=".venv"; PYBIN="python3"; FORCE=""; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dir) VENV="$2"; shift 2;;
    --python) PYBIN="$2"; shift 2;;
    --cuda) FORCE="$2"; shift 2;;
    --dry-run) DRY=1; shift;;
    -h|--help) sed -n '2,8p' "$0"; exit 0;;
    *) echo "unknown option: $1 (see --help)"; exit 2;;
  esac
done
cd "$(dirname "$0")/.."

die() { echo; echo "ERROR: $*" >&2; exit 1; }

# ---- 1. driver version -> torch wheel -------------------------------------------------------------------------------------------------
# NVIDIA minor-version compatibility: a cuXYZ wheel carries its own CUDA runtime and needs at least this driver:
#   cu126 >= 525.60 (best >= 560), cu124 >= 525.60 (best >= 550), cu121 >= 525.60, cu118 >= 450.80
DRIVER="${FAKE_DRIVER_VERSION:-}"
if [ -z "$DRIVER" ]; then
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found.
  - Is this machine really the GPU machine (and are you inside the GPU container / session)?
  - If nvidia-smi is not available the GPU driver is not visible here; ask the platform administrator.
  - To only prepare the venv without a GPU, use:  bash deploy/setup_venv.sh --cuda cu124"
  DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ' || true)"
fi
if [ -z "$FORCE" ] && [ -z "$DRIVER" ]; then
  die "could not read the driver version from nvidia-smi (run it by hand and look at 'Driver Version'), then use --cuda cuXXX"
fi
ver_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1)" = "$2" ]; }   # ver_ge A B  <=>  A >= B

PYV="$("$PYBIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)" || die "$PYBIN not found. Install python3 (sudo apt install python3 python3-venv python3-pip)."
PYMINOR="${PYV#*.}"
[ "${PYV%%.*}" = "3" ] && [ "$PYMINOR" -ge 9 ] || die "python $PYV is too old (need 3.9 - 3.13). Use --python python3.10 or install a newer one."

TAG="$FORCE"
if [ -z "$TAG" ]; then
  if   ver_ge "$DRIVER" 560; then TAG=cu126
  elif ver_ge "$DRIVER" 550; then TAG=cu124
  elif ver_ge "$DRIVER" 525.60; then TAG=cu121
  elif ver_ge "$DRIVER" 450.80; then TAG=cu118
  else
    die "driver $DRIVER is too old for any current PyTorch CUDA wheel (needs >= 450.80, CUDA 12 wheels need >= 525.60).
  Ask the platform administrator to update the NVIDIA driver to 525.60 or newer. A Docker image cannot help: the driver limits the CUDA version.
  If it cannot be updated, report to your supervisor that Direction 2 cannot run on this machine."
fi
fi
case "$TAG" in
  cu126) TORCH=2.6.0;;
  cu124) if [ "$PYMINOR" -ge 13 ]; then TORCH=2.6.0; else TORCH=2.5.1; fi;;
  cu121|cu118) TORCH=2.5.1
     [ "$PYMINOR" -le 12 ] || die "python $PYV is too new for the $TAG wheels (they stop at python 3.12). Use --python python3.12 or --cuda cu124.";;
  *) die "unknown --cuda value '$TAG' (use cu118, cu121, cu124 or cu126)";;
esac
[ "$TAG" = cu118 ] && echo "WARNING: cu118 is a CUDA 11 wheel (driver is old). It works for sm_75, but report this in your notes."
echo "driver version : ${DRIVER:-unknown (forced)}"
echo "python         : $PYBIN ($PYV)"
echo "torch wheel    : torch==$TORCH from https://download.pytorch.org/whl/$TAG"
echo "venv directory : $VENV"
if [ "$DRY" = 1 ]; then echo "(dry run: nothing installed)"; exit 0; fi

# ---- 2. venv + installs ---------------------------------------------------------------------------------------------------------------
if [ ! -x "$VENV/bin/python" ]; then
  "$PYBIN" -m venv "$VENV" || die "'$PYBIN -m venv' failed.
  - Ubuntu usually needs:  sudo apt install python3-venv     (or python3.X-venv)
  - Without sudo: '$PYBIN -m pip install --user virtualenv' then '$PYBIN -m virtualenv $VENV', or use conda: 'conda create -n d2 python=3.10'."
fi
PIP="$VENV/bin/python -m pip"
$PIP install --upgrade pip >/dev/null
echo "installing torch (large download, a few minutes) ..."
$PIP install "torch==$TORCH" --index-url "https://download.pytorch.org/whl/$TAG" || die "torch install failed.
  - No internet / blocked? Ask the platform for a PyPI + download.pytorch.org proxy, or download the wheel on another machine and 'pip install' the file.
  - Out of disk space? check 'df -h .'"
echo "installing requirements.txt ..."
$PIP install -r requirements.txt

# ---- 3. verify ------------------------------------------------------------------------------------------------------------------------
"$VENV/bin/python" - <<'PY'
import sys, torch
arch = torch.cuda.get_arch_list()
print("torch", torch.__version__, "| built for CUDA", torch.version.cuda, "| compiled archs", arch)
if "sm_75" not in arch:
    sys.exit("FAIL: this torch build has no sm_75 kernels (Quadro RTX 8000 needs them)")
if not torch.cuda.is_available():
    sys.exit("FAIL: torch.cuda.is_available() is False. Check 'nvidia-smi' and the driver version; see deploy/README_zh.md section 'common problems'")
x = torch.ones(1024, 1024, dtype=torch.float64, device="cuda")
assert float((x @ x).sum()) == 1024.0 ** 3
print("GPU:", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0), "| float64 matmul OK")
PY
echo
echo "OK. Next:  source $VENV/bin/activate && bash scripts/d2_target_check.sh"
