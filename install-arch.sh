#!/usr/bin/env bash
set -euo pipefail

# Manjaro/Arch setup helper for this repository
TIMEOUT="${TIMEOUT:-25s}"

if ! command -v pacman >/dev/null 2>&1; then
  echo "This script is intended for Manjaro/Arch (pacman). Exiting."
  exit 1
fi

sudo -v || { echo "sudo privileges are required"; exit 1; }

sudo pacman -S --needed --noconfirm base-devel git curl wget jq ripgrep fd pkgconf cmake make unzip

# Python setup
sudo pacman -S --needed --noconfirm python python-pip python-virtualenv
if [ -f "requirements.txt" ]; then
  python -m venv .venv
  source .venv/bin/activate
  pip install -U pip
  pip install -r requirements.txt
fi

# GUI bindings if a GUI is detected
if rg -q --no-messages 'PyQt|tkinter|PySide' -n . 2>/dev/null; then
  sudo pacman -S --needed --noconfirm pyqt5 python-pyqt5 || true
fi

echo "[OK] Manjaro setup complete for $(basename "$PWD")."
