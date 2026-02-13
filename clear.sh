#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p results

find results -type f \( -name "*.csv" -o -name "*.csv.lock" \) -delete
find logs -type f \( -name "*.out" \) -delete
find results -maxdepth 1 -type f \( -name ".cleanup_*.lock" -o -name ".cleanup_*.done" \) -delete
rm -rf results/_array_tmp

echo "Cleared result CSV outputs and temporary array files under results/."
