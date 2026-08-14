#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root/src/PTL"
exec python run_pearl_other_stations_v3_2021_2024.py "$@"
