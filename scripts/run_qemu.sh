#!/usr/bin/env bash
set -euo pipefail

cat <<'EOF'
QEMU runtime skeleton is ready, but the platform command depends on your kernel/rootfs choice.

Suggested next step:
  1. Prepare an ARM rootfs or static binary flow
  2. Decide whether to use qemu-system-arm or qemu-arm
  3. Fill this script with the exact boot and run command

Use docs/phase_task_checklist.md phase 5 as the implementation checklist.
EOF
