#!/bin/sh
# docker-entrypoint.sh
#
# Runs before the server starts on every container launch.
# Creates missing workspace subdirectories so the app never fails on a
# fresh mount — safe to re-run because mkdir -p is idempotent.
#
# Expected container layout after this runs:
#   /workspace/data_src/aligned/
#   /workspace/data_dst/aligned/
#   /workspace/model/
#   /workspace/output/
#   /workspace/output_mask/   ← merge mask images
#   /backups/
#
# If the host already has these dirs they are untouched.
# Data inside the dirs is never created or deleted here.

set -e

mkdir -p \
    /workspace/data_src/aligned \
    /workspace/data_dst/aligned \
    /workspace/model \
    /workspace/output \
    /workspace/output_mask \
    /backups

exec "$@"
