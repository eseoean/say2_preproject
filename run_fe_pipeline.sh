#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /Users/skku_aws2_18/pre_project/toolchain_env.sh
source "$SCRIPT_DIR/project_env.sh"

PROFILE="${NXF_PROFILE:-awsbatch}"
if [[ $# -gt 0 ]]; then
  PROFILE="$1"
  shift
fi

cd "$SCRIPT_DIR/nextflow"

nextflow run main.nf \
  -profile "$PROFILE" \
  -c nextflow.config \
  -resume \
  "$@"
