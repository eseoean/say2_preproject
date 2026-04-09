#!/usr/bin/env bash

# Project-specific runtime defaults.
# Edit this file when you want to change user-specific settings.

export PROJECT_ROOT="/Users/skku_aws2_18/pre_project/say2_preproject"
export PROJECT_REMOTE_URL="https://github.com/eseoean/say2_preproject"

export S3_BASE="s3://say2-4team/20260409_eseo"
export DATA_DIR="$S3_BASE/data"
export FE_OUTPUT_DIR="$S3_BASE/fe_output"
export FE_RUN_ID="20260409_newfe_v8_eseo"

export TAG_KEY="pre-batch-2-4-team"
export TAG_VAL="20260409_eseo"

export AWS_REGION="ap-northeast-2"
export AWS_BATCH_QUEUE="team4-fe-queue-cpu"
export FE_CONTAINER="666803869796.dkr.ecr.ap-northeast-2.amazonaws.com/fe-v2-nextflow:v2-pip-awscli"
export NXF_S3_WORKDIR="$S3_BASE/nextflow_work"
