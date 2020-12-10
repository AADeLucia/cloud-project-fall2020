#!/bin/bash

PROJECT_HOME="/home/aadelucia/files/course_projects/cloud-project-fall2020"
DATA_DIR="${PROJECT_HOME}/flows"
OUTPUT_DIR="${PROJECT_HOME}/results"
mkdir -p "${OUTPUT_DIR}"

python "${PROJECT_HOME}/code/xgboost_learn.py" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --log-file "${OUTPUT_DIR}/log_xgboost.out" \
    --debug \
