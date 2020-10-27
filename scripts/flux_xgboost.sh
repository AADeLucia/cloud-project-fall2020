#!/bin/bash

PROJECT_HOME="/Users/alexandradelucia/cloud_project"
DATA_DIR="${PROJECT_HOME}/flux/data/ml"
OUTPUT_DIR="${PROJECT_HOME}/flux/ml/results"

python ../flux/ml/xgboost_learn.py \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --log-file "${OUTPUT_DIR}/log_xgboost.out" \
    --debug \

