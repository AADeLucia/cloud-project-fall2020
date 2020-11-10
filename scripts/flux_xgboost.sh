#!/bin/bash

PROJECT_HOME="/Users/kevinsherman/cloud-project-fall2020"
DATA_DIR="${PROJECT_HOME}/flux/data/ml"
OUTPUT_DIR="${PROJECT_HOME}/flux/ml/results_spark_features"
mkdir -p ${OUTPUT_DIR}

python3 "${PROJECT_HOME}/flux/ml/xgboost_learn.py" \
    --tests KMeans PageRank SGD \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --log-file "${OUTPUT_DIR}/log_xgboost.out" \
    --debug \
