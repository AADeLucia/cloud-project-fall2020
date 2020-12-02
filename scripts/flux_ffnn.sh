#!/bin/bash

PROJECT_HOME="/home/aadelucia/files/course_projects/cloud-project-fall2020"
DATA_DIR="${PROJECT_HOME}/flows"
OUTPUT_DIR="${PROJECT_HOME}/flux/ml/results"
mkdir -p "${OUTPUT_DIR}"

python "${PROJECT_HOME}/flux/ml/ffnn.py" \
    --tests KMeans \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --log-file "${OUTPUT_DIR}/log.out" \
    --debug \
    --predict-only \

