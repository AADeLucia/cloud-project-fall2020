#!/usr/bin/env python
"""
Modified by Alexandra DeLucia. Added:
- logging
- model saving
- switched to scikit-learn API to work with LIME
(https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn)
"""

import os
import math
import pickle
from argparse import ArgumentParser
import random
import logging

import xgboost
import xgboost_util
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tests", nargs="+", default=["KMeans", "PageRank", "SGD", "web_server", "tensorflow"])
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--log-file", type=str)
    parser.add_argument("--look-back", type=int, default=5, help="Also referred to as window size")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--predict-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results_dict = {}

    # Customize logging
    logger = logging.getLogger()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    if args.log_file:
        filehandler = logging.FileHandler(args.log_file, "w+")
        logger.addHandler(filehandler)
    
    # Set seed
    random.seed(args.seed)

    # Train model for each dataset
    TARGET_COLUMN = 'flow_size'
    for TEST_NAME in args.tests:
        logging.info(f"On test {TEST_NAME}")
        results_dict[TEST_NAME] = {}

        TRAINING_PATH = f"{args.data_dir}/{TEST_NAME}/training/"
        TEST_PATH = f"{args.data_dir}/{TEST_NAME}/test/"
        VALIDATION_PATH = f"{args.data_dir}/{TEST_NAME}/validation/"
        MODEL_PATH = f"{args.output_dir}/xgboost_{TEST_NAME}"
        
        training_files = [os.path.join(TRAINING_PATH, f) for f in os.listdir(TRAINING_PATH)]
        test_files = [os.path.join(TEST_PATH, f) for f in os.listdir(TEST_PATH)]
        validation_files = [os.path.join(VALIDATION_PATH, f) for f in os.listdir(VALIDATION_PATH)]

        scaling = xgboost_util.calculate_scaling(training_files)
        data = xgboost_util.prepare_files(training_files, args.look_back, scaling, TARGET_COLUMN)

        inputs, outputs = xgboost_util.make_io(data)
        
        # Original code set "eval_metric" to "mae"
        clf = xgboost.XGBRegressor(
            n_estimators=50,
            objective="reg:linear",
            booster="gbtree",
            max_depth=10,
            base_score=2,
            importance_type="gain", # For feature importance
            random_state=args.seed # Not set in original code
            )
        logging.info('Training started')
        clf.fit(inputs, outputs)

#        training = xgboost.DMatrix(inputs, outputs, feature_names = data[0][0].columns)
#        logging.info(f"Len outputs: {len(outputs)}")
#        model = xgboost.train(param, training, param['num_epochs'])
        clf.save_model(MODEL_PATH)
        
        def evaluate_model(files, write_to_simulator=False):
            real = []
            predicted = []
            for f in files:
                data = xgboost_util.prepare_files([f], args.look_back, scaling, TARGET_COLUMN)
                inputs, outputs = xgboost_util.make_io(data)

                y_pred = clf.predict(inputs)
                pred = y_pred.tolist()

                real += outputs
                predicted += pred

            return xgboost_util.score_predictions(real, predicted)

        for name, files in [("train", training_files), ("test", test_files), ("validation", validation_files)]:
            mae, mse, r2 = evaluate_model(files)
            results_dict[TEST_NAME][name] = {
                "mae": mae,
                "mse": mse,
                "r2": r2
            }
            logging.info(f"{name}\tMAE: {mae:.2}\tMSE: {mse:.2}\tR2: {r2:.2}")
        
        logging.info("------------------------------------")

    # Save all scores
    logging.info("Saving all results")
    with open(f"{args.output_dir}/xgboost_results.pkl", "wb") as f:
        pickle.dump(results_dict, f)


