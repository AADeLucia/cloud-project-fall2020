#!/usr/bin/env python
"""
Modified by Alexandra DeLucia. Added:
- logging
- model saving
- switched to scikit-learn API to work with LIME
(https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn)
- train on 2 and test on 1

Modified by Kevin Sherman. Added:
- train on one set and test on another
"""

import os
import sys
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
    parser.add_argument("--spark-features-only", action="store_true", help="Only use shared Spark features")
    parser.add_argument("--mode", type=str, default="normal",
                        choices=["normal", "trainAllTestOne", "trainOneTestAll", "leaveOneOut", "trainAllTestAll"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Customize logging
    logger = logging.getLogger()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    if args.log_file:
        filehandler = logging.FileHandler(args.log_file, "w+")
        logger.addHandler(filehandler)
    
    # Set seed
    random.seed(args.seed)

    names, train_paths, valid_paths, test_paths = [], [], [], []
    if len(args.tests) == 1 and args.mode != "normal":
        logging.error(f"Mode must be 'normal' for only one test")
        sys.exit(1)
    if args.mode == "normal":
        for test_name in args.tests:
            names.append(test_name)
            train_paths.append([f"{args.data_dir}/{test_name}/training/"])
            valid_paths.append([f"{args.data_dir}/{test_name}/validation/"])
            test_paths.append([f"{args.data_dir}/{test_name}/test/"])
    elif args.mode == "leaveOneOut":
        for test_name in args.tests:
            names.append(f"trainAllTest_{test_name}")
            train_paths.append([f"{args.data_dir}/{t}/training/" for t in args.tests if t != test_name])
            valid_paths.append([f"{args.data_dir}/{test_name}/validation/"])
            test_paths.append([f"{args.data_dir}/{test_name}/test/"])
    elif args.mode == "trainOneTestAll":
        for test_name in args.tests:
            names.append(f"train_{test_name}_testAll")
            train_paths.append([f"{args.data_dir}/{test_name}/training/"])
            valid_paths.append([f"{args.data_dir}/{t}/validation/" for t in args.tests])
            test_paths.append([f"{args.data_dir}/{t}/test/" for t in args.tests])
    elif args.mode == "trainAllTestAll":
        names.append("all_" + "_".join(args.tests))
        train_paths.append([f"{args.data_dir}/{t}/training/" for t in args.tests])
        valid_paths.append([f"{args.data_dir}/{t}/validation/" for t in args.tests])
        test_paths.append([f"{args.data_dir}/{t}/test/" for t in args.tests])
    else:
        logging.error(f"Invalid mode")
        sys.exit(1)

    # Train model for each dataset
    results_dict = {}
    TARGET_COLUMN = 'flow_size'
    for TEST_NAME, TRAINING_PATH, VALIDATION_PATH, TEST_PATH  in zip(names, train_paths, valid_paths, test_paths):
        logging.info(f"On test {TEST_NAME}")
        results_dict[TEST_NAME] = {}
        MODEL_PATH = f"{args.output_dir}/xgboost_{TEST_NAME}"

        training_files = [os.path.join(t, f) for t in TRAINING_PATH for f in os.listdir(t)]
        validation_files = [os.path.join(t, f) for t in VALIDATION_PATH for f in os.listdir(t)]
        test_files = [os.path.join(t, f) for t in TEST_PATH for f in os.listdir(t)]
        logging.debug(f"{training_files}")

        scaling = xgboost_util.calculate_scaling(training_files)
        data = xgboost_util.prepare_files(training_files, args.look_back, scaling,
                                          TARGET_COLUMN, use_shared_features=args.spark_features_only)

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
        clf.save_model(MODEL_PATH)
        
        def evaluate_model(files, write_to_simulator=False):
            real = []
            predicted = []
            for f in files:
                data = xgboost_util.prepare_files([f], args.look_back, scaling, TARGET_COLUMN, use_shared_features=args.spark_features_only)
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


