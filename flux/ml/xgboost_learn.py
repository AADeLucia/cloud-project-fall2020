#!/usr/bin/env python

import xgboost
import os
import xgboost_util
import math

from argparse import ArgumentParser
import pandas as pd
import numpy as np
import random
import logging

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
        filehandler = logging.FileHandler(args.log_file, 'a')
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

        training_files = [os.path.join(TRAINING_PATH, f) for f in os.listdir(TRAINING_PATH)]
        test_files = [os.path.join(TEST_PATH, f) for f in os.listdir(TEST_PATH)]
        validation_files = [os.path.join(VALIDATION_PATH, f) for f in os.listdir(VALIDATION_PATH)]

        scaling = xgboost_util.calculate_scaling(training_files)
        data = xgboost_util.prepare_files(training_files, args.look_back, scaling, TARGET_COLUMN)

        inputs, outputs = xgboost_util.make_io(data)

        # fit model no training data
        param = {
            'num_epochs' : 50,
            'max_depth' : 10,
            'objective' : 'reg:linear',
            'booster' : 'gbtree',
            'base_score' : 2,
            'silent': 1,
            'eval_metric': 'mae'
        }

        training = xgboost.DMatrix(inputs, outputs, feature_names = data[0][0].columns)
        logging.info(f"Len outputs: {len(outputs)}")
        logging.info('Training started')
        model = xgboost.train(param, training, param['num_epochs'])

        def evaluate_model(files, write_to_simulator=False):
            real = []
            predicted = []
            for f in files:
                data = xgboost_util.prepare_files([f], args.look_back, scaling, TARGET_COLUMN)
                inputs, outputs = xgboost_util.make_io(data)

                y_pred = model.predict(xgboost.DMatrix(inputs, feature_names = data[0][0].columns))
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


