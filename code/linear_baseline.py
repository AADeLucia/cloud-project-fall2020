"""
Linear regression model which uses previous 4 flows as input.

Author: Alexandra DeLucia
"""
import pickle
import time
import os
from argparse import ArgumentParser
import logging

import numpy
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

# Configure default logging
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = ArgumentParser()
    # Experiment settings
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tests", nargs="+", default=["KMeans", "PageRank", "SGD", "web_server", "tensorflow"])
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--log-file", type=str)

    # Model settings
    parser.add_argument("--scale-data", action="store_true", help="Scale the dataset")
    parser.add_argument("--normalize-data", action="store_true", help="Normalize the dataset")
    parser.add_argument("--look-back", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back-1)].copy()
        dataX.append(a.flatten())
        dataY.append(dataset[i + look_back - 1])
        logging.debug(f"x={a}, y={dataset[i + look_back - 1]}")
    return numpy.array(dataX), numpy.array(dataY)


def load_dataset(path, ):
    """
    Load dataset and only keep flow size
    """
    dfs = []
    for f in os.listdir(path):
        df = pd.read_csv(path + f, engine='python', skipfooter=1)
        df = df.drop(columns=['index'])
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    logging.debug(f"Loaded data from {path}\n\tFeature columns: {', '.join(df.columns)}\n\tSamples: {len(df):,}")
    dataset = df.flow_size.values
    dataset = dataset.astype('float32').reshape(-1, 1)
    return dataset


def main():
    # Parse commandline options
    args = parse_args()
    results_dict = {}
    
    # Customize logging
    logger = logging.getLogger()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    if args.log_file:
        filehandler = logging.FileHandler(args.log_file, "w+")
        logger.addHandler(filehandler)

    # Set seed for reproducibility
    numpy.random.seed(args.seed)

    # Set base save file
    base_save = f"linear_{'scaled_' if args.scale_data else ''}{args.look_back}"

    # Process each test case
    for TEST_NAME in args.tests:
        logging.info(f"Started running experiment {TEST_NAME}")

        TRAIN_PATH = f"{args.data_dir}/{TEST_NAME}/training/"
        TEST_PATH = f"{args.data_dir}/{TEST_NAME}/test/"
        VALIDATION_PATH = f"{args.data_dir}/{TEST_NAME}/validation/"
        MODEL_PATH = f"{args.output_dir}/{base_save}_{TEST_NAME}.pkl"

        # Load the data
        train = load_dataset(TRAIN_PATH)
        test = load_dataset(TEST_PATH)
        validation = load_dataset(VALIDATION_PATH)

        # Scale dataset
        if args.scale_data:
            scaler = MinMaxScaler(feature_range=(0, 1))
            train = scaler.fit_transform(train)
            test = scaler.transform(test)
            validation = scaler.transform(validation)

        trainX, trainY = create_dataset(train, args.look_back)
        testX, testY = create_dataset(test, args.look_back)
        validationX, validationY = create_dataset(validation, args.look_back)
        logging.debug(f"train={trainX}\n{trainY}")

        # Check if model needs to be trained
        if not args.predict_only:
            # Build and train model
            logging.info("Building model")
            model = LinearRegression(normalize=args.normalize_data)

            start = time.time()
            model.fit(trainX, trainY)
            end = time.time()
            logging.debug(f"Model took {end-start/60:.2} minutes to train")

            # Save model
            logging.info("Saving model")
            with open(MODEL_PATH, "wb") as f:
                pickle.dump(model, f)
        else:
            logging.info(f"Loading pre-trained model from {MODEL_PATH}")
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)

        # Make predictions
        logging.debug(f"TrainX: {trainX}")
        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)
        validationPredict = model.predict(validationX)
        
        trainScore = r2_score(trainY, trainPredict)
        logging.info('Train Score: %.2f R2' % (trainScore))
        testScore = r2_score(testY, testPredict)
        logging.info('Test Score: %.2f R2' % (testScore))
        validationScore = r2_score(validationY, validationPredict)
        logging.info('Validation Score: %.2f R2' % (validationScore))
        
        results_dict[TEST_NAME] = {
            "train": trainScore,
            "test": testScore,
            "validation": validationScore
        }
        logging.info("---------------------------------------------------------------------\n")

    # Save all scores
    logging.info("Saving all results")
    with open(f"{args.output_dir}/{base_save}_results.pkl", "wb") as f:
        pickle.dump(results_dict, f)


if __name__ == "__main__":
    main()

