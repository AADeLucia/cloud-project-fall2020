"""
Edited by Alexandra DeLucia for logging and model save.
"""
# LSTM for international airline passengers problem with regression framing
import pickle
import time
import os
from argparse import ArgumentParser
import logging

import numpy
import pandas as pd
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

# Configure default logging
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tests", nargs="+", default=["KMeans", "PageRank", "SGD", "tensorflow", "web_server"])
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--log-file", type=str)
    parser.add_argument("--look-back", type=int, default=5)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--predict-only", action="store_true")
    return parser.parse_args()


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), :].copy()
        a[-1,-1] = 0
        dataX.append(a.flatten())
        dataY.append(dataset[i + look_back - 1, -1])
    return numpy.array(dataX), numpy.array(dataY)


def load_dataset(path, ):
    dfs = []
    for f in os.listdir(path):
        df = pd.read_csv(path + f, engine='python', skipfooter=1)
        df = df.drop(columns=['index'])
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded data from {path}\n\tFeature columns: {', '.join(df.columns)}\n\tSamples: {len(df):,}")
    dataset = df.values
    dataset = dataset.astype('float32')
    dataset = dataset[:-1,:]
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

    # Process each test case
    for TEST_NAME in args.tests:
        logging.info(f"Started running experiment {TEST_NAME}")

        TRAIN_PATH = f"{args.data_dir}/{TEST_NAME}/training/"
        TEST_PATH = f"{args.data_dir}/{TEST_NAME}/test/"
        VALIDATION_PATH = f"{args.data_dir}/{TEST_NAME}/validation/"
        MODEL_PATH = f"{args.output_dir}/ffnn_{TEST_NAME}"
        
        # Load and normalize the dataset
        scaler = MinMaxScaler(feature_range=(0, 1))

        train = load_dataset(TRAIN_PATH)
        train = scaler.fit_transform(train)

        test = load_dataset(TEST_PATH)
        test = scaler.transform(test)

        validation = load_dataset(VALIDATION_PATH)
        validation = scaler.transform(validation)

        trainX, trainY = create_dataset(train, args.look_back)
        testX, testY = create_dataset(test, args.look_back)
        validationX, validationY = create_dataset(validation, args.look_back)
 
        # Check if model needs to be trained
        if not args.predict_only:
            # Build and train model
            logging.info("Building model")
            model = Sequential()
            model.add(Dense(5, input_dim=trainX.shape[1], activation='relu'))
            model.add(Dense(5, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='mean_absolute_error', optimizer='adam')

            start = time.time()
            model.fit(trainX, trainY, epochs=250, batch_size=10, verbose=2)
            end = time.time()
            logging.debug(f"Model took {end-start/60:.2} minutes to train")

            # Save model
            logging.info("Saving model")
            model.save(MODEL_PATH)

        else:
            logging.info("Loading pre-trained model")
            model = load_model(MODEL_PATH)
        
        # Make predictions
        logging.debug(f"TrainX: {trainX}")
        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)
        validationPredict = model.predict(validationX)
        
        trainScore = r2_score(trainY.flatten(), trainPredict.flatten())
        logging.info('Train Score: %.2f R2' % (trainScore))
        testScore = r2_score(testY.flatten(), testPredict.flatten())
        logging.info('Test Score: %.2f R2' % (testScore))
        validationScore = r2_score(validationY.flatten(), validationPredict.flatten())
        logging.info('Validation Score: %.2f R2' % (validationScore))
        
        results_dict[TEST_NAME] = {
            "train": trainScore,
            "test": testScore,
            "validation": validationScore
        }
        logging.info("---------------------------------------------------------------------\n")

    # Save all scores
    logging.info("Saving all results")
    with open(f"{args.output_dir}/ffnn_results.pkl", "wb") as f:
        pickle.dump(results_dict, f)


if __name__ == "__main__":
    main()


