import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sklearn.metrics as metrics
import sklearn.preprocessing as proc
import pandas as pd

import math
import time
import logging

import utils
from model import StockPredictor, StockPredictorAttention


# config for logger
logging.basicConfig(filename='run.log', filemode='a', format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)


def train(model, input_seq, y_target):
    """
    trains the model and returns the loss
    :param model: the model to be trained
    :param input_seq: sequence containing data for the last n days, used as input to the model
    :param y_target: target values, used for computing the error
    :return: returns the prediction error / loss
    """

    # set model to train mode
    model.train()

    # predict 'close' value of next day based on sequence of last days
    y_pred = model(input_seq)

    # compute error
    loss = criterion(y_pred, y_target)

    # zero gradients
    optimizer.zero_grad()

    # backpropagation
    loss.backward()

    # update model
    optimizer.step()

    # return training loss of current epoch
    return loss.item()


def evaluate(model, input_seq, y_target):
    """
    evaluates the model to measure its accuracy
    :param model: the model to be evaluated
    :param input_seq: sequence containing data for the last n days, used as input to the model
    :param y_target: target values, used for computing the error
    :return: returns the root mean squared error (RMSE)
    """

    # set model to eval mode
    model.eval()

    with torch.no_grad():
        y_pred = model(input_seq)

    mse = metrics.mean_squared_error(y_target, y_pred)
    return math.sqrt(mse)


if __name__ == '__main__':
    # init logger
    logger = logging.getLogger(__name__)

    # define hyperparameters
    input_size = 1
    hidden_size = 64
    num_layers = 1
    attention = False
    alignment = 'concat' if attention else None
    learning_rate = 0.001
    split_ratio = (0.8, 0.2)
    seq_len = 20
    epochs = 250
    hyperparameters = {'input_size': input_size, 'hidden_size': hidden_size, 'num_layers': num_layers,
                       'attention': attention, 'alignment': alignment, 'learning_rate': learning_rate,
                       'split_ratio': split_ratio, 'seq_len': seq_len, 'epochs': epochs}

    # check if gpu is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # log hyperparameters and device
    logger.info(f'hyperparameters: {hyperparameters}')
    logger.info(f'device: {device}')

    # init model
    if attention:
        stock_predictor = StockPredictorAttention(input_size, hidden_size, num_layers, device, alignment)
    else:
        stock_predictor = StockPredictor(input_size, hidden_size, num_layers, device)

    # init optimizer and loss criterion
    optimizer = optim.Adam(stock_predictor.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # load dataset from file
    dataframe = utils.load_data('djia30.csv', name='IBM')

    # init scaler for normalization
    scaler = proc.MinMaxScaler(feature_range=(0, 1))

    # normalize dataset and generate sequences
    scaler, *data = utils.transform_data(dataframe['Close'].values.reshape(-1, 1), scaler, split_ratio, seq_len)
    x_train, y_train, x_validate, y_validate, x_complete, y_complete = data

    # init history
    losses = np.zeros(epochs)
    accuracy = np.zeros(epochs)

    # start loop
    start = time.time()
    for e in range(epochs):
        epoch_loss = train(stock_predictor, x_train, y_train)
        validation_error = evaluate(stock_predictor, x_validate, y_validate)

        losses[e] = epoch_loss
        accuracy[e] = validation_error

        if e % 10 == 0 or e == epochs - 1:
            logger.info(f'epoch {e}: training loss {epoch_loss:.6f}, RMSE of validation data: {validation_error:.6f}')

    logger.info(f'training complete! overall time: {time.time() - start:.2f}s')
    logger.info(f'min training loss: {min(losses):.6f}')
    logger.info(f'min RMSE of validation data: {min(accuracy):.6f}')

    # set model to eval mode and predict complete sequence for plotting
    stock_predictor.eval()
    with torch.no_grad():
        prediction = stock_predictor(x_complete)

    # inverse transform normalized data for plotting
    prediction = scaler.inverse_transform(prediction)
    ground_truth = scaler.inverse_transform(y_complete)

    # init new dataframe for plotting
    start = dataframe['Date'].values[seq_len]
    periods = len(y_complete)
    df = pd.DataFrame({'date': pd.date_range(start=start, periods=periods),
                       'ground_truth': ground_truth[:, 0],
                       'prediction': prediction[:, 0]})

    # make plots and save
    utils.plot_result(df, losses, accuracy, hyperparameters)
