import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import seaborn as sns
import logging


logger = logging.getLogger(__name__)


def gen_seq(data, seq_len):
    """
    generates a pair of sequences of features and its corresponding 'close' value on the following day
    :param data: numpy array containing the data
    :param seq_len: integer value defining the length of each generated sequence
    :return: pytorch tensors containing sequence data and corresponding next day data
    """
    pairs = [(data[i:seq_len + i, :], data[seq_len + i, -1]) for i in range(len(data) - seq_len)]
    return map(lambda t: torch.tensor(t, dtype=torch.float32), zip(*pairs))


def load_data(path, name):
    """
    loads the dataset from disk based on passed variables
    :param path: string that represents directory information
    :param name: string that holds market share name, used for filtering
    :return: sorted and filtered pandas dataframe
    """

    # read dataset
    df = pd.read_csv(path)

    # sort values by 'date'
    df.sort_values('Date')

    # slice dataframe -> take rows by 'name'
    df = df.loc[df['Name'] == name]

    # drop columns 'open', 'high', 'low', 'name', 'volume' -> take only 'close' as feature
    df = df.drop(columns=['Open', 'High', 'Low', 'Name', 'Volume'])

    logger.info('dataset loaded')
    return df


def transform_data(dataset, scaler, split_ratio=(0.8, 0.2), seq_len=20):
    """
    applies several transformations on the passed data and returns it
    :param dataset: pandas dataframe that contains the data
    :param scaler: min-max scaler for normalization purposes
    :param split_ratio: tuple of ints defining the split-ratio of the dataset
    :param seq_len: integer value defining the length of each generated sequence
    :return: transformed data and multiple data chunks as pytorch tensors
    """

    # calc sizes of dataset chunks
    size = len(dataset)
    train_len = int(split_ratio[0] * size)
    validate_len = int(split_ratio[1] * size)

    # split dataset in train, validation and test data
    train = dataset[:train_len]
    validate = dataset[train_len:train_len + validate_len]

    # fit scaler to data and normalize chunks
    scaler.fit(train)
    train = scaler.transform(train)
    validate = scaler.transform(validate)
    complete = scaler.transform(dataset)

    # generate sequences for model training and evaluation
    x_train, y_train = gen_seq(train, seq_len)
    x_validate, y_validate = gen_seq(validate, seq_len)
    x_complete, y_complete = gen_seq(complete, seq_len)

    # unsqueeze to match dimensions
    y_train.unsqueeze_(dim=1)
    y_validate.unsqueeze_(dim=1)
    y_complete.unsqueeze_(dim=1)

    # log infos concerning dataset
    logger.info(f'data split ratio: {split_ratio}')
    logger.info(f'dataset length: {size}')
    logger.info(f'length of training chunk: {len(train)}')
    logger.info(f'length of validation chunk: {len(validate)}')

    return scaler, x_train, y_train, x_validate, y_validate, x_complete, y_complete


def plot_result(df, losses, accuracy, hyperparameters):
    """
    generates and shows or saves the collected data
    :param df: pandas dataframe containing the data
    :param losses: numpy array containing the training loss for each training epoch
    :param accuracy: numpy array containing the models accuracy values for each evaluation epoch
    :param hyperparameters: dictionary containing the used hyperparameters
    :return: None
    """

    # set style to seaborn and show ticks
    sns.set(rc={'xtick.bottom': True, 'ytick.left': True})
    fig = plt.figure(figsize=(15, 10))

    # add subplot and plot actual and predicted stock price
    sub_1 = fig.add_subplot(2, 1, 1)
    sub_1.plot(df[['date']], df[['ground_truth']], label='ground truth')
    sub_1.plot(df[['date']], df[['prediction']], label='prediction')

    # set major ticks every year
    sub_1.xaxis.set_major_locator(dates.YearLocator())
    sub_1.xaxis.set_major_formatter(dates.DateFormatter('\n%Y'))

    # set minor ticks every 3 months
    sub_1.xaxis.set_minor_locator(dates.MonthLocator(bymonth=[1, 4, 7, 10]))
    sub_1.xaxis.set_minor_formatter(dates.DateFormatter('%b'))

    # set title and axis labels
    sub_1.set_title('stock price', fontsize=16)
    sub_1.set_xlabel('date', fontsize=14)
    sub_1.set_ylabel('close', fontsize=14)
    sub_1.legend()

    # add subplot, plot loss and set labels
    sub_2 = fig.add_subplot(2, 2, 3)
    sub_2.set_xlabel('epoch', fontsize=14)
    sub_2.set_ylabel('loss', fontsize=14)
    sub_2.plot(losses)

    # add subplot, plot accuracy and set labels
    sub_3 = fig.add_subplot(2, 2, 4)
    sub_3.set_xlabel('epoch', fontsize=14)
    sub_3.set_ylabel('validation error', fontsize=14)
    sub_3.plot(accuracy)

    # add text
    plt.suptitle(f'hyperparameters = {hyperparameters}', fontsize=10)

    # adjust spacing
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.35)

    # show plot
    # plt.show()

    # save plot and close
    plt.savefig('result.png')
    plt.close()
