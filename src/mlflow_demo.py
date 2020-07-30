"""
Author: Ren Gibbons
Organization: Edge Analytics

Script must be run from /MLflowDemo/src
"""
import os
import sys
import numpy as np
import tensorflow as tf
import yaml
import datetime as dt
from pprint import pprint
import pytz

import mlflow

TIMEZONE = 'US/Pacific'

tracking_uri = 'file://' + os.path.join(os.path.dirname(os.getcwd()), 'data', 'mlruns')
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("my-experiment")


def get_time():
    """
    Returns the time as a string.
    """
    return dt.datetime.now(pytz.timezone(TIMEZONE)).strftime('%Y-%m-%d-%H:%M:%S')


def get_dataset(train_size=60000, test_size=10000):
    """
    Loads the MNIST dataset from Keras and returns training and test splits.
    """
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return x_train[:train_size], y_train[:train_size], x_test[:test_size], y_test[:test_size]


def create_model(cfg):
    """
    Creates a feedforward model.
    """
    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=cfg['input_shape'])])
    for fc_layer in cfg['fc_layers']:
        model.add(tf.keras.layers.Dense(fc_layer, activation=cfg['activation']))
        model.add(tf.keras.layers.Dropout(cfg['dropout']))
    model.add(tf.keras.layers.Dense(cfg['output_shape']))

    model.summary()

    model.compile(
        optimizer=cfg['optimizer'],
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=cfg['metrics']
    )       
    return model


def main():
    """
    Trains a model on the MNIST dataset.
    """
    print('{} Starting mlflow_demo.py.'.format(get_time()))
    # Load configuration file and default column ordering for df_models.
    config_root = os.path.join(os.path.expanduser('~'), 'Documents', 'EdgeAnalytics', 'BizDev', 'MLflowDemo', 'config')

    # Check if the correct command line arguments are provied.
    if len(sys.argv) not in [2, 3]:
        raise Exception('Usage: $ python3 mflow_demo.py <CONFIG_FILE.yml> optional: <RUN_NAME>')

    config_file = sys.argv[1]
    run_name = sys.argv[2] if len(sys.argv) == 3 else 'default'

    with mlflow.start_run(run_name=run_name):
        with open(os.path.join(config_root, config_file)) as fin:
            cfg = yaml.load(fin, Loader=yaml.FullLoader)

        print('{} Configuration parameters.'.format(get_time()))
        pprint(cfg)
        mlflow.log_params(cfg)

        x_train, y_train, x_test, y_test = get_dataset(cfg['train_size'], cfg['test_size'])

        model = create_model(cfg)
        model.fit(x_train, y_train, epochs=cfg['epochs'])


if __name__ == '__main__':
    """
    Calls main.
    """
    main()
