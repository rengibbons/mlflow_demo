"""
Author: Ren Gibbons
Organization: Edge Analytics

Script must be run from /MLflowDemo/src
"""
import os
import sys
import numpy as np
import yaml
import pickle
from pprint import pprint
import datetime as dt
import pytz
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python.saved_model import tag_constants

import mlflow
import mlflow.tensorflow

EXPERIMENT_NAME = "my-experiment"
DATA_PATH = os.path.join(os.path.dirname(os.getcwd()), 'data')

mlflow.set_tracking_uri('file://' + os.path.join(DATA_PATH, 'mlruns'))
mlflow.set_experiment(EXPERIMENT_NAME)

# Enable auto-logging to MLflow to capture TensorBoard metrics.
mlflow.tensorflow.autolog(every_n_iter=1)


def get_time(timezone='US/Pacific'):
    """
    Returns the time as a string.
    """
    return dt.datetime.now(pytz.timezone(timezone)).strftime('%Y%m%d-%H%M%S')


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


def get_tensorboard_callbacks(run_name):
    """
    Gets the tensorboard callbacks.
    """
    return tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(get_model_path_local(run_name), 'tensorboard'), update_freq='epoch'
    )


def get_model_path_local(run_name):
    """
    .
    """
    experiment_id = mlflow.get_experiment_by_name(name=EXPERIMENT_NAME).experiment_id
    return os.path.join(DATA_PATH, 'edge_models', experiment_id, run_name)


def main():
    """
    Trains a model on the MNIST dataset.
    """
    print('{} Starting mlflow_demo.py.'.format(get_time()))

    # Load configuration file and default column ordering for df_models.
    config_root = os.path.join(os.path.dirname(os.getcwd()), 'config')

    # Check if the correct command line arguments are provied.
    if len(sys.argv) not in [2, 3]:
        raise Exception('Usage: $ python3 mflow_demo.py <CONFIG_FILE.yml> optional: <RUN_NAME>')

    config_file = sys.argv[1]
    run_name = get_time()

    with mlflow.start_run(run_name=run_name):
        with open(os.path.join(config_root, config_file)) as fin:
            cfg = yaml.load(fin, Loader=yaml.FullLoader)

        print('{} Configuration parameters for run_name = {} run_id = {}.'.format(
            get_time(), run_name, mlflow.active_run().info.run_id)
        )
        pprint(cfg)
        mlflow.log_params(cfg)

        x_train, y_train, x_test, y_test = get_dataset(cfg['train_size'], cfg['test_size'])

        model = create_model(cfg)
        model.fit(
            x=x_train,
            y=y_train,
            validation_split=cfg['validation_split'],
            callbacks=[get_tensorboard_callbacks(run_name)],
            epochs=cfg['epochs'])

        # Export SavedModel
        model_local_path = get_model_path_local(run_name)
        if not os.path.exists(model_local_path):
            os.makedirs(model_local_path)

        model.save(os.path.join(model_local_path, 'final_model'))

        mlflow.tensorflow.log_model(
            tf_saved_model_dir=os.path.join(model_local_path, 'final_model'),
            tf_meta_graph_tags=[tag_constants.SERVING],
            tf_signature_def_key='serving_default',
            artifact_path='model'
        )

        # Log a pickle file.
        pickle.dump(x_train, open(os.path.join(model_local_path, 'x_train.p'), 'wb'))
        mlflow.log_artifact(os.path.join(model_local_path, 'x_train.p'))

        # Log an image.
        fig, ax = plt.subplots()
        ax.scatter(np.random.normal(0.0, 1.0, 1000), np.random.normal(0.0, 1.0, 1000), alpha=0.3)
        ax.set_title('Random plot')
        fig.savefig(os.path.join(model_local_path, 'random_figure.png'), dpi=300, bbox_inches='tight')
        mlflow.log_artifact(os.path.join(model_local_path, 'random_figure.png'))


if __name__ == '__main__':
    """
    Calls main.
    """
    main()
