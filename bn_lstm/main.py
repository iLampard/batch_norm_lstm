""" A scipt to run the model """

import os
import sys

CUR_PATH = os.path.abspath(os.path.dirname(__file__))
ROOT_PATH = os.path.split(CUR_PATH)[0]
sys.path.append(ROOT_PATH)

from absl import app
from absl import flags

FLAGS = flags.FLAGS

# model runner_params
flags.DEFINE_bool('write_summary', False, 'Whether to write summary of epoch in training using Tensorboard')
flags.DEFINE_integer('max_epoch', 10, 'Max epoch number of training')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate')

# model params
flags.DEFINE_bool('batch_norm', False, 'Whether to apply batch normalization')
flags.DEFINE_integer('rnn_dim', 32, 'Dimension of LSTM cell')
flags.DEFINE_float('dropout', 0.0, 'Dropout rate of the model')
flags.DEFINE_string('save_dir', 'logs', 'Root path to save logs and models')

def main(argv):
    return


if __name__ == '__main__':
    app.run(main)
