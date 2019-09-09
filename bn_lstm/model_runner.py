""" Model Runner that controls the pipeline of running the model   """

from absl import logging
import enum
from datetime import datetime as dt
import functools
import numpy as np
import logging as logging_base
import operator
import os
import sys
import tensorflow as tf


class StrEnum(str, enum.Enum):
    """ Define a string enum class """
    pass


class RunnerPhase(StrEnum):
    """ Model Runner Phase Enum """
    TRAIN = 'train'
    VALIDATE = 'validate'
    PREDICT = 'predict'


class ModelRunner:
    def __init__(self,
                 model,
                 flags):
        self.model_wrapper = ModelWrapper(model)
        self.model_checkpoint_path = flags.save_dir
        self.lr = flags.learning_rate
        self.write_summary = flags.write_summary
        self.rnn_dim = flags.rnn_dim
        self.dropout = flags.dropour_rate
        self.batch_size = flags.batch_size

        logging.get_absl_logger().addHandler(logging_base.StreamHandler(sys.path))

        return

    @staticmethod
    def num_params():
        """ Return total num of params of the model  """
        total_num = 0
        for var in tf.trainable_variables():
            shape = var.get_shape()
            total_num += functools.reduce(operator.mul, [dim.value for dim in shape], 1)
        return total_num

    def restore(self, restore_path):
        """ Restore model  """
        self.model_wrapper.restore(restore_path)

    def train(self,
              train_dataset,
              valid_dataset,
              test_dataset,
              max_epochs,
              valid_gap_epochs=1,
              auto_save=True):
        """ Run the training """
        folder_path, model_path = self.make_dir()
        logging.get_absl_handler().use_absl_log_file('log', folder_path)
        logging.info('Start training with max epochs {}'.format(max_epochs))
        logging.info('Model and logs saved in {}'.format(folder_path))
        logging.info('Number of trainable parameters - {}'.format(self.num_params()))
        if self.write_summary:
            self.model_wrapper.init_summary_writer(folder_path)

        best_eval_loss = float('inf')

        for i in range(1, max_epochs + 1):
            loss, _, _ = self.run_one_epoch(train_dataset, RunnerPhase.TRAIN, self.lr)

            # Record the train loss
            loss_metrics = {'loss': np.mean(loss)}

            logging.info('Epoch {0} -- Training loss {1}'.format(i, loss_metrics['loss']))
            self.model_wrapper.write_summary(i, loss_metrics, RunnerPhase.TRAIN)

            # Evaluate the model
            if i % valid_gap_epochs == 0:
                loss, prediction, _ = self.run_one_epoch(valid_dataset, RunnerPhase.VALIDATE, self.lr)
                # Record the validation loss
                loss_metrics = {'loss': np.mean(loss)}
                logging.info('Epoch {0} -- Validation loss {1}'.format(i, loss_metrics['loss']))
                self.model_wrapper.write_summary(i, loss_metrics, RunnerPhase.VALIDATE)

                # If it is the best model on valid set
                if best_eval_loss > loss_metrics['loss']:
                    metrics = self.evaluate(test_dataset)
                    self.model_wrapper.write_summary(i, metrics, RunnerPhase.PREDICT)

                    best_eval_loss = loss_metrics['loss']
                    if auto_save:
                        self.model_wrapper.save(model_path)

        if not auto_save:
            self.model_wrapper.save(model_path)
        else:
            # use the saved model based on valid performance
            self.restore(model_path)

        logging.info('Training finished')

    def evaluate(self, dataset):
        """ Evaluate the model on valid / test set """
        logging.info('Start evaluation')

        loss, predictions, labels = self.run_one_epoch(dataset, RunnerPhase.PREDICT, self.lr)

        accuracy = np.mean(predictions == labels)

        logging.info('Accuracy {}'.format(accuracy))

        logging.info('Evaluation finished')
        return

    def run_one_epoch(self, dataset, phase, lr):
        """ Run one complete epoch """
        epoch_loss = []
        epoch_predictions = []
        epoch_labels = []
        for input_x, label in dataset.next_batch(self.batch_size):
            loss, prediction = self.model_wrapper.run_batch(input_x,
                                                            lr,
                                                            phase=phase)
            epoch_loss.extend(loss)
            epoch_predictions.extend(prediction)
            epoch_labels.extend(label)

        return epoch_loss, epoch_predictions, epoch_labels

    def make_dir(self):
        """ Initialize the directory to save models and logs """
        folder_name = list()
        model_tags = {'lr': self.lr,
                      'dim': self.rnn_dim}

        for key, value in model_tags.items():
            folder_name.append('{}-{}'.format(key, value))
        folder_name = '_'.join(folder_name)
        current_time = dt.now().strftime('%Y%m%d-%H%M%S')
        folder_path = os.path.join(self.model_checkpoint_path,
                                   self.model_wrapper.__class__.__name__,
                                   folder_name,
                                   current_time)
        os.mkdir(folder_path)
        model_path = os.path.join(folder_path, 'saved_model')
        return folder_path, model_path


class ModelWrapper:

    def __init__(self, model):
        """ Init model wrapper """
        self.model = model
        sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        self.sess = tf.Session(config=sess_config)
        self.model.build()

        # initialize saver and tensorboard summary writer
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        self.train_summary_writer = None
        self.valid_summary_writer = None
        self.test_summary_writer = None

    def save(self, checkpoint_path):
        """ Save the model """
        self.saver.save(self.sess, checkpoint_path)

    def restore(self, checkpoint_path):
        """ Restore the model """
        self.saver.restore(self.sess, checkpoint_path)

    def init_summary_writer(self, root_dir):
        """ Init tensorboard writer  """
        tf_board_dir = 'tfb_dir'
        folder = os.path.join(root_dir, tf_board_dir)
        self.train_summary_writer = tf.summary.FileWriter(os.path.join(folder, 'train'), self.sess.graph)
        self.valid_summary_writer = tf.summary.FileWriter(os.path.join(folder, 'valid'))
        self.test_summary_writer = tf.summary.FileWriter(os.path.join(folder, 'test'))

    def write_summary(self, epoch_num, kv_pairs, phase):
        """ Write summary into tensorboard """
        if phase == RunnerPhase.TRAIN:
            summary_writer = self.train_summary_writer
        elif phase == RunnerPhase.VALIDATE:
            summary_writer = self.valid_summary_writer
        elif phase == RunnerPhase.PREDICT:
            summary_writer = self.test_summary_writer
        else:
            raise RuntimeError('Unknow phase: ' + phase)

        if summary_writer is None:
            return

        for key, value in kv_pairs.item():
            metrics = tf.Summary()
            metrics.value.add(tag=key, simple_value=value)
            summary_writer.add_summary(metrics, epoch_num)

        summary_writer.flush()

    def run_batch(self, x, lr, phase):
        """ Run one batch """
        if phase == RunnerPhase.TRAIN:
            loss, prediction = self.model.train(self.sess, x, lr=lr)
        else:
            loss, prediction = self.model.predict(self.sess, x)

        return loss, prediction
