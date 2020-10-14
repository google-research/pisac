r"""Run PI-SAC training and evaluation."""
import os

from absl import app
from absl import flags
from absl import logging

import gin
import gin.tf
from pisac import train_pisac
import tensorflow.compat.v2 as tf


flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_multi_string('gin_file', None,
                          'Path to the trainer config files.')
flags.DEFINE_multi_string('gin_bindings', None, 'Gin binding to pass through.')

FLAGS = flags.FLAGS


def main(_):
  tf.compat.v1.enable_v2_behavior()
  logging.set_verbosity(logging.INFO)
  gin.parse_config_files_and_bindings(
      FLAGS.gin_file, FLAGS.gin_bindings, skip_unknown=False)
  root_dir = FLAGS.root_dir
  train_pisac.train_eval(root_dir)


if __name__ == '__main__':
  flags.mark_flag_as_required('root_dir')
  app.run(main)
