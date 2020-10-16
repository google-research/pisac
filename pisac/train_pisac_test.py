# coding=utf-8
# Copyright 2020 The PI-SAC Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for pisac.sac_agent."""

import os

from absl import flags
from pisac import train_pisac
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.utils import test_utils

FLAGS = flags.FLAGS


class PisacTrainEvalTest(test_utils.TestCase):

  def test_simple_iteration(self):
    # Check that the method runs and checkpoints are created.
    root_dir = os.path.join(FLAGS.test_tmpdir, 'pisac')
    train_pisac.train_eval(root_dir=root_dir,
                           frame_shape=(16, 16, 3),
                           num_env_steps=1,
                           eval_env_interval=100,
                           initial_feature_step=10,
                           initial_collect_steps=10,
                           replay_buffer_capacity=20,
                           batch_size=1,
                           checkpoint_env_interval=100000,
                           log_env_interval=1,
                           summary_interval=1,
                           image_summary_interval=0,
                           conv_feature_dim=2,
                           ceb_feature_dim=2,
                           num_eval_episodes=1)


# Main function so that users of `test_utils.TestCase` can also call
# `test_utils.main()`.
if __name__ == '__main__':
  tf.test.main()
