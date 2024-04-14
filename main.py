# coding=utf-8
# Copyright 2020 The Google Research Authors.
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
#python main_fastmri.py --config configs/ve/MAR_416_ncsnpp_continuous.py --eval_folder eval/MAR_416 --mode 'train' --workdir=workdir/MAR_416
"""Training and evaluation"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
#import run_lib_fastmri
import run_lib_fastmri
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import logging

import tensorflow as tf

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_enum("mode", None, ["train", "train_regression", "eval"], "Running mode: train, train_regression, or eval")
flags.DEFINE_string("eval_folder", "eval",
                    "The folder name for storing evaluation results")
flags.mark_flags_as_required(["workdir", "config", "mode"])


def main(argv):
  print(FLAGS.config)
  if FLAGS.mode == "train" or FLAGS.mode == "train_regression":
    # Create the working directory
    tf.io.gfile.makedirs(FLAGS.workdir)
    # Set logger so that it outputs to both console and file
    # Make logging work for both disk and Google Cloud Storage
    gfile_stream = open(os.path.join(FLAGS.workdir, 'stdout.txt'), 'w')
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')
    # Run the training pipeline
    print('start')
    if FLAGS.mode == "train": 
        #run_lib_fastmri_CT.train(FLAGS.config, FLAGS.workdir)
        run_lib_fastmri.train(FLAGS.config, FLAGS.workdir)
    elif FLAGS.mode == "train_regression":
      run_lib_fastmri.train_regression(FLAGS.config, FLAGS.workdir)
  elif FLAGS.mode == "eval":
    # Run the evaluation pipeline
    run_lib_fastmri.evaluate(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder)
  else:
    raise ValueError(f"Mode {FLAGS.mode} not recognized.")


if __name__ == "__main__":
  app.run(main)
