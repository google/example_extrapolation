# coding=utf-8
# Copyright 2021 Google LLC
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
"""Postprocess predictions from the Ex2 teacher."""
import random
from absl import app
from absl import flags
import tensorflow as tf
import tf_utils

FLAGS = flags.FLAGS

flags.DEFINE_string("teacher_intents_path", None,
                    "Path to teacher_intents.txt")
flags.DEFINE_string("teacher_predictions_path", None,
                    "Path to teacher_predictions.txt")
flags.DEFINE_string("original_train_path", None,
                    "Path to student_train.tfr")
flags.DEFINE_string("output_train_path", None,
                    "Path to student_train_with_ex2.tfr")


def main(_):
  examples = list(
      tf.data.TFRecordDataset(FLAGS.original_train_path).as_numpy_iterator())

  print(f"Found {len(examples)} original examples.")

  with tf.io.gfile.GFile(FLAGS.teacher_intents_path) as intents_file:
    with tf.io.gfile.GFile(FLAGS.teacher_predictions_path) as predictions_file:
      for intent, prediction in zip(intents_file, predictions_file):
        example = tf_utils.create_tf_example(prediction.strip(), intent.strip())
        examples.append(example.SerializeToString())

  print(f"Outputting {len(examples)} total examples.")

  random.shuffle(examples)
  with tf.io.TFRecordWriter(FLAGS.output_train_path) as writer:
    for e in examples:
      writer.write(e)

if __name__ == "__main__":
  app.run(main)
