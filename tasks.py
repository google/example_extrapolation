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
"""Register tasks."""
import collections
import os

import t5
import tensorflow as tf

CLINC150_DATA_DIR = os.environ["CLINC150_DATA_DIR"]
assert CLINC150_DATA_DIR

TEACHER_FILENAMES = {
    "train": "teacher_train.tfr",
    "validation": "teacher_val.tfr",
}

STUDENT_FILENAMES = {
    "train": "student_train.tfr",
    "overall_validation": "student_overall_val.tfr",
    "fewshot_validation": "student_fewshot_val.tfr"
}

STUDENT_WITH_EX2_FILENAMES = {
    "train": "student_train_with_ex2.tfr",
    "overall_validation": "student_overall_val.tfr",
    "fewshot_validation": "student_fewshot_val.tfr"
}

FEATURE_DESCRIPTION = {
    "inputs": tf.io.FixedLenFeature([], tf.string),
    "targets": tf.io.FixedLenFeature([], tf.string)
}


def macro_f1(targets, predictions):
  """Macro F1."""
  # True positive, target counts, predicted counts.
  counts_by_label = collections.defaultdict(lambda: [0, 0, 0])

  for target_label, predicted_label in zip(targets, predictions):
    counts_by_label[target_label][0] += int(target_label == predicted_label)
    counts_by_label[target_label][1] += 1
    counts_by_label[predicted_label][2] += 1

  def _safe_divide(x, y):
    return x / y if y != 0 else 0.0

  def _f1(tp, tc, pc):
    p = _safe_divide(tp, pc)
    r = _safe_divide(tp, tc)
    return _safe_divide(2 * p * r, p + r)

  def _average(x):
    return _safe_divide(sum(x), len(x))

  return {
      "macro_f1": _average([_f1(*v) for v in counts_by_label.values()]) * 100.0
  }


t5.data.TaskRegistry.add(
    "clinc150_teacher",
    t5.data.TFExampleTask,
    split_to_filepattern={k: os.path.join(CLINC150_DATA_DIR, v)
                          for k, v in TEACHER_FILENAMES.items()},
    feature_description=FEATURE_DESCRIPTION,
    text_preprocessor=None,
    metric_fns=[],
    supports_caching=False)

t5.data.TaskRegistry.add(
    "clinc150_student",
    t5.data.TFExampleTask,
    split_to_filepattern={k: os.path.join(CLINC150_DATA_DIR, v)
                          for k, v in STUDENT_FILENAMES.items()},
    feature_description=FEATURE_DESCRIPTION,
    text_preprocessor=None,
    metric_fns=[t5.evaluation.metrics.accuracy, macro_f1],
    supports_caching=False)

t5.data.TaskRegistry.add(
    "clinc150_student_with_ex2",
    t5.data.TFExampleTask,
    split_to_filepattern={k: os.path.join(CLINC150_DATA_DIR, v)
                          for k, v in STUDENT_WITH_EX2_FILENAMES.items()},
    feature_description=FEATURE_DESCRIPTION,
    text_preprocessor=None,
    metric_fns=[t5.evaluation.metrics.accuracy, macro_f1],
    supports_caching=False)
