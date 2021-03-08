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
"""Tensorflow utilities."""

import tensorflow as tf


def add_text_feature(name, feature, example):
  example.features.feature[name].bytes_list.value.append(
      feature.encode("utf-8"))


def create_tf_example(inputs, targets):
  example = tf.train.Example()
  add_text_feature("inputs", inputs, example)
  add_text_feature("targets", targets, example)
  return example
