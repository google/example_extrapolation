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
"""Preprocess CLINC150 data from https://arxiv.org/abs/1909.02027."""
import collections
import hashlib
import json
import os
import random


from absl import app
from absl import flags
import numpy as np
import tensorflow as tf
import tf_utils

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "data_path", None,
    "Path to https://github.com/clinc/oos-eval/blob/master/data/data_full.json")
flags.DEFINE_string("output_dir", None, "Output directory.")
flags.DEFINE_integer("num_exemplars", 10,
                     "Number of exemplars as input to each Ex2 example.")
flags.DEFINE_string("validation_domain", "banking", "The held-out domain.")

# See https://github.com/clinc/oos-eval/blob/master/supplementary.pdf.
DOMAIN_TO_INTENT = {
    "banking": [
        "transfer", "transactions", "balance", "freeze_account", "pay_bill",
        "bill_balance", "bill_due", "interest_rate", "routing", "min_payment",
        "order_checks", "pin_change", "report_fraud", "account_blocked",
        "spending_history"
    ],
    "credit_card": [
        "credit_score", "report_lost_card", "credit_limit", "rewards_balance",
        "new_card", "application_status", "card_declined", "international_fees",
        "apr", "redeem_rewards", "credit_limit_change", "damaged_card",
        "replacement_card_duration", "improve_credit_score", "expiration_date"
    ],
    "dining": [
        "recipe", "restaurant_reviews", "calories", "nutrition_info",
        "restaurant_suggestion", "ingredients_list", "ingredient_substitution",
        "cook_time", "food_last", "meal_suggestion", "restaurant_reservation",
        "confirm_reservation", "how_busy", "cancel_reservation",
        "accept_reservations"
    ],
    "home": [
        "shopping_list", "shopping_list_update", "next_song", "play_music",
        "update_playlist", "todo_list", "todo_list_update", "calendar",
        "calendar_update", "what_song", "order", "order_status", "reminder",
        "reminder_update", "smart_home"
    ],
    "auto": [
        "traffic", "directions", "gas", "gas_type", "distance",
        "current_location", "mpg", "oil_change_when", "oil_change_how",
        "jump_start", "uber", "schedule_maintenance", "last_maintenance",
        "tire_pressure", "tire_change"
    ],
    "travel": [
        "book_flight", "book_hotel", "car_rental", "travel_suggestion",
        "travel_alert", "travel_notification", "carry_on", "timezone",
        "vaccines", "translate", "flight_status", "international_visa",
        "lost_luggage", "plug_type", "exchange_rate"
    ],
    "utility": [
        "time", "alarm", "share_location", "find_phone", "weather", "text",
        "spelling", "make_call", "timer", "date", "calculator",
        "measurement_conversion", "flip_coin", "roll_dice", "definition"
    ],
    "work": [
        "direct_deposit", "pto_request", "taxes", "payday", "w2", "pto_balance",
        "pto_request_status", "next_holiday", "insurance", "insurance_change",
        "schedule_meeting", "pto_used", "meeting_schedule", "rollover_401k",
        "income"
    ],
    "small_talk": [
        "greeting", "goodbye", "tell_joke", "where_are_you_from",
        "how_old_are_you", "what_is_your_name", "who_made_you", "thank_you",
        "what_can_i_ask_you", "what_are_your_hobbies", "do_you_have_pets",
        "are_you_a_bot", "meaning_of_life", "who_do_you_work_for", "fun_fact"
    ],
    "meta": [
        "change_ai_name", "change_user_name", "cancel", "user_name",
        "reset_settings", "whisper_mode", "repeat", "no", "yes", "maybe",
        "change_language", "change_accent", "change_volume", "change_speed",
        "sync_device"
    ],
    "oos": ["oos"]
}


def get_hash(x):
  return int(hashlib.sha1(x.encode("utf-8")).hexdigest(), 16)


def truncate_data(data, num_examples):
  """Used to simulate the few-shot setting for validation intents."""
  return sorted(data, key=get_hash)[:num_examples]


def exemplars_to_inputs(exemplar_queries):
  return " | ".join(exemplar_queries)


def write_student_data(data,
                       output_path):
  """Write student data."""
  examples = []
  for d in data:
    for intent, queries in d.items():
      examples.extend(tf_utils.create_tf_example(q, intent) for q in queries)
  random.shuffle(examples)
  print(f"Writing {len(examples)} student examples to {output_path}")
  with tf.io.TFRecordWriter(output_path) as writer:
    for e in examples:
      writer.write(e.SerializeToString())


def write_teacher_data(data,
                       output_path,
                       num_exemplars):
  """Write teacher data."""
  examples = []
  for queries in data.values():
    for new_query in queries:
      candidate_exemplars = [q for q in queries if q != new_query]
      exemplar_queries = random.sample(candidate_exemplars, num_exemplars)
      examples.append(tf_utils.create_tf_example(
          exemplars_to_inputs(exemplar_queries), new_query))
  random.shuffle(examples)
  print(f"Writing {len(examples)} teacher examples to {output_path}")
  with tf.io.TFRecordWriter(output_path) as writer:
    for e in examples:
      writer.write(e.SerializeToString())


def main(_):
  ###############################
  ## Read original CLINC150 data.
  ###############################
  data = collections.defaultdict(lambda: collections.defaultdict(list))
  with tf.io.gfile.GFile(FLAGS.data_path, "r") as data_file:
    for split_name, split_data in json.load(data_file).items():
      for query, intent in split_data:
        data[split_name][intent].append(query)

  ##################
  ## Split the data.
  ##################
  train_data = data["train"]
  val_data = data["val"]
  validation_intents = set(DOMAIN_TO_INTENT[FLAGS.validation_domain])

  # Train and validation instances for the training intents.
  train_train_data = {k: v for k, v in
                      train_data.items() if k not in validation_intents}
  train_val_data = {k: v for k, v in
                    val_data.items() if k not in validation_intents}

  # Train and validation instances for the validation intents.
  val_train_data = {k: truncate_data(v, FLAGS.num_exemplars) for k, v in
                    train_data.items() if k in validation_intents}
  val_val_data = {k: v for k, v in
                  val_data.items() if k in validation_intents}

  ##################
  ## Write the data.
  ##################
  tf.io.gfile.makedirs(FLAGS.output_dir)

  # Student training data.
  write_student_data([train_train_data, val_train_data],
                     os.path.join(FLAGS.output_dir, "student_train.tfr"))
  write_student_data([train_val_data, val_val_data],
                     os.path.join(FLAGS.output_dir, "student_overall_val.tfr"))
  write_student_data([val_val_data],
                     os.path.join(FLAGS.output_dir, "student_fewshot_val.tfr"))

  # Teacher training data.
  write_teacher_data(train_train_data,
                     os.path.join(FLAGS.output_dir, "teacher_train.tfr"),
                     FLAGS.num_exemplars)
  write_teacher_data(train_val_data,
                     os.path.join(FLAGS.output_dir, "teacher_val.tfr"),
                     FLAGS.num_exemplars)

  # Teacher inference data.
  synthetic_examples_per_intent = int(np.median(
      [len(e) for e in train_train_data.values()]))
  with tf.io.gfile.GFile(os.path.join(
      FLAGS.output_dir, "teacher_intents.txt"), "w") as intents_file:
    with tf.io.gfile.GFile(os.path.join(
        FLAGS.output_dir, "teacher_inputs.txt"), "w") as inputs_file:
      for intent, queries in val_train_data.items():
        for _ in range(synthetic_examples_per_intent):
          exemplar_queries = random.sample(queries, FLAGS.num_exemplars)
          inputs_file.write(exemplars_to_inputs(exemplar_queries))
          intents_file.write(intent)
          inputs_file.write("\n")
          intents_file.write("\n")


if __name__ == "__main__":
  app.run(main)
