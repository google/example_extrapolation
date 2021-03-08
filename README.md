# Introduction

This directory contains a partial implementation of


> [Neural Data Augmentation via Example Extrapolation](https://arxiv.org/abs/2102.01335)

> Kenton Lee*, Kelvin Guu*, Luheng He*, Tim Dozat*, Hyung Won Chung*

> (* equal contribution)

# Getting started

We require Python 3.6+ and the latest versions of `mesh_tensorflow` and `t5`:

```bash
pip install -r requirements.txt
```

# Data preprocessing
Create a Google Cloud Storage bucket to store the data that we will generate:

```bash
export CLINC150_DATA_DIR="gs://<YOUR_BUCKET>/<PATH_TO_DATA_DIR>"
```

Download the original data from
https://github.com/clinc/oos-eval/blob/master/data/data_full.json, and run our
preprocessing script to split the data and generate examples for both the Ex2
teacher and the CLINC150 student.

```bash
python -m preprocess_clinc150 \
  --data_path=<PATH_TO_DOWNLOADED_data_full.json> \
  --output_dir=${CLINC150_DATA_DIR}
```

# Setting up T5 training

We provide a few minimal commands for using T5. We recommend taking a look at
https://github.com/google-research/text-to-text-transfer-transformer for details
and following the latest recommendations for how to run T5.

Please set the following environment variables based on your GCP project.

```bash
export PROJECT=<YOUR_PROJECT_NAME>
export ZONE=<YOUR_PROJECT_ZONE>
export TPU_NAME=ex2-tpu
export TPU_SIZE=v3-8
export MODELS_DIR="gs://<YOUR_BUCKET>/<PATH_TO_MODELS_DIR>"
export CLINC150_DATA_DIR="gs://<YOUR_BUCKET>/<PATH_TO_DATA_DIR>"
export TOTAL_TEACHER_TRAIN_STEPS=1001000
export TOTAL_STUDENT_TRAIN_STEPS=1020000
```

Note that the `CLINC150_DATA_DIR` environment variable is essential because
`tasks.py` uses it to determine where to read the data from.

Launch a Cloud TPU instance ([CTPU reference](https://cloud.google.com/tpu/docs/ctpu-reference)):

```bash
ctpu up --name=$TPU_NAME --project=$PROJECT --zone=$ZONE --tpu-size=$TPU_SIZE \
        --tpu-only --noconf
```

# Ex2 teacher

First we train the Ex2 teacher on the many-shot data and use it to
infer new synthetic data on the few-shot data.

## Teacher training

```bash
python3 -m t5.models.mesh_transformer_main \
  --tpu="${TPU_NAME}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODELS_DIR}/clinc150_teacher" \
  --module_import="tasks" \
  --gin_file="dataset.gin" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'" \
  --gin_param="utils.run.sequence_length = {'inputs': 512, 'targets': 128}" \
  --gin_param="utils.run.batch_size = ('sequences_per_batch', 128)" \
  --gin_param="MIXTURE_NAME = 'clinc150_teacher'" \
  --gin_file="gs://t5-data/pretrained_models/t5.1.1.xl/operative_config.gin" \
  --gin_param="run.train_steps = ${TOTAL_TEACHER_TRAIN_STEPS}"
```

## Teacher inference

```bash
python3 -m t5.models.mesh_transformer_main \
  --tpu="${TPU_NAME}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODELS_DIR}/clinc150_teacher" \
  --module_import="tasks" \
  --gin_file="${MODELS_DIR}/clinc150_teacher/operative_config.gin" \
  --gin_file="infer.gin" \
  --gin_file="sample_decode.gin" \
  --gin_param="input_filename = '${CLINC150_DATA_DIR}/teacher_inputs.txt'"\
  --gin_param="output_filename = '${CLINC150_DATA_DIR}/teacher_predictions.txt'"\
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'"\
  --gin_param="utils.run.sequence_length = {'inputs': 512, 'targets': 128}" \
  --gin_param="utils.run.batch_size = ('sequences_per_batch', 128)" \
  --gin_param="Bitransformer.decode.max_decode_length = 128" \
  --gin_param="infer_checkpoint_step = ${TOTAL_TEACHER_TRAIN_STEPS}"
```

The predictions need to be stitched back together with the original data using
our postprocessing script:

```bash
python3 -m postprocess_teacher_predictions \
  --teacher_intents_path="${CLINC150_DATA_DIR}/teacher_intents.txt" \
  --teacher_predictions_path="${CLINC150_DATA_DIR}/teacher_predictions.txt-${TOTAL_TEACHER_TRAIN_STEPS}" \
  --original_train_path="${CLINC150_DATA_DIR}/student_train.tfr" \
  --output_train_path="${CLINC150_DATA_DIR}/student_train_with_ex2.tfr"
```

# CLINC150 Student

Finally, we train the CLINC150 student on either the original training data or
the augmented training data.

## Student training

```bash
python3 -m t5.models.mesh_transformer_main \
  --tpu="${TPU_NAME}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODELS_DIR}/clinc150_student_with_ex2" \
  --module_import="tasks" \
  --gin_file="dataset.gin" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'" \
  --gin_param="utils.run.sequence_length = {'inputs': 128, 'targets': 32}" \
  --gin_param="utils.run.batch_size = ('sequences_per_batch', 128)" \
  --gin_param="MIXTURE_NAME = 'clinc150_student_with_ex2'" \
  --gin_file="gs://t5-data/pretrained_models/t5.1.1.xl/operative_config.gin" \
  --gin_param="run.train_steps = ${TOTAL_STUDENT_TRAIN_STEPS}"
```

## Student evaluation

```bash
python3 -m t5.models.mesh_transformer_main \
  --tpu="${TPU_NAME}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODELS_DIR}/clinc150_student_with_ex2" \
  --gin_file="${MODELS_DIR}/clinc150_student_with_ex2/operative_config.gin" \
  --module_import="tasks" \
  --gin_file="eval.gin" \
  --gin_file="greedy_decode.gin" \
  --gin_param="run.dataset_split = 'fewshot_validation'" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'" \
  --gin_param="utils.run.sequence_length = {'inputs': 128, 'targets': 32}" \
  --gin_param="utils.run.batch_size = ('sequences_per_batch', 128)" \
  --gin_param="Bitransformer.decode.max_decode_length = 128" \
  --gin_param="MIXTURE_NAME = 'clinc150_student'" \
  --gin_param="eval_checkpoint_step = ${TOTAL_STUDENT_TRAIN_STEPS}"
```

Replace all instances of `clinc150_student_with_ex2` above with
`clinc150_student` to run the baseline that does not use any synthetic data.

Replace `fewshot_validation` above with `overall_validation` to compute overall
results.

# Replicating results

For conceptual simplicity, this directory only contains the minimal code
necessary to demonstrate the Ex2 recipe for CLINC150.

In order to replicate the
CLINC150 results from the paper, the following additional steps are neccessary:

* Continuous student and teacher evaluation for early stopping.
  * We simply use a fixed number of steps.
* Preprocessing and evaluating on the test set for the student model.
  * We only provide code for evaluating on the validation set.
* Aggregating results by cross-validating the few-shot domain.
  * We only provide code for experimenting with a single few-shot domain.

For reference, in the released implementation, when the `banking` domain is the
few-shot domain, you should expect the following results:

| Model     | Overall Acc. | Overall Macro F1 | Few-shot Acc. | Few-shot Macro F1 |
| --------- | ------------ | ---------------- | ------------- | ----------------- |
| T5        | ~97%         | ~96%             | ~95%          | ~63%              |
| T5 w/ Ex2 | ~97%         | ~96%             | ~96%          | ~90%              |
