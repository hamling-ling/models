#!/bin/bash
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# This script performs the following operations:
# 1. Downloads the Cifar10 dataset
# 2. Trains a CifarNet model on the Cifar10 training set.
# 3. Evaluates the model on the Cifar10 testing set.
#
# Usage:
# cd slim
# ./scripts/train_cifarnet_on_cifar10.sh
set -e

# Where the checkpoint and logs will be saved to.
TRAIN_DIR=cifarnet-model

# Where the dataset is saved to.
DATASET_DIR=cifar10_data

# Download the dataset
python download_and_convert_data.py \
  --dataset_name=cifar10 \
  --dataset_dir=${DATASET_DIR}

# Run training.
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=cifar10 \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=cifarnet \
  --preprocessing_name=cifarnet \
  --max_number_of_steps=100000 \
  --batch_size=128 \
  --save_interval_secs=120 \
  --save_summaries_secs=120 \
  --log_every_n_steps=100 \
  --optimizer=sgd \
  --learning_rate=0.1 \
  --learning_rate_decay_factor=0.1 \
  --num_epochs_per_decay=200 \
  --weight_decay=0.004

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=cifar10 \
  --dataset_split_name=test \
  --dataset_dir=${DATASET_DIR} \
  --model_name=cifarnet

echo -------- exporting --------
export LAST_CHECKPOINT=`head -n1 $TRAIN_DIR/checkpoint | cut -d'"' -f2`
echo LAST_CHECKPOINT=$LAST_CHECKPOINT

python export_inference_graph.py \
    --model_name=cifarnet \
    --dataset_name=cifar10 \
    --output_file=cifarnet_inf_graph.pb

echo -------- freezing --------
python -m tensorflow.python.tools.freeze_graph \
--input_graph=cifarnet_inf_graph.pb \
--input_checkpoint=$TRAIN_DIR/${LAST_CHECKPOINT} \
--input_binary=true \
--output_graph=frozen_cifarnet.pb \
--output_node_names=CifarNet/Predictions/Softmax

echo -------- converting --------
tflite_convert --graph_def_file=frozen_cifarnet.pb \
--output_file=quantized_cifarnet.tflite \
--input_format=TENSORFLOW_GRAPHDEF \
--output_format=TFLITE \
--inference_type=QUANTIZED_UINT8 \
--output_arrays=CifarNet/Predictions/Softmax \
--input_arrays=input \
--mean_values 121 \
--std_dev_values 64
