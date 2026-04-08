#!/bin/bash

python DNABERT2_testSNP_feature_pred.py \
  --model_path ./fine-tuned_model/dnabert2/sc-PBMC/dnabert2_finetune_ABC/checkpoint-2400 \
  --input_dir ./SNP_for_pred \
  --output_file ./dnabert2_feature/SNP_for_pred_ABC.csv \
  --max_length 120 \
  --batch_size 256

python DNABERT2_testSNP_feature_pred.py \
  --model_path ./fine-tuned_model/dnabert2/sc-PBMC/dnabert2_finetune_Bmemory/checkpoint-3200 \
  --input_dir ./SNP_for_pred \
  --output_file ./dnabert2_feature/SNP_for_pred_Bmemory.csv \
  --max_length 120 \
  --batch_size 256

python DNABERT2_testSNP_feature_pred.py \
  --model_path ./fine-tuned_model/dnabert2/sc-PBMC/dnabert2_finetune_Bnaive/checkpoint-3400 \
  --input_dir ./SNP_for_pred \
  --output_file ./dnabert2_feature/SNP_for_pred_Bnaive.csv \
  --max_length 120 \
  --batch_size 256

python DNABERT2_testSNP_feature_pred.py \
  --model_path ./fine-tuned_model/dnabert2/sc-PBMC/dnabert2_finetune_CD14mono/checkpoint-3800 \
  --input_dir ./SNP_for_pred \
  --output_file ./dnabert2_feature/SNP_for_pred_CD14mono.csv \
  --max_length 120 \
  --batch_size 256

python DNABERT2_testSNP_feature_pred.py \
  --model_path ./fine-tuned_model/dnabert2/sc-PBMC/dnabert2_finetune_CD16mono/checkpoint-3600 \
  --input_dir ./SNP_for_pred \
  --output_file ./dnabert2_feature/SNP_for_pred_CD16mono.csv \
  --max_length 120 \
  --batch_size 256

python DNABERT2_testSNP_feature_pred.py \
  --model_path ./fine-tuned_model/dnabert2/sc-PBMC/dnabert2_finetune_CD4Naive/checkpoint-3600 \
  --input_dir ./SNP_for_pred \
  --output_file ./dnabert2_feature/SNP_for_pred_CD4Naive.csv \
  --max_length 120 \
  --batch_size 256

python DNABERT2_testSNP_feature_pred.py \
  --model_path ./fine-tuned_model/dnabert2/sc-PBMC/dnabert2_finetune_CD4Tcm/checkpoint-2600 \
  --input_dir ./SNP_for_pred \
  --output_file ./dnabert2_feature/SNP_for_pred_CD4Tcm.csv \
  --max_length 120 \
  --batch_size 256

python DNABERT2_testSNP_feature_pred.py \
  --model_path ./fine-tuned_model/dnabert2/sc-PBMC/dnabert2_finetune_CD8Naive/checkpoint-3400 \
  --input_dir ./SNP_for_pred \
  --output_file ./dnabert2_feature/SNP_for_pred_CD8Naive.csv \
  --max_length 120 \
  --batch_size 256

python DNABERT2_testSNP_feature_pred.py \
  --model_path ./fine-tuned_model/dnabert2/sc-PBMC/dnabert2_finetune_CD8Tem/checkpoint-3000 \
  --input_dir ./SNP_for_pred \
  --output_file ./dnabert2_feature/SNP_for_pred_CD8Tem.csv \
  --max_length 120 \
  --batch_size 256

python DNABERT2_testSNP_feature_pred.py \
  --model_path ./fine-tuned_model/dnabert2/sc-PBMC/dnabert2_finetune_DC/checkpoint-1800 \
  --input_dir ./SNP_for_pred \
  --output_file ./dnabert2_feature/SNP_for_pred_DC.csv \
  --max_length 120 \
  --batch_size 256

python DNABERT2_testSNP_feature_pred.py \
  --model_path ./fine-tuned_model/dnabert2/sc-PBMC/dnabert2_finetune_NK/checkpoint-3200 \
  --input_dir ./SNP_for_pred \
  --output_file ./dnabert2_feature/SNP_for_pred_NK.csv \
  --max_length 120 \
  --batch_size 256

python DNABERT2_testSNP_feature_pred.py \
  --model_path ./fine-tuned_model/dnabert2/sc-PBMC/dnabert2_finetune_PB/checkpoint-2200 \
  --input_dir ./SNP_for_pred \
  --output_file ./dnabert2_feature/SNP_for_pred_PB.csv \
  --max_length 120 \
  --batch_size 256
  