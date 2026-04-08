#!/bin/bash

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 --cell <CELL> --list <VARIANT_LIST.bim>"
    exit 1
fi

if [ "$1" = "--cell" ] && [ "$3" = "--list" ]; then
    cell="$2"
    list_file="$4"
else
    echo "Usage: $0 --cell <CELL> --list <VARIANT_LIST.bim>"
    exit 1
fi

GENOME="./data/ref_file/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
CHROM_SIZES="./data/ref_file/GRCh38_EBV.standard.chrom.sizes.tsv"
MODEL_ROOT="/fine-tuned_model/chrombpnet/sc-PBMC"
PEAK_DIR="./data/chrombpnet_peaks"

list_name=$(basename "${list_file%.*}")
out_dir="./chrombpnet_feature/${list_name}_${cell}"
peak_file="${PEAK_DIR}/${cell}__peaks_bpnet.narrowPeak"

mkdir -p "${out_dir}"

declare -a score_files=()

export PYTHONPATH="${PYTHONPATH}:/root/autodl-tmp/variant_scoring/src"
CUDA_VISIBLE_DEVICES=1

# 遍历 fold 0 到 4
for fold in {0..4}; do
    model="${MODEL_ROOT}/${cell}/fold_${fold}/models/chrombpnet_nobias.h5"
    bias="${MODEL_ROOT}/${cell}/fold_${fold}/models/bias_model_scaled.h5"
    out_prefix="${out_dir}/${list_name}.${cell}.fold${fold}"

    echo "Running ${cell} fold ${fold} for ${list_name}..."

    time python ./variant_scoring/src/variant_scoring_noshuffle.py \
        -l "${list_file}" \
        -g "${GENOME}" \
        -m "${model}" \
        -b "${bias}" \
        -o "${out_prefix}" \
        -s "${CHROM_SIZES}" \
        -p "${peak_file}" \
        --no_hdf5 
    
    if [[ ! -f "${out_prefix}.variant_scores.tsv" ]]; then
        echo "Error: scoring failed for fold ${fold}"
        exit 1
    fi

    score_files+=("$(basename "${out_prefix}.variant_scores.tsv")")
done

# 汇总 across folds
summary_prefix="${out_dir}/${list_name}.${cell}.across_folds"
python ./variant_scoring/src/variant_summary_across_folds.py \
    --score_dir "${out_dir}" \
    --score_list "${score_files[@]}" \
    --out_prefix "${summary_prefix}" \
    --schema chrombpnet

echo "✅ Done! Results in: ${out_dir}/"
echo "Final file: ${summary_prefix}.mean.variant_scores.tsv"