#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import torch
import csv
import numpy as np
import gc
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


print(f"PyTorch version: {torch.__version__}")


class SNPDataset(Dataset):
    def __init__(self, ids, ref_seqs, alt_seqs):
        self.ids = ids
        self.ref_seqs = ref_seqs
        self.alt_seqs = alt_seqs

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.ids[idx], self.ref_seqs[idx], self.alt_seqs[idx]


def collate_fn(batch, tokenizer, max_length):
    ids, ref_seqs, alt_seqs = zip(*batch)
    all_seqs = list(ref_seqs) + list(alt_seqs)

    encodings = tokenizer(
        all_seqs,
        padding="longest",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    bsz = len(ref_seqs)
    ref_inputs = {k: v[:bsz] for k, v in encodings.items()}
    alt_inputs = {k: v[bsz:] for k, v in encodings.items()}

    return list(ids), ref_inputs, alt_inputs


def main(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------- Tokenizer -------------------
    tokenizer = AutoTokenizer.from_pretrained(
        "zhihan1996/DNABERT-2-117M",
        trust_remote_code=True,
        use_fast=True,
        padding_side="right",
        model_max_length=args.max_length
    )

    # ------------------- Model (FP16) -------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    model.to(device)
    model.eval()
    print(f"Model loaded in FP16: {args.model_path}")

    # ------------------- Parameters -------------------
    positive_class = 1
    use_amp = (device.type == "cuda")
    print(f"AMP enabled: {use_amp}")

    # ------------------- Input / Output -------------------
    input_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.csv')])
    if not input_files:
        raise FileNotFoundError(f"No .csv files in {args.input_dir}")
    print(f"Found {len(input_files)} files.")

    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    print(f"Output: {os.path.abspath(args.output_file)}")

    # Full output header
    header = [
        "ID",
        "ref_logit_1", "alt_logit_1", "logit_diff",
        "ref_prob_1", "alt_prob_1", "prob_diff",
        "prob_log_odds_diff", "l2_dist_cls_emb"
    ]

    all_results = []
    epsilon = 1e-8

    # ------------------- Process each input file -------------------
    for file_name in tqdm(input_files, desc="Processing files"):
        file_path = os.path.join(args.input_dir, file_name)
        df = pd.read_csv(file_path)

        if not {'ID', 'REF_seq', 'ALT_seq'}.issubset(df.columns):
            print(f"Skipping {file_name}: missing required columns")
            continue

        dataset = SNPDataset(df['ID'].tolist(), df['REF_seq'].tolist(), df['ALT_seq'].tolist())
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, tokenizer, args.max_length),
            num_workers=2,
            pin_memory=True
        )

        # ------------------- Inference -------------------
        with torch.no_grad():
            for batch_ids, ref_inputs, alt_inputs in dataloader:
                try:
                    # Move tensors to GPU
                    ref_inputs = {k: v.to(device, non_blocking=True) for k, v in ref_inputs.items()}
                    alt_inputs = {k: v.to(device, non_blocking=True) for k, v in alt_inputs.items()}

                    with torch.cuda.amp.autocast(enabled=use_amp):
                        ref_out = model(**ref_inputs, output_hidden_states=True)
                        alt_out = model(**alt_inputs, output_hidden_states=True)

                        ref_logits = ref_out.logits
                        alt_logits = alt_out.logits

                        # CLS embeddings
                        ref_cls = ref_out.hidden_states[:, 0, :].cpu().numpy()
                        alt_cls = alt_out.hidden_states[:, 0, :].cpu().numpy()

                        # Probabilities and logits
                        ref_prob = torch.softmax(ref_logits, dim=-1)[:, positive_class].cpu().numpy()
                        alt_prob = torch.softmax(alt_logits, dim=-1)[:, positive_class].cpu().numpy()
                        ref_logit1 = ref_logits[:, positive_class].cpu().numpy()
                        alt_logit1 = alt_logits[:, positive_class].cpu().numpy()

                    # ---------- Feature calculation ----------
                    logit_diff = alt_logit1 - ref_logit1
                    prob_diff = alt_prob - ref_prob

                    logodds_ref = np.log((ref_prob + epsilon) / (1 - ref_prob + epsilon))
                    logodds_alt = np.log((alt_prob + epsilon) / (1 - alt_prob + epsilon))
                    logodds_diff = logodds_alt - logodds_ref

                    l2_dist = np.linalg.norm(ref_cls - alt_cls, ord=2, axis=1)

                    # ---------- Save all results (no filtering) ----------
                    for j in range(len(batch_ids)):
                        all_results.append({
                            "ID": batch_ids[j],
                            "ref_logit_1": round(float(ref_logit1[j]), 6),
                            "alt_logit_1": round(float(alt_logit1[j]), 6),
                            "logit_diff": round(float(logit_diff[j]), 6),
                            "ref_prob_1": round(float(ref_prob[j]), 6),
                            "alt_prob_1": round(float(alt_prob[j]), 6),
                            "prob_diff": round(float(prob_diff[j]), 6),
                            "prob_log_odds_diff": round(float(logodds_diff[j]), 6),
                            "l2_dist_cls_emb": round(float(l2_dist[j]), 6)
                        })

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print("CUDA out-of-memory detected. Reducing batch size is recommended.")
                        torch.cuda.empty_cache()
                        break
                    else:
                        raise e

                # Memory cleanup
                del ref_inputs, alt_inputs, ref_logits, alt_logits
                del ref_out, alt_out, ref_cls, alt_cls
                gc.collect()
                torch.cuda.empty_cache()

    # ------------------- Write all results -------------------
    with open(args.output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nDone! {len(all_results)} SNPs processed and saved (no filtering).")
    print(f"Output: {os.path.abspath(args.output_file)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DNABERT-2: Full Feature Prediction (No Threshold Filtering)"
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to fine-tuned DNABERT-2 model")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing input CSV files (ID, REF_seq, ALT_seq)")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output CSV containing full features for all SNPs")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Large batch size works well with FP16 + AMP")
    parser.add_argument("--gpu_id", type=int, default=0)

    args = parser.parse_args()
    main(args)