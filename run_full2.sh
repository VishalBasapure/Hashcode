#!/bin/bash
# Round 4b: FULL fine-tuning of every transformer layer, token embeddings frozen (so J1 fits on a 15 GB T4),
# identical for B1 and J1 so the comparison stays fair. Hindi only, 1 epoch, LR 5e-5, seed 0.
#   bash run_full2.sh
set -u
cd /kaggle/working/Hashcode
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=/kaggle/working/logs
mkdir -p $L /tmp/merged
say() { echo "[$(date +%H:%M)] $*"; }

train() {  # method gpu tag micro_bsz grad_accum
  local n=$1_fullfe_s0
  CUDA_VISIBLE_DEVICES=$2 python train/train.py --method $1 --seed 0 --full --freeze_emb --epochs 1 --lr 5e-5 \
      --tag $3 --micro_bsz $4 --grad_accum $5 --model /tmp/llama --merge_to /tmp/merged/$n >> $L/$n.train.log 2>&1
}

run() {  # method gpu
  local n=$1_fullfe_s0
  say "START train $n (full fine-tuning, embeddings frozen) on GPU $2"
  if train $1 $2 _fullfe 2 16; then
    say "DONE  train $n"
  elif grep -qi "out of memory" $L/$n.train.log; then
    say "OOM - retrying $n with micro batch 1"
    train $1 $2 _fullfer 1 32 && say "DONE  train $n (micro batch 1)" || { say "FAIL  train $n"; tail -5 $L/$n.train.log; return 1; }
  else
    say "FAIL  train $n"; tail -25 $L/$n.train.log; return 1
  fi
  grep -h '"sec_per_step"\|"peak_gpu_gb"\|"train_seconds"' /kaggle/working/runs/$1_fullfe*_s0_*/summary.json 2>/dev/null | tr -d '\n'; echo
  say "START eval $n"
  CUDA_VISIBLE_DEVICES=$2 python eval/run_eval.py --model /tmp/merged/$n --name $n >> $L/$n.eval.log 2>&1 \
    && say "DONE  eval $n: $(python -c "import json;m=json.load(open('/kaggle/working/evals/$n/metrics.json'));print('IFEval',round(m.get('ifeval',0),3),'EA',round(m.get('EA',0),3),'HindiNLL',round(m.get('hindi_val_nll',0),3))")" \
    || say "FAIL  eval $n"
}

run J1 0 2>&1 | sed -u 's/^/[gpu0] /' &
run B1 1 2>&1 | sed -u 's/^/[gpu1] /' &
wait
python eval/drift.py --models P1=/tmp/llama B1_fullfe_s0=/tmp/merged/B1_fullfe_s0 J1_fullfe_s0=/tmp/merged/J1_fullfe_s0 \
    >> $L/drift.log 2>&1 && say "DONE drift" && cat /kaggle/working/drift.json | head -c 900; echo
python analysis/report.py
say "ALL DONE - download the Output tab"