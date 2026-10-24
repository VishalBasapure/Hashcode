#!/bin/bash
# Extra, exploratory forgetting test (added after the 3-epoch run showed no forgetting):
# 5x higher learning rate (1e-3) for 2 epochs, B1 vs J1, seed 0. Labelled "exploratory" in all results.
set -u
cd /kaggle/working/Hashcode
export TOKENIZERS_PARALLELISM=false
L=/kaggle/working/logs
mkdir -p $L /tmp/merged
say() { echo "[$(date +%H:%M)] $*"; }

run() {  # method gpu
  local n=$1_lr_s0
  say "START train $n on GPU $2"
  CUDA_VISIBLE_DEVICES=$2 python train/train.py --method $1 --seed 0 --epochs 2 --lr 1e-3 --tag _lr --model /tmp/llama \
      --merge_to /tmp/merged/$n >> $L/$n.train.log 2>&1 && rm -rf /kaggle/working/runs/${n}_*/checkpoint-* \
    && say "DONE  train $n" || { say "FAIL  train $n"; return 1; }
  say "START eval $n"
  CUDA_VISIBLE_DEVICES=$2 python eval/run_eval.py --model /tmp/merged/$n --name $n >> $L/$n.eval.log 2>&1 \
    && say "DONE  eval $n: $(python -c "import json;m=json.load(open('/kaggle/working/evals/$n/metrics.json'));print('IFEval',round(m.get('ifeval',0),3),'EA',round(m.get('EA',0),3),'HindiNLL',round(m.get('hindi_val_nll',0),3))")" \
    || say "FAIL  eval $n"
}

run B1 0 2>&1 | sed -u 's/^/[gpu0] /' &
run J1 1 2>&1 | sed -u 's/^/[gpu1] /' &
wait
python eval/drift.py --models P1=/tmp/llama B1_lr_s0=/tmp/merged/B1_lr_s0 J1_lr_s0=/tmp/merged/J1_lr_s0 \
    >> $L/drift.log 2>&1 && say "DONE drift"
python analysis/report.py
say "ALL DONE - download the Output tab"