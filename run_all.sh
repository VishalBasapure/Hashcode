#!/bin/bash
# Full overnight pipeline: train + evaluate every model on both GPUs, then make the report.
# Skips any model that already has evals/<name>/metrics.json, so it is safe to re-run.
#   bash run_all.sh
set -u
cd /kaggle/working/Hashcode
export TOKENIZERS_PARALLELISM=false
L=/kaggle/working/logs
mkdir -p $L /tmp/merged

say() { echo "[$(date +%H:%M)] $*"; }

train() {  # method seed gpu
  local n=$1_s$2
  if [ -f /tmp/merged/$n/tokenizer_config.json ]; then say "skip train $n (exists)"; return 0; fi
  say "START train $n on GPU $3"
  CUDA_VISIBLE_DEVICES=$3 python train/train.py --method $1 --seed $2 --model /tmp/llama \
      --merge_to /tmp/merged/$n >> $L/$n.train.log 2>&1 \
    && rm -rf /kaggle/working/runs/${n}_*/checkpoint-* \
    && say "DONE  train $n" || say "FAIL  train $n (see logs/$n.train.log)"
}

evalm() {  # name gpu
  if [ -f /kaggle/working/evals/$1/metrics.json ]; then say "skip eval $1 (exists)"; return 0; fi
  say "START eval $1 on GPU $2"
  CUDA_VISIBLE_DEVICES=$2 python eval/run_eval.py --model /tmp/merged/$1 --name $1 >> $L/$1.eval.log 2>&1 \
    && say "DONE  eval $1: $(python -c "import json;m=json.load(open('/kaggle/working/evals/$1/metrics.json'));print('IFEval',round(m.get('ifeval',0),3),'EA',round(m.get('EA',0),3),'HindiNLL',round(m.get('hindi_val_nll',0),3))")" \
    || say "FAIL  eval $1 (see logs/$1.eval.log)"
}

# Core comparison first (seed 0), then the extra seeds for B1 and J1.
q0() { train B1 0 0; evalm B1_s0 0; train B4 0 0; evalm B4_s0 0; train B1 1 0; evalm B1_s1 0; train B1 2 0; evalm B1_s2 0; }
q1() { train J1 0 1; evalm J1_s0 1; train B2 0 1; evalm B2_s0 1; train J1 1 1; evalm J1_s1 1; train J1 2 1; evalm J1_s2 1; }

q0 2>&1 | sed -u 's/^/[gpu0] /' &
q1 2>&1 | sed -u 's/^/[gpu1] /' &
wait
say "all queues finished"
say "START drift"
python eval/drift.py --models P1=/tmp/llama B1_s0=/tmp/merged/B1_s0 B2_s0=/tmp/merged/B2_s0 \
    B4_s0=/tmp/merged/B4_s0 J1_s0=/tmp/merged/J1_s0 >> $L/drift.log 2>&1 && say "DONE  drift" || say "FAIL  drift"
python analysis/report.py
say "ALL DONE - download the Output tab"