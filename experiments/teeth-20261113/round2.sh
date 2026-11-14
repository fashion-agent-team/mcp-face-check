#!/bin/bash
# 2차 실험 (젯슨): 데이터 만들기 → 5가지 학습 → 같은 시험 세트로 평가.
# 준비: TEETH_WORK 폴더에 mouths_tight/ (teeth.tar), 이 폴더의 스크립트, python에 torch torchvision opencv numpy pillow.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
export TEETH_WORK=${TEETH_WORK:-$HOME/teeth-work}
PY=${PY:-python3}
cd "$TEETH_WORK"
cp "$HERE"/crop_rule.py . 2>/dev/null || true
mkdir -p runs/v4_384 && cp "$HERE"/models/v4_384_best.pt runs/v4_384/best.pt && cp "$HERE"/models/config.json runs/v4_384/teeth_mobilenetv3.json
[ -f dataset_v2/labels.json ] || $PY "$HERE"/synth_v2_old.py --src mouths_tight --out dataset_v2
[ -f dataset_v4/labels.json ] || $PY "$HERE"/synth_v4.py --variants 4 --src mouths_tight --out dataset_v4
[ -f dataset_v5/labels.json ] || $PY "$HERE"/synth_v5.py --variants 4 --gen-seed 7 --src mouths_tight --out dataset_v5
B=${BATCH:-16}
run() { name=$1; shift; [ -f runs/$name/best.pt ] || $PY "$HERE"/train_adapt.py --name $name --batch $B "$@" 2>&1 | tee runs_$name.log | grep -v Warn | tail -3; }
run full_lr3e5        --mode full --data dataset_v5 --epochs 4 --lr 3e-5
run full_mix          --mode full --data dataset_v5 dataset_v4 --epochs 2 --lr 1e-4
run lora_r4           --mode lora --data dataset_v5 --epochs 4 --lr 1e-3 --rank 4 --alpha 8
run lora_r16          --mode lora --data dataset_v5 --epochs 4 --lr 1e-3 --rank 16 --alpha 32
run lora_mix          --mode lora --data dataset_v5 dataset_v4 --epochs 2 --lr 1e-3 --rank 8
$PY "$HERE"/compare.py runs/v4_384/best.pt runs/full_lr3e5/best.pt runs/full_mix/best.pt runs/lora_r4/best.pt runs/lora_r16/best.pt runs/lora_mix/best.pt | tee round2_compare.txt
for m in full_lr3e5 full_mix lora_r4 lora_r16 lora_mix; do echo "$m $(grep -o '"train_sec": [0-9]*' runs/$m/train.json) $(grep -o '"trainable": [0-9]*' runs/$m/train.json)"; done | tee round2_time.txt
