#!/bin/bash
# 젯슨에서 멈추라고 할 때까지 눈화장 모델을 차례로 학습·평가. 결과: training/out/jetson/results.jsonl, best.txt
# 사용: nohup bash training/jetson/loop.sh > training/out/jetson/loop.log 2>&1 &
# 다른 스레드가 젯슨 GPU를 쓰려면: echo 'closet ...' > ~/gpu-train.lock (끝나면 rm) → 이 학습은 5분 안에 저장된 에폭에서 멈췄다가 잠금이 풀리면 이어서 함
# 멈추기: touch training/out/jetson/STOP (지금 학습 중인 에폭이 끝나면 저장돼 있어서 다음에 이어서 가능)
# 메모리 감시: 남은 메모리가 1GB 아래면 학습이 스스로 멈추고(코드 75), 10분 쉬었다 이어서 함
cd "$(dirname "$0")"
OUTD=../out/jetson; mkdir -p $OUTD
PY=${PY:-python3}
stop() { [ -f $OUTD/STOP ]; }
# 젯슨 GPU 같이 쓰기: 다른 스레드(옷장 등)는 돌리는 동안 ~/gpu-train.lock에 자기 이름을 적고 끝나면 지움 → 그동안 기다림
LOCK=${GPU_LOCK:-$HOME/gpu-train.lock}
mine() { [ -f "$LOCK" ] && head -c 10 "$LOCK" | grep -q '^face-check'; }
take() { while [ -f "$LOCK" ] && ! mine && ! stop; do echo "$(date) 다른 학습 중: $(head -1 "$LOCK") → 5분 기다림"; sleep 300; done; echo "face-check eye loop $(date)" > "$LOCK"; }
trap 'mine && rm -f "$LOCK"' EXIT
run() {  # 이름 나머지-인자
  local name=$1; shift
  [ -f $OUTD/$name/eval.json ] && return
  while ! stop; do
    take
    $PY train_eye.py --name "$name" "$@"; rc=$?
    [ $rc -eq 0 ] && break
    if [ $rc -eq 75 ]; then echo "$(date) $name 메모리 부족 또는 다른 학습 → 잠금 풀고 10분 쉼"; mine && rm -f "$LOCK"; sleep 600; continue; fi
    echo "$(date) $name 실패 rc=$rc → 건너뜀"; return
  done
  $PY ens.py
}
# 기준점: 지금 서버 모델(거른 평가 약 71%)
[ -f $OUTD/server/eval.json ] || $PY train_eye.py --name server --eval-onnx ../../src/main/resources/models/eye_makeup.onnx
# 1) 서버 모델과 같은 설정, 씨앗만 바꿔 길게  2) 더 넓은 망  3) 더 긴 학습  4) 낮은 학습률
run s13e16 --seed 13 --epochs 16
run w15s13 --seed 13 --epochs 16 --width 1.5
run s21e24 --seed 21 --epochs 24
run w20s13 --seed 13 --epochs 16 --width 2.0
run s13e16lr2 --seed 13 --epochs 16 --lr 2e-3
# 그다음은 멈출 때까지 씨앗을 바꿔 앙상블 후보를 쌓음
s=31
while ! stop; do run s${s}e16 --seed $s --epochs 16; s=$((s+10)); done
echo "$(date) STOP"
