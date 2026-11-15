#!/bin/bash
# 눈화장 학습 데이터(약 8.5만 쌍)와 평가 묶음 만들기. 맥(M2) 권장: mediapipe가 필요하고 CPU 4프로세스로 1~2시간.
# 사용: bash training/jetson/make_data.sh <CelebA 폴더(…/CelebA, Anno·Img/img_celeba 있는 곳)> <FFHQ 사진 폴더>
#   CelebA가 없으면: python3 training/fetch_celeba.py <받을 폴더> --n 30000  → <받을 폴더>/home/ubuntu/CelebA
# 결과: training/out/pack/{a,b,y}.npy(학습, 약 5.9GB) + training/out/pack/eval_*.npy(평가 FFHQ 앞 600장 중 정면 약 346장)
# → 젯슨에 training/out/pack 폴더째 복사(rsync -a). CelebA·FFHQ는 비상업: 사진·만든 파일은 레포에 올리지 마세요.
set -e
C=$(cd "$1" && pwd); F=$(cd "$2" && pwd)
cd "$(dirname "$0")/.."
mkdir -p out
run() { [ -f "out/$1.npz" ] && { echo "skip $1"; return; }; shift; echo "== $*"; python3 "$@"; }
run eye_makeup_1_train eye_makeup_data.py "$C" 27000 4 1
run eye_makeup_2_train eye_makeup_data.py "$C" 27000 4 2
run eye_makeup_ffhq_train eye_ffhq_data.py "$F" 40
run eye_hn_ffhq eye_hardneg_data.py ffhq "$F" 25 eye_hn_ffhq
run eye_hn_celeba eye_hardneg_data.py celeba "$C" 14000 eye_hn_celeba 5
python3 jetson/pack.py
python3 jetson/eval_cache.py "$F"
du -sh out/pack
