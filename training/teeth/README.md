# 치아 이물질 모델 학습 (teeth_check)

`teeth_check`가 쓰는 `src/main/resources/models/teeth_mobilenetv3.onnx`를 만드는 스크립트입니다.
공개 얼굴 사진(FFHQ)에서 입을 잘라 음식 조각을 코드로 그려 넣고(합성), MobileNetV3-Small로 "깨끗함 / 낀 것 있음"을 학습합니다.

- FFHQ(NVIDIA)는 CC BY-NC-SA 4.0, **비상업용**입니다. 얼굴 사진·잘라 낸 입·합성 사진은 레포에 올리지 않습니다(`.gitignore`).
  발표 자료에는 몇 장만 쓰고 "FFHQ (NVIDIA), CC BY-NC-SA 4.0"을 밝힙니다.
- 사용자 사진은 학습에 쓰지 않고, 외부 AI로 보내지 않습니다.

## 순서

| 단계 | 스크립트 | 결과 |
|---|---|---|
| 1. 얼굴 | (FFHQ 1024px 0~2999번 받아 두기) | `~/Downloads/ffhq-subset/` |
| 2. 이가 보이는 입 | `crop_mouths.py` (MediaPipe) | `mouths/`, `processed.jsonl` |
| 3. 혀·어두운 사진 빼기 | `flag_crops.py` | `mouths_flagged/` |
| 4. v2 자르기(입 안쪽 박스 × 1.36) | `crop_tight.py` | `mouths_tight/` (1,310명) |
| 5. 합성 | `synth.py` | `dataset_v4/` |
| 6. 학습 + ONNX | `train.py` | `runs/<이름>/` |
| 7. 실제 사진 | `eval_onnx.py` | 출력 / CSV |

```bash
python crop_tight.py
python synth.py --variants 4 --src mouths_tight --out dataset_v4
python train.py --data dataset_v4 --size 384 --name v4_384 --init runs/v3_320/best.pt --epochs 10 --patience 3 --lr 1.5e-4 --batch 32
```

`--init`은 이전 학습 결과(best.pt)에서 이어 학습합니다(ImageNet 가중치를 받을 수 없을 때도 씀).
v4는 v2(ImageNet 시작) → v3(`dataset_v3`, 320px, 3장씩) → v4(`dataset_v4`, 384px, 4장씩) 순서로 이어 학습했습니다.
시험용 얼굴(262명)은 v2부터 같고 학습에 쓰지 않았습니다.

## v4에서 바꾼 것 (2026-11-12)

앞니 사이에 낀 고춧가루가 또렷한 웃는 사진을 v2가 "없음"(확률 0.03)으로 봤습니다. 같은 사진에 v2 합성 방식으로 고춧가루를
그려 넣어도 20번 중 0번 알아봐서, 사진이 특이한 게 아니라 모델이 빨간 조각에 약한 것으로 판단했습니다.

- 합성(`synth.py`): 고춧가루를 3배 자주, 크기 0.03~0.08(입 너비 대비)·길쭉함 1~3배로 넓히고 밝은 주황~짙은 빨강 색을 추가.
  35% 사진은 깨끗한 것·낀 것 모두 입술에 립스틱 색을 칠해 "이 근처의 빨강 = 음식"으로 배우지 않게 함.
  저화질(줄였다 키우기), 보정 필터(bilateral), 선명하게 하기를 섞음. 얼굴마다 4장씩.
- 입력 320 → 384px(입 자른 크기 그대로라 작은 조각이 덜 뭉개짐).
- 서버: 한 번만 보면 입 박스가 조금만 달라도 같은 사진이 0.2와 0.97로 갈려서, 여백 1.25·1.36·1.5배 × 좌우 뒤집기 6번의 평균을 씀.

## 결과

| | v2 (이전) | v4 |
|---|---|---|
| 합성 시험 v2 세트 1,048장 정확도 / 낀 것 알아봄 | 0.872 / 0.781 | 0.912 / 0.870 |
| ┗ 고춧가루 / 깨 / 밥알 | 0.73 / 0.72 / 0.68 | 0.86 / 0.81 / 0.82 |
| 합성 시험 v4 세트 2,096장 정확도 / 낀 것 알아봄 | 0.833 / 0.721 | 0.899 / 0.849 |
| 깨끗한 실제 이 262명(서버와 같은 6번 평균): "낀 것 있음" 잘못 알림 | 1.1% (한 번만 봄) | 1.1~1.5% |
| ┗ "확실하지 않아요" | 6.1% | 6.5% |
| 고춧가루 실사진 1장, 입 박스 4가지 + 박스 없음 | 0.03~0.12 (없음) | 0.96~1.0 (있음) |
| 같은 사진에 고춧가루 합성 20번 중 알아봄(한 번만 봄) | 0 | 9 |

깨끗한 이 잘못 알림은 파이썬 평가 1.1%, 서버(Java) 경로 4/262 = 1.5%입니다(크기 줄이는 방법 차이).

## 한계

- 실제로 음식이 낀 사진은 아직 1장으로만 확인했습니다. 팀원이 고춧가루·김·깨를 이에 붙여 찍은 사진 10~20장으로
  `eval_onnx.py --crop-faces --label food`를 돌려 봐야 진짜 인식률을 알 수 있습니다.
- 밥알·깨처럼 이와 색이 비슷한 것은 여전히 약합니다(합성 0.8 안팎).

## v4 이어 학습·LoRA 실험 (2026-11-13~14, 모델 교체 없음)

강사님이 주신 세 가지(이물질 낀 사진으로 시험, v4에 추가 학습, LoRA 붙여 학습)를 해 봤습니다.
결과는 [docs/lab.md](../../docs/lab.md)의 "1-4"에 있고, 결론은 **v4 유지**입니다.
합성 사진 점수는 올랐지만(AUC 0.958 → 최고 0.967) 깨끗한 실제 이 잘못 알림이 1.5% → 1.9~3.1%로 늘었습니다.

스크립트는 [adapt/](adapt/)에 있고, 모델 파일과 사진은 올리지 않았습니다.

| 스크립트 | 하는 일 |
|---|---|
| `adapt/onnx_to_pt.py` | 서버 ONNX(BN 합쳐진 것)에서 torchvision 가중치(best.pt)를 되살림 |
| `adapt/synth_v5.py` | 합성 v5: 깨·밥알·국수 2.5배, 고춧가루 3배 |
| `adapt/synth_v2.py` | 시험용 합성 v2 세트(1,048장)를 다시 만들 때 쓰는 옛 합성 |
| `adapt/train_adapt.py` | `--mode full`(전체 이어 학습) / `--mode lora`(1×1 conv·Linear에 LoRA, 학습 뒤 합쳐서 같은 크기 ONNX). BN은 고정 |
| `adapt/evalset.py`, `adapt/compare.py` | 합성 시험, 깨끗한 실제 이(6번 평균) 평가와 모델 비교 |
| `adapt/eval_gen.py` | AI로 만든 시험 사진 폴더(food/clean/bases) 평가 |
| `adapt/color_rule.py`, `adapt/color_hist.py`, `adapt/hybrid.py` | 색으로 찾기 실험(손 규칙, 색 분포 + GBM, v4와 섞기) |
| `adapt/round2.sh` | 2차 다섯 가지 학습 + 비교를 한 번에(젯슨 GPU에서 판당 약 5분) |

```bash
export TEETH_WORK=~/teeth-work   # mouths_tight/ 가 있는 폴더
bash training/teeth/adapt/round2.sh
```
