# 눈화장 모델 젯슨 GPU 학습

클라우드 CPU(4코어)에서는 16에폭에 약 11시간 걸려서, 학습을 젯슨 Orin GPU로 옮겼어요. 데이터는 맥에서 만들어요(mediapipe 필요).

1. **맥: 데이터 만들기** (1~2시간)
   `bash training/jetson/make_data.sh <CelebA 폴더> <FFHQ 사진 폴더>`
   - CelebA는 원본(Anno/list_attr_celeba.txt, list_bbox_celeba.txt, Img/img_celeba) 형식이어야 해요. 없으면 `python3 training/fetch_celeba.py <폴더> --n 30000`을 쓰세요.
   - FFHQ는 정렬 순서로 앞 600장을 평가에 쓰고, 나머지를 학습에 써요.
   - 결과: `training/out/pack/`(약 7GB). a/b/y.npy는 학습용이고, eval_p.npy와 eval_meta.npz는 평가 묶음이에요.
2. **젯슨에 복사**
   `rsync -a training/out/pack/ <젯슨>:<레포>/training/out/pack/`
3. **젯슨: 반복 학습** (멈추라고 할 때까지)
   `nohup bash training/jetson/loop.sh > training/out/jetson/loop.log 2>&1 &`
   - 처음에 지금 서버 모델을 평가해요. 서버식으로 거르면 약 71%가 나와야 해요.
   - 그다음 같은 설정의 긴 학습, 넓은 망, 다른 씨앗을 차례로 돌려요.
   - 결과는 `out/jetson/results.jsonl`과 `out/jetson/best.txt`에 쌓여요. best.txt에는 하나씩 돌린 성적과 앙상블 성적이 들어 있어요.
   - 멈추려면 `touch training/out/jetson/STOP`를 실행하세요.
   - 에폭마다 `ck.pt`를 저장해요. 그래서 끊겨도 같은 명령으로 다시 시작하면 이어서 돌아요.
   - 젯슨 GPU를 다른 스레드와 같이 써요. 다른 학습을 돌릴 땐 `~/gpu-train.lock`에 이름을 적고 끝나면 지우세요. 그동안 이 학습은 멈췄다가 잠금이 풀리면 이어서 돌아요.
   - 젯슨은 서버(face-check 등)와 메모리 8GB를 같이 써요. 남은 메모리가 1GB 아래로 떨어지면 학습이 스스로 멈추고, 10분 뒤 다시 시작해요. 학습 데이터는 mmap으로 조금씩 읽어요.

평가 방법은 서버와 같아요. 안경이 눈 조각의 2%를 넘거나 눈 밑 밝기 차이가 15를 넘는 사진은 빼요. 남은 사진의 맨얼굴 점수에서 상위 5%(p95)를 기준선으로 잡고, 한쪽에만 섀도(.3/.5/.7)나 아이라인(.5/.8)을 넣었을 때 기준선을 넘는 비율을 재요.

더 나은 모델이 나오면 이렇게 반영해요.
1. `eye_<이름>.onnx`를 `src/main/resources/models/eye_makeup.onnx`로 바꿔요.
2. `training/eye_server_eval.py`로 서버 기준값(eye-makeup-model)을 다시 재요.
3. 결과를 보여 주고 승인을 받은 뒤 PR을 올려요.

CelebA와 FFHQ는 비상업용이에요. 사진과 만든 파일은 레포에 올리지 마세요(`training/out/`는 .gitignore에 들어 있어요).
