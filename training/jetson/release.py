"""서버 반영 묶음 만들기: 모델마다 점수 크기를 맞춘 onnx(scale_onnx.py와 같음) + 서버 기준값(앙상블 실사진 p95) + 설정 한 줄.
고르는 규칙(같은 평가 사진에서 조합을 뒤지면 점수가 부풀려지므로 미리 정한 규칙으로): 지금 서버 모델 + 하나씩 점수 상위 N개.
사용: python3 release.py --top 3            (server + 상위 3개)
      python3 release.py r18s41e5 r18s13e4 server   (직접 지정)
결과: out/jetson/release/ (eye_makeup.onnx, eye_makeup_2.onnx …, release.txt) → 레포 src/main/resources/models/로 복사
주의: server는 레포의 지금 eye_makeup.onnx를 씀. 묶음을 레포에 복사한 뒤 다시 돌리면 크기 맞춤이 두 번 들어가니, 원래 파일(git)로 되돌린 뒤 돌리세요."""
import sys, json, pathlib, shutil, subprocess, numpy as np
HERE = pathlib.Path(__file__).resolve().parent; TR = HERE.parent; J = TR / 'out/jetson'; PACK = TR / 'out/pack'
M = np.load(PACK / 'eval_meta.npz'); SG = M['sg']; g = (M['glasses'] <= 0.02) & (M['light'] <= 15)
K = ['shadow.3', 'shadow.5', 'shadow.7', 'liner.5', 'liner.8']


def score(R):
    Rm = R.mean(2); th = float(np.percentile(np.abs(Rm[g, 0]), 95))
    d = {k: round(float(np.mean(Rm[g, i + 1] * SG[g] > th)), 3) for i, k in enumerate(K)}
    return th, round(float(np.mean(list(d.values()))), 3), d


Rs = {p.parent.name: np.load(p) for p in J.glob('*/R.npy')}
if sys.argv[1:2] == ['--top']:
    n = int(sys.argv[2]); single = sorted((score(R)[1], k) for k, R in Rs.items() if k != 'server')[::-1]
    names = ['server'] + [k for _, k in single[:n]]
else:
    names = sys.argv[1:]
names = ['server'] + [k for k in names if k != 'server'] if 'server' in names else names
OUT = J / 'release'; shutil.rmtree(OUT, ignore_errors=True); OUT.mkdir()
files, scaled = [], []
for i, k in enumerate(names):
    src = TR.parent / 'src/main/resources/models/eye_makeup.onnx' if k == 'server' else J / k / f'eye_{k}.onnx'
    dst = OUT / ('eye_makeup.onnx' if i == 0 else f'eye_makeup_{i + 1}.onnx')
    subprocess.run([sys.executable, str(HERE / 'scale_onnx.py'), str(src), str(J / k / 'R.npy'), str(dst)], check=True)
    files.append(dst.name); scaled.append(Rs[k] / (np.std(Rs[k][:, 0]) + 1e-9))
th, allv, d = score(np.mean(scaled, 0))
_, sv, sd = score(Rs['server']) if 'server' in Rs else (0, 0, {})
txt = f"""모델: {' + '.join(names)}
실사진(거른 {int(g.sum())}장, 잘못 알림 5%): 전체 {allv:.3f} {json.dumps(d)}
지금 서버 모델 혼자: {sv:.3f} {json.dumps(sd)}
서버 기준값 face.symmetry.eye-makeup-model: {th:.3f}   (맞춘 점수 단위. 서버에 올린 뒤 eye_server_eval.py로 다시 확인)
설정 face.eye-makeup.model: {','.join('classpath:models/' + f for f in files)}
"""
(OUT / 'release.txt').write_text(txt); print(txt)
