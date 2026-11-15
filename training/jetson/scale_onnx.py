"""앙상블용: 서버(OnnxPairModel)는 여러 모델 점수를 그냥 평균하므로, 모델마다 실사진(맨얼굴) 점수 퍼짐(std)으로 나눠 크기를 맞춘 onnx를 만듦.
ens.py가 쓰는 맞춤과 같음. 사용: python3 scale_onnx.py <in.onnx> <R.npy> <out.onnx>   (R.npy = out/jetson/<이름>/R.npy)"""
import sys, numpy as np, onnx
from onnx import helper, numpy_helper
src, rpath, dst = sys.argv[1:4]
R = np.load(rpath); f = 1.0 / float(np.std(R[:, 0]) + 1e-9)
m = onnx.load(src); g = m.graph; out = g.output[0].name
g.node.append(helper.make_node('Mul', [out + '_raw', 'scale_k'], [out]))
for n in g.node[:-1]:
    n.output[:] = [out + '_raw' if o == out else o for o in n.output]
g.initializer.append(numpy_helper.from_array(np.array([f], np.float32), 'scale_k'))
onnx.checker.check_model(m); onnx.save(m, dst)
print(f'{src} → {dst} ×{f:.4f}')
