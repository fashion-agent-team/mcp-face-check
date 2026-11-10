"""서버가 쓰는 얼굴 분할 모델(BiSeNet ResNet34, CelebAMask-HQ 19분류)을 받아 서버용 ONNX(fp16)로 만든다.
→ src/main/resources/models/face_parsing_bisenet_r34_fp16.onnx (서버 OnnxFaceParser가 읽음)

출처: yakhyo/face-parsing(MIT) 릴리스의 resnet34.onnx. CelebAMask-HQ로 학습했으므로 가중치는 비상업 연구용 데이터에서 나왔습니다.
받은 ONNX는 그래프(데이터)만 읽고, 끝에 ArgMax → uint8을 붙여 서버 입출력(정규화된 1x3x512x512 → 1x512x512 분류 번호)에 맞춥니다.
분류 번호 순서는 그전 모델(export_face_parsing.py)과 같습니다. 왜 이 모델인지는 docs/face-parsing-models.md.

필요: pip install onnx onnxconverter-common
실행: python3 fetch_face_parsing_r34.py
"""
import hashlib
import pathlib
import urllib.request

URL = "https://github.com/yakhyo/face-parsing/releases/download/weights/resnet34.onnx"
SHA256 = "5b805bba7b5660ab7070b5a381dcf75e5b3e04199f1e9387232a77a00095102e"
CACHE = pathlib.Path(__file__).parent / ".cache/resnet34.onnx"
OUT = pathlib.Path(__file__).parent.parent / "src/main/resources/models/face_parsing_bisenet_r34_fp16.onnx"


def main():
    import onnx
    from onnx import TensorProto, helper
    from onnxconverter_common import float16
    if not CACHE.exists():
        CACHE.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(URL, CACHE)
    if hashlib.sha256(CACHE.read_bytes()).hexdigest() != SHA256:
        raise SystemExit("모델 파일이 예상과 달라요(sha256)")
    m = onnx.load(CACHE)
    g = m.graph
    logits = g.output[0].name  # 1x19x512x512 (나머지 두 출력은 학습용 보조 출력)
    g.node.extend([helper.make_node("ArgMax", [logits], ["class"], axis=1, keepdims=0),
                   helper.make_node("Cast", ["class"], ["labels"], to=TensorProto.UINT8)])
    while len(g.output):
        g.output.pop()
    g.output.append(helper.make_tensor_value_info("labels", TensorProto.UINT8, [1, 512, 512]))
    g.input[0].type.tensor_type.shape.dim[0].dim_value = 1
    onnx.save(float16.convert_float_to_float16(m, keep_io_types=True), OUT)
    print("저장:", OUT, OUT.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
