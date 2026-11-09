"""CelebA(속성 라벨 + 원본 사진 일부)를 받는다. public_baseline.py용.

공식 배포처(Google Drive)가 막힌 곳에서도 받을 수 있게, CelebA를 담은 공개 Docker Hub 이미지(yhjo/celeba-hq)의
레이어를 흘려 읽으면서 필요한 파일만 꺼낸다(레이어 전체는 약 79GB라 내려받아 두지 않음).
받는 것: Anno/(속성·얼굴 상자 등), Img/img_celeba/ 처음 N장(원본 크기). 레이어 앞쪽의 정렬 사진 두 벌(약 13GB)을
지나야 해서 시간이 걸린다(26MB/s에서 약 10분).

CelebA는 비상업 연구용입니다(https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). 사진은 레포에 올리지 않습니다.

실행: python3 fetch_celeba.py <받을 폴더> [--n 15000]
  → <받을 폴더>/home/ubuntu/CelebA/{Anno,Img/img_celeba}
"""
import argparse
import json
import os
import subprocess
import tarfile
import urllib.request

REPO = "yhjo/celeba-hq"
ACCEPT = ",".join(["application/vnd.docker.distribution.manifest.v2+json", "application/vnd.oci.image.manifest.v1+json"])


def token():
    url = f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{REPO}:pull"
    return json.load(urllib.request.urlopen(url))["token"]


def biggest_layer(tok):
    req = urllib.request.Request(f"https://registry-1.docker.io/v2/{REPO}/manifests/latest",
                                 headers={"Authorization": f"Bearer {tok}", "Accept": ACCEPT})
    layers = json.load(urllib.request.urlopen(req))["layers"]
    return max(layers, key=lambda l: l["size"])["digest"]


def main(out, n):
    tok = token()
    url = f"https://registry-1.docker.io/v2/{REPO}/blobs/{biggest_layer(tok)}"
    p = subprocess.Popen(["curl", "-sL", "-H", f"Authorization: Bearer {tok}", url], stdout=subprocess.PIPE)
    got = 0
    try:
        tar = tarfile.open(fileobj=p.stdout, mode="r|gz")
        for m in tar:
            d = os.path.dirname(m.name)
            if not m.isfile():
                continue
            if d.endswith("CelebA/Anno"):
                tar.extract(m, out, filter="data")
            elif d.endswith("Img/img_celeba"):
                tar.extract(m, out, filter="data")
                got += 1
                if got % 1000 == 0:
                    print(got, flush=True)
                if got >= n:
                    break
    finally:
        p.kill()
    print(f"원본 사진 {got}장 →", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=15000)
    a = ap.parse_args()
    main(a.out, a.n)
