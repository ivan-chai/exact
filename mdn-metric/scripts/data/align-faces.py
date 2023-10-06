import argparse
import os
import shutil

import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop


EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPEG"}


def parse_arguments():
    parser = argparse.ArgumentParser("Scale dataset images")
    parser.add_argument("src", help="Images root")
    parser.add_argument("dst", help="Target root")
    parser.add_argument("--min-size", help="Minimum face size. Keep original image if less.", type=int, default=10)
    return parser.parse_args()


def imread(filename):
    """Read BGR image."""
    with open(filename, "rb") as fp:
        buf = bytearray(fp.read())
        return cv2.imdecode(np.asarray(buf, dtype=np.uint8), cv2.IMREAD_UNCHANGED)


def imwrite(filename, image):
    """Write BGR image."""
    ext = os.path.splitext(filename)[1]
    success, buf = cv2.imencode(ext, image)
    if not success:
        print("Failed encode {}".format(filename))
        import pdb
        pdb.set_trace()
    with open(filename, "wb") as fp:
        fp.write(buf.tobytes())


class Worker:
    def __init__(self, detector, src, dst, min_size):
        self._src = src
        self._dst = dst
        self._detector = detector
        self._min_size = min_size

    def __call__(self, src_path):
        if not os.path.isfile(src_path):
            return
        dst_path = self._dst / src_path.relative_to(self._src)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if src_path.suffix.lower() not in EXTENSIONS:
            shutil.copy2(src_path, dst_path)
            return
        image = imread(src_path)
        faces = self._detector.get(image)
        if not faces:
            print("No faces found for {}. Keep original face.".format(src_path))
            crop = image
        else:
            face = max(faces, key=lambda face: face["det_score"])
            crop = norm_crop(image, face["kps"])
        imwrite(str(dst_path), crop)


def main(args):
    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(exist_ok=True)

    providers = ["CPUExecutionProvider"]
    if torch.cuda.is_available():
        providers.append("CUDAExecutionProvider")
    detector = FaceAnalysis(allowed_modules=["detection"], providers=providers)
    detector.prepare(ctx_id=0, det_size=(640, 640))

    worker = Worker(detector, src, dst, args.min_size)
    src_paths = list(src.rglob("*"))
    for src_path in tqdm(src_paths):
        worker(src_path)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
