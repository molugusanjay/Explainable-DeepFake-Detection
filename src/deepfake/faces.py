"""Shared training/inference face preprocessing with OpenCV YuNet."""
from pathlib import Path
import urllib.request
import cv2
import numpy as np
from PIL import Image, ImageOps

YUNET_URL = ('https://media.githubusercontent.com/media/opencv/opencv_zoo/main/'
             'models/face_detection_yunet/face_detection_yunet_2023mar.onnx')


def download_yunet(path='models/assets/face_detection_yunet_2023mar.onnx'):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix('.download')
        urllib.request.urlretrieve(YUNET_URL, temp)
        if temp.stat().st_size < 100000:
            raise RuntimeError('Incomplete YuNet download; please retry')
        temp.replace(path)
    return path


def read_rgb(image):
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    return ImageOps.exif_transpose(image).convert('RGB')


class FaceCropper:
    def __init__(self, weights='models/assets/face_detection_yunet_2023mar.onnx'):
        path = download_yunet(weights)
        self.detector = cv2.FaceDetectorYN.create(str(path), '', (320, 320), .85, .3, 5000)

    def crop(self, image):
        image = read_rgb(image)
        rgb = np.asarray(image)
        h, w = rgb.shape[:2]
        if min(w, h) < 48:
            raise ValueError('Image is too small. Use a clearer, larger facial photograph.')
        scale = min(1., 960/max(w, h))
        small = cv2.resize(rgb, (max(1, round(w*scale)), max(1, round(h*scale))))
        self.detector.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self.detector.detect(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
        if faces is None or len(faces) == 0:
            print('No clear face detected. Analyzing the entire image instead.')
            return image, (0, 0, w, h)
        if len(faces) != 1:
            print('Multiple faces detected. Analyzing the entire image instead.')
            return image, (0, 0, w, h)
        x, y, fw, fh = faces[0, :4] / scale
        if min(fw, fh) < 48:
            raise ValueError('The face is too small for reliable analysis. Use a closer photograph.')
        # Tight common crop: same policy for training and prediction.
        margin = .10
        x0, y0 = max(0, int(x-margin*fw)), max(0, int(y-margin*fh))
        x1, y1 = min(w, int(x+(1+margin)*fw)), min(h, int(y+(1+margin)*fh))
        box = (x0, y0, x1, y1)
        return image.crop(box), box
