# upscale.py model in.png out.png  — Real-ESRGAN x4 anime via onnxruntime (tiles not needed for small crops)
import sys, numpy as np, onnxruntime as ort
from PIL import Image
sess = ort.InferenceSession(sys.argv[1], providers=['CPUExecutionProvider'])
im = np.asarray(Image.open(sys.argv[2]).convert('RGB')).astype(np.float32) / 255
y = sess.run(None, {'input': im.transpose(2, 0, 1)[None]})[0][0]
Image.fromarray((np.clip(y.transpose(1, 2, 0), 0, 1) * 255 + 0.5).astype(np.uint8)).save(sys.argv[3])
