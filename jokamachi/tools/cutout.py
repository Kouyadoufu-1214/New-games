import sys, numpy as np, onnxruntime as ort
from PIL import Image
sess = ort.InferenceSession(sys.argv[1], providers=['CPUExecutionProvider'])
inp = sess.get_inputs()[0]
print('input', inp.name, inp.shape)
for name in sys.argv[2:]:
    im = Image.open(f'art/{name}_src.webp').convert('RGB')
    W, H = im.size
    x = np.asarray(im.resize((1024, 1024), Image.LANCZOS)).astype(np.float32)
    x = x / x.max()
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / 1.0
    x = x.transpose(2, 0, 1)[None]
    out = sess.run(None, {inp.name: x})[0][0, 0]
    out = (out - out.min()) / (out.max() - out.min())
    m = Image.fromarray((out * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    m.save(f'art/{name}_mask.png')
    rgba = im.copy(); rgba.putalpha(m); rgba.save(f'art/{name}_cut.png')
    print(name, W, H)
