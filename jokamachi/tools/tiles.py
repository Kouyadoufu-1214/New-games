# cut each expression tile out of the sheet and upscale it x4 with Real-ESRGAN anime
import json, numpy as np, onnxruntime as ort
from PIL import Image
sheet = Image.open('diff/sheet.jpg').convert('RGB')
C3 = [(13, 278), (297, 561), (580, 843)]
L = {
  'koharu': ([(81, 346), (363, 628)], C3, ['normal', 'happy', 'blush', 'angry', 'sad', 'surprise']),
  'yukino': ([(757, 992), (1038, 1272)], C3, ['normal', 'happy', 'blush', 'angry', 'sad', 'surprise']),
  'chizuru': ([(1438, 1700), (1717, 1976)], [(17, 278), (297, 561), (580, 839)], ['normal', 'happy', 'blush', 'angry', 'sad', 'surprise']),
  'kaede': ([(2098, 2302), (2313, 2517)], [(6, 209), (220, 424), (435, 639), (649, 852)], ['m_normal', 'm_happy', 'm_blush', 'm_angry', 'm_sad', 'm_surprise', 'u_normal', 'u_happy']),
}
sess = ort.InferenceSession('models/anime6b.onnx', providers=['CPUExecutionProvider'])
boxes = {}
for k, (rows, cols, names) in L.items():
    i = 0
    for (y0, y1) in rows:
        for (x0, x1) in cols:
            t = sheet.crop((x0, y0, x1, y1)); n = names[i]; i += 1
            t.save(f'diff/tiles/{k}_{n}.png'); boxes[f'{k}_{n}'] = [x0, y0, x1, y1]
            a = np.asarray(t).astype(np.float32) / 255
            y = sess.run(None, {'input': a.transpose(2, 0, 1)[None]})[0][0]
            Image.fromarray((np.clip(y.transpose(1, 2, 0), 0, 1) * 255 + 0.5).astype(np.uint8)).save(f'diff/tiles/{k}_{n}_x4.png')
            print(k, n, t.size)
json.dump(boxes, open('diff/tiles/boxes.json', 'w'))
