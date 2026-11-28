# Cut out NPC busts: isnet-anime mask -> clean -> decontaminate edges -> crop -> face detect
import sys, os, json, glob, numpy as np, onnxruntime as ort, cv2
from PIL import Image
from scipy import ndimage as ndi
B = '/tmp/claude-0/-home-user-New-games/e10e8c9b-88e9-5bdd-8465-a555008c7672/scratchpad/'
sess = ort.InferenceSession(B + 'models/isnet-anime.onnx', providers=['CPUExecutionProvider'])
casc = cv2.CascadeClassifier(B + 'models/lbpcascade_animeface.xml')
def mask_of(im):
    x = np.asarray(im.resize((1024, 1024), Image.LANCZOS)).astype(np.float32) / 255
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)).transpose(2, 0, 1)[None]
    o = sess.run(None, {'img': x})[0][0, 0]; o = (o - o.min()) / (o.max() - o.min() + 1e-9)
    return np.asarray(Image.fromarray((o * 255).astype(np.uint8)).resize(im.size, Image.BILINEAR)).astype(np.float32) / 255
def cut(src, out, maxdim=600, keep_frac=0.02):
    im = Image.open(src).convert('RGB'); C = np.asarray(im).astype(np.float32) / 255
    a = mask_of(im)
    lab, n = ndi.label(a > 0.3)
    if n:
        sz = ndi.sum(np.ones_like(a), lab, range(1, n + 1)); keep = np.zeros(n + 1, bool); keep[1:] = sz > sz.max() * keep_frac
        a = np.where(ndi.binary_dilation(keep[lab], iterations=5), a, 0)
    a = np.clip((a - 0.06) / 0.88, 0, 1)
    bg = a < 0.02; glob_bg = np.median(C[bg], axis=0) if bg.sum() > 100 else np.array([1, 1, 1], np.float32)
    Bk = np.zeros_like(C)
    for y in range(C.shape[0]):
        m = bg[y]; Bk[y] = np.median(C[y][m], axis=0) if m.sum() > 30 else glob_bg
    am = a[..., None]
    F = np.where(am > 0.03, np.clip((C - (1 - am) * Bk) / np.maximum(am, 1e-3), 0, 1), C); F = np.where(am > 0.97, C, F)
    ys, xs = np.where(a > 0.04); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    pad = 6; y0 = max(0, y0 - pad); x0 = max(0, x0 - pad); y1 = min(a.shape[0], y1 + pad); x1 = min(a.shape[1], x1 + pad)
    rgba = (np.dstack([F, a]) * 255 + 0.5).astype(np.uint8)
    o = Image.fromarray(rgba, 'RGBA').crop((x0, y0, x1, y1))
    s = min(1, maxdim / max(o.size)); o = o.resize((round(o.width * s), round(o.height * s)), Image.LANCZOS) if s < 1 else o
    o.save(out)
    # face detection on the crop (grey, equalised)
    g = cv2.cvtColor(np.asarray(o.convert('RGB')), cv2.COLOR_RGB2GRAY); g = cv2.equalizeHist(g)
    fs = casc.detectMultiScale(g, scaleFactor=1.05, minNeighbors=4, minSize=(int(o.width * 0.15), int(o.width * 0.15)))
    faces = sorted([list(map(int, f)) for f in fs], key=lambda f: -f[2])
    return {'w': o.width, 'h': o.height, 'faces': faces[:3], 'cutBottom': bool(y1 >= a.shape[0] - 1)}
if __name__ == '__main__':
    jobs = json.loads(sys.argv[1]); res = {}
    for key, src in jobs.items():
        res[key] = cut(src, B + f'npc/{key}.png', maxdim=int(os.environ.get('MAXDIM', 600))); print(key, res[key], flush=True)
    old = json.load(open(B + 'npc/info.json')) if os.path.exists(B + 'npc/info.json') else {}
    old.update(res); json.dump(old, open(B + 'npc/info.json', 'w'), indent=1)
