# Build an ONNX graph of Real-ESRGAN x4plus_anime_6B (basicsr RRDBNet, 6 blocks) from the .pth weights, without torch.
import sys, zipfile, pickle, collections, numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper

def load_pth(path):
    z = zipfile.ZipFile(path)
    root = [n for n in z.namelist() if n.endswith('data.pkl')][0].rsplit('/', 1)[0]
    DT = {'FloatStorage': np.float32, 'HalfStorage': np.float16, 'LongStorage': np.int64, 'IntStorage': np.int32}
    class Storage:
        def __init__(self, dt, key): self.dt, self.key = dt, key
    def rebuild(storage, offset, size, stride, *a):
        buf = np.frombuffer(z.read(f'{root}/data/{storage.key}'), dtype=storage.dt)
        if not size: return buf[offset:offset + 1].reshape(())
        return np.lib.stride_tricks.as_strided(buf[offset:], shape=size, strides=[s * buf.itemsize for s in stride]).copy()
    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if name == '_rebuild_tensor_v2': return rebuild
            if name in DT: return name
            if mod == 'collections' and name == 'OrderedDict': return collections.OrderedDict
            return super().find_class(mod, name)
        def persistent_load(self, pid):
            _, typ, key, *_ = pid
            return Storage(DT[typ if isinstance(typ, str) else typ], key)
    return U(z.open(f'{root}/data.pkl')).load()

sd = load_pth(sys.argv[1])
sd = sd.get('params_ema', sd.get('params', sd))
nodes, inits, n = [], [], [0]
def nm(p): n[0] += 1; return f'{p}{n[0]}'
def conv(x, key):
    w, b = sd[key + '.weight'].astype(np.float32), sd[key + '.bias'].astype(np.float32)
    inits.extend([numpy_helper.from_array(w, key + '.w'), numpy_helper.from_array(b, key + '.b')])
    o = nm('c'); nodes.append(helper.make_node('Conv', [x, key + '.w', key + '.b'], [o], pads=[1, 1, 1, 1], kernel_shape=[3, 3])); return o
def lrelu(x): o = nm('l'); nodes.append(helper.make_node('LeakyRelu', [x], [o], alpha=0.2)); return o
def cat(xs): o = nm('k'); nodes.append(helper.make_node('Concat', xs, [o], axis=1)); return o
def const(v):
    k = nm('s'); inits.append(numpy_helper.from_array(np.array(v, np.float32), k)); return k
def mul(x, v): o = nm('m'); nodes.append(helper.make_node('Mul', [x, const(v)], [o])); return o
def add(a, b): o = nm('a'); nodes.append(helper.make_node('Add', [a, b], [o])); return o
def up2(x):
    o = nm('u'); nodes.append(helper.make_node('Resize', [x, '', const([1, 1, 2, 2])], [o], mode='nearest')); return o
def rdb(x, p):
    x1 = lrelu(conv(x, p + '.conv1')); x2 = lrelu(conv(cat([x, x1]), p + '.conv2')); x3 = lrelu(conv(cat([x, x1, x2]), p + '.conv3'))
    x4 = lrelu(conv(cat([x, x1, x2, x3]), p + '.conv4')); x5 = conv(cat([x, x1, x2, x3, x4]), p + '.conv5')
    return add(mul(x5, 0.2), x)
nb = len({k.split('.')[1] for k in sd if k.startswith('body.')})
feat = conv('input', 'conv_first'); h = feat
for i in range(nb):
    o = rdb(rdb(rdb(h, f'body.{i}.rdb1'), f'body.{i}.rdb2'), f'body.{i}.rdb3'); h = add(mul(o, 0.2), h)
feat = add(feat, conv(h, 'conv_body'))
feat = lrelu(conv(up2(feat), 'conv_up1')); feat = lrelu(conv(up2(feat), 'conv_up2'))
out = conv(lrelu(conv(feat, 'conv_hr')), 'conv_last')
nodes.append(helper.make_node('Identity', [out], ['output']))
g = helper.make_graph(nodes, 'rrdb', [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, None, None])],
                      [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, None, None])], inits)
m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)]); m.ir_version = 8
onnx.checker.check_model(m); onnx.save(m, sys.argv[2]); print('blocks', nb, 'nodes', len(nodes))
