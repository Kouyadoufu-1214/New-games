# 絵の下ごしらえ（ヒロインの立ち絵と表情差分）

ゲームには結果だけを埋め込んでいるので、遊ぶのにこのフォルダは要りません。絵を差し替えたり作り直したりするときの手順です。

作業用のフォルダに `art/`（元画像は `art/<名前>_src.webp`）と `diff/`、`models/` を作り、そこで実行します。Python は onnxruntime・Pillow・numpy・scipy・onnx を使います。

1. 立ち絵の切り抜き：`models/isnet-anime.onnx`（rembg のリリースにあるアニメ向けモデル）で `cutout.py`、つづけて `finalize.py`（花びらの除去、縁の色の補正、高さ1650pxにそろえる）
2. 拡大モデル：Real-ESRGAN のリリースにある `RealESRGAN_x4plus_anime_6B.pth` を `esrgan_onnx.py` で ONNX に組み直す（PyTorch は要らない）
3. 表情差分：`tiles.py` で表（`diff/sheet.jpg`）からコマを切り出して4倍に拡大 → `register.py data/regspec.json diff/reg.json` でコマどうしの位置合わせ → `export_faces.py` で立ち絵に合わせた顔を書き出す
   - 立ち絵との対応点（両目・あご・口）は `compose.py` の `LM`、顔を重ねる範囲は `data/shape_<名前>.json`。雪乃の照れと悲しみは眼鏡の四隅も使う（`affine`）
4. 書き出した `final/hero_face_b64.js` の中身を `index.html` の `HERO_FACE` と差し替える
