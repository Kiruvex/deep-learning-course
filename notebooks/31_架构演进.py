import marimo

__generated_with = "0.25.2"
app = marimo.App(app_title="31 · 架构演进", width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import numpy as np
    return (np,)


@app.cell
def _():
    import matplotlib.pyplot as plt

    # ==== 中文乱码修复（每课开头的固定仪式，一次设置全程生效）====
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Microsoft YaHei", "SimHei",
        "Noto Sans SC", "WenQuanYi Zen Hei",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # 31 · 架构演进：从 LeNet 到 ResNet

        **从零到 Transformer · 手写深度学习** ｜ 第 5 部分 · 卷积神经网络（6/6 · 收官）
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📖 本课目标（先读我）": mo.md(
                """
                **最后一课不讲新零件，讲历史——以及历史里最重要的
                一个主意。**

                1998 年你刚训练过的 LeNet-5 已经用上了卷积、池化、
                反向传播。可深度学习的大爆发偏偏拖到 **2012 年**。
                为什么？爆发的引线是什么？爆发之后，AlexNet →
                VGG → GoogLeNet → ResNet 四年四代，每一代各自
                回答了什么问题？

                这一课有**两场动手实验**（不是光读历史）：

                - **深度退化 vs 残差救援**：亲手复现 ResNet 论文
                  里最著名的现象——普通网络越深越差（33 层第 1
                  轮只有 57%），残差连接让 33 层**比 5 层还好**
                - **两个 3×3 顶一个 5×5**：把 VGG 的哲学用最小
                  二乘算个明白——线性版严格等价（对账 True），
                  中间插一个 ReLU 后 5×5 就模仿不动了（最优拟合
                  误差 92%）

                学完你将能：

                - 讲清 2012 大爆发的**三件套**：ImageNet 大数据、
                  GPU 算力、ReLU + Dropout 算法创新——缺一不可
                - 复述四代架构的核心思想各一句话，并说出它们的
                  ImageNet top-5 错误率（15.3% → 7.3% / 6.7% →
                  3.57%）
                - **手写残差块的前向与反向**，解释跳跃连接为什么
                  是"梯度的高速公路"
                - 把第 5 部分六课的能力装进行囊，准备进第 6
                  部分（序列的世界）

                前置：30 课 LeNet-5（实验底盘）、28 课感受野
                （VGG 的账要用它算）、25 课 He 初始化（退化实验
                的初始化）。

                这是第 5 部分的收官课——轻松一点，代码不多，
                但两场实验都值得亲手跑。
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "🌋 概念 1/4 · 大爆发前夜：为什么是 2012？": mo.md(
                """
                LeNet-5 之后，CNN 沉寂了整整十四年。不是思想停滞
                ——卷积、池化、反向传播全都就位——而是**三样东西
                凑不齐**：

                | 要素 | 1998 年 | 2012 年 |
                |---|---|---|
                | 数据 | 手写数字（6 万张） | **ImageNet**：1000 类、128 万张 |
                | 算力 | CPU | **GPU**：两个 GTX 580 训了五六天 |
                | 算法 | sigmoid 易饱和 | **ReLU**（19 课的老朋友）+ Dropout（25 课） |

                2009 年 ImageNet 建成，2012 年 AlexNet 一战封神：
                top-5 错误率 **15.3%**，比亚军（传统特征工程方法，
                26.1%）低了近 11 个百分点——深度学习不是"又一种
                方法"，是**换了一个时代**。

                > top-5 错误率：模型猜的前 5 个类别全都不对的
                > 概率。1000 类里猜 5 次还不对，比想象的难。

                AlexNet 的架构其实和 LeNet 同宗（卷积→池化→
                卷积→池化→全连接），只是大得多：8 层、6000 万
                参数。它的真正遗产是把"三件套"焊在了一起：
                **从此所有人都用大数据 + GPU + ReLU 训深度网络**。
                """
            ),
            "🧱 概念 2/4 · 两个流派：VGG 的深 与 GoogLeNet 的巧": mo.md(
                """
                2014 年，两个流派同时给出答卷：

                **VGG：大力出奇迹。** 全部用 3×3 小核（27 课说过
                same 配方最爱奇数小核），一层层堆到 16/19 层。
                核心算术：**两个 3×3 的感受野 = 一个 5×5**（
                3 + 2×2 = 5，28 课的递推），但参数 2×9 = 18 < 25，
                还多插一个 ReLU。演示 2 会把这笔账算到底：线性
                情况下两个 3×3 **严格等价**于某个 5×5（数值对账
                True），中间插了 ReLU 之后 5×5 就模仿不动了
                （最优拟合误差 92%）。代价：VGG-16 有 **1.38 亿**
                参数——LeNet 的 2000 多倍。

                **GoogLeNet：多尺度并联。** Inception 模块让
                1×1、3×3、5×5 卷积和池化**并行**跑一遍再拼起来
                ——"不知道该看多大，就都看"。配套发明了 **1×1
                卷积当降维阀**：先用 1×1 把通道压扁再做 5×5，
                参数能砍一个数量级。22 层只有 680 万参数，比
                AlexNet 少 89%，成绩反而更好（top-5 6.7% vs
                15.3%）。

                一句话记住两家：**VGG 证明了"深"值钱，GoogLeNet
                证明了"深"不一定要贵。**
                """
            ),
            "🛣️ 概念 3/4 · ResNet：退化问题与梯度的高速公路": mo.md(
                """
                堆到 20+ 层，人们撞上一堵诡异的墙——**退化问题**
                （degradation）：56 层的**训练误差**比 20 层还高。
                注意：不是过拟合（验证误差高），是**连训练集都
                拟合不好**——纯优化难题。深网络理论上应该至少
                不输浅网络（多出的层学成恒等映射就行），可 SGD
                就是学不动"恒等映射"这四个字。

                ResNet（2015）的答案简单到离谱——**给梯度修一条
               高速公路**：

                ```
                普通块：  y = F(x)            # 想让 y=x，得让 F 学出恒等
                残差块：  y = x + F(x)        # 想让 y=x，让 F=0 就行（容易得多）
                ```

                反向传播时，梯度沿跳跃线**原样回流**：∂y/∂x = 1 +
                ∂F/∂x——那个 **+1** 保证梯度永不消失，无论多少层。
                演示 1 里你会亲眼看到：普通 33 层第 1 轮只有 57%
                训练准确率、爬到 94.7% 就到头；残差 33 层第 1 轮
                就 91%、终盘 99.4%——**比 5 层普通网络还好**。

                两个工程细节（我们的实验会用上）：

                - 真实 ResNet 每层配 **BatchNorm**（批归一化）稳住
                  激活值的分布；我们的 NumPy 版用 **α 缩放**
                  （残差分支乘 0.1，ReZero/Fixup 一族的做法）——
                  作用相同：防止方差逐块累加，实现只要一行
                - 残差块的维度必须一致（x 和 F(x) 才能相加）；
                  维度变化处（如池化后）用 1×1 卷积做投影适配

                战绩：ResNet 把 ImageNet top-5 干到 **3.57%**，
                首次超过人类水平（约 5.1%），深度直上 **152 层**。
                跳跃连接从此成为深度学习的通用零件——Transformer
                （第 7 部分）里也到处是它。
                """
            ),
            "🔮 概念 4/4 · 之后的世界：轻量化与注意力": mo.md(
                """
                ResNet 之后，卷积网络的故事进入下半场：

                - **轻量化**（2017~）：MobileNet 的**深度可分离
                  卷积**——把"C_in×K×K 的大核"拆成"每个通道
                  各自 K×K + 一次 1×1 合通道"，参数和计算量砍
                  近一个数量级。手机上跑视觉，靠的就是这类手艺
                - **注意力进军视觉**（2020）：**ViT**（Vision
                  Transformer）把图像切成 16×16 的小方块当
                  "词"处理，用纯注意力（不用卷积）打平甚至
                  超越 CNN——论文标题就叫《An Image is Worth
                  16×16 Words》。你在这部分学的卷积没有白学：
                  ViT 的"图像切块"正是**局部窗口**思想的延续，
                  只是换了一副注意力的心脏
                - **融合**：ConvNeXt（2022）用现代配方重做卷积
                  网络，再次打平 ViT——两条路线至今并驾齐驱

                编年史一张表记住（数字来自各论文与 ILSVRC 官方
                战绩）：

                | 年份 | 模型 | 层数 | 参数 | top-5 错误率 |
                |---|---|---|---|---|
                | 1998 | LeNet-5 | 5 | 0.06M | —（MNIST 时代） |
                | 2012 | AlexNet | 8 | 60M | 15.3% |
                | 2014 | VGG-19 | 19 | 144M | 7.3% |
                | 2014 | GoogLeNet | 22 | 6.8M | 6.7% |
                | 2015 | ResNet-152 | 152 | 60M | **3.57%** |
                | — | 人类 | — | — | ≈5.1% |

                规律藏在参数列：**错误率一路降，参数量却不是
                一路涨**——GoogLeNet（6.8M）比 AlexNet（60M）
                准且小，ResNet（60M）靠"深"而不是"肥"登顶。
                架构设计，本质是**参数的经济学**。
                """
            ),
        }
    )
    return


@app.cell
def _(mo, np):
    # MNIST 加载器（与 25~30 课同款）
    def load_mnist():
        """返回 (X_tr, y_tr, X_te, y_te)，像素归一化 [0,1]，X 形状 (n, 784)。"""
        import os, gzip, urllib.request
        from pathlib import Path
        _files = {"train-images": ("train-images-idx3-ubyte.gz", 9912422),
                  "train-labels": ("train-labels-idx1-ubyte.gz", 28881),
                  "t10k-images": ("t10k-images-idx3-ubyte.gz", 1648877),
                  "t10k-labels": ("t10k-labels-idx1-ubyte.gz", 4542)}
        _mirrors = ["https://ossci-datasets.s3.amazonaws.com/mnist",
                    "https://storage.googleapis.com/cvdf-datasets/mnist"]
        _cands = []
        try:
            _nb = mo.notebook_location()
            if _nb is not None:
                _cands += [Path(_nb).parent.parent / "data", Path(_nb).parent / "data"]
        except Exception:
            pass
        _cands += [Path("../data"), Path("data")]
        _dir = next((_d for _d in _cands if _d.is_dir()), None)
        if _dir is None:
            for _d in _cands:
                try:
                    _d.mkdir(parents=True, exist_ok=True)
                    _dir = _d
                    break
                except OSError:
                    continue
            if _dir is None:
                raise RuntimeError("找不到可写的数据目录：请看 25 课的手动下载说明")
        _cache = _dir / "mnist.npz"
        if _cache.exists():
            with np.load(_cache) as _z:
                return _z["X_tr"], _z["y_tr"], _z["X_te"], _z["y_te"]
        for _key, (_fname, _size) in _files.items():
            _path = _dir / _fname
            if _path.exists() and _path.stat().st_size == _size:
                continue
            _ok = False
            for _m in _mirrors:
                try:
                    print(f"下载 {_fname}（{_size/1e6:.1f} MB）……")
                    urllib.request.urlretrieve(f"{_m}/{_fname}", _path)
                    if _path.stat().st_size == _size:
                        _ok = True
                        break
                except Exception:
                    continue
            if not _ok:
                raise RuntimeError(
                    f"MNIST 下载失败：{_fname}。请看 25 课的手动下载说明，"
                    "把 4 个文件放进 data/ 目录后重新运行。")
        def _images(_path):
            with gzip.open(_path, "rb") as _f:
                _raw = _f.read()
            _magic, _n, _r, _c = np.frombuffer(_raw[:16], dtype=">i4")
            assert _magic == 2051, "图片文件魔数不对"
            return np.frombuffer(_raw[16:], dtype=np.uint8).reshape(_n, _r * _c).astype(np.float32) / 255.0
        def _labels(_path):
            with gzip.open(_path, "rb") as _f:
                _raw = _f.read()
            _magic, _n = np.frombuffer(_raw[:8], dtype=">i4")
            assert _magic == 2049, "标签文件魔数不对"
            return np.frombuffer(_raw[8:], dtype=np.uint8).astype(np.int64)
        X_tr, y_tr = _images(_dir / _files["train-images"][0]), \
                     _labels(_dir / _files["train-labels"][0])
        X_te, y_te = _images(_dir / _files["t10k-images"][0]), \
                     _labels(_dir / _files["t10k-labels"][0])
        np.savez_compressed(_cache, X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te)
        return X_tr, y_tr, X_te, y_te

    X_mnist, y_mnist, X_te31, y_te31 = load_mnist()
    X_sub31, y_sub31 = X_mnist[:10000], y_mnist[:10000]        # 迷你战场：1 万张（17 课的三分天下迷你版）
    X_va31, y_va31 = X_mnist[10000:12000], y_mnist[10000:12000]
    print("迷你战场：训练", X_sub31.shape, " 验证", X_va31.shape)
    return (X_mnist, X_sub31, X_va31, y_mnist, y_sub31, y_te31, y_va31)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 🎬 演示
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 1 · 深度退化 vs 残差救援：五台网络同场竞技

        同样的数据（MNIST 前 1 万张）、同样的预算（Adam、
        batch 128、10 个 epoch）、同样的宽度 64——只有两个
        变量：**深度**和**有没有跳跃连接**。

        写作时实测的战报（你的数字应可复现，全程约 40 秒）：

        | 网络 | 第 1 轮 train | 终盘 train | 终盘 val |
        |---|---|---|---|
        | 普通 5 层 | 91.0% | 99.05% | 94.05% |
        | 普通 21 层 | 71.3% | 97.52% | 93.55% |
        | 普通 33 层 | 56.5% | **94.74%** | 92.15% |
        | 残差 21 层 | 91.3% | 99.34% | 95.05% |
        | 残差 33 层 | **91.1%** | **99.38%** | **95.05%** |

        三条铁律浮出水面：

        - **普通网络越深越差**：第 1 轮 91% → 71% → 57%，
          终盘 99.1% → 94.7%——这正是 ResNet 论文里的退化
          问题（连训练误差都更高，这不是过拟合）
        - **残差网络的第 1 轮和 5 层一样快**：91.1% vs 91.0%
          ——跳跃线让梯度瞬间直达每一层，深网络"起步即跑车"
        - **残差 33 层 > 普通 5 层**：深度从负担变成了纯红利
        """
    )
    return


@app.cell
def _(X_sub31, X_va31, mo, np, plt, y_sub31, y_va31):
    # 演示 1 · 深度退化 vs 残差救援（普通 5/21/33 层 + 残差 21/33 层）
    import time as _time

    def build_mlp31(depth, width, seed=0):
        """He 初始化的 MLP：depth 个隐藏层 + 输出层。"""
        _r = np.random.default_rng(seed)
        _dims = [784] + [width] * depth + [10]
        _Ws = [_r.normal(0, 1, (_dims[i], _dims[i+1])) * np.sqrt(2 / _dims[i])
               for i in range(depth + 1)]
        _bs = [np.zeros(_d) for _d in _dims[1:]]
        return _Ws, _bs

    def forward31(X, Ws, bs, residual):
        """前向。残差版：第 1 层 plain，其后每 2 层组成残差块。
        块：out = x + 0.1 · (W2·ReLU(W1·x+b1)+b2)——α 缩放防方差累加（概念 3）。"""
        _depth = len(Ws) - 1
        _cache = []
        _h = X
        _i = 0
        while _i < _depth:
            if residual and _i > 0:
                _h_in = _h
                _z1 = _h @ Ws[_i] + bs[_i]
                _a1 = np.maximum(_z1, 0)
                _z2 = _a1 @ Ws[_i+1] + bs[_i+1]
                _out = _h_in + 0.1 * _z2                       # 残差块：跳跃 + α 缩放分支
                _cache.append(("r", _i, _h_in, _z1, _a1, _out))
                _h = _out
                _i += 2
            else:
                _z = _h @ Ws[_i] + bs[_i]
                _out = np.maximum(_z, 0)
                _cache.append(("p", _i, _z, _out))
                _h = _out
                _i += 1
        return _h @ Ws[_depth] + bs[_depth], _cache

    def backward31(dlogits, Ws, cache, X):
        """反向。残差块的关键行：dh = dz1 @ W1.T + dh——跳跃线让梯度原样回流。"""
        _depth = len(Ws) - 1
        _gW = [np.zeros_like(_w) for _w in Ws]
        _gb = [np.zeros(_w.shape[1]) for _w in Ws]
        _gW[_depth] = cache[-1][5].T @ dlogits if cache[-1][0] == "r" else cache[-1][3].T @ dlogits
        _gb[_depth] = dlogits.sum(0)
        _dh = dlogits @ Ws[_depth].T
        for _e in reversed(cache):
            if _e[0] == "r":
                _, _i, _h_in, _z1, _a1, _out = _e
                _dz2 = _dh * 0.1                              # α 缩放的反向
                _gW[_i+1] = _a1.T @ _dz2
                _gb[_i+1] = _dz2.sum(0)
                _da1 = _dz2 @ Ws[_i+1].T
                _dz1 = _da1 * (_z1 > 0)
                _gW[_i] = _h_in.T @ _dz1
                _gb[_i] = _dz1.sum(0)
                _dh = _dz1 @ Ws[_i].T + _dh                   # 梯度高速公路：+dh，原样回流
            else:
                _, _i, _z, _out = _e
                _dz = _dh * (_z > 0)
                _h_prev = X if _i == 0 else None
                if _h_prev is None:
                    for _f in reversed(cache[:cache.index(_e)]):
                        _h_prev = _f[5] if _f[0] == "r" else _f[3]
                        break
                _gW[_i] = _h_prev.T @ _dz
                _gb[_i] = _dz.sum(0)
                _dh = _dz @ Ws[_i].T
        return _gW, _gb

    def train31(depth, residual, width=64, epochs=10, lr=1e-3, seed=0):
        """Adam 训练（24 课），返回每轮 (train_acc, val_acc)。"""
        Ws31, bs31 = build_mlp31(depth, width, seed)
        _mW = [np.zeros_like(_w) for _w in Ws31]
        _mB = [np.zeros(_w.shape[1]) for _w in Ws31]
        _sW = [np.zeros_like(_w) for _w in Ws31]
        _sB = [np.zeros(_w.shape[1]) for _w in Ws31]
        _r = np.random.default_rng(seed + 1)
        _step, _hist = 0, []
        for _ep in range(epochs):
            _order = _r.permutation(len(X_sub31))
            for _st in range(0, len(X_sub31), 128):
                _ib = _order[_st:_st + 128]
                _xb, _yb = X_sub31[_ib], y_sub31[_ib]
                _logits, _cache = forward31(_xb, Ws31, bs31, residual)
                _e = np.exp(_logits - _logits.max(1, keepdims=True))
                _P = _e / _e.sum(1, keepdims=True)
                _n = len(_yb)
                _dl = _P.copy()
                _dl[np.arange(_n), _yb] -= 1
                _dl /= _n
                _gW, _gb = backward31(_dl, Ws31, _cache, _xb)
                _step += 1
                for _k in range(len(Ws31)):
                    _mW[_k] = 0.9 * _mW[_k] + 0.1 * _gW[_k]
                    _sW[_k] = 0.999 * _sW[_k] + 0.001 * _gW[_k] ** 2
                    Ws31[_k] -= lr * (_mW[_k] / (1 - 0.9 ** _step)) / \
                        (np.sqrt(_sW[_k] / (1 - 0.999 ** _step)) + 1e-8)
                    _mB[_k] = 0.9 * _mB[_k] + 0.1 * _gb[_k]
                    _sB[_k] = 0.999 * _sB[_k] + 0.001 * _gb[_k] ** 2
                    bs31[_k] -= lr * (_mB[_k] / (1 - 0.9 ** _step)) / \
                        (np.sqrt(_sB[_k] / (1 - 0.999 ** _step)) + 1e-8)
            _lg, _ = forward31(X_sub31, Ws31, bs31, residual)
            _lgv, _ = forward31(X_va31, Ws31, bs31, residual)
            _hist.append((float(np.mean(np.argmax(_lg, 1) == y_sub31)),
                          float(np.mean(np.argmax(_lgv, 1) == y_va31))))
        return _hist

    _t0 = _time.time()
    _configs = [(5, False), (21, False), (33, False), (21, True), (33, True)]
    _curves = {}
    for _d, _res in _configs:
        _curves[( _d, _res)] = train31(_d, _res)
        _h = _curves[(_d, _res)]
        print(f"  {'残差' if _res else '普通'}-{_d}层：第1轮 train {_h[0][0]:.3f}"
              f"｜终盘 train {_h[-1][0]:.4f}｜val {_h[-1][1]:.4f}")
    print(f"五台网络全部跑完：{_time.time()-_t0:.0f}s")

    _fig, _ax = plt.subplots(figsize=(9.4, 5.2))
    _styles = {(5, False): ("tab:blue", "-"), (21, False): ("tab:blue", "--"),
               (33, False): ("tab:blue", ":"), (21, True): ("tab:red", "--"),
               (33, True): ("tab:red", "-")}
    for (_d, _res), _h in _curves.items():
        _c, _ls = _styles[(_d, _res)]
        _ax.plot(range(1, 11), [t for t, v in _h], _ls, color=_c, lw=2.2,
                 marker="o", ms=4, label=f"{'残差' if _res else '普通'}-{_d}层")
    _ax.set_xlabel("epoch")
    _ax.set_ylabel("训练准确率（迷你战场 1 万张）")
    _ax.set_ylim(0.5, 1.01)
    _ax.grid(True, alpha=0.3)
    _ax.legend(fontsize=10, loc="lower right")
    _ax.set_title("退化与救赎：普通网络越深越差（蓝），残差让深度变成红利（红）", fontsize=11.5)
    _fig.tight_layout()
    _fig
    return (backward31, build_mlp31, forward31, train31)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 2 · 两个 3×3 顶一个 5×5：VGG 哲学的数学账

        三步把这笔账算死：

        1. **线性情况**：随机取两个 3×3 核，堆叠卷积的输出
           与某个 5×5 核（复合核 K_eff）的输出**逐元素相同**
           （对账 True）——两个 3×3 能表达的线性变换是 5×5
           的子集，参数还少（18 vs 25）
        2. **插入 ReLU**：中间加一个 ReLU 后，用最小二乘找
           **最优的 5×5 核**去拟合堆叠输出——相对误差 **92%**
           （线性版是 0）。多出来的那个 ReLU 是 5×5 永远模仿
           不了的
        3. **整网账本**：4 站 conv5 与 8 站 conv3 的感受野都是
           **17**，参数 100 vs 72（省 28%），ReLU 4 个 vs
           **8 个**——同样的视野，更少的参数、双倍的非线性
        """
    )
    return


@app.cell
def _(np, plt):
    # 演示 2 · 两个 3×3 的数学账
    from numpy.lib.stride_tricks import sliding_window_view

    def conv2_31(x, K):
        _win = sliding_window_view(x, K.shape)
        return np.einsum("ijkl,kl->ij", _win, K)

    _rng = np.random.default_rng(0)
    K1_31 = _rng.normal(size=(3, 3))
    K2_31 = _rng.normal(size=(3, 3))
    _x = _rng.normal(size=(12, 12))

    # ---- ① 线性复合核 ----
    K_eff = np.zeros((5, 5))
    for _i in range(5):
        for _j in range(5):
            _s = 0.0
            for _u in range(3):
                for _v in range(3):
                    if 0 <= _i - _u < 3 and 0 <= _j - _v < 3:
                        _s += K1_31[_u, _v] * K2_31[_i - _u, _j - _v]
            K_eff[_i, _j] = _s
    _stack_lin = conv2_31(conv2_31(_x, K1_31), K2_31)
    _ok1 = np.allclose(conv2_31(_x, K_eff), _stack_lin, atol=1e-10)
    print(f"① 两个 3×3（线性）== 复合 5×5 核：{_ok1}")

    # ---- ② 插入 ReLU 后的最优 5×5 拟合 ----
    _Xs = _rng.normal(size=(6, 12, 12))
    _rows, _ys = [], []
    for _xx in _Xs:
        _out = conv2_31(np.maximum(conv2_31(_xx, K1_31), 0), K2_31)   # 中间插 ReLU
        _win = sliding_window_view(_xx, (5, 5))
        for _a in range(_out.shape[0]):
            for _b in range(_out.shape[1]):
                _rows.append(_win[_a, _b].ravel())
                _ys.append(_out[_a, _b])
    _A, _y = np.array(_rows), np.array(_ys)
    _Kfit, *_ = np.linalg.lstsq(_A, _y, rcond=None)
    _resid = float(np.linalg.norm(_A @ _Kfit - _y) / (np.linalg.norm(_y) + 1e-12))
    print(f"② 插 ReLU 后，最优 5×5 核的相对拟合误差：{_resid:.1%}（线性版是 0）")

    # ---- ③ 整网账本 ----
    _rf, _jump = 1, 1
    for _ in range(4):
        _rf += (5 - 1) * _jump
    _rf5 = _rf
    _rf, _jump = 1, 1
    for _ in range(8):
        _rf += (3 - 1) * _jump
    _rf3 = _rf
    print(f"③ 感受野：conv5×4 站 = {_rf5} vs conv3×8 站 = {_rf3}（相同）")
    print(f"   参数：4×25 = {4*25} vs 8×9 = {8*9}（省 {1 - 72/100:.0%}）｜ReLU：4 个 vs 8 个")

    _fig, _axes = plt.subplots(1, 3, figsize=(11.6, 3.9))
    _m = max(np.abs(K1_31).max(), np.abs(K2_31).max(), np.abs(K_eff).max())
    _axes[0].imshow(K1_31, cmap="RdBu_r", vmin=-_m, vmax=_m)
    _axes[0].set_title("核 1（3×3）")
    _axes[1].imshow(K2_31, cmap="RdBu_r", vmin=-_m, vmax=_m)
    _axes[1].set_title("核 2（3×3）")
    _axes[2].imshow(K_eff, cmap="RdBu_r", vmin=-_m, vmax=_m)
    _axes[2].set_title(f"复合核 K_eff（5×5）\n两个 3×3 线性堆叠的等价核")
    for _ax in _axes:
        _ax.axis("off")
    _fig.suptitle("两个 3×3 = 一个 5×5（线性）；插上 ReLU，5×5 就追不上了", fontsize=12)
    _fig.tight_layout()
    _fig
    return (conv2_31,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 3 · ImageNet 编年史：一条折线讲完九年

        蓝线：ILSVRC 各年冠军的 **top-5 错误率**——2012 年
        断崖（AlexNet），2015 年跌破人类水平线（ResNet 3.57%
        vs 人类约 5.1%）。柱状：参数量——注意它**不是**单调
        上涨：GoogLeNet 用 6.8M 打赢 AlexNet 的 60M，架构的
        进步一半是"花得更省"。

        （数字来自各论文与 ILSVRC 官方战绩；LeNet-5 列在
        1998 年做参照，它属于 MNIST 时代。）
        """
    )
    return


@app.cell
def _(plt):
    # 演示 3 · ImageNet 编年史
    _years = [2010, 2011, 2012, 2013, 2014, 2015]
    _errs = [28.2, 25.8, 15.3, 11.7, 6.7, 3.57]
    _names = ["传统方法", "传统方法", "AlexNet", "ZFNet", "GoogLeNet", "ResNet"]

    _fig, _axes = plt.subplots(1, 2, figsize=(12.4, 4.6),
                               gridspec_kw={"width_ratios": [1.35, 1]})
    _ax = _axes[0]
    _ax.plot(_years, _errs, "o-", color="tab:blue", lw=2.5, ms=8)
    for _yr, _er, _nm in zip(_years, _errs, _names):
        _ax.annotate(f"{_nm}\n{_er}%", (_yr, _er), textcoords="offset points",
                     xytext=(0, 14 if _yr in (2010, 2012, 2014) else -34),
                     ha="center", fontsize=8.5)
    _ax.axhline(5.1, color="tab:red", ls="--", lw=1.8, label="人类水平（约 5.1%）")
    _ax.axvspan(2011.5, 2012.5, color="tab:orange", alpha=0.12)
    _ax.text(2012, 27, "2012：AlexNet\n大爆发", ha="center", fontsize=9, color="tab:orange")
    _ax.set_ylim(0, 32)
    _ax.set_xlabel("年份")
    _ax.set_ylabel("ImageNet top-5 错误率（%）")
    _ax.set_xticks(_years)
    _ax.grid(True, alpha=0.3)
    _ax.legend(fontsize=9)
    _ax.set_title("错误率九年：2012 断崖，2015 跌破人类", fontsize=11)

    _ax2 = _axes[1]
    _models = ["LeNet-5\n(1998)", "AlexNet\n(2012)", "VGG-19\n(2014)", "GoogLeNet\n(2014)", "ResNet-152\n(2015)"]
    _pars = [0.06, 60, 144, 6.8, 60]
    _bars = _ax2.bar(_models, _pars, color=["tab:gray", "tab:orange", "tab:red",
                                            "tab:green", "tab:blue"])
    _ax2.set_yscale("log")
    _ax2.set_ylabel("参数量（百万，对数坐标）")
    for _b, _v in zip(_bars, _pars):
        _ax2.text(_b.get_x() + _b.get_width() / 2, _v * 1.2, f"{_v}M" if _v >= 1 else f"{_v}M",
                  ha="center", fontsize=9.5)
    _ax2.set_ylim(0.02, 400)
    _ax2.grid(True, axis="y", alpha=0.3)
    _ax2.set_title("参数量：不是越堆越肥，是越花越省", fontsize=11)
    _ax2.tick_params(axis="x", labelsize=8.5)
    _fig.suptitle("CNN 编年史（数字来自各论文与 ILSVRC 官方战绩）", fontsize=12)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ---

        ## 🎛️ 交互实验
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 交互实验 1 · 深度旋钮 + 残差开关：亲手踩一遍退化曲线

        拖深度（5~33 层，奇数）、开关残差，迷你战场（width 32、
        8 个 epoch）上 2~4 秒出结果。判词会对照同配置的参考
        锚点（写作时实测）。

        玩法建议：

        - 残差关、深度从 5 拉到 33：看第 1 轮准确率一路跳水
          （82% → 62% → 56%）——退化问题的亲手复现
        - 残差开、同样拉深度：第 1 轮稳在 87%~89% 不掉——
          跳跃线把"深"的起步价抹平了
        - 找到"残差 33 层 ≥ 普通 5 层"的组合（参考：0.9746
          vs 0.9679）
        """
    )
    return


@app.cell
def _(mo):
    # 交互控制：深度 + 残差开关
    depth_slider = mo.ui.slider(5, 33, value=33, step=2, label="🏗️ 隐藏层数（5~33，奇数）")
    residual_switch = mo.ui.switch(value=True, label="🛣️ 残差连接（跳跃线）")
    mo.hstack([depth_slider, residual_switch])
    return (depth_slider, residual_switch)


@app.cell
def _(depth_slider, mo, np, plt, residual_switch, train31):
    # ===== 交互实验 1 · 深度旋钮 + 残差开关（响应单元，width 32 迷你版）=====
    import time as _time
    _d = depth_slider.value
    _res = residual_switch.value
    _t0 = _time.time()
    _hist = train31(_d, _res, width=32, epochs=8)
    _dt = _time.time() - _t0

    _fig, _ax = plt.subplots(figsize=(8.6, 4.4))
    _ax.plot(range(1, 9), [t for t, v in _hist], "o-", lw=2.2,
             color="tab:red" if _res else "tab:blue")
    _ax.set_xlabel("epoch")
    _ax.set_ylabel("训练准确率（迷你战场）")
    _ax.set_ylim(0.4, 1.01)
    _ax.grid(True, alpha=0.3)
    _ax.set_title(f"{'残差' if _res else '普通'}-{_d} 层（width 32）：{_dt:.1f}s 跑完", fontsize=11.5)
    _fig.tight_layout()
    _fig

    # ---- 判词（对照 width 32 参考锚点，写作时实测）----
    _anchors = {(5, False): (0.819, 0.9679), (21, False): (0.622, 0.9478),
                (33, False): (0.559, 0.9210), (9, True): (0.873, 0.9616),
                (21, True): (0.875, 0.9671), (33, True): (0.894, 0.9746)}
    _ep1, _final = _hist[0][0], _hist[-1][0]
    if _res:
        _verdict = (f"残差-{_d} 层：第 1 轮 **{_ep1:.1%}**（起步即跑车）→ 终盘 **{_final:.1%}**。"
                    "跳跃线让梯度原样回流每一层，深度不再拖慢起步——"
                    "参考锚点：残差 33 层可达 97.5%，反超普通 5 层的 96.8%")
    else:
        if _ep1 < 0.70:
            _verdict = (f"普通-{_d} 层：第 1 轮只有 **{_ep1:.1%}**——深度退化的现场。"
                        f"终盘 {_final:.1%} 也追不回浅网络的水平（普通 5 层终盘 96.8%）。"
                        "把上面的残差开关打开，同样的深度再跑一次对比")
        else:
            _verdict = (f"普通-{_d} 层：第 1 轮 {_ep1:.1%}、终盘 {_final:.1%}——"
                        "这个深度普通网络还扛得住；再拉深几档，或打开残差开关对比")

    mo.vstack([
        _fig,
        mo.callout(mo.md(f"**实时判词：{_verdict}**"), kind="info"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✏️ 动手练习（每课必做，本课 3 题）": mo.md(
                """
                **练习 1 · 手写残差块前向**：补全跳跃连接那一行
                ——`out = h_in + 0.1 × z2`（α 缩放版）。算完和
                参考实现对账，并检查形状：块不改变维度。

                **练习 2 · VGG 账本**：4 站 conv5 与 8 站 conv3
                两条路线，笔算四个空：两边的感受野、conv5 路线
                参数、conv3 路线参数（28 课的递推公式 + 小学
                算术）。

                **练习 3 · 编年史侦探**：2012 年 AlexNet 赢下
                ImageNet 靠"三件套"——四张卡片里有一张**不是**
                三件套成员，找出它（提示：其中一件事 1986 年
                就发生了）。

                规则不变：先把 `None` 换成你的答案再运行；写完再看参考答案。
                """
            )
        }
    )
    return


@app.cell
def _(mo, np):
    # ===== 练习 1 · 手写残差块前向 =====
    rng_ex1 = np.random.default_rng(7)
    h_ex1 = rng_ex1.normal(size=(4, 32))          # 块输入
    W1_ex1 = rng_ex1.normal(size=(32, 32)) * 0.1
    b1_ex1 = np.zeros(32)
    W2_ex1 = rng_ex1.normal(size=(32, 32)) * 0.1
    b2_ex1 = np.zeros(32)

    def res_block_ex1(h, W1, b1, W2, b2, alpha=0.1):
        """残差块：两层 + 跳跃。补全最后一行。"""
        z1 = h @ W1 + b1
        a1 = np.maximum(z1, 0)
        z2 = a1 @ W2 + b2
        out = None                                 # ← ① 补：跳跃连接 + α 缩放的残差分支
        return out

    res_ready_ex1 = None                           # ← 补完后，把这行换成 res_block_ex1

    if res_ready_ex1 is not None:
        try:
            _got = res_ready_ex1(h_ex1, W1_ex1, b1_ex1, W2_ex1, b2_ex1)
            _z1 = h_ex1 @ W1_ex1 + b1_ex1
            _ref = h_ex1 + 0.1 * (np.maximum(_z1, 0) @ W2_ex1 + b2_ex1)
            if np.allclose(_got, _ref, atol=1e-12) and _got.shape == h_ex1.shape:
                _msg = ("✅ 对账通过，形状 (4, 32) 不变——残差块的铁律：x 和 F(x) "
                        "同形才能相加。少了 0.1 会怎样？把 α 去掉跑一遍，"
                        "看演示 1 的方差爆炸版（我们实验初期真踩过这坑）")
            elif _got.shape != h_ex1.shape:
                _msg = "❌ 形状变了——块不改变维度，out 必须和 h 同形"
            else:
                _msg = "❌ 数值对不上——跳跃线是 h_in（块的输入），不是 z1 或 a1；分支要乘 0.1"
        except (TypeError, ValueError) as _e:
            _msg = f"❌ 运行出错：{_e}——提示：out = h + 0.1 * z2"
    else:
        _msg = "⏳ 先补跳跃行，再把 res_ready_ex1 换成 res_block_ex1"
    mo.md(_msg)
    return (res_block_ex1, res_ready_ex1)


@app.cell
def _(mo):
    # ===== 练习 2 · VGG 账本 =====
    print("两条路线（单通道，28 课的递推公式 RF += (K−1)×jump）：")
    print("  A 路线：4 站 conv5（每站感受野 +4）")
    print("  B 路线：8 站 conv3（每站感受野 +2）")
    print("笔算四个空：")

    rf_a_ex2 = None                                # ← ① A 路线的最终感受野
    rf_b_ex2 = None                                # ← ② B 路线的最终感受野
    params_a_ex2 = None                            # ← ③ A 路线参数（4 站 × 每站几个权重？）
    params_b_ex2 = None                            # ← ④ B 路线参数
    return (params_a_ex2, params_b_ex2, rf_a_ex2, rf_b_ex2)


@app.cell
def _(mo, params_a_ex2, params_b_ex2, rf_a_ex2, rf_b_ex2):
    # ===== 练习 2 · 判词 =====
    _truth = {"① A 感受野": (rf_a_ex2, 17), "② B 感受野": (rf_b_ex2, 17),
              "③ A 参数": (params_a_ex2, 100), "④ B 参数": (params_b_ex2, 72)}
    if any(_v[0] is None for _v in _truth.values()):
        _msg = "⏳ 四个数都算出来再填（起点 RF=1；每站参数 = K×K 个权重）"
    else:
        _wrong = [k for k, (a, t) in _truth.items() if a != t]
        if not _wrong:
            _msg = ("✅ 17 = 17、100 vs 72（省 28%）——同样的视野，B 路线还多出 "
                    "4 个 ReLU（8 vs 4）。这就是 VGG 敢全用 3×3 的全部底气")
        else:
            _hint = {"① A 感受野": "1 + 4×4",
                     "② B 感受野": "1 + 8×2",
                     "③ A 参数": "4 站 × 5×5",
                     "④ B 参数": "8 站 × 3×3"}
            _msg = "❌ 还差：" + "；".join(f"{k}（{_hint[k]}）" for k in _wrong)
    mo.md(_msg)
    return


@app.cell
def _(mo):
    # ===== 练习 3 · 编年史侦探 =====
    print("2012 年 AlexNet 赢下 ImageNet 的「三件套」，四张卡片里有一张混进来的：")
    print("  A. GPU 算力（两块 GTX 580）")
    print("  B. ImageNet 大数据（1000 类 128 万张）")
    print("  C. 反向传播算法的发明")
    print("  D. ReLU + Dropout 算法创新")

    ans_card_ex3 = None                            # ← ① 不属于三件套的卡片字母（"A"/"B"/"C"/"D"）
    return (ans_card_ex3,)


@app.cell
def _(ans_card_ex3, mo):
    # ===== 练习 3 · 判词 =====
    if ans_card_ex3 is None:
        _msg = "⏳ 选一张卡片字母填进来"
    elif ans_card_ex3 == "C":
        _msg = ("✅ 答对了：反向传播 1986 年就由 Rumelhart、Hinton 等人推广开来，"
                "LeNet-5 时代天天在用——2012 年的新东西是数据（B）、算力（A）和"
                " ReLU+Dropout（D）的组合。三件套凑齐，时代才爆")
    else:
        _msg = "❌ 再想想：三件套是 2012 年「新凑齐」的条件——哪件事 26 年前就发生了？"
    mo.md(_msg)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 1 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                out = h + 0.1 * z2
                ```

                - **h** 是块的输入（跳跃线的起点），**z2** 是
                  两层变换的输出（残差分支）
                - **0.1（α 缩放）**：让残差分支"先小声说话"，
                  防止方差逐块累加——真实 ResNet 用 BatchNorm
                  达成同样目的；ReZero/Fixup 一族证明直接缩放
                  也行
                - 反向传播里最关键的一行（演示 1 的 backward31）：
                  `dh = dz1 @ W1.T + dh`——**那个 +dh 就是跳跃
                  线**：梯度不衰减、不过门，原样回流。深网络的
                  梯度高速公路
                """
            ),
            "✅ 练习 2 参考答案（先自己写过再看！）": mo.md(
                """
                **17、17、100、72。**

                - A 感受野：1 + 4 站 × (5−1) = **17**
                - B 感受野：1 + 8 站 × (3−1) = **17**
                - A 参数：4 × 25 = **100**
                - B 参数：8 × 9 = **72**（省 28%）

                别忘了账本外的第三项：B 路线有 **8 个 ReLU**，
                A 只有 4 个——非线性容量翻倍。参数、视野、
                非线性三项全算上，小核堆叠才完胜。VGG 的
                "philosophy" 就这三个数字。

                （进阶：若 C_in=C_out=多通道，参数差距更大——
                5×5 核的花费是 C_in×C_out×25，两个 3×3 是
                C_in×C_out×9×2，比例不变。）
                """
            ),
            "✅ 练习 3 参考答案（先自己写过再看！）": mo.md(
                """
                **C——反向传播不是 2012 年的事。**

                反向传播 1986 年（Rumelhart、Hinton、Williams）
                广为人知，LeCun 1998 年训 LeNet-5 用的就是它。
                2012 年真正"新凑齐"的三件套：

                - **B 数据**：ImageNet 2009 年建成——128 万张、
                  1000 类。深度学习的胃口，终于喂得饱
                - **A 算力**：GTX 580 的并行浮点算力，把训练
                  时间从"年"缩到"天"
                - **D 算法**：ReLU 解决饱和（19 课）、Dropout
                  压过拟合（25 课）——深度网络终于训得动

                一句常被引用的总结：深度学习 = **大数据 ×
                大算力 × 可训练的深架构**。三者相乘，缺一个
                都不是 2012。
                """
            ),
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📝 本课小结（展开自查：6 条都答得上来吗？）": mo.md(
                """
                1. **2012 大爆发三件套**：ImageNet 大数据 +
                   GPU 算力 + ReLU/Dropout；AlexNet top-5
                   15.3% vs 传统方法 26.1%——反向传播不是新
                   玩意（1986），三件套凑齐才是关键
                2. **VGG 哲学**：小核堆深。两个 3×3 线性堆叠
                   严格等价某个 5×5（对账 True），插 ReLU 后
                   5×5 最优拟合误差 92%；同 RF 17 下参数
                   72 < 100、ReLU 8 > 4
                3. **GoogLeNet 哲学**：Inception 多尺度并联 +
                   1×1 卷积降维——22 层 6.8M 参数打赢 AlexNet
                   的 60M（top-5 6.7% vs 15.3%）
                4. **退化问题**：普通网络越深**训练误差**越高
                   （我们实测第 1 轮 91%→71%→57%、终盘
                   99.1%→94.7%）——纯优化难题，不是过拟合
                5. **残差连接**：y = x + F(x)，想让块变恒等
                   只需 F=0；反向 ∂y/∂x = 1 + ∂F/∂x，跳跃线
                   是不衰减的梯度高速公路；实测残差 33 层
                   起步 91%（= 普通 5 层）终盘 99.4%（> 普通
                   5 层的 99.1%）
                6. **参数的经济学**：编年史的隐藏规律——错误率
                   一路降（15.3→3.57%，跌破人类 5.1%），参数
                   不一路涨（GoogLeNet 6.8M < AlexNet 60M）；
                   架构演进 = 把参数越花越准
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(
                """
                ## 🎓 第 5 部分收官：卷积神经网络（6/6）

                六课连打，你现在手里有一台**完整的、自己拧过每颗
                螺丝的 CNN**。清点行囊：

                | 课 | 里程碑：你现在能做到什么 |
                |---|---|
                | **26** 卷积直觉 | 小手电扫图、权值共享、1256 倍的参数差距 |
                | **27** 卷积数学 | 尺寸公式、same 配方、多通道合议制 |
                | **28** 池化与感受野 | max 池化的平移容忍（7% → 40%）、RF 递推 |
                | **29** 实现卷积层 | im2col + 反向传播 + 10⁻¹⁰ 数值对账 + max 路由 |
                | **30** LeNet-5 | 端到端训练 CNN，98.51% 打赢 MLP（参数少 56%） |
                | **31** 架构演进 | 亲手复现退化与残差救援；两个 3×3 的数学账 |

                用一句话回顾第 5 部分：图像的规律是**局部的、
                平移可复用的**——卷积把这两个先验焊进结构，
                池化把粒度换成稳定，感受野随深度长大，残差让
                深度免费——最后，44,426 个参数打赢 101,770 个。
                """
            ),
            mo.callout(
                mo.md(
                    """
                    26 课埋的那句话现在可以兑现了：**全连接看图，
                    像把照片剪成碎屑再认**；卷积看图，是保留了
                    空间的"语法"去读。这个"结构先验"的思想，
                    下一部分会换个战场再演一遍——

                    图像的空间有"局部性"，**序列的时间也有**：
                    一句话的下一个词，取决于**前面**的词——
                    而且通常是很近的那几个。怎么让网络记住
                    "上文"？循环神经网络（RNN）登台。
                    """
                ),
                kind="success",
            ),
            mo.md(
                """
                ---

                **下一课预告** — `32_序列数据`（第 6 部分 · 循环神经网络）：

                到现在为止，所有网络吃的都是**定长输入**（784
                个像素、一张 28×28）。可语言不是：一句话三个
                词，下一句三十个词。第 6 部分从"序列长什么样"
                开始：变长、有序、上文决定下文——以及为什么
                全连接在序列上又一次水土不服（权重没法跟
                   着词的位置走）。

                你会先在玩具序列上练手（预测下一个字符、
                补全数字模式），然后一步步逼近 RNN 的核心：
                **同一组权重，沿着时间循环使用**——听起来
                耳熟吗？**权值共享**，你在 26 课刚见过它
                的空间版。

                🌊 图像通关，去序列的海！
                """
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
