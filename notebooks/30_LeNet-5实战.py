import marimo

__generated_with = "0.25.2"
app = marimo.App(app_title="30 · LeNet-5 实战", width="medium")


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
        # 30 · LeNet-5 实战：向 MLP 正式宣战

        **从零到 Transformer · 手写深度学习** ｜ 第 5 部分 · 卷积神经网络（5/6）
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📖 本课目标（先读我）": mo.md(
                """
                **第 5 部分收官之战：训练一个真正的 CNN。**

                1998 年，Yann LeCun 用 LeNet-5 把手写支票识别
                铺进了全美国的银行——那是卷积网络第一次大规模
                服役。今天我们用前四课攒下的全部零件，把这台
                1998 年的机器亲手复活：

                - 26/27 课的**卷积**（局部窗口 + 权值共享 +
                  步长填充）
                - 28 课的**池化**（下采样 + 平移容忍）
                - 29 课的**im2col 前向 + 反向传播**（今天唯一
                  的"新代码"，其余全是组装）
                - 22~23 课的反向传播纪律、24 课的 **Adam**、
                  25 课的**完全体训练循环**（He 初始化、洗牌
                  切批、早停）

                对手是 25 课的老将：784-128-10 的 MLP，
                **97.78%**、101,770 个参数。LeNet-5 只有
                **44,426** 个参数——**少 56% 的参数，能打
                赢吗？**

                学完你将能：

                - 画出 LeNet-5 的七层结构图，说清每层的形状账
                  和分工（卷积提特征、池化压尺寸、全连接判决）
                - 把 29 课的 conv/pool 反向接进 25 课的训练
                  循环，跑通**第一个端到端 CNN**
                - 读懂训练曲线（val 峰值 → 耐心耗尽 → 早停
                  回滚），并对比 MLP 与 LeNet 的**分类别战报**
                - 亲眼看到 conv1 的 6 个核训练后**长出了方向
                  分工**——没人教过它们什么是边缘

                前置：29 课全部（今天的代码直接复用其思想）、
                25 课训练循环、24 课 Adam。本课训练实测约
                **4 分钟**（普通 CPU）——泡杯茶，值得。
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "🏛️ 概念 1/2 · LeNet-5 解剖：七层流水线": mo.md(
                """
                LeNet-5 的名字里 5 = 五层**可训练**层（两个卷积
                + 三个全连接），加上池化共七站。数据从进到出：

                | 站 | 层 | 形状变化 | 干什么 | 参数 |
                |---|---|---|---|---|
                | 输入 | — | 28×28×1 | 灰度图（28 版；原论文 pad 到 32） | 0 |
                | C1 | conv 5×5×6 | → 24×24×6 | 6 把尺子提 6 种局部特征 | 156 |
                | S2 | pool 2×2 | → 12×12×6 | 减半 + 容忍抖动 | 0 |
                | C3 | conv 5×5×16 | → 8×8×16 | 6 种特征组合出 16 种更复杂特征 | 2,416 |
                | S4 | pool 2×2 | → 4×4×16 | 再减半 | 0 |
                | F5 | fc | → 120 | 把 256 维特征图摊平判决 | 30,840 |
                | F6 | fc | → 84 | 再抽象一层 | 10,164 |
                | OUT | fc | → 10 | 10 类打分 | 850 |

                **三个设计思想**，正是前四课的浓缩：

                1. **特征层次化**：C1 看 5×5 的笔画片段，C3 的
                   感受野 14×14（28 课算过）——看的已经是"部件"。
                   越深越抽象，这是所有 CNN 的通则
                2. **卷积管提特征、池化管压尺寸、全连接管判决**：
                   各司其职。参数大头在全连接（41,854/44,426 =
                   94%）——卷积层用 2,572 个参数干了最重的活
                3. **参数经济学**：如果第一层用全连接看 28×28，
                   784×6 = 4,704 个参数；卷积 156 个，**差 30 倍**，
                   还附赠平移稳定性——这就是结构先验的价值

                对比 25 课 MLP 的 101,770：LeNet 少 56% 参数。
                省下的钱花在哪了？**花在"更深"上**——两层卷积
                + 两层池化的特征加工，MLP 一步都做不了。
                """
            ),
            "⚔️ 概念 2/2 · 战前清点：训练计划的每个决定": mo.md(
                """
                训练配置不是拍脑袋，每一项都有出处：

                - **He 初始化**（25 课）：conv 层 fan_in =
                  C·K·K（C1 是 25、C3 是 150），std = sqrt(2/fan_in)
                - **Adam，lr = 0.002**（24 课）：卷积网络梯度
                  量级和 MLP 不同，lr 稍大收敛更快
                - **batch 128，每轮洗牌**（24 课）：781 → 391
                  步/epoch，比 batch 64 少一半 Python 开销
                - **softmax + 交叉熵**（21 课）：输出层配对表
                  的标配；反向就是 p − y
                - **早停 patience = 3**（25 课）：val 创新高就
                  存档，三轮不涨就刹车回滚
                - **速度工程**（29 课的延伸）：im2col 换成
                  **滑窗视图零循环版**、反向的 einsum 摊平成
                  **一次大矩阵乘**——训练要跑几千步，每步省
                  一点，全程就省几分钟。数学和 29 课逐字相同，
                  只是搬运方式更快

                预期战况（写作时实测，供对照）：epoch 1 就到
                97.3%（MLP 要 1~2 个 epoch），epoch 6 峰值
                98.50%，早停于 epoch 9，**测试集 98.51%**，
                全程约 3.7 分钟。你的数字会因 CPU 与 BLAS
                线程略有浮动，小数点后一两位内都算复现。
                """
            ),
        }
    )
    return


@app.cell
def _(mo, np):
    # MNIST 加载器（与 25~29 课同款）
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

    X_mnist, y_mnist, X_te30, y_te30 = load_mnist()
    print("训练集:", X_mnist.shape, " 测试集:", X_te30.shape)
    return (X_mnist, X_te30, y_mnist, y_te30)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 1 · 架构海报：数据的一路七站

        每个方块是一站的输出（**方块高 ∝ 特征图边长**，
        块内标注尺寸），方块下方是参数账。注意三件事：

        - **方块越走越小、通道越走越多**：空间分辨率换成
          特征丰富度（28×28×1 → 4×4×16），这是"卷积提特征、
          池化压尺寸"的几何呈现
        - **卷积层的参数小得可怜**（柱状图里几乎看不见），
          却承担了全部的特征提取
        - F5 一个全连接就占 30,840 参数（69%）——LeNet 的
          钱，一半花在了"判决席"上
        """
    )
    return


@app.cell
def _(np, plt):
    # 演示 1 · LeNet-5 架构海报
    _stations = [
        ("输入", 28, 28, 1, 0, "gray"),
        ("C1 conv", 24, 24, 6, 156, "tab:blue"),
        ("S2 pool", 12, 12, 6, 0, "lightgray"),
        ("C3 conv", 8, 8, 16, 2416, "tab:blue"),
        ("S4 pool", 4, 4, 16, 0, "lightgray"),
        ("F5 fc", 1, 1, 120, 30840, "tab:orange"),
        ("F6 fc", 1, 1, 84, 10164, "tab:orange"),
        ("OUT", 1, 1, 10, 850, "tab:red"),
    ]
    _fig, _axes = plt.subplots(1, 2, figsize=(12.6, 4.6),
                               gridspec_kw={"width_ratios": [2.6, 1]})
    _ax = _axes[0]
    _x = 0.0
    for _nm, _h, _w, _c, _p, _col in _stations:
        _size = max(np.sqrt(_h * _w) / 28 * 7.2, 0.9)   # 方块高 ∝ 特征图边长（归一到输入）
        _ax.add_patch(plt.Rectangle((_x, 5 - _size / 2), 1.15, _size,
                                    facecolor=_col, alpha=0.75 if _p else 0.4,
                                    edgecolor="black", lw=0.8))
        _ax.text(_x + 0.575, 5 + _size / 2 + 0.3, _nm, ha="center", fontsize=9.5)
        _lbl = f"{_h}×{_w}×{_c}" if _h > 1 else f"{_c}"
        _ax.text(_x + 0.575, 5, _lbl, ha="center", va="center", fontsize=8.5,
                 rotation=90 if _h > 8 else 0)
        _ax.text(_x + 0.575, 5 - _size / 2 - 0.32,
                 f"{_p:,}" if _p else "0", ha="center", fontsize=8.5, color="dimgray")
        _x += 1.55
    _ax.set_xlim(-0.4, _x)
    _ax.set_ylim(0.6, 9.6)
    _ax.axis("off")
    _ax.set_title("数据的一路七站（方块高 ∝ 特征图边长，方块下 = 参数量）", fontsize=11)

    _ax2 = _axes[1]
    _names = ["C1", "C3", "F5", "F6", "OUT"]
    _pars = [156, 2416, 30840, 10164, 850]
    _bars = _ax2.bar(_names, _pars, color=["tab:blue", "tab:blue", "tab:orange",
                                           "tab:orange", "tab:red"])
    _ax2.set_ylabel("参数量")
    _ax2.set_title(f"参数账：共 {sum(_pars):,}（全连接占 "
                   f"{sum(_pars[2:]) / sum(_pars):.0%}）", fontsize=11)
    for _b, _v in zip(_bars, _pars):
        _ax2.text(_b.get_x() + _b.get_width() / 2, _v * 1.05, f"{_v:,}",
                  ha="center", fontsize=9)
    _ax2.set_ylim(0, 36000)
    _ax2.grid(True, axis="y", alpha=0.3)
    _fig.suptitle("LeNet-5（28×28 输入版）：越走越小、越走越抽象", fontsize=12)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 2 · 主战场：4 分钟的训练完全体

        下面这个单元是全课程迄今最长的代码——但拆开看全是
        熟人：`im2col_fast` 是 29 课 im2col 的零循环近亲
        （滑窗视图 + transpose，29 练习 1 见过）；`conv_backward`
        的三步数学与 29 课逐字相同，只是把 einsum 摊平成一次
        大矩阵乘（更快）；训练循环是 25 课的完全体。

        运行约 **4 分钟**（普通 CPU）。观察打印的战报：

        - **epoch 1 就到 97.3%**——卷积特征先验的威力：
          MLP 同期还在 95% 附近
        - epoch 6 摸到 val 峰值 **98.50%**，之后三轮不涨，
          早停刹车于 epoch 9，参数回滚到 epoch 6 版本
        - **测试集 98.51%**：总错分从 MLP 的 222 个（97.78%）
          降到 **149 个**
        """
    )
    return


@app.cell
def _(X_mnist, X_te30, mo, np, plt, y_mnist, y_te30):
    # 演示 2 · LeNet-5：前向 + 反向 + Adam 完全体训练（本课核心单元）
    from numpy.lib.stride_tricks import sliding_window_view
    import time as _time

    # ---- 速度件：im2col 零循环版（29 课 im2col 的滑窗视图近亲） ----
    def im2col_fast(x, K, stride=1, pad=0):
        """(N,C,H,W) → (N, C·K·K, OH·OW)。数学同 29 课，搬运零循环。"""
        if pad:
            x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        _N, _C, _H, _W = x.shape
        _oh, _ow = (_H - K) // stride + 1, (_W - K) // stride + 1
        _win = sliding_window_view(x, (K, K), axis=(2, 3))[:, :, ::stride, ::stride]
        return _win.transpose(0, 1, 4, 5, 2, 3).reshape(_N, _C * K * K, _oh * _ow)

    def col2im_fast(dcols, x_shape, K, stride=1, pad=0):
        """散射回填的 K² 向量化版（等价于 29 课的 OH·OW 循环）。"""
        _N, _C, _H, _W = x_shape
        _oh = (_H + 2 * pad - K) // stride + 1
        _ow = (_W + 2 * pad - K) // stride + 1
        _dc = dcols.reshape(_N, _C, K, K, _oh, _ow)
        _dxp = np.zeros((_N, _C, _H + 2 * pad, _W + 2 * pad))
        for _k1 in range(K):
            for _k2 in range(K):
                _dxp[:, :, _k1:_k1+stride*_oh:stride, _k2:_k2+stride*_ow:stride] += \
                    _dc[:, :, _k1, _k2, :, :]
        return _dxp[:, :, pad:_H+pad, pad:_W+pad] if pad else _dxp

    def conv_forward30(x, W, b):
        """conv 5×5、stride 1、pad 0（LeNet 专用，cols 供反向复用）。"""
        _N, _C, _H, _W = x.shape
        _F, _, _K, _ = W.shape
        _cols = im2col_fast(x, _K)
        _W2 = W.reshape(_F, _C * _K * _K)
        _Z = _W2 @ _cols + b[:, None]                      # 一次矩阵乘 + 广播（29 课）
        return _Z.reshape(_N, _F, _H - _K + 1, _W - _K + 1), _cols

    def conv_backward30(dout, x, W, cols):
        """29 课三步：db 求和、dW 误差×窗口、dX 散射。einsum 摊平成大矩阵乘。"""
        _N, _C, _H, _W = x.shape
        _F, _, _K, _ = W.shape
        _oh, _ow = dout.shape[2], dout.shape[3]
        _W2 = W.reshape(_F, _C * _K * _K)
        _dout2 = dout.reshape(_N, _F, _oh * _ow)
        _db = dout.sum(axis=(0, 2, 3))                     # ① db
        _dout_flat = _dout2.transpose(1, 0, 2).reshape(_F, _N * _oh * _ow)
        _cols_flat = cols.transpose(1, 0, 2).reshape(_C * _K * _K, _N * _oh * _ow)
        _dW2 = _dout_flat @ _cols_flat.T                   # ② dW：一次大矩阵乘
        _dcols = (_W2.T @ _dout_flat).reshape(_C * _K * _K, _N, _oh * _ow).transpose(1, 0, 2)
        _dx = col2im_fast(_dcols, x.shape, _K)             # ③ 散射累加
        return _dW2.reshape(W.shape), _db, _dx

    def pool2_fwd30(h):
        """2×2 max 池化，顺手记赢家掩码（29 练习 4 的批量版）。"""
        _N, _C, _H, _W = h.shape
        _hb = h[:, :, :_H//2*2, :_W//2*2].reshape(_N, _C, _H//2, 2, _W//2, 2)
        _out = _hb.max(axis=(3, 5))
        _mask = _hb == _out[:, :, :, None, :, None]        # 记账：谁是赢家
        return _out, _mask

    def pool2_bwd30(dout, mask):
        """梯度只走赢家（并列时都给——亚梯度，与 argmax 版等价到浮点误差）。"""
        return (dout[:, :, :, None, :, None] * mask).reshape(
            mask.shape[0], mask.shape[1], mask.shape[2] * 2, mask.shape[4] * 2)

    class LeNet5:
        """七层流水线。参数是 dict，Adam（24 课）按 key 记账。"""
        def __init__(self, seed=0):
            _r = np.random.default_rng(seed)
            self.p = {
                "Wc1": _r.normal(0, 1, (6, 1, 5, 5)) * np.sqrt(2 / 25),    # He（25 课）
                "bc1": np.zeros(6),
                "Wc3": _r.normal(0, 1, (16, 6, 5, 5)) * np.sqrt(2 / 150),
                "bc3": np.zeros(16),
                "Wf5": _r.normal(0, 1, (120, 256)) * np.sqrt(2 / 256),
                "bf5": np.zeros(120),
                "Wf6": _r.normal(0, 1, (84, 120)) * np.sqrt(2 / 120),
                "bf6": np.zeros(84),
                "Wout": _r.normal(0, 1, (10, 84)) * np.sqrt(2 / 84),
                "bout": np.zeros(10),
            }

        def forward(self, X, cache=None):
            _p = self.p
            _z1, _cols1 = conv_forward30(X, _p["Wc1"], _p["bc1"])       # C1
            _h1 = np.maximum(_z1, 0)                                     # ReLU（19 课）
            _h1p, _m1 = pool2_fwd30(_h1)                                 # S2
            _z3, _cols3 = conv_forward30(_h1p, _p["Wc3"], _p["bc3"])     # C3
            _h3 = np.maximum(_z3, 0)
            _h3p, _m3 = pool2_fwd30(_h3)                                 # S4
            _flat = _h3p.reshape(len(X), -1)                             # flatten（07 课）
            _z5 = _flat @ _p["Wf5"].T + _p["bf5"]                        # F5
            _h5 = np.maximum(_z5, 0)
            _z6 = _h5 @ _p["Wf6"].T + _p["bf6"]                          # F6
            _h6 = np.maximum(_z6, 0)
            _logits = _h6 @ _p["Wout"].T + _p["bout"]                    # OUT
            if cache is not None:
                cache.update(dict(X=X, cols1=_cols1, h1=_h1, m1=_m1, h1p=_h1p,
                                  cols3=_cols3, h3=_h3, m3=_m3, h3p=_h3p,
                                  flat=_flat, h5=_h5, h6=_h6))
            return _logits

        def backward(self, dout_logits, c):
            """逐站反着走：fc 三段（23 课）→ pool 路由 → conv 反向（29 课）。"""
            _p, _g = self.p, {}
            _N = len(dout_logits)
            _g["Wout"] = dout_logits.T @ c["h6"]                          # OUT
            _g["bout"] = dout_logits.sum(0)
            _dz6 = (dout_logits @ _p["Wout"]) * (c["h6"] > 0)             # ReLU 反向（19 课）
            _g["Wf6"] = _dz6.T @ c["h5"]                                  # F6
            _g["bf6"] = _dz6.sum(0)
            _dz5 = (_dz6 @ _p["Wf6"]) * (c["h5"] > 0)
            _g["Wf5"] = _dz5.T @ c["flat"]                                # F5
            _g["bf5"] = _dz5.sum(0)
            _dflat = _dz5 @ _p["Wf5"]
            _dz3 = pool2_bwd30(_dflat.reshape(_N, 16, 4, 4), c["m3"]) * (c["h3"] > 0)   # S4 路由
            _g["Wc3"], _g["bc3"], _dh1p = conv_backward30(_dz3, c["h1p"], _p["Wc3"], c["cols3"])   # C3
            _dz1 = pool2_bwd30(_dh1p, c["m1"]) * (c["h1"] > 0)            # S2 路由
            _g["Wc1"], _g["bc1"], _ = conv_backward30(_dz1, c["X"], _p["Wc1"], c["cols1"])          # C1
            return _g

        def predict(self, X, bs=512):
            _outs = []
            for _i in range(0, len(X), bs):
                _outs.append(np.argmax(self.forward(X[_i:_i+bs]), 1))
            return np.concatenate(_outs)

    # ---- 训练完全体（25 课的循环 + 29 课的层） ----
    def train_lenet30(epochs=10, lr=0.002, batch=128, seed=0, patience=3):
        _X_all = X_mnist.reshape(-1, 1, 28, 28)
        _rs = np.random.default_rng(42)                    # 三分天下（17 课）
        _idx = _rs.permutation(len(_X_all))
        _Xa, _ya = _X_all[_idx], y_mnist[_idx]
        _Xtr, _ytr = _Xa[:50000], _ya[:50000]
        _Xva, _yva = _Xa[50000:], _ya[50000:]
        _Xte = X_te30.reshape(-1, 1, 28, 28)

        _net = LeNet5(seed=seed)
        _keys = list(_net.p.keys())
        _m = {k: np.zeros_like(v) for k, v in _net.p.items()}     # Adam 四本账（24 课）
        _s = {k: np.zeros_like(v) for k, v in _net.p.items()}
        _r = np.random.default_rng(seed + 1)
        _hist = {"acc_va": [], "L": []}
        _best_va, _best_pack, _wait = -1.0, None, 0
        _t0 = _time.time()
        _step = 0
        for _ep in range(epochs):
            _order = _r.permutation(len(_Xtr))             # 每轮洗牌（24 课）
            for _st in range(0, len(_Xtr), batch):
                _ib = _order[_st:_st + batch]
                _xb, _yb = _Xtr[_ib], _ytr[_ib]
                _cache = {}
                _logits = _net.forward(_xb, _cache)
                _e = np.exp(_logits - _logits.max(1, keepdims=True))     # 稳定 softmax（21 课）
                _P = _e / _e.sum(1, keepdims=True)
                _L = -float(np.mean(np.log(_P[np.arange(len(_yb)), _yb] + 1e-12)))
                _dl = _P.copy()                                          # p − y（21 课）
                _dl[np.arange(len(_yb)), _yb] -= 1
                _dl /= len(_yb)
                _grads = _net.backward(_dl, _cache)
                _step += 1
                for _k in _keys:                                        # Adam（24 课）
                    _m[_k] = 0.9 * _m[_k] + 0.1 * _grads[_k]
                    _s[_k] = 0.999 * _s[_k] + 0.001 * _grads[_k] ** 2
                    _net.p[_k] -= lr * (_m[_k] / (1 - 0.9 ** _step)) / \
                        (np.sqrt(_s[_k] / (1 - 0.999 ** _step)) + 1e-8)
            _acc_va = float(np.mean(_net.predict(_Xva) == _yva))
            _hist["acc_va"].append(_acc_va)
            _hist["L"].append(_L)
            if _acc_va > _best_va + 1e-4:                 # 早停记账（25 课）
                _best_va, _wait = _acc_va, 0
                _best_pack = {k: v.copy() for k, v in _net.p.items()}
                _hist["best_ep"] = _ep + 1
            else:
                _wait += 1
                if _wait >= patience:
                    _hist["stopped"] = _ep + 1
                    break
            print(f"  epoch {_ep+1}: L={_L:.4f} acc_va={_acc_va:.4f} "
                  f"best={_best_va:.4f}（{_time.time()-_t0:.0f}s）")
        _hist.setdefault("stopped", epochs)
        _net.p = _best_pack                                 # 回滚最佳版本
        _hist["total_s"] = _time.time() - _t0
        _hist["acc_te"] = float(np.mean(_net.predict(_Xte) == y_te30))
        _hist["best_va"] = _best_va
        return _net, _hist

    lenet_net, lenet_hist = train_lenet30()
    print(f"\n⏱ 总耗时 {lenet_hist['total_s']:.0f}s｜val 峰值 {lenet_hist['best_va']:.4f}"
          f"（epoch {lenet_hist['best_ep']}）｜早停于 epoch {lenet_hist['stopped']}")
    print(f"🎯 测试集 {lenet_hist['acc_te']:.4f}（对照：25 课 MLP 97.78%｜瞎猜 10%）")

    # ---- 训练曲线 ----
    _fig, _ax = plt.subplots(figsize=(8.2, 4.2))
    _ep = np.arange(1, len(lenet_hist["acc_va"]) + 1)
    _ax.plot(_ep, lenet_hist["acc_va"], "o-", color="tab:blue", lw=2, label="验证准确率")
    _ax.scatter([lenet_hist["best_ep"]], [lenet_hist["best_va"]], s=160, facecolors="none",
                edgecolors="tab:red", linewidths=2.5, zorder=5,
                label=f"峰值 epoch {lenet_hist['best_ep']}（{lenet_hist['best_va']:.2%}）")
    _ax.axhline(0.9778, color="gray", ls="--", lw=1.5, label="25 课 MLP 水平线（97.78%）")
    _ax.axvline(lenet_hist["stopped"], color="tab:red", ls=":", lw=1.5,
                label=f"早停刹车（epoch {lenet_hist['stopped']}）")
    _ax.set_xlabel("epoch")
    _ax.set_ylabel("验证准确率")
    _ax.set_ylim(0.955, 1.0)
    _ax.grid(True, alpha=0.3)
    _ax.legend(fontsize=9, loc="lower right")
    _ax.set_title(f"LeNet-5 训练战报：测试集 {lenet_hist['acc_te']:.2%}，"
                  f"用时 {lenet_hist['total_s']:.0f}s")
    _fig.tight_layout()
    _fig
    return (LeNet5, conv_forward30, im2col_fast, lenet_hist, lenet_net, pool2_fwd30, sliding_window_view)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 3 · conv1 的六把尺子：训练后长成了什么样

        初始化时它们是 6 张随机噪声；训练 10 个 epoch 后——
        看下排的特征图：**同一个数字（测试集 #0，是个 7）在
        六张图上留下六种不同的指纹**。

        量化一下方向性：每把尺子的**左右权重差**和**上下
        权重差**（核的左两列减右两列、上两行减下两行）。
        写作时实测：核 3 左右差 +2.7（竖边倾向，它在 7 的
        斜笔画两侧点亮红蓝）、核 1 上下差 −2.8（横边倾向）、
        核 0 和核 5 方向性弱、更像"墨迹探测器"（有笔画就亮）。

        它们不像 26 课手写的 Sobel 那样工整——**没人教过
        网络什么是边缘**，这些形状是交叉熵损失逼出来的自发
        分工。1998 年 LeCun 论文里的第一层，画出来也是
        这副"长歪了的边缘检测器"模样。
        """
    )
    return


@app.cell
def _(X_te30, lenet_net, np, plt, sliding_window_view):
    # 演示 3 · conv1 学到的 6 个滤波器 + 特征图
    _Wc1 = lenet_net.p["Wc1"]                       # (6,1,5,5)
    _img = X_te30[0].reshape(28, 28)                # 测试集 #0
    _win = sliding_window_view(_img, (5, 5))
    _fig, _axes = plt.subplots(2, 6, figsize=(13.2, 4.8))
    for _f in range(6):
        _W = _Wc1[_f, 0]
        _r = np.einsum("ijkl,kl->ij", _win, _W)
        _lr = float(_W[:, :2].sum() - _W[:, 3:].sum())
        _ud = float(_W[:2, :].sum() - _W[3:, :].sum())
        _axes[0, _f].imshow(_W, cmap="RdBu_r",
                            vmin=-max(abs(_Wc1.min()), abs(_Wc1.max())),
                            vmax=max(abs(_Wc1.min()), abs(_Wc1.max())))
        _axes[0, _f].set_title(f"核{_f}：左右差 {_lr:+.1f}\n上下差 {_ud:+.1f}", fontsize=9)
        _vmax = max(np.abs(_r).max(), 1e-9)
        _axes[1, _f].imshow(_r, cmap="RdBu_r", vmin=-_vmax, vmax=_vmax)
        _axes[1, _f].set_title(f"核{_f} 的响应图", fontsize=9)
    for _ax in _axes.ravel():
        _ax.axis("off")
    _fig.suptitle("训练后的 conv1：六把学出来的尺子，六种指纹（输入 = 测试集 #0）", fontsize=12)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 4 · 分类别战报 + 错案画廊：LeNet 赢在哪、输在哪

        公平起见，先把 25 课的 MLP 原配置复刻训练一遍（同
        样的 50000/10000 划分、Adam、batch 64；复切实测
        97.75%，与 25 课战报 97.78% 在浮动范围内一致），
        然后逐类对比。

        写作时实测的分类别战报（错分个数）：

        | 数字 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 合计 |
        |---|---|---|---|---|---|---|---|---|---|---|---|
        | MLP | 14 | 14 | 19 | 28 | 22 | 24 | 15 | 18 | 38 | 33 | **225** |
        | LeNet | 9 | **2** | 12 | 32 | 7 | 6 | 25 | 10 | **18** | 28 | **149** |

        三条读法：

        - **大赎回**：8（38→18，两个圈的复杂局部结构）、
          5（24→6）、4（22→7）、1（14→2）——卷积特征
          最值钱的战场
        - **小退步**：3（28→32）、6（15→25）——LeNet 不是
          逐类碾压，个别字反而更难伺候；但总分 98.5% vs
          97.8%，总错分 **−34%** 才是硬道理
        - **最难的字换了**：MLP 的老大难是 8 和 9；LeNet 的
          老大难是 3、9、6——全卷积之后剩下的错案，多是
          "烂字"（看错案画廊：残缺、粘连、人眼都悬）
        """
    )
    return


@app.cell
def _(X_mnist, X_te30, lenet_net, np, plt, y_mnist, y_te30):
    # 演示 4 · MLP 复刻 + 分类别对比 + LeNet 错案画廊
    _rs = np.random.default_rng(42)                     # 与 LeNet 完全相同的划分
    _idx = _rs.permutation(len(X_mnist))
    _Xa, _ya = X_mnist[_idx], y_mnist[_idx]
    _Xtr, _ytr = _Xa[:50000], _ya[:50000]

    _ri = np.random.default_rng(0)                      # 25 课 MLP 原配方
    _params = [_ri.normal(0, 1, (784, 128)) * np.sqrt(2 / 784), np.zeros(128),
               _ri.normal(0, 1, (128, 10)) * np.sqrt(2 / 128), np.zeros(10)]
    _m = [np.zeros_like(_p) for _p in _params]
    _s = [np.zeros_like(_p) for _p in _params]
    _r = np.random.default_rng(1)
    _step = 0
    for _ep in range(12):
        _order = _r.permutation(len(_Xtr))
        for _st in range(0, len(_Xtr), 64):
            _ib = _order[_st:_st + 64]
            _xb, _yb = _Xtr[_ib], _ytr[_ib]
            _W1, _b1, _W2, _b2 = _params
            _Z1 = _xb @ _W1 + _b1
            _H = np.maximum(0, _Z1)
            _logits = _H @ _W2 + _b2
            _e = np.exp(_logits - _logits.max(1, keepdims=True))
            _P = _e / _e.sum(1, keepdims=True)
            _dl = _P.copy()
            _dl[np.arange(len(_yb)), _yb] -= 1
            _dl /= len(_yb)
            _dW2, _db2 = _H.T @ _dl, _dl.sum(0)
            _dZ1 = (_dl @ _W2.T) * (_Z1 > 0)
            _dW1, _db1 = _xb.T @ _dZ1, _dZ1.sum(0)
            _grads = [_dW1, _db1, _dW2, _db2]
            _step += 1
            for _i in range(4):
                _m[_i] = 0.9 * _m[_i] + 0.1 * _grads[_i]
                _s[_i] = 0.999 * _s[_i] + 0.001 * _grads[_i] ** 2
                _params[_i] -= 0.001 * (_m[_i] / (1 - 0.9 ** _step)) / \
                    (np.sqrt(_s[_i] / (1 - 0.999 ** _step)) + 1e-8)
    _pred_mlp = np.argmax(np.maximum(0, X_te30 @ _params[0] + _params[1])
                          @ _params[2] + _params[3], 1)
    _acc_mlp = float(np.mean(_pred_mlp == y_te30))
    _err_mlp = [int((_pred_mlp[y_te30 == d] != d).sum()) for d in range(10)]

    _pred_lenet = lenet_net.predict(X_te30.reshape(-1, 1, 28, 28))
    _err_lenet = [int((_pred_lenet[y_te30 == d] != d).sum()) for d in range(10)]
    print(f"MLP 复刻：测试集 {_acc_mlp:.4f}（总错分 {sum(_err_mlp)}）")
    print(f"LeNet-5：测试集 {float(np.mean(_pred_lenet == y_te30)):.4f}（总错分 {sum(_err_lenet)}）")
    print("每类错分 MLP  :", _err_mlp)
    print("每类错分 LeNet:", _err_lenet)

    _fig, _axes = plt.subplots(1, 2, figsize=(13.0, 4.4),
                               gridspec_kw={"width_ratios": [1.3, 1]})
    _x = np.arange(10)
    _axes[0].bar(_x - 0.2, _err_mlp, 0.4, color="tab:red", label="MLP（97.8%）")
    _axes[0].bar(_x + 0.2, _err_lenet, 0.4, color="tab:blue", label="LeNet-5（98.5%）")
    _axes[0].set_xticks(_x)
    _axes[0].set_xlabel("数字类别")
    _axes[0].set_ylabel("测试集错分个数")
    _axes[0].set_title("分类别战报：8/5/4/1 大赎回，3/6 小退步", fontsize=11)
    _axes[0].legend(fontsize=9)
    _axes[0].grid(True, axis="y", alpha=0.3)

    _wrong = np.where(_pred_lenet != y_te30)[0][:8]     # 错案画廊
    for _k, _wi in enumerate(_wrong):
        _axes[1].imshow(X_te30[_wi].reshape(28, 28), cmap="gray_r",
                        extent=[_k * 1.1, _k * 1.1 + 1, 0, 1])
    _axes[1].set_xlim(0, 8.8)
    _axes[1].set_ylim(0, 1)
    _axes[1].set_yticks([])
    _axes[1].set_xticks([_k * 1.1 + 0.5 for _k in range(8)])
    _axes[1].set_xticklabels([f"真{int(y_te30[_w])}\n认{int(_pred_lenet[_w])}" for _w in _wrong],
                             fontsize=8)
    _axes[1].set_title("LeNet 的错案画廊（前 8 案）：\n残缺、粘连、连人眼都要眯一下", fontsize=10)
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
        ### 交互实验 1 · 逐层 CT：给测试样张做全身扫描

        拖动样张编号（0~999），用**训练好的** LeNet-5 给它做
        逐层透视：conv1 的六种指纹 → 池化后的粗指纹 →
        conv2 的十六种"部件指纹" → 最后一根 softmax 概率条。

        玩法建议：

        - 换几张不同数字，看 conv1 指纹怎么变、conv2 哪些
          格子常亮
        - 找一张预测错的样张（999 号以内藏着几个），看它的
          概率条有多纠结
        - 对比 conv1 和 pool1：池化后的指纹更粗、更"神似"
          ——位置细节被扔掉，留下的是"这附近有没有"
        """
    )
    return


@app.cell
def _(mo):
    # 交互控制：测试样张编号
    sample_slider = mo.ui.slider(0, 999, value=0, step=1, label="🖼️ 测试样张编号（0~999）")
    mo.hstack([sample_slider])
    return (sample_slider,)


@app.cell
def _(X_te30, conv_forward30, lenet_net, mo, np, plt, pool2_fwd30, sample_slider, y_te30):
    # ===== 交互实验 1 · 逐层 CT（响应单元）=====
    _si = sample_slider.value
    _x = X_te30[_si].reshape(1, 1, 28, 28)
    _p = lenet_net.p

    _z1, _ = conv_forward30(_x, _p["Wc1"], _p["bc1"])
    _h1 = np.maximum(_z1, 0)
    _h1p, _ = pool2_fwd30(_h1)
    _z3, _ = conv_forward30(_h1p, _p["Wc3"], _p["bc3"])
    _h3 = np.maximum(_z3, 0)
    _h3p, _ = pool2_fwd30(_h3)
    _flat = _h3p.reshape(-1)
    _h5 = np.maximum(_flat @ _p["Wf5"].T + _p["bf5"], 0)
    _h6 = np.maximum(_h5 @ _p["Wf6"].T + _p["bf6"], 0)
    _logits = _h6 @ _p["Wout"].T + _p["bout"]
    _e = np.exp(_logits - _logits.max())
    _prob = _e / _e.sum()
    _pred = int(np.argmax(_prob))

    def montage(maps, n_row, n_col, gap=1):
        """把 (C,h,w) 的多张图拼成一张大图（拼贴画）。"""
        _C, _h, _w = maps.shape
        _big = np.zeros((n_row * _h + (n_row - 1) * gap,
                         n_col * _w + (n_col - 1) * gap))
        for _c in range(min(_C, n_row * n_col)):
            _r, _cc = divmod(_c, n_col)
            _big[_r*(_h+gap):_r*(_h+gap)+_h, _cc*(_w+gap):_cc*(_w+gap)+_w] = maps[_c]
        return _big

    _fig = plt.figure(figsize=(13.4, 4.6))
    _gs = _fig.add_gridspec(2, 5, height_ratios=[1, 1], hspace=0.55, wspace=0.25)
    _ax0 = _fig.add_subplot(_gs[:, 0])
    _ax0.imshow(_x[0, 0], cmap="gray_r")
    _ax0.set_title(f"输入（真值 {int(y_te30[_si])}）", fontsize=11)
    _ax0.axis("off")
    _ax1 = _fig.add_subplot(_gs[:, 1])
    _ax1.imshow(montage(_h1[0], 2, 3), cmap="viridis")
    _ax1.set_title("conv1+ReLU：6 张指纹\n（24×24 ×6）", fontsize=9.5)
    _ax1.axis("off")
    _ax2 = _fig.add_subplot(_gs[:, 2])
    _ax2.imshow(montage(_h1p[0], 2, 3), cmap="viridis")
    _ax2.set_title("pool1：粗指纹\n（12×12 ×6）", fontsize=9.5)
    _ax2.axis("off")
    _ax3 = _fig.add_subplot(_gs[:, 3])
    _ax3.imshow(montage(_h3[0], 4, 4), cmap="viridis")
    _ax3.set_title("conv2+ReLU：部件指纹\n（8×8 ×16）", fontsize=9.5)
    _ax3.axis("off")
    _ax4 = _fig.add_subplot(_gs[:, 4])
    _ybars = np.arange(10)
    _ax4.barh(_ybars, _prob, color=["tab:red" if _i == _pred else "tab:gray" for _i in range(10)])
    _ax4.set_yticks(_ybars)
    _ax4.invert_yaxis()
    _ax4.set_xlim(0, 1.05)
    _ax4.set_title(f"softmax：预测 {_pred}\n（置信 {_prob[_pred]:.1%}）", fontsize=10)
    _ax4.grid(True, axis="x", alpha=0.3)
    _fig.suptitle(f"测试样张 #{_si} 的逐层 CT 扫描", fontsize=12)

    if _pred == int(y_te30[_si]):
        _second = float(np.sort(_prob)[-2])
        _verdict = (f"✅ 预测 **{_pred}** 正确，置信 {_prob[_pred]:.1%}——"
                    f"第二名只拿到 {_second:.1%}")
    else:
        _verdict = (f"❌ 预测 {_pred}，真值 {int(y_te30[_si])}——烂字翻车的现场，"
                    f"置信 {_prob[_pred]:.1%}（真值类别只拿到 {_prob[int(y_te30[_si])]:.1%}）")

    mo.vstack([
        _fig,
        mo.callout(mo.md(_verdict), kind="info"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✏️ 动手练习（每课必做，本课 3 题）": mo.md(
                """
                **练习 1 · 形状表填空**：把 LeNet-5 数据流里四个
                空缺的形状数字填上（S2 输出的高、C3 输出的
                通道数、flatten 的长度、F6 的输出宽度）。判词
                会用真实前向的形状对账。

                **练习 2 · 参数账手算**：C1 的 156 已给出，笔算
                C3、F5、OUT 三层的参数量和全网总账（4 个空）。

                **练习 3 · 架构侦探 · VGG 迷你版**：一个新架构
                `conv3×3(P=1)×4 → pool2 → conv3×3(P=1)×8 →
                pool2 → fc10`。用 27 课的公式笔算：conv1 输出
                边长、pool1 输出形状的通道数、conv2 输出边长、
                flatten 长度、fc 的参数量（5 个空）。这是 27~29
                课的综合测——填完判词会真的搭一遍这个网络对账。

                规则不变：先把 `None` 换成你的答案再运行；写完再看参考答案。
                """
            )
        }
    )
    return


@app.cell
def _(LeNet5, mo, np):
    # ===== 练习 1 · 形状表填空 =====
    _net = LeNet5(seed=0)
    _xd = np.random.default_rng(5).normal(size=(2, 1, 28, 28))
    _c = {}
    _logits = _net.forward(_xd, _c)
    print("真实形状账（对账用）：")
    print(f"  C1 输出 {_c['h1'].shape} → S2 输出 {_c['h1p'].shape}")
    print(f"  C3 输出 {_c['h3'].shape} → S4 输出 {_c['h3p'].shape}")
    print(f"  flatten {_c['flat'].shape} → F5 → F6 → OUT {_logits.shape}")

    ans_s2_h_ex1 = None                       # ← ① S2 输出的高（和宽）
    ans_c3_c_ex1 = None                       # ← ② C3 输出的通道数
    ans_flat_ex1 = None                       # ← ③ flatten 后的长度
    ans_f6_ex1 = None                         # ← ④ F6 输出的宽度
    return (ans_c3_c_ex1, ans_f6_ex1, ans_flat_ex1, ans_s2_h_ex1)


@app.cell
def _(ans_c3_c_ex1, ans_f6_ex1, ans_flat_ex1, ans_s2_h_ex1, mo):
    # ===== 练习 1 · 判词 =====
    _truth = {"① S2 高": (ans_s2_h_ex1, 12), "② C3 通道": (ans_c3_c_ex1, 16),
              "③ flatten 长": (ans_flat_ex1, 256), "④ F6 宽": (ans_f6_ex1, 84)}
    if any(_v[0] is None for _v in _truth.values()):
        _msg = "⏳ 四个形状都算出来再填（提示：conv5 无填充每边缩 4，pool2 减半）"
    else:
        _wrong = [k for k, (a, t) in _truth.items() if a != t]
        if not _wrong:
            _msg = ("✅ 四个形状全对：24×24×6 → pool → 12×12×6 → conv5 → 8×8×16 → "
                    "pool → 4×4×16 → flatten 256 → 120 → 84 → 10")
        else:
            _hint = {"① S2 高": "C1 是 28−5+1=24，pool2 减半",
                     "② C3 通道": "C3 有 16 个核 → 16 通道",
                     "③ flatten 长": "S4 是 4×4×16 = 256",
                     "④ F6 宽": "F6 的输出宽度写在架构里是 84"}
            _msg = "❌ 还差：" + "；".join(f"{k}（{_hint[k]}）" for k in _wrong)
    mo.md(_msg)
    return


@app.cell
def _(mo):
    # ===== 练习 2 · 参数账手算 =====
    print("笔算三层的参数量（权重 + 偏置）：")
    print("  C1（已给）：5×5×1×6 + 6 = 156")
    print("  C3：5×5×6×16 + 16 = ?")
    print("  F5：256×120 + 120 = ?")
    print("  OUT：84×10 + 10 = ?")
    print("  全网总账 = ?")

    ans_c3_ex2 = None                         # ← ① C3 参数量
    ans_f5_ex2 = None                         # ← ② F5 参数量
    ans_out_ex2 = None                        # ← ③ OUT 参数量
    ans_total_ex2 = None                      # ← ④ 全网总账
    return (ans_c3_ex2, ans_f5_ex2, ans_out_ex2, ans_total_ex2)


@app.cell
def _(ans_c3_ex2, ans_f5_ex2, ans_out_ex2, ans_total_ex2, mo):
    # ===== 练习 2 · 判词 =====
    _truth = {"① C3": (ans_c3_ex2, 2416), "② F5": (ans_f5_ex2, 30840),
              "③ OUT": (ans_out_ex2, 850), "④ 总账": (ans_total_ex2, 44426)}
    if any(_v[0] is None for _v in _truth.values()):
        _msg = "⏳ 四个数都算出来再填（别忘了偏置，别忘了个十百位的零）"
    else:
        _wrong = [k for k, (a, t) in _truth.items() if a != t]
        if not _wrong:
            _msg = ("✅ 账目分毫不差：156 + 2,416 + 30,840 + 10,164 + 850 = 44,426。"
                    "对照 MLP 的 101,770：少 56% 的参数打赢 0.73 个百分点——"
                    "这就是结构先验的价格优势")
        else:
            _msg = (f"❌ 还差：{'、'.join(_wrong)}。C3 = 25×6×16+16；"
                    "F5 = 256×120+120；OUT = 84×10+10；总账把 C1 的 156 和 "
                    "F6 的 10,164 也加进去")
    mo.md(_msg)
    return


@app.cell
def _(mo, np):
    # ===== 练习 3 · 架构侦探（VGG 迷你版）=====
    print("新架构：conv3×3(P=1)×4核 → pool2 → conv3×3(P=1)×8核 → pool2 → flatten → fc10")
    print("笔算五个空（27 课公式 out = (H+2P−K)/S + 1）")

    ans_conv1_ex3 = None                      # ← ① conv1 输出边长（28×28 进）
    ans_pool1c_ex3 = None                     # ← ② pool1 输出的通道数
    ans_conv2_ex3 = None                      # ← ③ conv2 输出边长（14×14 进）
    ans_flat_ex3 = None                       # ← ④ flatten 长度
    ans_fc_ex3 = None                         # ← ⑤ fc10 的参数量
    return (ans_conv1_ex3, ans_conv2_ex3, ans_fc_ex3, ans_flat_ex3, ans_pool1c_ex3)


@app.cell
def _(ans_conv1_ex3, ans_conv2_ex3, ans_fc_ex3, ans_flat_ex3, ans_pool1c_ex3, mo, np, sliding_window_view):
    # ===== 练习 3 · 判词：真的搭一遍对账 =====
    _rng = np.random.default_rng(6)
    _x = _rng.normal(size=(2, 1, 28, 28))
    _x = np.pad(_x, ((0, 0), (0, 0), (1, 1), (1, 1)))                     # P=1
    _win = sliding_window_view(_x, (3, 3), axis=(2, 3))
    _W1 = _rng.normal(size=(4, 1, 3, 3)) * 0.1
    _h1 = np.maximum(np.einsum("ncijkl,fckl->nfij", _win, _W1), 0)
    _p1 = _h1.reshape(2, 4, 14, 2, 14, 2).max(axis=(3, 5))          # 2×2 max 池化（28→14）
    _x2 = np.pad(_p1, ((0, 0), (0, 0), (1, 1), (1, 1)))
    _win2 = sliding_window_view(_x2, (3, 3), axis=(2, 3))
    _W2 = _rng.normal(size=(8, 4, 3, 3)) * 0.1
    _h2 = np.maximum(np.einsum("ncijkl,fckl->nfij", _win2, _W2), 0)
    _p2 = _h2.reshape(2, 8, 7, 2, 7, 2).max(axis=(3, 5))            # 14→7
    _flat = _p2.reshape(2, -1)
    _fc_params = _flat.shape[1] * 10 + 10
    print(f"搭出来的真实形状：conv1 → {_h1.shape[2]}｜pool1 通道 {_p1.shape[1]}｜"
          f"conv2 → {_h2.shape[2]}｜flatten {_flat.shape[1]}｜fc 参数 {_fc_params}")

    _truth = {"① conv1 边长": (ans_conv1_ex3, _h1.shape[2]),
              "② pool1 通道": (ans_pool1c_ex3, _p1.shape[1]),
              "③ conv2 边长": (ans_conv2_ex3, _h2.shape[2]),
              "④ flatten 长": (ans_flat_ex3, _flat.shape[1]),
              "⑤ fc 参数": (ans_fc_ex3, _fc_params)}
    if any(_v[0] is None for _v in _truth.values()):
        _msg = "⏳ 五个空都填上再运行（same 配方：P=(K−1)/2=1，输出边长不变）"
    else:
        _wrong = [k for k, (a, t) in _truth.items() if a != t]
        if not _wrong:
            _msg = (f"✅ 五空全对！这个 VGG 迷你版：28→28→14→14→7→7，flatten "
                    f"{_flat.shape[1]}，fc 只有 {_fc_params:,} 参数——"
                    "same 配方 + 小核堆叠，就是 31 课 VGG 的路数")
        else:
            _hint = {"① conv1 边长": "(28+2−3)/1+1",
                     "② pool1 通道": "pool 不改通道数，通道 = conv1 的核数",
                     "③ conv2 边长": "(14+2−3)/1+1",
                     "④ flatten 长": "pool2 后是 7×7×8",
                     "⑤ fc 参数": "flatten 长 ×10 + 10"}
            _msg = "❌ 还差：" + "；".join(f"{k}（{_hint[k]}）" for k in _wrong)
    mo.md(_msg)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 1 参考答案（先自己写过再看！）": mo.md(
                """
                **12、16、256、84。**

                - S2 高：C1 输出 28−5+1 = 24，pool2 减半 → **12**
                - C3 通道：16 个核 → **16** 张响应图
                - flatten：S4 是 4×4×16 = **256**
                - F6 宽：**84**（LeNet-5 的原设计数字，来自
                  1989 年版字符识别器的输出层——历史遗留，无深意）

                形状账是 CNN 的会计学：**每一层的输出形状由
                三个数字完全决定**（输入形状、K/S/P、核数）。
                能白纸画出这张表，任何框架的报错都难不倒你。
                """
            ),
            "✅ 练习 2 参考答案（先自己写过再看！）": mo.md(
                """
                **C3 = 2,416，F5 = 30,840，OUT = 850，总账 44,426。**

                - C3：每个核是 5×5×6 的三维块 = 150 个权重，
                  16 个核 = 2,400，加 16 个偏置 = **2,416**
                - F5：256×120 + 120 = **30,840**
                - OUT：84×10 + 10 = **850**
                - 总账：156 + 2,416 + 30,840 + 10,164 + 850 =
                  **44,426**

                一个值得记住的比例：全连接三层共 41,854 个
                参数（94%），卷积两层只有 2,572 个（6%）——
                **卷积层是性价比之王，全连接是开销大户**。
                现代网络（31 课）干脆把全连接砍到只剩一层，
                甚至全用卷积。
                """
            ),
            "✅ 练习 3 参考答案（先自己写过再看！）": mo.md(
                """
                **28、4、14、392、3,930。**

                - ① conv1：(28+2−3)/1+1 = **28**——same 配方
                  （P=(K−1)/2=1）边长不变
                - ② pool1 通道：池化不碰通道，= conv1 的核数
                  = **4**
                - ③ conv2：(14+2−3)/1+1 = **14**
                - ④ flatten：pool2 后 7×7×8 = **392**
                - ⑤ fc：392×10+10 = **3,930**

                这就是 **VGG 的配方**：小核（3×3）+ same 填充
                + 池化减半，一层层堆。对比 LeNet 的大核（5×5
                无填充）：每站尺寸账完全不同（28→28→14→14→7
                vs 28→24→12→8→4），但思想同源。31 课把这些
                配方的历史演进一次讲透。
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
                1. **LeNet-5 七站流水线**：28×28×1 → C1(6@24²) →
                   S2(6@12²) → C3(16@8²) → S4(16@4²) → 256 →
                   120 → 84 → 10；卷积提特征、池化压尺寸、
                   全连接判决
                2. **参数经济学**：44,426 vs MLP 101,770（−56%）；
                   卷积层只占 6% 参数干了全部特征提取；
                   全连接占 94% 却只是判决席——结构先验的价值
                3. **训练完全体拼装**：He 初始化 + Adam(0.002) +
                   batch 128 洗牌 + 稳定 softmax + 早停 patience 3；
                   每个决定都有前课出处，没有一项是新的
                4. **战报**：epoch 1 即 97.3%；val 峰值 98.50%
                   （epoch 6）→ 早停 epoch 9 → 回滚；测试集
                   **98.51%**，总错分 222 → **149**（−34%）
                5. **分类别战况**：8（38→18）、5（24→6）、4（22→7）
                   大赎回；3（28→32）、6（15→25）小退步——CNN
                   赢在局部结构复杂的字，不是逐类碾压
                6. **学出来的滤波器**：conv1 六核自发长出方向
                   分工（左右差 −1.9 ~ +2.7）——没人教过边缘，
                   是损失逼出来的；与 LeCun 1998 论文的
                   第一层同款气质
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
                ## 🎓 下一站

                98.51% 的 LeNet-5 打赢了 MLP，第 5 部分的技术
                故事到此完整。但历史才讲到 1998 年——之后 CNN
                沉寂了十四年，然后在 2012 年一夜之间改天换地。

                最后一课不讲新代码，讲**这十四年和之后发生了
                什么**：为什么深度学习在 2012 年爆发？残差连接
                解决了什么难题？那些几十层、上百层的网络是
                怎么堆起来的？
                """
            ),
            mo.callout(
                mo.md(
                    """
                    **下一课预告** — `31_架构演进`（第 5 部分收官）：

                    从 **AlexNet**（2012，三件套：ReLU + Dropout +
                    GPU）到 **VGG**（"两个 3×3 顶一个 5×5"的哲学）
                    到 **ResNet**（"抄近路"的残差连接，把网络
                    堆到 152 层）——外加一场动手实验：亲手看
                    **没有残差的深网络为什么越深越差**，以及
                    残差怎么救场。

                    第 5 部分通关后，图像的世界告一段落；第 6
                    部分转向**序列**——文字、时间、语言。

                    📜 去看 CNN 的编年史！
                    """
                ),
                kind="success",
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
