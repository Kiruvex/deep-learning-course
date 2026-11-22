import marimo

__generated_with = "0.25.2"
app = marimo.App(app_title="29 · NumPy 实现卷积层", width="medium")


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
        # 29 · NumPy 实现卷积层：im2col 与反向传播

        **从零到 Transformer · 手写深度学习** ｜ 第 5 部分 · 卷积神经网络（4/6）
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📖 本课目标（先读我）": mo.md(
                """
                **第 5 部分最硬核的一课：把卷积变成可训练的层。**

                到现在为止，卷积核都是你手写的（竖边核、横边核）。
                手写的核能找边，但认不出"这是 0 的边还是 6 的边"。
                要让网络**自己学核**，就必须回答 22~23 课问过
                MLP 的那个问题：**梯度怎么流过卷积层？**

                今天的主角是 **im2col**（image to column）——
                一个朴素的搬运技巧：把每个窗口的小块**摊开、
                摆齐**，卷积就变成了**一次矩阵乘法**。它同时
                解决两个问题：

                - **速度**：四重 Python 循环 → BLAS 矩阵乘，
                  实测加速两百多倍（演示 1）
                - **梯度**：前向是矩阵乘，反向就套 23 课的四条
                  矩阵梯度——dW、dX 的推导一气呵成

                学完你将能：

                - 亲手实现 **im2col / conv_forward / conv_backward
                  / col2im**，并说出每步的形状账
                - 用 22/23 课的**数值梯度对账**验证全部反向代码
                  （本课实测相对误差 10⁻¹⁰ 上下）
                - 写出**池化的反向**：max 路由——梯度只给每块的
                  赢家
                - 把 conv→relu→pool→flatten→fc 按 LeNet-5 的
                  形状串成流水线（30 课的底盘）

                前置：23 课的四条矩阵梯度与形状检查法（今天的
                发动机）、12 课"矩阵乘 = 万能线性算子"（im2col
                的哲学依据）、26~28 课的卷积/池化语义。

                代码量是全课程之最，但每一行都是旧知识的新组装。
                慢慢来，跑通一个是一个。
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📦 概念 1/4 · im2col：把卷积变成矩阵乘": mo.md(
                """
                26 课的卷积是四重循环：批 × 核 × 行 × 列，每个
                位置做一次小内积。Python 循环慢得出名——但循环
                里干的事**本质上是矩阵乘**（12 课：任何"加权
                求和"的线性操作都是矩阵乘法）。im2col 的思路就
                一句话：

                > **与其在循环里找数据，不如先把所有窗口搬到
                > 矩阵里，然后一次乘完。**

                具体搬法：输入 (N, C, H, W)，核 K×K、步长 S、
                填充 P，输出 OH×OW 个位置——为**每个输出位置**
                准备一**列**：把该位置窗口里的 C×K×K 个数
                （按通道、行、列排好）竖着摆进去：

                ```
                cols：(N, C·K·K, OH·OW)     每列 = 一个窗口摊平
                W2  ：(F, C·K·K)            每行 = 一个核摊平
                Z   = W2 @ cols + b         一次矩阵乘 + 广播偏置
                Z   → reshape (N, F, OH, OW)
                ```

                代价是**内存**：窗口互相重叠，数据被复制多份
                （27 课的覆盖计数：中心像素进 9 个窗口 = 被复制
                9 份）。演示 3 和交互实验 1 会算这笔账：K=3 时
                单张图 6,084 个元素（47.5 KB），K=7 涨到 23,716
                （185.3 KB）——**用内存换速度**，深度学习最常见
                的交易。
                """
            ),
            "🔧 概念 2/4 · 完整前向：批量、多通道、偏置各就各位": mo.md(
                """
                把 26~27 课的全部语义装进一个函数，形状账先对
                齐再写码（23 课的纪律）：

                ```
                x   : (N, C, H, W)        N 张 C 通道的图
                W   : (F, C, K, K)        F 个核，每个都是三维块
                b   : (F,)                每张输出图一个偏置
                out : (N, F, OH, OW)      OH = (H + 2P − K)/S + 1
                ```

                前向四步：**pad → im2col → reshape 核 → 矩阵乘
                + 偏置**。其中偏置靠广播（06 课）：`b[:, None]`
                加到每个位置上；`W.reshape(F, C·K·K)` 只是把
                四维核"摊平"成 W2 的行（07 课 reshape 不复制
                数据）。

                为什么执着于矩阵乘？因为 NumPy 的 `@` 背后是
                几十年调优的 BLAS 库：缓存友好的分块、SIMD
                指令、多线程。**把计算重排成矩阵乘，就等于免费
                雇了一支施工队。** PyTorch、cuDNN 的卷积底层
                干的就是同一件事（格式细节不同，思想一致）。
                """
            ),
            "🔁 概念 3/4 · 反向传播：dW 是回收，dX 是散射": mo.md(
                """
                前向是 `Z = W2 @ cols + b`，反向全部套 23 课的
                矩阵梯度。设上游送来 `dout`（N, F, OH, OW），
                先摊平成 `dout2`（N, F, OH·OW）：

                **① db（最简单）**：偏置广播到每个位置 →
                `db = dout 对 (N, OH, OW) 三个轴求和`，剩 (F,)。

                **② dW（误差与窗口的"相关"）**：Z[f, p] =
                Σ_k W2[f, k]·cols[k, p]，所以
                `dW2[f, k] = Σ dout2[f, p] · cols[k, p]`——
                **每个窗口位置的误差，乘上窗口内容，再对全部
                位置求和**。写成 einsum 就是
                `dW2 = einsum("nfp,nkp->fk", dout2, cols)`。
                白话：**哪个窗口长什么样、误差又大，核就往哪个
                方向改**——这就是"学核"。

                **③ dX（散射回填）**：对 cols 的梯度
                `dcols = W2ᵀ @ dout2`（同一条乘法的另一个方向）。
                但 cols 是"摊开的副本"——同一个输入像素躺在
                好几列里（覆盖计数！），它的梯度 = **所有包含
                它的窗口列回流的梯度之和**。把 dcols 散射回
                (N, C, H, W) 的操作叫 **col2im**：逐窗口位置
                把 K×K 的梯度块 **累加** 回原图。累加是关键——
                中心像素被 9 个窗口看过，它的 dX 是 9 份回流
                之和。

                形状检查法全程护航：dW 和 W 同形、db 和 b 同形、
                dX 和 x 同形，差一个字母都能立刻发现。
                """
            ),
            "✅ 概念 4/4 · 对账方法论 + 池化的 max 路由": mo.md(
                """
                **数值对账**（22/23 课的家法）：对任意参数 w，
                用双侧差分 `(L(w+ε) − L(w−ε)) / 2ε 算数值梯度，
                和解析梯度比相对误差。ε=10⁻⁵ 时机器精度给到
                ~10⁻¹⁰ 量级；本课实测 dW/db/dX 三项全部
                **10⁻¹¹**——四个九的对账通过，反向代码可信。

                **池化的反向**（max 版）：前向每块只留赢家，
                那么梯度也**只归赢家**——对每个 2×2 块找
                argmax，把 dout[i,j] 塞回赢家的位置，其余三格
                梯度为零。这就是 **max 路由**（winner-takes-all）。

                两个细节值得记：

                - 梯度总量守恒：`sum(dX) == sum(dout)`——
                  池化不产生也不消灭梯度，只做搬运
                - **平顶陷阱**：块内最大值并列时，argmax 只选
                  一个（数值差分在这种点不稳定，属于已知的
                  数学边界，工程上忽略）

                avg 池化的反向更民主：每块 4 格各拿 1/4。
                dropout 的 mask（25 课）记路径、池化的 argmax
                记赢家——**反向传播的本质是"前向记账、反向
                照账还钱"**，这句话你现在应该有体感了。
                """
            ),
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 🎬 演示
        """
    )
    return


@app.cell
def _(mo, np):
    # MNIST 加载器（与 25~28 课同款）
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

    X_mnist, y_mnist, X_te29, y_te29 = load_mnist()
    print("训练集:", X_mnist.shape, " 测试集:", X_te29.shape)
    return (X_mnist, X_te29, y_mnist, y_te29)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 1 · 速度对账：四重循环 vs 一次矩阵乘

        同一个任务：batch 64 张 28×28、8 个 3×3 核，跑两版
        实现。左：26 课的四重循环版（conv_naive）；右：im2col
        版。计时结果随机器不同，**量级不会变：百倍级**。

        （本课写作时实测约 1230 ms → 6 ms，**200 倍出头**；
        你运行时打印的是你机器的现场数字，倍数量级相同就是健康。）

        加速的来源不是算法变巧——计算量几乎一样——而是
        **把工作交给了 BLAS**：矩阵乘的内层循环在 C/Fortran
        里以缓存友好的方式跑满 CPU。这就是 im2col 在所有
        深度学习框架里当家几十年的原因。
        """
    )
    return


@app.cell
def _(X_mnist, np):
    # 演示 1 · im2col + conv_forward + 计时
    def im2col(x, K, stride=1, pad=0):
        """x: (N,C,H,W) → cols: (N, C·K·K, OH·OW)。每个输出位置一列：窗口摊平。"""
        _N, _C, _H, _W = x.shape
        if pad:
            x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        _Hp, _Wp = x.shape[2], x.shape[3]
        _oh, _ow = (_Hp - K) // stride + 1, (_Wp - K) // stride + 1
        _cols = np.zeros((_N, _C * K * K, _oh * _ow))
        for _i in range(_oh):
            for _j in range(_ow):
                _patch = x[:, :, _i*stride:_i*stride+K, _j*stride:_j*stride+K]   # (N,C,K,K)
                _cols[:, :, _i*_ow + _j] = _patch.reshape(_N, _C * K * K)        # 摊平摆列
        return _cols

    def conv_forward(x, W, b, stride=1, pad=0):
        """x:(N,C,H,W) W:(F,C,K,K) b:(F,) → (N,F,OH,OW)。im2col 版。"""
        _N, _C, _H, _W = x.shape
        _F, _, _K, _ = W.shape
        _cols = im2col(x, _K, stride, pad)
        _W2 = W.reshape(_F, _C * _K * _K)                 # 核摊平成行（07 课）
        _Z = _W2 @ _cols + b[:, None]                     # 一次矩阵乘 + 偏置广播（06/12 课）
        _oh = (_H + 2*pad - _K) // stride + 1
        _ow = (_W + 2*pad - _K) // stride + 1
        return _Z.reshape(_N, _F, _oh, _ow)

    def conv_naive29(x, W, b, stride=1, pad=0):
        """四重循环版（26 课的血统），作为正确性与速度的对照。"""
        _N, _C, _H, _W = x.shape
        _F, _, _K, _ = W.shape
        _xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad))) if pad else x
        _oh = (_xp.shape[2] - _K) // stride + 1
        _ow = (_xp.shape[3] - _K) // stride + 1
        _out = np.zeros((_N, _F, _oh, _ow))
        for _n in range(_N):
            for _f in range(_F):
                for _i in range(_oh):
                    for _j in range(_ow):
                        _out[_n, _f, _i, _j] = np.sum(
                            _xp[_n, :, _i*stride:_i*stride+_K, _j*stride:_j*stride+_K] * W[_f]) + b[_f]
        return _out

    import time as _time
    _rng = np.random.default_rng(0)
    _xb = X_mnist[:64].reshape(64, 1, 28, 28)             # batch 64
    _Wb = _rng.normal(size=(8, 1, 3, 3)) * 0.5
    _bb = np.zeros(8)

    _t0 = _time.time()
    _o_naive = conv_naive29(_xb, _Wb, _bb)
    _t_naive = _time.time() - _t0
    _t0 = _time.time()
    _o_fast = conv_forward(_xb, _Wb, _bb)
    _t_fast = _time.time() - _t0

    print(f"naive 四重循环：{_t_naive*1000:7.1f} ms")
    print(f"im2col 矩阵乘：{_t_fast*1000:7.1f} ms   → 加速 {_t_naive/_t_fast:.0f} 倍")
    print(f"两版输出逐元素一致：{np.allclose(_o_naive, _o_fast, atol=1e-10)}")
    print(f"输出形状 {_o_fast.shape} = (N=64, F=8, OH=26, OW=26)")
    return (conv_forward, conv_naive29, im2col)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 2 · 反向传播 + 数值对账：10⁻¹¹ 的签字画押

        `conv_backward` 按概念 3 的三步走：db 求和、dW 用
        einsum、dX 先算 dcols 再 col2im 散射累加。

        对账用 22/23 课的家法：小随机数据（2×3×6×6 输入、
        4 个 3×3 核），损失取 `sum(out · dout)`，对 W、b、x
        逐个参数做双侧差分，与解析梯度比**相对误差**。

        本课写作时实测三项都在 10⁻¹⁰ ~ 10⁻¹¹（ε=10⁻⁵ 的理论天花板附近）。
        对账通过的那一刻，这个卷积层就"可训练"了——30 课
        直接把它挂进训练循环。
        """
    )
    return


@app.cell
def _(conv_forward, im2col, np):
    # 演示 2 · conv_backward + 数值梯度对账
    def conv_backward(dout, x, W, b, stride=1, pad=0):
        """dout:(N,F,OH,OW) → (dW, db, dx)。三步：求和、einsum、散射。"""
        _N, _C, _H, _W = x.shape
        _F, _, _K, _ = W.shape
        _oh, _ow = dout.shape[2], dout.shape[3]
        _cols = im2col(x, _K, stride, pad)                          # (N, CKK, OHW)
        _W2 = W.reshape(_F, _C * _K * _K)
        _dout2 = dout.reshape(_N, _F, _oh * _ow)                    # 摊平 (N,F,OHW)
        _db = dout.sum(axis=(0, 2, 3))                              # ① db：三轴求和
        _dW2 = np.einsum("nfp,nkp->fk", _dout2, _cols)              # ② dW：误差×窗口，对位置求和
        _dcols = np.einsum("fk,nfp->nkp", _W2, _dout2)              # ③ dX 前半：W 转置回流
        # col2im：把 dcols 散射回 (N,C,H+2P,W+2P)，重叠处累加
        _xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad))) if pad else x
        _dxp = np.zeros_like(_xp)
        for _i in range(_oh):
            for _j in range(_ow):
                _p = _i * _ow + _j
                _dxp[:, :, _i*stride:_i*stride+_K, _j*stride:_j*stride+_K] += \
                    _dcols[:, :, _p].reshape(_N, _C, _K, _K)        # += 是累加，不是覆盖！
        _dx = _dxp[:, :, pad:-pad, pad:-pad] if pad else _dxp       # 撕掉填充
        return _dW2.reshape(_F, _C, _K, _K), _db, _dx

    def num_grad(f, x, eps=1e-5):
        """双侧差分数值梯度（22 课家法）。f 是无参损失函数，x 是被扰动的数组。"""
        _g = np.zeros_like(x)
        _it = np.nditer(x, flags=["multi_index"])
        while not _it.finished:
            _ix = _it.multi_index
            _old = x[_ix]
            x[_ix] = _old + eps
            _fp = f()
            x[_ix] = _old - eps
            _fm = f()
            x[_ix] = _old
            _g[_ix] = (_fp - _fm) / (2 * eps)
            _it.iternext()
        return _g

    _rng = np.random.default_rng(1)
    x_ck = _rng.normal(size=(2, 3, 6, 6))
    W_ck = _rng.normal(size=(4, 3, 3, 3))
    b_ck = _rng.normal(size=(4,))
    dout_ck = _rng.normal(size=(2, 4, 4, 4))

    def _loss():
        return float(np.sum(conv_forward(x_ck, W_ck, b_ck) * dout_ck))

    dW_ck, db_ck, dx_ck = conv_backward(dout_ck, x_ck, W_ck, b_ck)
    gW_ck = num_grad(_loss, W_ck)
    gb_ck = num_grad(_loss, b_ck)
    gx_ck = num_grad(_loss, x_ck)

    def _relerr(a, b):
        return float(np.max(np.abs(a - b)) / (np.abs(b).max() + 1e-12))

    print(f"dW 相对误差：{_relerr(dW_ck, gW_ck):.2e}   （形状 {dW_ck.shape} ✓ 与 W 同形）")
    print(f"db 相对误差：{_relerr(db_ck, gb_ck):.2e}")
    print(f"dx 相对误差：{_relerr(dx_ck, gx_ck):.2e}   （形状 {dx_ck.shape} ✓ 与 x 同形）")
    print("三项全部 < 1e-8：对账通过，反向传播可信")
    return (conv_backward, num_grad)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 3 · 内存账 + LeNet-5 形状流水线：30 课的底盘

        上表：im2col 的内存账——核越大，窗口越大、位置越少，
        但**总账单调上涨**（重叠复制越来越多）。K=7 比 K=3
        多吃近 4 倍内存，这就是"大核贵"的隐藏成本。

        下表：把今天的三件套（conv / relu / pool）和 20 课的
        fc 串起来，过一遍 **LeNet-5（28×28 输入版）** 的完整
        形状账。参数总计 **44,426**——比 25 课的 MLP（101,770）
        少 56%，却将在 30 课拿到更高的准确率：**参数少而结构
        对，就是卷积的胜利**。

        > 冷知识：经典 LeNet-5 论文把输入 pad 到 32×32，第一
        > 层全连接是 400→120，总参数 61,706。我们直接吃 28×28，
        > flatten 后只有 256 个特征——工程上完全合法，30 课
        > 按我们的账本训练。
        """
    )
    return


@app.cell
def _(X_mnist, conv_forward, im2col, np):
    # 演示 3 · 内存账 + LeNet-5 形状流水线
    _x1 = X_mnist[:1].reshape(1, 1, 28, 28)
    print("== im2col 内存账（单张 28×28，stride=1，pad=0）==")
    for _K in [3, 5, 7]:
        _cols = im2col(_x1, _K)
        _oh = 28 - _K + 1
        print(f"  K={_K}: cols 形状 ({_K*_K}, {_oh*_oh})，共 {_K*_K*_oh*_oh:,} 元素"
              f"（{_K*_K*_oh*_oh*8/1024:.1f} KB，float64）——原始图只有 784 元素")

    print("\n== LeNet-5（28×28 版）形状流水线（batch 64，随机权重演示形状）==")
    _rng = np.random.default_rng(2)
    _x = X_mnist[:64].reshape(64, 1, 28, 28)
    Wc1 = _rng.normal(size=(6, 1, 5, 5)) * 0.1
    Wc3 = _rng.normal(size=(16, 6, 5, 5)) * 0.1
    b1, b3 = np.zeros(6), np.zeros(16)

    def pool2_29(h):
        """2×2 最大池化（28 课 reshape 技巧的批量版）。"""
        _N, _C, _H, _W = h.shape
        return h[:, :, :_H//2*2, :_W//2*2].reshape(_N, _C, _H//2, 2, _W//2, 2).max(axis=(3, 5))

    _h1 = np.maximum(conv_forward(_x, Wc1, b1), 0)               # C1: conv5×5 + ReLU
    _h1p = pool2_29(_h1)                                          # S2: pool2×2
    _h3 = np.maximum(conv_forward(_h1p, Wc3, b3), 0)              # C3: conv5×5 + ReLU
    _h3p = pool2_29(_h3)                                          # S4: pool2×2
    _flat = _h3p.reshape(64, -1)                                  # flatten（07 课）
    Wf5 = _rng.normal(size=(256, 120)) * 0.05
    Wf6 = _rng.normal(size=(120, 84)) * 0.05
    Wout = _rng.normal(size=(84, 10)) * 0.05
    _f5 = np.maximum(_flat @ Wf5, 0)                              # F5
    _f6 = np.maximum(_f5 @ Wf6, 0)                                # F6
    _out = _f6 @ Wout                                             # OUT

    _shapes = [("输入", (64, 1, 28, 28)), ("C1 conv5×5+ReLU", _h1.shape), ("S2 pool2×2", _h1p.shape),
               ("C3 conv5×5+ReLU", _h3.shape), ("S4 pool2×2", _h3p.shape),
               ("flatten", _flat.shape), ("F5 fc+ReLU", _f5.shape),
               ("F6 fc+ReLU", _f6.shape), ("OUT fc", _out.shape)]
    for _nm, _s in _shapes:
        print(f"  {_nm:<16} {str(_s):>18}")

    _p = {"C1": 6*1*5*5 + 6, "C3": 16*6*5*5 + 16, "F5": 256*120 + 120,
          "F6": 120*84 + 84, "OUT": 84*10 + 10}
    print(f"\n  参数账：C1 {_p['C1']} + C3 {_p['C3']} + F5 {_p['F5']:,} + "
          f"F6 {_p['F6']:,} + OUT {_p['OUT']} = {sum(_p.values()):,}")
    print(f"  对照 25 课 MLP（784-128-10）：101,770 → 少 {1 - sum(_p.values())/101770:.0%}")
    return (pool2_29,)


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
        ### 交互实验 1 · im2col 透视器：看一眼摊开的矩阵

        拖动核大小，看三样东西：**摊开的矩阵长什么样**（中图，
        每一列是一个窗口）、**内存账怎么涨**、**卷积输出怎么
        缩**。矩阵的"列纹"有规律——每列 9/25/49 个数，列与列
        之间大量内容重叠（那就是复制的代价）。
        """
    )
    return


@app.cell
def _(mo):
    # 交互控制：核大小
    ksize29_slider = mo.ui.slider(3, 7, value=3, step=2, label="📐 核大小 K（3 / 5 / 7）")
    mo.hstack([ksize29_slider])
    return (ksize29_slider,)


@app.cell
def _(X_mnist, conv_forward, im2col, ksize29_slider, mo, np, plt):
    # ===== 交互实验 1 · im2col 透视器（响应单元）=====
    _K = int(ksize29_slider.value)
    _img = X_mnist[1].reshape(1, 1, 28, 28)                # 数字 0
    _cols = im2col(_img, _K)                               # (1, K*K, OH*OH)
    _oh = 28 - _K + 1
    _n_elem = _K * _K * _oh * _oh

    # 一个匹配当前 K 的竖边核（左 +1 右 −1 的加宽版），看输出图
    _half = _K // 2
    _row = [1] * (_K - _half - 1) + [0] + [-1] * (_K - _half - 1)
    _Kv = np.array([_row] * _K, dtype=np.float64).reshape(1, 1, _K, _K)
    _out = conv_forward(_img, _Kv, np.zeros(1))[0, 0]

    _fig, _axes = plt.subplots(1, 3, figsize=(12.4, 4.1),
                               gridspec_kw={"width_ratios": [0.75, 1.35, 0.75]})
    _axes[0].imshow(_img[0, 0], cmap="gray_r")
    _axes[0].set_title("输入 28×28\n（784 元素）", fontsize=10)
    _axes[1].imshow(_cols[0], cmap="viridis", aspect="auto", interpolation="nearest")
    _axes[1].set_title(f"im2col 矩阵：({_K*_K}, {_oh*_oh})\n"
                       f"每列 = 一个 {_K}×{_K} 窗口摊平", fontsize=10)
    _axes[1].set_xlabel("输出位置（列）")
    _axes[1].set_ylabel("窗口内元素（行）")
    _vmax = np.abs(_out).max()
    _axes[2].imshow(_out, cmap="RdBu_r", vmin=-_vmax, vmax=_vmax)
    _axes[2].set_title(f"卷积输出 {_oh}×{_oh}", fontsize=10)
    for _ax in [_axes[0], _axes[2]]:
        _ax.axis("off")
    _fig.tight_layout()
    _fig

    _ratio = _n_elem / 784
    _verdict = (f"**K={_K} 的账本**：矩阵 {_K*_K}×{_oh*_oh} = **{_n_elem:,} 元素"
                f"（{_n_elem*8/1024:.1f} KB）**，是原图 784 元素的 **{_ratio:.1f} 倍**——"
                f"窗口重叠让数据被复制了多份。对照：K=3 时 6,084 元素（7.8 倍）、"
                "K=7 时 23,716（30.2 倍）。大核的内存税，在这里看得一清二楚。")

    mo.vstack([
        _fig,
        mo.callout(mo.md(_verdict), kind="info"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 交互实验 2 · col2im 散射器：一个像素的梯度从哪来

        前向时每个像素被复制进若干个窗口列（27 课覆盖计数）；
        反向时这些列各还一份梯度，**加起来**才是它的 dX。拖动
        像素位置：右图点亮"包含它的窗口位置"——

        - 角落像素：1 个窗口 → 梯度只从 1 列回流
        - 边缘像素：2 个（对角）/ 3 个（直边）窗口
        - 中心像素：K×K = 9 个窗口 → 9 份回流之和

        这就是 col2im 必须用 **+= 累加** 而不是覆盖的原因，
        也是演示 2 里 dx 能对账到 10⁻¹¹ 的机制保证。
        """
    )
    return


@app.cell
def _(mo):
    # 交互控制：像素位置
    pix_i_slider = mo.ui.slider(0, 27, value=14, step=1, label="⬇️ 像素行 i（0~27）")
    pix_j_slider = mo.ui.slider(0, 27, value=14, step=1, label="➡️ 像素列 j（0~27）")
    mo.hstack([pix_i_slider, pix_j_slider])
    return (pix_i_slider, pix_j_slider)


@app.cell
def _(X_mnist, mo, np, plt, pix_i_slider, pix_j_slider):
    # ===== 交互实验 2 · col2im 散射器（响应单元）=====
    _i, _j = pix_i_slider.value, pix_j_slider.value
    _K = 3
    _img = X_mnist[1].reshape(28, 28)

    # 包含 (i,j) 的窗口（左上角位置）：窗口 (a,b) 覆盖 (i,j) ⟺ a ≤ i ≤ a+2 且 b ≤ j ≤ b+2
    _wins = [(a, b) for a in range(max(0, _i - 2), min(_i, 25) + 1)
                    for b in range(max(0, _j - 2), min(_j, 25) + 1)]
    _n_win = len(_wins)

    _fig, _axes = plt.subplots(1, 2, figsize=(10.6, 4.6))
    _axes[0].imshow(_img, cmap="gray_r")
    _axes[0].scatter(_j, _i, s=120, color="red", zorder=5, marker="s")
    for _a, _b in _wins:
        _axes[0].add_patch(plt.Rectangle((_b - .5, _a - .5), 3, 3,
                                         fill=False, color="tab:orange", lw=1.4, alpha=0.85))
    _axes[0].set_title(f"输入：红方块 = 像素({_i},{_j})\n橙框 = 包含它的 {_n_win} 个窗口", fontsize=10)
    _axes[0].axis("off")
    _cover = np.zeros((26, 26))
    for _a, _b in _wins:
        _cover[_a, _b] = 1
    _axes[1].imshow(_cover, cmap="Oranges", vmin=0, vmax=1.4)
    _axes[1].set_title(f"窗口位置图（26×26）\n点亮 {_n_win} 个位置 = 该像素被复制进 {_n_win} 列", fontsize=10)
    _axes[1].axis("off")
    _fig.tight_layout()
    _fig

    if _n_win == 9:
        _verdict = ("**中心像素**：9 个窗口都包含它——反向时 9 列各还一份梯度，"
                    "col2im 把它们**全部加起来**写进 dX[i,j]。覆盖计数（27 课）= 回流通道数")
    elif _n_win in (2, 3):
        _verdict = (f"**边缘像素**：{_n_win} 个窗口包含它——比中心少，但不止一个；"
                    "注意对角边缘是 2、直边是 3")
    else:
        _verdict = (f"**角落像素**：只有 {_n_win} 个窗口包含它——梯度只有一条回流通道，"
                    "这就是无填充时边缘信息吃亏的梯度版（27 课覆盖计数的续集）")

    mo.vstack([
        _fig,
        mo.callout(mo.md(_verdict), kind="info"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✏️ 动手练习（每课必做，本课 4 题——第 5 部分的反向传播大课）": mo.md(
                """
                **练习 1 · 实现 im2col**：补全"往第几列摆什么"的
                一行——窗口切片、reshape、写列。用 26 课的
                sliding_window_view 向量化版对账（滑窗视图展开
                后 reshape 的结果应逐元素一致）。

                **练习 2 · 实现 conv_forward**：补全"一次矩阵乘
                + 广播偏置"的一行。和四重循环版
                （conv_naive29）在两组配置上逐元素对账。

                **练习 3 · 实现 col2im**：补全散射累加的一行。
                两道验收：① dcols 全 1 时，dX 应恰好等于
                27 课的**覆盖计数图**（角落 1、直边 3、中心 9）
                ——散射的"累加"就是覆盖的"计数"；② 随机 dcols
                与参考实现逐元素一致。

                **练习 4 · 池化反向（max 路由）**：补全"把梯度
                塞给赢家"的一行。验收：28 课练习 1 的那张 4×4
                矩阵 + 你指定的 dout，梯度必须只落在 4 个赢家
                格上、总量分毫不差。

                规则不变：先把 `None` 换成你的答案再运行；写完再看参考答案。
                """
            )
        }
    )
    return


@app.cell
def _(X_mnist, mo, np):
    # ===== 练习 1 · 实现 im2col =====
    def im2col_ex1(x, K, stride=1, pad=0):
        """补全第 ① 行：把窗口摊平摆进正确的列。"""
        _N, _C, _H, _W = x.shape
        if pad:
            x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        _oh = (x.shape[2] - K) // stride + 1
        _ow = (x.shape[3] - K) // stride + 1
        _cols = np.zeros((_N, _C * K * K, _oh * _ow))
        for _i in range(_oh):
            for _j in range(_ow):
                _cols[:, :, _i * _ow + _j] = None     # ← ① 补：窗口切片 → reshape → 摆列
        return _cols

    im2col_ready_ex1 = None                        # ← 补完后，把这行换成 im2col_ex1

    if im2col_ready_ex1 is not None:
        try:
            from numpy.lib.stride_tricks import sliding_window_view
            _x = X_mnist[:4].reshape(4, 1, 28, 28)
            _got = im2col_ready_ex1(_x, 3)
            _win = sliding_window_view(_x, (3, 3), axis=(2, 3))          # (N,C,26,26,3,3)
            _ref = _win.transpose(0, 1, 4, 5, 2, 3).reshape(4, 9, 26 * 26)
            if np.allclose(_got, _ref, atol=1e-10):
                _msg = ("✅ 4×676 列全部与滑窗视图一致！你的 im2col 与"
                        "演示 1 的工业版同源——注意列的排列顺序约定："
                        "列号 = i*OW + j（行主序）")
            else:
                _msg = ("❌ 列内容对了但顺序可能不对——列号必须是 _i*_ow+_j；"
                        "窗口切片 x[:, :, _i*stride:_i*stride+K, _j*stride:_j*stride+K]")
        except (TypeError, ValueError) as _e:
            _msg = f"❌ 运行出错：{_e}——提示：_patch.reshape(_N, _C*K*K)"
    else:
        _msg = "⏳ 先补第 ① 行，再把 im2col_ready_ex1 换成 im2col_ex1"
    mo.md(_msg)
    return (im2col_ex1, im2col_ready_ex1)


@app.cell
def _(conv_naive29, mo, np):
    # ===== 练习 2 · 实现 conv_forward =====
    def conv_forward_ex2(x, W, b, stride=1, pad=0):
        """补全第 ① 行：核摊平 → 一次矩阵乘 + 偏置广播。"""
        _N, _C, _H, _W = x.shape
        _F, _, _K, _ = W.shape
        _cols = None                                # ← ① 补：调用 im2col（提示：im2col(x, _K, stride, pad)）
        _W2 = W.reshape(_F, _C * _K * _K)
        _Z = None                                   # ← ② 补：W2 @ cols + b 广播到每个位置
        _oh = (_H + 2 * pad - _K) // stride + 1
        _ow = (_W + 2 * pad - _K) // stride + 1
        return _Z.reshape(_N, _F, _oh, _ow)

    conv_ready_ex2 = None                          # ← 补完后，把这行换成 conv_forward_ex2

    if conv_ready_ex2 is not None:
        try:
            _rng = np.random.default_rng(3)
            _x = _rng.normal(size=(2, 3, 8, 8))
            _W = _rng.normal(size=(4, 3, 3, 3))
            _b = _rng.normal(size=(4,))
            _ok = all(
                np.allclose(conv_ready_ex2(_x, _W, _b, _s, _p),
                            conv_naive29(_x, _W, _b, _s, _p), atol=1e-10)
                for _s, _p in [(1, 1), (2, 0)])
            if _ok:
                _msg = ("✅ 两组配置（S=1,P=1 与 S=2,P=0）都与四重循环版逐元素一致——"
                        "你的卷积层前向毕业了。偏置那步是 b[:, None]："
                        "(F,) → (F,1)，靠广播（06 课）铺到每一列")
            else:
                _msg = "❌ 有配置对不上——检查 ① im2col 的参数顺序、② b[:, None]"
        except (TypeError, ValueError) as _e:
            _msg = f"❌ 运行出错：{_e}——① im2col(x, _K, stride, pad)；② _W2 @ _cols + b[:, None]"
    else:
        _msg = "⏳ 先补两行，再把 conv_ready_ex2 换成 conv_forward_ex2"
    mo.md(_msg)
    return (conv_forward_ex2, conv_ready_ex2)


@app.cell
def _(mo, np):
    # ===== 练习 3 · 实现 col2im =====
    def col2im_ex3(dcols, x_shape, K, stride=1, pad=0):
        """dcols: (N, C·K·K, OH·OW) → dX: x_shape。补全散射累加的一行。"""
        _N, _C, _H, _W = x_shape
        _oh = (_H + 2 * pad - K) // stride + 1
        _ow = (_W + 2 * pad - K) // stride + 1
        _dxp = np.zeros((_N, _C, _H + 2 * pad, _W + 2 * pad))
        for _i in range(_oh):
            for _j in range(_ow):
                _p = _i * _ow + _j
                _patch = dcols[:, :, _p].reshape(_N, _C, K, K)
                _dxp[:, :, _i*stride:_i*stride+K, _j*stride:_j*stride+K] += None   # ← ① 补
        return _dxp[:, :, pad:_H + pad, pad:_W + pad] if pad else _dxp

    col2im_ready_ex3 = None                        # ← 补完后，把这行换成 col2im_ex3

    if col2im_ready_ex3 is not None:
        try:
            # 验收 ①：dcols 全 1 → dX = 覆盖计数图（27 课续集）
            _ones = np.ones((1, 9, 4))             # 4×4 输入、K=3 → 2×2 窗口
            _got = col2im_ready_ex3(_ones, (1, 1, 4, 4), 3)
            _cover = np.zeros((4, 4))
            for _a in range(2):
                for _b in range(2):
                    _cover[_a:_a+3, _b:_b+3] += 1
            _ok1 = np.allclose(_got[0, 0], _cover, atol=1e-12)
            # 验收 ②：随机 dcols 与参考实现一致
            _rng = np.random.default_rng(4)
            _dc = _rng.normal(size=(2, 3 * 3 * 2, 3 * 3))    # (N, CKK, OHW)
            _xr = (2, 2, 8, 8)
            _ref = np.zeros((2, 2, 8, 8))
            _oh = _ow = 6
            for _i in range(_oh):
                for _j in range(_ow):
                    _p = _i * _ow + _j
                    _ref[:, :, _i:_i+3, _j:_j+3] += _dc[:, :, _p].reshape(2, 2, 3, 3)
            _ok2 = np.allclose(col2im_ready_ex3(_dc, _xr, 3), _ref, atol=1e-12)
            if _ok1 and _ok2:
                _msg = ("✅ 双验收全过！全 1 的 dcols 散射出 [[1,2,2,1],[2,4,4,2],...] "
                        "——正是 27 课的覆盖计数：角落 1、直边 2/3、中心 4（4×4 图的版本）。"
                        "累加的物理意义：N 列回流，各还各的，一分不差")
            else:
                _msg = f"❌ 验收①覆盖计数 {'过' if _ok1 else '没过'}、验收②随机对账 {'过' if _ok2 else '没过'}——散射行是 += _patch（不是 = _patch！）"
        except (TypeError, ValueError) as _e:
            _msg = f"❌ 运行出错：{_e}——提示：直接把 _patch 累加到对应窗口切片上"
    else:
        _msg = "⏳ 先补散射行，再把 col2im_ready_ex3 换成 col2im_ex3"
    mo.md(_msg)
    return (col2im_ex3, col2im_ready_ex3)


@app.cell
def _(mo, np):
    # ===== 练习 4 · 池化反向（max 路由）=====
    def maxpool_bwd_ex4(im, dout):
        """im: (H,W) 原图，dout: (H/2, W/2) 上游梯度 → dx: (H,W)。补全路由行。"""
        _h, _w = im.shape
        _dx = np.zeros_like(im)
        for _i in range(_h // 2):
            for _j in range(_w // 2):
                _block = im[_i*2:_i*2+2, _j*2:_j*2+2]
                _wi, _wj = np.unravel_index(np.argmax(_block), (2, 2))    # 找赢家
                _dx[_i*2+_wi, _j*2+_wj] = None    # ← ① 补：把哪份梯度塞给赢家？
        return _dx

    pool_bwd_ready_ex4 = None                      # ← 补完后，把这行换成 maxpool_bwd_ex4

    if pool_bwd_ready_ex4 is not None:
        try:
            _im = np.array([[1, 0, 3, 2], [4, 1, 2, 0], [0, 5, 2, 1], [1, 0, 4, 3]], float)
            _dy = np.array([[10., 20.], [30., 40.]])
            _got = pool_bwd_ready_ex4(_im, _dy)
            _exp = np.zeros((4, 4))
            _exp[1, 0], _exp[0, 2], _exp[2, 1], _exp[3, 2] = 10, 20, 30, 40
            _ok = np.allclose(_got, _exp, atol=1e-12)
            _sum_ok = abs(_got.sum() - _dy.sum()) < 1e-12
            if _ok and _sum_ok:
                _msg = ("✅ 路由正确：4 个赢家 (1,0)/(0,2)/(2,1)/(3,2) 各拿走自己的那份，"
                        "其余 12 格梯度为零；总量 100 分毫不差（守恒）。"
                        "argmax 是前向的记账（概念 4），路由是反向的还钱")
            else:
                _msg = (f"❌ 赢家位置或数额不对（对账 {'过' if _sum_ok else '不过'}）。"
                        "塞给赢家的是 dout[_i,_j]——它那一份，不是全图的总和")
        except (TypeError, ValueError) as _e:
            _msg = f"❌ 运行出错：{_e}——提示：_dx[行, 列] = _dout 对应块的值"
    else:
        _msg = "⏳ 先补路由行，再把 pool_bwd_ready_ex4 换成 maxpool_bwd_ex4"
    mo.md(_msg)
    return (maxpool_bwd_ex4, pool_bwd_ready_ex4)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 1 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                _cols[:, :, _i * _ow + _j] = x[:, :, _i*stride:_i*stride+K,
                                                _j*stride:_j*stride+K].reshape(_N, _C * K * K)
                ```

                三个动作一气呵成：**切片**取窗口（04 课）、
                **reshape** 把 (N,C,K,K) 摊平成 (N, C·K·K)（07 课，
                注意 C 在前 K 在后的排列约定）、**赋值**摆进
                第 `i*OW + j` 列（行主序，和图像的读序一致）。

                滑窗视图对账版用 transpose 把 (N,C,OH,OW,K,K)
                挪成 (N,C,K,K,OH,OW) 再 reshape——同一个矩阵，
                两种搬运路径，互为铁证。
                """
            ),
            "✅ 练习 2 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                _cols = im2col(x, _K, stride, pad)
                _Z = _W2 @ _cols + b[:, None]
                ```

                `_W2 @ _cols`：(F, CKK) @ (N, CKK, OHW) →
                (N, F, OHW)——矩阵乘自动广播到批量维（NumPy 的
                matmul 对高维按批处理，12 课进阶用法）。
                `b[:, None]` 把 (F,) 变 (F,1)，让偏置加到
                **每个输出位置**上（06 课广播）。

                最后 reshape 回 (N, F, OH, OW)。四步里的每一步
                都是旧课的零件：pad → im2col → matmul+broadcast →
                reshape。**没有新数学，只有新组装。**
                """
            ),
            "✅ 练习 3 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                _dxp[:, :, _i*stride:_i*stride+K, _j*stride:_j*stride+K] += _patch
                ```

                灵魂在 **+=**：窗口之间互相重叠（stride < K 时），
                同一个输入像素躺在多列里，散射必须**累加**。
                写成 `=` 会把前面窗口的还钱覆盖掉——梯度对账
                立刻爆掉（你可以故意改回 `=` 跑一遍，看相对误差
                从 10⁻¹¹ 跳到 10⁰ 量级，体验"覆盖"的破坏力）。

                全 1 验收是最妙的自检：dcols 全 1 时，dX[i,j] =
                包含它的窗口数——**散射的累加就是覆盖的计数**。
                27 课用四重循环数出来的 1/3/9，今天从梯度的
                方向又推了一遍。两张视角，一个事实。
                """
            ),
            "✅ 练习 4 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                _dx[_i*2+_wi, _j*2+_wj] = dout[_i, _j]
                ```

                赢家坐标是全局的 `(块起点 + 块内偏移)`：行
                `_i*2+_wi`、列 `_j*2+_wj`；塞给它的就是这一块的
                上游梯度 `dout[_i, _j]`。

                三个性质记住即可：

                - **每块恰好一个赢家**拿走全部梯度（并列时
                  argmax 取第一个——数值差分在这种"平顶"点不稳
                  定，是已知边界，工程忽略）
                - **总量守恒**：sum(dX) == sum(dout)
                - **稀疏**：4×4 的例子里 16 格只有 4 格非零——
                  max 路由是"赢者通吃"的梯度世界

                对照 dropout 的反向（25 课）：那边梯度过 mask
                  （被删的拿零），这边梯度过 argmax（输的拿零）
                  ——**前向怎么选择的，反向就怎么分账**。
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
                1. **im2col**：每个输出位置一列，窗口摊平摆齐；
                   cols (N, C·K·K, OH·OW)、W2 (F, C·K·K)，
                   卷积 = W2 @ cols + b——12 课"线性操作皆矩阵乘"
                   的落地
                2. **速度与内存的交易**：四重循环 1230 ms →
                   im2col 6 ms（本机约 210 倍，BLAS 施工队）；
                   代价是复制：K=3 单图 6,084 元素（7.8 倍原图）、
                   K=7 达 23,716（30.2 倍）
                3. **反向三步**：db 三轴求和；dW = einsum 误差×窗口
                   对位置求和（"哪个窗口像、误差大，核就往哪改"）；
                   dX = Wᵀ 回流再 col2im 散射累加
                4. **col2im 的 += 是生命线**：像素被几个窗口覆盖，
                   就有几列回流；全 1 dcols 散射出覆盖计数图——
                   27 课 1/3/9 的梯度版证明
                5. **数值对账**：双侧差分 vs 解析，dW/db/dX 相对
                   误差全在 10⁻¹⁰ 上下；四个九的签字，22/23 课家法
                   在卷积层照常生效
                6. **池化反向 = max 路由**：argmax 记赢家、梯度
                   只给赢家、总量守恒 sum(dX)==sum(dout)；
                   dropout 过 mask、pool 过 argmax——前向的
                   记账决定反向的分账
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

                引擎全部到手：im2col 前向（快）、conv_backward
                （准）、pool 路由（稳）。LeNet-5 的形状账也核对
                完毕：44,426 个参数，比 25 课的 MLP 还少 56%。

                只剩一件事没做——**训练它**。
                """
            ),
            mo.callout(
                mo.md(
                    """
                    **下一课预告** — `30_LeNet-5实战`：

                    第 5 部分收官之战：把 C1-S2-C3-S4-F5-F6-OUT
                    七层组装成完整网络，接上 22~23 课的反向传播、
                    24 课的 Adam、25 课的完全体训练循环，在
                    MNIST 上向 25 课 MLP 的 97.78% 正式宣战。

                    战报预告：训练完成后，你会亲眼看到 conv1 的
                    6 个核**自己长成了边缘检测器**——没人教它
                    竖边核长什么样，是损失逼出来的。

                    ⚔️ 全部家当带上，去打 LeNet-5 的最后一仗！
                    """
                ),
                kind="success",
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
