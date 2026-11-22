import marimo

__generated_with = "0.25.2"
app = marimo.App(app_title="28 · 池化与感受野", width="medium")


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
        # 28 · 池化与感受野：把「大概在哪」变成够用的信息

        **从零到 Transformer · 手写深度学习** ｜ 第 5 部分 · 卷积神经网络（3/6）
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📖 本课目标（先读我）": mo.md(
                """
                **卷积扫完的响应图很大，而且记得太细了。**

                27 课的卷积层吐出 26×26 的响应图——每一格都忠实
                记录"这个精确位置有多像模板"。但认字不需要这么细：
                笔画往右挪一格，"这里有竖边"这个事实没变。

                这节课的两件事，一件做减法，一件算视野：

                - **最大池化**：每 2×2 的小块只留一个最大的数。
                  图砍半、参数为零、最强信号保住——顺便买一份
                  对小抖动的容忍
                - **感受野**：层层堆叠后，最深处的一个输出到底
                  "看得见"原图多大范围？一条递推公式算清楚，
                  你就明白为什么网络要**深**

                学完你将能：

                - 手写 max pooling（一行 reshape + max 的 NumPy
                  拼装），说清它**为什么零参数**
                - 用**固定读出实验**定量讲出池化的平移容忍：
                  数字挪 1 像素，盯着固定位置看的下游神经元，
                  读卷积图崩、读池化图稳（500 样本：高保持率
                  7% vs 39%）
                - 用递推公式 **RF += (K−1)×jump；jump ×= S** 算
                  任意网络栈的感受野，并解释"深度换视野"
                - 看懂 LeNet-5 的 5→6→14→16 视野生长路线
                  （30 课的地图）

                前置：26 课的响应图与平移实验（上半场）、
                27 课的步长（池化就是步长 2 的特殊扫描）、
                07/08 课的 reshape 与 axis（池化的实现零件）。
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "🏔️ 概念 1/3 · 最大池化：留最大，扔细节": mo.md(
                """
                池化（pooling）是卷积的减法搭档：**2×2 的窗口，
                步长 2，每块只留一个数**。

                - **最大池化**（max pooling）：留块内最大值——
                  "这 2×2 的小区域里，模板匹配得最欢有多欢"。
                  竖边响应图里 2.98 的强信号，池化后还是 2.98
                  （演示 1 实测），一个不丢
                - **平均池化**（average pooling）：留块内平均——
                  "这片区域平均有多像"。亮度守恒（均值不变），
                  但强信号被稀释

                三个立竿见影的效果：

                1. **尺寸减半**：26×26 → 13×13。层越深图越小，
                   计算量雪崩式下降
                2. **零参数**：没有权重、没有偏置，纯机械运算。
                   对照 26 课账本：全连接 100,480、卷积 80、
                   池化 **0**——减法不花钱
                3. **粗粒度化**：输出的第 (i,j) 格说的是
                   "原图第 2i~2i+1 行、2j~2j+1 列那片"——
                   位置精度从 1 像素变成 2 像素。**用位置精度
                   换稳定性**，这是下一节的主题

                > 术语：池化窗口几乎总是 2×2 步长 2（"各取所需、
                > 不重叠"）。更大窗口（3×3 步 2）会重叠、也更
                > 激进地丢信息，LeNet-5 用的是 2×2。
                """
            ),
            "🛡️ 概念 2/3 · 平移容忍：块内抖动免疫，块间邻格接住": mo.md(
                """
                池化为什么能容忍小抖动？拆开看一个 2×2 块：

                只要强响应**挪了但没挪出块**（1 像素抖动对 2×2
                的块，大部分情况如此），块的最大值**一字不改**。
                下游神经元读的是块的最大值——它根本不知道里面
                挪过了。这就是"块内抖动免疫"。

                如果强响应**整块出走**呢？这个块确实没救了
                （演示 2 里数字 1 左移 1 像素的那一格：读数从
                2.99 掉到 0.60，池化也拦不住）——但**相邻的
                池化格接住了它**。关键在于下游读的是粗网格：
                特征"挪了一个池化格"在粗网格里只是**一格误差**，
                而在细网格里是**两格误差**。粒度越粗，同样的
                物理位移造成的**相对**错位越小。

                所以诚实的结论是三段式：

                - 权值共享（26 课）管"特征换地方照样认"
                - 池化管"特征在附近晃，读数不慌"
                - 真正的类别判决在深层，那里每个输出已经看了
                  一大片（本课下半场：感受野）

                顺带一句：31 课会讲现代趋势——**步长卷积**抢了
                池化的饭碗（下采样 + 可学习二合一），但"用粒度
                换稳定"的思想从未过时。
                """
            ),
            "🔭 概念 3/3 · 感受野：深度换视野": mo.md(
                """
                单层 3×3 卷积的输出只看了 9 个输入像素（26 课的
                伏笔）。那第二层卷积呢？它的输入是第一层的响应
                图——**响应图的每个格子背后站着一小片原图**。
                第二层的窗口盖住 9 个格子，就间接看了一大片原图。

                感受野（receptive field, RF）就是"一个输出往回
                走，能覆盖多大范围的输入"。逐层递推只需两行：

                ```
                RF_new  = RF + (K − 1) × jump      # 本层窗口在原图上扩张多少
                jump    = jump × S                  # 步长越大，往后每格跨得越远
                ```

                直觉版：新窗口里的每个格子"身后"都拖着 jump 个
                原图像素的间隔；K 个格子首尾拉开 (K−1) 个间隔。

                拿它算两条经典路线（演示 3 有完整表）：

                - **纯卷积堆**（4 层 3×3）：RF = 3→5→7→9——
                  每层只长 2。想看全 28×28 要堆 13 层
                - **LeNet 风**（conv5 → pool2 → conv5 → pool2）：
                  RF = 5→6→14→16——池化把 jump 从 1 提到 2，
                  后面的卷积一层就长 8。**下采样是视野的
                  加速器**

                "深度学习"的深，一半是为了非线性（19 课），
                另一半就是为了这份不断长大的视野。
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
    # MNIST 加载器（与 25~27 课同款）
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

    X_mnist, y_mnist, X_te28, y_te28 = load_mnist()

    # 本课工具箱：向量化卷积 + 双池化 + 平移
    def conv_fast(img, K):
        """滑窗视图版卷积（26 课练习 2 的老朋友，26×26 出）。"""
        from numpy.lib.stride_tricks import sliding_window_view
        _win = sliding_window_view(img, K.shape)
        return np.einsum("ijkl,kl->ij", _win, K)

    def maxpool2(im):
        """2×2 最大池化：reshape 技巧（07 课形状 + 08 课 axis 的合体技）。"""
        _h, _w = im.shape
        return im[:_h // 2 * 2, :_w // 2 * 2].reshape(_h // 2, 2, _w // 2, 2).max(axis=(1, 3))

    def avgpool2(im):
        """2×2 平均池化。"""
        _h, _w = im.shape
        return im[:_h // 2 * 2, :_w // 2 * 2].reshape(_h // 2, 2, _w // 2, 2).mean(axis=(1, 3))

    def shift_img28(im, dy, dx):
        """把图往 (下, 右) 平移，空位补零。"""
        _out = np.zeros_like(im)
        _h, _w = im.shape
        _ys, _ye = max(0, dy), min(_h, _h + dy)
        _xs, _xe = max(0, dx), min(_w, _w + dx)
        _out[_ys:_ye, _xs:_xe] = im[_ys - dy:_ye - dy, _xs - dx:_xe - dx]
        return _out

    K_vert28 = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float64)
    img_zero28 = X_mnist.reshape(-1, 28, 28)[1]       # 数字 0
    img_one28 = X_mnist.reshape(-1, 28, 28)[8]        # 数字 1（平移实验主角）
    print("训练集:", X_mnist.shape, "｜工具箱就位：conv_fast / maxpool2 / avgpool2 / shift_img28")
    return (K_vert28, X_mnist, avgpool2, conv_fast, img_one28, img_zero28, maxpool2, shift_img28, y_mnist)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 1 · 池化金字塔：图砍半，信号不丢

        上排：对**原图**连续最大池化 28→14→7——笔画还在，
        只是"墨有没有"的粒度变粗（max 会保住每一块里最亮的
        像素，笔画因此略微变粗壮）。

        下排：对**竖边响应图**（26×26）池化一次——强信号
        全数保留：max 2.98 → 2.98。均值账本也值得一看：
        原图均值 0.156，平均池化后还是 0.156（亮度守恒），
        最大池化后涨到 0.228（只留亮点，信号被"聚焦"）。

        这就是 max 与 avg 的性格差：**avg 是会计，max 是
        猎犬**。认笔画要的是猎犬。
        """
    )
    return


@app.cell
def _(K_vert28, avgpool2, conv_fast, img_zero28, maxpool2, np, plt):
    # 演示 1 · 池化金字塔 + max/avg 对比
    _p1, _p2 = maxpool2(img_zero28), maxpool2(maxpool2(img_zero28))
    _r = conv_fast(img_zero28, K_vert28)
    _rp_max, _rp_avg = maxpool2(_r), avgpool2(_r)
    print(f"原图 {img_zero28.shape} → {_p1.shape} → {_p2.shape}")
    print(f"响应图 max：{_r.max():.2f} → 池化后 {_rp_max.max():.2f}（强信号保留）")
    print(f"均值账：原图 {img_zero28.mean():.3f}｜avg池化后 {avgpool2(img_zero28).mean():.3f}（守恒）"
          f"｜max池化后 {maxpool2(img_zero28).mean():.3f}（只留亮点，被抬高）")

    _fig, _axes = plt.subplots(2, 4, figsize=(12.6, 6.6))
    _axes[0, 0].imshow(img_zero28, cmap="gray_r")
    _axes[0, 0].set_title("原图 28×28", fontsize=11)
    _axes[0, 1].imshow(_p1, cmap="gray_r")
    _axes[0, 1].set_title("max 池化 14×14", fontsize=11)
    _axes[0, 2].imshow(_p2, cmap="gray_r")
    _axes[0, 2].set_title("再池化 7×7", fontsize=11)
    _axes[0, 3].imshow(avgpool2(img_zero28), cmap="gray_r")
    _axes[0, 3].set_title("对照：avg 池化 14×14\n（笔画被稀释）", fontsize=10)
    _vmax = np.abs(_r).max()
    _axes[1, 0].imshow(_r, cmap="RdBu_r", vmin=-_vmax, vmax=_vmax)
    _axes[1, 0].set_title("竖边响应图 26×26", fontsize=11)
    _axes[1, 1].imshow(_rp_max, cmap="RdBu_r", vmin=-_vmax, vmax=_vmax)
    _axes[1, 1].set_title("max 池化 13×13\n（max 2.98 → 2.98）", fontsize=10)
    _axes[1, 2].imshow(_rp_avg, cmap="RdBu_r", vmin=-_vmax, vmax=_vmax)
    _axes[1, 2].set_title("对照：avg 池化 13×13\n（强信号被平均稀释）", fontsize=10)
    _axes[1, 3].axis("off")
    _axes[1, 3].text(0.5, 0.5, "账本速览\n\n参数量：池化 = 0\n\nmax 是猎犬：\n留最强的\n\navg 是会计：\n保均值守恒",
                     ha="center", va="center", fontsize=12, transform=_axes[1, 3].transAxes)
    for _ax in _axes.ravel():
        _ax.axis("off")
    _fig.suptitle("池化金字塔：尺寸雪崩、参数为零、最强信号存活", fontsize=12)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 2 · 固定读出实验：池化买来的容忍，明码标价

        26 课审判过"平移罪"，今天补下半场。设一个最简单的下游
        神经元：**盯着固定位置读数**（全连接的本能）。两个版本：

        - **读卷积格**：盯死响应图 (7,14)——原图上笔画最强处
        - **读池化格**：盯死池化图 (3,7)——同一片区域的 2×2 块

        然后把数字 1 往各方向挪，看两个读数各自的命运：

        | 平移 | 读卷积格 (7,14) | 读池化格 (3,7) |
        |---|---|---|
        | 原图 | +2.99 | 2.99 |
        | 右 1px | +2.39 | **2.99（纹丝不动）** |
        | 下 1px | +2.90 | 2.90 |
        | 左 1px | +0.60 | 0.60（笔画走出块，双双失手） |
        | 右 2px | **−2.76（符号翻转）** | +2.52（保持 84%） |

        规律浮出水面：**1 像素的小晃动，只要强响应没挪出 2×2
        的块，池化格的读数一个字都不改**（右 1px）；挪出块的
        （左 1px）谁都救不了这格——但相邻池化格接住了信号。
        整步长平移（右 2px）更妙：卷积格直接符号翻转，池化格
        只丢了 16%。

        500 个样本的大样本统计（各找其最强格，右移 1px 后再读）：

        | 指标 | 读卷积格 | 读池化格 |
        |---|---|---|
        | 保持率 > 80% 的比例 | 7% | **40%** |
        | 保持率中位数 | 0.43 | 0.60 |

        池化把"高保持率"的可能性抬高了 **5 倍多**——这就是
        "用位置精度换稳定性"的明码标价。
        """
    )
    return


@app.cell
def _(K_vert28, X_mnist, conv_fast, img_one28, maxpool2, np, plt, shift_img28):
    # 演示 2 · 固定读出实验
    r0_28 = conv_fast(img_one28, K_vert28)
    p0_28 = maxpool2(r0_28)
    i_fix, j_fix = np.unravel_index(np.argmax(r0_28), r0_28.shape)      # (7,14)
    pi_fix, pj_fix = i_fix // 2, j_fix // 2                             # (3,7)

    _shifts = [(0, 0, "原图"), (0, 1, "右1"), (0, -1, "左1"), (1, 0, "下1"),
               (0, 2, "右2"), (1, 1, "右下1")]
    print(f"卷积格 ({i_fix},{j_fix})、池化格 ({pi_fix},{pj_fix}) 的读数：")
    print(f"{'平移':>5} │ {'读卷积格':>8} │ {'读池化格':>8}")
    print("─" * 32)
    _rows = []
    for _dy, _dx, _tag in _shifts:
        _r1 = conv_fast(shift_img28(img_one28, _dy, _dx), K_vert28)
        _p1 = maxpool2(_r1)
        _rows.append((_tag, _r1[i_fix, j_fix], _p1[pi_fix, pj_fix]))
        print(f"{_tag:>5} │ {_r1[i_fix, j_fix]:>+8.2f} │ {_p1[pi_fix, pj_fix]:>8.2f}")

    # ---- 500 样本大统计 ----
    _keep_c, _keep_p = [], []
    for _k in range(500):
        _im = X_mnist.reshape(-1, 28, 28)[_k]
        _a = conv_fast(_im, K_vert28)
        if np.abs(_a).max() < 1.0:
            continue
        _pa = maxpool2(_a)
        _ii, _jj = np.unravel_index(np.argmax(np.abs(_a)), _a.shape)
        _b = conv_fast(shift_img28(_im, 0, 1), K_vert28)
        _pb = maxpool2(_b)
        if _a[_ii, _jj] != 0 and _pa[_ii // 2, _jj // 2] != 0:
            _keep_c.append(_b[_ii, _jj] / _a[_ii, _jj])
            _keep_p.append(_pb[_ii // 2, _jj // 2] / _pa[_ii // 2, _jj // 2])
    _kc, _kp = np.array(_keep_c), np.array(_keep_p)
    print(f"\n500 样本（右移 1px 后重读原最强格）：")
    print(f"  保持率>80%：读卷积格 {np.mean(_kc > 0.8):.0%} vs 读池化格 {np.mean(_kp > 0.8):.0%}")
    print(f"  保持率中位数：{np.median(_kc):.2f} vs {np.median(_kp):.2f}")

    # ---- 画图：六种平移的两组读数 ----
    _fig, _axes = plt.subplots(2, 6, figsize=(14.6, 5.2))
    for _c, (_dy, _dx, _tag) in enumerate(_shifts):
        _r1 = conv_fast(shift_img28(img_one28, _dy, _dx), K_vert28)
        _p1 = maxpool2(_r1)
        _vmax = np.abs(r0_28).max()
        _axes[0, _c].imshow(_r1, cmap="RdBu_r", vmin=-_vmax, vmax=_vmax)
        _axes[0, _c].scatter(j_fix, i_fix, s=130, facecolors="none",
                             edgecolors="green", linewidths=2, zorder=5)
        _axes[0, _c].set_title(f"{_tag}：{_r1[i_fix, j_fix]:+.2f}", fontsize=10)
        _axes[1, _c].imshow(_p1, cmap="RdBu_r", vmin=-_vmax, vmax=_vmax)
        _axes[1, _c].scatter(pj_fix, pi_fix, s=130, facecolors="none",
                             edgecolors="green", linewidths=2, zorder=5)
        _axes[1, _c].set_title(f"{_p1[pi_fix, pj_fix]:+.2f}", fontsize=10)
    _axes[0, 0].set_ylabel("读卷积格")
    _axes[1, 0].set_ylabel("读池化格")
    for _ax in _axes.ravel():
        _ax.axis("off")
    _axes[0, 0].text(-0.15, 0.5, "读卷积格\n(7,14)", fontsize=11, rotation=90,
                     va="center", ha="center", transform=_axes[0, 0].transAxes)
    _axes[1, 0].text(-0.15, 0.5, "读池化格\n(3,7)", fontsize=11, rotation=90,
                     va="center", ha="center", transform=_axes[1, 0].transAxes)
    _fig.suptitle("固定读出实验：同一个下游神经元，读卷积格（上）vs 读池化格（下）", fontsize=12)
    _fig.tight_layout()
    _fig
    return (i_fix, j_fix, p0_28, pi_fix, pj_fix, r0_28)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 3 · 感受野生长表 + 嵌套框：一层层把视野撑大

        左：两条经典路线的 RF 生长表（递推公式跑出来的）。
        纯卷积堆每层只长 2；LeNet 路线靠池化把 jump 提到 2，
        第二个 conv5 一层就长 8。

        右：把 LeNet 路线上一条真实链路（最终输出格 S4(1,1)
        一路回溯到输入）的感受野画在数字上——四层之后，
        一个输出格子已经**看到原图 16×16 的范围**（绿色框
        是第一层看到的 5×5，橙色 6×6，红色 14×14，最外层
        16×16）。四层网络还看不全 28×28——但全连接层读的
        是全部 4×4=16 个格子，**联合视野**已经覆盖全图；
        再垫一层 conv5，单格 RF 直接 16→32，超过整张图。
        """
    )
    return


@app.cell
def _(img_one28, plt):
    # 演示 3 · 感受野生长表 + 嵌套框可视化
    def rf_track(layers):
        """layers: [(名字, K, stride)]，返回每层后的 (名字, RF, jump)。"""
        _rf, _jump, _rows = 1, 1, []
        for _nm, _K, _S in layers:
            _rf = _rf + (_K - 1) * _jump
            _jump = _jump * _S
            _rows.append((_nm, _rf, _jump))
        return _rows

    _vgg = rf_track([("conv3", 3, 1), ("conv3", 3, 1), ("conv3", 3, 1), ("conv3", 3, 1)])
    _lenet = rf_track([("conv5", 5, 1), ("pool2", 2, 2), ("conv5", 5, 1), ("pool2", 2, 2)])
    _lenet_plus = rf_track([("conv5", 5, 1), ("pool2", 2, 2), ("conv5", 5, 1),
                            ("pool2", 2, 2), ("conv5", 5, 1)])
    print("纯卷积堆（4 层 3×3）： ", " → ".join(f"{_r[1]}" for _r in _vgg))
    print("LeNet 风（conv5-pool-conv5-pool）：", " → ".join(f"{_r[1]}" for _r in _lenet))
    print("LeNet+1 层 conv5：    ", " → ".join(f"{_r[1]}" for _r in _lenet_plus), "（超过 28！）")

    _fig, _axes = plt.subplots(1, 2, figsize=(11.8, 5.2),
                               gridspec_kw={"width_ratios": [1, 1.1]})
    _axes[0].axis("off")
    _y = 0.90
    _axes[0].text(0.02, _y, "感受野 RF 生长表", fontsize=13, fontweight="bold")
    _labels = [("纯卷积堆", _vgg, "tab:blue"), ("LeNet 风", _lenet, "tab:green"),
               ("LeNet+1 层", _lenet_plus, "tab:red")]
    for _nm, _rows, _c in _labels:
        _y -= 0.14
        _axes[0].text(0.02, _y, _nm, fontsize=11, color=_c, fontweight="bold")
        _y -= 0.07
        _axes[0].text(0.10, _y, " → ".join(f"{_r[0]}:{_r[1]}" for _r in _rows),
                      fontsize=10, family="monospace")
    _y -= 0.14
    _axes[0].text(0.02, _y, "规律：纯 conv3 每层 +2；\n池化把 jump 提到 2 后，\n每个 conv5 一层 +8。",
                  fontsize=10.5, color="dimgray")

    # 嵌套框：LeNet 链 S4(1,1) → C3(2,2) → S2(4,4) → C1(8,8) 的真实回投影
    _axes[1].imshow(img_one28, cmap="gray_r")
    _boxes = [(4, 4, 16, "tab:red", "S4 后 16×16"),      # (y, x, size, 颜色, 标签)
              (4, 4, 14, "tab:orange", "C3 后 14×14"),
              (8, 8, 6, "tab:purple", "S2 后 6×6"),
              (8, 8, 5, "tab:green", "C1 后 5×5")]
    for _by, _bx, _sz, _c, _lb in _boxes:
        _axes[1].add_patch(plt.Rectangle((_bx - .5, _by - .5), _sz, _sz,
                                         fill=False, color=_c, lw=2.2))
    _axes[1].annotate("16×16", (4.5 + 16, 4.5), fontsize=10, color="tab:red")
    _axes[1].annotate("14×14", (4.5 + 14.2, 20.5), fontsize=10, color="tab:orange")
    _axes[1].annotate("6×6", (8.5 + 6, 8.5 + 6.6), fontsize=10, color="tab:purple")
    _axes[1].annotate("5×5", (8.5 + 5, 8.5 - 1.2), fontsize=10, color="tab:green")
    _axes[1].set_title("四层之后：一个输出格看见 16×16\n（嵌套框 = 逐层回溯的真实范围）", fontsize=11)
    _axes[1].axis("off")
    _fig.suptitle("深度换视野：感受野的两种长法", fontsize=12)
    _fig.tight_layout()
    _fig
    return (rf_track,)


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
        ### 交互实验 1 · 池化稳定器：换池化类型、挪数字看读数

        拖动**平移方向**，同时观察两行图：上排是卷积响应图、
        下排是池化图，绿圈永远是同一个"固定读出位置"。
        再切换 max / avg 池化，看看猎犬和会计谁更扛晃。

        对照目标（演示 2 的表格）：右 1px 时读池化格应该
        纹丝不动（2.99）；左 1px 时两行一起跌落——这不是
        bug，是"笔画走出块"的物理事实。
        """
    )
    return


@app.cell
def _(mo):
    # 交互控制：池化类型 + 平移方向
    pool_dropdown = mo.ui.dropdown(options=["最大池化", "平均池化"], value="最大池化", label="🏔️ 池化类型")
    shift_dropdown = mo.ui.dropdown(
        options=["原图", "右 1px", "左 1px", "下 1px", "上 1px", "右 2px", "右下 1px"],
        value="右 1px", label="↔️ 平移方向")
    mo.hstack([pool_dropdown, shift_dropdown])
    return (pool_dropdown, shift_dropdown)


@app.cell
def _(K_vert28, avgpool2, conv_fast, img_one28, i_fix, j_fix, maxpool2, mo, np, pi_fix, pj_fix, plt, pool_dropdown, r0_28, shift_dropdown, shift_img28):
    # ===== 交互实验 1 · 池化稳定器（响应单元）=====
    _pool = maxpool2 if (pool_dropdown.value or "最大池化") == "最大池化" else avgpool2
    _shift_map = {"原图": (0, 0), "右 1px": (0, 1), "左 1px": (0, -1),
                  "下 1px": (1, 0), "上 1px": (-1, 0), "右 2px": (0, 2), "右下 1px": (1, 1)}
    _dy, _dx = _shift_map[shift_dropdown.value or "右 1px"]

    _r1 = conv_fast(shift_img28(img_one28, _dy, _dx), K_vert28)
    _p1 = _pool(_r1)
    _read_conv, _read_pool = float(_r1[i_fix, j_fix]), float(_p1[pi_fix, pj_fix])
    _base_conv, _base_pool = float(r0_28[i_fix, j_fix]), float(_pool(r0_28)[pi_fix, pj_fix])

    _fig, _axes = plt.subplots(1, 3, figsize=(12.0, 4.2),
                               gridspec_kw={"width_ratios": [0.8, 1, 1]})
    _axes[0].imshow(shift_img28(img_one28, _dy, _dx), cmap="gray_r")
    _axes[0].set_title(f"输入（{shift_dropdown.value}）", fontsize=11)
    _vmax = np.abs(r0_28).max()
    _axes[1].imshow(_r1, cmap="RdBu_r", vmin=-_vmax, vmax=_vmax)
    _axes[1].scatter(j_fix, i_fix, s=150, facecolors="none", edgecolors="green",
                     linewidths=2.5, zorder=5)
    _axes[1].set_title(f"卷积图 @({i_fix},{j_fix})\n读数 {_read_conv:+.2f}", fontsize=10)
    _axes[2].imshow(_p1, cmap="RdBu_r", vmin=-_vmax, vmax=_vmax)
    _axes[2].scatter(pj_fix, pi_fix, s=150, facecolors="none", edgecolors="green",
                     linewidths=2.5, zorder=5)
    _axes[2].set_title(f"池化图 @({pi_fix},{pj_fix})\n读数 {_read_pool:+.2f}", fontsize=10)
    for _ax in _axes:
        _ax.axis("off")
    _fig.suptitle(f"{pool_dropdown.value}遇上平移：固定读出位置（绿圈）的命运", fontsize=12)
    _fig.tight_layout()
    _fig

    # ---- 判词 ----
    _ratio_pool = _read_pool / _base_pool if _base_pool != 0 else 0
    if _dx == 0 and _dy == 0:
        _verdict = "基线状态：两个读数都拿到最强响应（2.99）。挪一下方向再来看"
    elif abs(_read_conv - _base_conv) < 1e-9 and abs(_read_pool - _base_pool) < 1e-9:
        _verdict = "两个读数都没变——这次平移连卷积格都没晃到（上 1px：笔画厚，格内自吸收）"
    elif abs(_read_pool - _base_pool) < 1e-9:
        _verdict = (f"🎯 池化格纹丝不动（{_base_pool:.2f} → {_read_pool:.2f}）："
                    "强响应在 2×2 块内挪了挪，块最大值照单全收；"
                    f"卷积格却从 {_base_conv:.2f} 变到 {_read_conv:+.2f}")
    elif abs(_ratio_pool) > 0.7:
        _verdict = (f"池化格保持 {_ratio_pool:.0%}（{_base_pool:.2f} → {_read_pool:+.2f}）；"
                    f"卷积格变到 {_read_conv:+.2f}——粗粒度读数明显更扛晃")
    else:
        _verdict = (f"这次连池化格都保不住（{_base_pool:.2f} → {_read_pool:+.2f}）："
                    "笔画整块出走。物理事实——信号去了隔壁池化格，"
                    "下游要在粗网格上接着找它")

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
                **练习 1 · 手算最大池化**：一个 4×4 的数字矩阵，
                2×2 最大池化后是 2×2——四个格子逐个笔算。

                **练习 2 · 实现 maxpool2**：补全双循环版本，然后
                用 **reshape 技巧**（把 (H,W) 折成 (H/2, 2, W/2, 2)
                再沿窗口两个 axis 取 max）对账——这是 07 课形状
                与 08 课 axis 的合体技，一行向量化搞定全部块。

                **练习 3 · 感受野表**：给一个四层栈
                conv3 → conv3 → pool2 → conv3，用递推公式算出
                每层之后的 RF，填四个空（jump 的变化也藏在内，
                小心第 4 层）。

                规则不变：先把 `None` 换成你的答案再运行；写完再看参考答案。
                """
            )
        }
    )
    return


@app.cell
def _(mo, np):
    # ===== 练习 1 · 手算最大池化 =====
    m_ex1 = np.array([[1, 0, 3, 2],
                      [4, 1, 2, 0],
                      [0, 5, 2, 1],
                      [1, 0, 4, 3]], dtype=float)
    print("4×4 输入：\n", m_ex1)
    print("2×2 最大池化 → 2×2，四个输出逐个笔算（左上块是 [1,0,4,1]）")

    out11_ex1 = None                          # ← ① 左上块的最大值
    out12_ex1 = None                          # ← ② 右上块的最大值
    out21_ex1 = None                          # ← ③ 左下块的最大值
    out22_ex1 = None                          # ← ④ 右下块的最大值

    _ref = m_ex1.reshape(2, 2, 2, 2).max(axis=(1, 3))
    if None not in (out11_ex1, out12_ex1, out21_ex1, out22_ex1):
        _got = np.array([[out11_ex1, out12_ex1], [out21_ex1, out22_ex1]], dtype=float)
        if np.allclose(_got, _ref):
            _msg = (f"✅ 四块全对！[[{out11_ex1:.0f},{out12_ex1:.0f}],[{out21_ex1:.0f},{out22_ex1:.0f}]]"
                    "——每块只留最大，剩下的 12 个数被扔掉")
        else:
            _msg = f"❌ 参考：\n{_ref}\n每块是 2×2 的四个相邻数：行 0-1/列 0-1、行 0-1/列 2-3……"
    else:
        _msg = "⏳ 四个最大值都找出来再填"
    mo.md(_msg)
    return (m_ex1,)


@app.cell
def _(K_vert28, conv_fast, img_zero28, mo, np):
    # ===== 练习 2 · 实现 maxpool2 =====
    def maxpool2_ex2(im, k=2):
        """补全：每个 k×k 块取最大。"""
        _h, _w = im.shape
        _out = np.zeros((_h // k, _w // k))
        for _i in range(_h // k):
            for _j in range(_w // k):
                _out[_i, _j] = None           # ← 补：第 (i,j) 块 = im 的 k×k 区域取 max
        return _out

    pool_ready_ex2 = None                     # ← 补完后，把这行换成 maxpool2_ex2

    if pool_ready_ex2 is not None:
        try:
            _r = conv_fast(img_zero28, K_vert28)
            _got = pool_ready_ex2(_r)
            _ref = _r.reshape(13, 2, 13, 2).max(axis=(1, 3))       # reshape 技巧参考
            if np.allclose(_got, _ref, atol=1e-12):
                _msg = (f"✅ 169 个池化值与 reshape 技巧全部一致！"
                        f"形状 {_got.shape}、max {_got.max():.2f}（强信号保留）。"
                        "reshape 技巧把四重循环压成一行——29 课 im2col 的同门功夫")
            else:
                _msg = (f"❌ 有 {(~np.isclose(_got, _ref, atol=1e-12)).sum()} 个值对不上——"
                        "第 (i,j) 块是 im[i*k:(i+1)*k, j*k:(j+1)*k] 再 .max()")
        except (TypeError, ValueError) as _e:
            _msg = f"❌ 运行出错：{_e}——块切片是 [i*k:(i+1)*k, j*k:(j+1)*k]"
    else:
        _msg = "⏳ 先补循环体，再把 pool_ready_ex2 换成 maxpool2_ex2"
    mo.md(_msg)
    return (maxpool2_ex2, pool_ready_ex2)


@app.cell
def _(mo):
    # ===== 练习 3 · 感受野表 =====
    print("四层栈：conv3 → conv3 → pool2 → conv3（RF 和 jump 逐层递推）")
    print("提示：RF += (K−1)×jump；jump ×= S。pool2 的 K=2、S=2")

    rf1_ex3 = None                            # ← ① 第 1 层 conv3 之后
    rf2_ex3 = None                            # ← ② 第 2 层 conv3 之后
    rf3_ex3 = None                            # ← ③ pool2 之后
    rf4_ex3 = None                            # ← ④ 第 3 层 conv3 之后（注意 jump 已经变了）
    return (rf1_ex3, rf2_ex3, rf3_ex3, rf4_ex3)


@app.cell
def _(mo, rf1_ex3, rf2_ex3, rf3_ex3, rf4_ex3):
    # ===== 练习 3 · 判词 =====
    _truth = [3, 5, 6, 10]
    _ans = [rf1_ex3, rf2_ex3, rf3_ex3, rf4_ex3]
    _names = ["conv3①", "conv3②", "pool2", "conv3③"]
    if any(_a is None for _a in _ans):
        _msg = "⏳ 四个 RF 值都算出来再填（递推时盯紧 jump：pool 之后它就不是 1 了）"
    else:
        _right = [i for i, (a, t) in enumerate(zip(_ans, _truth)) if a == t]
        if len(_right) == 4:
            _msg = ("✅ 3 → 5 → 6 → 10 全对！最后一跳最迷人：pool 后 jump=2，"
                    "conv3 一层就长 (3−1)×2 = 4——下采样是视野加速器")
        else:
            _hint = {0: "起点 RF=1，conv3：1+(3−1)×1",
                     1: "在 3 的基础上再加 (3−1)×1",
                     2: "pool2：RF+（2−1)×1，且 jump 翻倍成 2",
                     3: "conv3：6+(3−1)×jump，jump 现在是 2 不是 1"}
            _msg = "❌ 还差：" + "；".join(f"{_names[i]}（{_hint[i]}）" for i in range(4) if _ans[i] != _truth[i])
    mo.md(_msg)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 1 参考答案（先自己写过再看！）": mo.md(
                """
                **[[4, 3], [5, 4]]。**

                - 左上块 [1, 0, 4, 1] → **4**
                - 右上块 [3, 2, 2, 0] → **3**
                - 左下块 [0, 5, 1, 0] → **5**
                - 右下块 [2, 1, 4, 3] → **4**

                16 个数进、4 个数出：**75% 的信息被扔掉**，但
                每块"最响的那个声音"都活着。对照平均池化的
                [[1.5, 1.75], [1.5, 2.5]]：max 的输出比 avg
                更"尖"，这正是认笔画要的性格。
                """
            ),
            "✅ 练习 2 参考答案（先自己写过再看！）": mo.md(
                """
                循环体一行：

                ```python
                _out[_i, _j] = im[_i*k:(_i+1)*k, _j*k:(_j+1)*k].max()
                ```

                reshape 技巧的完整版：

                ```python
                im.reshape(h//2, 2, w//2, 2).max(axis=(1, 3))
                ```

                读法：把 (26,26) 折成 (13, 2, 13, 2)——块号和
                块内坐标分成四个 axis（07 课），然后**沿块内的
                两个 axis**（1 和 3）取 max（08 课），剩下的就是
                (13,13) 的池化结果。零循环、零拷贝。

                注意一个小坑：H 或 W 是奇数时要先裁掉末行/末列
                （26 恰好是偶数，躲过了）。30 课 LeNet 的 12×12、
                8×8 都专门设计成偶数，就是为了配 2×2 池化。
                """
            ),
            "✅ 练习 3 参考答案（先自己写过再看！）": mo.md(
                """
                **3 → 5 → 6 → 10。** 递推过程（盯住 jump 的变化）：

                | 层 | 计算 | RF | jump |
                |---|---|---|---|
                | 起点 | — | 1 | 1 |
                | conv3 | 1 + 2×1 | **3** | 1 |
                | conv3 | 3 + 2×1 | **5** | 1 |
                | pool2 | 5 + 1×1；jump×2 | **6** | **2** |
                | conv3 | 6 + 2×**2** | **10** | 2 |

                最后一跳 6→10 是本题的戏眼：**同样一个 conv3，
                jump=1 时长 2，jump=2 时长 4**。层"看见多少"
                不只取决于自己的核，还取决于**前面有没有人
                下采样**。这也是为什么 VGG 敢用全 3×3 的小核
                （31 课）：层数堆够，视野自然长；而 LeNet 用
                池化加速——四层就把 RF 推到 16。
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
                1. **最大池化**：2×2 窗口、步长 2、块内留最大；
                   26×26→13×13、参数 **0**、最强信号 2.98→2.98
                   原样保留
                2. **max vs avg**：猎犬 vs 会计——max 聚焦强信号
                   （均值 0.156→0.228）、avg 亮度守恒（0.156→
                   0.156）但把强响应稀释；认笔画用猎犬
                3. **平移容忍的机理**：块内 1px 抖动 → 块最大值
                   不变（固定读出实验：右 1px 池化格 2.99 纹丝
                   不动，卷积格掉到 2.39）；块间出走 → 本格没救
                   但邻格接住——粗网格上同样的位移是更小的相对
                   错位
                4. **大样本定价**：500 样本右移 1px，高保持率
                   （>80%）比例：读卷积格 7% vs 读池化格 40%
                   ——5 倍多的稳定性提升，代价是位置精度减半
                5. **感受野递推**：RF += (K−1)×jump、jump ×= S；
                   纯 conv3 堆每层 +2（3→5→7→9），池化把 jump
                   提到 2 后每个 conv5 +8（5→6→14→16）
                6. **深度换视野**：LeNet 四层后单格 RF=16×16，
                   还没盖住 28×28；FC 读全部 16 格联合覆盖全图；
                   再一层 conv5 → RF=32 超过全图。深，不只是
                   为了非线性，还为了视野
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

                三课下来，CNN 的"前向机器"已经完整：卷积扫边、
                池化减码、感受野逐层长大。但还差最后一台发动机——
                这些卷积核的权重，**到目前为止都是你手写的**。

                手写的竖边核能找边，可它认不出"这是 0 的边还是
                6 的边"。真正的功夫在网络自己**学出**成百上千
                个专用核的时候才开始。
                """
            ),
            mo.callout(
                mo.md(
                    """
                    **下一课预告** — `29_NumPy实现卷积层`：

                    第 5 部分最硬核的一课：**im2col**——把所有
                    窗口摊平成矩阵，卷积瞬间变成一次矩阵乘法
                    （12 课：任何线性操作都是矩阵乘）；然后是
                    压轴大戏 **卷积层的反向传播**：dW、dX 逐个
                    推导，配上 22/23 课同款的**数值梯度对账**
                    （目标精度 10⁻⁸）。

                    学完这课，CNN 的前向 + 反向全在你手里——
                    30 课直接组装 LeNet-5 打 MNIST 冲 99%。

                    🔧 去把发动机装上！
                    """
                ),
                kind="success",
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
