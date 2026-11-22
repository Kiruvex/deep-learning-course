import marimo

__generated_with = "0.25.2"
app = marimo.App(app_title="26 · 卷积直觉", width="medium")


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
        # 26 · 卷积直觉：换一双看图的眼睛

        **从零到 Transformer · 手写深度学习** ｜ 第 5 部分 · 卷积神经网络（1/6）
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📖 本课目标（先读我）": mo.md(
                """
                **这一课只回答一个问题：看图像，到底应该怎么"看"？**

                第 25 课的 MLP 在 MNIST 上拿了 97.78%，剩下的错分大多
                出在"写得歪、写得连"的烂字上。根子不在训练不够，
                而在**全连接的看图方式**本身：每个神经元直勾勾盯着
                全部 784 个像素——像素挪一挪、笔画歪一歪，它看到的
                就是完全不同的输入。

                图像的规律不长这样。认字靠的是**局部**的笔画形状
                （一小块区域内的边、角、端点），而且**同样的形状
                出现在哪都算数**。把这两条直觉写成算法，就是**卷积**。

                学完你将能：

                - 列出全连接处理图像的**三宗罪**：参数挥霍、平移脆弱、
                  结构盲视
                - 用"**小手电扫图**"讲清卷积在做什么：一个小小的
                  权重模板（卷积核）划过整张图，逐位置打分
                - 解释**权值共享**为什么同时解决了参数量和平移
                  两个问题（100,480 个参数 → 80 个，差 1256 倍）
                - 亲手用边缘检测核在真实 MNIST 数字上"找到笔画"

                前置：第 25 课的 MNIST 战报与 2% 错分（今天的案发
                现场）、第 20 课"每个隐单元配齐 784 个权重"的账本、
                第 11 课的内积 = 相似度（卷积核打分的数学本质）。

                本课代码很轻（核心只有一个双重循环），但概念是
                后面五课的全部地基——29 课手写卷积层、30 课 LeNet-5
                都长在今天这颗种子上。
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "🧓 概念 1/3 · 全连接的三宗罪": mo.md(
                """
                回顾 20 课：把 28×28 的图拉平成 784 维向量，第一层
                每个隐单元对**全部** 784 个像素各配一个权重。在图像
                世界里，这种"人人盯着所有像素"的连法有三宗罪：

                **罪一 · 参数挥霍。** 25 课 MLP 的 101,770 个参数里，
                第一层就占 100,480（98.7%）。一个隐单元要学会认
                "左上角有个圆圈"，就得单独长出 784 个权重；想认
                "右下角有个圆圈"，再长 784 个。**同一个形状挪个
                位置，就要重新学一遍。**

                **罪二 · 平移脆弱。** 把数字往右下挪 2 个像素，
                784 维向量就**面目全非**——每个像素位置的值都变了。
                对网络来说这是一张全新的图，之前学的全不作数。
                可人类一看就知道还是同一个字。

                **罪三 · 结构盲视。** 拉平成向量后，像素 (3,4) 和
                (3,5) 是相邻的、(3,4) 和 (20,25) 隔了十万八千里——
                这层信息在向量里**没有任何特殊标记**，网络只能靠
                硬学。而图像的语义恰恰藏在**相邻像素的组合**里
                （笔画、边、角）。

                本课演示 2 会当庭审判罪二：数字挪 2 像素，固定
                位置的神经元响应从 **+2.99 直接翻转到 −2.49**。
                """
            ),
            "🔦 概念 2/3 · 卷积：小手电扫图": mo.md(
                """
                卷积的看图方式只有两条规则：

                **规则一 · 只看局部。** 做一个 3×3 的小窗口（手电筒），
                每次只照亮图上 9 个相邻像素。窗口里装一个
                **卷积核**（kernel）：一个 3×3 的权重小矩阵。
                打分方式你早就认识——**内积**（11 课）：

                ```
                z = Σ (窗口像素 × 核的对应权重) = 9 对数字两两相乘再求和
                ```

                内积大 = 窗口内容和核"长得像"，得分高；内积为零 =
                不像。所以卷积核是一把**模板尺**：本课演示 1 里的
                竖边核 [[1,0,−1],[1,0,−1],[1,0,−1]]，专量"左边亮、
                右边暗"的竖直边缘——左列和减右列和。

                **规则二 · 扫过全图。** 窗口从左上角出发，逐格滑动
                （本课先只滑步长 1），每个位置打一个分。28×28 的图、
                3×3 的核，扫完得到 **26×26 的响应图**（feature
                map）——每个格子记录"这个位置有多像模板"。

                于是"找边"这个任务从"784 维向量上的模式识别"，
                变成了"一把 9 权重的小尺子，把全图量一遍"。
                """
            ),
            "🔁 概念 3/3 · 权值共享：一把尺子量全图": mo.md(
                """
                最关键的决定：**同一个核，扫过所有位置，权重一份
                不动。** 这叫权值共享（weight sharing）。它一箭双雕：

                **① 参数量崩塌式下降。** 8 个 3×3 核 = 8×9 + 8 =
                **80 个参数**，就把整张 28×28 扫了 8 遍（8 张响应图）。
                对照全连接第一层的 100,480 个——**1256 倍的差距**
                （演示 3 的账本）。

                **② 平移不变性免费附赠。** 数字往右挪 2 像素，
                同样的笔画形状只是出现在别的窗口里——**还是那把
                尺子来量它**，响应图整体跟着挪，最大响应纹丝不动
                （演示 2 的 2.99 → 2.99）。全连接做不到：它每个
                位置配独立的权重，形状挪走，原来位置的权重就成了
                废物。

                一句话总结卷积层的全部秘密：

                > **局部连接**（只看 3×3，尊敬图像的空间结构）+
                > **权值共享**（一把尺子量全图，尊敬"哪都有边"的
                > 重复性）——两个先验换来了参数量和平移稳定性。

                27 课补上数学细节（步长、填充、多通道），28 课讲
                怎么把响应图压缩，29 课把它变成可训练的层。
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

        先把 MNIST 请回来（25 课的加载器原样复用，数据已缓存秒开），
        然后做三场实验。
        """
    )
    return


@app.cell
def _(mo, np):
    # MNIST 加载器（与 25 课同款：优先读 data/mnist.npz 缓存，缺失则自动下载）
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

    X_mnist, y_mnist, X_te26, y_te26 = load_mnist()
    print("训练集:", X_mnist.shape, " 测试集:", X_te26.shape)
    return (X_mnist, X_te26, load_mnist, y_mnist, y_te26)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 1 · 三把手工尺：在真实数字上找边

        每行三张图：原图（灰度）→ 竖边核响应 → 横边核响应。
        响应图用红蓝发散色：**红 = 正响应**（长得像模板）、
        **蓝 = 负响应**（长得像模板的"反色"）、白 = 不像。

        - **数字 0**：既有竖笔画（左右两侧）又有横笔画（上下），
          两把尺都大面积点亮——竖边核 45 个强正窗 + 48 个强负窗，
          横边核 40 + 39
        - **数字 1**：几乎纯竖笔画。竖边核点亮 74 个窗口，
          横边核只找到 **5** 个——**钥匙配锁**：核里没有的形状，
          响应图里就是找不到

        注意一个反直觉的细节：竖边核在数字 0 的**左右两侧**分别
        点亮红和蓝——左边缘"左亮右暗"得正分，右边缘"左暗右亮"
        得负分。**符号告诉你边的朝向。**
        """
    )
    return


@app.cell
def _(X_mnist, np, plt, y_mnist):
    # 演示 1 · 手工边缘核 × 真实数字（conv_naive 供全课复用）
    def conv_naive(img, K):
        """最朴素的 2D 卷积：K 为 kh×kw 核，步长 1、无填充。
        输出 (h-kh+1, w-kw+1)。"""
        _h, _w = img.shape
        _kh, _kw = K.shape
        _out = np.zeros((_h - _kh + 1, _w - _kw + 1))
        for _i in range(_out.shape[0]):
            for _j in range(_out.shape[1]):
                _out[_i, _j] = np.sum(img[_i:_i + _kh, _j:_j + _kw] * K)
        return _out

    K_vert = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float64)  # 竖边
    K_horz = K_vert.T.copy()                                                   # 横边（转置即可）

    _X2d = X_mnist.reshape(-1, 28, 28)
    img_zero, img_one = _X2d[1], _X2d[8]      # 固定样张：数字 0 和数字 1
    _rows = [("数字 0（训练集 #1）", img_zero), ("数字 1（训练集 #8）", img_one)]

    _fig, _axes = plt.subplots(2, 3, figsize=(11.5, 7.0))
    for _r, (_ttl, _im) in enumerate(_rows):
        _rv, _rh = conv_naive(_im, K_vert), conv_naive(_im, K_horz)
        _axes[_r, 0].imshow(_im, cmap="gray_r")
        _axes[_r, 0].set_title(f"{_ttl}", fontsize=11)
        _vmax = max(np.abs(_rv).max(), np.abs(_rh).max())
        _axes[_r, 1].imshow(_rv, cmap="RdBu_r", vmin=-_vmax, vmax=_vmax)
        _axes[_r, 1].set_title(f"竖边核：强响应窗 {(np.abs(_rv) > 1.5).sum()} 个\n"
                               f"（正 {( _rv > 1.5).sum()} / 负 {(_rv < -1.5).sum()}）", fontsize=10)
        _axes[_r, 2].imshow(_rh, cmap="RdBu_r", vmin=-_vmax, vmax=_vmax)
        _axes[_r, 2].set_title(f"横边核：强响应窗 {(np.abs(_rh) > 1.5).sum()} 个\n"
                               f"（正 {(_rh > 1.5).sum()} / 负 {(_rh < -1.5).sum()}）", fontsize=10)
        if _r == 0:
            print(f"数字 0：竖边核范围 [{_rv.min():.2f}, {_rv.max():.2f}]，"
                  f"横边核范围 [{_rh.min():.2f}, {_rh.max():.2f}]")
        else:
            print(f"数字 1：竖边核强响应 {(np.abs(_rv) > 1.5).sum()} 窗，"
                  f"横边核强响应 {(np.abs(_rh) > 1.5).sum()} 窗——钥匙配锁")
    for _ax in _axes.ravel():
        _ax.axis("off")
    _fig.suptitle("手工卷积核在真实手写数字上的响应图（红=正 / 蓝=负 / 白=零）", fontsize=12)
    _fig.tight_layout()
    _fig
    return (K_horz, K_vert, conv_naive, img_one)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 2 · 当庭审判"平移罪"：固定窗 vs 扫描模板

        把数字 1 往右下平移 **2 像素**，看两种神经元各自的遭遇：

        - **固定位置神经元**（全连接的看图方式）：只盯死
          (7,14) 那个 3×3 窗口——原图里它恰好贴着笔画，响应
          **+2.99**；数字挪走后，窗口里只剩笔画的灰边——响应
          **翻转到 −2.49**。它看到的"世界"整个换了一批像素
        - **卷积模板**（同一把尺子扫全图）：最大响应
          **2.99 → 2.99，一个字都不带变的**。笔画挪到哪个窗口，
          那个窗口就报同样的分——响应图只是整体跟着挪了 2 格

        最右下角的"对齐差异图"是铁证：把两张响应图按平移量对齐
        后相减，**处处为零**（最大差 0.00）。而随机抽查 200 个
        固定窗，平均响应变化 0.54——**全连接式看法对平移的敏感
        是结构性的，不是巧合。**
        """
    )
    return


@app.cell
def _(K_vert, conv_naive, img_one, np, plt):
    # 演示 2 · 平移 2px：固定窗神经元 vs 卷积模板
    def shift_img(im, dy, dx):
        """把图往 (下, 右) 平移 dy/dx 像素，空出的地方填 0。"""
        _out = np.zeros_like(im)
        _h, _w = im.shape
        _ys, _ye = max(0, dy), min(_h, _h + dy)
        _xs, _xe = max(0, dx), min(_w, _w + dx)
        _out[_ys:_ye, _xs:_xe] = im[_ys - dy:_ye - dy, _xs - dx:_xe - dx]
        return _out

    img_one_shift = shift_img(img_one, 2, 2)                 # 右下平移 2 像素
    r_orig, r_shift = conv_naive(img_one, K_vert), conv_naive(img_one_shift, K_vert)

    i_fix, j_fix = 7, 14                                     # 固定窗盯死的位置
    z_fixed_orig = float(np.sum(img_one[i_fix:i_fix+3, j_fix:j_fix+3] * K_vert))
    z_fixed_shift = float(np.sum(img_one_shift[i_fix:i_fix+3, j_fix:j_fix+3] * K_vert))
    aligned_diff = r_orig[:-2, :-2] - r_shift[2:, 2:]        # 按平移量对齐后相减

    _rng = np.random.default_rng(0)                          # 200 个随机固定窗的统计
    _diffs = [abs(np.sum(img_one[_i:_i+3, _j:_j+3] * K_vert)
                  - np.sum(img_one_shift[_i:_i+3, _j:_j+3] * K_vert))
              for _i, _j in _rng.integers(0, 26, (200, 2))]
    print(f"固定窗 ({i_fix},{j_fix})：{z_fixed_orig:+.2f} → {z_fixed_shift:+.2f}（符号翻转）")
    print(f"卷积模板最大响应：{r_orig.max():.4f} → {r_shift.max():.4f}（纹丝不动）")
    print(f"200 个随机固定窗的平均响应变化：{np.mean(_diffs):.4f}")
    print(f"响应图对齐后最大差异：{np.abs(aligned_diff).max():.6f}")

    _fig, _axes = plt.subplots(2, 3, figsize=(11.5, 7.2))
    for _ax, _im, _ttl, _zf in [
        (_axes[0, 0], img_one, "原图（数字 1）", z_fixed_orig),
        (_axes[0, 1], img_one_shift, "平移 2 像素后", z_fixed_shift),
    ]:
        _ax.imshow(_im, cmap="gray_r")
        _ax.add_patch(plt.Rectangle((j_fix - .5, i_fix - .5), 3, 3,
                                    fill=False, color="red", lw=2))
        _ax.set_title(f"{_ttl}\n红框固定窗响应：{_zf:+.2f}", fontsize=10)
        _ax.axis("off")
    _axes[0, 2].axis("off")
    _axes[0, 2].text(0.5, 0.5,
                     f"固定窗的遭遇：\n\n{z_fixed_orig:+.2f} → {z_fixed_shift:+.2f}\n\n"
                     "笔画走出窗口，\n响应直接翻转符号\n（全连接视角）",
                     ha="center", va="center", fontsize=13, color="tab:red",
                     transform=_axes[0, 2].transAxes)
    _vmax = max(np.abs(r_orig).max(), np.abs(r_shift).max())
    _axes[1, 0].imshow(r_orig, cmap="RdBu_r", vmin=-_vmax, vmax=_vmax)
    _axes[1, 0].set_title(f"原图响应图\n最大 {r_orig.max():.2f}", fontsize=10)
    _axes[1, 1].imshow(r_shift, cmap="RdBu_r", vmin=-_vmax, vmax=_vmax)
    _axes[1, 1].set_title(f"平移后响应图（跟着挪 2 格）\n最大 {r_shift.max():.2f}", fontsize=10)
    _axes[1, 2].imshow(aligned_diff, cmap="RdBu_r", vmin=-0.05, vmax=0.05)
    _axes[1, 2].set_title(f"对齐后差异图：最大 |diff| = {np.abs(aligned_diff).max():.4f}", fontsize=10)
    _axes[1, 2].text(13, 13, "整张雪白\n正是铁证", ha="center", va="center",
                     fontsize=15, color="dimgray", fontweight="bold")
    _axes[1, 2].text(13, 22.5, "两张响应图按平移量对齐后相减：\n676 个位置全部为零——\n响应图只是整体挪了 2 格，\n内容分毫未动",
                     ha="center", va="center", fontsize=9.5, color="gray")
    for _ax in _axes[1]:
        _ax.axis("off")
    _fig.suptitle("审判平移罪：固定窗翻车（红框），扫描模板稳如泰山（下排）", fontsize=12)
    _fig.tight_layout()
    _fig
    return (img_one_shift, r_orig, r_shift, shift_img, z_fixed_orig, z_fixed_shift)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 3 · 参数账本：100,480 vs 80

        同样"看一遍 28×28"，两种连法的开销（对数坐标柱状图，
        柱子高度差 1256 倍）：

        | 方案 | 看图方式 | 参数量 | 每个输出看了多少输入 |
        |---|---|---|---|
        | 全连接 784→128 | 每个神经元盯全部 784 像素 | **100,480**（784×128+128） | 784 |
        | 卷积 8×(3×3) | 8 把小尺，局部扫描、权重共享 | **80**（8×9+8） | 9 |

        顺带一笔账：25 课 MLP 的 101,770 个参数里，第一层全连接
        就占了 98.7%——**大头全花在"看图"上了**。卷积把这笔钱
        省下来，30 课会用省下的参数去堆更多层、看得更远。

        还有一个伏笔：卷积每个输出只看了 9 个输入像素（"感受野"
        只有 3×3）——视野这么小，凭什么能认整个数字？答案是
        **层叠**：28 课讲感受野如何一层层长大。
        """
    )
    return


@app.cell
def _(np, plt):
    # 演示 3 · 参数账本（对数坐标）
    fc_params, conv_params = 784 * 128 + 128, 8 * (3 * 3) + 8
    mlp_total = 784 * 128 + 128 + 128 * 10 + 10          # 25 课 MLP 全部参数
    print(f"全连接 784→128：{fc_params:,} 参数")
    print(f"卷积 8×(3×3)：{conv_params} 参数，倍数 {fc_params / conv_params:,.0f}×")
    print(f"25 课 MLP 共 {mlp_total:,} 参数，第一层占 {fc_params / mlp_total:.1%}")

    _fig, _ax = plt.subplots(figsize=(7.8, 4.3))
    _bars = _ax.bar(["全连接第一层\n784→128", "卷积层\n8×(3×3)"],
                    [fc_params, conv_params], color=["tab:red", "tab:blue"], width=0.5)
    _ax.set_yscale("log")
    _ax.set_ylim(1, 1e6)
    _ax.set_ylabel("参数量（对数坐标）")
    for _b, _v in zip(_bars, [fc_params, conv_params]):
        _ax.text(_b.get_x() + _b.get_width() / 2, _v * 3.0, f"{_v:,}",
                 ha="center", fontsize=12, fontweight="bold")
    _ax.annotate("", xy=(1.12, conv_params * 1.4), xytext=(0.18, fc_params * 0.6),
                 arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
    _ax.text(0.52, fc_params * 0.09, f"差 {fc_params / conv_params:,.0f} 倍",
             fontsize=13, color="gray", rotation=-18)
    _ax.set_title("两种「看一遍 28×28」的开销：局部 + 共享 vs 全连接")
    _ax.grid(True, axis="y", alpha=0.3)
    _fig.tight_layout()
    _fig
    return (conv_params, fc_params)


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
        ### 交互实验 1 · 核工坊：亲手打造一把尺子

        左边选**预设核**（或选"自定义"后拖动中间 3×3 的九个滑块），
        右边换测试样张。看三样东西：**核长什么样**、**响应图在哪亮**、
        **强响应窗有几个**。

        试试这些玩法：

        - 把竖边核换成横边核，数字 1 的响应图几乎全灭，数字 0 依旧坚挺
        - 自定义一个"对角核"，专找斜笔画（提示：数字 7 的顶端）
        - 把核全部填 1：求和核——它像谁？（想想 08 课的 sum）
        - 把核改成只有中心是 1：响应图会变成"缩了一圈的原图"
        """
    )
    return


@app.cell
def _(mo):
    # 交互控制：预设核 + 自定义 3×3 滑块 + 样张选择
    preset_dropdown = mo.ui.dropdown(
        options=["竖边", "横边", "对角", "模糊", "自定义（拖下面的滑块）"], value="竖边",
        label="📐 卷积核",
    )
    digit_slider = mo.ui.slider(0, 15, value=8, step=1, label="🖼️ 测试样张（训练集前 16 张）")
    kgrid = [mo.ui.slider(-2, 2, value=v, step=1, debounce=True, label=f"({r},{c})")
             for r, c, v in [(0, 0, 1), (0, 1, 0), (0, 2, -1),
                             (1, 0, 1), (1, 1, 0), (1, 2, -1),
                             (2, 0, 1), (2, 1, 0), (2, 2, -1)]]
    mo.vstack([
        mo.hstack([preset_dropdown, digit_slider]),
        mo.md("**自定义核的 9 个格子**（选「自定义」时生效，摆成 3×3 矩阵的样子）："),
        mo.vstack([mo.hstack(kgrid[0:3]), mo.hstack(kgrid[3:6]), mo.hstack(kgrid[6:9])]),
    ])
    return (digit_slider, kgrid, preset_dropdown)


@app.cell
def _(K_horz, K_vert, X_mnist, conv_naive, digit_slider, kgrid, mo, np, plt, preset_dropdown, y_mnist):
    # ===== 交互实验 1 · 核工坊（响应单元）=====
    _presets = {"竖边": K_vert, "横边": K_horz,
                "对角": np.array([[0, 1, 0], [1, 0, -1], [0, -1, 0]], dtype=np.float64),
                "模糊": np.ones((3, 3)) / 9.0}
    _sel = preset_dropdown.value or "竖边"
    if _sel == "自定义（拖下面的滑块）":
        K_user = np.array([[_s.value for _s in _row] for _row in
                           [kgrid[0:3], kgrid[3:6], kgrid[6:9]]], dtype=np.float64)
        _kname = "自定义"
    else:
        K_user = _presets[_sel]
        _kname = _sel

    _img = X_mnist.reshape(-1, 28, 28)[digit_slider.value]
    _resp = conv_naive(_img, K_user)
    _vmax = max(np.abs(_resp).max(), 1e-9)

    _fig, _axes = plt.subplots(1, 3, figsize=(11.6, 3.9),
                               gridspec_kw={"width_ratios": [1, 1, 1.1]})
    _axes[0].imshow(K_user, cmap="RdBu_r", vmin=-2, vmax=2)
    for _i in range(3):
        for _j in range(3):
            _axes[0].text(_j, _i, f"{K_user[_i, _j]:.2f}".rstrip("0").rstrip("."),
                          ha="center", va="center", fontsize=13,
                          color="white" if abs(K_user[_i, _j]) > 1 else "black")
    _axes[0].set_title(f"核：{_kname}", fontsize=11)
    _axes[1].imshow(_img, cmap="gray_r")
    _axes[1].set_title(f"输入（数字 {int(y_mnist[digit_slider.value])}）", fontsize=11)
    _axes[2].imshow(_resp, cmap="RdBu_r", vmin=-_vmax, vmax=_vmax)
    _axes[2].set_title("响应图（红=正 / 蓝=负）", fontsize=11)
    for _ax in _axes:
        _ax.axis("off")
    _fig.tight_layout()

    # ---- 判词：认一认这把尺子是什么类型 ----
    _n_strong = int((np.abs(_resp) > 1.5).sum())
    if np.abs(K_user).max() < 1e-9:
        _kind = "全零核——响应图一片白，什么都看不见"
    elif np.abs(K_user.max() - K_user.min()) < 1e-9:
        _kind = "均匀核（全是一样的数）——它在求窗口的平均/总和，是个模糊尺"
    elif abs(K_user.sum()) < 1e-9 and np.abs(K_user).max() > 1e-9:
        _kind = "差分核（权重总和为 0）——均匀区域一律零响应，专门找「变化」：边"
    else:
        _kind = "中心核（中心权重大）——强化中心像素、压制邻居：锐化或复制"
    _verdict = (f"**{_kname}核** · 强响应窗 **{_n_strong}**/676 · "
                f"响应范围 [{_resp.min():.2f}, {_resp.max():.2f}] ｜ 识别：{_kind}")

    mo.vstack([
        _fig,
        mo.callout(mo.md(_verdict), kind="info"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 交互实验 2 · 扫描小手电：亲眼看一个输出像素的诞生

        拖动**行 / 列**滑块，把 3×3 小手电照到数字 1 的任意位置。
        左图的红框是手电照亮的地方，右图红圈是这次的得分落在
        响应图的哪一格。

        竖边核的打分公式有个可爱的化简：核的三列是 [1, 0, −1]，
        所以 **z = 窗口左列之和 − 右列之和**——它只比较窗口左右
        两半谁更亮。默认位置 (7,14) 就是全图最强响应 +2.99 的
        出生地：笔画正好贴着窗口左列。
        """
    )
    return


@app.cell
def _(mo):
    # 交互控制：手电位置
    row_slider = mo.ui.slider(0, 25, value=7, step=1, label="⬇️ 窗口所在行 i（0~25）")
    col_slider = mo.ui.slider(0, 25, value=14, step=1, label="➡️ 窗口所在列 j（0~25）")
    mo.hstack([row_slider, col_slider])
    return (col_slider, row_slider)


@app.cell
def _(K_vert, conv_naive, col_slider, img_one, mo, np, plt, row_slider):
    # ===== 交互实验 2 · 扫描小手电（响应单元）=====
    _i, _j = row_slider.value, col_slider.value
    _patch = img_one[_i:_i + 3, _j:_j + 3]                 # 手电照亮的 3×3
    _z = float(np.sum(_patch * K_vert))
    _left_sum, _right_sum = float(_patch[:, 0].sum()), float(_patch[:, 2].sum())
    _resp = conv_naive(img_one, K_vert)
    _rank = int((np.abs(_resp) > abs(_z)).sum())           # |z| 排名

    _fig, _axes = plt.subplots(1, 3, figsize=(11.6, 3.9),
                               gridspec_kw={"width_ratios": [1, 1, 1.1]})
    _axes[0].imshow(img_one, cmap="gray_r")
    _axes[0].add_patch(plt.Rectangle((_j - .5, _i - .5), 3, 3,
                                     fill=False, color="red", lw=2.5))
    _axes[0].set_title(f"输入：窗口({_i},{_j}) 处的 3×3", fontsize=11)
    _axes[1].imshow(_patch, cmap="gray_r", vmin=0, vmax=1)
    _axes[1].set_title(f"窗口内容（放大）\n左列和 {_left_sum:.2f} − 右列和 {_right_sum:.2f}", fontsize=11)
    _vmax = max(np.abs(_resp).max(), 1e-9)
    _axes[2].imshow(_resp, cmap="RdBu_r", vmin=-_vmax, vmax=_vmax)
    _axes[2].scatter(_j, _i, s=160, facecolors="none", edgecolors="red",
                     linewidths=2.5, zorder=5)
    _axes[2].set_title(f"响应图：({_i},{_j}) 处 = {_z:+.2f}", fontsize=11)
    for _ax in _axes:
        _ax.axis("off")
    _fig.tight_layout()

    if abs(_z) > 1.5:
        _side = "左亮右暗" if _z > 0 else "左暗右亮"
        _verdict = (f"**强响应 {_z:+.2f}**（全图第 {_rank + 1} 强）——"
                    f"窗口里有一条{_side}的竖直边缘，正中竖边核下怀")
    elif abs(_z) > 0.5:
        _verdict = f"**中等响应 {_z:+.2f}**——窗口里有边缘的影子（抗锯齿的灰边）"
    else:
        _verdict = (f"**安静区 {_z:+.2f}**——窗口左右两半一样亮（全白或全黑），"
                    "竖边核对它无话可说")

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
                **练习 1 · 手算一个卷积输出**：给一张 5×5 的小图
                （竖笔画在第 1 列）和竖边核，**笔算**两个输出值：
                笔画在窗口正中间列的 out[1,0]、笔画贴着窗口最左列的
                out[1,1]。先在纸上算，再填进代码对账。

                **练习 2 · 亲手实现 conv_naive**：补全双重循环里的
                一行——窗口 × 核再求和。跑在数字 0 × 横边核上，
                和 NumPy 滑窗视图的向量化实现逐元素对账
                （26×26 = 676 个数全对才算过）。

                **练习 3 · 核侦探**：三个匿名核 A/B/C 在数字 0 上的
                响应图已画出，旁边有四张"身份卡"：恒等、模糊、锐化、
                竖边。看核的形状 + 响应图的长相，给每个核配上身份
                （有一张卡是烟雾弹）。

                规则不变：先把 `None` 换成你的答案再运行；写完再看参考答案。
                """
            )
        }
    )
    return


@app.cell
def _(K_vert, conv_naive, mo, np):
    # ===== 练习 1 · 手算卷积输出 =====
    img_ex1 = np.zeros((5, 5))
    img_ex1[1:4, 1] = 1.0                    # 一条 3 像素高的竖笔画，贴着第 1 列
    print("5×5 输入图（笔画在第 1 列）：\n", img_ex1)
    print("竖边核：\n", K_vert)

    out_mid_ex1 = None                       # ← ① 笔算 out[1,0]：笔画在窗口「正中间」列
    out_left_ex1 = None                      # ← ② 笔算 out[1,1]：笔画贴着窗口「最左」列

    _ref = conv_naive(img_ex1, K_vert)
    if out_mid_ex1 is not None and out_left_ex1 is not None:
        _ok = (abs(out_mid_ex1 - _ref[1, 0]) < 1e-9
               and abs(out_left_ex1 - _ref[1, 1]) < 1e-9)
        _msg = (f"✅ 全对！out[1,0] = {_ref[1,0]:.0f}（中间列：左右两半一样亮，得 0）、"
                f"out[1,1] = {_ref[1,1]:.0f}（最左列：左列 3 − 右列 0）"
                if _ok else
                f"❌ 参考：out[1,0] = {_ref[1,0]:.0f}、out[1,1] = {_ref[1,1]:.0f}。"
                "提示：z = 窗口左列和 − 右列和；把窗口 img_ex1[i:i+3, j:j+3] 抄到纸上算")
    else:
        _msg = "⏳ 两个值都笔算出来再填（提示：z = 窗口左列之和 − 窗口右列之和）"
    mo.md(_msg)
    return (img_ex1, out_left_ex1, out_mid_ex1)


@app.cell
def _(X_mnist, mo, np):
    # ===== 练习 2 · 亲手实现 conv_naive =====
    def conv_loop_ex2(img, K):
        """补全循环体：窗口 × 核，再求和，写进 _out[i, j]。"""
        _h, _w = img.shape
        _kh, _kw = K.shape
        _out = np.zeros((_h - _kh + 1, _w - _kw + 1))
        for _i in range(_out.shape[0]):
            for _j in range(_out.shape[1]):
                _out[_i, _j] = None          # ← 补：img 的窗口 与 K 逐元素相乘再求和
        return _out

    conv_ready_ex2 = None                    # ← 补完后，把这行换成 conv_loop_ex2

    if conv_ready_ex2 is not None:
        try:
            _K = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float64)  # 横边核
            _img = X_mnist.reshape(-1, 28, 28)[1]                                  # 数字 0
            _got = conv_ready_ex2(_img, _K)
            from numpy.lib.stride_tricks import sliding_window_view
            _win = sliding_window_view(_img, (3, 3))                               # (26,26,3,3) 滑窗视图
            _ref = np.einsum("ijkl,kl->ij", _win, _K)                              # 向量化参考
            if np.allclose(_got, _ref, atol=1e-10):
                _msg = (f"✅ 676 个输出值与滑窗视图实现全部一致！"
                        f"数字 0 × 横边核：max {_got.max():.2f}，min {_got.min():.2f}"
                        "——27 课正式给这套操作上数学户口")
            else:
                _msg = (f"❌ 有 {(~np.isclose(_got, _ref, atol=1e-10)).sum()} 个位置对不上——"
                        "检查循环体：img[_i:_i+3, _j:_j+3] * K 再 .sum()")
        except (TypeError, ValueError) as _e:
            _msg = f"❌ 运行出错：{_e}——循环体那行还没补对（np.sum(窗口 * 核)）"
    else:
        _msg = "⏳ 先补循环体一行，再把 conv_ready_ex2 换成 conv_loop_ex2"
    mo.md(_msg)
    return (conv_loop_ex2, conv_ready_ex2)


@app.cell
def _(X_mnist, mo, np, plt):
    # ===== 练习 3 · 核侦探：看响应图配身份 =====
    K_A = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float64)
    K_B = np.ones((3, 3)) / 9.0
    K_C = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float64)

    def conv3_ex3(img, K):
        _h, _w = img.shape
        _out = np.zeros((_h - 2, _w - 2))
        for _i in range(_h - 2):
            for _j in range(_w - 2):
                _out[_i, _j] = np.sum(img[_i:_i + 3, _j:_j + 3] * K)
        return _out

    _img = X_mnist.reshape(-1, 28, 28)[1]                   # 数字 0
    _fig, _axes = plt.subplots(2, 3, figsize=(10.8, 6.6))
    for _c, (_K, _nm) in enumerate([(K_A, "核 A"), (K_B, "核 B"), (K_C, "核 C")]):
        _axes[0, _c].imshow(_K, cmap="RdBu_r", vmin=-1, vmax=8)
        for _i in range(3):
            for _j in range(3):
                _axes[0, _c].text(_j, _i, f"{_K[_i, _j]:.2f}", ha="center", va="center",
                                  fontsize=10, color="black")
        _axes[0, _c].set_title(_nm, fontsize=12)
        _axes[0, _c].axis("off")
        _r = conv3_ex3(_img, _K)
        _axes[1, _c].imshow(_r, cmap="RdBu_r", vmin=-2, vmax=4)
        _axes[1, _c].set_title(f"{_nm} 在数字 0 上的响应", fontsize=11)
        _axes[1, _c].axis("off")
    _fig.suptitle("三把匿名尺子：它们各自的响应图说明了什么身份？", fontsize=12)
    _fig.tight_layout()
    _fig

    # 身份卡：恒等 / 模糊 / 锐化 / 竖边（有一张是烟雾弹）
    ans_A_ex3 = None                         # ← ① 核 A 的身份（四选一字符串）
    ans_B_ex3 = None                         # ← ② 核 B 的身份
    ans_C_ex3 = None                         # ← ③ 核 C 的身份
    return (K_A, K_B, K_C, ans_A_ex3, ans_B_ex3, ans_C_ex3)


@app.cell
def _(ans_A_ex3, ans_B_ex3, ans_C_ex3, mo):
    # ===== 练习 3 · 判词 =====
    _truth = {"A": "恒等", "B": "模糊", "C": "锐化"}
    _ans = {"A": ans_A_ex3, "B": ans_B_ex3, "C": ans_C_ex3}
    if any(_v is None for _v in _ans.values()):
        _msg = "⏳ 三个身份都填上（字符串：'恒等' / '模糊' / '锐化' / '竖边'）"
    else:
        _right = [_k for _k in _truth if _ans[_k] == _truth[_k]]
        _wrong = [_k for _k in _truth if _ans[_k] != _truth[_k]]
        if not _wrong:
            _msg = ("✅ 三案全破！A 只看中心像素 → 响应图就是缩小的原图；"
                    "B 是均值 → 笔画被抹圆；C 是 8×中心−邻居 → 只在笔画边缘亮，"
                    "数字的轮廓清晰浮出。至于「竖边」——从响应图左右对称就能排除")
        else:
            _hint = {"A": "A 的响应图看着是不是特别像原图本身？",
                     "B": "B 的核九个格子一样大，它在求平均——像谁？",
                     "C": "C 的核中心 8、四周 −1，均匀区域得 0、边缘得高分"}
            _msg = "❌ 还差：" + "；".join(f"{_k}（{_hint[_k]}）" for _k in _wrong)
    mo.md(_msg)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 1 参考答案（先自己写过再看！）": mo.md(
                """
                **out[1,0] = 0，out[1,1] = 3。**

                - out[1,0] 的窗口是 img[1:4, 0:3]：笔画（第 1 列）
                  正好落在窗口**中间列**。竖边核 z = 左列和 − 右列和
                  = 0 − 0 = **0**——笔画居中时，"左边 vs 右边"
                  比不出差别。这就是"边缘检测器"与"线条检测器"
                  的区别：竖边核找的是**明暗跳变**，不是线本身
                - out[1,1] 的窗口是 img[1:4, 1:4]：笔画贴着**最左列**，
                  左列和 = 3、右列和 = 0，z = **3**——满分的
                  "左亮右暗"

                顺手算 out[1,2]（笔画在窗口最左列再过去一格？
                  不，img[1:4, 2:5] 三列全空）= 0。整张响应图：

                ```
                [[0, 2, 0],
                 [0, 3, 0],     ← 笔画最亮的一段（3 行长的中部）
                 [0, 2, 0]]
                ```

                只在笔画**左侧边缘**的窗口上有响应——和演示 1
                里数字 0 左边缘亮红、右边缘亮蓝是同一回事。
                """
            ),
            "✅ 练习 2 参考答案（先自己写过再看！）": mo.md(
                """
                循环体只需一行：

                ```python
                _out[_i, _j] = np.sum(img[_i:_i + 3, _j:_j + 3] * K)
                ```

                切片 `img[_i:_i+3, _j:_j+3]` 取出窗口（04 课），
                `* K` 逐元素相乘（06 课广播的特例：形状相同的
                两个数组），`np.sum` 全部加起来（08 课 axis 的
                全体）。**卷积的前向 = 三课旧知识的组装。**

                对账用的滑窗视图 `sliding_window_view` 是 NumPy
                藏的宝：不复制数据，把 28×28 的图"看"成
                26×26×3×3 的四维窗阵，再让 einsum 一次算完所有
                窗口——**零 Python 循环**。它就是 29 课 im2col
                工业加速的近亲，届时细讲。
                """
            ),
            "✅ 练习 3 参考答案（先自己写过再看！）": mo.md(
                """
                **A = 恒等，B = 模糊，C = 锐化。竖边是烟雾弹。**

                - **A（只有中心是 1）**：z = 中心像素值。响应图
                  精确等于原图往里缩一圈——所谓"恒等"，就是
                  28×28 → 26×26 的复制机。验算：
                  `np.allclose(conv3(数字0, K_A), 数字0[1:27, 1:27])` 为 True
                - **B（九个 1/9）**：z = 窗口 9 个像素的平均值
                  （08 课的 mean）。笔画边缘的抗锯齿灰阶被抹匀，
                  整张图变"糊"——模糊
                - **C（中心 8、四周 −1）**：均匀区域 8a − 8a = 0
                  （权重和为 0），任何跳变处大幅点亮——锐化 /
                  边缘增强，数字 0 的轮廓在响应图里发亮
                - **竖边怎么排除**：三个核里没有一个长着
                  [1,0,−1] 的列结构；A/B 的响应图都关于数字
                  对称轴左右对称，C 只在轮廓亮——都找不到
                  "左红右蓝"的竖边签名
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
                1. **全连接三宗罪**：参数挥霍（第一层 100,480 个，
                   占 25 课 MLP 的 98.7%）、平移脆弱（固定窗响应
                   +2.99 → −2.49 符号翻转）、结构盲视（拉平后
                   相邻关系无标记）
                2. **卷积 = 小手电扫图**：3×3 窗口 + 9 权重的核，
                   逐位置打分；打分就是内积（11 课）——核是模板，
                   内积量相似度
                3. **权值共享**：同一把尺子量全图。参数 80 个
                   （8 核），比全连接少 1256 倍；平移时响应图
                   整体跟着挪，最大响应 2.99 → 2.99 不动
                4. **响应图的读法**：红正蓝负白零；竖边核在
                   数字 0 左右两侧分别亮红与亮蓝——**符号编码
                   边的朝向**；钥匙配锁：数字 1 上横边核只有
                   5 个强响应窗
                5. **边界 vs 线条**：竖边核对"笔画居中"的窗口
                   得 0（左右一样亮），只在明暗跳变处得分
                   （练习 1 的 out[1,0]=0 vs out[1,1]=3）
                6. **28×28 → 26×26**：核扫过的合法位置数
                   (28−3+1)²；这个"缩水"规律 27 课升级成通用
                   公式，配步长与填充两枚旋钮
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

                今天你把"看图的正确姿势"建立了起来：**局部窗口 +
                权重模板 + 扫全图**。但现在的卷积还有三个"没说清"：

                - 为什么 28×28 扫完只剩 26×26？一层层扫下去，
                  图会不会缩到没有？（**输出尺寸公式**）
                - 真实图像是彩色 RGB 三通道的，核怎么同时看三个
                  通道？（**多通道卷积**）
                - 扫描能不能跳着扫、边上能不能补一圈？（**步长
                  与填充**）
                """
            ),
            mo.callout(
                mo.md(
                    """
                    **下一课预告** — `27_卷积数学`：

                    把今天的手电筒升级成带旋钮的仪器：**步长**
                    （跳着扫，一步并两步）、**填充**（在图边补零，
                    让输出不缩水）、**输出尺寸公式** (H−K+2P)/S+1
                    ——一行公式回答所有"扫完剩多大"的问题。

                    然后是三通道合议制：一张 RGB 图进、多个核出，
                    每个核把三个通道的"意见"加权求和成一张响应图。
                    28 课的池化、29 课的手写卷积层，全都站在今天
                    这套数学上。

                    🔭 手电筒已点亮，去把公式钉死！
                    """
                ),
                kind="success",
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
