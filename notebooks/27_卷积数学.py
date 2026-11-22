import marimo

__generated_with = "0.25.2"
app = marimo.App(app_title="27 · 卷积数学", width="medium")


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
        # 27 · 卷积数学：步长、填充与通道

        **从零到 Transformer · 手写深度学习** ｜ 第 5 部分 · 卷积神经网络（2/6）
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📖 本课目标（先读我）": mo.md(
                """
                **上一课的手电筒很好用，但还没装旋钮。**

                26 课的卷积只会一种扫法：3×3 窗口、一格一格挪、
                贴着边扫。这节课把三个旋钮装上，卷积就从"玩具"
                变成工程里真正可调的零件：

                - **步长 S**：跳着扫。S=2 一步并两步，输出直接减半
                - **填充 P**：在图四周补零。既保住输出尺寸，
                  又让边缘像素被公平地看见
                - **多通道**：真实图像是 RGB 三层的，核得学会
                  同时看三层、把意见**加权合议**成一张响应图

                三件东西最后汇成**一行公式**——输出尺寸公式，
                它回答所有"扫完剩多大"的问题，也是你读任何
                CNN 论文、看任何框架报错时的翻译器。

                学完你将能：

                - 用公式 **out = (H + 2P − K) / S + 1** 秒算任意
                  配置的输出尺寸，并说出每个字母的几何含义
                - 讲清**填充为什么存在**：不是保尺寸这么简单——
                  无填充时角落像素只被 1 个窗口看过、中心被看
                  9 次（演示 2 的覆盖计数热图）
                - 解释**多通道卷积**：核是 C_in×K×K 的三维块，
                  通道维与窗口维**一起求和**——"合议制"
                - 算出 LeNet-5 第一层为什么是 28→24（30 课的伏笔）

                前置：26 课的 conv_naive、响应图读法（红正蓝负）。
                本课全部公式都配"几何直觉版推导"，不看代数也能懂。
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📏 概念 1/4 · 输出尺寸公式：一行公式管所有": mo.md(
                """
                26 课埋的疑问：28×28 的图、3×3 的核，扫完为什么剩
                26×26？几何直觉一秒讲清——**窗口的左上角能落在哪**：

                窗口左上角最远只能落在第 H−K 行（再往下窗口就出界
                了），所以合法位置从 0 到 H−K 共 **H−K+1** 个。
                这就是 26×26 的 26 = 28−3+1。

                加上步长（窗口每次右挪 S 格）和填充（四周各补 P
                圈零，等效把图放大到 H+2P），合法位置变成：

                ```
                out = (H + 2P − K) / S + 1        （先除后加 1：位置数 = 范围/步长 + 1）
                ```

                读法：**分子 = 窗口左上角的活动范围**（放大后的边长
                减去核宽），除以步长 = 能停几站，加 1 = 起点那一站。

                两个必考推论：

                - **不整除怎么办？** 比如 H=28, K=3, S=2, P=1：
                  (28+2−3)/2+1 = 14.5。真实框架（PyTorch 等）一律
                  **向下取整**得 14——代价是最右边/最下边有半格
                  像素被跳过没扫到。工程里通常**调 P 让它整除**
                - **same 配方**：想让输出=输入尺寸，解方程得
                  **P = (K−1)/2**。K=3 → P=1；K=5 → P=2；K=7 → P=3
                  ——这就是为什么全世界的卷积核都用**奇数**尺寸：
                  偶数核的 (K−1)/2 不是整数，same 配方配不出来
                """
            ),
            "👟 概念 2/4 · 步长：跳着扫的下采样": mo.md(
                """
                步长 S=1 是"一步一步走"；S=2 是"每两格停一站"。

                - S=1：H=28, K=3, P=0 → 26×26，每个窗口位置都打分
                - S=2：同样的图 → **13×13**，计算量约剩 1/4
                  （站数减半 × 每站照算 → 输出像素数 26²→13²）

                它的真正身份是**下采样器**：分辨率换速度。你将看到
                响应图明显变"马赛克"——每 2×2 的输入块只保留一个
                采样点。28 课会讲它的老搭档**池化**（另一种下采样，
                路数不同）；31 课讲现代网络"用步长卷积替代池化"
                的趋势。

                直觉类比：看航拍图找机场，不需要一米一米地看——
                每十米看一眼（大步长）足够锁定位置，然后对小区域
                再细看（细步长）。**粗扫定位、细扫确认**。
                """
            ),
            "🖼️ 概念 3/4 · 填充：给边缘像素一个公平的机会": mo.md(
                """
                无填充扫描有个隐藏的不公平：**角落像素只被 1 个窗口
                看过，中心像素被看 9 次**（3×3 的窗口盖住中心的
                方式有 3×3 = 9 种摆放）。边缘信息天生吃亏——
                而手写数字的起笔、收笔恰恰常在边缘。

                填充 P=1（四周补一圈零）之后：

                - 输出回到 28×28（same 配方）
                - 角落被看 4 次、边缘 6 次、中心 9 次——
                  从 1:9 的悬殊拉平到 4:9

                演示 2 会画出**覆盖计数热图**：无填充时左上角一个
                深坑，P=1 后坑被填平大半。想完全平（处处 9 次）？
                P=2 可以，但输出会涨到 30×30 比原图还大——
                **公平和尺寸不可兼得**，same 配方是工程上的折中。

                为什么要补**零**而不是补别的？零 = "这里没有信号"，
                对绝大多数核（权重和不为零的）意味着补丁区贡献小、
                又不引入虚假边缘。它是中性陪衬。
                """
            ),
            "🧵 概念 4/4 · 多通道：合议制与三维核": mo.md(
                """
                彩色图像是三层楼：R、G、B 各一张 H×W 的灰度图，
                叠成 **C_in × H × W** 的三维块（MNIST 是单通道，
                C_in = 1）。多通道卷积只改一件事——**核也变三维**：

                ```
                核形状：C_in × K × K      （每层楼配一页 K×K 的权重）
                z = Σ(通道、行、列三个维度一起加权求和)   ← 合议制
                ```

                打分时，窗口在每一层楼各盖住一个 K×K 的小块，
                三块各自与核的对应页做内积，然后**把三层的分数
                直接加起来**，得到一个数。通道维被"折叠"掉了：
                **输出还是一张平面响应图**。

                - 核页 [1, 0, 0]：只听 R 层的意见 → 退化成单通道卷积
                - 核页 [1, −1, 0]：听"R 减 G"——两版不一致的地方亮
                - 核页 [⅓, ⅓, ⅓]：三票平均——笔画重合处互相加强、
                  错位处互相抵消

                最后，一个卷积层想要 **C_out 张**不同的响应图？
                准备 C_out 个三页的核，各自扫一遍全图。所以卷积层
                的权重形状是 **C_out × C_in × K × K**——四本账
                （29 课实现时你会亲手摸到这个四维张量）。

                > 📌 术语彩蛋：深度学习里说的"卷积"其实是不翻转
                > 核的**互相关**（cross-correlation）。数学系的真
                > 卷积要先把核旋转 180°。两者只差一个固定的翻转，
                > 不影响网络能学什么——核本来就是学出来的，翻不翻
                > 等价于换一套参数。行业将错就错，你面试时能讲出
                > 这个区别就够了。
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

        样张沿用 26 课的数字 0（训练集 #1），竖边核当唯一的模板——
        变的是**扫描方式**，不是尺子。
        """
    )
    return


@app.cell
def _(mo, np):
    # MNIST 加载器（与 25/26 课同款：优先读 data/mnist.npz 缓存，缺失则自动下载）
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

    X_mnist, y_mnist, X_te27, y_te27 = load_mnist()

    # 带步长与填充的卷积参考实现（本课地基，供全课与练习对账）
    def conv_sp(img, K, stride=1, pad=0):
        """循环版 2D 卷积：先四周补 pad 圈零，再按 stride 跳格扫。"""
        if pad:
            img = np.pad(img, pad)                    # 填充：四周补零（06 课广播的亲戚）
        _h, _w = img.shape
        _kh, _kw = K.shape
        _oh = (_h - _kh) // stride + 1                # 输出尺寸公式（本课概念 1）
        _ow = (_w - _kw) // stride + 1
        _out = np.zeros((_oh, _ow))
        for _i in range(_oh):
            for _j in range(_ow):
                _out[_i, _j] = np.sum(img[_i*stride:_i*stride+_kh,
                                          _j*stride:_j*stride+_kw] * K)
        return _out

    K_vert27 = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float64)
    img_zero27 = X_mnist.reshape(-1, 28, 28)[1]       # 数字 0（26 课同款样张）
    print("训练集:", X_mnist.shape, "｜conv_sp 就位：", conv_sp(img_zero27, K_vert27).shape)
    return (K_vert27, X_mnist, conv_sp, img_zero27, y_mnist)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 1 · 公式验证台：8 组配置逐一过堂

        下表每一行都**手算公式、真跑卷积**两边对账——公式说多大，
        代码就吐多大。重点看三行：

        - `K=3 S=1 P=1` 与 `K=5 S=1 P=2`：same 配方，28 进 28 出
        - `K=3 S=2 P=0`：**不整除**（公式 13.5）——框架取整到 13，
          最右/最下有半格被浪费
        - `K=4 S=3 P=0`：(28−4)/3+1 = 9 整除——偶数核不是不能用，
          只是配不出 same
        """
    )
    return


@app.cell
def _(K_vert27, conv_sp, img_zero27, np):
    # 演示 1 · 公式验证台
    _configs = [(3, 1, 0), (3, 1, 1), (5, 1, 2), (3, 2, 1), (5, 2, 2),
                (2, 2, 0), (3, 2, 0), (4, 3, 0)]
    print(f"{'K':>2} {'S':>2} {'P':>2} │ {'公式精确值':>8} │ {'实际输出':>8} │ 备注")
    print("─" * 52)
    for _K, _S, _P in _configs:
        _exact = (28 + 2 * _P - _K) / _S + 1
        _out = conv_sp(img_zero27, np.ones((_K, _K)), stride=_S, pad=_P)
        _note = ""
        if abs(_exact - round(_exact)) > 1e-9:
            _note = "⚠️ 不整除 → 向下取整（最右/最下半格被跳过）"
        elif _out.shape[0] == 28:
            _note = "same 配方：28 进 28 出"
        print(f"{_K:>2} {_S:>2} {_P:>2} │ {_exact:>8.1f} │ {str(_out.shape[0]):>8} │ {_note}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 2 · 覆盖计数热图：边缘像素的委屈与填充的平反

        对每个输入像素数一数：**有多少个 3×3 窗口盖住过它**
        （它被"看"了几次）。颜色越亮，被看次数越多。

        - **左（无填充）**：角落 1 次、边缘 3 次、中心 9 次——
          一个四周深、中心亮的金字塔。角落的笔画信息天生
          只能被采样一次
        - **右（P=1）**：角落 4 次、边缘 6 次、中心 9 次——
          坑被填平大半，输出同时回到 28×28

        想要处处 9 次的绝对公平？P=2 可以做到，但输出会涨到
        30×30（比原图还大）。工程选择 same 配方：**够公平 +
        尺寸不变**。
        """
    )
    return


@app.cell
def _(np, plt):
    # 演示 2 · 覆盖计数热图
    def cover_counts(H, K, S, P):
        """每个输入像素被多少个窗口盖住。"""
        _cnt = np.zeros((H, H))
        for _i in range(-P, H - K + P + 1, S):
            for _j in range(-P, H - K + P + 1, S):
                for _di in range(K):
                    for _dj in range(K):
                        _y, _x = _i + _di, _j + _dj
                        if 0 <= _y < H and 0 <= _x < H:
                            _cnt[_y, _x] += 1
        return _cnt

    c_nopad = cover_counts(28, 3, 1, 0)
    c_pad1 = cover_counts(28, 3, 1, 1)
    print(f"无填充：角落 {c_nopad[0,0]:.0f} 次、边缘 {c_nopad[0,14]:.0f} 次、中心 {c_nopad[14,14]:.0f} 次")
    print(f"P=1：   角落 {c_pad1[0,0]:.0f} 次、边缘 {c_pad1[0,14]:.0f} 次、中心 {c_pad1[14,14]:.0f} 次")

    _fig, _axes = plt.subplots(1, 2, figsize=(9.6, 4.5))
    for _ax, _c, _ttl in [
        (_axes[0], c_nopad, "无填充：角落 1 / 边缘 3 / 中心 9"),
        (_axes[1], c_pad1, "P=1：角落 4 / 边缘 6 / 中心 9"),
    ]:
        _im = _ax.imshow(_c, cmap="viridis", vmin=0, vmax=9)
        _ax.set_title(_ttl, fontsize=11)
        _ax.add_patch(plt.Rectangle((0.5, 0.5), 10, 10, fill=False, color="white", lw=1.5, alpha=0.6))
        _ax.axis("off")
    _cbar = _fig.colorbar(_im, ax=_axes, shrink=0.85, ticks=range(10))
    _cbar.set_label("该像素被多少个窗口看过")
    _fig.suptitle("覆盖计数热图：填充把「边缘的坑」填平大半", fontsize=12)
    _fig
    return (c_nopad, c_pad1, cover_counts)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 3 · 多通道合议：同一个字的三层楼

        MNIST 是单通道，我们**人为盖三层楼**：通道 1 = 原图、
        通道 2 = 原图右移 2 像素、通道 3 = 原图下移 2 像素
        （想象同一个人写了三遍、每遍位置略歪）。

        三个"合议核"（都是 3×3×3 的三维块，竖边核打底）：

        | 核 | 通道投票权 | 响应图在说什么 | 强响应窗 |
        |---|---|---|---|
        | A 只听通道 1 | [1, 0, 0] | 和 26 课的单通道卷积**逐像素相同** | 93 |
        | B 听两版之差 | [1, −1, 0] | 两版笔画错位的地方亮——"变化检测器" | 149 |
        | C 三票平均 | [⅓, ⅓, ⅓] | 三版重合的边互相加强、错位的互相抵消 | 34 |

        注意 A 的验证：`np.allclose(多通道版, 单通道版) = True`——
        单通道卷积就是"只有一个选民"的合议制特例。

        C 的强响应从 93 掉到 34 也值得咀嚼：**平均会杀死分歧**。
        三版一致的边（数字 0 的主轮廓）活了下来，三版各说各话的
        区域被平均抹平——这就是"合议制用加权控制话语权"。
        """
    )
    return


@app.cell
def _(K_vert27, X_mnist, conv_sp, img_zero27, np, plt):
    # 演示 3 · 多通道合议
    def shift_img27(im, dy, dx):
        _out = np.zeros_like(im)
        _h, _w = im.shape
        _ys, _ye = max(0, dy), min(_h, _h + dy)
        _xs, _xe = max(0, dx), min(_w, _w + dx)
        _out[_ys:_ye, _xs:_xe] = im[_ys - dy:_ye - dy, _xs - dx:_xe - dx]
        return _out

    def conv_multi(x, K):
        """x: (C,H,W)，K: (C,kh,kw) → (H-kh+1, W-kw+1)。通道维与窗口维一起求和。"""
        _C, _H, _W = x.shape
        _, _kh, _kw = K.shape
        _out = np.zeros((_H - _kh + 1, _W - _kw + 1))
        for _i in range(_out.shape[0]):
            for _j in range(_out.shape[1]):
                _out[_i, _j] = np.sum(x[:, _i:_i + _kh, _j:_j + _kw] * K)   # C×K×K 全加（08 课 axis）
        return _out

    x3_27 = np.stack([img_zero27, shift_img27(img_zero27, 0, 2), shift_img27(img_zero27, 2, 0)])
    K_pick = np.zeros((3, 3, 3)); K_pick[0] = K_vert27            # 只听通道 1
    K_diff = np.zeros((3, 3, 3)); K_diff[0] = K_vert27; K_diff[1] = -K_vert27
    K_avg = np.stack([K_vert27 / 3] * 3)                          # 三票平均

    r_pick, r_diff, r_avg = conv_multi(x3_27, K_pick), conv_multi(x3_27, K_diff), conv_multi(x3_27, K_avg)
    print(f"核A 只听通道1：与单通道卷积逐像素相同 = "
          f"{np.allclose(r_pick, conv_sp(img_zero27, K_vert27))}；强响应窗 {(np.abs(r_pick) > 1.5).sum()}")
    print(f"核B 两版之差：范围 [{r_diff.min():.2f}, {r_diff.max():.2f}]；强响应窗 {(np.abs(r_diff) > 1.5).sum()}")
    print(f"核C 三票平均：范围 [{r_avg.min():.2f}, {r_avg.max():.2f}]；强响应窗 {(np.abs(r_avg) > 1.5).sum()}")

    _fig, _axes = plt.subplots(1, 4, figsize=(13.0, 3.7))
    _axes[0].imshow(img_zero27, cmap="gray_r")
    _axes[0].set_title("输入：数字 0\n（1 通道）", fontsize=10)
    _vmax = np.abs(r_diff).max()
    for _ax, _r, _ttl in [
        (_axes[1], r_pick, "核A [1,0,0]\n只听通道1（=单通道）"),
        (_axes[2], r_diff, "核B [1,−1,0]\n两版之差（变化检测）"),
        (_axes[3], r_avg, "核C [1/3,1/3,1/3]\n三票平均（分歧被抹平）"),
    ]:
        _ax.imshow(_r, cmap="RdBu_r", vmin=-_vmax, vmax=_vmax)
        _ax.set_title(_ttl, fontsize=10)
    for _ax in _axes:
        _ax.axis("off")
    _fig.suptitle("多通道合议制：通道投票权（核的通道页）决定响应图说什么", fontsize=12)
    _fig.tight_layout()
    _fig
    return (conv_multi, r_avg, r_diff, r_pick, shift_img27, x3_27)



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
        ### 交互实验 1 · 尺寸试验台：三个旋钮一起拧

        核大小（竖边核加宽版）、步长、填充三个旋钮随意组合。
        每次改动，下方的**公式黑板**实时代入数字算出输出尺寸，
        右图是真扫出来的响应图——公式和代码永远同步。

        试试这些任务：

        - 调出 **same 配方**（输出 28×28）：K=5 时 P 得是多少？
        - 找一个**不整除**的组合（黑板会亮黄牌警告）
        - 把 S 拉到 3、P=0：数字 0 还能认出来吗？马赛克到什么程度？
        - 核加到 7×7（响应量级变大——一次求和的项更多了）
        """
    )
    return


@app.cell
def _(mo):
    # 交互控制：核大小 / 步长 / 填充
    ksize_dropdown = mo.ui.dropdown(options=[3, 5, 7], value=3, label="📐 核大小 K")
    stride_slider = mo.ui.slider(1, 3, value=1, step=1, label="👟 步长 S（1~3）")
    pad_slider = mo.ui.slider(0, 4, value=0, step=1, label="🖼️ 填充 P（0~4）")
    mo.hstack([ksize_dropdown, stride_slider, pad_slider])
    return (ksize_dropdown, pad_slider, stride_slider)


@app.cell
def _(K_vert27, conv_sp, img_zero27, ksize_dropdown, mo, np, pad_slider, plt, stride_slider):
    # ===== 交互实验 1 · 尺寸试验台（响应单元）=====
    _K_size = int(ksize_dropdown.value or 3)
    _S, _P = stride_slider.value, pad_slider.value

    def make_vert(K):                          # 竖边核的加宽版：左半 +1、右半 −1、中间 0
        _half = K // 2
        _row = [1] * (K - _half - 1) + [0] + [-1] * (K - _half - 1)
        return np.array([_row] * K, dtype=np.float64)

    K_now = make_vert(_K_size)
    _exact = (28 + 2 * _P - _K_size) / _S + 1   # 公式精确值
    _out = conv_sp(img_zero27, K_now, stride=_S, pad=_P)
    _oh = _out.shape[0]
    _warn = abs(_exact - round(_exact)) > 1e-9

    _img_padded = np.pad(img_zero27, _P) if _P else img_zero27
    _fig, _axes = plt.subplots(1, 2, figsize=(9.8, 4.6),
                               gridspec_kw={"width_ratios": [1, 1]})
    _axes[0].imshow(_img_padded, cmap="gray_r")
    if _P:
        _axes[0].add_patch(plt.Rectangle((-.5, -.5), _img_padded.shape[1], _img_padded.shape[0],
                                         fill=False, color="tab:orange", lw=2))
        _axes[0].set_title(f"输入（28×28）+ 填充 P={_P}\n橙框内是补的零圈", fontsize=10)
    else:
        _axes[0].set_title("输入（28×28）无填充", fontsize=10)
    _axes[0].axis("off")
    _vmax = max(np.abs(_out).max(), 1e-9)
    _axes[1].imshow(_out, cmap="RdBu_r", vmin=-_vmax, vmax=_vmax)
    _axes[1].set_title(f"响应图 {_out.shape[0]}×{_out.shape[1]}\n"
                       f"范围 [{_out.min():.1f}, {_out.max():.1f}]", fontsize=10)
    _axes[1].axis("off")
    _fig.suptitle(f"K={_K_size} · S={_S} · P={_P} 的扫描结果", fontsize=12)
    _fig.tight_layout()
    _fig

    # ---- 公式黑板 + 判词 ----
    _board = (f"**公式黑板**：out = (H + 2P − K) / S + 1 = "
              f"(28 + 2×{_P} − {_K_size}) / {_S} + 1 = **{_exact:.1f}**"
              f"{' → 向下取整' if _warn else ''} → 实际输出 **{_oh}×{_oh}**")
    if _warn:
        _verdict = (f"⚠️ 不整除（{_exact:.1f}）——最右/最下有半格被跳过。"
                    "工程上通常调 P 让它整除")
    elif _oh == 28:
        _verdict = "✅ same 配方达成：28 进 28 出——这正是「奇数核 + P=(K−1)/2」的好处"
    else:
        _verdict = (f"尺寸 28 → {_oh}。{_S > 1 and '步长在跳格下采样；' or ''}"
                    f"每站计算量不变，总站数从 26² 降到 {_oh}²")

    mo.vstack([
        _fig,
        mo.callout(mo.md(_board + "<br>" + _verdict), kind="info"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✏️ 动手练习（每课必做，本课 3 题）": mo.md(
                """
                **练习 1 · 手算尺寸表**：四组 (H, K, S, P)，用公式
                笔算输出尺寸，填进四个空。其中一题藏着
                **不整除的陷阱**——想想框架会怎么处理。

                **练习 2 · 实现 conv_sp 的步长与填充**：给你留好
                函数骨架，补两处关键代码——`np.pad` 的填充与循环里
                按步长取窗口的切片。用「S=1、P=0 时必须退化成
                26 课的 conv_naive」对账，再跑两组带旋钮的配置
                和参考实现逐元素比对。

                **练习 3 · 多通道核手算**：2 通道 4×4 输入 +
                2×2×2 核，手算两个输出值（一个常规、一个恰好为零）。
                算完就知道"通道维一起求和"不是句空话。

                规则不变：先把 `None` 换成你的答案再运行；写完再看参考答案。
                """
            )
        }
    )
    return


@app.cell
def _(mo):
    # ===== 练习 1 · 手算尺寸表 =====
    quiz1_rows = [(32, 5, 1, 0), (32, 3, 2, 1), (28, 2, 2, 0), (24, 3, 1, 1)]
    print("用 out = (H + 2P − K) / S + 1 笔算下表的「输出」列：")
    print(f"{'H':>3} {'K':>2} {'S':>2} {'P':>2} │ 输出")
    print("─" * 24)
    for _H, _K, _S, _P in quiz1_rows:
        print(f"{_H:>3} {_K:>2} {_S:>2} {_P:>2} │ ???")

    ans1_ex1 = None                           # ← ① H=32 K=5 S=1 P=0
    ans2_ex1 = None                           # ← ② H=32 K=3 S=2 P=1
    ans3_ex1 = None                           # ← ③ H=28 K=2 S=2 P=0
    ans4_ex1 = None                           # ← ④ H=24 K=3 S=1 P=1
    return (ans1_ex1, ans2_ex1, ans3_ex1, ans4_ex1, quiz1_rows)


@app.cell
def _(ans1_ex1, ans2_ex1, ans3_ex1, ans4_ex1, mo, quiz1_rows):
    # ===== 练习 1 · 判词 =====
    _truth = [(_H + 2 * _P - _K) // _S + 1 for _H, _K, _S, _P in quiz1_rows]
    _ans = [ans1_ex1, ans2_ex1, ans3_ex1, ans4_ex1]
    if any(_a is None for _a in _ans):
        _msg = "⏳ 四个尺寸都算出来再填（记住：先除后加 1）"
    else:
        _right = [i for i, (a, t) in enumerate(zip(_ans, _truth)) if a == t]
        if len(_right) == 4:
            _msg = (f"✅ 四题全对：{_truth}。最后一路 H=24, K=3, S=1, P=1 是 "
                    "same 配方（P=(K−1)/2=1）——24 进 24 出")
        else:
            _detail = "；".join(f"第{i+1}题（H={r[0]},K={r[1]},S={r[2]},P={r[3]}）"
                               f"参考 {t}" for i, (a, t), r in
                               zip(range(4), zip(_ans, _truth), quiz1_rows) if a != t)
            _msg = f"❌ {_detail}。口诀：分子是窗口活动范围，除 S 是站数，加 1 是起点站"
    mo.md(_msg)
    return


@app.cell
def _(K_vert27, X_mnist, conv_sp, img_zero27, mo, np):
    # ===== 练习 2 · 实现 conv_sp 的步长与填充 =====
    def conv_sp_ex2(img, K, stride=1, pad=0):
        """补全两处：① 填充 ② 按步长取窗口。"""
        if pad:
            img = None                         # ← ① 补：把 img 四周各补 pad 圈零
        _h, _w = img.shape
        _kh, _kw = K.shape
        _oh = (_h - _kh) // stride + 1
        _ow = (_w - _kw) // stride + 1
        _out = np.zeros((_oh, _ow))
        for _i in range(_oh):
            for _j in range(_ow):
                _out[_i, _j] = None            # ← ② 补：从 _i*stride 起取 K 大小的窗口 × 核求和
        return _out

    conv_ready_ex2 = None                      # ← 补完后，把这行换成 conv_sp_ex2

    if conv_ready_ex2 is not None:
        try:
            _img = img_zero27
            _r0 = conv_ready_ex2(_img, K_vert27, 1, 0)              # 必须退化成 26 课结果
            _ref0 = conv_sp(_img, K_vert27, 1, 0)
            _r1 = conv_ready_ex2(_img, K_vert27, 2, 1)              # 旋钮组合
            _ref1 = conv_sp(_img, K_vert27, 2, 1)
            _r2 = conv_ready_ex2(_img, K_vert27, 2, 0)
            _ref2 = conv_sp(_img, K_vert27, 2, 0)
            if (np.allclose(_r0, _ref0) and np.allclose(_r1, _ref1)
                    and np.allclose(_r2, _ref2)):
                _msg = ("✅ 三组配置全部与参考一致！S=1,P=0 → 26×26；"
                        "S=2,P=1 → 14×14；S=2,P=0 → 13×13——旋钮全听你指挥")
            else:
                _msg = ("❌ 有配置对不上。检查①np.pad(img, pad)；"
                        "②窗口起点是 _i*stride 和 _j*stride，终点各加 _kh/_kw")
        except (TypeError, ValueError, IndexError) as _e:
            _msg = f"❌ 运行出错：{_e}——两处 None 还没补对"
    else:
        _msg = "⏳ 先补两处代码，再把 conv_ready_ex2 换成 conv_sp_ex2"
    mo.md(_msg)
    return (conv_ready_ex2, conv_sp_ex2)


@app.cell
def _(mo, np):
    # ===== 练习 3 · 多通道核手算 =====
    x_ex3 = np.stack([
        np.array([[1, 0, 2, 0], [1, 0, 2, 0], [1, 0, 2, 0], [0, 0, 0, 0]], float),
        np.array([[0, 0, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]], float),
    ])                                          # (2, 4, 4)：两层的 4×4
    K_ex3 = np.stack([
        np.array([[1, 0], [0, 1]], float),      # 通道1 页：主对角
        np.array([[0, 1], [1, 0]], float),      # 通道2 页：副对角
    ])                                          # (2, 2, 2)
    print("输入 通道1：\n", x_ex3[0])
    print("输入 通道2：\n", x_ex3[1])
    print("核 通道1页：\n", K_ex3[0], "\n核 通道2页：\n", K_ex3[1])
    print("提示：out[1,1] 的窗口是 通道1 的 [1:3, 1:3] 和 通道2 的 [1:3, 1:3]")

    out_mid_ex3 = None                          # ← ① 手算 out[1,1]（窗口在正中）
    out_bl_ex3 = None                           # ← ② 手算 out[2,0]（窗口在左下）
    return (K_ex3, out_bl_ex3, out_mid_ex3, x_ex3)


@app.cell
def _(K_ex3, mo, np, out_bl_ex3, out_mid_ex3, x_ex3):
    # ===== 练习 3 · 判词 =====
    _C, _H, _W = x_ex3.shape
    _, _kh, _kw = K_ex3.shape
    _ref = np.zeros((_H - _kh + 1, _W - _kw + 1))
    for _i in range(_ref.shape[0]):
        for _j in range(_ref.shape[1]):
            _ref[_i, _j] = np.sum(x_ex3[:, _i:_i + 2, _j:_j + 2] * K_ex3)
    print("参考输出（对账用，别偷看答案前先算）：\n", _ref)

    if out_mid_ex3 is not None and out_bl_ex3 is not None:
        _ok = (abs(out_mid_ex3 - _ref[1, 1]) < 1e-9 and abs(out_bl_ex3 - _ref[2, 0]) < 1e-9)
        if _ok:
            _msg = (f"✅ 全对！out[1,1] = {_ref[1,1]:.0f}（通道1 的 2 + 通道2 的 1）、"
                    f"out[2,0] = {_ref[2,0]:.0f}（两层恰好都贡献 0——窗口里的 1 "
                    "全躲开了核的 1）")
        else:
            _msg = (f"❌ 参考：out[1,1] = {_ref[1,1]:.0f}、out[2,0] = {_ref[2,0]:.0f}。"
                    "把两个通道的窗口抄下来，逐位置乘、九个积一起加（两层共 8 个积）")
    else:
        _msg = "⏳ 两个值都笔算出来再填"
    mo.md(_msg)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 1 参考答案（先自己写过再看！）": mo.md(
                """
                **28、16、14、24。**

                - (32+0−5)/1+1 = **28**
                - (32+2−3)/2+1 = 15.5+1 = **16.5**——陷阱题！
                  31/2 不整除，框架向下取整：floor(15.5)+1 = **16**
                - (28+0−2)/2+1 = **14**
                - (24+2−3)/1+1 = **24**（same 配方：P=(K−1)/2=1）

                第 2 题的教训值得记一辈子：**公式算出小数不是
                世界末日**，框架一律向下取整；但"被跳过的半格"
                意味着最右一列像素没被任何窗口完整扫过。顺带一个
                数字游戏：这组配置里 32−3 = 29 是奇数，加任何 2P
                都改不了奇偶性——**无论 P 填多少都不整除**；想
                整除只能动 K 或 H（K=4 就行）。配置是设计出来的，
                不是撞大运。
                """
            ),
            "✅ 练习 2 参考答案（先自己写过再看！）": mo.md(
                """
                两处补全：

                ```python
                if pad:
                    img = np.pad(img, pad)          # ① 四周各 pad 圈零
                ...
                    _out[_i, _j] = np.sum(
                        img[_i*stride:_i*stride+_kh,
                            _j*stride:_j*stride+_kw] * K)   # ② 窗口起点 = 序号×步长
                ```

                ② 的精髓：循环变量 `_i` 是**第几站**（0,1,2,…），
                窗口的**像素坐标**是 `_i*stride`。步长藏在"站号
                →坐标"的换算里——S=2 时第 3 站的窗口从像素 6
                开始，而不是 2。

                `np.pad(img, pad)` 的默认模式就是补零；它还有
                `'edge'`（复制边缘）、`'reflect'`（镜像）等模式，
                CNN 里 99% 的场合用零填充——零是"无信号"的
                中性表达。
                """
            ),
            "✅ 练习 3 参考答案（先自己写过再看！）": mo.md(
                """
                **out[1,1] = 3，out[2,0] = 0。**

                out[1,1] 的窗口：通道 1 取 [[0,2],[0,2]]、
                通道 2 取 [[1,0],[1,0]]。

                - 通道 1：0×1 + 2×0 + 0×0 + 2×1 = **2**
                  （主对角页只给 (左上,右下) 位置的 2 记分）
                - 通道 2：1×0 + 0×1 + 1×1 + 0×0 = **1**
                  （副对角页只给 (左下,右上) 位置的 1 记分）
                - 合计 z = 2 + 1 = **3**——两个通道的分数**直接相加**，
                  通道维被折叠掉了

                out[2,0] 的窗口：通道 1 取 [[0,2],[0,0]]、
                通道 2 取 [[0,0],[0,0]]——第二层这一窗全是 0！

                - 通道 1：0×1 + 2×0 + 0×0 + 0×1 = 0
                - 通道 2：0
                - 合计 **0**：就算某一层信号很强，**另一层照样
                  能用零把它对冲掉**——合议制的权力制衡
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
                1. **输出尺寸公式**：out = (H + 2P − K) / S + 1；
                   分子是窗口左上角的活动范围，除 S 是站数，加 1
                   是起点站；不整除时框架一律向下取整（跳过半格）
                2. **same 配方**：P = (K−1)/2 让输出 = 输入；
                   奇数核独享——偶数核配不出整数的 P
                3. **步长是下采样器**：S=2 时输出边长减半、总计算量
                   约 1/4；粗扫定位、细扫确认
                4. **填充的双重使命**：保尺寸 + 平反边缘——无填充
                   角落 1 次/边缘 3 次/中心 9 次，P=1 变 4/6/9；
                   处处 9 要 P=2 但输出会膨胀到 30×30
                5. **多通道 = 合议制**：核是 C_in×K×K 的三维块，
                   通道维与窗口维一起求和，输出仍是单张响应图；
                   通道页 [1,0,0] 退化成单通道（验证 True）、
                   平均页把强响应从 93 窗杀到 34 窗
                6. **层的完整形状账**：权重 C_out × C_in × K × K，
                   偏置 C_out；一层读进 C_in 通道、吐出 C_out 张
                   响应图——29 课动手实现时的四维账本
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

                至此，"怎么扫"的全部数学已就位：窗口多大、跳几格、
                边上补几圈、几层楼一起看。但卷积层吐出的响应图
                还是**原尺寸的忠实记录**——28×28 的字，响应图
                也差不多 28×28，一点没省。

                下一课解决**怎么把图变小**、以及一个更深的问题：
                层层堆叠之后，**每个输出到底"看得到"原图多大范围**？
                """
            ),
            mo.callout(
                mo.md(
                    """
                    **下一课预告** — `28_池化与感受野`：

                    **最大池化**：2×2 的窗口里只留最大的那个数
                    ——"这片区域有没有笔画"比"笔画精确在哪格"
                    更重要。28→14→7 的金字塔，配合卷积把
                    **感受野**从 3×3 一路养大到吞下整张图。

                    还有一场平移容忍的定量实验：数字挪 1 像素，
                    只卷积的响应图变化多大？加了池化又变小多少？
                    ——上一课平移实验的下半场。

                    🏊 去看池化怎么把"大概在哪"变成够用的信息！
                    """
                ),
                kind="success",
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
