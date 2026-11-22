import marimo

__generated_with = "0.25.2"
app = marimo.App(app_title="19 · 激活函数", width="medium")


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
        # 19 · 激活函数：把直线掰弯的那只手

        **从零到 Transformer · 手写深度学习** ｜ 第 4 部分 · 神经网络（2/8）
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📖 本课目标（先读我）": mo.md(
                """
                **这一课只解决一个问题：为什么神经元之间必须有一个"非线性关节"？**

                第 18 课结尾立了两桩案子：sign **不可导**（z = 0 处突然跳变，
                14 课的链式法则没法从它身上穿过去）、**只给判决不给概率**。
                这一课请出修复两案的主角——**激活函数**（activation function）：
                接在 `w·x + b` 后面的一道非线性工序，可导、平滑、有脾气。

                打法依然稳扎稳打：先看一个细思极恐的事实——**线性摞线性还是线性**，
                不加关节，摞一百层等于一层（12 课的复合变换早埋好了伏笔）；
                再把三位主力选手（sigmoid / tanh / ReLU）的**函数和导数**一起请上台；
                然后亲手用 ReLU 拼折线、逼近 sin 曲线，提前看见"万能逼近"的雏形；
                最后查两桩悬案：**饱和**（导数消失的平地）与**死 ReLU**
                （一睡不醒的神经元）。

                学完你将能：

                - 证明给自己看：**没有非线性，深度就是一句空话**
                - 画出 sigmoid / tanh / ReLU 的**函数与导数**，说出各自的地盘：
                  sigmoid 守概率输出（16 课），tanh 守隐层（零中心），ReLU 守深度
                - 说出**饱和区**和**死 ReLU** 分别是什么、为什么危险、怎么预防
                - 用"**斜率补丁**"的眼光看 ReLU：每个神经元到折点才生效，
                  改写量 = 权重——这是下一课 MLP 的核心直觉

                前置：第 16 课的 sigmoid 与 σ' = σ(1-σ)、第 12 课的复合变换
                （A@B 还是一个矩阵）、第 13 课的对称差分（本课的验钞机）、
                第 17 课的容量旋钮（本课的容量 = 折点数量）。

                激活函数本身只有一行代码（`np.maximum(0, z)` 而已）——
                本课真正的功课是看清**它的导数如何在 22 课的反向传播里
                决定生死**。今天认识的每一位选手，都在为那一刻排队。
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "🔗 概念 1/4 · 线性摞线性 = 线性（深度的魔咒）": mo.md(
                """
                12 课讲过：矩阵相乘 = 复合变换，`A @ B` 从右往左读，
                两个矩阵干完活，**效果等于一个新的矩阵**。把这句话翻译到网络上：

                ```
                层 1:  h1 = x @ W1 + b1
                层 2:  h2 = h1 @ W2 + b2
                层 3:  h3 = h2 @ W3 + b3
                ```

                层层代入（先忽略 b，一会儿单独说），h3 = x @ (W1 @ W2 @ W3)——
                **三个矩阵的连乘还是一个矩阵**。也就是说：不管你摞多少层，
                输入到输出之间的映射永远是**一次线性变换**，
                和一层没有区别（准确说：等价于某个 W = W1@W2@W3 和某个 b）。

                偏置 b 也救不了场：代入之后它只是被后面的矩阵搬运、缩放、
                合并成一个新的 b——线性加线性，还是线性。

                而线性变换能做的事 12 课全看过了：旋转、剪切、拉伸、平移——
                **弯不了**。18 课的 XOR 需要弯，任何非直线的边界需要弯，
                哪怕只想拟一条 sin 曲线也需要弯。结论只有一条：

                > **每一层之后必须插一道非线性工序**（激活函数 σ），
                > 让"摞层数"真正变成"弯折能力"。深度学习里的"深度"，
                > 深的是非线性，不是矩阵乘法。

                顺便立个规矩：从本课起，"一层网络"的完整定义是
                **h = σ(X @ W + b)**——12 课的几何（W 变形、b 平移）
                加今天的新工序（σ 弯折）。演示 2 会用数值实验把这桩案子办成铁案。
                """
            ),
            "🌊 概念 2/4 · sigmoid 与 tanh：平滑的弯": mo.md(
                """
                两位老朋友 + 一位新朋友：

                **sigmoid**（16 课）：σ(z) = 1/(1+e⁻ᶻ)。把任意打分压进 (0, 1)，
                天生适合当**输出层**说概率。它有一条 16 课推过的漂亮导数：

                ```
                σ'(z) = σ(z)·(1 - σ(z))     （导数用函数自己就能算）
                ```

                注意 σ' 的最大值：在 z = 0 处 = 0.25。**它从没大过四分之一**——
                这个不起眼的天花板，是 22 课反向传播里的一颗地雷（概念 4 细说）。

                **tanh**（新朋友）：tanh(z) = 2σ(2z) - 1。把 sigmoid 拉伸成 (-1, 1)：
                形状一样、档位更宽。两个好处：**零中心**（输出有正有负、均值约 0，
                下一层拿到的输入更对称，梯度更新更均衡——24 课还会再遇到这个话题）；
                **导数天花板是 1**（z = 0 处），比 sigmoid 的 0.25 宽裕四倍。

                ```
                tanh'(z) = 1 - tanh(z)²      （z = 0 处 = 1，两端趋 0）
                ```

                两位的共同软肋叫**饱和**（saturation）：|z| 一大，曲线就躺平，
                导数趋近 0——"变化率"本身没了变化。z = ±4 附近，sigmoid' 只剩
                0.017、tanh' 只剩 0.0013；z = ±10 时分别是 4.5×10⁻⁵ 和
                8×10⁻⁹（演示 5 有全表）。**饱和 = 梯度的荒漠**：
                一个神经元一旦长期工作在饱和区，反向传播传到它这里的水流
                就近乎枯竭。
                """
            ),
            "⚡ 概念 3/4 · ReLU：深度学习的实干派": mo.md(
                """
                2010 年前后，深度网络被饱和折磨得够呛，一位极简主义者登场：

                ```
                ReLU(z) = max(0, z)      （负的归零，正的原样放行）
                ```

                就这么一行——**没有 exp，没有除法**，一次比较搞定（01 课的教训：
                简单运算才扛得住亿万次重复）。它的三大优势全是实战派打法：

                - **正区不饱和**：z > 0 时导数恒为 **1**。梯度穿过来不衰减——
                  深层网络能训起来，这一条居功至伟
                - **稀疏激活**：约一半神经元输出 0，网络每次只动用一部分兵力，
                  特征之间互不掺和
                - **折线机器**：ReLU 网络的输出是**分段线性**的——
                  每个神经元贡献一个折点和一段斜率改写。折点够多，
                  折线就能逼近任何连续曲线（演示 3 亲手拼给你看）

                但它也有一个著名的命门——**死 ReLU**（dead ReLU）：
                负区导数恒为 0。如果一个神经元的 z 对**所有**样本都小于 0，
                它输出恒 0、梯度恒 0、**永远不会再被更新**——睡着了就叫不醒。
                两种死法：初始化时偏置太负（一出生就睡，演示 4 数给你看）；
                训练中一次离谱的大梯度把 b 打飞（被打晕，演示 4 复盘现场）。

                药方也都记下：He 初始化（25 课）、Leaky ReLU（负区留 0.1 的
                小斜率，漏点血保条命）。工业界的默认答案至今仍是 ReLU——
                **便宜、不饱和、够用**。
                """
            ),
            "📉 概念 4/4 · 导数才是主角：反向传播的传令兵": mo.md(
                """
                这节课反复念叨导数，不是数学洁癖——是因为 **22 课的反向传播，
                传的就是导数**。先剧透一句规则：梯度每穿过一层激活函数，
                就要**乘上一次 f'(z)**（链式法则，14 课）。

                现在把概念 2、3 的导数排成一列，看看"连乘"会发生什么：

                | 层数 | 每层乘 σ'（最坏 0.25） | 每层乘 tanh'（最坏 1） | 每层乘 ReLU'（正区 1） |
                |---|---|---|---|
                | 10 层 | 0.25¹⁰ ≈ **9.5×10⁻⁷** | ≤ 1 | 1 |
                | 20 层 | 0.25²⁰ ≈ **9.1×10⁻¹³** | ≤ 1 | 1 |

                梯度消失（vanishing gradient）的雏形就在这张表里：sigmoid 网络
                深到 10 层，底层收到的梯度可能是百万分之一——**知道自己错了，
                却迈不动腿**（16 课说 MSE 配 sigmoid 会得这个病，病根相同）。
                这正是深度网络集体迁移到 ReLU 的头号原因。

                对称的病叫**梯度爆炸**（连乘的数都大于 1，梯度冲上天）——
                它在 35 课讲 RNN 时会真正爆发，到时候再算总账。

                本课的验钞机还是 13 课的**对称差分**：任何一个手写导数，
                都拿 (f(z+h) - f(z-h)) / (2h) 对一遍账，h 取 1e-5 量级。
                练习 1 你就要亲手对一次。
                """
            ),
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 🖥️ 代码演示

        五组演示，把"弯"这件事看透：

        - **演示 1** · 三位选手同台：函数与导数并排亮相
        - **演示 2** · 铁案：线性摞线性 = 线性，插一道 ReLU 立刻不同
        - **演示 3** · 折线魔术：用 ReLU 拼帐篷、逼近 sin（容量阶梯）
        - **演示 4** · 验尸报告：死 ReLU 的两种死法
        - **演示 5** · 饱和荒漠：导数表与梯度连乘的坍缩
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 1 · 三位选手同台

        上排是**函数** f(z)，下排是**导数** f'(z)——从今天起养成习惯：
        认识一个激活函数，必须连它的导数一起认识（22 课的传令兵就是它）。

        三个细节先划重点：sigmoid 的导数峰值只有 **0.25**（z = 0 处，
        图上那座小山的高度）；tanh 的导数峰值是 **1**，但两端同样塌陷；
        ReLU 的导数是一个**台阶**——正区恒 1（不衰减），负区恒 0（死区），
        z = 0 处严格说不可导，工程上随手约定取 0 或 1 都行（测度零的单点，
        不影响积分式的梯度计算）。
        """
    )
    return


@app.cell
def _(np, plt):
    # 演示 1 · sigmoid / tanh / ReLU：函数（上排）与导数（下排）同台
    _z = np.linspace(-6, 6, 600)                   # 采样点（03 课）
    _sig = 1 / (1 + np.exp(-_z))                   # sigmoid（16 课）

    # 三位选手：名字、函数、导数（导数全部用"函数自己"或定义式算）
    _acts = [
        ("sigmoid", _sig, _sig * (1 - _sig)),      # σ' = σ(1-σ)（16 课推过）
        ("tanh", np.tanh(_z), 1 - np.tanh(_z) ** 2),   # tanh' = 1 - tanh²
        ("ReLU", np.maximum(0, _z), (_z > 0).astype(float)),  # 正区 1，负区 0
    ]
    _colors = ["tab:blue", "tab:orange", "tab:green"]

    _fig, _axes = plt.subplots(2, 3, figsize=(12.5, 6.2), sharex=True)
    for _j, (_name, _f, _df) in enumerate(_acts):
        _axes[0][_j].plot(_z, _f, color=_colors[_j], lw=2.2)
        _axes[0][_j].set_title(f"{_name}(z)")
        _axes[0][_j].grid(True, alpha=0.3)
        _axes[0][_j].axhline(0, color="gray", lw=0.7, alpha=0.5)
        _axes[1][_j].plot(_z, _df, color=_colors[_j], lw=2.2)
        _axes[1][_j].set_title(f"{_name}'(z)   ← 22 课要传的就是它")
        _axes[1][_j].set_xlabel("z")
        _axes[1][_j].grid(True, alpha=0.3)
    _axes[0][0].set_ylabel("f(z)")
    _axes[1][0].set_ylabel("f'(z)")
    _axes[1][0].axhline(0.25, color="gray", ls=":", lw=1)
    _axes[1][0].annotate("峰值只有 0.25", xy=(0, 0.25), xytext=(1.5, 0.32),
                         fontsize=9, color="gray",
                         arrowprops=dict(arrowstyle="->", color="gray"))
    _fig.suptitle("上排：函数长什么样；下排：导数有多能打", fontsize=13)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 2 · 铁案：线性摞线性 = 线性

        4 个随机 5×5 矩阵当 4 层，同一个输入 x 分别走两条路：

        - **无激活**：x → W1 → W2 → W3 → W4，一路矩阵乘到底
        - **带 ReLU**：每层乘完先过一遍 max(0, ·) 再进下一层

        先验证理论：无激活版的输出，和 x 直接乘**等价矩阵** W1@W2@W3@W4
        的输出之差是 **2.7×10⁻¹⁵**——浮点噪声水平（09 课的老朋友），
        数学上就是同一个变换。**摞 4 层线性 = 1 层，分毫不差。**

        再看带 ReLU 的版本：5 个输出分量里 4 个被砍成了 0，剩下的 0.36
        和无激活版面目全非——**弯折真的发生了**，而且这一路的映射
        再也写不成"x 乘某个矩阵"的形式。深度，从这一道 max 开始才算数。
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # 演示 2 · 4 层网络：无激活 = 1 层（数值铁证）；插 ReLU 立刻不同
    _rng = np.random.default_rng(42)               # 固定种子（10 课）
    _d = 5
    _Ws = [_rng.normal(0, 1, (_d, _d)) for _ in range(4)]   # 4 层随机权重
    _x = _rng.normal(0, 1, _d)                     # 一个 5 维输入

    # ---- 路线 A：无激活，一路矩阵乘 ----
    _z_lin = _x.copy()
    for _W in _Ws:
        _z_lin = _z_lin @ _W                       # 12 课：连乘 = 复合变换

    # ---- 等价单矩阵：W1 @ W2 @ W3 @ W4（12 课：从右往左读）----
    _W_eq = _Ws[0] @ _Ws[1] @ _Ws[2] @ _Ws[3]
    _z_direct = _x @ _W_eq
    _gap = float(np.max(np.abs(_z_lin - _z_direct)))         # 应为浮点噪声

    # ---- 路线 B：每层后面插一道 ReLU ----
    _z_rel = _x.copy()
    for _W in _Ws:
        _z_rel = np.maximum(0, _z_rel @ _W)        # 19 课主角上岗

    # ---- 画图：两组柱状对比 ----
    _fig, _ax = plt.subplots(figsize=(8.6, 4.2))
    _idx = np.arange(_d)                           # 5 个分量的位置（03 课）
    _w_bar = 0.36
    _ax.bar(_idx - _w_bar / 2, _z_lin, _w_bar, color="tab:gray", label="4 层无激活（= 1 层）")
    _ax.bar(_idx + _w_bar / 2, _z_rel, _w_bar, color="tab:green", label="4 层带 ReLU")
    _ax.axhline(0, color="gray", lw=0.8)
    _ax.set_xticks(_idx)
    _ax.set_xlabel("输出分量的编号")
    _ax.set_ylabel("输出值")
    _ax.set_title("同一个输入、同样的 4 个矩阵：插不插 ReLU，两个世界")
    _ax.legend()
    _ax.grid(True, alpha=0.3, axis="y")

    mo.vstack([
        _fig,
        mo.md(
            f"""
            | 检查项 | 数值 | 结论 |
            |---|---|---|
            | 无激活输出 | {np.round(_z_lin, 2)} | |
            | 等价单矩阵输出 | {np.round(_z_direct, 2)} | 两者最大差 **{_gap:.1e}** = 浮点噪声 |
            | 带 ReLU 输出 | {np.round(_z_rel, 2)} | {_d} 个分量里 {int(np.sum(_z_rel == 0))} 个被砍成 0，映射面目全非 |

            一个补充的冷知识：把 100 个小尺度矩阵（标准差 0.3）连乘，
            乘完的向量范数从 1.8 缩到 **10⁻²⁰** 量级——线性层连乘不只是
            "等价于一层"，还会把数值本身**乘没了**。深度网络里每一层
            的尺度都需要精心照顾，25 课的初始化专治这个病。
            """
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 3 · 折线魔术：ReLU 拼出曲线

        ReLU 网络的几何真相：**每个隐单元 = 一段斜率补丁**。
        `a·relu(x - k)` 的意思是"**过了折点 k，斜率加 a**"——
        折点之前它安安静静输出 0，折点之后才开始贡献一段固定斜率。

        左图拼一个最小的例子（帐篷函数）：

        ```
        tent(x) = relu(x) - 2·relu(x - 1) + relu(x - 2)
        ```

        x < 0 时三个补丁都没醒（0）；0 到 1 之间斜率 +1；1 到 2 之间
        第一个补丁继续 +1、第二个补丁 **-2** 上场，合计 -1（下坡）；
        x > 2 三个补丁全醒：+1 - 2 + 1 = 0（躺平）。三段拼出一顶帐篷。

        右图上强度：撒 **48 个随机折点**、随机权重，再用最小二乘
        （17 课的 solve 直接回收）解出组合系数去逼近 sin——
        折线贴着曲线走，肉眼分不出彼此。容量阶梯一目了然：

        | 随机折点数 | 3 | 6 | 12 | 24 | 48 |
        |---|---|---|---|---|---|
        | 逼近 MSE | 0.27 | 0.033 | 0.0045 | 0.004 | **0.0002** |

        3 个折点时 MSE 高达 0.27，而且那 3 个随机折点里有两个几乎重叠
        （1.83 和 1.85）——**容量浪费**；48 个折点时误差压到 0.0002。
        这就是 17 课容量旋钮的神经网络版：**折点 = 容量的最小单位**。
        （今天折点是随机撒的、只有组合系数是解出来的——22 课之后，
        连折点位置都会被梯度自己学出来。）
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # 演示 3 · ReLU 拼折线：帐篷函数（左）+ 48 个随机折点逼近 sin（右）
    _xx = np.linspace(-3, 3, 600)
    _relu = lambda _v: np.maximum(0, _v)           # 主角一行流

    # ---- 左图：帐篷 = 三个斜率补丁的合力 ----
    _p1 = _relu(_xx)                               # 补丁 1：过 0，斜率 +1
    _p2 = -2 * _relu(_xx - 1)                      # 补丁 2：过 1，斜率 -2
    _p3 = _relu(_xx - 2)                           # 补丁 3：过 2，斜率 +1
    _tent = _p1 + _p2 + _p3                        # 帐篷

    # ---- 右图：48 个随机折点 + 最小二乘读出，逼近 sin ----
    _rng = np.random.default_rng(5)                # 固定种子：折点每次一样
    _n_feat = 48
    _k = _rng.uniform(-3, 3, _n_feat)              # 随机折点（撒在 x 范围内）
    _a = _rng.normal(0, 1, _n_feat)                # 随机特征权重
    _H = _relu(_xx[:, None] - _k[None, :]) * _a[None, :]     # (600, 48) 特征矩阵
    # 读出层最小二乘：解 (HᵀH)·w = Hᵀ·sin(x)（17 课的 solve）
    _w_out = np.linalg.solve(_H.T @ _H + 1e-8 * np.eye(_n_feat),
                            _H.T @ np.sin(_xx))
    _fit = _H @ _w_out
    _mse48 = float(np.mean((_fit - np.sin(_xx)) ** 2))

    _fig, _axes = plt.subplots(1, 2, figsize=(12.5, 4.3))
    _axes[0].plot(_xx, _p1, lw=1.2, ls="--", color="tab:blue", alpha=0.6, label="relu(x)（斜率 +1）")
    _axes[0].plot(_xx, _p2, lw=1.2, ls="--", color="tab:orange", alpha=0.6, label="-2·relu(x-1)（斜率 -2）")
    _axes[0].plot(_xx, _p3, lw=1.2, ls="--", color="tab:green", alpha=0.6, label="relu(x-2)（斜率 +1）")
    _axes[0].plot(_xx, _tent, color="tab:red", lw=2.6, label="三者相加：帐篷")
    _axes[0].set_title("帐篷函数：三个斜率补丁接力")
    _axes[0].set_xlabel("x")
    _axes[0].set_ylabel("y")
    _axes[0].legend(fontsize=8.5, loc="upper left")
    _axes[0].grid(True, alpha=0.3)

    _axes[1].plot(_xx, np.sin(_xx), color="gray", lw=1.6, alpha=0.75, label="sin(x) 真曲线")
    _axes[1].plot(_xx, _fit, color="tab:red", lw=2.2,
                  label=f"48 个随机 ReLU 的最优组合（MSE = {_mse48:.4f}）")
    for _kk in _k[::6]:                            # 抽稀画折点竖线，不喧宾夺主
        _axes[1].axvline(_kk, color="tab:red", alpha=0.12, lw=0.8)
    _axes[1].set_title("折线贴曲线：万能逼近的雏形")
    _axes[1].set_xlabel("x")
    _axes[1].set_ylabel("y")
    _axes[1].legend(fontsize=9)
    _axes[1].grid(True, alpha=0.3)

    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 4 · 验尸报告：死 ReLU 的两种死法

        造 64 个神经元（50 维输入、小随机权重），只动一个旋钮：偏置 b。
        数一数"对所有 200 个样本都输出 0"的神经元个数：

        | b | -2.5 | -2.0 | -1.5 | -1.0 | -0.5 | 0 | +0.5 |
        |---|---|---|---|---|---|---|---|
        | 死亡数 / 64 | 59 | 41 | 4 | 0 | 0 | 0 | 0 |

        注意这条**死亡悬崖**有多陡：b 从 -1.5 到 -2.0，死亡数从 4 跳到 41。
        原因在高维：单个样本 z < 0 的概率约 0.998（b = -2 时），
        看起来只比"大概率"大一点，但 **200 个样本全负的概率 = 0.998²⁰⁰ ≈ 0.67**——
        维度一多，小概率事件变成必然。高维世界的常识，经常和直觉相反。

        第二种死法更凶险：**训练中被打晕**。假设某次更新传来一个离谱的
        大梯度（比如误差信号 60），b 被一脚从 0 踹到 -60——此后 z 恒为负、
        导数恒为 0、**梯度再也传不回来**，这个神经元当场死亡且无药可救。
        防守姿势：学习率别太野（15 课）、初始化别太偏（25 课）、
        或者干脆用 Leaky ReLU 给负区留 0.1 的活口。
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # 演示 4 · 死 ReLU：偏置扫描（左）+ 一次离谱更新前后的 z 分布（右）
    _rng = np.random.default_rng(0)                # 固定种子
    _X = _rng.normal(0, 1, size=(200, 50))         # 200 个样本 × 50 维
    _W = _rng.normal(0, 0.1, size=(50, 64))        # 64 个神经元的小权重

    _b_list = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5]
    _dead_counts = []
    for _b in _b_list:
        _H = np.maximum(0, _X @ _W + _b)           # 前向
        _dead_counts.append(int(np.all(_H == 0, axis=0).sum()))   # 08 课：沿样本方向 all

    # ---- 右图素材：同一个神经元，b = 0 vs b = -60 的 z 分布 ----
    _z_alive = _X @ _W[:, 0] + 0.0                 # 健在版：z 有正有负
    _z_dead = _X @ _W[:, 0] - 60.0                 # 被打晕版：z 全在 -60 附近

    _fig, _axes = plt.subplots(1, 2, figsize=(12.5, 4.3))
    _axes[0].bar(range(len(_b_list)), _dead_counts, color="tab:red", alpha=0.85)
    _axes[0].set_xticks(range(len(_b_list)))
    _axes[0].set_xticklabels([f"{_b:+.1f}" for _b in _b_list])
    _axes[0].set_xlabel("偏置 b")
    _axes[0].set_ylabel("死神经元个数（共 64 个）")
    _axes[0].set_title("死亡悬崖：b 从 -1.5 到 -2.0，死亡数从 4 跳到 41")
    _axes[0].grid(True, alpha=0.3, axis="y")

    _axes[1].hist(_z_alive, bins=40, color="tab:green", alpha=0.75, label="健在：b = 0（z 跨正负）")
    _axes[1].hist(_z_dead, bins=40, color="tab:red", alpha=0.85, label="被打晕：b = -60（z 全负）")
    _axes[1].axvline(0, color="gray", lw=1.2, ls="--")
    _axes[1].set_xlabel("神经元的打分 z")
    _axes[1].set_ylabel("样本个数")
    _axes[1].set_title("死法二：一次离谱更新后，z 永远回不到正区")
    _axes[1].legend(fontsize=9)
    _axes[1].grid(True, alpha=0.3)

    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 5 · 饱和荒漠：导数表与梯度连乘的坍缩

        概念 4 的表格，用真实数字填满（对称差分逐一验过，13 课的规矩）：

        | z | 0 | ±2 | ±4 | ±6 | ±10 |
        |---|---|---|---|---|---|
        | sigmoid'(z) | 0.25 | 0.105 | 0.0177 | 2.5×10⁻³ | **4.5×10⁻⁵** |
        | tanh'(z) | 1.0 | 0.0707 | 1.3×10⁻³ | 2.5×10⁻⁵ | **8.2×10⁻⁹** |

        左图把三条导数曲线放在**同一把 log 尺**上（数值跨 9 个数量级，
        线性尺会把它们全压扁——09 课 log 的老家）：sigmoid 的导数山
        峰值 0.25，两侧跳水；tanh 起点更高、跳水更快；ReLU 的导数
        是一条水平线（正区恒 1）加一段深渊（负区 0）。

        右图算**连乘坍缩**：一个 20 层网络，每层梯度乘一次 f'（都取
        各自的"典型值"：sigmoid 取 0.25、tanh 取 0.5、ReLU 取 1），
        看梯度从顶层传到底层还剩多少——sigmoid 剩 **9.1×10⁻¹³**，
        相当于彻底失联；ReLU 原样送达。**深度网络为什么认 ReLU 当主力，
                这张图就是全部答案。**
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # 演示 5 · 左：三条导数曲线同台（log 尺）；右：20 层连乘后梯度剩多少
    _z = np.linspace(-8, 8, 800)
    _sig = 1 / (1 + np.exp(-_z))
    _dsig = _sig * (1 - _sig)                      # σ' = σ(1-σ)
    _dtanh = 1 - np.tanh(_z) ** 2                  # tanh' = 1 - tanh²
    _drelu = (_z > 0).astype(float)                # ReLU'：台阶

    _fig, _axes = plt.subplots(1, 2, figsize=(12.5, 4.3))

    # 左图：log 尺看导数（0 的地方 log 无定义，用极小值垫底再截断）
    _eps = 1e-12
    _axes[0].semilogy(_z, np.maximum(_dsig, _eps), color="tab:blue", lw=2.2, label="sigmoid'")
    _axes[0].semilogy(_z, np.maximum(_dtanh, _eps), color="tab:orange", lw=2.2, label="tanh'")
    _axes[0].semilogy(_z, np.maximum(_drelu, _eps), color="tab:green", lw=2.2, label="ReLU'")
    _axes[0].axhline(0.25, color="gray", ls=":", lw=1)
    _axes[0].text(-7.7, 0.30, "sigmoid' 天花板 0.25", fontsize=9, color="gray")
    _axes[0].set_xlabel("z")
    _axes[0].set_ylabel("f'(z)（log 尺）")
    _axes[0].set_ylim(_eps, 3)
    _axes[0].set_title("导数的荒漠地图：两端塌陷 vs 正区不塌")
    _axes[0].legend()
    _axes[0].grid(True, alpha=0.3, which="both")

    # 右图：20 层连乘，每层乘一个"典型导数"
    _n_layers = np.arange(1, 21)
    _axes[1].semilogy(_n_layers, 0.25 ** _n_layers, "o-", ms=4, color="tab:blue",
                      label="sigmoid（每层 ×0.25）")
    _axes[1].semilogy(_n_layers, 0.5 ** _n_layers, "s-", ms=4, color="tab:orange",
                      label="tanh（每层 ×0.5，典型值）")
    _axes[1].semilogy(_n_layers, 1.0 ** _n_layers, "^-", ms=4, color="tab:green",
                      label="ReLU（正区 ×1）")
    _axes[1].axhline(1e-6, color="gray", ls=":", lw=1)
    _axes[1].text(1.2, 2e-6, "失联线 10⁻⁶", fontsize=9, color="gray")
    _axes[1].set_xlabel("网络层数")
    _axes[1].set_ylabel("梯度残留比例（log 尺）")
    _axes[1].set_title("连乘坍缩：20 层后 sigmoid 梯度剩 9×10⁻¹³")
    _axes[1].legend(fontsize=9)
    _axes[1].grid(True, alpha=0.3, which="both")

    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 🎛️ 交互实验

        本课两组（关键课加倍）：

        1. **激活函数探索器**——下拉选函数、滑块选工作点 z₀，
           函数图上画出切线、导数图上标出落点，判词实时提醒你
           有没有踩进饱和荒漠或死区
        2. **折线魔术师**——四个滑块（两个折点 + 两个斜率补丁），
           亲手拼一条三段折线，看 ReLU 的"到点改斜率"机制怎么运转

        玩法提示：探索器里把 z₀ 拖到 ±6 以外看看 sigmoid 的导数读数；
        魔术师里试试让两个折点靠近、再让补丁一正一负——
        「帐篷」和「峡谷」都拼得出来。
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 交互实验 1 · 激活函数探索器
        """
    )
    return


@app.cell
def _(mo):
    # 交互控制：选函数 + 选工作点（debounce：松手才重算）
    act_dropdown = mo.ui.dropdown(
        options=["sigmoid", "tanh", "ReLU", "Leaky ReLU"], value="sigmoid",
        label="🧪 激活函数",
    )
    z0_slider = mo.ui.slider(-8, 8, value=0.0, step=0.1, debounce=True,
                             label="🎛️ 工作点 z₀（-8 ~ 8）")
    mo.hstack([act_dropdown, z0_slider])
    return (act_dropdown, z0_slider)


@app.cell
def _(act_dropdown, mo, np, plt, z0_slider):
    # ===== 交互实验 1 · 激活函数探索器（响应单元）=====
    _name = act_dropdown.value or "sigmoid"
    _z0 = z0_slider.value

    # 四位选手：函数与导数（Leaky ReLU 负区斜率 0.1）
    if _name == "sigmoid":
        _f = lambda _v: 1 / (1 + np.exp(-_v))
        _df = lambda _v: _f(_v) * (1 - _f(_v))
        _color = "tab:blue"
    elif _name == "tanh":
        _f = np.tanh
        _df = lambda _v: 1 - np.tanh(_v) ** 2
        _color = "tab:orange"
    elif _name == "ReLU":
        _f = lambda _v: np.maximum(0, _v)
        _df = lambda _v: (_v > 0).astype(float)
        _color = "tab:green"
    else:
        _f = lambda _v: np.where(_v > 0, _v, 0.1 * _v)
        _df = lambda _v: np.where(_v > 0, 1.0, 0.1)
        _color = "tab:purple"

    _z = np.linspace(-8, 8, 800)
    _y0, _s0 = float(_f(np.array(_z0))), float(_df(np.array(_z0)))

    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4.1))
    _axes[0].plot(_z, _f(_z), color=_color, lw=2.4, label=f"{_name}(z)")
    _t = np.linspace(_z0 - 1.5, _z0 + 1.5, 10)    # 切线的一段
    _axes[0].plot(_t, _y0 + _s0 * (_t - _z0), color="tab:red", lw=1.8, ls="--",
                  label=f"z₀ 处切线（斜率 {_s0:.3f}）")
    _axes[0].scatter([_z0], [_y0], s=90, color="tab:red", zorder=4)
    _axes[0].set_title(f"{_name} 与它在 z₀ = {_z0:.1f} 处的切线")
    _axes[0].set_xlabel("z")
    _axes[0].set_ylabel("f(z)")
    _axes[0].legend(fontsize=9)
    _axes[0].grid(True, alpha=0.3)

    _axes[1].plot(_z, _df(_z), color=_color, lw=2.4, label=f"{_name}'(z)")
    _axes[1].scatter([_z0], [_s0], s=90, color="tab:red", zorder=4)
    _axes[1].annotate(f"f'(z₀) = {_s0:.4f}", xy=(_z0, _s0), xytext=(_z0 + 1.0, _s0 + 0.25),
                      fontsize=11, color="tab:red",
                      arrowprops=dict(arrowstyle="->", color="tab:red"))
    _axes[1].set_title("导数曲线：红点是当前工作点")
    _axes[1].set_xlabel("z")
    _axes[1].set_ylabel("f'(z)")
    _axes[1].set_ylim(-0.15, 1.35)
    _axes[1].legend(fontsize=9)
    _axes[1].grid(True, alpha=0.3)

    # ---- 判词：工作点健康状况 ----
    if _s0 < 0.01:
        _verdict = (f"🚨 危险：f'(z₀) = {_s0:.2e}——"
                    + ("死区（ReLU 负区，导数恒 0，一睡不醒）" if _name in ("ReLU", "Leaky ReLU") and _s0 == 0
                       else "深度饱和（梯度传到这里近乎枯竭，22 课的大敌）"))
    elif _s0 < 0.1:
        _verdict = f"⚠️ 半饱和：f'(z₀) = {_s0:.3f}，梯度已衰减到只剩零头"
    else:
        _verdict = f"✅ 健康：f'(z₀) = {_s0:.3f}，梯度畅通"

    mo.vstack([
        _fig,
        mo.callout(mo.md(f"**实时判词：{_verdict}**"), kind="info"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 交互实验 2 · 折线魔术师

        基线是一条斜率 0.5 的直线，两个 ReLU 补丁随时待命：

        ```
        y = 0.5·x  +  a·relu(x - k₁)  +  b·relu(x - k₂)
        ```

        三段的斜率分别是 **0.5**（两个折点都没到）、**0.5 + a**
        （只翻过 k₁）、**0.5 + a + b**（两个都翻过）——每个 ReLU 神经元
        就是一次"到点生效的斜率改写"，改写量就是它的权重。
        20 课的 MLP 里，隐层干的就是这个活，只是折点和改写量由梯度自己学。
        """
    )
    return


@app.cell
def _(mo):
    # 交互控制：两个折点 + 两个斜率补丁
    knot1_slider = mo.ui.slider(-2.0, 0.0, value=-1.0, step=0.1, debounce=True,
                                label="📐 折点 k₁（-2 ~ 0，先醒的补丁）")
    knot2_slider = mo.ui.slider(0.5, 2.5, value=1.5, step=0.1, debounce=True,
                                label="📐 折点 k₂（0.5 ~ 2.5，后醒的补丁）")
    patch_a_slider = mo.ui.slider(-2.0, 2.0, value=-1.5, step=0.1, debounce=True,
                                  label="✂️ 补丁 a：翻过 k₁ 后斜率的变化量")
    patch_b_slider = mo.ui.slider(-2.0, 2.0, value=-1.0, step=0.1, debounce=True,
                                  label="✂️ 补丁 b：翻过 k₂ 后斜率的变化量")
    mo.vstack([knot1_slider, knot2_slider, patch_a_slider, patch_b_slider])
    return (knot1_slider, knot2_slider, patch_a_slider, patch_b_slider)


@app.cell
def _(knot1_slider, knot2_slider, mo, np, patch_a_slider, patch_b_slider, plt):
    # ===== 交互实验 2 · 折线魔术师（响应单元）=====
    _k1, _k2 = knot1_slider.value, knot2_slider.value
    _a, _b = patch_a_slider.value, patch_b_slider.value

    _xx = np.linspace(-3, 3, 600)
    _yy = 0.5 * _xx + _a * np.maximum(0, _xx - _k1) + _b * np.maximum(0, _xx - _k2)
    _relu = lambda _v: np.maximum(0, _v)

    _fig, _ax = plt.subplots(figsize=(7.8, 4.5))
    _ax.plot(_xx, 0.5 * _xx, color="gray", lw=1.2, ls="--", alpha=0.7, label="基线：0.5·x")
    _ax.plot(_xx, _a * _relu(_xx - _k1), color="tab:blue", lw=1.2, ls=":", alpha=0.8,
             label=f"补丁 a：{_a:+.1f}·relu(x−{_k1:.1f})")
    _ax.plot(_xx, _b * _relu(_xx - _k2), color="tab:green", lw=1.2, ls=":", alpha=0.8,
             label=f"补丁 b：{_b:+.1f}·relu(x−{_k2:.1f})")
    _ax.plot(_xx, _yy, color="tab:red", lw=2.8, label="合体折线", zorder=4)
    for _kk in (_k1, _k2):
        _ax.axvline(_kk, color="tab:red", alpha=0.25, lw=1.2)
    # 三段斜率标注（取每段中点立牌子）
    _seg_style = dict(fontsize=10, ha="center",
                      bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="tab:red", alpha=0.9))
    _ax.text((-3 + _k1) / 2, 0.5 * (-3 + _k1) / 2 + 0.45, f"斜率 0.5", **_seg_style)
    _ax.text((_k1 + _k2) / 2, 0.5 * (_k1 + _k2) / 2 + 0.45, f"斜率 {0.5 + _a:.1f}", **_seg_style)
    _ax.text((_k2 + 3) / 2, 0.5 * (_k2 + 3) / 2 + 0.45, f"斜率 {0.5 + _a + _b:.1f}", **_seg_style)
    _ax.set_title("两个 ReLU 补丁拼出的三段折线")
    _ax.set_xlabel("x")
    _ax.set_ylabel("y")
    _ax.set_ylim(-3.2, 3.2)
    _ax.legend(fontsize=8.5, loc="upper left")
    _ax.grid(True, alpha=0.3)

    _slopes = (0.5, 0.5 + _a, 0.5 + _a + _b)
    _verdict = (f"三段斜率：**{_slopes[0]:.1f} → {_slopes[1]:.1f} → {_slopes[2]:.1f}**。"
                f"补丁 a = {_a:+.1f}、补丁 b = {_b:+.1f}——"
                "ReLU 只在翻过折点后改写斜率，改写量 = 权重，一次生效、永不重复计费")

    mo.vstack([
        _fig,
        mo.callout(mo.md(f"**读数：{_verdict}**"), kind="info"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✏️ 动手练习（每课必做，本课 3 题）": mo.md(
                """
                **练习 1 · 手写 sigmoid 和它的导数**：两行函数 + 13 课的
                对称差分对账——这是 22 课反向传播的入场券，必须一次写对。

                **练习 2 · 数死神经元**：给定数据和权重（都造好了），
                统计 b = -2 和 b = +0.5 两种情况下"对所有样本都输出 0"的神经元
                个数——布尔掩码 + axis 的大练兵（05、08 课）。

                **练习 3 · 挑战 · softplus**：ReLU 的平滑替身
                softplus(z) = log(1 + eᶻ)。手写它、算出 z = 0 处的函数值和导数，
                再想想它和 ReLU 各自的取舍。

                规则不变：先把 `None` 换成你的答案再运行；写完再看参考答案。
                """
            )
        }
    )
    return


@app.cell
def _(mo, np):
    # ===== 练习 1 · 手写 sigmoid 与其导数 + 数值对账 =====
    def sigmoid_ex1(z):
        return None        # ← ① 补这行：1 / (1 + np.exp(-z))

    def dsigmoid_ex1(z):
        return None        # ← ② 补这行：σ(z)·(1 - σ(z))——直接调用上面的 sigmoid_ex1

    deriv_at_1_ex1 = None  # ← ③ 补完 ② 后填 dsigmoid_ex1(1.0)

    if deriv_at_1_ex1 is not None:
        try:
            _h = 1e-5
            _num = (sigmoid_ex1(1 + _h) - sigmoid_ex1(1 - _h)) / (2 * _h)   # 13 课对称差分
            _ana = float(deriv_at_1_ex1)
            if abs(_ana - _num) < 1e-6:
                _msg = f"✅ σ'(1) = **{_ana:.6f}**，对称差分对账误差 {abs(_ana - _num):.1e}——完美"
            else:
                _msg = (f"❌ 你的 σ'(1) = {_ana:.6f}，对称差分算出 {_num:.6f}——"
                        "② 的公式再对一对（是 σ·(1-σ)，不是 1-σ）")
        except TypeError:
            _msg = "⏳ 先把 ①② 两行补上，函数才跑得起来"
    else:
        _msg = "⏳ 补完 ①②，再算出 ③ 的数值"
    mo.md(_msg)
    return (deriv_at_1_ex1, dsigmoid_ex1, sigmoid_ex1)


@app.cell
def _(mo, np):
    # ===== 练习 2 · 数死神经元 =====
    _rng = np.random.default_rng(9)                # 数据已造好，别动种子
    _Xd = _rng.normal(0, 1, size=(200, 50))        # 200 样本 × 50 维
    _Wd = _rng.normal(0, 0.1, size=(50, 64))       # 64 个神经元的小权重

    # 提示骨架：
    #   _H = np.maximum(0, _Xd @ _Wd + b值)        # 前向（b 值自己填）
    #   int(np.all(_H == 0, axis=0).sum())         # 沿"样本"方向查全零（08 课 axis）
    #   np.all(..., axis=0) 把 (200, 64) 压成 (64,)：第 j 个元素 = 第 j 个神经元是否对全部样本输出 0

    dead_b_minus2_ex2 = None    # ← ① b = -2.0 时的死神经元个数（0~64 的整数）
    dead_b_plus05_ex2 = None    # ← ② b = +0.5 时的死神经元个数

    if dead_b_minus2_ex2 is not None and dead_b_plus05_ex2 is not None:
        if (dead_b_minus2_ex2, dead_b_plus05_ex2) == (38, 0):
            _msg = "✅ 38 和 0——死亡悬崖亲眼见：偏置太负，神经元成片睡死"
        else:
            _msg = (f"❌ 你数出 {dead_b_minus2_ex2} 和 {dead_b_plus05_ex2}"
                    "（期望 38 和 0）——检查 axis 是不是 0，以及 all/any 用对了没有")
    else:
        _msg = "⏳ 照提示骨架把两个数字数出来"
    mo.md(_msg)
    return (dead_b_minus2_ex2, dead_b_plus05_ex2)


@app.cell
def _(mo, np):
    # ===== 练习 3 · 挑战：softplus =====
    def softplus_ex3(z):
        return None        # ← ① 补这行：np.log(1 + np.exp(z))（大 z 会溢出？先不管，本题 z 很小）

    sp0_ex3 = None         # ← ② softplus(0) 的值（提示：和 ln 2 有关）
    spd0_ex3 = None        # ← ③ softplus 在 z = 0 处的导数（提示：softplus' 恰好就是 sigmoid！）

    if sp0_ex3 is not None and spd0_ex3 is not None:
        try:
            _ok = abs(float(sp0_ex3) - 0.6931) < 1e-3 and abs(float(spd0_ex3) - 0.5) < 1e-3
            _msg = (f"softplus(0) = **{float(sp0_ex3):.4f}**，softplus'(0) = **{float(spd0_ex3):.4f}**。"
                    + ("✅ ln 2 和 0.5——平滑版 ReLU 的身份证到手" if _ok
                       else "❌ 期望 0.6931（= ln 2）和 0.5（= σ(0)），再核一遍"))
        except (TypeError, ValueError):
            _msg = "⏳ 两个数字都补上再运行"
    else:
        _msg = "⏳ ①②③ 逐个补：先写函数，再算两个数"
    mo.md(_msg)
    return (softplus_ex3, sp0_ex3, spd0_ex3)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 1 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                def sigmoid_ex1(z):
                    return 1 / (1 + np.exp(-z))          # ①

                def dsigmoid_ex1(z):
                    s = sigmoid_ex1(z)
                    return s * (1 - s)                   # ② 导数用函数自己算

                deriv_at_1_ex1 = dsigmoid_ex1(1.0)       # ③ = 0.196612
                ```

                σ'(1) = **0.196612**，对称差分对账误差 ~10⁻¹²。
                两个工程习惯值得抄走：

                - **导数写成函数的组合**（先算 s 再返回 s·(1-s)），
                  不重复实现公式——一处改动，处处生效
                - 这个"用函数自己表示导数"的套路在 22 课反向传播里天天用：
                  前向时存下 σ(z)，反向时顺手就算出 σ(1-σ)，**不用重算 exp**
                """
            ),
            "✅ 练习 2 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                _H = np.maximum(0, _Xd @ _Wd - 2.0)      # b = -2 的前向
                dead_b_minus2_ex2 = int(np.all(_H == 0, axis=0).sum())   # 38

                _H2 = np.maximum(0, _Xd @ _Wd + 0.5)     # b = +0.5 的前向
                dead_b_plus05_ex2 = int(np.all(_H2 == 0, axis=0).sum())  # 0
                ```

                答案：**38 个**和 **0 个**。咀嚼三个细节：

                - `axis=0` 沿样本方向压：把 (200, 64) 压成 (64,) 的布尔，
                  第 j 个元素回答"第 j 个神经元是否对**所有**样本输出 0"——
                  08 课的 axis 直觉在这里是生死攸关的：axis 选错，
                  数出来的就是"每个样本上全零的维度数"，完全另一个问题
                - 38/64 ≈ 59% 的死亡率，和理论值 0.998²⁰⁰ ≈ 0.67 吻合
                  （每个神经元"全负"的概率，66 个里活下来 38 个，量级对得上）
                - b = +0.5 一个都不死：P(全负) = 0.31²⁰⁰ ≈ 10⁻¹⁰⁰——
                  高维世界里"有点危险"和"必然发生"之间隔着指数级的墙
                """
            ),
            "✅ 练习 3 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                def softplus_ex3(z):
                    return np.log(1 + np.exp(z))         # ① （工程版：np.logaddexp(0, z)）

                sp0_ex3 = softplus_ex3(0.0)              # ② = 0.6931 = ln 2
                spd0_ex3 = 1 / (1 + np.exp(0.0))         # ③ = 0.5 = σ(0)
                ```

                **softplus(0) = ln 2 ≈ 0.693，softplus'(0) = σ(0) = 0.5。**
                ②③ 两个数就是它的身份证：z = 0 处 ReLU 输出 0，
                softplus 输出 0.693——**处处平滑、没有死区**（导数恒为
                σ(z) > 0，再深的负区也留着一线生机）。

                取舍清单：

                | | ReLU | softplus |
                |---|---|---|
                | z = ±5 处的值 | 0 / 5 | 0.007 / 5.007 |
                | 计算成本 | 一次比较 | 一次 exp + log |
                | 死区 | 有（负区导数 0） | 无（导数恒正） |
                | 工业界 | **默认主力** | 偶尔见于需要平滑的场合 |

                一句话：softplus 用计算成本买"无死区"，ReLU 用"可能有死区"
                换极致便宜——深度学习大多数时候选后者，因为**快就是硬道理**
                （01 课的教训，在 activation 界重演）。
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
                1. **线性摞线性 = 线性**：矩阵连乘还是一个矩阵（12 课复合变换），
                   演示 2 用 2.7×10⁻¹⁵ 的浮点噪声办成铁案——没有非线性关节，
                   深度就是一句空话
                2. **一层的完整定义**：h = σ(X @ W + b)——W 变形（12 课）、
                   b 平移（12 课）、σ 弯折（本课）；从此"层数"才等于"弯折能力"
                3. **三位选手的地盘**：sigmoid 守输出层说概率（16 课），
                   tanh 零中心适合隐层（导数峰值 1），ReLU 正区导数恒 1、
                   计算最便宜——深度网络的默认主力
                4. **ReLU = 斜率补丁**：a·relu(x-k) 表示"过了折点 k，
                   斜率加 a"；折点数量就是容量（演示 3 的容量阶梯：
                   3 → 48 个折点，MSE 从 0.27 压到 0.0002）——
                   17 课容量旋钮的神经网络版
                5. **两大职业病**：饱和（sigmoid/tanh 两端导数趋 0，
                   z = ±10 时只剩 4.5×10⁻⁵ / 8×10⁻⁹）；死 ReLU（负区导数 0，
                   一睡不醒；死亡悬崖在 b = -1.5 到 -2 之间：4 → 41 个）
                6. **导数才是主角**：22 课反向传播每过一层乘一次 f'——
                   0.25²⁰ ≈ 10⁻¹³（梯度消失）vs ReLU 正区 1²⁰ = 1（原样送达）。
                   任何手写导数都要过 13 课对称差分这道验钞机
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ---

        **下一课预告** — `20_多层感知机`：

        零件全齐了：直线（W、b）会画、弯折（激活函数）会拐、
        连折线都能拼了。下一课把零件**摞成整机**——多层感知机（MLP）：
        隐层提特征、输出层做判决，一层喂一层。

        名场面预告：四个 XOR 点摆上桌，两个手设权重的隐层神经元
        各画一条线，输出层把两个结论**与**起来——18 课那堵墙，
        被一个 2-2-1 的小网络当场拆掉。你还会数清一个网络里到底
        有多少参数，以及为什么 784-128-10 的 MLP 有十万零八百多个旋钮。

        唯一的坏消息：手设权重拼得出 XOR，**随机权重怎么学**还不知道——
        这个悬念，正是 22 课反向传播的开场。

        🎆 关节装好了，去把层摞起来！
        """
    )
    return


if __name__ == "__main__":
    app.run()
