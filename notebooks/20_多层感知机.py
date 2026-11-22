import marimo

__generated_with = "0.25.2"
app = marimo.App(app_title="20 · 多层感知机", width="medium")


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
        # 20 · 多层感知机：把层摞起来

        **从零到 Transformer · 手写深度学习** ｜ 第 4 部分 · 神经网络（3/8）
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📖 本课目标（先读我）": mo.md(
                """
                **这一课只解决一个问题：把零件摞成整机——多层感知机（MLP）。**

                上一课立好了规矩：一层 = `h = σ(X @ W + b)`，σ 负责弯折。
                这一课把**两层**接起来：隐层先提特征，输出层再对特征做判决——
                "层"这个字从形容词变成动词。18 课那堵 XOR 之墙，会被一个
                只有 9 个参数的 2-2-1 小网络当场拆掉，而且**每个零件你都认识**。

                打法依然稳扎稳打：先记一本**维度账本**（矩阵形状怎么对上）；
                再手设权重拆 XOR 墙，顺藤摸瓜看清**隐层到底干了什么**
                （剧透：它把输入空间重新编码，让难题在新坐标系里变成
                简单题）；然后数清一个网络里有多少**参数**（旋钮）；
                最后看看**随机权重**和手设权重的天壤之别——结构一样、
                参数不同、功能千里之差，这正是"训练"二字的意义所在。

                学完你将能：

                - 一眼读出 MLP 的**维度账本**：(n, d) → (n, h) → (n, o)，
                  每一步哪个矩阵、谁乘谁、偏置加在哪
                - 用**手设权重**的 2-2-1 网络拆掉 XOR 墙，讲清两个隐单元
                  各自在量什么、输出层怎么把结论拼起来
                - 说出**隐层的本质**：空间折叠——数据在原空间不可分，
                  在隐层的输出空间里一条直线就分开了
                - 数出任意 MLP 的**参数量**（如 784-128-10 = 101,770 个旋钮），
                  并解释为什么参数得靠梯度去学、不能靠手拍

                前置：第 19 课的激活函数与"斜率补丁"、第 12 课的
                X @ W + b（行样本约定）与矩阵连乘、第 17 课的 solve
                （演示里读出层的老朋友）、第 18 课的 XOR 之墙（今天拆它）。

                一个提醒：本课全程**手设权重**——爽是爽，但"权重怎么自己学"
                的悬念会越攒越大。22 课的反向传播就是来兑现这张支票的。
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "🏗️ 概念 1/4 · 维度账本：矩阵形状怎么对上": mo.md(
                """
                第 12 课定过规矩：**数据按行摆**，X 是 (n, d)——n 个样本、
                每行 d 个特征。一层的全部形状问题，一张账本说清：

                ```
                X: (n, d)      W: (d, h)      b: (h,)
                Z = X @ W + b  →  (n, h)
                H = σ(Z)       →  (n, h)      （激活不改形状，只改数值）
                ```

                三条对账规则（以后天天对）：

                - **W 的行数 = 进来的特征数，列数 = 出去的神经元数**——
                  `X @ W` 要求 (n, d) @ (d, h)：内维 d 必须咬合（09 课的规矩）
                - **b 永远是出去那一维的长度** (h,)，靠广播加到每一行上（06 课）
                - **样本数 n 全程不变**——批量的意义：n 个样本一把算完，
                  不写 for 循环（01 课的老教训）

                一个名字先记住：中间这层的神经元个数 h 叫**宽度**（width），
                也叫隐单元数。h = 4 就是"这层有 4 个神经元"——
                每个神经元管输出的一列，各算各的 z、各过各的 σ。
                """
            ),
            "🧱 概念 2/4 · 摞两层：隐层提特征，输出层做判决": mo.md(
                """
                把两层接起来，就是本课的主角——**多层感知机**（MLP，
                Multi-Layer Perceptron；也叫前馈网络）：

                ```
                Z1 = X @ W1 + b1        第一层打分：(n, d) → (n, h)
                H  = ReLU(Z1)           隐层激活（弯折工序，19 课）
                out = H @ W2 + b2       输出层打分：(n, h) → (n, o)
                ```

                读这段代码的正确姿势是**分工**：

                - **隐层（H）**：h 个特征探测器。每个神经元从 d 个原始特征里
                  挑自己关心的组合（W1 的一**列**就是一个探测器的配方），
                  ReLU 决定"够不够格上报"——不够格就输出 0（19 课的斜率补丁）
                - **输出层（out）**：对 h 个新特征做一次**线性判决**——
                  和 18 课感知机一模一样，只不过它看到的不再是原始的 x，
                  而是隐层提炼过的 H

                所以 MLP 的本质是：**先用非线性把空间折一折，再在新空间里
                画直线**。折好了，直线的活就简单了——这正是下一概念要
                亲眼看到的事。

                术语对齐：`out` 后面通常还要接激活（回归接恒等、二分类接
                sigmoid、多分类接 softmax），那是 21 课损失函数的配对问题，
                本课先专注于"层怎么摞、形状怎么对"。
                """
            ),
            "🎭 概念 3/4 · 拆墙预告：XOR 的两把剪刀": mo.md(
                """
                18 课的 XOR 之墙：四个点、对角同号，一条直线怎么摆都
                至多对 75%。现在亮出拆墙方案——**两个隐单元、两把剪刀**：

                ```
                h1 = ReLU( x1 − x2 )      量"x1 比 x2 大多少"
                h2 = ReLU( x2 − x1 )      量"x2 比 x1 大多少"
                out = h1 + h2 − 0.5       两个探测器加起来，过 0.5 的门槛
                ```

                逐点验算（纸上就能算）：

                | 输入 | x1 − x2 | h1 | h2 | out | 判决 | 真值 |
                |---|---|---|---|---|---|---|
                | (0, 0) | 0 | 0 | 0 | −0.5 | −1 | −1 ✓ |
                | (1, 0) | 1 | 1 | 0 | +0.5 | +1 | +1 ✓ |
                | (0, 1) | −1 | 0 | 1 | +0.5 | +1 | +1 ✓ |
                | (1, 1) | 0 | 0 | 0 | −0.5 | −1 | −1 ✓ |

                四点全对。看懂这两把剪刀在量什么，XOR 的谜底就揭开了：
                **XOR 问的是"两个输入相不相等"**——h1 + h2 = |x1 − x2|
                恰好就是"差多少"。差得远（>0.5）判 +1，差不多（<0.5）判 −1。
                决策边界是两条对角线夹出的一条带子，18 课的"两条线围出
                一个角落"预言完全应验。

                更深一层的启示：原空间里线性不可分的数据，经过隐层这
                一折，**在 (h1, h2) 坐标下变得线性可分**——(0,0) 和 (1,1)
                都被折到了原点附近，(1,0) 和 (0,1) 各占一根轴。
                "空间折叠"这四个字，就是深度学习的全部野心。
                """
            ),
            "⚙️ 概念 4/4 · 参数量与「手拍不可行」定理": mo.md(
                """
                网络里所有可学习的旋钮叫**参数**（parameter）：

                ```
                参数量 = W1 的个数 + b1 的个数 + W2 的个数 + b2 的个数
                       = d·h   +   h   +  h·o   +   o
                ```

                小网络手算：2-4-1 → 2×4 + 4 + 4×1 + 1 = **17 个**。
                MNIST 手写数字的网络（784 个像素、128 个隐单元、10 个输出）：
                784×128 + 128 + 128×10 + 10 = **101,770 个**。

                十万零一千七百七十。假设你每秒钟能目测调好一个权重、
                一天调 8 小时，调完这套旋钮要 **3.5 年**——而且 18 课的
                经验告诉我们，单个权重的改动会牵动整个边界，根本不是
                "逐个调优"能收敛的活。

                这就是为什么"训练"必须是**自动的**：给一个损失函数打分
                （21 课），让梯度告诉每个旋钮该往哪边拧（22 课）。
                手设权重今天只干一件事：**证明结构有能力**——剩下的，
                交给梯度。

                顺带立个对照实验的flag：同样的 2-2-1 结构、换成随机
                权重，决策地图长什么样、准确率多少——演示 5 揭晓
                （剧透：连抛硬币都不如）。
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

        五组演示，把整机装配看个透：

        - **演示 1** · 维度账本：3 样本过 2-4-1 网络，形状逐站对账
        - **演示 2** · 拆墙名场面：手设权重的 2-2-1 网络按住 XOR 摩擦
        - **演示 3** · 隐层解剖：两把剪刀各量什么、h 空间里发生了什么
        - **演示 4** · 数旋钮：从 17 到 101,770，log 尺上看清参数爆炸
        - **演示 5** · 抽奖现场：同一结构换随机权重，地图面目全非
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 1 · 维度账本：3 个样本过一遍 2-4-1

        一批 3 个样本（3, 2），过一个 2-4-1 的网络。左边是流程图
        （14 课计算图的亲戚——从数据到输出，一站一站），右边把每个
        中间量的形状打印出来对账。

        对账三连问（试着自己抢答）：

        - `X @ W1` 时哪个维度必须咬合？（**内维 2**：X 的列 = W1 的行）
        - ReLU 之后形状变了吗？（**不变**，只改数值）
        - 样本数 3 到输出还剩几个？（**还是 3**——批量的意义）
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # 演示 1 · 维度账本：3 样本 × 2 特征 → 隐 4 → 出 1，逐站对形状
    from matplotlib.patches import FancyBboxPatch as _Box   # 圆角矩形 = 一站
    from matplotlib.patches import FancyArrowPatch as _Arrow

    _rng = np.random.default_rng(1)                # 固定种子
    _Xs = _rng.uniform(-1, 1, (3, 2))              # 3 个样本、2 个特征
    _W1s = _rng.normal(0, 1, (2, 4))               # 第一层权重：(进 2, 出 4)
    _b1s = _rng.normal(0, 0.1, 4)                  # 偏置：出去那维的长度
    _W2s = _rng.normal(0, 1, (4, 1))               # 第二层权重：(进 4, 出 1)
    _b2s = _rng.normal(0, 0.1, 1)

    _Z1 = _Xs @ _W1s + _b1s                        # (3,2)@(2,4)+(4,) → (3,4)
    _Hs = np.maximum(0, _Z1)                       # ReLU：形状不动，负数归零（19 课）
    _out = _Hs @ _W2s + _b2s                       # (3,4)@(4,1)+(1,) → (3,1)

    # ---- 左图：流程图 ----
    _fig, _axes = plt.subplots(1, 2, figsize=(12.5, 4.0),
                               gridspec_kw={"width_ratios": [1.4, 1]})
    _ax = _axes[0]
    _ax.set_xlim(0, 12)
    _ax.set_ylim(0.2, 4.2)
    _ax.axis("off")
    _stations = [
        ("X\n(3, 2)", 1.0), ("Z1 = X@W1+b1\n(3, 4)", 3.4),
        ("H = ReLU(Z1)\n(3, 4)", 6.0), ("out = H@W2+b2\n(3, 1)", 8.9),
    ]
    for _label, _xc in _stations:
        _ax.add_patch(_Box((_xc - 0.85, 1.6), 1.7, 1.2, boxstyle="round,pad=0.12",
                           facecolor="#e8f2fb", edgecolor="#1f77b4", lw=1.6))
        _ax.text(_xc, 2.2, _label, ha="center", va="center", fontsize=10)
    for _xa, _xb, _lab in [(1.95, 2.5, "W1: (2,4)\nb1: (4,)"),
                            (4.4, 5.1, ""), (7.0, 8.0, "W2: (4,1)\nb2: (1,)")]:
        _ax.add_patch(_Arrow((_xa, 2.2), (_xb, 2.2), arrowstyle="-|>",
                             mutation_scale=15, color="#1f77b4", lw=1.8))
    _ax.text(2.2, 3.35, "W1: (2, 4)\nb1: (4,)", ha="center", fontsize=8.5, color="gray")
    _ax.text(5.2, 3.35, "ReLU\n（19 课）", ha="center", fontsize=8.5, color="gray")
    _ax.text(7.5, 3.35, "W2: (4, 1)\nb2: (1,)", ha="center", fontsize=8.5, color="gray")
    _ax.set_title("一站一站：数据 → 隐层 → 输出")

    # ---- 右图：形状对账表（文字表格）----
    _ax2 = _axes[1]
    _ax2.axis("off")
    _rows = [
        ("X", str(_Xs.shape)), ("X @ W1", str((_Xs @ _W1s).shape)),
        ("+ b1 → Z1", str(_Z1.shape)), ("ReLU(Z1) → H", str(_Hs.shape)),
        ("H @ W2 + b2 → out", str(_out.shape)),
    ]
    _ax2.text(0.5, 0.95, "维度账本", ha="center", fontsize=12, weight="bold")
    for _i, (_nm, _sh) in enumerate(_rows):
        _yc = 0.82 - _i * 0.16
        _ax2.text(0.12, _yc, _nm, fontsize=10, family="monospace")
        _ax2.text(0.85, _yc, _sh, fontsize=10, family="monospace", color="tab:blue")
    _ax2.text(0.12, 0.02, "参数量 = 2×4 + 4 + 4×1 + 1 = 17", fontsize=10,
              color="tab:red")

    mo.vstack([
        _fig,
        mo.md(
            f"""
            | 站点 | 形状 | 谁在干活 |
            |---|---|---|
            | X | {_Xs.shape} | 3 个样本 × 2 个特征 |
            | Z1 = X@W1 + b1 | {_Z1.shape} | W1 变形（12 课），b1 平移（06 课广播） |
            | H = ReLU(Z1) | {_Hs.shape} | 19 课的弯折：负数归零 |
            | out = H@W2 + b2 | {_out.shape} | 输出层：对 4 个新特征做线性判决 |

            三个数顺手记住：**样本数 3 全程不变**、**ReLU 不改形状**、
            **参数量 17 个**（演示 4 会拿它和 MNIST 版对比）。
            """
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 2 · 拆墙名场面：手设权重按住 XOR

        概念 3 的两把剪刀，原样写进代码——`W1` 的第 1 列 (1, −1)
        是 h1 的配方、第 2 列 (−1, 1) 是 h2 的配方（注意：**W1 的列 =
        探测器配方**，维度账本第一课的行/列视角在此兑现）。

        背景色是网络打分 out 的地图（红蓝分明），黑线是决策边界
        out = 0——正是概念 3 预告的"两条对角线夹出一条带子"。
        四个干净点全部判对；连 40 个带噪声的点（18 课演示 4 的
        同款配方）也**一个不错，100%**。

        18 课那堵墙，倒了。用到的参数一共 9 个（W1 四个 + b1 两个 +
        W2 两个 + b2 一个）——**深度学习的第一次胜利，便宜得惊人**。
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # 演示 2 · 手设 2-2-1 网络拆 XOR 墙（参数全部手拍，无梯度）
    # ---- 拆墙网络（概念 3 的两把剪刀，原样照抄）----
    W1 = np.array([[1.0, -1.0],                   # 第 1 行：x1 对两个隐单元的投票
                   [-1.0, 1.0]])                  # 第 2 行：x2 的投票（列 = 探测器配方）
    b1 = np.array([0.0, 0.0])                     # 隐层门槛：0
    W2 = np.array([[1.0], [1.0]])                 # 输出层：h1 + h2
    b2 = np.array([-0.5])                         # 输出门槛：0.5

    # ---- 数据：干净 4 点 + 噪声 40 点（18 课演示 4 同款配方）----
    pts4 = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    labs4 = np.array([-1, 1, 1, -1])              # XOR 标签
    _rng = np.random.default_rng(7)
    _centers = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
    X_xor = np.vstack([_rng.normal(_c, 0.15, (10, 2)) for _c in _centers])
    y_xor = np.repeat(np.array([-1, -1, 1, 1]), 10)

    # ---- 前向：干净 4 点逐点验算 ----
    _H4 = np.maximum(0, pts4 @ W1 + b1)
    _out4 = (_H4 @ W2 + b2).ravel()
    _pred4 = np.where(_out4 > 0, 1, -1)

    # ---- 噪声 40 点的准确率 ----
    _pred_noise = np.where((np.maximum(0, X_xor @ W1 + b1) @ W2 + b2).ravel() > 0, 1, -1)
    _acc = float(np.mean(_pred_noise == y_xor))

    # ---- 决策地图的通用画法（后面反复用，先包好）----
    def draw_map(ax, W1_, b1_, W2_, b2_, X=None, y=None, title="", lim=(-0.7, 1.7), n=220):
        """画 2-?-1 网络的打分地图：红蓝着色 + out=0 黑线 + 可选数据点"""
        _gx = np.linspace(lim[0], lim[1], n)      # 网格（07 课 linspace）
        _gy = np.linspace(lim[0], lim[1], n)
        _GX, _GY = np.meshgrid(_gx, _gy)          # 展开成两张网格纸
        _G = np.column_stack([_GX.ravel(), _GY.ravel()])   # (n², 2)：每个网格点当样本
        _Z = (np.maximum(0, _G @ W1_ + b1_) @ W2_ + b2_).reshape(_GX.shape)
        ax.contourf(_GX, _GY, _Z, levels=25, cmap="RdBu", alpha=0.55)   # 红负蓝正
        ax.contour(_GX, _GY, _Z, levels=[0], colors="k", linewidths=2)  # 决策边界
        if X is not None:
            ax.scatter(X[y == 1, 0], X[y == 1, 1], s=30, color="tab:blue",
                       edgecolors="white", linewidths=0.7, zorder=3)
            ax.scatter(X[y == -1, 0], X[y == -1, 1], s=30, color="tab:red",
                       marker="s", edgecolors="white", linewidths=0.7, zorder=3)
        ax.set_title(title)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")

    _fig, _ax = plt.subplots(figsize=(7.2, 5.0))
    draw_map(_ax, W1, b1, W2, b2, X=X_xor, y=y_xor,
             title=f"手设 2-2-1：XOR 墙塌了（噪声点准确率 {_acc:.0%}）")
    for _p, _o in zip(pts4, _out4):
        _ax.annotate(f"{_o:+.1f}", xy=tuple(_p), xytext=(_p[0] - 0.04, _p[1] + 0.09),
                     fontsize=9, ha="center", weight="bold",
                     color="white" if _o < 0 else "black")
    _fig.tight_layout()
    _fig
    return (W1, W2, X_xor, b1, b2, draw_map, labs4, pts4, y_xor)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        读图（黑线 = 决策边界，红蓝 = 两类领地）：

        - 四个标注的数是干净 4 点的 out 值：**(0,0) 和 (1,1) 都是 −0.5，
          (1,0) 和 (0,1) 都是 +0.5**——两把剪刀把"相不相等"翻译成了
          一个带符号的分数，对称得漂亮
        - 边界是 |x1 − x2| = 0.5 的两条平行线，夹出一条对角带子：
          **带子外 = 不相等 = +1**，带子内 = 相等 = −1
        - 噪声点 100% 落对领地——σ=0.15 的抖动还没有大到能把点
          甩过边界（18 课手摆直线只有 75% 的天花板，这里是 100%）

        一个值得停三秒的对比：18 课的感知机在这份数据上挣扎 30 轮
        只拿到 52.5%；今天 9 个手拍参数拿到 100%。**结构（多层+弯折）
        带来的能力跃迁，比任何"努力训练单层"都大**——这是深度学习
        的第一性原理。
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 3 · 隐层解剖：两把剪刀各量什么，h 空间发生了什么

        左中两图：h1 和 h2 各自的输出地图（亮 = 输出大，黑 = 输出 0）。
        h1 沿"左上 → 右下"方向渐亮（x1 越比 x2 大，值越高）；
        h2 镜像对称。**两个探测器方向互补，各管半边天**。

        右图是本课的灵魂：把 40 个噪声点**变换到 (h1, h2) 坐标**重新画——
        (0,0) 和 (1,1) 两团被折到了原点附近，(1,0) 和 (0,1) 各占一根轴。
        18 课不可分的 XOR，在隐层的输出空间里**一条直线（黑线，
        h1 + h2 = 0.5）就分开了**——输出层干的活，就是这条直线。

        这就是"隐层 = 特征提取器、输出层 = 简单判决器"的全部含义：
        **难，难在坐标；换好坐标，难题变简单题**。往后每一层网络，
        都在干"换坐标"这件事——层数越深，坐标换得越狠。
        """
    )
    return


@app.cell
def _(W1, X_xor, b1, np, plt, y_xor):
    # 演示 3 · 隐层解剖：h1 / h2 的地图 + h 空间的散点
    _gx = np.linspace(-0.7, 1.7, 220)
    _GX, _GY = np.meshgrid(_gx, _gx)
    _G = np.column_stack([_GX.ravel(), _GY.ravel()])
    _Hg = np.maximum(0, _G @ W1 + b1)             # 网格点过隐层：(n², 2)

    _Hn = np.maximum(0, X_xor @ W1 + b1)          # 40 个噪声点的隐层输出

    _fig, _axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    _im0 = _axes[0].contourf(_GX, _GY, _Hg[:, 0].reshape(_GX.shape),
                             levels=20, cmap="Blues", alpha=0.9)
    _axes[0].set_title("h₁ = ReLU(x₁ − x₂)\n量「x₁ 比 x₂ 大多少」")
    _axes[0].set_xlabel("x1")
    _axes[0].set_ylabel("x2")
    _fig.colorbar(_im0, ax=_axes[0], shrink=0.85)

    _im1 = _axes[1].contourf(_GX, _GY, _Hg[:, 1].reshape(_GX.shape),
                             levels=20, cmap="Blues", alpha=0.9)
    _axes[1].set_title("h₂ = ReLU(x₂ − x₁)\n量「x₂ 比 x₁ 大多少」（镜像）")
    _axes[1].set_xlabel("x1")
    _axes[1].set_ylabel("x2")
    _fig.colorbar(_im1, ax=_axes[1], shrink=0.85)

    _axes[2].scatter(_Hn[y_xor == 1, 0], _Hn[y_xor == 1, 1], s=32, color="tab:blue",
                     label="+1（不相等）")
    _axes[2].scatter(_Hn[y_xor == -1, 0], _Hn[y_xor == -1, 1], s=32, color="tab:red",
                     marker="s", label="-1（相等）")
    _hk = np.linspace(-0.1, 1.4, 50)              # h 空间里的判决线：h1 + h2 = 0.5
    _axes[2].plot(_hk, 0.5 - _hk, color="k", lw=2.2, label="输出层：h₁+h₂ = 0.5")
    _axes[2].annotate("(0,0) 团和 (1,1) 团\n被折到原点附近", xy=(0.02, 0.06),
                      xytext=(0.42, 0.62), fontsize=9,
                      arrowprops=dict(arrowstyle="->", color="gray"))
    _axes[2].set_title("h 空间：XOR 变成一条直线的活")
    _axes[2].set_xlabel("h1（x₁−x₂ 的正半边）")
    _axes[2].set_ylabel("h2（x₂−x₁ 的正半边）")
    _axes[2].set_xlim(-0.1, 1.15)
    _axes[2].set_ylim(-0.1, 1.15)
    _axes[2].legend(fontsize=8.5, loc="upper right")
    _axes[2].grid(True, alpha=0.3)

    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 4 · 数旋钮：从 17 到 795,010

        把参数量公式套在四个网络上，log 尺一排（数值跨 4 个半数量级，
        线性尺会把小网络压成看不见的火柴棍——09 课的老规矩）：

        | 网络 | 参数量 | 算式 |
        |---|---|---|
        | 2-4-1（演示 1） | 17 | 2×4 + 4 + 4×1 + 1 |
        | 2-8-1（宽一点） | 33 | 2×8 + 8 + 8×1 + 1 |
        | 784-128-10（MNIST 入门款） | **101,770** | 784×128 + 128 + 128×10 + 10 |
        | 784-1000-10（宽奢款） | **795,010** | 784×1000 + 1000 + 1000×10 + 10 |

        看 log 尺的台阶：**宽度从 128 涨到 1000（不到 8 倍），
        参数量涨到 7.8 倍**——参数量对宽度是近似线性的；但对**输入维度 d**
        也是线性的：MNIST 的 784 维输入，光 W1 一层就是十万级。
        25 课训练 MNIST 时，这十万个旋钮会全部交给梯度接管。

        顺带一个反差萌：今天拆 XOR 墙只用了 9 个参数——问题的难度
        和参数的多少，从来不是简单挂钩（18 课的收敛定理早就提醒过：
        关键在几何，不在大小）。
        """
    )
    return


@app.cell
def _(np, plt):
    # 演示 4 · 参数量大排行（log 尺）
    _names = ["2-4-1", "2-8-1", "784-128-10", "784-1000-10"]
    _counts = [2*4 + 4 + 4*1 + 1,                # 17
               2*8 + 8 + 8*1 + 1,                # 33
               784*128 + 128 + 128*10 + 10,      # 101770
               784*1000 + 1000 + 1000*10 + 10]   # 795010

    _fig, _ax = plt.subplots(figsize=(7.8, 4.2))
    _bars = _ax.bar(_names, _counts, color=["tab:blue", "tab:cyan", "tab:orange", "tab:red"],
                    alpha=0.88)
    _ax.set_yscale("log")                         # 数值跨 4 个半数量级（09 课）
    for _b, _c in zip(_bars, _counts):
        _ax.text(_b.get_x() + _b.get_width() / 2, _c * 1.25, f"{_c:,}",
                 ha="center", fontsize=10, weight="bold")
    _ax.set_ylabel("参数量（log 尺）")
    _ax.set_title("数旋钮：宽度和输入维度一涨，参数量起飞")
    _ax.set_ylim(8, 3e6)
    _ax.grid(True, alpha=0.3, axis="y", which="both")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 5 · 抽奖现场：随机权重的地图

        同样的 2-2-1 结构，权重换成随机数（三个种子各抽一次），
        和手设版摆在一起看。地图的"地形"完全由参数决定：
        随机版要么一条歪线切出两块不搭边的领地（50%）、
        要么几乎全线判负（55%）、甚至学反了（35%）——
        **结构给了能力，参数才给功能**。

        35% 比瞎猜还差这个现象值得多看一眼：瞎猜基线是 50%，
        "学反了"的网络把 +1 领地和 −1 领地整个对调——
        55% 和 35% 其实是同一种烂，只是方向相反
        （把输出取个负号，35% 立刻变 65%）。

        训练，就是在参数空间里从"抽奖现场"走到"拆墙现场"的路。
        21 课先造计分板（损失函数），22 课再造导航仪（反向传播）。
        """
    )
    return


@app.cell
def _(W1, X_xor, b1, b2, draw_map, np, plt, y_xor):
    # 演示 5 · 同结构换随机权重：三个种子 + 手设版，四连拍
    _fig, _axes = plt.subplots(1, 4, figsize=(13.6, 3.7), sharey=True)
    for _ax, _seed in zip(_axes[:3], [0, 1, 2]):
        _r = np.random.default_rng(_seed)         # 每个种子抽一套随机权重
        _Wr = _r.normal(0, 1, (2, 2))             # 隐层权重随机
        _br = _r.normal(0, 0.5, 2)
        _Vr = _r.normal(0, 1, (2, 1))             # 输出层权重随机
        _cr = _r.normal(0, 0.5, 1)
        _pred = np.where((np.maximum(0, X_xor @ _Wr + _br) @ _Vr + _cr).ravel() > 0, 1, -1)
        _acc = float(np.mean(_pred == y_xor))
        draw_map(_ax, _Wr, _br, _Vr, _cr, X=X_xor, y=y_xor,
                 title=f"随机权重（种子 {_seed}）：{_acc:.0%}")
    draw_map(_axes[3], W1, b1, np.array([[1.0], [1.0]]), b2, X=X_xor, y=y_xor,
             title="手设权重：100%")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 🎛️ 交互实验

        **输出层调参师**——隐层的两把剪刀固定不动（h1、h2 照常量差），
        输出层的三个旋钮交给你：w₁、w₂（两个特征的表决权）和 b（门槛）。

        ```
        out = w₁·h₁ + w₂·h₂ + b
        ```

        拖动滑块，看决策带怎么变形、噪声点的准确率怎么跳。玩法建议：

        1. 先把 b 从 −0.5 拖到 +0.5：整条带子翻面——XOR 变 XNOR
           （练习 3 的答案就藏在这一拖里）
        2. 把 w₁ 拖成 −2：带子拧成斜的，单边失守——**两个特征
           缺一不可**，这正是"加法判决"的纪律
        3. 试着找出 100% 之外你最喜欢的一版参数，体会"参数空间"
           里有多少种摆法
        """
    )
    return


@app.cell
def _(mo):
    # 交互控制：输出层三旋钮（debounce：松手才重算）
    wout1_slider = mo.ui.slider(-2.0, 2.0, value=1.0, step=0.1, debounce=True,
                                label="🎛️ w₁：h₁ 的表决权（-2 ~ 2）")
    wout2_slider = mo.ui.slider(-2.0, 2.0, value=1.0, step=0.1, debounce=True,
                                label="🎛️ w₂：h₂ 的表决权（-2 ~ 2）")
    bias_slider = mo.ui.slider(-2.0, 2.0, value=-0.5, step=0.1, debounce=True,
                               label="🎛️ b：输出层门槛（-2 ~ 2）")
    mo.vstack([wout1_slider, wout2_slider, bias_slider])
    return (bias_slider, wout1_slider, wout2_slider)


@app.cell
def _(W1, X_xor, b1, bias_slider, draw_map, mo, np, plt, wout1_slider, wout2_slider, y_xor):
    # ===== 交互实验 · 输出层调参师（响应单元）=====
    _w1, _w2, _b = wout1_slider.value, wout2_slider.value, bias_slider.value
    _W2_now = np.array([[_w1], [_w2]])
    _b2_now = np.array([_b])

    _pred = np.where((np.maximum(0, X_xor @ W1 + b1) @ _W2_now + _b2_now).ravel() > 0, 1, -1)
    _acc = float(np.mean(_pred == y_xor))
    _n_pos = int(np.sum(_pred == 1))

    _fig, _ax = plt.subplots(figsize=(7.0, 4.8))
    draw_map(_ax, W1, b1, _W2_now, _b2_now, X=X_xor, y=y_xor,
             title=f"out = {_w1:+.1f}·h₁ + {_w2:+.1f}·h₂ {_b:+.1f}　准确率 {_acc:.0%}")

    # ---- 判词：这条直线摆得怎么样 ----
    if _acc == 1.0:
        _verdict = "✅ 100%：输出层这条直线摆对了——隐层折好的空间里，判决就是这么简单"
    elif _n_pos == 0 or _n_pos == len(y_xor):
        _verdict = (f"🚨 一边倒：{_n_pos}/{len(y_xor)} 个点全判 {'+1' if _n_pos else '-1'}——"
                    "门槛 b 太高或太低，直线整个跑出了数据带")
    elif _acc >= 0.9:
        _verdict = f"⚠️ 接近了：{_acc:.0%}——带子的位置差一口气，微调 b 试试"
    else:
        _verdict = f"❌ 领地切歪了：{_acc:.0%}——注意 w₁ 和 w₂ 是不是一正一负把带子拧翻了"

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
                **练习 1 · 补全前向函数**：`forward_mlp` 的②③两行被抽走了——
                这是全课程出镜率最高的函数，必须闭眼写对。

                **练习 2 · 数参数**：笔算（或用代码数）两个网络的参数量：
                2-4-1 和 784-128-10。答案就是两个整数。

                **练习 3 · 挑战 · XNOR 改装**：拆墙网络的其他零件全不动，
                只换输出层的 `W2`（两个数）和 `b2`（一个数），
                把 XOR 网络改装成 XNOR 网络——相等判 +1、不相等判 −1。
                提示：交互实验里"把 b 翻面"那一幕。

                规则不变：先把 `None` 换成你的答案再运行；写完再看参考答案。
                """
            )
        }
    )
    return


@app.cell
def _(W1, W2, b1, b2, labs4, mo, np, pts4):
    # ===== 练习 1 · 补全 forward_mlp =====
    def forward_mlp(X, W1, b1, W2, b2):
        """两层 MLP 前向：返回 (Z1, H, out)——三个中间量都要交出去"""
        Z1 = X @ W1 + b1              # ① 第一层打分（这行不变）
        H = None                      # ← ② 补：np.maximum(0, Z1)
        out = None                    # ← ③ 补：H @ W2 + b2
        return Z1, H, out

    fwd_ex1 = None                    # ← ④ 补完后，把这行换成 forward_mlp（就写函数名）

    if fwd_ex1 is not None:
        try:
            _Z1, _H, _out = fwd_ex1(pts4, W1, b1, W2, b2)
            _H_want = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
            if np.allclose(_H, _H_want) and np.allclose(_out.ravel(), [-0.5, 0.5, 0.5, -0.5]):
                _msg = ("✅ 四个点的 H 和 out 全部对上——前向传播，会写了"
                        "（Z1 顺手也验了：形状 (4, 2)）")
            else:
                _msg = (f"❌ H = {np.round(_H, 2).tolist()}，期望 {_H_want.tolist()}；"
                        "out 应为 [-0.5, 0.5, 0.5, -0.5]——检查 ②③ 两行的顺序和形状")
        except TypeError:
            _msg = "⏳ H 或 out 还是 None——②③ 两行补了吗？"
    else:
        _msg = "⏳ 先补 ②③，再把 fwd_ex1 换成 forward_mlp"
    mo.md(_msg)
    return (forward_mlp, fwd_ex1)


@app.cell
def _(mo, np):
    # ===== 练习 2 · 数参数 =====
    # 公式：参数量 = d·h + h + h·o + o（概念 4）
    #   2-4-1：        2×4 + 4 + 4×1 + 1 = ?
    #   784-128-10：   784×128 + 128 + 128×10 + 10 = ?

    n_241_ex2 = None      # ← ① 2-4-1 的参数量（一个整数）
    n_mnist_ex2 = None    # ← ② 784-128-10 的参数量（一个整数）

    if n_241_ex2 is not None and n_mnist_ex2 is not None:
        if (n_241_ex2, n_mnist_ex2) == (17, 101770):
            _msg = (f"✅ **{n_241_ex2}** 和 **{n_mnist_ex2:,}**——数对了。"
                    "顺便记住：MNIST 入门款的十万旋钮，25 课全交给梯度接管")
        else:
            _msg = (f"❌ 你数出 {n_241_ex2} 和 {n_mnist_ex2}（期望 17 和 101,770）——"
                    "别忘了偏置也要数：每个神经元一个 b")
    else:
        _msg = "⏳ 两个整数，笔算或代码都行"
    mo.md(_msg)
    return (n_241_ex2, n_mnist_ex2)


@app.cell
def _(W1, b1, labs4, mo, np, pts4):
    # ===== 练习 3 · 挑战：XOR 网络改装成 XNOR =====
    # 隐层两把剪刀原样保留：h1 = ReLU(x1-x2)、h2 = ReLU(x2-x1)
    # 目标：相等 (0,0)、(1,1) → +1；不相等 → -1

    W2_new_ex3 = None     # ← ① 新的输出层权重（形如 np.array([[?], [?]])）
    b2_new_ex3 = None     # ← ② 新的输出层偏置（一个数）

    if W2_new_ex3 is not None and b2_new_ex3 is not None:
        try:
            _W2n = np.asarray(W2_new_ex3, dtype=float).reshape(2, 1)
            _bn = float(b2_new_ex3)
            _pred = np.where((np.maximum(0, pts4 @ W1 + b1) @ _W2n + _bn).ravel() > 0, 1, -1)
            if np.all(_pred == -labs4):
                _msg = f"✅ 预测 {_pred.tolist()}——XNOR 改装成功（你的方案：W2={_W2n.ravel().tolist()}, b={_bn}）"
            else:
                _msg = (f"❌ 预测 {_pred.tolist()}，期望 [1, -1, -1, 1]——"
                        "提示：把原判决整个翻个面（三个数同时变号）就行")
        except (TypeError, ValueError):
            _msg = "⏳ W2 要是两个数（np.array([[-1.0], [-1.0]]) 这种形状），b 是一个数"
    else:
        _msg = "⏳ 补上两个数和一个数（提示：交互实验里 b 翻面的那一幕）"
    mo.md(_msg)
    return (W2_new_ex3, b2_new_ex3)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 1 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                H = np.maximum(0, Z1)        # ② 隐层激活（19 课的弯折）
                out = H @ W2 + b2            # ③ 输出层线性判决
                ```

                验算：四个点的 H = [[0,0], [1,0], [0,1], [0,0]]，
                out = [−0.5, +0.5, +0.5, −0.5]。

                三个值得抄走的工程习惯：

                - **中间量全部交出去**（返回 Z1、H、out 三个）——22 课
                  反向传播要用它们算梯度，这个"前向顺手存中间量"的套路
                  叫 **缓存**（cache），从今天养成
                - 变量名用 Z1 / H / out 分工命名：Z 是"打分"、H 是"激活后"，
                  一眼看出过了没过弯折
                - 形状自查：Z1 和 H 都是 (4, 2)，out 是 (4, 1)——
                  写完 `print(x.shape)` 对一遍账，胜过 debug 一小时
                """
            ),
            "✅ 练习 2 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                n_241_ex2 = 2*4 + 4 + 4*1 + 1          # = 17
                n_mnist_ex2 = 784*128 + 128 + 128*10 + 10   # = 101770
                ```

                拆开看 784-128-10 的十万旋钮都花在哪：

                | 部件 | 个数 | 占比 |
                |---|---|---|
                | W1（784×128） | 100,352 | 98.6% |
                | b1（128） | 128 | 0.1% |
                | W2（128×10） | 1,280 | 1.3% |
                | b2（10） | 10 | 0.01% |

                **98.6% 的参数在 W1**——输入维度 d 才是参数量大头
                （每个隐单元都要配齐 784 个权重）。这就是第 5 部分要讲的故事：
                图像的局部结构不需要每个像素连每个神经元，卷积
                把"784 个权重/单元"砍成"9 个权重/单元"——26 课见。
                """
            ),
            "✅ 练习 3 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                W2_new_ex3 = np.array([[-1.0], [-1.0]])
                b2_new_ex3 = 0.5
                ```

                预测翻成 [1, −1, −1, 1]——XNOR 到手。玄机一句话：
                **XNOR = −XOR**，所以把输出层整个取负（w₁、w₂、b 三个
                同时变号），判决自动翻面。交互实验里"把 b 从 −0.5 拖到
                +0.5、同时把两个 w 拖成 −1"就是这个动作。

                两点品味：

                - **隐层一毫米没动**。h1、h2 还是那两把剪刀——特征不变，
                  只换对特征的**解读**。"表示"和"判决"分层，正是 MLP
                  的设计哲学，也是深度学习"特征学习"四个字的来历
                - 等价答案有无穷多组：任何 (w₁, w₂, b) 与 (−1, −1, 0.5)
                  成正比的都行（18 课就说过：等比例放大不改变边界）——
                  参数空间里"好解"是一个方向，不是一个点。22 课的梯度
                  下降找到的，也未必和手拍的同一个，但一样好用
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
                1. **维度账本**：(n, d) @ (d, h) + (h,) → (n, h)；
                   W 的行数=进、列数=出，b 长度=出，样本数 n 全程不变，
                   激活不改形状只改数值
                2. **MLP = 隐层提特征 + 输出层做判决**：
                   Z1 = X@W1+b1 → H = σ(Z1) → out = H@W2+b2；
                   本质是"先非线性折空间，再在新空间画直线"
                3. **拆墙实录**：2-2-1 手设网络 9 个参数拿下 XOR 100%——
                   两把剪刀 ReLU(x₁−x₂)、ReLU(x₂−x₁) 量"差多少"，
                   输出层 h₁+h₂−0.5 判"差得够不够远"
                4. **隐层 = 空间折叠**：原空间不可分的数据，在 (h₁, h₂)
                   坐标下一条直线就分开——(0,0) 和 (1,1) 团被折到原点，
                   (1,0)、(0,1) 各占一轴。换坐标，难题变简单题
                5. **参数量**：d·h + h + h·o + o；2-4-1 = 17，
                   784-128-10 = 101,770（98.6% 在 W1——输入维度是大头，
                   26 课卷积的伏笔）
                6. **结构给能力，参数给功能**：同一结构随机权重只有
                   35%~55%（连抛硬币都不如），手设权重 100%——
                   "训练"就是从抽奖现场走到拆墙现场的路，下一课先造计分板
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

        **下一课预告** — `21_损失函数与softmax`：

        网络会前向了，但还欠两样东西：**计分板**和**导航仪**。
        下一课先造计分板——损失函数怎么和网络输出"配对上岗"：

        - 回归 → 恒等输出 + MSE（15 课的旧相识）
        - 二分类 → sigmoid + 交叉熵（16 课的旧相识）
        - **多分类 → softmax + 交叉熵（本课新主角）**——十个神经元
          各报一个分数，softmax 把分数变成"一套加起来等于 1 的概率"

        还有两个工程硬菜：logits 冲到 1000 时 softmax 会当场溢出，
        得靠 **log-sum-exp** 这剂退烧药（09 课浮点噪声的老账）；
        以及一个漂亮到不像话的结论——softmax 配交叉熵，
        对 logit 的梯度恰好是 **p − y**（预测减真值，一行就是全部），
        22 课反向传播的最后一环，就等它上桌。

        🎆 整机装好了，去给它装计分板！
        """
    )
    return


if __name__ == "__main__":
    app.run()
