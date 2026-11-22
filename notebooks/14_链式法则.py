import marimo

__generated_with = "0.25.2"
app = marimo.App(app_title="14 · 链式法则", width="medium")


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

    # ==== 中文乱码修复（11 课的老朋友，一次设置全程生效）====
    # matplotlib 默认字体（DejaVu Sans）不含汉字，中文会显示成方框 □□□
    # 修法：给 font.sans-serif 一张"候选字体名单"，matplotlib 从左到右
    # 挑第一个系统里装了的：
    #   PingFang SC                  → macOS 自带"苹方"
    #   Microsoft YaHei / SimHei     → Windows 自带"微软雅黑 / 黑体"
    #   Noto Sans SC / WenQuanYi Zen Hei → Linux 常见"思源黑体 / 文泉驿"
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Microsoft YaHei", "SimHei",
        "Noto Sans SC", "WenQuanYi Zen Hei",
    ]
    # 顺手修负号：中文字体常缺"数学负号"字形，不设这行负数会变方框
    # （本课的导数动不动就是负数，这行必须开）
    plt.rcParams["axes.unicode_minus"] = False
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # 14 · 链式法则

        **从零到 Transformer · 手写深度学习** ｜ 第 2 部分 · 深度学习的数学（4/4）
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📖 本课目标（先读我）": mo.md(
                """
                **这一课只解决一个问题：函数套着函数，导数怎么传？**

                第 13 课收尾时留了一句预告：单层函数的梯度会算了，但神经网络是
                函数套函数的**俄罗斯套娃**，对最外层求导必须一环一环传进去——
                链式法则，反向传播的引擎。今天兑现承诺：**把套娃求导拆成齿轮逐级传动**。

                学完你将能：

                - 把 f(g(x)) 看成两级齿轮：x 先过内层机器 g，输出再进外层机器 f
                - 默写链式法则 `d/dx f(g(x)) = f'(g(x)) × g'(x)`，并说清
                  "**外层导数要在中间结果处取值**"这个最容易踩的坑
                - 用"放大倍数连乘"把变化率的传递讲成齿轮传动（并用一张数字表格把它钉死）
                - 画出**计算图**：前向从左到右算值、反向从右到左传梯度——
                  第 22 课手写反向传播的蓝图，本课先画好

                前置：第 13 课的导数与对称差分（本课每组演示都请它当"数值裁判"）、
                第 11 课的 matplotlib 画图基本功、第 09 课的 `np.allclose` 浮点容差。
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "🎯 概念 1/4 · 复合函数 = 俄罗斯套娃": mo.md(
                """
                第 13 课的导数都是"一层"函数：x 进去，y 出来。可现实里（尤其深度学习里）
                的函数几乎都是**套娃**：

                ```
                f(g(x))     ← 打开最外层，里面还藏着一个 g(x)
                ```

                - **内层机器 g**：x 先进它，吐出**中间结果** u = g(x)
                - **外层机器 f**：u 再进它，吐出最终结果 f(g(x))

                比如 (3x+1)²：内层 g(x) = 3x+1（先乘 3 加 1），外层 f(u) = u²（再平方）。
                先里后外，一层一层打开——俄罗斯套娃。

                深度学习就是一台巨型套娃机：数据进第 1 层 → 输出进第 2 层 → …… →
                第 N 层的输出进损失函数。想训练最里面的参数，就得对最外层的损失求导——
                **导数必须穿过所有层**。怎么穿？这正是本课主角要回答的。
                """
            ),
            "⚙️ 概念 2/4 · 链式法则本体：变化率是乘法传递的": mo.md(
                """
                ```
                d/dx f(g(x)) = f'(g(x)) × g'(x)
                ```

                读法（每个字都有讲究）：

                - **f'(g(x))**：外层函数的导数，但要在**中间结果 g(x) 处**取值——
                  不是随便找个地方求，是在"内层刚吐出来的那个数"上求
                - **g'(x)**：内层函数对 x 的导数
                - **×**：两层的变化率**相乘**

                直觉是一台**齿轮组**：你摇动手柄（x 动一点），第一级齿轮转 g' 圈，
                带动第二级又放大 f' 倍——手柄转速 × 各级齿轮比 = 末端转速。
                一句话：**变化率是乘法传递的**。

                三层就再乘一环：`d/dx f(g(h(x))) = f'(g(h(x))) × g'(h(x)) × h'(x)`。
                N 层 = N 个导数连乘——这个"连乘"，就是第 4 部分神经网络的心跳。
                """
            ),
            "🧠 概念 3/4 · 为什么深度学习离了它就死": mo.md(
                """
                把一个 N 层网络写成公式：

                ```
                Loss = Loss(第N层(第N-1层(...(第1层(x))...)))
                ```

                训练要更新第 1 层的参数 w1，就得算 dLoss/dw1——从损失一路穿到最底层，
                路上每一层都得乘上那层的导数。链式法则保证：只要一环一环乘下去、
                不重不漏，答案就是对的。

                反过来：**链一断，全盘错**。哪一层忘了乘自己的导数，传到最底层的梯度
                就彻底错了——模型看起来在训练，其实在朝一个错误的方向学。
                变化像多米诺一样逐级传导，而每级都会被本级导数"变速"：
                漏乘一环，骨牌的受力就全变了。

                所以**反向传播（backpropagation）不是新数学**——它就是一套
                "系统化、不重不漏地执行链式法则"的工程方案：第 22 课手写它，
                第 8 部分过渡到 PyTorch 时，autograd 干的还是这件事。
                """
            ),
            "🗺️ 概念 4/4 · 计算图：链式法则的图纸": mo.md(
                """
                复合函数一旦超过三层，公式开始难读。工程解法：**画图**。

                - **节点 = 数**（输入 x、中间结果、损失 L）
                - **箭头 = 运算**（乘、加、平方……）

                ```
                x →(×3)→ u →(+1)→ v →(平方)→ L
                ```

                两条走法：

                - **前向（从左到右）**：把每个节点的值算出来——这就是"跑一遍网络算损失"
                - **反向（从右到左）**：每个节点把"上游传回来的变化率"乘上"本级导数"，
                  往左传——这就是"算梯度"

                这张图叫**计算图**（computational graph）。PyTorch 的 autograd、
                各家框架的求导器，骨子里都在维护这样一张图；第 22 课手写反向传播时，
                跑的就是它。本课先把蓝图画好。
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

        四组演示，把"连乘"从公式变成眼见为实：

        - **演示 1** · 两层齿轮：链式法则的解析导数 vs 数值导数，两条曲线完全重合
        - **演示 2** · 三层套娃：sin(3x²+1) 在一点逐环节算"转速"，三环连乘成总导数
        - **演示 3** · 齿轮传动实验：亲手把 x 拧动 0.001，看变化量逐级传导、放大倍数连乘
        - **演示 4** · 画出计算图：前向算值（蓝）、反向传梯度（橙）——第 22 课的预告海报
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 1 · 两层齿轮：链式法则 vs 数值导数

        最简单的复合：内层 g(x) = 3x+1（线性机器），外层 f(u) = u²（平方机器），
        复合 F(x) = (3x+1)²。

        链式法则给出解析导数：f'(g(x)) × g'(x) = 2(3x+1) × 3 = 6(3x+1)。
        对不对？请第 13 课的**对称差分**当裁判：如果链式法则算错了，
        两条导数曲线立刻分家。

        （fig/ax 二件套、双子图、图例网格——第 11 课练的画图基本功，直接上岗。）
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # 演示 1 · 两层齿轮：f(g(x)) = (3x+1)²，链式法则 vs 数值导数
    _x = np.linspace(-1.5, 1.5, 301)              # 采样 301 个点（03 课 linspace）

    # 两层齿轮：
    #   内层 g(x) = 3x + 1   —— 线性机器：先乘 3 再加 1
    #   外层 f(u) = u²       —— 平方机器
    #   复合 F(x) = f(g(x)) = (3x+1)²
    _g = 3 * _x + 1                               # 内层输出：中间结果 u（一整条数组）
    _y = _g ** 2                                  # 外层输出：复合函数值

    # 链式法则的解析导数：f'(g(x)) × g'(x) = 2(3x+1) × 3 = 6(3x+1)
    # 注意第一项是在"中间结果 g(x) 处"取值——最常见的错就是忘掉这一点
    _d_chain = 2 * _g * 3

    # 数值导数（13 课对称差分）：[F(x+h) - F(x-h)] / (2h)，当独立裁判
    _h = 1e-5
    _d_num = ((3 * (_x + _h) + 1) ** 2 - (3 * (_x - _h) + 1) ** 2) / (2 * _h)

    _gap = float(np.max(np.abs(_d_chain - _d_num)))   # 两条导数曲线的逐点最大偏差

    # 双子图（11 课的基本功）：上 = 原函数，下 = 两种导数当面对撞
    _fig, (_ax_top, _ax_bot) = plt.subplots(2, 1, figsize=(7, 6.2), sharex=True)

    _ax_top.plot(_x, _y, color="tab:blue", lw=2)
    _ax_top.set_title("上层：复合函数 F(x) = (3x+1)² —— x 先过内层，再进外层")
    _ax_top.set_ylabel("F(x)")
    _ax_top.grid(True, alpha=0.3)

    _ax_bot.plot(_x, _d_chain, color="tab:orange", lw=2.2,
                 label="链式法则：2(3x+1) × 3 = 6(3x+1)")
    _ax_bot.scatter(_x[::15], _d_num[::15], s=16, color="tab:purple", zorder=3,
                    label="对称差分（13 课的数值裁判）")
    _ax_bot.set_title("下层：解析导数 vs 数值导数 —— 散点全部落在曲线上")
    _ax_bot.set_xlabel("x")
    _ax_bot.set_ylabel("dF/dx")
    _ax_bot.legend(loc="upper left")
    _ax_bot.grid(True, alpha=0.3)

    _fig.tight_layout()
    mo.vstack([
        _fig,
        mo.md(
            f"""
            两条导数曲线不是"看起来重合"——逐点最大偏差只有 **{_gap:.1e}**
            （浮点噪声级别；判相等请用 09 课的 `np.allclose`，别用 `==`）。

            彩蛋：对二次函数，对称差分在**代数上是精确的**——把 (F(x+h) - F(x-h))
            展开，h 的平方项恰好相消，剩下的只有浮点舍入。链式法则与数值微分，
            两条完全独立的路，指向同一个答案：这就是"推导对了"的味道。
            """
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 2 · 三层套娃：在 x = 0.5 处逐环节算"转速"

        加一层：内层 h(x) = x² → 中层 g(u) = 3u+1 → 外层 f(v) = sin(v)，
        总函数 **sin(3x² + 1)**。

        在 x = 0.5 这一点，把三个齿轮的"转速"逐个算出来再连乘；
        对称差分在旁边当裁判。**特别注意外层那一级的符号**——
        cos(1.75) 是负数，整条链的总转速跟着变负。
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # 演示 2 · 三层套娃：sin(3x²+1) 在 x=0.5 处逐环节算转速
    _x0 = 0.5                                      # 观测点

    # 三层齿轮，从里到外：
    #   内层 h(x) = x² → 中层 g(u) = 3u+1 → 外层 f(v) = sin(v)
    #   总函数 F(x) = f(g(h(x))) = sin(3x² + 1)
    _h_val = _x0 ** 2                              # 内层输出：0.25
    _g_val = 3 * _h_val + 1                        # 中层输出：1.75
    _f_val = np.sin(_g_val)                        # 外层输出：sin(1.75) ≈ 0.984

    # 三个齿轮各自的"转速"（本级导数——注意各自在"上一级的输出处"取值）：
    _rate_h = 2 * _x0                              # h'(0.5) = 2×0.5 = 1
    _rate_g = 3.0                                  # g'(u) = 3（线性机器，处处是 3）
    _rate_f = np.cos(_g_val)                       # f'(1.75) = cos(1.75) ≈ -0.178（负数！）
    _total = _rate_f * _rate_g * _rate_h           # 链式法则：三环连乘

    # 独立裁判：13 课对称差分，直接对总函数 sin(3x²+1) 整体求
    _eps = 1e-5
    _num = (np.sin(3 * (_x0 + _eps) ** 2 + 1)
            - np.sin(3 * (_x0 - _eps) ** 2 + 1)) / (2 * _eps)

    # 画总函数曲线 + x0 处的切线（13 课的切线手艺：切线斜率 = 导数）
    _x = np.linspace(-1.4, 1.4, 401)
    _y = np.sin(3 * _x ** 2 + 1)
    _tangent = _f_val + _total * (_x - _x0)        # 过 (x0, F(x0))、斜率为总导数的直线

    _fig, _ax = plt.subplots(figsize=(7, 4.6))
    _ax.plot(_x, _y, color="tab:blue", lw=2, label="F(x) = sin(3x² + 1)")
    _ax.plot(_x, _tangent, "--", color="tab:orange", lw=1.8,
             label=f"x = 0.5 处切线（斜率 = {_total:.4f}）")
    _ax.scatter([_x0], [_f_val], s=55, color="tab:red", zorder=3, label="观测点 x = 0.5")
    _ax.axvline(_x0, color="gray", lw=0.8, ls=":")
    _ax.set_xlabel("x")
    _ax.set_ylabel("F(x)")
    _ax.legend(loc="best")
    _ax.grid(True, alpha=0.3)
    _ax.set_title("三层套娃的总函数，与链式法则算出的切线")

    mo.vstack([
        _fig,
        mo.md(
            f"""
            | 齿轮 | 函数 | x = 0.5 时的值 | 本级转速（导数） |
            |---|---|---|---|
            | 内层 | h(x) = x² | h(0.5) = **{_h_val}** | h'(0.5) = 2×0.5 = **{_rate_h:.0f}** |
            | 中层 | g(u) = 3u + 1 | g(0.25) = **{_g_val}** | g'(u) = **{_rate_g:.0f}** |
            | 外层 | f(v) = sin(v) | f(1.75) ≈ **{_f_val:.4f}** | f'(1.75) = cos(1.75) ≈ **{_rate_f:.4f}** |

            三环连乘：{_rate_h:.0f} × {_rate_g:.0f} × ({_rate_f:.4f}) = **{_total:.4f}**；
            对称差分裁判：**{_num:.4f}** —— 吻合到小数点后四位。

            **注意符号**：cos(1.75) ≈ -0.178 是负数——外层齿轮在这一点"倒转"，
            整条链的总转速跟着变负，切线自然朝下。链条会如实传递大小，
            也会如实传递方向。
            """
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 3 · 齿轮传动实验：亲手拧 0.001

        导数说的是"变化率"，那就真的变一次：把 x 从 0.5 拧到 0.501（Δx = 0.001），
        盯着每一层各动了多少。

        - 每层的**实测放大倍数** = Δ输出 / Δ输入——它应该约等于这层的导数
        - 三个实测倍数**连乘**，应该约等于总放大倍数 Δf/Δx

        "约等于"差在哪？差分的 Δ 不够"无穷小"——第 13 课权衡过的老问题，
        这里再亲眼见一次。
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # 演示 3 · 齿轮传动实验：把 x 拧动 0.001，看变化像多米诺一样逐级传导
    _x0, _x1 = 0.5, 0.501                          # 输入只动这么一点
    _dx = _x1 - _x0                                # Δx = 0.001

    # 三层齿轮在两个输入下的输出（还是 h=x² → g=3u+1 → f=sin）：
    _h0, _h1 = _x0 ** 2, _x1 ** 2                  # 内层
    _g0, _g1 = 3 * _h0 + 1, 3 * _h1 + 1            # 中层
    _f0, _f1 = np.sin(_g0), np.sin(_g1)            # 外层

    # 每一级"被带动"的变化量（多米诺一张接一张倒下去）：
    _dh = _h1 - _h0                                # ≈ +0.001001
    _dg = _g1 - _g0                                # ≈ +0.003003
    _df = _f1 - _f0                                # ≈ -0.000540

    # 每级的实测放大倍数 = Δ输出 / Δ输入，与"本级导数"（理论）对照：
    _r_h = _dh / _dx                               # ≈ 1.001    vs  h'(0.5) = 1
    _r_g = _dg / _dh                               # =  3.000   vs  g' = 3
    _r_f = _df / _dg                               # ≈ -0.1797  vs  cos(1.75) ≈ -0.1782
    _r_total = _df / _dx                           # ≈ -0.5397  vs  1×3×cos(1.75) ≈ -0.5347

    # 柱状图：每级"实测放大倍数" vs "本级导数"——两根柱子应该几乎一样高
    _stages = ["x → h\n(平方)", "h → g\n(×3 再 +1)", "g → f\n(正弦)", "总账 x → f\n(三环连乘)"]
    _measured = [_r_h, _r_g, _r_f, _r_total]       # 实测放大倍数
    _theory = [2 * _x0, 3.0, np.cos(_g0), 3.0 * np.cos(_g0)]   # 理论：本级导数（及连乘）

    _fig, _ax = plt.subplots(figsize=(7, 4.6))
    _pos = np.arange(4)                            # 四组柱子的位置
    _ax.bar(_pos - 0.18, _measured, width=0.36, color="tab:blue",
            label="实测放大倍数（Δ比）")
    _ax.bar(_pos + 0.18, _theory, width=0.36, color="tab:orange", alpha=0.75,
            label="理论（本级导数）")
    for _i in range(4):                            # 每根柱子头顶（或脚下）标数值
        _ax.text(_pos[_i] - 0.18, _measured[_i] + (0.05 if _measured[_i] >= 0 else -0.10),
                 f"{_measured[_i]:.3f}", ha="center", fontsize=9, color="tab:blue")
        _ax.text(_pos[_i] + 0.18, _theory[_i] + (0.05 if _theory[_i] >= 0 else -0.10),
                 f"{_theory[_i]:.3f}", ha="center", fontsize=9, color="tab:orange")
    _ax.axhline(0, color="gray", lw=0.8)           # 零线：负放大 = 方向被反转
    _ax.set_xticks(_pos, _stages)
    _ax.set_ylabel("放大倍数")
    _ax.set_ylim(-0.8, 3.7)
    _ax.legend(loc="upper right")
    _ax.grid(True, axis="y", alpha=0.3)
    _ax.set_title("齿轮传动实验：每级放大倍数 ≈ 本级导数")

    mo.vstack([
        _fig,
        mo.md(
            f"""
            | 传动环节 | 输入端变化 | 输出端变化 | 实测放大倍数 | 理论（本级导数） |
            |---|---|---|---|---|
            | x → h（平方） | {_dx:.3f} | {_dh:+.6f} | **{_r_h:.4f}** | 1 |
            | h → g（×3 再 +1） | {_dh:.6f} | {_dg:+.6f} | **{_r_g:.4f}** | 3 |
            | g → f（正弦） | {_dg:.6f} | {_df:+.6f} | **{_r_f:.4f}** | cos(1.75) ≈ -0.1782 |
            | **总账：x → f** | {_dx:.3f} | {_df:+.6f} | **{_r_total:.4f}** | 1×3×cos(1.75) ≈ -0.5347 |

            把"乘法传递"钉死的两句话：

            1. **连乘自洽**：三个实测倍数连乘 {_r_h:.4f} × {_r_g:.4f} × ({_r_f:.4f})
               = **{_r_h * _r_g * _r_f:.4f}**，恰好等于总实测放大倍数——中间量 Δh
               在连乘里分子分母正好约掉。这就是"**变化率是乘法传递的**"的数字版。
            2. **误差来自 Δ 不够小**：正弦那级实测 {_r_f:.4f} vs 理论 -0.1782，
               差在第三位——因为 Δg = 0.003 不是无穷小；把 Δx 缩到 1e-5
               （演示 2 的做法），误差就缩进小数点后四位。
            """
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 4 · 画出计算图：第 22 课的预告海报

        取一个两层网络片段：x →(×w1)→ u1 →(×w2)→ u2 →(平方)→ L，
        参数取 x = 2、w1 = 3、w2 = 2。

        - **蓝色向右**：前向，把每个节点的值算出来（2 → 6 → 12 → 144）
        - **橙色向左**：反向，每条边把"上游传回来的变化率"乘上"本级导数"，
          再往左传（24 → 48 → 144）

        第 22 课手写反向传播时，跑的就是这张图——今天先把海报画好贴在墙上。
        """
    )
    return


@app.cell
def _(mo, plt):
    # 演示 4 · 画出计算图：前向算值（蓝 →），反向传梯度（橙 ←）
    from matplotlib.patches import Circle as _Circle            # 圆圈 = 节点（一个数）
    from matplotlib.patches import FancyArrowPatch as _FArrow   # 带箭头的连线 = 运算

    # 两层网络片段：x →(×w1)→ u1 →(×w2)→ u2 →(平方)→ L
    _x, _w1, _w2 = 2.0, 3.0, 2.0

    # ---- 第一遍：前向，从左到右把每个节点的值算出来 ----
    _u1 = _w1 * _x                                 # 6
    _u2 = _w2 * _u1                                # 12
    _L = _u2 ** 2                                  # 144

    # ---- 第二遍：反向，从右到左把梯度一环一环传回来 ----
    _g_L = 1.0                                     # 起点：dL/dL = 1
    _g_u2 = _g_L * 2 * _u2                         # dL/du2 = 24（本级导数 2×u2）
    _g_u1 = _g_u2 * _w2                            # dL/du1 = 48（上游 24 × 本级 w2）
    _g_x = _g_u1 * _w1                             # dL/dx  = 144（上游 48 × 本级 w1）
    _g_w1 = _g_u1 * _x                             # dL/dw1 = 96（参数的梯度——训练要更新它！）
    _g_w2 = _g_u2 * _u1                            # dL/dw2 = 144

    # 数值抽查（13 课对称差分）：三个梯度各验一遍，谁也不许裸奔
    _eps = 1e-5
    _num_x = ((_w2 * _w1 * (_x + _eps)) ** 2 - (_w2 * _w1 * (_x - _eps)) ** 2) / (2 * _eps)
    _num_w1 = ((_w2 * (_w1 + _eps) * _x) ** 2 - (_w2 * (_w1 - _eps) * _x) ** 2) / (2 * _eps)
    _num_w2 = (((_w2 + _eps) * _w1 * _x) ** 2 - ((_w2 - _eps) * _w1 * _x) ** 2) / (2 * _eps)

    # ---- 画图 ----
    _fig, _ax = plt.subplots(figsize=(8.6, 4.4))
    _ax.set_xlim(0, 10.8)
    _ax.set_ylim(1.7, 6.5)
    _ax.axis("off")                                # 计算图不需要坐标轴

    _y_node = 4.7                                  # 节点所在的水平线
    _nodes = [("x", 1.1), ("u1", 4.0), ("u2", 6.9), ("L", 9.6)]   # 名字与横坐标
    _vals = [2, 6, 12, 144]                        # 前向算出的值

    for (_name, _xc), _val in zip(_nodes, _vals):
        _ax.add_patch(_Circle((_xc, _y_node), 0.62, facecolor="#e8f2fb",
                              edgecolor="#1f77b4", lw=1.8))       # 圆圈 = 一个数
        _ax.text(_xc, _y_node, _name, ha="center", va="center", fontsize=15)
        _ax.text(_xc, _y_node - 0.95, f"值 = {_val}", ha="center",
                 va="center", fontsize=11, color="#1f77b4")

    # 前向箭头（蓝，向右）+ 边上的运算标签
    _fwd = [("× w1  (w1 = 3)", 1.1, 4.0),
            ("× w2  (w2 = 2)", 4.0, 6.9),
            ("平方", 6.9, 9.6)]
    for _label, _xa, _xb in _fwd:
        _ax.add_patch(_FArrow((_xa + 0.68, _y_node), (_xb - 0.68, _y_node),
                              arrowstyle="-|>", mutation_scale=16,
                              color="#1f77b4", lw=2))
        _ax.text((_xa + _xb) / 2, _y_node + 0.33, _label, ha="center",
                 fontsize=11.5, color="#1f77b4")

    # 反向箭头（橙，向左）+ 每条边上回传的梯度
    _bwd = [("dL/du2 = 2 × 12 = 24", 9.6, 6.9),
            ("dL/du1 = 24 × 2 = 48", 6.9, 4.0),
            ("dL/dx = 48 × 3 = 144", 4.0, 1.1)]
    for _label, _xa, _xb in _bwd:                  # 注意方向：起点在右，终点在左
        _ax.add_patch(_FArrow((_xa, 2.85), (_xb, 2.85),
                              arrowstyle="-|>", mutation_scale=16,
                              color="#d95f02", lw=2))
        _ax.text((_xa + _xb) / 2, 2.42, _label, ha="center",
                 fontsize=11, color="#d95f02")

    # 图例说明（手工摆两行字，代替 legend）
    _ax.text(0.12, 6.12, "前向 →：从左到右，算出每个节点的值",
             fontsize=11, color="#1f77b4")
    _ax.text(0.12, 5.70, "反向 ←：从右到左，传回每条边的变化率",
             fontsize=11, color="#d95f02")
    _ax.text(9.6, 5.62, "回传起点：dL/dL = 1", ha="center",
             fontsize=9.5, color="gray")

    _ax.set_title("计算图：两层网络片段 —— 第 22 课反向传播的预告海报", fontsize=13)

    mo.vstack([
        _fig,
        mo.md(
            f"""
            数值抽查（13 课对称差分，各把输入拧动 1e-5）：

            | 梯度 | 链式法则（图上橙字连乘） | 数值抽查 |
            |---|---|---|
            | dL/dx | 24 × 2 × 3 = **{_g_x:.0f}** | **{_num_x:.4f}** |
            | dL/dw1 | 48 × x = **{_g_w1:.0f}** | **{_num_w1:.4f}** |
            | dL/dw2 | 24 × u1 = **{_g_w2:.0f}** | **{_num_w2:.4f}** |

            前向 2 → 6 → 12 → 144，反向 24 → 48 → 144——两个方向在同一条链上跑，
            算出来的数被对称差分逐一核对。
            """
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        读图三问（答案都藏在图里）：

        1. **反向第一跳为什么是 24？**——L = u2²，本级导数 2 × u2 = 2 × 12 = 24
        2. **传到 u1 为什么变成 48？**——上游 24 × 本级导数 w2 = 24 × 2 = 48
        3. **传到 x 为什么是 144？**——上游 48 × 本级导数 w1 = 48 × 3 = 144

        每一步都是同一个动作：**收到的梯度 × 本级导数 = 往左传的梯度**。
        这条铁律你今天已经手算三遍了（演示 2、演示 3、这张图）——
        第 22 课的反向传播，无非是把它写成循环。

        另外，dL/dw1 = 96、dL/dw2 = 144 这两个梯度特别金贵：x 是数据、
        u1/u2 是中间结果，都不用学；**真正被梯度下降更新的是参数 w**——
        训练的目标从来就是它们。
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 🎛️ 交互实验

        **齿轮调速台**——复合函数 f(a·x + b)，你来当齿轮厂厂长：调内层齿轮的
        斜率 a 和截距 b，还能整体换掉外层齿轮（平方 / 正弦）。

        观测点固定在 x0 = 0.8。画面上是复合曲线 + 该点切线
        （斜率由链式法则**实时**算出：f'(a·x0 + b) × a）；数值表里是
        "内层变化率 × 外层变化率 = 总变化率"，旁边站着对称差分的抽查值。

        两个必做实验：

        1. 把 a 拉到 0：内层齿轮卡死，曲线躺平，总转速归零——
           链式法则给出 0 × f'(u0) = 0，自动正确，不用你特判
        2. 换正弦齿轮，把 b 拧到让 u0 落进 cos 为负的区间：
           总转速变负、切线朝下——外层"倒转"会顺着链条传染给总转速

        （画框高度固定，齿轮拧太猛时曲线会冲出画框——数学没错，只是窗户装不下。）
        """
    )
    return


@app.cell
def _(mo):
    # 交互控制：内层两个旋钮 + 外层齿轮切换（debounce=True：松手才重算，画面不闪烁）
    gear_a_slider = mo.ui.slider(-3, 3, value=1.5, step=0.1, debounce=True,
                                 label="🎛️ 内层斜率 a（g(x) = a·x + b 的 a，-3 ~ 3）")
    gear_b_slider = mo.ui.slider(-2, 2, value=0.5, step=0.1, debounce=True,
                                 label="🎛️ 内层截距 b（-2 ~ 2）")
    outer_dropdown = mo.ui.dropdown(
        options=["平方 f(u) = u²", "正弦 f(u) = sin(u)"],
        value="平方 f(u) = u²",
        label="🔀 换一个外层齿轮",
    )
    mo.vstack([gear_a_slider, gear_b_slider, outer_dropdown])
    return (gear_a_slider, gear_b_slider, outer_dropdown)


@app.cell
def _(gear_a_slider, gear_b_slider, mo, np, outer_dropdown, plt):
    # ===== 交互实验 · 齿轮调速台（响应单元）=====
    _a = gear_a_slider.value                       # 内层斜率 a
    _b = gear_b_slider.value                       # 内层截距 b
    _outer = outer_dropdown.value or "平方 f(u) = u²"   # 外层齿轮（防一手空值）
    _is_square = _outer.startswith("平方")

    # 外层齿轮换件：函数和它的导数必须成对换——链式法则的第一环就是它
    def _f(u):
        return u ** 2 if _is_square else np.sin(u)

    def _df(u):
        return 2 * u if _is_square else np.cos(u)

    _x0 = 0.8                                      # 固定观测点
    _u0 = _a * _x0 + _b                            # 中间结果：内层机器的输出
    _y0 = _f(_u0)                                  # 复合函数在观测点的值

    # 链式法则：总转速 = 外层转速（在 u0 处取值！） × 内层转速
    _rate_inner = _a                               # g'(x) = a
    _rate_outer = _df(_u0)                         # f'(u0)
    _rate_total = _rate_outer * _rate_inner        # 总变化率 = 切线斜率

    # 独立裁判：13 课对称差分，对复合函数整体求
    _eps = 1e-5
    _num = (_f(_a * (_x0 + _eps) + _b) - _f(_a * (_x0 - _eps) + _b)) / (2 * _eps)

    # 画复合曲线 + 观测点切线（13 课的切线手艺：斜率 = 链式法则的总转速）
    _x = np.linspace(-2, 2, 401)
    _y = _f(_a * _x + _b)
    _tan = _y0 + _rate_total * (_x - _x0)
    _seg = (_x >= _x0 - 0.9) & (_x <= _x0 + 0.9)   # 切线只画观测点附近一段

    _fig, _ax = plt.subplots(figsize=(7, 4.6))
    _ax.plot(_x, _y, color="tab:blue", lw=2, label="复合函数 F(x) = f(a·x + b)")
    _ax.plot(_x[_seg], _tan[_seg], "--", color="tab:orange", lw=1.8,
             label=f"x0 = 0.8 处切线（斜率 = {_rate_total:.2f}）")
    _ax.scatter([_x0], [_y0], s=55, color="tab:red", zorder=3, label="观测点 x0 = 0.8")
    _ax.axvline(_x0, color="gray", lw=0.8, ls=":")
    if _is_square:                                 # 两种外层齿轮配不同高度的画框
        _ax.set_ylim(-4, 10)
    else:
        _ax.set_ylim(-2.5, 2.5)
    _ax.set_xlabel("x")
    _ax.set_ylabel("F(x)")
    _ax.legend(loc="best")
    _ax.grid(True, alpha=0.3)
    _ax.set_title(f"齿轮调速台：外层 = {'u²' if _is_square else 'sin(u)'}，"
                  f"a = {_a:.1f}，b = {_b:.1f}")

    # 外层转速的"来历"写进表格：平方齿轮是 2×u0，正弦齿轮是 cos(u0)
    _fname = f"2 × {_u0:.2f}" if _is_square else f"cos({_u0:.2f})"
    _verdict = (
        "🔧 内层齿轮卡死（a = 0）：u 恒等于 b，曲线躺平，"
        "总转速 = 0 × f'(u0) = 0——链式法则自动给出正确答案，不用你特判。"
        if _a == 0
        else f"⚙️ 内层变化率 {_rate_inner:.2f} × 外层变化率 {_rate_outer:.2f}"
             f" = 总变化率 {_rate_total:.2f}；数值抽查 {_num:.2f}——分毫不差。"
    )

    mo.vstack([
        _fig,
        mo.md(
            f"""
            | 量 | 公式 | 值 |
            |---|---|---|
            | 中间结果 | u0 = a·x0 + b | **{_u0:.2f}** |
            | 内层变化率 | g'(x) = a | **{_rate_inner:.2f}** |
            | 外层变化率 | f'(u0) = {_fname} | **{_rate_outer:.2f}** |
            | **总变化率（链式法则）** | **a × f'(u0)** | **{_rate_total:.2f}** |
            | 数值抽查 | 对称差分（13 课） | **{_num:.2f}** |
            """
        ),
        mo.callout(mo.md(_verdict), kind="info"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✏️ 动手练习（每课必做，本课 3 题）": mo.md(
                """
                **练习 1 · 手算两层链**：f(x) = (3x+1)² 在 x = 1 处的导数。
                三步：先算 g(1)，再算外层导数在 g(1) 处的值，最后乘上 g'(1)。
                把结果填进 `deriv_ex1`。

                **练习 2 · 手算三层链 + 数值验证**：f(x) = sin(x²) 在 x = 1 处的导数。
                解析答案填 `deriv_ex2`；再用 13 课的对称差分对 sin(x²) 整体求一个
                数值答案填 `num_ex2`——让两个自己互相印证。

                **练习 3 · 计算图填表**：链条 x →(×3)→ u →(+1)→ v →(平方)→ L，
                输入 x = 2。前向把每个节点的值填出来；反向从 dL/dv = 2v 起步，
                一环一环往左乘。最终的 dL/dx 填进 `grad_x_ex3`。

                规则不变：先把 `None` 换成你的答案再运行；写完再看参考答案。
                """
            )
        }
    )
    return


@app.cell
def _(mo, np):
    # ===== 练习 1 · 手算两层链 =====
    # f(x) = (3x + 1)²，在 x = 1 处求导。三步走：
    #   ① 内层：g(1) = 3×1 + 1 = 4
    #   ② 外层导数在中间结果处取值：f'(u) = 2u → f'(g(1)) = 2 × 4 = 8
    #   ③ 乘上内层导数：g'(x) = 3 → 链式法则 = 8 × 3 = ?
    deriv_ex1 = None       # ← 把手算结果填进来（一个数）

    # 机器裁判（13 课对称差分）——填完答案自动核对
    _eps = 1e-5
    _num = ((3 * (1 + _eps) + 1) ** 2 - (3 * (1 - _eps) + 1) ** 2) / (2 * _eps)

    if deriv_ex1 is None:
        _msg = "⏳ 先在纸上走完三步：g(1) → f'(g(1)) → 乘 g'(1)，把结果填进 `deriv_ex1` 再运行。"
    elif np.allclose(deriv_ex1, _num, atol=1e-3):
        _msg = f"✅ 正确！你的答案 {deriv_ex1}，机器裁判 {_num:.4f}——两层齿轮严丝合缝。"
    else:
        _msg = (f"🤔 再查一遍：你的答案 {deriv_ex1}，机器裁判 {_num:.4f}。"
                "两个常见的坑：外层导数没在 g(1) 处取值？忘了乘 g'(x) = 3？")

    mo.md(_msg)
    return (deriv_ex1,)


@app.cell
def _(mo, np):
    # ===== 练习 2 · 手算三层链 + 数值验证 =====
    # f(x) = sin(x²)，在 x = 1 处求导：
    #   内层 h(x) = x²：    h(1) = 1，h'(1) = 2×1 = 2
    #   外层 f(v) = sin(v)：f'(v) = cos(v) → f'(h(1)) = cos(1)   （1 是弧度！）
    #   链式法则：cos(1) × 2 = ?
    deriv_ex2 = None   # ← 解析答案（可以直接写 np.cos(1) * 2）

    # 数值验证：用对称差分对 sin(x²) 整体求导（13 课的手艺）
    #   提示：[sin((1+h)²) - sin((1-h)²)] / (2h)，h 取 1e-5
    num_ex2 = None     # ← 把差分结果填进来（一个数）

    if deriv_ex2 is None or num_ex2 is None:
        _msg = ("⏳ 两个空都要填：`deriv_ex2` = 链式法则手算；"
                "`num_ex2` = 对称差分（提示：np.sin((1 + 1e-5) ** 2)）。")
    elif np.allclose(deriv_ex2, num_ex2, atol=1e-4):
        _msg = (f"✅ 手算 {float(deriv_ex2):.6f} ≈ 数值 {float(num_ex2):.6f}——"
                "解析与数值互相印证（判相等用 09 课的 np.allclose，容差 atol=1e-4）。")
    else:
        _msg = (f"🤔 两个值对不上：手算 {deriv_ex2}，数值 {num_ex2}。"
                "要么链上漏乘/错乘了一环，要么差分式子的括号写错了——分开检查。")

    mo.md(_msg)
    return (deriv_ex2, num_ex2)


@app.cell
def _(mo, np):
    # ===== 练习 3 · 计算图填表 =====
    # 链条：x →(×3)→ u →(+1)→ v →(平方)→ L，输入 x = 2
    #
    # 第一遍·前向（从左到右算值）——代码帮你算好，自己手算核对：
    _x = 2.0
    _u = _x * 3        # u = 3x    → ?
    _v = _u + 1        # v = u + 1 → ?
    _L = _v ** 2       # L = v²    → ?
    #
    # 第二遍·反向（从右到左传梯度）——你来填：
    #   dL/dv = 2v（v 的值你已经有了）
    #   dL/du = dL/dv × dv/du = ? × 1
    #   dL/dx = dL/du × du/dx = ? × 3
    grad_x_ex3 = None   # ← 把最终的 dL/dx 填进来

    # 机器裁判：把 x 拧动一点点，看 L 动多少（13 课对称差分）
    _eps = 1e-5
    _num = ((3 * (_x + _eps) + 1) ** 2 - (3 * (_x - _eps) + 1) ** 2) / (2 * _eps)

    if grad_x_ex3 is None:
        _msg = (f"⏳ 前向：x = {_x:.0f} → u = {_u:.0f} → v = {_v:.0f} → L = {_L:.0f}。"
                "反向从 dL/dv = 2v 起步，一环一环往左乘，把最终结果填进 `grad_x_ex3`。")
    elif np.allclose(grad_x_ex3, _num, atol=1e-3):
        _msg = f"✅ 正确！dL/dx = {grad_x_ex3}，机器裁判 {_num:.4f}——反向三步连乘，一环没漏。"
    else:
        _msg = (f"🤔 再想想：你的答案 {grad_x_ex3}，机器裁判 {_num:.4f}。"
                "反向每一行都是「上游梯度 × 本级导数」，别跳步。")

    mo.md(_msg)
    return (grad_x_ex3,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 1 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                deriv_ex1 = 2 * (3 * 1 + 1) * 3    # = 24
                ```

                手算三步：g(1) = 3×1+1 = **4**；f'(u) = 2u → f'(g(1)) = 2×4 = **8**；
                g'(x) = **3**；链式法则 8 × 3 = **24**。

                交叉验证（展开法）：(3x+1)² = 9x² + 6x + 1，导数 18x + 6，
                在 x=1 处 = **24**——殊途同归。但别高兴太早：展开法只对这种
                恰好能展开的复合有效，sin(3x²+1) 这类就展不开了——
                **链式法则才是通用引擎**。
                """
            ),
            "✅ 练习 2 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                deriv_ex2 = np.cos(1) * 2                                            # ≈ 1.0806
                num_ex2 = (np.sin((1 + 1e-5) ** 2) - np.sin((1 - 1e-5) ** 2)) / 2e-5
                ```

                手算：h(1) = 1、h'(1) = **2**；f'(v) = cos(v) → f'(h(1)) = **cos(1)**；
                连乘 cos(1) × 2 ≈ **1.0806**（注意 cos 的自变量是弧度，1 弧度 ≈ 57.3°）。

                数值 ≈ **1.0806**，与手算在小数点后四位内一致——两个独立的方法
                互相作证。差分里 h 取 1e-5 是 13 课的权衡：再大误差肉眼可见，
                再小浮点噪声就会上位。
                """
            ),
            "✅ 练习 3 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                grad_x_ex3 = 42
                ```

                | 节点 | 前向值（左 → 右） | 反向梯度（右 → 左：上游 × 本级导数） |
                |---|---|---|
                | L | 49 | 1（起点：损失对自己的导数） |
                | v | 7 | 1 × 2v = 2 × 7 = **14** |
                | u | 6 | 14 × 1 = **14** |
                | x | 2 | 14 × 3 = **42** |

                彩蛋：整条链展开就是 L = (3x+1)²——**练习 1 的同款函数**！
                练习 1 在 x=1 处（导数 24），这里在 x=2 处（18×2+6 = 42）。
                同一个函数、两种求导走法，答案必须一致——这就是"链没断"的自查法。
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
                1. **复合函数 = 套娃/齿轮**：x 先过内层机器 g，输出 u 再进外层机器 f；
                   深度学习里数据就这样一层层穿过网络，直到最外层的损失
                2. **链式法则**：d/dx f(g(x)) = f'(g(x)) × g'(x)——外层在**中间结果处**的
                   变化率 × 内层的变化率；N 层复合 = N 个导数**连乘**
                3. **变化率是乘法传递的**（演示 3 的数字表格钉死过）：每级实测放大倍数
                   Δ输出/Δ输入 ≈ 本级导数，三级连乘 = 总放大倍数，中间量恰好约掉
                4. **深度学习的损失是 N 层复合**：对第 1 层参数求导要穿过全部 N 层；
                   链一断（哪层忘了乘本级导数），梯度就全错——
                   **反向传播 = 系统化、不重不漏地执行链式法则**
                5. **计算图 = 链式法则的可视化**：节点是数、箭头是运算；前向从左到右
                   算值、反向从右到左传梯度（上游 × 本级导数）——第 22 课跑在这张图上
                6. **数学三件套集齐**：几何给画面（11-12 课）、微积分给方向（13 课）、
                   链式法则给传动装置（本课）——下一部分马上用它组装第一个训练循环
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
                ## 🎓 第 2 部分毕业清点

                **从零到 Transformer · 手写深度学习** ｜ 第 2 部分 · 深度学习的数学（4/4）——**收官！**

                四课之前，矩阵还只是一张数字表，导数还只是课本里的极限符号。现在，
                **数学三件套集齐**：

                - **几何**给了画面（向量是箭头、矩阵是变形机器）
                - **微积分**给了方向（导数是斜率、梯度是最陡的上坡）
                - **链式法则**给了传动装置（变化率逐级乘法传递）

                画面、方向、传动都到手，一台"会学习的机器"就差组装了。
                毕业清点，四行对账：
                """
            ),
            mo.md(
                """
                | 课 | 里程碑：你现在能做到什么 |
                |---|---|
                | **11** 向量的几何直觉 | 把向量看成箭头：分量是坐标、范数是长度，用点积与余弦相似度比较"方向像不像" |
                | **12** 矩阵即变换 | 把矩阵看成操作手册：X @ W 是把一批点送进同一台变形机器，(n,k)@(k,m)→(n,m) 脱口而出 |
                | **13** 导数与梯度 | 用对称差分算导数、画切线；知道梯度指最陡上坡、训练要沿反方向下山 |
                | **14** 链式法则 | 对函数套函数逐级连乘求导；画出计算图，前向算值、反向传梯度 |
                """
            ),
            mo.callout(
                mo.md(
                    """
                    路线图给第 2 部分的承诺是"**看懂公式的底气**"——从下一课起，
                    再看到 `dL/dw`、`反向传播`、`梯度`这些词，你都能在心里画出
                    齿轮组和计算图。这就是底气。🎓
                    """
                ),
                kind="success",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ---

        **下一课预告** — `15_线性回归`：

        恭喜——**第 2 部分 · 深度学习的数学，四课功成！** 🎓 数学热身到此为止，
        第 3 部分"经典机器学习热身"开箱：真刀真枪的建模，从一条直线开始。

        下一课，你将用集齐的三件套组装全课程**第一个完整的训练循环**：

        - **前向**：算出预测和损失（矩阵乘法，12 课）
        - **反向**：用链式法则算出损失的梯度（本课）
        - **更新**：沿梯度反方向挪一点参数（13 课指的方向）

        三步一个循环，循环转起来，你就将亲眼看着损失一路下降、
        直线自己学会贴上数据——"**学习**"这个词，从比喻变成你手写的十几行 NumPy。

        你要亲手养第一个会学习的模型了。

        🚀 齿轮已咬合，去把第一台学习机器造出来！
        """
    )
    return


if __name__ == "__main__":
    app.run()
