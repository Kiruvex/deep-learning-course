import marimo

__generated_with = "0.25.2"
app = marimo.App(app_title="22 · 反向传播上", width="medium")


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
        # 22 · 反向传播（上）：从计算图到第一条梯度

        **从零到 Transformer · 手写深度学习** ｜ 第 4 部分 · 神经网络（5/8）
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📖 本课目标（先读我）": mo.md(
                """
                **这一课只解决一个问题：梯度怎么穿过一个真正的网络？**

                这是全课程最核心的一课——**反向传播**（backpropagation），
                深度学习的发动机。好消息从第一行就开始：**你早就见过它的海报**。
                第 14 课演示 4 画过那张计算图（x →×w₁→ u₁ →×w₂→ u₂ →平方→ L），
                说"第 22 课跑的就是这张图"——今天，海报变正片。

                剧情分四幕：先把海报兑成现金（前向算值、反向传率，
                每个数字对一遍账）；再提炼**两条铁律**（加法分发、乘法交换）
                ——它们是反向传播全部的复杂度，剩下的只是重复；
                然后上一个**真正的微型网络**（tanh 隐层 + sigmoid 输出 +
                交叉熵，4 个参数 7 个节点），一个节点一个节点地手推反向，
                每条梯度过对称差分的验钞机；最后按下训练键——200 步之后，
                **参数第一次"自己学会了"一件事**。

                学完你将能：

                - 在计算图上**双向跑通**：前向把每个节点的值算出来，
                  反向把 dL/d· 一个节点一个节点传回去
                - 背诵并运用两条铁律：**穿过加法，梯度原样分发；
                  穿过乘法，梯度乘上"另一个因子"**
                - 对 4 参数微型网络**手推全部梯度**（包括最长的四连环
                  dL/dw₁），并说清每一步用的是哪条铁律
                - 用**对称差分给反向传播对账**（13 课验钞机，误差 < 10⁻⁸），
                  然后亲手训练它：损失 0.478 → 0.005

                前置：第 14 课的链式法则与计算图（海报）、第 13 课的
                对称差分、第 19 课的激活函数导数（tanh'、σ'）、
                第 21 课的 p − y（输出层的最后一环）、第 15 课的四步曲
                （本课把第 ③ 步"求梯度"真正造出来）。

                一句定心丸：反向传播**不是新数学**——它就是链式法则 +
                一张图 + 两句口诀。难的是把它读熟，而不是把它想懂。
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "🗺️ 概念 1/4 · 海报回收：前向算值，反向传率": mo.md(
                """
                把 14 课的海报再贴一遍——两层网络片段：

                ```
                x →(×w₁)→ u₁ →(×w₂)→ u₂ →(平方)→ L
                ```

                参数取 x = 2、w₁ = 3、w₂ = 2。**前向**（蓝色，向右）：

                ```
                u₁ = w₁·x = 6        u₂ = w₂·u₁ = 12       L = u₂² = 144
                ```

                **反向**（橙色，向左）从最右端起跳——起跳线永远是
                **dL/dL = 1**（损失对自己的变化率当然是 1）。然后每过
                一条边，做同一件事：

                > **传给左边的梯度 = 上游传来的梯度 × 这条边的本级导数**

                ```
                dL/du₂ = 1 × (2u₂) = 24        （平方的导数 2u，在 u₂=12 处取值）
                dL/du₁ = 24 × w₂ = 48           （×w₂ 的本级导数就是 w₂）
                dL/dx  = 48 × w₁ = 144
                ```

                参数也是节点，也有梯度（乘法铁律：想知道对 w₂ 的导数，
                就把另一个因子 u₁ 乘上去）：

                ```
                dL/dw₂ = 24 × u₁ = 144        dL/dw₁ = 48 × x = 96
                ```

                14 课算过这两个数；今天除了重算，还要加一道新工序：
                **对称差分对账**——手推的和数值的，一个数都不许有分歧。
                （顺带一个小细节：dL/dx 和 dL/dw₂ 恰好都是 144——
                平方外层 × 两条不同的链，殊途同归，对账时会再遇上。）

                前后两遍扫过同一张图，就是反向传播的**全部动作**。
                前向攒信息（每个节点的值），反向花信息（本级导数要在
                **节点当前值**处取值——14 课"外层导数在中间结果处取值"，
                此话此处兑现）。
                """
            ),
            "🔀 概念 2/4 · 两条铁律：加法分发，乘法交换": mo.md(
                """
                网络再深，运算拆到原子也只有两种：**加法和乘法**。
                各配一条铁律，反向传播就通关了。

                **铁律一 · 加法 = 原样分发。**
                节点 s = a + b：a 和 b 各加多少，s 就变多少——

                ```
                ∂s/∂a = 1,  ∂s/∂b = 1
                ```

                上游传来 dL/ds，往两条边各发一份**原样**的 dL/ds。
                偏置 b 的梯度永远是 δ 本身（dL/db = δ·1），
                就是这条铁律的直通车。

                **铁律二 · 乘法 = 交换因子。**
                节点 m = a × b：固定 b 看 a，导数是 b；固定 a 看 b，导数是 a——

                ```
                ∂m/∂a = b,  ∂m/∂b = a
                ```

                上游传来 dL/dm，往 a 的边送 dL/dm × **b**，
                往 b 的边送 dL/dm × **a**——**想知道自己对谁的导数，
                就把另一个因子乘上去**。权重的梯度 = δ × 输入
                （dL/dw = δ · x），输入的梯度 = δ × 权重（dL/dx = δ · w），
                全是这一条。

                tanh、σ、平方这些"一元运算"更简单：本级导数查表
                （19 课背过的 f'），乘上传来的梯度就行。

                于是**每个节点只需要两样东西**：自己输出的**值**（前向攒下的），
                和自己的**本级导数公式**（查表）。它不需要知道网络长什么样、
                损失在多远——**局部性**。这就是反向传播能扛住百万参数的原因：
                复杂度不随深度爆炸，每层只做"收梯度、查导数、乘、往左传"。
                现代框架（PyTorch 的 autograd，46 课见）自动干的就是这件事。
                """
            ),
            "🧩 概念 3/4 · 微型网络：7 个节点、4 个参数，全部手推": mo.md(
                """
                真枪实弹。一个**能打的两层网络**（单样本、全标量）：

                ```
                z₁ = w₁·x + b₁        第一层打分（乘法 + 加法）
                h  = tanh(z₁)          隐层弯折（19 课）
                z₂ = w₂·h + b₂        第二层打分
                p  = σ(z₂)             输出概率（16 课）
                L  = −log(p)           交叉熵（y = 1 时；21 课）
                ```

                参数 4 个：w₁、b₁、w₂、b₂。输入 x = 1.5，真值 y = 1，
                初始参数 w₁ = 0.8、b₁ = −0.2、w₂ = 1.3、b₂ = −0.5。

                **前向**（演示 2 逐步算）：z₁ = 1.0 → h = tanh(1) = 0.762 →
                z₂ = 0.490 → p = σ(0.490) = 0.620 → L = −log(0.620) = **0.478**。
                它现在押"62% 是正类"——方向对，不够自信。

                **反向**（演示 3 逐节点填表），从右往左：

                | 步 | 公式 | 铁律 | 数值 |
                |---|---|---|---|
                | δ₂ = dL/dz₂ | (p − y)（21 课！） | σ 配 CE 的合体导数 | **−0.380** |
                | dL/dw₂ | δ₂ · h | 乘法交换 | **−0.289** |
                | dL/db₂ | δ₂ · 1 | 加法分发 | **−0.380** |
                | dL/dh | δ₂ · w₂ | 乘法交换 | **−0.494** |
                | δ₁ = dL/dz₁ | dL/dh · (1 − h²) | tanh' = 1 − h²（19 课） | **−0.207** |
                | dL/dw₁ | δ₁ · x | 乘法交换 | **−0.311** |
                | dL/db₁ | δ₁ · 1 | 加法分发 | **−0.207** |

                最长的一条链是 **dL/dw₁ = δ₂ × w₂ × (1−h²) × x**——四连环，
                每一环都是一条铁律。22 课下篇（23 课）把它推广成矩阵版，
                这张标量表就是那时的"对账底单"。
                """
            ),
            "✅ 概念 4/4 · 对账、训练，与「第一次自己学会」": mo.md(
                """
                手推的梯度对不对？13 课的**验钞机**伺候：对每个参数，
                把它拧动 ±10⁻⁷ 算对称差分，和反向传播的答案对表。
                四个参数全部对上（误差 ~10⁻⁸）——**梯度，验明正身**。

                然后是历史性的一刻：把四步曲（15 课）挂上自动挡——

                ```
                重复 200 次：
                    ① 前向：z₁ → h → z₂ → p → L（顺手存中间量）
                    ③ 反向：δ₂ → 四个梯度
                    ④ 更新：每个参数 -= lr × 自己的梯度
                ```

                损失 0.478 → **0.005**，押注 0.62 → **0.995**。
                没有手拍权重（20 课）、没有最小二乘（19 课）——
                **参数沿着梯度自己走到了好位置**。这就是"训练"二字的
                完整含义，从这一刻起你写的每一个网络都是这么学的。

                两个先埋伏笔的观察（交互实验里亲手摸到）：

                - **梯度自己会刹车**：p 越接近 1，δ₂ = p − y 越接近 0，
                  步子自动变小——"错多少、学多少"（21 课的比例阀）
                - **换损失只换最后一环**：把 CE 换成 MSE，反向传播其余
                  六步一字不改，只改 δ₂ 的公式——练习 4 亲自换一次，
                  顺便看 16 课"MSE 迈不动腿"的病在此发作
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

        五组演示，把发动机点着：

        - **演示 1** · 海报兑现：14 课那张图，跑起来 + 全梯度对账
        - **演示 2** · 微型网络前向：5 步算到损失，图上每个节点标值
        - **演示 3** · 反向逐节点：两条铁律连用七次，橙字一路向左
        - **演示 4** · 验钞机：4 条梯度 vs 对称差分，误差 10⁻⁸
        - **演示 5** · 点火：200 步训练，损失 0.478 → 0.005
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 1 · 海报兑现：14 课的图跑起来

        还是那张图、还是那组数（x = 2、w₁ = 3、w₂ = 2）：前向
        2 → 6 → 12 → 144，反向 24 → 48 → 144。这次的新工序是
        **验钞**：三个手推梯度，逐个用对称差分（h = 10⁻⁵）重算——

        | 梯度 | 手推（链式法则） | 对称差分 | 判决 |
        |---|---|---|---|
        | dL/dx | 24 × 2 × 3 = 144 | 144.0000 | ✅ |
        | dL/dw₁ | 48 × x = 96 | 96.0000 | ✅ |
        | dL/dw₂ | 24 × u₁ = 144 | 144.0000 | ✅ |

        顺带解开概念 1 埋的小彩蛋：dL/dx 和 dL/dw₂ **都是 144**——
        前者走"×w₂ × ×w₁"的长链，后者一步到位乘 u₁，两条路在数字上
        撞车纯属这组参数的巧合（x = 2 恰好等于 u₁/3……不对，
        u₁ = 6、x = 2：24×6 = 144、48×2 = 96，撞车的是 dL/dx 与 dL/dw₂
        的 144 = 24×6 与 48×3——各自成立，互不相干）。
        **对账表的意义就在这：不靠巧合，只靠复算。**
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # 演示 1 · 海报兑现 + 全梯度对账（13 课对称差分）
    from matplotlib.patches import Circle as _Circle
    from matplotlib.patches import FancyArrowPatch as _FArrow

    _x, _w1, _w2 = 2.0, 3.0, 2.0
    _u1 = _w1 * _x                                 # 6
    _u2 = _w2 * _u1                                # 12
    _L = _u2 ** 2                                  # 144

    # ---- 反向：每条边 = 上游梯度 × 本级导数 ----
    _g_L = 1.0                                     # 起跳线：dL/dL = 1
    _g_u2 = _g_L * 2 * _u2                         # 24（平方的导数 2u₂）
    _g_u1 = _g_u2 * _w2                            # 48（乘法交换）
    _g_x = _g_u1 * _w1                             # 144
    _g_w1 = _g_u1 * _x                             # 96
    _g_w2 = _g_u2 * _u1                            # 144

    # ---- 验钞：对称差分 ----
    _eps = 1e-5
    _num_x = ((_w2 * _w1 * (_x + _eps)) ** 2 - (_w2 * _w1 * (_x - _eps)) ** 2) / (2 * _eps)
    _num_w1 = ((_w2 * (_w1 + _eps) * _x) ** 2 - (_w2 * (_w1 - _eps) * _x) ** 2) / (2 * _eps)
    _num_w2 = (((_w2 + _eps) * _w1 * _x) ** 2 - ((_w2 - _eps) * _w1 * _x) ** 2) / (2 * _eps)

    # ---- 画图（14 课同款，配色不动）----
    _fig, _ax = plt.subplots(figsize=(8.6, 4.4))
    _ax.set_xlim(0, 10.8)
    _ax.set_ylim(1.7, 6.5)
    _ax.axis("off")
    _y_node = 4.7
    _nodes = [("x", 1.1, _x), ("u1", 4.0, _u1), ("u2", 6.9, _u2), ("L", 9.6, _L)]
    for _name, _xc, _val in _nodes:
        _ax.add_patch(_Circle((_xc, _y_node), 0.62, facecolor="#e8f2fb",
                              edgecolor="#1f77b4", lw=1.8))
        _ax.text(_xc, _y_node, _name, ha="center", va="center", fontsize=15)
        _ax.text(_xc, _y_node - 0.95, f"值 = {_val:g}", ha="center",
                 va="center", fontsize=11, color="#1f77b4")
    for _label, _xa, _xb in [("× w1 (3)", 1.1, 4.0), ("× w2 (2)", 4.0, 6.9), ("平方", 6.9, 9.6)]:
        _ax.add_patch(_FArrow((_xa + 0.68, _y_node), (_xb - 0.68, _y_node),
                              arrowstyle="-|>", mutation_scale=16,
                              color="#1f77b4", lw=2))
        _ax.text((_xa + _xb) / 2, _y_node + 0.33, _label, ha="center",
                 fontsize=11.5, color="#1f77b4")
    for _label, _xa, _xb in [("dL/du2 = 24", 9.6, 6.9), ("dL/du1 = 48", 6.9, 4.0),
                             ("dL/dx = 144", 4.0, 1.1)]:
        _ax.add_patch(_FArrow((_xa, 2.85), (_xb, 2.85), arrowstyle="-|>",
                              mutation_scale=16, color="#d95f02", lw=2))
        _ax.text((_xa + _xb) / 2, 2.42, _label, ha="center", fontsize=11.5, color="#d95f02")
    _ax.text(0.12, 6.12, "前向 →：从左到右，算出每个节点的值", fontsize=11, color="#1f77b4")
    _ax.text(0.12, 5.70, "反向 ←：从右到左，传回每条边的变化率", fontsize=11, color="#d95f02")
    _ax.text(9.6, 5.62, "起跳线：dL/dL = 1", ha="center", fontsize=9.5, color="gray")
    _ax.set_title("计算图：14 课的海报，今天兑现", fontsize=13)

    mo.vstack([
        _fig,
        mo.md(
            f"""
            | 梯度 | 手推 | 对称差分 | 判决 |
            |---|---|---|---|
            | dL/dx | **{_g_x:.0f}** | {_num_x:.4f} | ✅ |
            | dL/dw₁ | **{_g_w1:.0f}** | {_num_w1:.4f} | ✅ |
            | dL/dw₂ | **{_g_w2:.0f}** | {_num_w2:.4f} | ✅ |
            """
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 2 · 微型网络前向：五步算到损失

        正片主角登场。网络（概念 3 的公式）、参数、输入全部就位——
        上排小圈是**参数节点**（它们也要领梯度），下排大圈是**数据流**。
        前向五步，每步只做一件小事：

        ```
        z₁ = 0.8×1.5 − 0.2 = 1.000      （第一层打分）
        h  = tanh(1.000)   = 0.762      （隐层弯折，19 课）
        z₂ = 1.3×0.762 − 0.5 = 0.490    （第二层打分）
        p  = σ(0.490)      = 0.620      （输出概率，16 课）
        L  = −log(0.620)   = 0.478      （交叉熵，y = 1，21 课）
        ```

        网络押"62% 是正类"——方向对，不够自信。**下一步：告诉每个参数
        "你该怎么动，损失才能降"**。
        """
    )
    return


@app.cell
def _(np, plt):
    # 演示 2 · 微型网络：定义 + 前向 + 计算图（前向部分）
    from matplotlib.patches import Circle as _Circle
    from matplotlib.patches import FancyArrowPatch as _FArrow

    # ---- 网络与数据（全课共用这一套）----
    x_in = 1.5                                     # 输入
    y_true = 1.0                                   # 真值（正类）
    w1, b1, w2, b2 = 0.8, -0.2, 1.3, -0.5          # 4 个初始参数

    # ---- 前向五步（顺手存中间量：20 课埋的 cache 习惯）----
    z1 = w1 * x_in + b1                            # 1.000
    h = np.tanh(z1)                                # 0.762
    z2 = w2 * h + b2                               # 0.490
    p = 1 / (1 + np.exp(-z2))                      # 0.620
    L_val = -np.log(p)                             # 0.478（y=1 时交叉熵 = −log p）

    # ---- 画计算图（前向）----
    _fig, _ax = plt.subplots(figsize=(12.8, 4.4))
    _ax.set_xlim(0, 14.6)
    _ax.set_ylim(0.4, 6.4)
    _ax.axis("off")
    _ymain = 3.4                                   # 主链高度
    _main = [("x", 1.1, x_in, f"{x_in:g}"), ("z1", 3.7, z1, f"{z1:.3f}"),
             ("h", 6.3, h, f"{h:.3f}"), ("z2", 8.9, z2, f"{z2:.3f}"),
             ("p", 11.4, p, f"{p:.3f}"), ("L", 13.6, L_val, f"{L_val:.3f}")]
    for _name, _xc, _val, _txt in _main:
        _ax.add_patch(_Circle((_xc, _ymain), 0.52, facecolor="#e8f2fb",
                              edgecolor="#1f77b4", lw=1.7))
        _ax.text(_xc, _ymain, _name, ha="center", va="center", fontsize=12.5)
        _ax.text(_xc, _ymain - 0.92, _txt, ha="center", fontsize=10,
                 color="#1f77b4", weight="bold")
    # 参数节点（小圈，悬在主链上方）
    _params = [("w1", 2.9, w1), ("b1", 4.5, b1), ("w2", 8.1, w2), ("b2", 9.7, b2)]
    for _name, _xc, _val in _params:
        _ax.add_patch(_Circle((_xc, 5.3), 0.38, facecolor="#fdf3e3",
                              edgecolor="#e6a23c", lw=1.5))
        _ax.text(_xc, 5.3, _name, ha="center", va="center", fontsize=9.5)
        _ax.text(_xc, 5.86, f"{_val:g}", ha="center", fontsize=8.5, color="#b88230")
    # 边（主链 + 参数汇入）
    for _xa, _xb, _lab in [(1.66, 3.16, "×w₁+b₁"), (4.24, 5.76, "tanh"),
                           (6.84, 8.36, "×w₂+b₂"), (9.44, 10.86, "σ"),
                           (11.94, 13.06, "−log")]:
        _ax.add_patch(_FArrow((_xa, _ymain), (_xb, _ymain), arrowstyle="-|>",
                              mutation_scale=14, color="#1f77b4", lw=1.9))
        _ax.text((_xa + _xb) / 2, _ymain + 0.26, _lab, ha="center",
                 fontsize=9.5, color="#1f77b4")
    for _xc in [2.9, 4.5]:
        _ax.add_patch(_FArrow((_xc, 4.9), (3.7, 4.0), arrowstyle="-|>",
                              mutation_scale=10, color="#e6a23c", lw=1.3))
    for _xc in [8.1, 9.7]:
        _ax.add_patch(_FArrow((_xc, 4.9), (8.9, 4.0), arrowstyle="-|>",
                              mutation_scale=10, color="#e6a23c", lw=1.3))
    _ax.text(0.15, 1.15, "前向 →：五步算到损失；上排橙色小圈是参数节点——下一步反向时，每个都要领梯度",
             fontsize=10, color="#1f77b4")
    _ax.set_title("微型网络计算图（前向）：4 参数、7 节点", fontsize=12.5)
    _fig
    return (b1, b2, h, L_val, p, w1, w2, x_in, y_true, z1, z2)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 3 · 反向逐节点：两条铁律连用七次

        橙色箭头从右往左，每过一条边就是一次"**上游梯度 × 本级导数**"。
        表格把七次乘法一一登记（对照图读，一个都别跳）：

        | # | 算什么 | 用哪条 | 数值 |
        |---|---|---|---|
        | 1 | δ₂ = (p − y) | 21 课的合体导数 | −0.380 |
        | 2 | dL/dw₂ = δ₂·h | 乘法交换 | −0.289 |
        | 3 | dL/db₂ = δ₂·1 | 加法分发 | −0.380 |
        | 4 | dL/dh = δ₂·w₂ | 乘法交换 | −0.494 |
        | 5 | δ₁ = dL/dh·(1−h²) | tanh'（19 课） | −0.207 |
        | 6 | dL/dw₁ = δ₁·x | 乘法交换 | **−0.311**（最长链的终点） |
        | 7 | dL/db₁ = δ₁·1 | 加法分发 | −0.207 |

        盯住第 6 行——它把四条边串成一串：
        **δ₂ × w₂ × (1−h²) × x**——第 14 课说"深层的参数要对最外层的损失求导，
        必须一环一环传进去"，这四连环就是那句话的最小实例。
        23 课把它升级成矩阵版（dW₁ = δ₁·xᵀ），骨架不变、只是把数换成块。
        """
    )
    return


@app.cell
def _(b1, b2, h, mo, np, p, plt, w1, w2, x_in, y_true, z1, z2):
    # 演示 3 · 反向传播：从 δ₂ 一路传回 w₁（每步都用两条铁律之一）
    delta2 = p - y_true                            # ① 21 课：σ 配 CE 的合体导数
    grad_w2 = delta2 * h                           # ② 乘法交换（另一个因子是 h）
    grad_b2 = delta2 * 1.0                         # ③ 加法分发
    dL_dh = delta2 * w2                            # ④ 乘法交换（另一个因子是 w₂）
    delta1 = dL_dh * (1 - h ** 2)                  # ⑤ tanh' = 1 − h²（19 课）
    grad_w1 = delta1 * x_in                        # ⑥ 乘法交换（另一个因子是 x）
    grad_b1 = delta1 * 1.0                         # ⑦ 加法分发

    # ---- 画图：在演示 2 的图上加橙色反向行 ----
    from matplotlib.patches import Circle as _Circle
    from matplotlib.patches import FancyArrowPatch as _FArrow
    _fig, _ax = plt.subplots(figsize=(12.8, 4.6))
    _ax.set_xlim(0, 14.6)
    _ax.set_ylim(0.2, 6.4)
    _ax.axis("off")
    _ymain = 3.9
    _main = [("x", 1.1), ("z1", 3.7), ("h", 6.3), ("z2", 8.9), ("p", 11.4), ("L", 13.6)]
    _vtxt = [f"{x_in:g}", f"{z1:.3f}", f"{h:.3f}", f"{z2:.3f}", f"{p:.3f}",
             f"{-np.log(p):.3f}"]
    for (_name, _xc), _txt in zip(_main, _vtxt):
        _ax.add_patch(_Circle((_xc, _ymain), 0.52, facecolor="#e8f2fb",
                              edgecolor="#1f77b4", lw=1.7))
        _ax.text(_xc, _ymain, _name, ha="center", va="center", fontsize=12.5)
        _ax.text(_xc, _ymain - 0.92, _txt, ha="center", fontsize=10,
                 color="#1f77b4", weight="bold")
    for _xa, _xb, _lab in [(1.66, 3.16, "×w₁+b₁"), (4.24, 5.76, "tanh"),
                           (6.84, 8.36, "×w₂+b₂"), (9.44, 10.86, "σ"),
                           (11.94, 13.06, "−log")]:
        _ax.add_patch(_FArrow((_xa, _ymain), (_xb, _ymain), arrowstyle="-|>",
                              mutation_scale=14, color="#1f77b4", lw=1.9))
        _ax.text((_xa + _xb) / 2, _ymain + 0.26, _lab, ha="center",
                 fontsize=9.5, color="#1f77b4")
    # 反向箭头（橙，向左）+ 每个节点收到的梯度
    _bwd = [("dL/dp = −1.61", 11.4, 9.9), ("δ₂ = −0.38", 9.4, 7.9),
            ("dL/dh = −0.49", 6.9, 5.4), ("δ₁ = −0.21", 5.4, 3.9),
            ("dL/dx = −0.31", 3.4, 1.7)]
    for _lab, _xa, _xb in _bwd:
        _ax.add_patch(_FArrow((_xa, 1.55), (_xb, 1.55), arrowstyle="-|>",
                              mutation_scale=14, color="#d95f02", lw=1.9))
        _ax.text((_xa + _xb) / 2, 1.05, _lab, ha="center", fontsize=9.5, color="#d95f02")
    _ax.text(13.35, 2.0, "起跳：dL/dL = 1", fontsize=8.5, color="gray", ha="center")
    # 参数梯度（橙色，挂在参数位置）
    _pg = [("w₁: −0.311", 2.9), ("b₁: −0.207", 4.7), ("w₂: −0.289", 8.1), ("b₂: −0.380", 9.9)]
    for _lab, _xc in _pg:
        _ax.text(_xc, 5.9, _lab, ha="center", fontsize=9.5, color="#d95f02", weight="bold")
    _ax.set_title("反向传播：橙箭头逐边回传，参数各自领梯度", fontsize=12.5)
    _fig

    mo.vstack([
        _fig,
        mo.md(
            f"""
            | # | 公式 | 数值 |
            |---|---|---|
            | ① | δ₂ = p − y | **{delta2:+.4f}** |
            | ② | dL/dw₂ = δ₂·h | **{grad_w2:+.4f}** |
            | ③ | dL/db₂ = δ₂ | **{grad_b2:+.4f}** |
            | ④ | dL/dh = δ₂·w₂ | **{dL_dh:+.4f}** |
            | ⑤ | δ₁ = dL/dh·(1−h²) | **{delta1:+.4f}** |
            | ⑥ | dL/dw₁ = δ₁·x（四连环终点） | **{grad_w1:+.4f}** |
            | ⑦ | dL/db₁ = δ₁ | **{grad_b1:+.4f}** |

            四个梯度**全是负数**——含义一读就懂：四个参数**全部增大**，
            损失都会降（现在 p = 0.62 还不够自信，网络想押更高）。
            更新一步（lr = 0.5）之后会发生什么？演示 5 见。
            """
        ),
    ])
    return (delta2, grad_b1, grad_b2, grad_w1, grad_w2)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 4 · 验钞机：四条梯度对账

        13 课的对称差分再上岗：把每个参数单独拧动 ±10⁻⁷，
        损失的实差 ÷ 扰动量 = 数值梯度。四条全对——

        | 梯度 | 反向传播 | 对称差分 | 误差 |
        |---|---|---|---|
        | dL/dw₁ | −0.3111 | −0.31109986 | ~10⁻⁸ |
        | dL/db₁ | −0.2074 | −0.20739990 | ~10⁻⁸ |
        | dL/dw₂ | −0.2893 | −0.28931173 | ~10⁻⁸ |
        | dL/db₂ | −0.3799 | −0.37987651 | ~10⁻⁸ |

        注意成本账：数值法每个参数要**跑两遍完整前向**，4 个参数 8 遍；
        反向传播**一趟**全拿下——参数一多（101,770 个？）差距是
        五十万比一。这就是"为什么不用数值梯度训练、却必须用它对账"
        的完整答案（13 课埋的线，今天收回）。
        """
    )
    return


@app.cell
def _(b1, b2, delta2, grad_b1, grad_b2, grad_w1, grad_w2, mo, np, w1, w2, x_in, y_true):
    # 演示 4 · 验钞机：反向传播 vs 对称差分
    def _loss_at(_w1_, _b1_, _w2_, _b2_):          # 完整前向的函数版
        _z1_ = _w1_ * x_in + _b1_
        _h_ = np.tanh(_z1_)
        _z2_ = _w2_ * _h_ + _b2_
        _p_ = 1 / (1 + np.exp(-_z2_))
        return -(y_true * np.log(_p_ + 1e-300) + (1 - y_true) * np.log(1 - _p_ + 1e-300))

    _eps = 1e-7
    _num_w1 = (_loss_at(w1 + _eps, b1, w2, b2) - _loss_at(w1 - _eps, b1, w2, b2)) / (2 * _eps)
    _num_b1 = (_loss_at(w1, b1 + _eps, w2, b2) - _loss_at(w1, b1 - _eps, w2, b2)) / (2 * _eps)
    _num_w2 = (_loss_at(w1, b1, w2 + _eps, b2) - _loss_at(w1, b1, w2 - _eps, b2)) / (2 * _eps)
    _num_b2 = (_loss_at(w1, b1, w2, b2 + _eps) - _loss_at(w1, b1, w2, b2 - _eps)) / (2 * _eps)

    _rows = [("dL/dw₁", grad_w1, _num_w1), ("dL/db₁", grad_b1, _num_b1),
             ("dL/dw₂", grad_w2, _num_w2), ("dL/db₂", grad_b2, _num_b2)]
    _max_err = max(abs(_bp - _nm) for _, _bp, _nm in _rows)

    mo.md(
        f"""
        | 梯度 | 反向传播 | 对称差分 | 误差 |
        |---|---|---|---|
        | dL/dw₁ | **{grad_w1:+.8f}** | {_num_w1:+.8f} | {abs(grad_w1-_num_w1):.1e} |
        | dL/db₁ | **{grad_b1:+.8f}** | {_num_b1:+.8f} | {abs(grad_b1-_num_b1):.1e} |
        | dL/dw₂ | **{grad_w2:+.8f}** | {_num_w2:+.8f} | {abs(grad_w2-_num_w2):.1e} |
        | dL/db₂ | **{grad_b2:+.8f}** | {_num_b2:+.8f} | {abs(grad_b2-_num_b2):.1e} |

        **最大误差 {_max_err:.1e}**——验钞通过。
        （δ₂ = {delta2:+.4f} 恰好等于 p − y：21 课的结论在真网络里原样兑现。）
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 5 · 点火：200 步，参数第一次自己学会

        四步曲挂自动挡（15 课的骨架，第 ③ 步换成今天的反向传播）。
        lr = 0.5，跑 200 步：

        左图损失曲线：0.478 一路俯冲，第 100 步破 0.01，
        第 200 步停在 **0.005**。右图押注曲线：0.62 稳步爬到 **0.995**——
        网络从"方向对但犹豫"学到"斩钉截铁"。

        三个值得驻足的细节：

        - **梯度自己会刹车**：p 越接近 1，δ₂ = p − y 越小，曲线越平缓——
          不是 lr 变了，是"错得越来越少"（21 课的比例阀在工作）
        - **四个参数同步进化**：w₁ 0.8 → 1.49、b₁ −0.2 → 0.26、
          w₂ 1.3 → 3.50、b₂ −0.5 → 1.84——没有谁指挥谁，
          各自跟着自己的梯度走，整体却协调得像排练过
        - **没有一步是手拍的**：20 课拆 XOR 墙靠人设权重，
          这次——**参数自己走到了好位置**。这两个演示之间，
          隔着的正是"训练"二字的全部含义
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # 演示 5 · 训练微型网络：四步曲 × 200 步（train_micro 供交互实验复用）
    def train_micro(lr=0.5, n_steps=200, init=(0.8, -0.2, 1.3, -0.5),
                    x=1.5, y=1.0, loss="bce"):
        """训练微型网络。返回历史记录 dict（每步的 L、p、z1、h、z2、δ₂、grad_w1、参数）"""
        _W1, _B1, _W2, _B2 = init
        _hist = {"L": [], "p": [], "z1": [], "h": [], "z2": [], "d2": [],
                 "gw1": [], "w1": [], "b1": [], "w2": [], "b2": []}
        for _t in range(n_steps):
            # ① 前向（顺手存中间量）
            _z1 = _W1 * x + _B1
            _h = np.tanh(_z1)
            _z2 = _W2 * _h + _B2
            _p = 1 / (1 + np.exp(-_z2))
            # ② 损失（③ 反向在下面两行）
            if loss == "bce":
                _L = -(y * np.log(_p + 1e-300) + (1 - y) * np.log(1 - _p + 1e-300))
                _d2 = _p - y                             # 21 课：p − y
            else:                                        # MSE 版（练习 4）
                _L = (_p - y) ** 2
                _d2 = 2 * (_p - y) * _p * (1 - _p)       # 链多两环：MSE' × σ'
            _gw1 = _d2 * _W2 * (1 - _h ** 2) * x         # 四连环
            # ④ 更新
            _W1 -= lr * _gw1
            _B1 -= lr * _d2 * _W2 * (1 - _h ** 2)
            _W2 -= lr * _d2 * _h
            _B2 -= lr * _d2
            _hist["L"].append(_L); _hist["p"].append(_p)
            _hist["z1"].append(_z1); _hist["h"].append(_h); _hist["z2"].append(_z2)
            _hist["d2"].append(_d2); _hist["gw1"].append(_gw1)
            _hist["w1"].append(_W1); _hist["b1"].append(_B1)
            _hist["w2"].append(_W2); _hist["b2"].append(_B2)
        return _hist

    _hist = train_micro(lr=0.5, n_steps=200)
    _reach = next((_i + 1 for _i, _v in enumerate(_hist["L"]) if _v < 0.01), None)

    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4.2))
    _axes[0].plot(range(1, 201), _hist["L"], color="tab:blue", lw=2.2)
    _axes[0].axhline(0.01, color="gray", ls=":", lw=1.2)
    _axes[0].annotate(f"第 {_reach} 步破 0.01", xy=(_reach, 0.01), xytext=(_reach + 30, 0.12),
                      fontsize=10, arrowprops=dict(arrowstyle="->", color="gray"))
    _axes[0].set_title(f"损失：0.478 → {_hist['L'][-1]:.4f}")
    _axes[0].set_xlabel("步数")
    _axes[0].set_ylabel("L")
    _axes[0].grid(True, alpha=0.3)

    _axes[1].plot(range(1, 201), _hist["p"], color="tab:green", lw=2.2)
    _axes[1].axhline(1.0, color="gray", ls=":", lw=1)
    _axes[1].set_title(f"押注：0.62 → {_hist['p'][-1]:.4f}（真值 = 1）")
    _axes[1].set_xlabel("步数")
    _axes[1].set_ylabel("p")
    _axes[1].set_ylim(0.5, 1.02)
    _axes[1].grid(True, alpha=0.3)

    _fig.tight_layout()
    _fig
    return (train_micro,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        读一遍最终成绩单（lr = 0.5、200 步）：

        | 量 | 起点 | 终点 |
        |---|---|---|
        | 损失 L | 0.478 | **0.005** |
        | 押注 p | 0.620 | **0.995** |
        | w₁, b₁ | 0.80, −0.20 | **1.49, 0.26** |
        | w₂, b₂ | 1.30, −0.50 | **3.50, 1.84** |
        | 达到 L < 0.01 | — | 第 **101** 步（L₁₀₀ = 0.0100，一步之差） |

        一个诚实的小观察：这个玩具网络"饱和保护"极好——学习率
        拉到 3 都不翻车（p 被 sigmoid 压着，梯度到顶就消失）。
        **真实的多参数网络可没这么乖**，学习率大了照样震荡崩盘——
        15 课的"三种命运"在 24 课的优化器大会上还会重演，届时对手
        换成真正的多参数地形。
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 🎛️ 交互实验

        本课两组（核心课加倍）：

        1. **步长调速台**——拖动学习率，看损失曲线的节奏变化，
           判词报告"达到 L < 0.01 用了几步"，和 lr = 0.5 的
           基准（101 步）比个快慢
        2. **单步透视镜**——快进到任意一步，把**该步的全部内脏**
           （参数、节点值、梯度）摆上解剖台；重点盯 δ₂ 怎么随
           训练推进自己变小——"错多少、学多少"的实时读数

        玩法提示：调速台里把 lr 拖到 0.05 看"磨蹭"、拖到 3 看"狂飙"
        （这个玩具网络饱和保护极好，不会翻车）；透视镜里对比
        第 1 步和第 200 步的 |δ₂|，差了一个数量级。
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 交互实验 1 · 步长调速台
        """
    )
    return


@app.cell
def _(mo):
    # 交互控制：学习率（debounce：松手才重算）
    lr_slider = mo.ui.slider(0.05, 3.0, value=0.5, step=0.05, debounce=True,
                             label="🎛️ 学习率 lr（0.05 ~ 3.0）")
    lr_slider
    return (lr_slider,)


@app.cell
def _(lr_slider, mo, np, plt, train_micro):
    # ===== 交互实验 1 · 步长调速台（响应单元）=====
    _lr = lr_slider.value
    _hist = train_micro(lr=_lr, n_steps=200)
    _reach = next((_i + 1 for _i, _v in enumerate(_hist["L"]) if _v < 0.01), None)

    _fig, _ax = plt.subplots(figsize=(7.8, 4.2))
    _ax.plot(range(1, 201), _hist["L"], color="tab:blue", lw=2.2,
             label=f"lr = {_lr:.2f}")
    _ax.axhline(0.01, color="gray", ls=":", lw=1.2, label="达标线 L = 0.01")
    if _reach is not None:
        _ax.scatter([_reach], [_hist["L"][_reach - 1]], s=110, color="tab:red", zorder=4)
        _ax.annotate(f"第 {_reach} 步达标", xy=(_reach, _hist["L"][_reach - 1]),
                     xytext=(_reach + 25, 0.18), fontsize=10,
                     arrowprops=dict(arrowstyle="->", color="tab:red"))
    _ax.set_xlabel("步数")
    _ax.set_ylabel("损失 L")
    _ax.set_ylim(0, 0.55)
    _ax.set_title(f"步长调速台：lr = {_lr:.2f}")
    _ax.legend()
    _ax.grid(True, alpha=0.3)

    # ---- 判词：和基准（lr = 0.5 → 101 步）比快慢 ----
    if _reach is None:
        _verdict = (f"🐌 lr = {_lr:.2f}：200 步还没破 0.01（当前 {_hist['L'][-1]:.4f}）——"
                    "磨蹭型：步子太小，200 步只走了半程")
    elif _lr < 0.3:
        _verdict = f"🐢 lr = {_lr:.2f}：第 {_reach} 步达标——比基准（101 步）慢，但稳"
    elif _lr <= 1.5:
        _verdict = f"✅ lr = {_lr:.2f}：第 {_reach} 步达标——和基准（lr=0.5 的 101 步）同量级"
    else:
        _verdict = (f"🚀 lr = {_lr:.2f}：第 {_reach} 步就达标——狂飙。"
                    "注意：这是玩具网络的饱和保护，大网络早翻车了（24 课见）")

    mo.vstack([
        _fig,
        mo.callout(mo.md(f"**实时判词：{_verdict}**"), kind="info"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 交互实验 2 · 单步透视镜

        拖到第 t 步，解剖台亮灯：本步的**前向五连**（z₁ → h → z₂ → p → L）、
        **回传起点** δ₂、**最长链终点** grad_w₁，以及**本步更新后的四个参数**。
        表格右端那列 |δ₂| 是重点——它就是"这一步学得多用力"。
        """
    )
    return


@app.cell
def _(mo):
    # 交互控制：透视哪一步
    step_slider = mo.ui.slider(1, 200, value=1, step=1, debounce=True,
                               label="🔬 快进到第几步？（1 ~ 200）")
    step_slider
    return (step_slider,)


@app.cell
def _(mo, step_slider, train_micro):
    # ===== 交互实验 2 · 单步透视镜（响应单元）=====
    _t = step_slider.value                            # 第 t 步（1 起）
    _hist = train_micro(lr=0.5, n_steps=200)
    _k = _t - 1                                       # 数组下标

    _d2_abs = abs(_hist["d2"][_k])
    _d2_first = abs(_hist["d2"][0])
    _shrink = _d2_first / max(_d2_abs, 1e-12)

    mo.vstack([
        mo.md(
            f"""
            | 本步读数（第 {_t} 步） | 值 |
            |---|---|
            | z₁ → h → z₂ | {_hist['z1'][_k]:+.4f} → {_hist['h'][_k]:+.4f} → {_hist['z2'][_k]:+.4f} |
            | 押注 p → 损失 L | **{_hist['p'][_k]:.4f}** → **{_hist['L'][_k]:.4f}** |
            | δ₂ = p − y | **{_hist['d2'][_k]:+.4f}**（学习力度） |
            | grad_w₁（四连环） | **{_hist['gw1'][_k]:+.4f}** |
            | 更新后 w₁, b₁ | {_hist['w1'][_k]:+.3f}, {_hist['b1'][_k]:+.3f} |
            | 更新后 w₂, b₂ | {_hist['w2'][_k]:+.3f}, {_hist['b2'][_k]:+.3f} |
            """
        ),
        mo.callout(
            mo.md(
                f"**实时判词：**此刻 |δ₂| = {_d2_abs:.4f}，是第 1 步"
                f"（{_d2_first:.4f}）的 **1/{_shrink:.0f}**——错得越少，学得越轻；"
                "21 课的比例阀，在训练里实时工作。"
            ),
            kind="info",
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✏️ 动手练习（每课必做，本课 4 题）": mo.md(
                """
                **练习 1 · 补全前向**：`forward_micro` 的②④两行被抽走
                （tanh 和 sigmoid 各一处）——验收标准：四个输出和演示 2 分毫不差。

                **练习 2 · 补全 δ₂ 与 dL/dw₂**：反向的起跳和第一跳
                （21 课的 p − y，加上乘法交换）。

                **练习 3 · 最长链 dL/dw₁**：四连环的后三环——
                乘法交换、tanh'、再乘法交换。和演示 3 的答案对账。

                **练习 4 · 挑战 · 换损失：MSE 的"迈不动腿"**：
                把起点挪到坏处（p₀ = 0.079，错得离谱），分别用
                交叉熵和 MSE 训练 200 步，填出两个终点押注——
                亲眼看 16 课预言的病发作。

                规则不变：先把 `None` 换成你的答案再运行；写完再看参考答案。
                """
            )
        }
    )
    return


@app.cell
def _(mo, np):
    # ===== 练习 1 · 补全前向 =====
    def forward_micro(w1, b1, w2, b2, x=1.5):
        """微型网络前向。返回 (z1, h, z2, p)"""
        z1 = w1 * x + b1          # ① 第一层打分（不变）
        h = None                  # ← ② 补：np.tanh(z1)
        z2 = w2 * h + b2          # ③ 第二层打分（不变）
        p = None                  # ← ④ 补：1 / (1 + np.exp(-z2))
        return z1, h, z2, p

    fwd_ready_ex1 = None          # ← ⑤ 补完后，把这行换成 forward_micro

    if fwd_ready_ex1 is not None:
        try:
            _z1_, _h_, _z2_, _p_ = fwd_ready_ex1(0.8, -0.2, 1.3, -0.5)
            _want = (1.0000, 0.7616, 0.4901, 0.6201)
            _got = (_z1_, _h_, _z2_, _p_)
            if all(abs(_g - _w) < 1e-3 for _g, _w in zip(_got, _want)):
                _msg = f"✅ z₁={_z1_:.4f} h={_h_:.4f} z₂={_z2_:.4f} p={_p_:.4f}——与演示 2 分毫不差"
            else:
                _msg = (f"❌ 你算出 ({_z1_:.4f}, {_h_:.4f}, {_z2_:.4f}, {_p_:.4f})，"
                        f"期望 {_want}——检查 ② 是 tanh、④ 是 sigmoid")
        except TypeError:
            _msg = "⏳ h 或 p 还是 None——②④ 两行补了吗？"
    else:
        _msg = "⏳ 先补 ②④，再把 fwd_ready_ex1 换成 forward_micro"
    mo.md(_msg)
    return (forward_micro, fwd_ready_ex1)


@app.cell
def _(forward_micro, mo, np, y_true):
    # ===== 练习 2 · 补全 δ₂ 与 dL/dw₂ =====
    try:
        _z1_, _h_, _z2_, _p_ = forward_micro(0.8, -0.2, 1.3, -0.5)   # 练习 1 的成果
        _fwd_ok = _h_ is not None and _p_ is not None
    except TypeError:
        _fwd_ok = False

    delta2_ex2 = None     # ← ① 补：_p_ - y_true        （21 课：p − y）
    grad_w2_ex2 = None    # ← ② 补：delta2_ex2 * _h_    （乘法交换：另一个因子是 h）

    if not _fwd_ok:
        _msg = "⏳ 练习 1 的 forward_micro 还没补完（h/p 还是 None）——先回上一题"
    elif delta2_ex2 is not None and grad_w2_ex2 is not None:
        _ok = (abs(delta2_ex2 - (-0.3799)) < 1e-3 and abs(grad_w2_ex2 - (-0.2893)) < 1e-3)
        _msg = (f"δ₂ = **{delta2_ex2:+.4f}**，dL/dw₂ = **{grad_w2_ex2:+.4f}**。"
                + ("✅ 和演示 3 的表对上了——反向的起跳与第一跳，会写了" if _ok
                   else "❌ 期望 −0.3799 和 −0.2893——检查①是不是 p−y、②乘的是不是 h"))
    else:
        _msg = "⏳ 两行都补上（① 用练习 1 算出的 _p_；② 乘法交换）"
    mo.md(_msg)
    return (delta2_ex2, grad_w2_ex2)


@app.cell
def _(forward_micro, grad_w1, mo, np, x_in, y_true):
    # ===== 练习 3 · 最长链 dL/dw₁ =====
    try:
        _z1_, _h_, _z2_, _p_ = forward_micro(0.8, -0.2, 1.3, -0.5)
        _fwd_ok = _h_ is not None and _p_ is not None
    except TypeError:
        _fwd_ok = False

    delta1_ex3 = None     # ← ① 补：(_p_ - y_true) * 1.3 * (1 - _h_**2)
                          #    （两环：乘法交换拿 w₂，tanh' 拿 1−h²）
    grad_w1_ex3 = None    # ← ② 补：delta1_ex3 * x_in   （最后一环：乘法交换拿 x）

    if not _fwd_ok:
        _msg = "⏳ 练习 1 的 forward_micro 还没补完——先回第一题"
    elif delta1_ex3 is not None and grad_w1_ex3 is not None:
        if abs(grad_w1_ex3 - grad_w1) < 1e-9:
            _msg = (f"δ₁ = **{delta1_ex3:+.4f}**，dL/dw₁ = **{grad_w1_ex3:+.4f}**——"
                    "✅ 与演示 3 的手推值精确一致，四连环贯通")
        else:
            _msg = (f"❌ 你算出 dL/dw₁ = {grad_w1_ex3:+.4f}，演示 3 是 {grad_w1:+.4f}——"
                    "四连环逐环检查：δ₂ → ×w₂ → ×(1−h²) → ×x")
    else:
        _msg = "⏳ 两行都补上（①②各一环，共补全四连环的后三环）"
    mo.md(_msg)
    return (delta1_ex3, grad_w1_ex3)


@app.cell
def _(mo, train_micro):
    # ===== 练习 4 · 挑战：坏起点下 BCE vs MSE =====
    _bad_init = (-2.0, 0.5, 3.0, 0.5)     # 坏起点：p₀ = 0.079，错得离谱（y = 1）

    # 提示：train_micro 支持两种损失——
    #   train_micro(lr=0.5, n_steps=200, init=_bad_init, loss="bce")["p"][-1]
    #   train_micro(lr=0.5, n_steps=200, init=_bad_init, loss="mse")["p"][-1]
    # （MSE 版的 δ₂ 已经写在 train_micro 里：2(p−y)·p(1−p)，去演示 5 的代码里看一眼）

    p_bce_200_ex4 = None    # ← ① 交叉熵版 200 步后的 p
    p_mse_200_ex4 = None    # ← ② MSE 版 200 步后的 p

    if p_bce_200_ex4 is not None and p_mse_200_ex4 is not None:
        _ok_bce = abs(float(p_bce_200_ex4) - 0.9946) < 0.01
        _ok_mse = abs(float(p_mse_200_ex4) - 0.9597) < 0.02
        if _ok_bce and _ok_mse:
            _msg = (f"BCE → **{float(p_bce_200_ex4):.3f}**，MSE → **{float(p_mse_200_ex4):.3f}**。"
                    "✅ 差距看清了——参考答案里有这场病的完整病理报告")
        else:
            _msg = (f"❌ 你填的 BCE={p_bce_200_ex4}、MSE={p_mse_200_ex4}"
                    "（期望约 0.995 和 0.960）——注意 init 用坏起点、lr=0.5、200 步")
    else:
        _msg = "⏳ 两个终点押注都跑出来（一行一个，loss 参数换一下）"
    mo.md(_msg)
    return (p_bce_200_ex4, p_mse_200_ex4)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 1 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                h = np.tanh(z1)                    # ② 隐层弯折（19 课）
                p = 1 / (1 + np.exp(-z2))          # ④ 输出概率（16 课）
                ```

                验收：z₁ = 1.0000、h = 0.7616、z₂ = 0.4901、p = 0.6201。

                两个工程习惯再敲一遍：

                - **中间量全存**（z₁、h、z₂、p 都返回）——反向传播
                  每一环都要用它们，前向不存、反向就得重算一遍
                - **激活写在打分之后**：`z2 = w2 * h + b2` 先算完，
                  再 `p = σ(z2)`——把"线性"和"非线性"分两行写，
                  梯度公式（δ₂、1−h²）才能一一对应
                """
            ),
            "✅ 练习 2 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                delta2_ex2 = _p_ - y_true          # ① p − y（21 课）
                grad_w2_ex2 = delta2_ex2 * _h_     # ② 乘法交换
                ```

                δ₂ = **−0.3799**、dL/dw₂ = **−0.2893**，与演示 3 对上。

                为什么 δ₂ 一行就够：它其实是**两步合一步**——
                dL/dp = −1/p，dp/dz₂ = p(1−p)，相乘：
                (−1/p)·p(1−p) = −(1−p) = p−1 = p−y（y=1 时）。
                σ 配 CE 的导数化简后干干净净，这就是 21 课
                "漂亮到不像话"的全部来历。**合体导数**值得单记：
                以后见到 sigmoid+交叉熵，直接写 p − y。
                """
            ),
            "✅ 练习 3 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                delta1_ex3 = (_p_ - y_true) * 1.3 * (1 - _h_**2)   # ① δ₂ × w₂ × tanh'
                grad_w1_ex3 = delta1_ex3 * x_in                    # ② × x
                ```

                δ₁ = **−0.2074**、dL/dw₁ = **−0.3111**——与演示 3 精确一致。

                读一遍这条四连环的"传令路径"：

                ```
                dL/dw₁ = δ₂ × w₂ × (1 − h²) × x
                        （输出层误差）（往回穿一层）（tanh 的坡）（自己的输入）
                ```

                每一环都是**本地信息**：w₂、h、x 全是前向攒下的值——
                这就是概念 2 说的"局部性"。23 课把它矩阵化：
                `dW₁ = δ₁ · xᵀ`，四个标量变成两块矩阵，骨架一模一样。
                """
            ),
            "✅ 练习 4 参考答案（先自己写过再看！）": mo.md(
                """
                答案：BCE 版 200 步后 p = **0.9946**；MSE 版只有 **0.9597**。

                病理报告（16 课"迈不动腿"的完整发作过程）：

                | 时刻 | BCE 的 δ₂ | MSE 的 δ₂ |
                |---|---|---|
                | 起点 p = 0.079 | p − y = **−0.921**（全力纠错） | 2(p−y)·p(1−p) = 2×(−0.921)×0.0726 = **−0.134**（有气无力） |
                | 前 10 步推进 | 0.079 → **0.850** | 0.079 → 0.359 |
                | 达到 p > 0.9 | **14 步** | **48 步**（3.4 倍慢） |

                病根在**双重平坦**：错得离谱时 p 贴着 0，sigmoid 的导数
                p(1−p) ≈ 0.073 也贴着 0——MSE 直接把这个小数字再乘上去，
                梯度被压成零头；交叉熵的 p − y 恰好**铲平**了 sigmoid
                这块平地（练习 2 的合体导数，分母的 p(1−p) 被约掉了）。

                一句话：**错得越离谱，越需要大步子——CE 给得起，
                MSE 给不起**。这就是配对表（21 课概念 1）背后最深的一条理由。
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
                1. **反向传播 = 链式法则 + 一张图**：前向从左到右算值，
                   反向从右到左传率；起跳线永远是 dL/dL = 1；
                   本级导数要在**节点当前值**处取值（14 课的老话）
                2. **两条铁律走天下**：加法 = 原样分发（dL/db = δ）；
                   乘法 = 交换因子（dL/dw = δ·输入，dL/dx = δ·w）。
                   tanh/σ/平方等一元运算查导数表（19 课）
                3. **局部性是引擎的 scalable 之本**：每个节点只要
                   "自己的值 + 自己的导数公式"，不需要知道全局——
                   百万参数的网络也只是这条规则的百万次重复
                   （PyTorch 的 autograd，46 课见）
                4. **微型网络全手推**：δ₂ = p − y（21 课合体导数）→
                   dL/dw₂ = δ₂·h → dL/dh = δ₂·w₂ → δ₁ = δ₂·w₂·(1−h²) →
                   dL/dw₁ = δ₁·x（四连环）；四个参数的梯度对账误差 ~10⁻⁸
                5. **训练 = 前向存值 + 反向传率 + 梯度更新**：
                   200 步把损失从 0.478 压到 0.005、押注 0.62 → 0.995——
                   没有一步手拍，参数自己走到好位置
                6. **换损失只换最后一环**：CE 的 δ₂ = p − y，
                   MSE 的 δ₂ = 2(p−y)·p(1−p)——其余六步一字不改。
                   坏起点下 BCE 14 步达标、MSE 要 48 步：
                   "错得越离谱越要大步子"，CE 给得起，MSE 给不起
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

        **下一课预告** — `23_反向传播下`：

        标量版的发动机转起来了，下一课把它装上真正的底盘——
        **矩阵版反向传播**：

        - 一个样本变**一批**：X (n, d) 一把过，for 循环扔掉
        - 四条矩阵梯度公式（dW₂ = δ₂ᵀ·H 这一家子），用**形状检查法**
          推导——结果形状必须和参数形状一致，错一点都不行
        - 新数据集**双月牙**登场：不是一条直线能对付的形状，
        正好检验两层 MLP + 手写反向传播的成色
        - 完整训练 + 决策边界 + 验证集体检（17 课的纪律全程在线），
          隐单元数的容量实验预告 25 课的正则化

        练习 3 那条四连环，将原样升级成 `δ₁ = (δ₂·W₂ᵀ) ⊙ (1−H²)`——
        标量变矩阵，骨架一毫米不变。

        🎆 发动机点火完毕，装上车轮！
        """
    )
    return


if __name__ == "__main__":
    app.run()
