import marimo

__generated_with = "0.25.2"
app = marimo.App(app_title="11 · 向量的几何直觉", width="medium")


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

    # ==== 中文乱码修复（中文用户必踩的坑，一次设置全程生效）====
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
    plt.rcParams["axes.unicode_minus"] = False
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # 11 · 向量的几何直觉

        **从零到 Transformer · 手写深度学习** ｜ 第 2 部分 · 深度学习的数学（1/4）
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📖 本课目标（先读我）": mo.md(
                """
                **这一课只解决一个问题：怎么让公式长出画面？**

                第 1 部分十课功成：NumPy 工具箱已就位。第 2 部分给它装上"数学眼睛"——
                第 09 课埋了一个悬念：点积的**三种走向**（大 → 一致、≈0 → 垂直、负 → 相反）
                背后是一套完整的几何故事，今天正式揭晓。

                还记得第 10 课结尾的承诺吗？"梯度下降"这四个字，将变成小球在山谷里
                滚动时选定的方向——而"方向"这个词，从今天起有了精确含义。

                学完你将能：

                - 把 NumPy 里的向量看成几何世界里的**箭头**：分量是坐标、范数是长度、方向是角度
                - 说出点积的几何真身：**a·b = |a||b|cosθ**，并用"投影"讲清楚它为什么对
                - 用**余弦相似度**比较两个向量的"方向"——深度学习里"这俩像不像"的默认答案
                - 正式启用 **matplotlib**：画曲线、画散点、画箭头、画热力图，顺手修好中文乱码

                前置：第 09 课的点积与范数（本课的地基）、第 06/07/08 课的广播 / newaxis / axis
                （练习里会一次性回收它们）。
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "🎯 概念 1/4 · 向量是支箭头": mo.md(
                """
                第 09 课里，向量是"一串数字"。几何视角里，向量是**一支箭头**：
                有起点（通常画在原点）、有方向、有长度。

                - **分量 = 坐标**：向量 (3, 2) 就是"从原点出发，向右走 3、向上走 2"的箭头
                - **范数 = 长度**（09 课的 `np.linalg.norm`）：箭头有多长，
                  √(3² + 2²) = √13 ≈ 3.61——勾股定理，第 09 课就验算过
                - **方向 = 与 x 轴的夹角**：`np.arctan2(y, x)` 给出方向角
                  （注意参数顺序是 **y 在前**，它比 arctan(y/x) 聪明：能分辨四个象限）

                还有第 09 课的老朋友**单位向量**：向量除以自己的范数——长度变成 1，
                只留方向。今天它会以"余弦相似度"的身份重新登场。

                为什么值得换视角？因为深度学习的关键词全是几何词：**梯度**是"最陡上坡的方向"，
                **梯度下降**是"朝低谷走"，**注意力**是"query 和 key 的夹角有多大"。
                没有画面，这些词只是汉字；有了画面，它们是直觉。
                """
            ),
            "📐 概念 2/4 · 点积的几何真身：|a||b|cosθ": mo.md(
                """
                第 09 课玩点积罗盘时你亲眼看到：方向一致点积大、垂直归零、相反变负。
                **为什么？** 因为点积有一个几何真身：

                ```
                a·b = |a| × |b| × cosθ        （θ 是两向量的夹角）
                ```

                换个写法更透亮：**a·b = |a| × (b 在 a 方向上的投影长度)**。

                "投影"= 让一束垂直于 a 的阳光照在 b 上，b 落在 a 这条线上的影子：
                影子长 = |b|cosθ。于是三种走向瞬间有了画面——

                - **夹角小**（cosθ > 0）：影子顺着 a 铺 → 点积为正，越齐越大
                - **垂直**（cosθ = 0）：影子缩成一个点，长度 0 → 点积恰好归零
                - **夹角超过 90°**（cosθ < 0）：影子"反着"投到 a 的延长线上 → 点积为负

                数值预告（代码演示 2 会画出来）：a = (3, 2)、b = (1, 3)，
                点积 = 9，两个长度 ≈ 3.61 和 3.16，夹角 ≈ 37.9°——两边算出的点积分毫不差。
                """
            ),
            "📏 概念 3/4 · 余弦相似度：只比方向，不比长短": mo.md(
                """
                第 09 课的罗盘实验里还有一个结论：**把 a 拉长一倍，点积翻倍，但"方向一致性"没变**。
                点积同时编码了方向和长度，可很多时候我们只想比方向——比如"苹果"和"香蕉"
                这两个词像不像，跟词向量写多长无关。

                修法你已经会了：各自**归一化**（除以长度），再做点积——

                ```
                cosθ = (a·b) / (|a| × |b|)      （余弦相似度 = 归一化点积）
                ```

                取值永远落在 **[-1, 1]**，一张含义表走天下：

                | cosθ | 含义 | 画面 |
                |---|---|---|
                | +1 | 完全同向 | 两支箭头重合 |
                | ≈ 0 | 垂直、无关 | 丁字尺 |
                | -1 | 正好相反 | 一支朝东一支朝西 |

                它是深度学习里"比较两个向量像不像"的默认答案：推荐系统算用户相似度、
                检索系统算句子相似度、第 7 部分的注意力机制给 query 和 key 打分——
                骨子里都是在算夹角的余弦。今天我们用它做一个"迷你语义搜索"。
                """
            ),
            "🎨 概念 4/4 · matplotlib 第一次握手": mo.md(
                """
                后面 30 多课的损失曲线、决策边界、注意力热力图，全靠 **matplotlib**。
                今天用十分钟把最核心的三件事学会：

                **① fig 与 ax 二件套。** `fig, ax = plt.subplots()` 一次拿到画布（figure）
                和坐标系（axes）——画布是纸，坐标系是纸上的坐标格。所有绘制都找 ax：
                `ax.plot(...)` 折线、`ax.scatter(...)` 散点、`ax.quiver(...)` 箭头。

                **② 图的"配套五件"。** 标题 `set_title`、轴标签 `set_xlabel/set_ylabel`、
                图例 `legend()`（读代码里的 `label=`）、网格 `grid(True)`、等比例
                `set_aspect("equal")`——最后这个是几何图的生命线：不设它，直角看起来
                会像斜角，圆会变成椭圆。

                **③ 中文乱码，一行修好。** matplotlib 默认字体不含汉字，中文标题会显示
                成方框。修法已经写在上面 import 单元里：给 `font.sans-serif` 一张候选字体
                名单（macOS 的苹方、Windows 的微软雅黑、Linux 的思源黑体），
                matplotlib 自动挑第一个装了的。你的本地机器上运行时同样生效。

                小差异提个醒：传统 Python 脚本画完图要写 `plt.show()` 才弹窗；
                在 marimo 里**把 fig 作为单元的最后表达式**就会自动显示（本课全程这么做）。
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

        五组演示，从"会画图"一路走到"迷你语义搜索"：

        - **演示 0** · matplotlib 第一次握手（画一条正弦曲线）
        - **演示 1** · 画出向量：两支箭头 + 长度 + 方向角
        - **演示 2** · 夹角：算 cosθ、arccos，把角画出来
        - **演示 3** · 投影：b 在 a 方向上的"影子"
        - **演示 4** · 余弦相似度矩阵：四个词向量，一张热力图
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 0 · matplotlib 第一次握手

        最经典的起点：画一条正弦曲线。重点看 `fig, ax` 二件套怎么拿、
        绘制函数怎么喂**两个等长的数组**（x 数组和 y 数组——第 03 课的 np.linspace 采点）。
        """
    )
    return


@app.cell
def _(np, plt):
    # 演示 0 · matplotlib 处女作：一条正弦曲线
    _x = np.linspace(0, 2 * np.pi, 100)        # 0 到 2π 均匀采 100 个点（03 课老朋友）

    _fig, _ax = plt.subplots(figsize=(7, 4))   # 二件套：fig=画布，ax=坐标系
    _ax.plot(_x, np.sin(_x), label="sin(x)")   # 折线图：喂 x 数组和 y 数组
    _ax.scatter([np.pi / 2], [1], color="red", zorder=3, label="最高点")  # 散点：标出峰
    _ax.set_title("matplotlib 第一次握手")       # 标题（中文字体已在 import 单元修好）
    _ax.set_xlabel("x")                        # x 轴标签
    _ax.set_ylabel("sin(x)")                   # y 轴标签
    _ax.legend()                               # 图例：自动读取上面写的 label=
    _ax.grid(True, alpha=0.3)                  # 网格：alpha=透明度，淡淡的就好
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 1 · 画出向量

        本课主角登场：a = (3, 2) 和 b = (1, 3)。画箭头用 `ax.quiver(起点x, 起点y, dx, dy)`，
        三个参数要背下来：`angles="xy", scale_units="xy", scale=1`——意思是
        "箭头长度按**真实坐标**画，不要自动缩放"（不写这三件，箭头长度会乱套）。

        顺手把两支箭头的长度（范数）和方向角标在旁边——从这一刻起，
        "分量、范数、方向"三个词都看得见了。
        """
    )
    return


@app.cell
def _(np, plt):
    # 演示 1 · 画出两支箭头
    _a = np.array([3.0, 2.0])                  # 向量 a：向右 3、向上 2
    _b = np.array([1.0, 3.0])                  # 向量 b：向右 1、向上 3

    _len_a = np.linalg.norm(_a)                # |a| = √13（09 课的范数）
    _len_b = np.linalg.norm(_b)                # |b| = √10
    _ang_a = np.degrees(np.arctan2(_a[1], _a[0]))   # a 的方向角：arctan2(y, x)，y 在前！
    _ang_b = np.degrees(np.arctan2(_b[1], _b[0]))   # b 的方向角

    _fig, _ax = plt.subplots(figsize=(6, 5.5))
    # 画箭头：quiver(0, 0, dx, dy) = 从原点出发；三件套参数保证按真实坐标画
    _ax.quiver(0, 0, _a[0], _a[1], angles="xy", scale_units="xy", scale=1,
               color="tab:red", width=0.012, label="a = (3, 2)")
    _ax.quiver(0, 0, _b[0], _b[1], angles="xy", scale_units="xy", scale=1,
               color="tab:blue", width=0.012, label="b = (1, 3)")

    # 在箭头旁边标注长度与方向角（放箭头中点再往外挪一点）
    _ax.text(_a[0] / 2 + 0.15, _a[1] / 2 + 0.15,
             f"|a| = √13 ≈ {_len_a:.2f}\n方向角 {_ang_a:.1f}°", color="tab:red")
    _ax.text(_b[0] / 2 - 0.15, _b[1] / 2 + 0.15,
             f"|b| = √10 ≈ {_len_b:.2f}\n方向角 {_ang_b:.1f}°", color="tab:blue",
             ha="right")

    _ax.set_xlim(-1, 4)                        # x 轴范围：留一点边距
    _ax.set_ylim(-1, 4)                        # y 轴范围
    _ax.set_aspect("equal")                    # 等比例！不设它，直角会看起来像斜角
    _ax.grid(True, alpha=0.3)
    _ax.axhline(0, color="gray", lw=0.8)       # 画出 x 轴（y=0 的横线）
    _ax.axvline(0, color="gray", lw=0.8)       # 画出 y 轴（x=0 的竖线）
    _ax.legend(loc="upper right")
    _ax.set_title("两支箭头：分量是坐标，范数是长度")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 2 · 夹角：两条路算出同一个点积

        代数路：`a @ b` 直接算，得 9。几何路：|a||b|cosθ——先算 cosθ，
        再用 `np.arccos`（反余弦，输入 [-1, 1]，输出弧度）拿夹角，最后
        `np.degrees` 把弧度换成度。

        灰色的弧就是把夹角"画"出来：`Arc(圆心, 宽, 高, theta1=起点角, theta2=终点角)`，
        角度单位是**度**、从 x 正轴逆时针数。
        """
    )
    return


@app.cell
def _(np, plt):
    # 演示 2 · 两条路算同一个点积：a@b 与 |a||b|cosθ
    from matplotlib.patches import Arc as _Arc   # 画弧的小工具（只在本单元用，_ 前缀）

    _a = np.array([3.0, 2.0])
    _b = np.array([1.0, 3.0])

    _dot = float(_a @ _b)                            # 代数路：逐位相乘再求和 → 9.0
    _na = np.linalg.norm(_a)                         # |a| ≈ 3.606
    _nb = np.linalg.norm(_b)                         # |b| ≈ 3.162
    _cos = _dot / (_na * _nb)                        # cosθ = 点积 / 长度乘积 ≈ 0.789
    _theta = np.degrees(np.arccos(_cos))             # θ = arccos(cos) ≈ 37.9°（弧度→度）

    _ang_a = np.degrees(np.arctan2(_a[1], _a[0]))    # a 的方向角 ≈ 33.7°
    _ang_b = np.degrees(np.arctan2(_b[1], _b[0]))    # b 的方向角 ≈ 71.6°

    _fig, _ax = plt.subplots(figsize=(6, 5.5))
    _ax.quiver(0, 0, _a[0], _a[1], angles="xy", scale_units="xy", scale=1,
               color="tab:red", width=0.012, label="a")
    _ax.quiver(0, 0, _b[0], _b[1], angles="xy", scale_units="xy", scale=1,
               color="tab:blue", width=0.012, label="b")

    # 夹角弧：从 a 的方向角画到 b 的方向角，半径 0.9（宽高 = 2×半径）
    _ax.add_patch(_Arc((0, 0), 1.8, 1.8, theta1=_ang_a, theta2=_ang_b,
                       color="dimgray", lw=1.5))
    # θ 字母放在弧的中间角度、稍微往外一点的地方
    _mid_ang = np.radians((_ang_a + _ang_b) / 2)
    _ax.text(1.18 * np.cos(_mid_ang), 1.18 * np.sin(_mid_ang), "θ", fontsize=14)

    _ax.set_xlim(-1, 4)
    _ax.set_ylim(-1, 4)
    _ax.set_aspect("equal")
    _ax.grid(True, alpha=0.3)
    _ax.axhline(0, color="gray", lw=0.8)
    _ax.axvline(0, color="gray", lw=0.8)
    _ax.legend(loc="upper right")
    _ax.set_title(f"夹角 θ ≈ {_theta:.1f}°：两条路算出同一个点积")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 3 · 投影：b 在 a 方向上的"影子"

        把 a 想成一束斜射的阳光方向，b 的**投影**就是 b 落在这条线上的影子：

        - 影子长度 = |b|·cosθ ≈ 3.16 × 0.789 ≈ 2.50（**带正负号**：顺着 a 为正、反着为负）
        - 点积 = |a| × 影子长度 ≈ 3.61 × 2.50 ≈ 9.0——和 `a @ b` 又对上了！

        影子本身也是一个向量（有方向有长度），公式一步到位：

        ```
        proj(b → a) = (a·b / |a|²) × a
        ```

        图里：绿色虚线是影子，灰色点线是 b 到影子的**垂线**——
        它和 a 垂直，这正是"投影"定义里那个直角。
        """
    )
    return


@app.cell
def _(np, plt):
    # 演示 3 · 投影：b 在 a 方向上的影子
    _a = np.array([3.0, 2.0])
    _b = np.array([1.0, 3.0])

    _dot = float(_a @ _b)                          # a·b = 9
    _na = np.linalg.norm(_a)                       # |a| ≈ 3.606
    _nb = np.linalg.norm(_b)                       # |b| ≈ 3.162

    _proj_len = _dot / _na                         # 影子长度（带符号）= 9 / 3.606 ≈ 2.50
    _proj_vec = (_dot / (_na * _na)) * _a          # 影子向量 = (a·b / |a|²) × a

    _fig, _ax = plt.subplots(figsize=(6, 5.5))
    # a 方向的整条延长线（点在 (0,0) 和 4a 之间）：影子落在这条线上
    _ax.plot([0, 4 * _a[0]], [0, 4 * _a[1]], color="tab:red", lw=1, alpha=0.35)
    _ax.quiver(0, 0, _a[0], _a[1], angles="xy", scale_units="xy", scale=1,
               color="tab:red", width=0.012, label="a")
    _ax.quiver(0, 0, _b[0], _b[1], angles="xy", scale_units="xy", scale=1,
               color="tab:blue", width=0.012, label="b")

    # 投影向量：绿色虚线段（plot 画线段：起点终点两组坐标）
    _ax.plot([0, _proj_vec[0]], [0, _proj_vec[1]], "--", color="tab:green", lw=2.5,
             label="b 的投影（影子）")
    # 垂线：从 b 的箭头尖垂到影子上——灰色点线
    _ax.plot([_b[0], _proj_vec[0]], [_b[1], _proj_vec[1]], ":", color="gray", lw=1.5)
    # 垂足：影子与垂线的交点，画个小圆点
    _ax.scatter([_proj_vec[0]], [_proj_vec[1]], s=35, color="tab:green", zorder=3)

    _ax.text(_proj_vec[0] / 2 - 0.1, _proj_vec[1] / 2 - 0.35,
             f"影子长度 ≈ {_proj_len:.2f}", color="tab:green", ha="center")
    _ax.text(_b[0] + 0.1, _b[1] + 0.1, "b", color="tab:blue", fontsize=13)

    _ax.set_xlim(-1, 4)
    _ax.set_ylim(-1, 4)
    _ax.set_aspect("equal")
    _ax.grid(True, alpha=0.3)
    _ax.axhline(0, color="gray", lw=0.8)
    _ax.axvline(0, color="gray", lw=0.8)
    _ax.legend(loc="upper right")
    _ax.set_title(f"点积 = |a| × 影子长度 ≈ {_na:.2f} × {_proj_len:.2f} ≈ {_na * _proj_len:.1f}")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 4 · 余弦相似度矩阵：迷你语义搜索

        给四个词各造一个 4 维"词向量"（水果味 / 甜度 / 会动 / 黏人，0~5 打分）——
        第 7 部分给注意力打分时，query 和 key 就是这样的向量，这是那套玩法的
        手工迷你版。然后**一次算出所有词两两的相似度**，
        正是第 09 课矩阵乘法的用武之地：

        - 点积矩阵：`X @ X.T`——(4, 4) @ (4, 4) 的转置相乘，第 i 行 j 列 = 第 i、j 个词的点积
        - 长度矩阵：`norms[:, None] * norms[None, :]`——第 07 课的 newaxis 造出 (4,1)×(1,4)，
          广播铺满成"两两长度乘积"表（第 06 课的外和模式！）
        - 相除：第 06 课的逐元素除法——余弦相似度矩阵到手

        最后用 `imshow` 把矩阵画成热力图（矩阵 → 彩色格子；16 课画概率地图时，
        还会回到"矩阵 → 颜色"这条思路）。
        """
    )
    return


@app.cell
def _(np, plt):
    # 演示 4 · 四个词的两两余弦相似度
    words = ["苹果", "香蕉", "猫", "狗"]          # 四个词
    # 每行一个词向量，4 个语义维度：[水果味, 甜度, 会动, 黏人]，0~5 打分
    words_vec = np.array([
        [5, 4, 0, 0],    # 苹果：水果味浓、比较甜、不会动、不黏人
        [5, 5, 0, 0],    # 香蕉：同上，更甜一点
        [0, 0, 5, 5],    # 猫：  没水果味、不甜、会动、黏人
        [0, 0, 5, 4],    # 狗：  同类
    ], dtype=float)

    # 三步走：
    _norms = np.linalg.norm(words_vec, axis=1)             # 每个词的长度 (4,)——08 课 axis=1
    _dots = words_vec @ words_vec.T                        # 两两点积 (4,4)——09 课 X@X.T
    words_sim = _dots / (_norms[:, None] * _norms[None, :])  # 广播相除——06+07 课联手

    # 热力图：矩阵 → 彩色格子
    _fig, _ax = plt.subplots(figsize=(5.8, 4.8))
    _im = _ax.imshow(words_sim, cmap="RdYlGn", vmin=0, vmax=1)   # 红绿配色，值域 0~1
    _ax.set_xticks(range(4), words)                       # x 轴刻度换成词名
    _ax.set_yticks(range(4), words)                       # y 轴刻度换成词名
    for _i in range(4):                                   # 每格标上数值（双循环标 16 格）
        for _j in range(4):
            _ax.text(_j, _i, f"{words_sim[_i, _j]:.2f}", ha="center", va="center",
                     fontsize=11)
    _fig.colorbar(_im, label="余弦相似度")                  # 右侧色条：颜色↔数值的对照表
    _ax.set_title("词向量相似度矩阵（手工迷你词嵌入）")
    _fig
    return (words, words_vec, words_sim)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        读图：**苹果-香蕉 ≈ 0.99、猫-狗 ≈ 0.99**（两对几乎满格），而**水果和动物之间全是 0.00**——
        正交 = 语义无关，第 09 课"三种走向"的第二种有了自己的用途。

        一行 `_dots / (_norms[:, None] * _norms[None, :])` 背后站着四课的功力：
        范数（09）、矩阵乘法（09）、newaxis（07）、广播（06）。这就是"稳扎稳打"的回报——
        新公式一点也不新，全是旧零件的组装。
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 🎛️ 交互实验

        **点积罗盘 2.0**——第 09 课罗盘的图形升级版。向量 a = (3, 2) 固定不动（红），
        你来调向量 b（蓝）：**方向角** 0~360°、**长度** 0.5~4。

        盯着右边的数值表做两个实验：

        1. 固定长度、只转角度：看 cosθ 和点积怎么随夹角起伏，垂直那一刻双双归零
        2. 固定角度、只拉长度：点积跟着变，**cosθ 纹丝不动**——"点积吃长度，余弦只看方向"
        """
    )
    return


@app.cell
def _(mo):
    # 交互控制：b 的方向角与长度（debounce=True：松手才重算，画面不闪烁）
    b_angle_slider = mo.ui.slider(0, 360, value=60, step=5, debounce=True,
                                  label="🎛️ 向量 b 的方向角（度，0~360）")
    b_len_slider = mo.ui.slider(0.5, 4, value=3, step=0.25, debounce=True,
                                label="🎛️ 向量 b 的长度（0.5~4）")
    mo.vstack([b_angle_slider, b_len_slider])
    return (b_angle_slider, b_len_slider)


@app.cell
def _(b_angle_slider, b_len_slider, mo, np, plt):
    # ===== 交互实验 · 点积罗盘 2.0（响应单元）=====

    _a = np.array([3.0, 2.0])                       # a 固定
    _theta_b = np.radians(b_angle_slider.value)     # 角度 → 弧度（三角函数只吃弧度）
    _r = b_len_slider.value                         # b 的长度
    _b = _r * np.array([np.cos(_theta_b), np.sin(_theta_b)])   # 极坐标 → 直角坐标

    _dot = float(_a @ _b)                           # 点积
    _na = np.linalg.norm(_a)                        # |a|
    _nb = np.linalg.norm(_b)                        # |b|
    _cos = _dot / (_na * _nb)                       # 余弦相似度
    _theta = np.degrees(np.arccos(_cos))            # 夹角（度）
    _proj_len = _dot / _na                          # b 在 a 上的影子长度
    _proj_vec = (_dot / (_na * _na)) * _a           # 影子向量

    _ang_a = np.degrees(np.arctan2(_a[1], _a[0]))   # a 的方向角
    _ang_b = b_angle_slider.value                   # b 的方向角（就是滑块值）

    _fig, _ax = plt.subplots(figsize=(6.5, 5.8))
    _ax.plot([0, 4.5 * _a[0] / _na], [0, 4.5 * _a[1] / _na], color="tab:red", lw=1, alpha=0.3)
    _ax.quiver(0, 0, _a[0], _a[1], angles="xy", scale_units="xy", scale=1,
               color="tab:red", width=0.012, label="a = (3, 2) 固定")
    _ax.quiver(0, 0, _b[0], _b[1], angles="xy", scale_units="xy", scale=1,
               color="tab:blue", width=0.012, label="b（你来调）")

    # 影子（绿色虚线）与垂线（灰色点线）：长度太小时影子可能缩成一点，照样画
    _ax.plot([0, _proj_vec[0]], [0, _proj_vec[1]], "--", color="tab:green", lw=2.5)
    _ax.plot([_b[0], _proj_vec[0]], [_b[1], _proj_vec[1]], ":", color="gray", lw=1.2)

    # 夹角弧：只画 a、b 之间不超过 180° 的那段劣弧——和数值表里的 θ 永远一致
    _diff = (_ang_b - _ang_a) % 360                 # 从 a 逆时针转到 b 需要多少度
    if _diff <= 180:                                # 逆时针这段就是劣弧：从 a 画到 b
        _lo, _hi = _ang_a, _ang_a + _diff
    else:                                           # 否则劣弧在另一侧：从 b 画回 a
        _lo, _hi = _ang_b, _ang_b + (360 - _diff)
    from matplotlib.patches import Arc as _Arc
    _ax.add_patch(_Arc((0, 0), 1.6, 1.6, theta1=_lo, theta2=_hi, color="dimgray", lw=1.5))

    _ax.set_xlim(-4.5, 4.5)
    _ax.set_ylim(-4.5, 4.5)
    _ax.set_aspect("equal")
    _ax.grid(True, alpha=0.3)
    _ax.axhline(0, color="gray", lw=0.8)
    _ax.axvline(0, color="gray", lw=0.8)
    _ax.legend(loc="upper left")
    _ax.set_title("点积罗盘 2.0")

    # 一句话判词（cosθ 的五档解读）
    _verdict = (
        "几乎同向（cosθ ≈ +1）" if _cos > 0.95
        else "大体同向" if _cos > 0.3
        else "接近垂直（cosθ ≈ 0）" if _cos > -0.3
        else "大体相反" if _cos > -0.95
        else "几乎完全相反（cosθ ≈ -1）"
    )

    mo.vstack([
        _fig,
        mo.md(
            f"""
            | 量 | 值 |
            |---|---|
            | 点积 a·b | **{_dot:.3f}** |
            | 余弦相似度 cosθ | **{_cos:.3f}** |
            | 夹角 θ | **{_theta:.1f}°** |
            | b 在 a 上的影子长度 | **{_proj_len:.3f}** |
            """
        ),
        mo.callout(mo.md(f"**实时判词：{_verdict}**"), kind="info"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✏️ 动手练习（每课必做，本课 3 题）": mo.md(
                """
                **练习 1 · 手算夹角**：a = (1, 2)、b = (2, 1)。
                先在纸上算出点积、两个范数、cosθ 和 θ（度），再把答案填进代码里的 `None`。

                **练习 2 · 写你自己的余弦相似度函数**：`cosine_similarity(X)`，
                输入 (n, d) 的矩阵（每行一个向量），返回 (n, n) 的相似度矩阵。
                演示 4 已经示范过一遍，这次把三步拆开自己写——然后填进验证行。

                **练习 3 · 投影到坐标轴**：把 v = (4, 5) 分别投影到 x 轴和 y 轴上。
                算完你会发现"分量"这个词的几何来历。

                规则不变：先把 `None` 换成你的答案再运行；写完再看参考答案。
                """
            )
        }
    )
    return


@app.cell
def _(mo, np):
    # ===== 练习 1 · 手算夹角 =====
    a_ex1 = np.array([1.0, 2.0])
    b_ex1 = np.array([2.0, 1.0])

    # 手算提示：
    #   点积 a·b = 1×2 + 2×1 = ?
    #   |a| = √(1+4) = √5，|b| 也是 √5 → cosθ = 点积 / (√5×√5) = ?
    #   θ = np.degrees(np.arccos(你的 cos))

    cos_ex1 = None       # ← 把手算的 cosθ 填进来（一个 0~1 之间的数）
    theta_ex1 = None     # ← 把夹角填进来（度数，可以用 np.degrees(np.arccos(...)) 算）

    mo.md(
        f"""
        代码验证：`a @ b` = **{a_ex1 @ b_ex1}**（和你手算一致吗？）

        你的答案：cosθ = **{cos_ex1}**，θ = **{theta_ex1}** 度

        （填完 None 再展开参考答案核对；`None` 说明还没填。）
        """
    )
    return (a_ex1, b_ex1, cos_ex1, theta_ex1)


@app.cell
def _(mo, np, words_vec):
    # ===== 练习 2 · 余弦相似度函数 =====
    def cosine_similarity(X):
        """输入 (n, d)：每行一个向量。返回 (n, n)：第 i 行 j 列 = 第 i、j 个向量的 cosθ"""
        _norms = None     # ← 每行向量的范数，shape (n,)。提示：08 课，axis=？
        _dots = None      # ← 两两点积矩阵，shape (n, n)。提示：09 课，X @ ？
        S = None          # ← 相似度矩阵 = _dots / (两两长度乘积)。提示：07 课 newaxis 造 (n,1)×(1,n)
        return S

    # 验证（先补完上面的函数，再把这行的 None 换成 cosine_similarity(words_vec)）：
    #   期望：对角线全是 1、矩阵对称、苹果-香蕉 ≈ 0.99
    S_ex2 = None

    _msg = (
        f"对角线 = {np.diag(S_ex2).round(3)}（应全为 1），对称 = {np.allclose(S_ex2, S_ex2.T)}，\n"
        f"苹果-香蕉 = {S_ex2[0, 1]:.3f}（应 ≈ 0.994）"
        if S_ex2 is not None
        else "⏳ 先把函数里的三步补完，再把 S_ex2 的 None 换成 cosine_similarity(words_vec)"
    )
    mo.md(_msg)
    return (cosine_similarity, S_ex2)


@app.cell
def _(mo, np):
    # ===== 练习 3 · 投影到坐标轴 =====
    v_ex3 = np.array([4.0, 5.0])
    e_x = np.array([1.0, 0.0])     # x 轴方向的单位向量
    e_y = np.array([0.0, 1.0])     # y 轴方向的单位向量

    # 投影公式：proj(v → u) = (v·u / u·u) × u
    # 把 v 分别投到 e_x 和 e_y 上：
    proj_x_ex3 = None   # ← v 投到 x 轴：结果是一个 2 维数组
    proj_y_ex3 = None   # ← v 投到 y 轴

    _msg = (
        f"v 投到 x 轴 = **{proj_x_ex3}**，投到 y 轴 = **{proj_y_ex3}**"
        if proj_x_ex3 is not None and proj_y_ex3 is not None
        else "⏳ 用投影公式把两个 None 补完（提示：e·e = 1，所以分母不用愁）"
    )
    mo.md(_msg)
    return (v_ex3, e_x, e_y, proj_x_ex3, proj_y_ex3)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 1 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                cos_ex1 = np.dot(a_ex1, b_ex1) / (np.linalg.norm(a_ex1) * np.linalg.norm(b_ex1))
                theta_ex1 = np.degrees(np.arccos(cos_ex1))
                ```

                手算过程：点积 = 2 + 2 = **4**；|a| = |b| = **√5**；
                cosθ = 4 / (√5 × √5) = 4/5 = **0.8**；θ = arccos(0.8) ≈ **36.87°**。

                彩蛋：|a| × |b| = √5 × √5 = **5**，点积 = **4**——
                这是一张藏好的 **3-4-5 直角三角形**：影子长 = 4/√5、垂线段 = 3/√5、
                斜边 = |b| = √5，三边同乘 √5 正好是 3、4、5
                （勾股定理的老朋友，09 课验算范数时见过它）。
                手算时遇到"整好除尽"的余弦，多半不是巧合，是出题人挑的数。
                """
            ),
            "✅ 练习 2 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                def cosine_similarity(X):
                    _norms = np.linalg.norm(X, axis=1)        # (n,)：每行的范数——08 课 axis=1
                    _dots = X @ X.T                           # (n, n)：两两点积——09 课
                    S = _dots / (_norms[:, None] * _norms[None, :])  # 06+07：newaxis + 广播
                    return S
                ```

                三个零件的"回收清单"：

                - `axis=1`：沿列方向压掉列，每行剩一个数（08 课"消灭那个轴"）
                - `X @ X.T`：自己和自己的转置相乘，第 i 行 j 列正好是第 i、j 个向量的点积（09 课）
                - `_norms[:, None] * _norms[None, :]`：(n,1) 乘 (1,n) 广播出 (n,n) 的
                  "两两长度乘积"表（06 课外和 + 07 课 newaxis）

                对角线自动 = 1（自己除以自己的长度平方），不用手动填——浮点上会是
                0.9999…，这就是参考答案里用 `np.allclose` 而不是 `==` 的原因（09 课浮点噪声）。
                """
            ),
            "✅ 练习 3 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                proj_x_ex3 = (v_ex3 @ e_x / (e_x @ e_x)) * e_x    # = 4 × (1, 0) = (4, 0)
                proj_y_ex3 = (v_ex3 @ e_y / (e_y @ e_y)) * e_y    # = 5 × (0, 1) = (0, 5)
                ```

                **(4, 0) 和 (0, 5)**——正是 v = (4, 5) 的两个分量。

                这不是巧合，是"**分量**"（component）一词的几何定义：
                v 的第 k 个分量 = v 投影到第 k 根坐标轴上的影子长度。
                你从第 01 课就开始写的 `v[0]`、`v[1]`，今天终于看到了它们的几何身世。
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
                1. 向量 = 箭头：分量是坐标、范数（`np.linalg.norm`）是长度、
                   `np.arctan2(y, x)` 是方向角（y 在前，能分辨四个象限）
                2. 点积的几何真身：**a·b = |a||b|cosθ = |a| × 影子长度**——
                   三种走向（正、零、负）= 影子顺着、缩成点、反着投
                3. 投影向量一步到位：`proj = (a·b / |a|²) × a`；投影到坐标轴 = 取出分量
                4. 余弦相似度 = **归一化点积**：只比方向不比长短，值域 [-1, 1]；
                   +1 同向、0 垂直无关、-1 相反——"这俩向量像不像"的默认答案
                5. matplotlib 三招（plot 折线 / scatter 散点 / quiver 箭头）+ fig/ax 二件套 +
                   几何图必设 `set_aspect("equal")`；中文乱码用 `font.sans-serif` 候选名单修
                6. 词向量相似度矩阵一行搞定：`X @ X.T / (norms[:,None] * norms[None,:])`——
                   06/07/08/09 四课零件的总装；第 7 部分讲注意力时它还回来
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

        **下一课预告** — `12_矩阵即变换`：

        今天，单个向量有了画面。下一课，让**整个空间动起来**。

        矩阵在你手里曾是"数字表格"；几何视角下它是一份**操作手册**——
        每一列写着"一根基向量被送往哪里"，整个空间便随之旋转、拉伸、剪切。
        旋转 90° 是矩阵、放大两倍是矩阵、镜像翻转也是矩阵；
        第 09 课的 `X @ W`，几何上就是"把一批点送进同一台变形机器"。

        你甚至能看到"矩阵乘法不可交换"（AB ≠ BA）的画面版解释：
        先转再拉 ≠ 先拉再转。

        🧭 箭头已就位，去把空间拧一拧！
        """
    )
    return


if __name__ == "__main__":
    app.run()
