import marimo

__generated_with = "0.25.2"
app = marimo.App(app_title="12 · 矩阵即变换", width="medium")


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

    # ==== 中文乱码修复（第 11 课配好的老三样，逐字照搬）====
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
        # 12 · 矩阵即变换

        **从零到 Transformer · 手写深度学习** ｜ 第 2 部分 · 深度学习的数学（2/4）
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📖 本课目标（先读我）": mo.md(
                """
                **这一课只解决一个问题：矩阵到底是什么？**

                第 11 课让"向量"长出了画面：一支支有方向、有长度的箭头。课尾埋了一句预告——
                **矩阵是一份操作手册：每一列写着"一根基向量被送往哪里"**，整个空间随之
                旋转、拉伸、剪切；`X @ W` 就是把一批点送进同一台变形机器；甚至能看到
                AB ≠ BA 的画面版。今天，逐字兑现这些承诺。

                学完你将能：

                - 把 `M @ v` 读成**一次空间变换**，说出线性变换的两条铁律：
                  原点是钉子、直线保持直线且平行线保持平行
                - 用**列视角**一眼看穿一个矩阵：第 k 列 = 第 k 根基向量的去向（本课灵魂）
                - 把复合变换读成矩阵相乘：`A @ B` **从右往左读**（先 B 后 A），
                  并亲眼确认 AB ≠ BA——先转再拉 ≠ 先拉再转
                - 给第 09 课的 `X @ W + b` 画上画面：一层网络 = 一次空间变形，
                  W 负责变形（旋转 / 拉伸 / 剪切），b 负责平移

                前置：第 09 课的矩阵乘法与转置（本课地基）、第 11 课的 quiver 画箭头与
                set_aspect("equal")（工具直接复用，不再重讲）。
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "🎯 概念 1/4 · 矩阵乘向量 = 一次空间变换": mo.md(
                """
                第 09 课里，`M @ v` 是一行数字游戏：逐位相乘再相加。几何视角下，
                它是一次**搬家**：矩阵 M 是一台"变形机器"，把向量 v 从原地送到新位置。

                但机器不能随便拧——**线性变换**有两条铁律：

                - **原点是钉子**：无论怎么旋转拉伸，原点纹丝不动（所以纯矩阵做不出
                  "平移"，想平移要请 b 帮忙——概念 4 见）
                - **直线变直线，平行线保持平行**：网格线再怎么被拉歪，也不会弯成弧线、
                  不会变得不平行——方格最多被拧成平行四边形

                数值预告（可以自己动手验：`M @ np.array([2, 1])`）：剪切矩阵 [[1, 1], [0, 1]]
                作用在 v = (2, 1) 上，M @ v = (3, 1)——这台机器把"y 越高的点向右推得越远"，
                (2, 1) 被推到 (3, 1)。演示 1 画的是整张网格被这台机器剪切的模样。
                """
            ),
            "📐 概念 2/4 · 列视角：第 k 列 = 第 k 根基向量的去向（本课灵魂）": mo.md(
                """
                平面里所有向量都能拆成两根**基向量**的组合：

                ```
                v = (3, 2) = 3·î + 2·ĵ        （î=(1,0) 向右，ĵ=(0,1) 向上）
                ```

                线性变换有个惊人的省事之处：**只要盯住 î 和 ĵ 去了哪，整个空间就全知道了**。
                因为变换保持"拆开再组装"的关系——

                ```
                M @ v = v[0] × (M 的第 0 列) + v[1] × (M 的第 1 列)
                ```

                也就是说：v 变换后 = 3 份"î 的新家" + 2 份"ĵ 的新家"。而矩阵正是把
                两个新家**并排写成一摞**——第 0 列是 î 的去向，第 1 列是 ĵ 的去向。

                四台经典机器的"列视角速查"（演示 2 逐个画）：

                | 矩阵 | 第 0 列 = î 去向 | 第 1 列 = ĵ 去向 | 整体效果 |
                |---|---|---|---|
                | [[0, -1], [1, 0]] | (0, 1)：转到 +y | (-1, 0)：转到 -x | 逆时针旋转 90° |
                | [[2, 0], [0, 2]] | (2, 0)：拉长 | (0, 2)：拉长 | 均匀放大 2 倍 |
                | [[1, 1], [0, 1]] | (1, 0)：不动 | (1, 1)：被推歪 | 剪切（沿 x 方向推） |
                | [[1, 0], [0, -1]] | (1, 0)：不动 | (0, -1)：翻下去 | 沿 x 轴镜像 |

                从此看矩阵别再一行行读数字——眯起眼看它的**两列**，画面自动浮现。
                """
            ),
            "⛓️ 概念 3/4 · 复合变换 = 矩阵相乘（从右往左读）": mo.md(
                """
                两台机器串联：一个点先过 B、再过 A。惊人的事实是——**串联后的整体
                仍是一台线性机器，而且它的手册恰好是 A @ B**。矩阵乘法被发明出来，
                本来就是为了给"复合变换"记账。

                - **从右往左读**：A @ B 作用在点上 = 先做 B，再做 A——和函数嵌套
                  f(g(x)) 一模一样（g 在右边，先执行）
                - **AB ≠ BA**：先转再拉 ≠ 先拉再转。穿衣服的顺序有时无所谓，
                  拧空间的方向从来都有所谓——演示 3 亲眼所见
                - 呼应第 09 课的形状口诀"**内维相消，外维成形**"：(n, k) @ (k, m) → (n, m)
                  说的是"两台机器要能串联，出口维度必须对上入口维度"——
                  形状规则不是官僚规定，是流水线的物理现实

                一句话：**矩阵相乘 = 变换的接力**，而接力的棒次（顺序）就写在乘法的次序里。
                """
            ),
            "🧠 概念 4/4 · 深度学习里的变换：一层网络 = 一次变形": mo.md(
                """
                第 09 课的高潮是一行公式：`z = X @ W + b`。当时它只是"批量加权求和"；
                今天它长出画面了：

                - **X @ W = 把一批点送进同一台变形机器**：X 的每行是一个点（一个样本），
                  W 决定机器怎么拧——旋转、拉伸、剪切，全看 W 里写着的"基向量新家"
                  （样本按行存时看 W 的行，和列视角差一次转置，演示 5 细说）
                - **b = 平移**：线性变换"原点钉死"的铁律不许平移，b 就是来补这一手的——
                  整片点云被同一支箭头推走（第 06 课的广播把它加到每个样本上）
                - **多层网络 = 多台机器串联**：X @ W1 的输出再过 W2、再过 W3……
                  第 09 课的矩阵连乘，画面上是变形的接力——**"深度"就是空间被变形的次数**。
                  （预告：层与层之间还有**激活函数**把空间掰弯——不然纯线性串联会塌缩成
                  一台机器，这是第 19 课的故事）

                第 20 课我们亲手把这台机器搭成真正的全连接层；第 7 部分的 Transformer 里
                满屏的 @，每一次都是"把空间拧向对下一层更顺手的方向"。今天先看懂一台机器。
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

        五组演示，从"一张网格"一路走到"一层网络"：

        - **演示 1** · 变换前 vs 变换后：一张网格的剪切之旅
        - **演示 2** · 标准变换陈列馆：旋转 / 缩放 / 剪切 / 镜像
        - **演示 3** · 复合变换：A @ B 从右往左读，AB ≠ BA 亲眼所见
        - **演示 4** · 把图形送进变换：字母 L 与面积缩放系数 det
        - **演示 5** · 一层网络的几何：X @ W + b 的画面版收官
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 1 · 变换前 vs 变换后

        本课主角 M = [[1, 1], [0, 1]]——剪切机器。左图是变换前的世界：
        9×9 的网格线（浅灰）+ 两根基向量（灰色箭头：向右的 x 基向量、向上的 y 基向量）。
        右图是同一张网格过完机器 M 之后的模样（绿色），叠着两根新箭头——
        **它们正是 M 的两个列**：第 0 列 (1, 0) 说"x 基向量原地不动"，
        第 1 列 (1, 1) 说"y 基向量被推歪 45°"。

        读图：竖线族全被推成 45° 斜线，横线族只是各自向右平移——但所有格子
        仍是平行四边形（铁律二：平行线保持平行）。

        画网格的小机关：每条线采 33 个点、整条线一起 `@ M.T` 再连线——
        这样就算变换把直线掰弯也画得出来（线性变换掰不弯，但这是通用的保险写法）。
        """
    )
    return


@app.cell
def _(np, plt):
    # 演示 1 · 变换前 vs 变换后：一张网格的剪切之旅
    _M = np.array([[1.0, 1.0],       # 第 0 列 (1, 0)：x 基向量的新家——原地不动
                   [0.0, 1.0]])      # 第 1 列 (1, 1)：y 基向量的新家——被推歪 45°

    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4.5))
    _ax0, _ax1 = _axes               # 左：变换前；右：变换后

    # ---- 左图：变换前的世界（方方正正的网格）----
    for _t in np.linspace(-2, 2, 9):                                    # 9 个刻度
        _ax0.plot([-2, 2], [_t, _t], color="0.75", lw=0.8, alpha=0.6)   # 横线 y = t
        _ax0.plot([_t, _t], [-2, 2], color="0.75", lw=0.8, alpha=0.6)   # 竖线 x = t
    # 两根基向量（灰色箭头；quiver 三件套参数 = 按真实坐标画，第 11 课讲过）
    _ax0.quiver(0, 0, 1, 0, angles="xy", scale_units="xy", scale=1,
                color="0.35", width=0.012, label="x 基向量")
    _ax0.quiver(0, 0, 0, 1, angles="xy", scale_units="xy", scale=1,
                color="0.55", width=0.012, label="y 基向量")

    # ---- 右图：过完机器 M 的同一张网格 ----
    _ts = np.linspace(-2, 2, 33)                     # 每条线采 33 个点（保险写法）
    for _t in np.linspace(-2, 2, 9):
        _h = np.stack([_ts, np.full_like(_ts, _t)], axis=1)   # 横线 y=t 上的点，每行一个
        _v = np.stack([np.full_like(_ts, _t), _ts], axis=1)   # 竖线 x=t 上的点
        _ht = _h @ _M.T                                       # 整条横线一起过机器
        _vt = _v @ _M.T                                       # 整条竖线一起过机器
        _ax1.plot(_ht[:, 0], _ht[:, 1], color="tab:green", lw=0.9, alpha=0.55)
        _ax1.plot(_vt[:, 0], _vt[:, 1], color="tab:green", lw=0.9, alpha=0.55)

    # 原基向量的旧位置（淡灰幽灵）+ 新基向量（= M 的两列！）
    _ax1.quiver(0, 0, 1, 0, angles="xy", scale_units="xy", scale=1,
                color="0.35", width=0.010, alpha=0.35)
    _ax1.quiver(0, 0, 0, 1, angles="xy", scale_units="xy", scale=1,
                color="0.55", width=0.010, alpha=0.35)
    _ax1.quiver(0, 0, _M[0, 0], _M[1, 0], angles="xy", scale_units="xy", scale=1,
                color="tab:red", width=0.012, zorder=3,
                label="x 基向量新家 = 第 0 列 (1, 0)")
    _ax1.quiver(0, 0, _M[0, 1], _M[1, 1], angles="xy", scale_units="xy", scale=1,
                color="tab:blue", width=0.012, zorder=3,
                label="y 基向量新家 = 第 1 列 (1, 1)")

    # 两图共用的收尾设置（几何图必须 aspect equal，第 11 课的教训）
    for _ax in (_ax0, _ax1):
        _ax.set_xlim(-4.3, 4.3)          # 剪切把最右边的点推到 x=4，范围留足
        _ax.set_ylim(-4.3, 4.3)
        _ax.set_aspect("equal")
        _ax.axhline(0, color="gray", lw=0.8)      # x 轴
        _ax.axvline(0, color="gray", lw=0.8)      # y 轴
    _ax0.set_title("变换前：方方正正的网格")
    _ax1.set_title("变换后：M = [[1, 1], [0, 1]] 剪切")
    _ax0.legend(loc="upper right")
    _ax1.legend(loc="upper left")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 2 · 标准变换陈列馆

        四台经典机器，各押着一个**单位方形**（顶点 (0,0)、(1,0)、(1,1)、(0,1)）：

        - **旋转 90°** [[0, -1], [1, 0]]：整块方方正正地转过去
        - **放大 2 倍** [[2, 0], [0, 2]]：从原点向外均匀膨胀
        - **剪切** [[1, 1], [0, 1]]：底边不动、顶边被推走，方块变平行四边形
        - **沿 x 轴镜像** [[1, 0], [0, -1]]：沿 x 轴翻个面，图形进了"镜子里"

        读图约定：灰色是变换前的单位方形，橙色是变换后；红箭头 = M 第 0 列
        （x 基向量的新家），蓝箭头 = M 第 1 列（y 基向量的新家）——
        **先看箭头、再看方块**，列视角就是这么用的。

        代码小机关：四个顶点按行存成 (4, 2)，`pts @ M.T` 一次全变完。
        为什么有个 `.T`？点按**行**存、手册按**列**读——转置就是两种记法之间的
        翻译官（第 09 课认识的老朋友，练习 3 还要亲手用它）。
        """
    )
    return


@app.cell
def _(np, plt):
    # 演示 2 · 标准变换陈列馆：四台经典机器，各押一个单位方形
    _mats = [
        ("旋转 90°",    np.array([[0.0, -1.0], [1.0, 0.0]])),   # î 转到 +y，ĵ 转到 -x
        ("放大 2 倍",   np.array([[2.0, 0.0], [0.0, 2.0]])),    # 两根基向量都拉长 2 倍
        ("剪切",        np.array([[1.0, 1.0], [0.0, 1.0]])),    # î 不动，ĵ 被推歪
        ("沿 x 轴镜像", np.array([[1.0, 0.0], [0.0, -1.0]])),   # î 不动，ĵ 翻到下面
    ]
    # 单位方形的四个顶点，每行一个点（第 09 课 X 的"行=样本"记法）
    _sq = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

    _fig, _axes = plt.subplots(2, 2, figsize=(10, 9))
    for _ax, (_name, _Mm) in zip(_axes.flat, _mats):
        _sq_t = _sq @ _Mm.T                        # 四个顶点一次过机器（行存点：@ M.T）
        _ax.fill(_sq[:, 0], _sq[:, 1], color="0.8", alpha=0.35,
                 edgecolor="0.5", lw=1.0)           # 原单位方形（淡灰）
        _ax.fill(_sq_t[:, 0], _sq_t[:, 1], color="tab:orange", alpha=0.30,
                 edgecolor="tab:orange", lw=1.8)    # 变换后（橙）
        # 两支新基向量箭头 = M 的两列（红 = 第 0 列，蓝 = 第 1 列）
        _ax.quiver(0, 0, _Mm[0, 0], _Mm[1, 0], angles="xy", scale_units="xy", scale=1,
                   color="tab:red", width=0.011, zorder=3)
        _ax.quiver(0, 0, _Mm[0, 1], _Mm[1, 1], angles="xy", scale_units="xy", scale=1,
                   color="tab:blue", width=0.011, zorder=3)
        # 子图标题把矩阵原文写出来，方便对照
        _ax.set_title(
            f"{_name}：[[{_Mm[0, 0]:.0f}, {_Mm[0, 1]:.0f}], [{_Mm[1, 0]:.0f}, {_Mm[1, 1]:.0f}]]"
        )
        _ax.set_xlim(-2.6, 2.6)
        _ax.set_ylim(-2.6, 2.6)
        _ax.set_aspect("equal")
        _ax.grid(True, alpha=0.25)
        _ax.axhline(0, color="gray", lw=0.8)
        _ax.axvline(0, color="gray", lw=0.8)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 3 · 复合变换：A @ B 从右往左读

        M1 = 旋转 90°，M2 = x 方向拉伸 2 倍。两台机器有两种接法：

        - **M1 @ M2**：从右往左读——先拉伸（橙色点线是中途站），再旋转（蓝色实心是终点）
        - **M2 @ M1**：先旋转（中途站），再拉伸（终点）

        两个终点都是面积 2 的长方形，但一个**竖高**（1 宽 2 高）、一个**横扁**
        （2 宽 1 高）——AB ≠ BA 不再是代数课的提醒，是你亲眼看到的两个不同画面。

        （"面积都是 2"这个观察先记着：面积缩放有个大名叫**行列式**，演示 4 马上讲。）
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # 演示 3 · 复合变换：A @ B 从右往左读——AB ≠ BA 亲眼所见
    _M1 = np.array([[0.0, -1.0],              # M1：逆时针旋转 90°
                    [1.0,  0.0]])
    _M2 = np.array([[2.0, 0.0],               # M2：x 方向拉伸 2 倍
                    [0.0, 1.0]])
    _sq = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])   # 单位方形

    _AB = _M1 @ _M2        # 合成手册一：先 M2（拉伸）再 M1（旋转）→ [[0, -1], [2, 0]]
    _BA = _M2 @ _M1        # 合成手册二：先 M1（旋转）再 M2（拉伸）→ [[0, -2], [1, 0]]
    _same = np.allclose(_AB, _BA)     # False！顺序换了，乘出的矩阵就换了

    # 两条路线各自的中途站与终点（行存点：pts @ M.T）
    _sq_m2 = _sq @ _M2.T       # 左图中途：先被拉成 2×1 的长方形
    _sq_ab = _sq @ _AB.T       # 左图终点：再整体旋转 90°（竖高条）
    _sq_m1 = _sq @ _M1.T       # 右图中途：先转 90°
    _sq_ba = _sq @ _BA.T       # 右图终点：再拉伸（横扁条）

    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4.5))

    def _draw_journey(_ax, _mid, _end, _title):
        """小助手：画原方形（灰虚线）→ 中途（橙点线）→ 终点（蓝实心）三段旅程"""
        _sq_c = np.vstack([_sq, _sq[:1]])       # 首点接到末尾，把多边形闭合
        _mid_c = np.vstack([_mid, _mid[:1]])
        _ax.plot(_sq_c[:, 0], _sq_c[:, 1], "--", color="0.5", lw=1.6, label="原单位方形")
        _ax.plot(_mid_c[:, 0], _mid_c[:, 1], ":", color="tab:orange", lw=2.4,
                 label="第 1 步后")
        _ax.fill(_end[:, 0], _end[:, 1], color="tab:blue", alpha=0.30,
                 edgecolor="tab:blue", lw=2.0, label="第 2 步后（最终）")
        _ax.set_title(_title)
        _ax.set_aspect("equal")
        _ax.grid(True, alpha=0.3)
        _ax.axhline(0, color="gray", lw=0.8)
        _ax.axvline(0, color="gray", lw=0.8)
        _ax.legend(loc="lower right")

    _draw_journey(_axes[0], _sq_m2, _sq_ab, "M1 @ M2：先拉伸，再旋转（从右往左读）")
    _draw_journey(_axes[1], _sq_m1, _sq_ba, "M2 @ M1：先旋转，再拉伸（从右往左读）")
    _axes[0].set_xlim(-1.6, 2.6)     # 左图内容：原方形 + 2×1 长方形 + 竖高终点
    _axes[0].set_ylim(-0.8, 2.7)
    _axes[1].set_xlim(-2.6, 1.6)     # 右图内容：原方形 + 旋转小方块 + 横扁终点
    _axes[1].set_ylim(-0.8, 1.9)

    mo.vstack(
        [
            _fig,
            mo.md(
                f"""
                两份"合成手册"（注意：乘出来的矩阵和 M1、M2 都长不一样——串联出的
                确实是一台**新机器**）：

                ```
                M1 @ M2 = [[{_AB[0, 0]:.0f}, {_AB[0, 1]:.0f}], [{_AB[1, 0]:.0f}, {_AB[1, 1]:.0f}]]    ← 先拉伸，再旋转
                M2 @ M1 = [[{_BA[0, 0]:.0f}, {_BA[0, 1]:.0f}], [{_BA[1, 0]:.0f}, {_BA[1, 1]:.0f}]]    ← 先旋转，再拉伸
                ```

                np.allclose(M1 @ M2, M2 @ M1) = **{_same}**——顺序不同，机器不同。

                左图终点竖高、右图终点横扁，面积却都是 2（原面积 1）：
                旋转不改变面积、拉伸让面积翻倍，谁先谁后都一样——
                但**形状**和**朝向**全变了。这就是 AB ≠ BA 的画面版。
                """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 4 · 把图形送进变换：字母 L 与面积缩放系数 det

        任意图形都一样送：把轮廓上的顶点按行存成 (n, 2)，`pts @ M.T` 整体过机器。
        这次送一个**字母 L** 进去，M = [[0.8, -0.6], [0.6, 0.8]]——

        - 它是纯旋转：cosθ = 0.8、sinθ = 0.6 → 逆时针转 **36.87°**
          （第 11 课练习 1 里那个 36.87° 的老朋友又回来了）
        - 顺手介绍一个**加分知识点**：det（行列式，`np.linalg.det`）= **面积缩放系数**。
          det = 1 → 面积不变（旋转、剪切都如此）；det = 4 → 胀成 4 倍；
          det < 0 → 图形被翻了面（镜像）。五台机器的 det 对照表见图下。

        图下还用"鞋带公式"实测了 L 变换前后的面积——比值恰好等于 det，
        一个字：准。
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # 演示 4 · 把图形送进变换：字母 L 的旋转 + det（面积缩放系数）
    _M = np.array([[0.8, -0.6],               # 纯旋转矩阵：cosθ = 0.8、sinθ = 0.6
                   [0.6,  0.8]])              #   → 逆时针转约 36.87°（11 课的老朋友）

    # 字母 L 的轮廓：6 个顶点按逆时针绕一圈（每行一个点，按顺序连线就是 L）
    _L = np.array([
        [0.0,  0.0],    # 左下
        [1.0,  0.0],    # 右下（底边右端）
        [1.0,  0.35],   # 底边右上拐点
        [0.35, 0.35],   # 竖条右壁下端
        [0.35, 1.0],    # 竖条右上
        [0.0,  1.0],    # 左上
    ])
    _Lt = _L @ _M.T                          # 整个 L 一次过机器（行存点：@ M.T）

    _det = np.linalg.det(_M)                 # = 1.0：面积不变
    _ang = np.degrees(np.arctan2(_M[1, 0], _M[0, 0]))   # 旋转角 ≈ 36.87°（arctan2(y, x)）

    # 加分小工具：鞋带公式量多边形面积（顶点按顺序绕一圈，相邻叉积求和取半）
    def _area(_pts):
        _s = 0.0
        for _i in range(len(_pts)):
            _j = (_i + 1) % len(_pts)        # 下一个顶点（末尾绕回首顶点）
            _s += _pts[_i, 0] * _pts[_j, 1] - _pts[_j, 0] * _pts[_i, 1]   # 叉积累加
        return 0.5 * abs(_s)

    _area_before = _area(_L)                 # 0.5775
    _area_after = _area(_Lt)                 # ≈ 0.5775（det = 1 → 面积应该不变）

    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # ---- 左图：变换前 ----
    _axes[0].fill(_L[:, 0], _L[:, 1], color="tab:blue", alpha=0.30,
                  edgecolor="tab:blue", lw=1.8)
    _axes[0].quiver(0, 0, 1, 0, angles="xy", scale_units="xy", scale=1,
                    color="0.4", width=0.010)                 # x 基向量
    _axes[0].quiver(0, 0, 0, 1, angles="xy", scale_units="xy", scale=1,
                    color="0.55", width=0.010)                # y 基向量
    _axes[0].text(1.05, -0.06, "x 基向量", fontsize=9, color="0.35")
    _axes[0].text(-0.44, 0.95, "y 基向量", fontsize=9, color="0.35")
    _axes[0].set_xlim(-0.6, 1.4)
    _axes[0].set_ylim(-0.35, 1.4)

    # ---- 右图：变换后（两支新箭头 = M 的两列）----
    _axes[1].fill(_Lt[:, 0], _Lt[:, 1], color="tab:orange", alpha=0.35,
                  edgecolor="tab:orange", lw=1.8)
    _axes[1].quiver(0, 0, _M[0, 0], _M[1, 0], angles="xy", scale_units="xy", scale=1,
                    color="tab:red", width=0.011, zorder=3,
                    label=f"第 0 列 ({_M[0, 0]:.1f}, {_M[1, 0]:.1f})")
    _axes[1].quiver(0, 0, _M[0, 1], _M[1, 1], angles="xy", scale_units="xy", scale=1,
                    color="tab:blue", width=0.011, zorder=3,
                    label=f"第 1 列 ({_M[0, 1]:.1f}, {_M[1, 1]:.1f})")
    _axes[1].set_xlim(-1.0, 1.3)
    _axes[1].set_ylim(-0.35, 1.4)

    for _ax in _axes:
        _ax.set_aspect("equal")
        _ax.grid(True, alpha=0.3)
        _ax.axhline(0, color="gray", lw=0.8)
        _ax.axvline(0, color="gray", lw=0.8)
    _axes[0].set_title(f"变换前：字母 L（面积 {_area_before:.4f}）")
    _axes[1].set_title(f"变换后：转过 {_ang:.2f}°，面积 {_area_after:.4f}")
    _axes[1].legend(loc="lower left")

    # det 对照表：一行一行现算现填，保证和图里的机器对得上
    _table_rows = "\n".join(
        f"| {_n} | **{np.linalg.det(_Dm):.2f}** | {_h} |"
        for _n, _Dm, _h in [
            ("旋转 90° [[0, -1], [1, 0]]", np.array([[0.0, -1.0], [1.0, 0.0]]), "不变"),
            ("放大 2 倍 [[2, 0], [0, 2]]", np.array([[2.0, 0.0], [0.0, 2.0]]), "胀成 4 倍"),
            ("剪切 [[1, 1], [0, 1]]", np.array([[1.0, 1.0], [0.0, 1.0]]), "不变（歪了但没胖）"),
            ("沿 x 轴镜像 [[1, 0], [0, -1]]", np.array([[1.0, 0.0], [0.0, -1.0]]), "翻面，面积不变"),
            ("本演示的 M", _M, "不变（纯转身）"),
        ]
    )

    mo.vstack(
        [
            _fig,
            mo.md(
                f"""
                **det（行列式，`np.linalg.det`）= 面积缩放系数**——一个数概括这台机器
                对面积的作为（加分知识点：记个直觉就够，后面课程用到再深挖）：

                | 变换 | det | 面积会怎样 |
                |---|---|---|
                {_table_rows}

                实测本演示：L 变换前面积 **{_area_before:.4f}**，变换后 **{_area_after:.4f}**，
                比值 **{_area_after / _area_before:.2f}**——和 det 分毫不差。

                det 还自带"方向感"：det < 0 说明图形被翻到了背面（镜像都这样）——
                就像纸片被掀了个面，原来的逆时针绕圈变成了顺时针。
                """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 5 · 一层网络的几何：X @ W + b 的画面版

        收官演示，回收第 09 课的高潮公式。40 个随机点当一批样本 X
        （种子固定，第 10 课的老规矩），W = [[1, 0.5], [0, 1]] 当这层网络的权重，
        b = (1, 2) 当偏置：

        - 左图：变换前的点云 X，两根灰色基向量
        - 右图：Z = X @ W + b——灰色小点是原位置（"幽灵"），橙色点是新位置，
          细灰箭头是 8 个样本的搬家路线，绿色大箭头是平移量 b

        一句话看懂一层网络：**W 负责变形，b 负责平移**。
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # 演示 5 · 一层网络的几何：X @ W + b 的画面版（09 课公式收官）
    _rng = np.random.default_rng(42)             # 第 10 课：种子固定，点云每次运行都一样
    _X = _rng.uniform(-1, 1, size=(40, 2))       # 40 个样本点，每行一个 (x, y)

    _W = np.array([[1.0, 0.5],                   # 权重：x 轴基向量的新家 (1, 0.5)（行视角）
                   [0.0, 1.0]])                  #        y 轴基向量原地不动 (0, 1)
    _b = np.array([1.0, 2.0])                    # 偏置：整片点云平移（右 1、上 2）

    _Z = _X @ _W + _b                            # (40,2) @ (2,2) + (2,) → (40,2)：09 课公式一行不改

    _idx = np.linspace(0, 39, 8).astype(int)     # 挑 8 个样本画"搬家路线"
    _d = _Z[_idx] - _X[_idx]                     # 每个样本的位移（起点 → 终点）

    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # ---- 左图：变换前的点云 ----
    _axes[0].scatter(_X[:, 0], _X[:, 1], s=22, color="tab:blue", alpha=0.75)
    _axes[0].quiver(0, 0, 1, 0, angles="xy", scale_units="xy", scale=1,
                    color="0.4", width=0.010)                 # x 基向量
    _axes[0].quiver(0, 0, 0, 1, angles="xy", scale_units="xy", scale=1,
                    color="0.55", width=0.010)                # y 基向量
    _axes[0].set_xlim(-1.7, 1.7)
    _axes[0].set_ylim(-1.7, 1.7)
    _axes[0].set_title("变换前：X（40 个样本点）")

    # ---- 右图：X @ W + b ----
    _axes[1].scatter(_X[:, 0], _X[:, 1], s=10, color="0.6", alpha=0.45)   # 原位置（幽灵）
    _axes[1].quiver(_X[_idx, 0], _X[_idx, 1], _d[:, 0], _d[:, 1],
                    angles="xy", scale_units="xy", scale=1,
                    color="0.45", width=0.005, alpha=0.85)    # 搬家路线
    _axes[1].scatter(_Z[:, 0], _Z[:, 1], s=22, color="tab:orange", alpha=0.85)   # 新位置
    _axes[1].quiver(0, 0, _W[0, 0], _W[0, 1], angles="xy", scale_units="xy", scale=1,
                    color="tab:red", width=0.011, zorder=3, label="x 基向量被抬高")   # W 第 0 行
    _axes[1].quiver(0, 0, _W[1, 0], _W[1, 1], angles="xy", scale_units="xy", scale=1,
                    color="tab:blue", width=0.011, zorder=3, label="y 基向量不动")    # W 第 1 行
    _axes[1].quiver(0, 0, _b[0], _b[1], angles="xy", scale_units="xy", scale=1,
                    color="tab:green", width=0.011, zorder=3, label="b = (1, 2) 平移")
    _axes[1].set_xlim(-1.5, 2.6)
    _axes[1].set_ylim(-1.2, 3.9)
    _axes[1].set_title("变换后：X @ W + b（剪切 + 平移）")

    for _ax in _axes:
        _ax.set_aspect("equal")
        _ax.grid(True, alpha=0.3)
        _ax.axhline(0, color="gray", lw=0.8)
        _ax.axvline(0, color="gray", lw=0.8)
    _axes[1].legend(loc="upper left")

    mo.vstack(
        [
            _fig,
            mo.md(
                f"""
                读图：**W 负责变形，b 负责平移。**

                - W = [[1, 0.5], [0, 1]]：x 轴基向量被抬到 (1, 0.5)、y 轴不动 →
                  点云被**剪切**——x 越大的点被抬得越高（看搬家路线的陡峭程度）
                - b = (1, 2)：整片点云**平移**——线性变换的"原点钉子"铁律不许平移，
                  b 就是来补这一手的
                - 形状账单：(40, 2) @ (2, 2) + (2,) → (40, 2)，内维相消外维成形（09 课）；
                  b 靠第 06 课的广播加到每一行

                抽样对账：第 0 个点 ({_X[0, 0]:.2f}, {_X[0, 1]:.2f}) →
                ({_Z[0, 0]:.2f}, {_Z[0, 1]:.2f})，即 (x, 0.5x + y) + (1, 2)，可手验。

                较真小注：样本按**行**存时，X @ W 里是 W 的**行**在当"基向量的新家"——
                和列视角只差一次转置（练习 3 会亲手搬这个转换）。

                再叠一层就是 X @ W1 @ W2 —— 两台变形机器串联（概念 3）。
                **"深度"网络，就是空间被连拧了好几次。**
                """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 🎛️ 交互实验

        **变换游乐场**——三个旋钮 + 一面镜子，亲手拼一台变形机器。

        拼装公式本身就是一次复合变换（从右往左读）：

        ```
        M = 旋转(θ) @ 缩放(sx, sy)
          = [[sx·cosθ, -sy·sinθ],
             [sx·sinθ,  sy·cosθ]]
        ```

        先沿两根轴缩放，再整体旋转。每动一个旋钮，下方实时重画：灰色是原网格、
        绿色是变换后的网格、灰色虚线是原单位方形、橙色是变换后的方形，
        红蓝两支箭头是 M 的两列——两根基向量的新家。

        三个实验建议：

        1. **只转 θ**（sx = sy = 1）：格子整整齐齐地转，det 恒等于 1
        2. **拉大 sx**：横向格子变疏，det 跟着涨——det 就是面积缩放系数
        3. **打开镜像开关**（θ = 0、sx = sy = 1）：M 变回演示 2 的镜像矩阵，
           det 变负——空间翻了面
        """
    )
    return


@app.cell
def _(mo):
    # 交互控制：三个旋钮 + 一面镜子（debounce=True：松手才重算，画面不闪烁）
    theta_slider = mo.ui.slider(0, 360, value=30, step=5, debounce=True,
                                label="🎛️ 旋转角 θ（度，0~360）")
    sx_slider = mo.ui.slider(0.5, 2, value=1, step=0.1, debounce=True,
                             label="🎛️ x 方向缩放 sx（0.5~2）")
    sy_slider = mo.ui.slider(0.5, 2, value=1, step=0.1, debounce=True,
                             label="🎛️ y 方向缩放 sy（0.5~2）")
    mirror_switch = mo.ui.switch(value=False, label="🪞 沿 x 轴镜像（给 y 翻面）")
    mo.vstack([theta_slider, sx_slider, sy_slider, mirror_switch])
    return (mirror_switch, sx_slider, sy_slider, theta_slider)


@app.cell
def _(mirror_switch, mo, np, plt, sx_slider, sy_slider, theta_slider):
    # ===== 交互实验 · 变换游乐场（响应单元）=====
    _theta = np.radians(theta_slider.value)     # 角度 → 弧度（三角函数只吃弧度，11 课讲过）
    _c, _s = np.cos(_theta), np.sin(_theta)     # 旋转零件
    _sx = sx_slider.value                       # x 方向缩放
    _sy = sy_slider.value * (-1.0 if mirror_switch.value else 1.0)   # 镜像 = 给 y 翻符号
    _M = np.array([[_sx * _c, -_sy * _s],       # M = 旋转(θ) @ 缩放（先缩放再旋转）
                   [_sx * _s,  _sy * _c]])
    _det = np.linalg.det(_M)                    # 面积缩放系数（带符号：负 = 翻面）

    # ---- 画图：原网格（灰）+ 变换后网格（绿）+ 单位方形 + 新基向量 ----
    _fig, _ax = plt.subplots(figsize=(6.5, 5.8))
    _ts = np.linspace(-2, 2, 33)                # 每条网格线采 33 个点
    for _t in np.linspace(-2, 2, 9):
        _h = np.stack([_ts, np.full_like(_ts, _t)], axis=1)   # 横线 y=t 上的点
        _v = np.stack([np.full_like(_ts, _t), _ts], axis=1)   # 竖线 x=t 上的点
        _ax.plot(_h[:, 0], _h[:, 1], color="0.75", lw=0.7, alpha=0.5)   # 原网格
        _ax.plot(_v[:, 0], _v[:, 1], color="0.75", lw=0.7, alpha=0.5)
        _ht, _vt = _h @ _M.T, _v @ _M.T         # 整条线一起过机器
        _ax.plot(_ht[:, 0], _ht[:, 1], color="tab:green", lw=0.9, alpha=0.5)
        _ax.plot(_vt[:, 0], _vt[:, 1], color="tab:green", lw=0.9, alpha=0.5)

    # 单位方形：原轮廓（灰虚线）+ 变换后（橙实心）
    _sq = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    _sq_t = _sq @ _M.T
    _ax.plot(np.append(_sq[:, 0], _sq[0, 0]), np.append(_sq[:, 1], _sq[0, 1]),
             "--", color="0.4", lw=1.4)
    _ax.fill(_sq_t[:, 0], _sq_t[:, 1], color="tab:orange", alpha=0.35,
             edgecolor="tab:orange", lw=2.0)

    # 新基向量 = M 的两列（红 = 第 0 列，蓝 = 第 1 列）
    _ax.quiver(0, 0, _M[0, 0], _M[1, 0], angles="xy", scale_units="xy", scale=1,
               color="tab:red", width=0.012, zorder=3, label="x 基向量新家（第 0 列）")
    _ax.quiver(0, 0, _M[0, 1], _M[1, 1], angles="xy", scale_units="xy", scale=1,
               color="tab:blue", width=0.012, zorder=3, label="y 基向量新家（第 1 列）")

    _ax.set_xlim(-4.8, 4.8)                     # 缩放拉满时网格会顶到边，正常（被裁掉一点）
    _ax.set_ylim(-4.8, 4.8)
    _ax.set_aspect("equal")
    _ax.axhline(0, color="gray", lw=0.8)
    _ax.axvline(0, color="gray", lw=0.8)
    _ax.legend(loc="upper left")
    _ax.set_title(
        f"变换游乐场：θ = {theta_slider.value}°，sx = {_sx:.1f}，"
        f"sy = {sy_slider.value:.1f}" + ("（镜像）" if mirror_switch.value else "")
    )

    # ---- 一句话判词（det 的三档解读）----
    if _det < 0:
        _verdict = f"镜像翻转：det = {_det:.2f} < 0——空间被翻了面（面积 ×{abs(_det):g}）"
    elif abs(_det - 1.0) < 1e-9:
        _verdict = "面积不变：det = 1——旋转、剪切都这样；一压一拉正好抵消时也是"
    elif _det < 1:
        _verdict = f"缩水：det = {_det:.2f}——面积缩成原来的 {abs(_det):g} 倍"
    else:
        _verdict = f"膨胀：det = {_det:.2f}——面积胀成 {abs(_det):g} 倍"

    mo.vstack(
        [
            _fig,
            mo.md(
                f"""
                当前变换手册（两列 = 两根基向量的新家）：

                ```
                M = [[{_M[0, 0]:.3f}, {_M[0, 1]:.3f}],
                     [{_M[1, 0]:.3f}, {_M[1, 1]:.3f}]]
                ```

                | 量 | 值 |
                |---|---|
                | 旋转角 θ | **{theta_slider.value}°** |
                | x / y 缩放 | **{_sx:.1f} / {sy_slider.value:.1f}** |
                | det（面积缩放系数） | **{_det:.3f}** |
                """
            ),
            mo.callout(mo.md(f"**实时判词：{_verdict}**"), kind="info"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✏️ 动手练习（每课必做，本课 3 题）": mo.md(
                """
                **练习 1 · 手推旋转矩阵**：逆时针旋转 90° 把 î 送到 +y、把 ĵ 送到 -x。
                用列视角把两个"新家"并排写成一个 (2, 2) 矩阵，填进 `rot90_ex1`，
                再让代码帮你验算两个方向。

                **练习 2 · 复合顺序验证**：A = 旋转 90°、B = x 拉伸 2 倍（都给你备好了）。
                分别算出 A @ B 与 B @ A，用 `np.allclose` 判断它们是否相等，
                把结论填进 `not_equal_ex2`。

                **练习 3 · 写 apply_transform 函数**：输入 M (2, 2) 和 points (n, 2)
                （每行一个点），返回变换后的 (n, 2)。写完用它把单位方形送进
                演示 4 的旋转矩阵。提示：行存的点要和 M 的转置相乘
                （演示 2 讲过的"翻译官"）。

                规则不变：先把 `None` 换成你的答案再运行；写完再看参考答案。
                """
            )
        }
    )
    return


@app.cell
def _(mo, np):
    # ===== 练习 1 · 手推旋转矩阵 =====
    # 逆时针旋转 90°：î (1, 0) 被送到 +y 轴上，ĵ (0, 1) 被送到 -x 轴上。
    # 列视角：把两个"新家"并排写进矩阵的两列——第 0 列放 î 的去向，第 1 列放 ĵ 的。

    rot90_ex1 = None    # ← 填一个 (2, 2) 的 np 数组（两个新家分别是 (0, 1) 和 (-1, 0)）

    _e_x = np.array([1.0, 0.0])   # î：向右
    _e_y = np.array([0.0, 1.0])   # ĵ：向上

    _msg = (
        f"验证：rot90_ex1 @ î = {rot90_ex1 @ _e_x}（应得 [0. 1.]）\n"
        f"验证：rot90_ex1 @ ĵ = {rot90_ex1 @ _e_y}（应得 [-1. 0.]）"
        if rot90_ex1 is not None
        else "⏳ 用列视角把 rot90_ex1 的 None 换成你手推的矩阵（提示：î 去了 (0, 1)，ĵ 去了 (-1, 0)）"
    )
    mo.md(_msg)
    return (rot90_ex1,)


@app.cell
def _(mo, np):
    # ===== 练习 2 · 复合顺序验证 =====
    A_ex2 = np.array([[0.0, -1.0],      # A：逆时针旋转 90°（演示 3 的 M1）
                      [1.0,  0.0]])
    B_ex2 = np.array([[2.0, 0.0],       # B：x 方向拉伸 2 倍（演示 3 的 M2）
                      [0.0, 1.0]])

    AB_ex2 = None        # ← A @ B：先做 B 再做 A 的那份合成手册
    BA_ex2 = None        # ← B @ A：先做 A 再做 B
    not_equal_ex2 = None # ← bool：两份手册不相等吗？提示：not np.allclose(AB_ex2, BA_ex2)

    _msg = (
        f"""
        你的计算结果：

        ```
        A @ B = [[{AB_ex2[0, 0]:.0f}, {AB_ex2[0, 1]:.0f}], [{AB_ex2[1, 0]:.0f}, {AB_ex2[1, 1]:.0f}]]    ← 先拉伸，再旋转
        B @ A = [[{BA_ex2[0, 0]:.0f}, {BA_ex2[0, 1]:.0f}], [{BA_ex2[1, 0]:.0f}, {BA_ex2[1, 1]:.0f}]]    ← 先旋转，再拉伸
        ```

        np.allclose(A @ B, B @ A) = **{np.allclose(AB_ex2, BA_ex2)}**——两个顺序算出的
        手册确实不一样！

        你的 not_equal_ex2 = **{not_equal_ex2}**（应得 **True**：顺序换了，机器就换了）
        """
        if AB_ex2 is not None and BA_ex2 is not None and not_equal_ex2 is not None
        else "⏳ 先算出 AB_ex2 与 BA_ex2（各一行 @），再用 not np.allclose(...) 把 not_equal_ex2 补上"
    )
    mo.md(_msg)
    return (A_ex2, AB_ex2, B_ex2, BA_ex2, not_equal_ex2)


@app.cell
def _(mo, np):
    # ===== 练习 3 · apply_transform：把一批点送进任意变形机器 =====
    def apply_transform(M, points):
        """输入：M (2, 2) 变换手册；points (n, 2) 每行一个点。返回：(n, 2) 变换后的点。"""
        _out = None    # ← 一行搞定：让每个点各做一次 M @ v。提示：行存的点 × M 的转置
        return _out

    # 单位方形的四个顶点（每行一个点）——拿它当试验品
    square_ex3 = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    M_ex3 = np.array([[0.8, -0.6],              # 演示 4 的旋转矩阵（cosθ=0.8, sinθ=0.6）
                      [0.6,  0.8]])

    square_t_ex3 = None    # ← apply_transform(M_ex3, square_ex3)

    _msg = (
        f"""
        变换后的四个顶点（每行一个点）：

        ```
        {square_t_ex3}
        ```

        对账：î 即 (1, 0) 的正确去向是 **(0.8, 0.6)**；
        你算出的第 1 行是 **({square_t_ex3[1, 0]:.1f}, {square_t_ex3[1, 1]:.1f})**。

        det(M_ex3) = {np.linalg.det(M_ex3):.2f} → 面积不变，只是转了约 36.87°。
        """
        if square_t_ex3 is not None
        else "⏳ 先把 apply_transform 里的一行补完（提示：points @ M.T），再填 square_t_ex3"
    )
    mo.md(_msg)
    return (M_ex3, apply_transform, square_ex3, square_t_ex3)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 1 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                rot90_ex1 = np.array([[0.0, -1.0],
                                      [1.0,  0.0]])
                ```

                手推过程：逆时针转 90°，î (1, 0) 落到 +y 轴 → 第 0 列写 (0, 1)；
                ĵ (0, 1) 落到 -x 轴 → 第 1 列写 (-1, 0)。两列并排一抄，矩阵到手。

                彩蛋：把 90° 换成任意角 θ，旋转矩阵是

                ```
                [[cosθ, -sinθ],
                 [sinθ,  cosθ]]
                ```

                交互实验里那台机器，就是它乘上缩放之后的样子。
                """
            ),
            "✅ 练习 2 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                AB_ex2 = A_ex2 @ B_ex2                        # [[0, -1], [2, 0]]：先拉伸再旋转
                BA_ex2 = B_ex2 @ A_ex2                        # [[0, -2], [1, 0]]：先旋转再拉伸
                not_equal_ex2 = not np.allclose(AB_ex2, BA_ex2)   # True
                ```

                演示 3 的画面：先拉再转得到"竖高条"，先转再拉得到"横扁条"——
                顺序换了，故事就换了。

                两个小提醒：

                - 比较矩阵要用 `np.allclose` 而不是 `==`——浮点噪声（09 课的老教训）
                - `not np.allclose(...)` 是"不相等"的标准写法：allclose 返回 bool，
                  取反正好是想要的答案
                """
            ),
            "✅ 练习 3 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                def apply_transform(M, points):
                    _out = points @ M.T     # 行存的点 × M 的转置 = 每个点各做一次 M @ v
                    return _out

                square_t_ex3 = apply_transform(M_ex3, square_ex3)
                ```

                为什么有个 `.T`？我们的约定是"**M 的列 = 基向量去向**"（列视角），
                而点按**行**存。行点 x 的变换是 M @ x（列向量约定），把它躺平写成行，
                就是 x @ M.T——转置是行/列两种记法之间的翻译官（09 课认识的）。

                验证：square_t_ex3 的四行依次是
                (0, 0)、(0.8, 0.6)、(0.2, 1.4)、(-0.6, 0.8)——
                单位方形被演示 4 的矩阵转了约 36.87°，面积分毫未变（det = 1）。
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
                1. **列视角（本课灵魂）**：矩阵的第 k 列 = 第 k 根基向量被送往哪里；
                   M @ v = v[0] × (第 0 列) + v[1] × (第 1 列)——抓住基向量，就抓住整个空间
                2. **常见变换速查**：旋转 90° [[0, -1], [1, 0]]、放大 2 倍 [[2, 0], [0, 2]]、
                   剪切 [[1, 1], [0, 1]]、镜像 [[1, 0], [0, -1]]；线性变换两铁律：
                   原点不动、直线保持直线且平行线保持平行
                3. **复合变换 = 矩阵相乘，从右往左读**：A @ B = 先 B 后 A；
                   09 课"内维相消，外维成形" = 流水线要能串联，出口必须对上入口
                4. **AB ≠ BA**：先转再拉 ≠ 先拉再转——矩阵乘法不可交换，顺序就是故事
                5. **det（行列式）= 面积缩放系数**：det = 1 面积不变（旋转 / 剪切）、
                   det = 4 胀 4 倍、det < 0 翻面（镜像）；`np.linalg.det` 一行算
                6. **X @ W + b 的几何**：W 把空间变形（旋转 / 拉伸 / 剪切），b 平移；
                   多层网络 = 多台机器串联——"深度"就是空间被变形的次数
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

        **下一课预告** — `13_导数与梯度`：

        空间会变形了，机器也看懂了。但还差最后一块拼图：**怎么知道该往哪个方向
        拧，损失才会变小？** 这需要"变化的数学"——导数与梯度。

        下一课，我们给曲线画切线、给山画等高线、给整个平面画梯度场；
        "梯度下降"将从四个汉字变成一支看得见的箭头——它永远指向最陡的上坡，
        而我们永远朝着它的反方向下山。

        🧭 变形机器已就位，去认路吧！
        """
    )
    return


if __name__ == "__main__":
    app.run()
