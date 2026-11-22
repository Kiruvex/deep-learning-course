import marimo

__generated_with = "0.25.2"
app = marimo.App(app_title="13 · 导数与梯度", width="medium")


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
        # 13 · 导数与梯度

        **从零到 Transformer · 手写深度学习** ｜ 第 2 部分 · 深度学习的数学（3/4）
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📖 本课目标（先读我）": mo.md(
                """
                **这一课只解决一个问题：往哪个方向变，损失才能变小？**

                第 12 课让整个空间动了起来：矩阵是变形机器，旋转、拉伸、剪切随手就来。
                但变形不是目的——训练网络时真正想问的是：**参数往哪边挪一点点，损失会降？**

                第 12 课结尾留了话：空间会变形了，但怎么知道该往哪个方向拧、损失才会变小？
                这需要"变化的数学"——"梯度下降"将从四个汉字变成一支看得见的箭头。
                今天就来兑现：把"变化"变成数字，把"方向"变成箭头。

                学完你将能：

                - 说出导数的三步走：**平均变化率 → h 缩小 → 瞬时变化率（切线斜率）**，
                  并秒算 f(x) = x² 的导数 2x
                - 用**对称差分** (f(x+h) - f(x-h)) / (2h) 给任何函数做数值求导，
                  并说清 h 的分寸：太大是截断误差、太小是浮点噪声（U 型曲线）
                - 把一串偏导排成**梯度 ∇f**：方向 = 最陡上坡、长度 = 陡峭程度
                  （11 课的"向量是箭头"在这里派上大用场）
                - 在等高线图上亲眼验证：**梯度处处垂直于等高线、指向碗外**
                - 手写深度学习的心跳公式 **x ← x - lr·∇f**，看它 12 步把点送回碗底

                前置：第 11 课的向量 / 范数 / quiver / set_aspect("equal")（梯度的画法
                全靠它）、第 09 课的浮点残渣与 np.linalg.norm（h 太小时前者回来捣乱）、
                第 06 课广播 + 第 08 课 axis（网格运算与按行压缩）。
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "🎯 概念 1/4 · 导数 = 瞬时变化率": mo.md(
                """
                开车时瞄一眼时速表：**这一瞬间车速是多少？** 这就是导数要回答的问题。

                "瞬时"没法直接算（时间间隔为零就没有变化了），只能**用平均去逼近**：

                ```
                平均变化率 = (f(x+h) - f(x)) / h      （高度差 ÷ 底边长）
                ```

                h 越小，平均越接近瞬时。几何画面：连接曲线上两点的直线叫**割线**，
                h 缩小 = 两点并成一点 = 割线的极限位置就是**切线**。
                **导数 = 切线的斜率 = 瞬时变化率**——同一个东西，三副面孔。

                拿本课主角 f(x) = x² 在 x = 1 处练手（结论先记下，不推严格证明）。
                前向差分 (f(1+h) - f(1)) / h 算"平均变化率"，h 一档档缩小：

                | h | 平均变化率 |
                |---|---|
                | 1 | 3.0 |
                | 0.1 | 2.1 |
                | 0.01 | 2.01 |
                | 0.001 | 2.001 |
                | → 0 | **→ 2** |

                平均值一步步收敛到 2——这就是 **f'(x) = 2x 在 x = 1 处等于 2** 的数值来历。
                推广到任意点：

                ```
                f'(x) = 2x
                ```

                x = -1.5 处斜率 -3（下坡）、x = 0 处 0（平地）、x = 1.5 处 +3（上坡）——
                同一条曲线，每个点的"脾气"不一样；导数就是把脾气逐点写下来。

                代码里怎么算？**对称差分**：往右看半步、往左看半步，取中间：

                ```
                f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
                ```

                比只往右看的"前向差分" (f(x+h)-f(x))/h 更准（误差从 h 级降到 h² 级）——
                交互实验里你会把这两条线摆在一起看。
                """
            ),
            "📐 概念 2/4 · 从导数到梯度": mo.md(
                """
                函数不止一个自变量时，"变化率"就有了方向。f(x, y) 里——

                - 固定 y 不动、只动 x：得到**偏导数 ∂f/∂x**（"偏"= 其他变量通通当常数）
                - 固定 x 不动、只动 y：得到 **∂f/∂y**

                把所有偏导排成一个向量，就是**梯度**（读作 nabla f 或 grad f）：

                ```
                ∇f(x, y) = (∂f/∂x, ∂f/∂y)
                ```

                第 11 课说过：向量是箭头。于是梯度天生带画面，还自带一条惊人的性质：

                - **方向**：指向"最陡上坡"——站在这一点，朝 ∇f 走，函数值涨得最快
                - **长度**（np.linalg.norm，09/11 课的老朋友）：坡有多陡——梯度越长越陡

                一句话：**梯度是山坡自带的"陡度计 + 指北针"**。
                演示 3 会把它画在等高线图上：一支支箭头全部垂直于等高线、齐刷刷指向碗外。

                自变量更多时照样排：∇f(x, y, z) = (∂f/∂x, ∂f/∂y, ∂f/∂z)。
                将来的神经网络动辄百万个参数，梯度就是百万维空间里的一个箭头——
                画不出来，但"方向 = 最陡上坡、长度 = 陡峭程度"这两条性质原封不动。
                今天在二维里把画面看牢，是为了以后在没有画面的高维里也敢想。
                """
            ),
            "⛰️ 概念 3/4 · 沿负梯度走 = 下山": mo.md(
                """
                想象浓雾里登山：看不见全貌、没有地图，只有脚底能感觉到坡度。
                最朴素也最有效的策略：**每一步都朝脚下最陡的下坡方向挪一小步**——
                也就是沿着**负梯度 -∇f** 走（梯指上坡，我们要下坡，所以加个负号）。

                写成一行，这就是深度学习的**心跳**：

                ```
                x ← x - lr · ∇f(x)
                ```

                - **x**：当前所有参数（这一课是平面上的一个点，将来是成千上万个权重）
                - **lr**（learning rate，学习率）：步子迈多大。太大容易跨过谷底来回震荡，
                  太小走一万步还在半山腰
                - 每走一步重新感受坡度（梯度随位置变），再走下一步——循环往复

                演示 4 从碗壁出发走 12 步，亲眼看着点滑进碗底；第 15 课，这个循环将
                原封不动变成第一个训练循环的代码。今天先看懂它的几何。
                """
            ),
            "🔬 概念 4/4 · 数值微分的分寸": mo.md(
                """
                对称差分 (f(x+h) - f(x-h)) / (2h) 里的 h 由你选。选多大？两头都是坑：

                - **h 太大**：割线还没贴上切线，算的是"平均变化率"，误差 ∝ h²（截断误差）
                - **h 太小**：分子是两个几乎相等的数相减——第 09 课的**浮点残渣**
                  （0.1 + 0.2 - 0.3，本该是 0、却算出 5.55e-17 那位老朋友）回来捣乱，
                  残渣除以极小的 2h 被急剧放大，信号被噪声淹没

                把 h 从 1e-1 扫到 1e-16，误差曲线是一条 **U 型**：右端栽在"割线"上，
                左端栽在"噪声"里，谷底大致落在 h ≈ 1e-5 ~ 1e-7——数值微分的经验区间。

                说句实话：深度学习真正的梯度不靠数值差分——对百万级参数逐个扰动太慢
                太脆，而是靠**反向传播**一步精确算出全部梯度，引擎是下一课的**链式
                法则**。但"梯度是什么、h 有多脆"的直觉，数值实验给得最扎实。
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

        五组演示，从"切线会动"一路走到"心跳公式"：

        - **演示 1** · 切线会动：f(x) = x² 上三个点的切线，斜率 = 2x 各不相同
        - **演示 2** · h 的分寸：数值导数的误差 U 型曲线（第一次用 log 轴）
        - **演示 3** · 等高线 + 梯度场：一只碗的全景地图
        - **演示 4** · 梯度下降预演：x ← x - lr·∇f 走 12 步
        - **演示 5** · 偏导 = "其他变量当常数"：切片 + 数值对答案
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 1 · 切线会动

        f(x) = x² 的导数是 f'(x) = 2x——**每个点的切线斜率都不一样**。
        在 x = -1.5、0.5、1.5 三个点各画一条切线：斜率分别是 -3、+1、+3，
        左边往下坡走、右边往上坡爬，一眼分清。

        过切点 (a, f(a))、斜率为 f'(a) 的直线就是切线，点斜式一步写出：

        ```
        y = f(a) + f'(a) · (x - a)      （从切点出发，按斜率爬）
        ```

        切线只画切点附近一小段（±1.1）——它本来就是"局部"的代言人。

        看图时盯两件事：三条切线的**倾斜程度**一眼不同（-3、+1、+3，
        图例里标了数）；切线再怎么不同，都**恰好贴着**曲线滑过切点——
        "过切点 + 斜率对"两个条件缺一不可。
        """
    )
    return


@app.cell
def _(np, plt):
    # 演示 1 · 切线会动：f(x) = x² 上三个点的切线
    _f = lambda x: x ** 2        # 主角：f(x) = x²（lambda = 一行的小函数，随用随扔）
    _df = lambda x: 2 * x        # 它的导数：f'(x) = 2x（每一点的切线斜率）

    _x = np.linspace(-2.5, 2.5, 400)     # 画曲线用的稠密采样（03 课老朋友）
    _pts = [-1.5, 0.5, 1.5]             # 三个切点：左边 / 缓坡 / 右边
    _cols = ["tab:red", "tab:green", "tab:blue"]   # 三条切线各配一色

    _fig, _ax = plt.subplots(figsize=(7.2, 4.8))   # 二件套：fig=画布，ax=坐标系
    _ax.plot(_x, _f(_x), color="dimgray", lw=2.2, label="f(x) = x²", zorder=2)

    # 三个点同一段流程，用 zip 打包循环（点、色一一配对）
    for _a, _c in zip(_pts, _cols):
        _k = _df(_a)                                # 该点切线斜率 = 2a
        _xs = np.linspace(_a - 1.1, _a + 1.1, 50)   # 切线只画切点附近一小段
        _yt = _f(_a) + _k * (_xs - _a)              # 点斜式：y = f(a) + f'(a)·(x - a)
        _ax.plot(_xs, _yt, "--", color=_c, lw=2,
                 label=f"x = {_a} 处切线（斜率 {_k:+.1f}）")
        _ax.scatter([_a], [_f(_a)], s=50, color=_c, zorder=3)   # 切点：颜色同切线

    _ax.set_xlim(-2.5, 2.5)          # x 轴范围
    _ax.set_ylim(-1.4, 6.8)          # y 轴范围：盖住曲线顶点 6.25 即可
    _ax.grid(True, alpha=0.3)        # 淡网格
    _ax.axhline(0, color="gray", lw=0.8)   # 画出 x 轴（y=0 的横线）
    _ax.axvline(0, color="gray", lw=0.8)   # 画出 y 轴（x=0 的竖线）
    _ax.legend(loc="upper center")   # 图例：斜率就标在这里
    _ax.set_title("切线会动：斜率 = 2x，每个点都不一样")
    _fig                              # 末表达式 = 自动显示（11 课约定）
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 2 · h 的分寸：数值导数 vs 精确导数

        对称差分里的 h 到底取多大？拿 11 课画过的 sin 当小白鼠：
        它在 x = 1 的精确导数是 cos(1) ≈ 0.5403（"sin 的导数是 cos"，先当已知用）。

        把 h 从 1e-1 一路扫到 1e-16（16 档，每档除以 10），画 |数值 - 精确|。

        第一次用 **log 轴**（`set_xscale("log")` / `set_yscale("log")`）：
        h 横跨 16 个数量级，普通坐标下左边全挤成一个点；log 轴把"每除以 10"
        变成"一格"，专看数量级——以后画损失曲线（从 1e0 一路降到 1e-5）也靠它。

        预测：右端 h 太大（割线没贴上切线），左端 h 太小（浮点噪声淹没信号），
        中间应该有个谷底。图上三条注释（红、红、绿）会把这三个位置点名。
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # 演示 2 · h 的分寸：数值导数的误差 U 型曲线（双对数轴）
    _x0 = 1.0                                        # 固定在 x = 1 求导
    _exact = np.cos(_x0)                             # 精确导数：sin' = cos → cos(1) ≈ 0.5403
    _hs = np.logspace(-1, -16, 16)                   # h = 1e-1, 1e-2, ..., 1e-16（16 档，
                                                     # logspace = 10 的等比数列，每档 ÷10）
    _num = (np.sin(_x0 + _hs) - np.sin(_x0 - _hs)) / (2 * _hs)   # 对称差分
                                                     # （x0 + _hs 整条数组一起算，06 课广播）
    _errs = np.abs(_num - _exact)                    # 每档 h 的绝对误差 |数值 - 精确|
    _i = int(_errs.argmin())                         # 谷底在哪一档（argmin = 最小值下标）

    _fig, _ax = plt.subplots(figsize=(7.5, 4.8))
    _ax.plot(_hs, _errs, "o-", color="tab:blue")     # 误差曲线：横 h、纵误差
    _ax.set_xscale("log")                            # 横轴 log：每除以 10 = 一格
    _ax.set_yscale("log")                            # 纵轴 log：误差跨 9 个数量级
    _ax.set_xlim(3e-17, 3e-1)                        # 留一点边距，注释有地方站
    _ax.set_ylim(1e-13, 3e-1)
    _ax.axvspan(1e-7, 1e-5, color="tab:green", alpha=0.15)   # 经验区间涂上淡绿
    _ax.text(1e-6, 2e-6, "经验区间\nh ≈ 1e-5 ~ 1e-7",
             ha="center", fontsize=9, color="seagreen")

    # annotate 第一次用：xy = 箭头指向的数据点，xytext = 文字摆放的位置
    # 右端注释：h 太大的病（截断误差）
    _ax.annotate("h 太大：割线还没贴上切线\n（截断误差跟 h² 成正比）",
                 xy=(_hs[0], _errs[0]), xytext=(8e-2, 2e-1),
                 fontsize=10, color="tab:red", ha="right", va="top",
                 arrowprops=dict(arrowstyle="->", color="tab:red"))
    # 左端注释：h 太小的病（浮点噪声）
    _ax.annotate("h 太小：浮点噪声淹没信号\n（09 课的浮点残渣回来捣乱）",
                 xy=(_hs[-1], _errs[-1]), xytext=(3e-16, 2e-1),
                 fontsize=10, color="tab:red", ha="left", va="top",
                 arrowprops=dict(arrowstyle="->", color="tab:red"))
    # 谷底注释：最甜的一档
    _ax.annotate(f"谷底 h ≈ {_hs[_i]:.0e}，误差仅 {_errs[_i]:.1e}",
                 xy=(_hs[_i], _errs[_i]), xytext=(1e-11, 5e-13),
                 fontsize=10, color="tab:green", ha="left", va="bottom",
                 arrowprops=dict(arrowstyle="->", color="tab:green"))

    _ax.set_xlabel("步长 h（log 轴）")
    _ax.set_ylabel("误差 |数值 - 精确|（log 轴）")
    _ax.set_title("h 的分寸：太大太小都不行（双对数轴）")
    _ax.grid(True, which="both", alpha=0.3)          # log 轴要 both：主刻度次刻度都画网格
    _fig.tight_layout()

    mo.vstack([
        _fig,
        mo.md(
            f"""
            读图（四个看点）：

            1. **右端**（h = 1e-1）：误差 9.0e-4——h 太大，割线还没贴上切线；
               h 每缩小 10 倍误差缩小 100 倍，这就是"截断误差 ∝ h²"的样子
            2. **谷底**（h ≈ 1e-5）：误差只有 {_errs[_i]:.1e}——不大不小，刚刚好
            3. **左端**（h = 1e-16）：误差反弹到 1.5e-2，比谷底差 9 个数量级——
               分子是两个几乎相等的数相减，09 课的浮点残渣被 2h 急剧放大
            4. 彩蛋：h = 1e-15 和 h = 1e-16 的误差**一模一样**——步子小到机器
               存不下，1 + h 存进浮点就被舍入到固定刻度，你嘴上的 h 和
               机器实际走的 h 已经不是一回事
            """
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 3 · 等高线 + 梯度场：一只碗的全景地图

        f(x, y) = x² + y² 的三维图像是一只开口朝上的**碗**。不画 3D，
        用两张 2D 图看透它：

        - **等高线** `ax.contour`：把"高度相同的点"连成圈——地形图的等高线，
          圈越小越接近碗底
        - **梯度场** `ax.quiver`：每个网格点画一支 ∇f = (2x, 2y) 箭头——
          11 课 quiver 的网格版，一次喂整片网格

        造网格的 `np.meshgrid` 第一次见，值得停一分钟：给它 x 轴刻度和 y 轴刻度，
        它吐回两个同形矩阵——X 的每格写着自己的 x 坐标、Y 的每格写着自己的 y 坐标。
        两者叠起来，棋盘上每个格子都知道"自己站在哪"；F = X² + Y²（06 课逐元素
        广播）再给每格标上海拔。

        一个小分工：等高线用**密网格**（200×200，圈才画得圆），箭头场用**疏网格**
        （9×9，不然箭头挤成一团）。演示 4 的等高线还靠这套网格。

        看图验证两件事：**箭头处处垂直于等高线**（不垂直就不配叫"最陡"），
        **箭头齐刷刷指向碗外**（梯度 = 最陡上坡 = 爬出碗的方向）。
        """
    )
    return


@app.cell
def _(np, plt):
    # 演示 3 · 等高线 + 梯度场：f(x, y) = x² + y² 是一只碗
    _xsf = np.linspace(-2, 2, 200)              # 密网格刻度：喂等高线
    _Xf, _Yf = np.meshgrid(_xsf, _xsf)          # 铺开：X 每格写 x 坐标，Y 每格写 y 坐标
    # meshgrid 长什么样？拿 3×3 迷你版看：meshgrid([1,2,3], [4,5,6]) 吐回
    #   X = [[1,2,3], [1,2,3], [1,2,3]]   （每格写自己的列号）
    #   Y = [[4,4,4], [5,5,5], [6,6,6]]   （每格写自己的行号）
    # 两者一叠，9 个格子个个知道"自己站在哪"——这就是网格的坐标系
    _Ff = _Xf ** 2 + _Yf ** 2                   # 每格的"海拔"（06 课逐元素广播）

    _xsc = np.linspace(-2, 2, 9)                # 疏网格刻度：喂箭头场
    _Xc, _Yc = np.meshgrid(_xsc, _xsc)          # 同一个 meshgrid，两种密度
    _GX, _GY = 2 * _Xc, 2 * _Yc                 # 梯度场：∂f/∂x = 2x、∂f/∂y = 2y

    _fig, _ax = plt.subplots(figsize=(6.8, 5.6))
    # 等高线：指定画哪几条"海拔线圈"，灰线 + 线上标数值
    _cs = _ax.contour(_Xf, _Yf, _Ff, levels=[0.5, 1.5, 2.5, 4.0],
                      colors="gray", linewidths=1.2)
    _ax.clabel(_cs, fmt="%.1f")                 # 线上的数字 = 这一圈的函数值
    # 梯度场：网格版 quiver——每个网格点一支箭头，箭头越长 = 梯度越长 = 越陡。
    # 注意网格版不写 angles/scale 三件套：让 matplotlib 自动统一缩放，
    # 箭头长短的"比例"仍然真实（最长的那支 = 最陡的地方）
    _ax.quiver(_Xc, _Yc, _GX, _GY, color="tab:red", alpha=0.85,
               width=0.004, zorder=3)
    # 碗底：梯度为 (0, 0) 的平地点
    _ax.scatter([0], [0], s=80, color="tab:blue", zorder=4, marker="*")
    _ax.text(0.1, 0.08, "碗底 (0, 0)", color="tab:blue", fontsize=10)

    _ax.set_xlim(-2.15, 2.15)                   # x 轴范围
    _ax.set_ylim(-2.15, 2.15)                   # y 轴范围
    _ax.set_aspect("equal")                     # 11 课铁律：几何图必须等比例，
    _ax.grid(True, alpha=0.3)                   # 不然"垂直"看起来不可信
    _ax.axhline(0, color="gray", lw=0.8)        # x 轴
    _ax.axvline(0, color="gray", lw=0.8)        # y 轴
    _ax.set_xlabel("x")
    _ax.set_ylabel("y")
    _ax.set_title("等高线 + 梯度场：箭头垂直于等高线，指向碗外")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 4 · 梯度下降预演：12 步下山

        概念 3 的雾中下山写成代码，核心就一行心跳：

        ```
        x ← x - lr · ∇f(x)
        ```

        从碗壁上的 (1.8, 1.2) 出发（f = 4.68），学习率 lr = 0.1，走 12 步。
        对这只碗，∇f = (2x, 2y)，代入心跳公式：x ← x - 0.1·2x = **0.8x**——
        每步坐标乘 0.8。离碗底越近梯度越短，步子自动变小：**自带刹车**，
        稳稳滑向碗底，而不是在谷底来回横跳。

        看图时盯三件事：脚印**间距越走越密**（刹车）；箭头方向**始终指向原点**
        （每一点的最陡下坡都直指碗底）；最后一点几乎贴住黑色 ×，但**永远差一点**——
        梯度下降是渐进逼近，不是一步到位。
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # 演示 4 · 梯度下降预演：从 (1.8, 1.2) 出发走 12 步
    _pt = np.array([1.8, 1.2])          # 出发点：碗壁上，f = 3.24 + 1.44 = 4.68
    _lr = 0.1                            # 学习率：每步挪 0.1 倍梯度那么长
    _path = [_pt.copy()]                 # 脚印列表（copy 很关键：不然存的是同一个数组的引用）
    for _step in range(12):              # 走 12 步
        _grad = 2.0 * _pt                # 感受脚下坡度：∇f = (2x, 2y)
        _pt = _pt - _lr * _grad          # ⭐ 心跳公式：x ← x - lr·∇f(x)（就这一行！）
        _path.append(_pt.copy())         # 记下新脚印
    _path = np.array(_path)              # (13, 2)：出发点 + 12 步 = 13 个脚印
    _fvals = np.sum(_path ** 2, axis=1)  # 每步的 f 值——08 课 axis=1：每行压成一个数
    _dists = np.linalg.norm(_path, axis=1)   # 每步离碗底的距离——09/11 课的范数

    # 画图：等高线当底图（密网格，沿用演示 3 的分工）
    _xs = np.linspace(-2.2, 2.2, 200)
    _X, _Y = np.meshgrid(_xs, _xs)
    _fig, _ax = plt.subplots(figsize=(6.8, 5.6))
    _cs = _ax.contour(_X, _Y, _X ** 2 + _Y ** 2, levels=[0.2, 1.0, 2.0, 3.5],
                      colors="gray", linewidths=1.1)
    _ax.clabel(_cs, fmt="%.1f")                 # 等高线数值：看 f 一圈圈往下掉
    # 相邻脚印之间画箭头：quiver(起点x, 起点y, dx, dy)——11 课的箭头语法。
    # 切片配对是 04 课的手艺：[:-1] 丢最后一个当"起点"，[1:] 丢第一个当"终点"，
    # 两列错开一位，正好一一对上——12 段位移一步到位
    _ax.quiver(_path[:-1, 0], _path[:-1, 1],
               _path[1:, 0] - _path[:-1, 0], _path[1:, 1] - _path[:-1, 1],
               angles="xy", scale_units="xy", scale=1,
               color="tab:blue", width=0.006, alpha=0.9,
               label="每步沿 -∇f 方向挪一小步")
    _ax.scatter(_path[:, 0], _path[:, 1], s=22, color="tab:blue", zorder=3,
                label="13 个脚印")
    _ax.scatter([1.8], [1.2], s=150, color="tab:green", zorder=4, marker="*",
                label="出发点 (1.8, 1.2)")
    _ax.scatter([_path[-1, 0]], [_path[-1, 1]], s=70, color="tab:red", zorder=4,
                label=f"第 12 步 ({_path[-1, 0]:.2f}, {_path[-1, 1]:.2f})")
    _ax.scatter([0], [0], s=200, color="black", marker="x", zorder=4,
                label="碗底 (0, 0)")

    _ax.set_xlim(-2.2, 2.2)              # x 轴范围
    _ax.set_ylim(-2.2, 2.2)              # y 轴范围
    _ax.set_aspect("equal")              # 等比例（11 课铁律）
    _ax.grid(True, alpha=0.3)
    _ax.set_xlabel("x")
    _ax.set_ylabel("y")
    _ax.legend(loc="upper left", fontsize=9)   # 左上角是空的：路全在右上→原点
    _ax.set_title("梯度下降预演：沿 -∇f 走 12 步滑向碗底")
    _fig.tight_layout()

    # 抽查几步的账本（0、4、8、12 步），数字与图互相印证
    # f 值用 08 课 axis=1 按行压缩，距离用 09/11 课的范数——旧零件全在岗
    _rows = "\n".join(
        f"| {_i} | ({_path[_i, 0]:.3f}, {_path[_i, 1]:.3f}) | {_fvals[_i]:.3f} | {_dists[_i]:.3f} |"
        for _i in [0, 4, 8, 12]
    )
    mo.vstack([
        _fig,
        mo.md(
            f"""
            | 步 | 位置 (x, y) | f(x, y) | 离碗底的距离 |
            |---|---|---|---|
            {_rows}

            12 步把 f 从 **{_fvals[0]:.2f}** 压到 **{_fvals[-1]:.3f}**
            （缩到约 1/{_fvals[0] / _fvals[-1]:.0f}），离碗底从 **{_dists[0]:.2f}**
            缩到 **{_dists[-1]:.2f}**（约 1/{_dists[0] / _dists[-1]:.0f}）。

            注意箭头的长度在缩短：越靠近碗底，梯度 (2x, 2y) 越短，步子自动变小——
            **自带刹车**。要是 lr 拿太大（这只碗超过 1），一步就会跨过碗底弹到
            对面更高处，越走越远——第 15 课调参时你会亲眼见到这种"发散"。
            """
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 5 · 偏导 = "其他变量当常数"

        f(x, y) = x²y 有两个自变量、两条偏导：

        ```
        ∂f/∂x = 2xy    （y 当常数：x² 求导得 2x，常数 y 乘在外面）
        ∂f/∂y = x²     （x 当常数：x² 整个变成系数，乘上 y 的导数 1）
        ```

        "当常数"最直观的画面是**切片**：在 (3, 2) 这一点——

        - 冻住 y = 2，沿 x 方向切一刀：切片 f(x, 2) = 2x² 是条曲线，
          它在 x = 3 处的斜率就是 ∂f/∂x
        - 冻住 x = 3，沿 y 方向切一刀：切片 f(3, y) = 9y 是条直线，
          斜率处处是 9，正是 ∂f/∂y

        左图右图各一刀，再用对称差分数值对答案——"偏导"两个字就落到地上了。
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # 演示 5 · 偏导 = 其他变量当常数：f(x, y) = x²y 的两刀切片
    _f = lambda x, y: x ** 2 * y        # 主角：f(x, y) = x²y
    _px, _py = 3.0, 2.0                 # 求导点：站在 (3, 2) 这一格
    _h = 1e-5                           # 对称差分的 h（概念 4 的经验区间）

    # ---- 左右两张切片图：subplots(1, 2) = 一行两列，一次拿两个坐标系 ----
    _xsl = np.linspace(1.5, 4.5, 200)   # 左图：y 冻在 2，x 从 1.5 到 4.5
    _ysl = np.linspace(0.0, 4.0, 200)   # 右图：x 冻在 3，y 从 0 到 4
    _kL = 2 * _px * _py                 # ∂f/∂x(3,2) = 2xy = 12（解析）
    _kR = _px ** 2                      # ∂f/∂y(3,2) = x² = 9（解析）

    _fig, (_axL, _axR) = plt.subplots(1, 2, figsize=(11, 4.5))
    # 左图：切片曲线 + x=3 处的切线（斜率 12）
    _axL.plot(_xsl, _f(_xsl, 2.0), color="dimgray", lw=2.2,
              label="切片 f(x, 2) = 2x²")
    _axL.plot(_xsl, _f(_px, 2.0) + _kL * (_xsl - _px), "--", color="tab:red",
              lw=2, label=f"x = 3 处切线（斜率 2xy = {_kL:g}）")
    _axL.scatter([_px], [_f(_px, _py)], s=55, color="tab:red", zorder=3)   # 切点
    _axL.set_xlabel("x（y 冻在 2）")     # 轴标签顺便提醒"谁被冻住了"
    _axL.set_ylabel("f(x, 2)")
    _axL.grid(True, alpha=0.3)
    _axL.legend()
    _axL.set_title("动 x、冻 y → 斜率是 ∂f/∂x")

    # 右图：切片本身是直线，切线与它重合（斜率处处 9）
    _axR.plot(_ysl, _f(3.0, _ysl), color="dimgray", lw=2.2,
              label="切片 f(3, y) = 9y")
    _axR.plot(_ysl, _f(_px, 2.0) + _kR * (_ysl - _py), "--", color="tab:blue",
              lw=2, alpha=0.7, label=f"切线与之重合（斜率 x² = {_kR:g}）")
    _axR.scatter([_py], [_f(_px, _py)], s=55, color="tab:blue", zorder=3)  # 切点
    _axR.set_xlabel("y（x 冻在 3）")
    _axR.set_ylabel("f(3, y)")
    _axR.grid(True, alpha=0.3)
    _axR.legend()
    _axR.set_title("动 y、冻 x → 斜率是 ∂f/∂y")
    _fig.tight_layout()

    # ---- 数值对答案：三个点上，解析 vs 对称差分 ----
    # 偏导的数值版 = 概念 1 的对称差分 + "另一维冻住不动"，没有任何新东西
    _pts = [(3.0, 2.0), (-1.0, 2.0), (2.0, 0.5)]   # 三个求导点
    _rows = []
    for _x, _y in _pts:
        _dfx = (_f(_x + _h, _y) - _f(_x - _h, _y)) / (2 * _h)   # 动 x、冻 y
        _dfy = (_f(_x, _y + _h) - _f(_x, _y - _h)) / (2 * _h)   # 动 y、冻 x
        _rows.append(
            f"| ({_x:g}, {_y:g}) | {2 * _x * _y:g} | {_dfx:.6f} | {_x ** 2:g} | {_dfy:.6f} |"
        )
    _table = "\n".join(_rows)

    mo.vstack([
        _fig,
        mo.md(
            f"""
            | 求导点 (x, y) | ∂f/∂x 解析 = 2xy | ∂f/∂x 数值 | ∂f/∂y 解析 = x² | ∂f/∂y 数值 |
            |---|---|---|---|---|
            {_table}

            六位小数全部吻合。要翻到小数点后第 11 位才能看出差别
            （比如 12.000000000078…）——这不是截断误差，而是纯纯的**浮点舍入**：
            对二次函数，对称差分在代数上是精确的（h 的平方项恰好相消，
            第 14 课演示 1 会把这个彩蛋摊开讲），剩下的只有浮点运算的毛刺。
            """
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 🎛️ 交互实验

        **切线滑板**——两个滑块，一次讲透"割线 → 切线"：

        1. **切点 x₀**（-2 ~ 2）：拖动它，切线（绿色粗线）跟着点在曲线上滑，
           斜率 = 2x₀ 实时变化——"导数逐点不同"的动态版
        2. **割线步长 h**（0.01 ~ 1）：橙色虚线连接曲线上 x₀ 和 x₀+h 两个点，
           是货真价实的**割线**（斜率 = 前向差分）

        建议做两个实验：

        - 固定 h = 1，拖 x₀：看割线与切线的夹角在陡的地方特别扎眼
        - 固定 x₀ = 0.5，把 h 从 1 慢慢调到 0.01：**亲眼看着割线转过去贴上
          切线**——"h → 0 时割线变切线"，导数定义的动态版

        顺手拖到 x₀ = 0 看一眼：切线变成水平的 y = 0（斜率 2×0 = 0）——
        碗底/谷底的导数归零，这正是演示 4 里"越靠近碗底、步子越小，最后几乎停在碗里"的原因。
        """
    )
    return


@app.cell
def _(mo):
    # 交互控制：切点 x0 与割线步长 h（debounce=True：松手才重算，画面不闪烁）
    x0_slider = mo.ui.slider(-2, 2, value=0.5, step=0.1, debounce=True,
                             label="🎛️ 切点 x₀（-2 ~ 2）")
    h_slider = mo.ui.slider(0.01, 1, value=0.5, step=0.01, debounce=True,
                            label="🎛️ 割线步长 h（0.01 ~ 1）")
    mo.vstack([x0_slider, h_slider])
    return (h_slider, x0_slider)


@app.cell
def _(h_slider, mo, np, plt, x0_slider):
    # ===== 交互实验 · 切线滑板（响应单元）=====
    _x0 = x0_slider.value              # 切点
    _h = h_slider.value                # 割线的"第二点"离切点多远
    _f = lambda x: x ** 2              # f(x) = x²

    _kt = 2 * _x0                      # 切线斜率：f'(x₀) = 2x₀（精确值）
    _itc = _x0 ** 2 - _kt * _x0        # 切线截距：算出来就是 -x₀²
    # 割线斜率用"前向差分"而不是对称差分：割线要连接曲线上真实存在的两点
    # (x₀, f(x₀)) 和 (x₀+h, f(x₀+h))——对称差分的 x₀-h 那个点看不见，画不出线
    _ks = (_f(_x0 + _h) - _f(_x0)) / _h     # 割线斜率 = 前向差分
    _diff = abs(_ks - _kt)             # 割线与切线的斜率差——对 x² 来说恰好 = h

    _x = np.linspace(-3, 3, 400)       # 画曲线与两条直线的公共 x 轴
    _fig, _ax = plt.subplots(figsize=(7, 4.8))
    _ax.plot(_x, _f(_x), color="dimgray", lw=2.2, label="f(x) = x²", zorder=2)
    _ax.plot(_x, _kt * _x + _itc, color="tab:green", lw=3, alpha=0.9,
             label=f"切线（斜率 2·x0 = {_kt:+.2f}）")
    _ax.plot(_x, _f(_x0) + _ks * (_x - _x0), "--", color="tab:orange", lw=2.2,
             label=f"割线（斜率 = {_ks:+.2f}）")
    _ax.scatter([_x0], [_f(_x0)], s=60, color="tab:red", zorder=4,
                label="切点 (x0, f(x0))")
    _ax.scatter([_x0 + _h], [_f(_x0 + _h)], s=45, color="tab:orange", zorder=4,
                label="割线第二点 (x0+h, f(x0+h))")

    _ax.set_xlim(-3.3, 3.3)            # x 轴范围：盖住最远的割线第二点 (2+1, 9) 且不切 marker
    _ax.set_ylim(-0.6, 9.8)            # y 轴范围
    _ax.grid(True, alpha=0.3)
    _ax.axhline(0, color="gray", lw=0.8)   # x 轴
    _ax.axvline(0, color="gray", lw=0.8)   # y 轴
    _ax.legend(loc="upper center")
    _ax.set_title("切线滑板：h 越小，割线越贴切线")

    # 一句话判词：割线贴上切线了吗？（阈值 0.05：肉眼基本看不出夹角）
    _verdict = (
        f"✅ 割线已经贴上切线（斜率差 {_diff:.2f} < 0.05）——继续调小 h，"
        "看两条线彻底合拢"
        if _diff < 0.05
        else f"⏳ 割线还没贴上切线（斜率差 {_diff:.2f}），把 h 调小试试"
    )

    mo.vstack([
        _fig,
        mo.md(
            f"""
            | 量 | 值 |
            |---|---|
            | 切点 x₀ | **{_x0:+.1f}** |
            | f(x₀) = x₀² | **{_f(_x0):.2f}** |
            | 切线斜率 f'(x₀) = 2x₀ | **{_kt:+.2f}** |
            | 割线斜率（前向差分） | **{_ks:+.2f}** |
            | 两者之差（理论值恰好 = h = {_h:.2f}） | **{_diff:.2f}** |

            切线方程：y = f(x₀) + f'(x₀)·(x − x₀)，代进去就是
            **y = {_kt:+.2f}·x + ({_itc:+.2f})**（截距 = -x₀²）。

            割线斜率 = (f(x₀+h) − f(x₀)) / h，把 (x₀+h)² 展开一算：
            它和切线斜率的差**恰好是 h**——f(x) = x² 的专属福利，可在纸上验证。
            """
        ),
        mo.callout(mo.md(f"**{_verdict}**"), kind="info"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✏️ 动手练习（每课必做，本课 3 题）": mo.md(
                """
                **练习 1 · 手算导数**：f(x) = x³ 的导数是 f'(x) = 3x²（幂函数法则：
                指数落下来当系数、指数减一）。x = 2 处的切线斜率是多少？
                填进 `slope_ex1`，再用对称差分（h = 1e-5）数值验证，填进 `num_ex1`。

                **练习 2 · 写一台数值求导器**：补完 `numerical_grad(f, xs, h=1e-5)`——
                f 是一元函数、xs 是数组，返回每个点的数值导数数组（06 课广播一步到位）。
                用它验证"sin 的导数是 cos"在 x = 0、π/4、π/2 三处的值。

                **练习 3 · 梯度指哪**：f(x, y) = x² + y² 在点 (1, 1) 的梯度是多少
                （2 维数组）？再用点积验证它确实指向远离原点的上坡一侧（bool）。

                规则不变：先把 `None` 换成你的答案再运行；写完再看参考答案。
                """
            )
        }
    )
    return


@app.cell
def _(mo):
    # ===== 练习 1 · 手算导数 =====
    # f(x) = x³，f'(x) = 3x²（幂函数法则：指数落下来当系数、指数减一）
    #   → x = 2 处：f'(2) = 3 × 2² = ？
    # 数值验证（对称差分，h = 1e-5）：
    #   num = ( (2 + 1e-5)³ - (2 - 1e-5)³ ) / (2 × 1e-5)
    #   在纸上展开：(2+h)³ = 8 + 12h + 6h² + h³，(2-h)³ = 8 - 12h + 6h² - h³
    #   相减 = 24h + 2h³，除以 2h = 12 + h²——和手算只差一个 h² = 1e-10，
    #   亲手摸一次"截断误差 ∝ h²"

    slope_ex1 = None    # ← 手算：f(x) = x³ 在 x = 2 处的导数（一个数）
    num_ex1 = None      # ← 数值验证：对称差分算出的导数（h = 1e-5）

    _msg = (
        f"你的答案：f'(2) = **{slope_ex1}**，对称差分数值 = **{num_ex1}**\n\n"
        "（两个数应该几乎一样——要到小数点后第 10 位才见分晓。）"
        if slope_ex1 is not None and num_ex1 is not None
        else "⏳ 先手算 slope_ex1（3 × 2² = ?），再把对称差分公式代进 num_ex1"
             "（提示：** 就是乘方）"
    )
    mo.md(_msg)
    return (num_ex1, slope_ex1)


@app.cell
def _(mo, np):
    # ===== 练习 2 · 写一台数值求导器 =====
    # 三步提示：
    #   1. f(xs + h)：数组加标量 → 同形数组（06 课广播）
    #   2. f 作用在数组上会逐元素求值（np.sin 是矢量化函数，03 课）
    #   3. 两个同形数组相减、再除以标量 2h——一行广播到底
    def numerical_grad(f, xs, h=1e-5):
        """一元函数 f 在数组 xs 各点的数值导数（对称差分），返回同形数组"""
        _plus = None    # ← f(xs + h)：数组整体加标量（06 课广播）
        _minus = None   # ← f(xs - h)
        grad = None     # ← 对称差分：(_plus - _minus) / (2h)
        return grad

    # 验证素材："sin 的导数是 cos"——在 x = 0、π/4、π/2 三处对答案
    _xs = np.array([0.0, np.pi / 4, np.pi / 2])
    # 期望值（cos）：1.0、≈ 0.7071、0（cos(π/2) 数学上恰好是 0）
    grad_ex2 = None    # ← 换成 numerical_grad(np.sin, _xs)

    _msg = (
        f"你的求导器：**{np.round(grad_ex2, 8)}**\n\n"
        f"精确值 cos：**{np.cos(_xs)}**\n\n"
        "前两处分毫不差；第三处最有趣：你算出干净的 0，而 np.cos(np.pi/2) "
        "吐出 6.12e-17——09 课的浮点残渣（同是 1e-17 量级的那位）换了个地方出现。"
        if grad_ex2 is not None
        else "⏳ 先把 numerical_grad 里的三个 None 补完，"
             "再把 grad_ex2 换成 numerical_grad(np.sin, _xs)"
    )
    mo.md(_msg)
    return (grad_ex2, numerical_grad)


@app.cell
def _(mo, np):
    # ===== 练习 3 · 梯度指哪 =====
    _p = np.array([1.0, 1.0])   # 站在碗壁上的这个点
    # 从原点指向 (1, 1) 的方向，就是"远离碗底"的上坡方向

    # f(x, y) = x² + y² → ∇f = (2x, 2y)。代入 (1, 1)：
    # 两个分量分别把 x 和 y 代进 2x、2y 就行——梯度不是什么新算子，
    # 就是把每根坐标轴方向的导数排成一排（概念 2）
    grad_ex3 = None    # ← 一个 2 维数组，比如 np.array([?, ?])

    # 方向验证：∇f 与 _p 的点积 > 0，说明两者夹角小于 90°
    # （11 课：cosθ > 0 → 大体同向），即梯度确实指向上坡一侧：
    uphill_ex3 = None  # ← bool：grad_ex3 @ _p > 0
    # 加餐（不改代码，心里答）：这个梯度的长度（范数）是多少？
    # 它和点 (1, 1) 离碗底的距离是什么关系？（提示：2 × 距离，演示 3 讲过）

    _msg = (
        f"你的答案：∇f(1, 1) = **{grad_ex3}**，指向上坡一侧 = **{uphill_ex3}**"
        if grad_ex3 is not None and uphill_ex3 is not None
        else "⏳ 先填 grad_ex3（用 np.array 造一个 2 维数组），"
             "再用点积判定 uphill_ex3（一个 bool）"
    )
    mo.md(_msg)
    return (grad_ex3, uphill_ex3)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 1 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                slope_ex1 = 3 * 2 ** 2                          # = 12
                num_ex1 = ((2 + 1e-5) ** 3 - (2 - 1e-5) ** 3) / (2e-5)   # ≈ 12.0000000002
                ```

                f'(2) = 3 × 2² = **12**。对称差分的分子展开是 24h + 2h³，
                除以 2h 得 **12 + h²**——理论上 12.0000000001，
                实际算出 12.0000000002（多出来的尾巴是浮点舍入）。

                这正是演示 2 的"截断误差 ∝ h²"：h 缩小 10 倍，误差缩小 100 倍。
                对称差分比前向差分（误差只有 h 级）准一个数量级起步——
                这就是概念 1 推荐它的原因。

                顺带一提：切线如果真要画出来，就是过 (2, 8)、斜率 12 的直线
                y = 12x - 16——和演示 1 同一个点斜式，换了主角而已。
                """
            ),
            "✅ 练习 2 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                def numerical_grad(f, xs, h=1e-5):
                    _plus = f(xs + h)                  # 06 课广播：数组 + 标量
                    _minus = f(xs - h)
                    grad = (_plus - _minus) / (2 * h)  # 逐元素除以标量，一步全出
                    return grad

                grad_ex2 = numerical_grad(np.sin, np.array([0.0, np.pi / 4, np.pi / 2]))
                ```

                结果 **[1.0, 0.70710678, 0.0]**，对照 cos 的
                [1.0, 0.70710678, 6.12e-17]——前两处严丝合缝，
                第三处你的机器反而更"干净"。

                这台求导器由三个旧零件组装：数组 ± 标量（06 课广播）、np.sin
                逐元素求值（03 课矢量化）、除以标量（06 课）。它对任何一元函数
                都生效——练习 1 的 x³ 也能喂：
                `numerical_grad(lambda x: x ** 3, np.array([2.0]))`。
                """
            ),
            "✅ 练习 3 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                grad_ex3 = np.array([2.0, 2.0])                     # ∇f(1,1) = (2×1, 2×1)
                uphill_ex3 = bool(grad_ex3 @ np.array([1.0, 1.0]) > 0)   # 2+2=4 > 0 → True
                ```

                (2, 2) 与 (1, 1) 的点积 = 4 > 0 → 夹角小于 90°，梯度确实指往碗外。

                再进一步：在 (1, 1) 处沿单位方向 u 走一小步，函数增量 ≈ ∇f·u =
                |∇f|·cosθ（11 课点积的几何真身）——cosθ = 1、即 u 与 ∇f 同向时
                涨得最快。**"梯度指向最陡上坡"不是口号，是 cosθ 的推论。**
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
                1. 导数 = 瞬时变化率 = 切线斜率：平均变化率 (f(x+h)-f(x))/h 在
                   h → 0 时的归宿；f(x) = x² 的导数是 2x，每个点脾气不同
                2. 数值求导用对称差分 (f(x+h)-f(x-h))/(2h)：误差 h² 级，
                   比前向差分（h 级）准
                3. h 的分寸是 U 型曲线：右端截断误差（割线没贴上切线）、左端浮点
                   噪声（09 课残渣被 2h 放大）；经验值 h ≈ 1e-5 ~ 1e-7
                4. 梯度 ∇f = 偏导排成的向量（11 课：向量是箭头）：方向 = 最陡上坡，
                   长度（范数）= 陡峭程度
                5. 等高线图上，梯度处处垂直于等高线、指向函数值增大的一侧（碗外）
                6. 心跳公式 x ← x - lr·∇f：沿负梯度下山，越近碗底步子越小
                   （自带刹车）——第 15 课它就是第一个训练循环
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

        **下一课预告** — `14_链式法则`：

        今天，一个函数的梯度会算了。但神经网络从来不是"一个函数"——
        它是**函数套函数套函数**的俄罗斯套娃：x 进第一层、第一层的输出进第二层、
        ……层层嵌套，最后才到损失。想对最外层求导，就得一环扣一环把导数传进去——
        这就是**链式法则**，反向传播的引擎。

        下一课把"套娃求导"拆成齿轮逐级传动：外层的导数 × 内层的导数 × 再内层的
        导数……一路乘到最里面。看懂它，反向传播的门票就到手了。

        💓 心跳已经响起，去把齿轮咬合上！
        """
    )
    return


if __name__ == "__main__":
    app.run()
