import marimo

__generated_with = "0.25.2"
app = marimo.App(app_title="03 · 创建数组的多种方式", width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import numpy as np
    return (np,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # 03 · 创建数组的多种方式

        **从零到 Transformer · 手写深度学习** ｜ 第 1 部分 · NumPy 从零开始（3/10）
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📖 本课目标（先读我）": mo.md(
                """
                **这一课只解决一个问题：数组从哪里来？**

                上一课我们解剖了 ndarray：shape、dtype、ndim 都认过了，但数组本身还没亲手造过。
                今天补上这块地基。学完你将能：

                - 用 **8 个函数**随心创建数组：`array` / `zeros` / `ones` / `full` / `empty` / `arange` / `linspace` / `eye`
                - 面对任何场景，立刻选出对的那个函数（小结里附"选哪个"小抄）
                - 避开两个新手坑：`empty` 里的"内存垃圾"、浮点步长 `arange` 的"点数不准"

                内容不多，但每个函数都会亲手运行一遍——创建数组，是后面 45 课每天都在做的事。
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📋 概念 1/3 · 从已有数据来：np.array 与嵌套列表": mo.md(
                """
                最直接的场景：**数据已经在手里**（一个 Python 列表），转成数组就能享受 NumPy 的全部本领。

                - 普通列表 → 1 维数组：`np.array([1, 2, 3])`
                - 嵌套列表 → 多维数组：**嵌套几层，就是几维**——两层是 2 维（一张表），三层是 3 维（一块千层饼）
                - **必须"规矩"**：每行的长度必须一致，才能拼成整齐的矩形；长短不齐 NumPy 直接报错（马上演示）

                创建时还可以用 `dtype` 参数指定类型：`np.array([1, 2, 3], dtype=np.float64)`。
                不指定的话，NumPy 会**自动推断**：

                - 全是整数 → `int64`
                - 混进任何一个小数 → 全体"升级"成 `float64`（整数 1 变成 1.0）

                这正是第 01 课"同类型纪律"的体现：数组必须同类型，所以一个老鼠坏一锅汤——
                只要混进一个小数，其他元素都得跟着变形。
                """
            ),
            "🧱 概念 2/3 · 凭空造：模具类与序列类": mo.md(
                """
                更多时候数据还没到手，我们要"凭空"造数组。两大流派：

                **模具类——先定形状，再灌内容**（形状是元组，如 `(2, 3)` = 2 行 3 列）：

                - `np.zeros((2, 3))`：全 0。深度学习的"空白画布"几乎都从全 0 开始
                - `np.ones((2, 3))`：全 1
                - `np.full((2, 3), 7)`：全 7——填什么值你说了算
                - `np.empty((2, 3))`：**只分配内存，不填值**。最快，但里面是"上任住客"留下的内存垃圾，
                  用之前必须每个位置都亲自赋值——**新手陷阱**，现阶段请用 zeros 代替

                **序列类——按规律生成一串数**：

                - `np.arange(起点, 终点, 步长)`：等差数列，**左闭右开**（含起点、不含终点），规则和 `range` 一致
                - `np.linspace(起点, 终点, 个数)`：在区间里**均匀取指定个数**的点，**两端都包含**

                一句话区分：arange 说"隔多远取一个"，linspace 说"一共要几个"。
                """
            ),
            "📏 概念 3/3 · arange 与 linspace 怎么选 + 浮点陷阱": mo.md(
                """
                **口诀：知道步长用 arange，知道个数用 linspace。**

                但有个坑：**浮点数步长的 arange 不可靠**。

                原因：0.1 这样的"干净小数"在二进制里是无限循环小数（就像十进制的 1/3 = 0.333...），
                只能近似存储。步长一次次累加，误差也跟着累积，最坏的结果是**点数多一个或少一个**，
                连"右开"的承诺都可能失效（下面马上实测给你看）。所以：

                > **浮点等距采样，一律优先 linspace**——个数和两端都有保证。

                最后混个脸熟：`np.eye(n)` 生成**单位矩阵**（对角线全 1、其余全 0），`np.identity(n)`
                是它的别名。单位矩阵在线代里的地位相当于数字里的 1：任何矩阵乘它都不变。
                它是第 09 课（线性代数）的主角，今天先认识一下就行。
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

        8 个创建函数，分四组出场，最后来一场综合演练：

        1. **从列表创建**：`np.array`——数据已在手
        2. **模具类**：`zeros` / `ones` / `full` / `empty`——先定形状再灌内容
        3. **序列类**：`arange` / `linspace`——按规律生成一串数
        4. **矩阵类**：`eye`——单位矩阵

        每个单元直接运行就能看效果，鼓励随手改数字。👉
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 1️⃣ 从列表创建：np.array

        手里已经有数据（Python 列表）时，`np.array` 是最直接的"翻译官"：
        列表翻一层 → 1 维，嵌套翻两层 → 2 维。重点观察 dtype 的自动推断。
        """
    )
    return


@app.cell
def _(mo, np):
    # 从 Python 列表创建：手里已有数据时的第一选择
    arr_int = np.array([1, 2, 3])       # 普通列表 → 1 维数组

    arr_nested = np.array([[1, 2, 3],   # 嵌套列表 → 2 维数组（嵌套几层 = 几维）
                           [4, 5, 6]])  # 外层 2 个元素 = 2 行；内层各 3 个元素 = 3 列

    arr_mixed = np.array([1, 2, 3.14])  # 混进一个小数，会发生什么？

    # 展示用 mo.plain_text(...)：按"原样"显示文本（等宽字体、保留换行），最适合摆数组
    mo.vstack(
        [
            mo.md("**普通列表 → 1 维**："),
            mo.plain_text(str(arr_int)),
            mo.md(
                f"shape = {arr_int.shape}，dtype = `{arr_int.dtype}`——全是整数，自动推断为 int64"
            ),
            mo.md("**嵌套列表 → 2 维**："),
            mo.plain_text(str(arr_nested)),
            mo.md(
                f"shape = {arr_nested.shape}，dtype = `{arr_nested.dtype}`——外层 2 个元素 = 2 行，内层各 3 个 = 3 列"
            ),
            mo.md("**混入一个 3.14 之后**："),
            mo.plain_text(str(arr_mixed)),
            mo.md(
                f"dtype 变成了 `{arr_mixed.dtype}`——整数 1 和 2 被迫\"升级\"成 1.0 和 2.0"
            ),
            mo.callout(
                mo.md(
                    """
                    **同类型纪律：一个老鼠坏一锅汤。**

                    第 01 课说过：数组必须类型统一，才能挤进同一块连续内存。
                    所以只要混进一个小数，全体元素都得跟着变成 float——这不是 bug，是纪律的代价。

                    想在创建时就定死类型，可以加参数：`np.array([1, 2, 3], dtype=np.float64)`。
                    """
                ),
                kind="warn",
            ),
        ]
    )
    return (arr_int, arr_nested, arr_mixed)


@app.cell
def _(mo, np):
    # 「不规则」的嵌套列表：第一行 2 个元素，第二行只有 1 个
    # 数组必须是整齐的矩形（每行等长），拼不成矩形 NumPy 就拒绝收货：
    try:
        _bad = np.array([[1, 2], [3]])   # 故意造一个"瘸腿"的嵌套列表
        _note = "竟然成功了？（老版本 NumPy 会这样，并给出弃用警告）"
    except ValueError as _err:           # NumPy 2.x：直接抛 ValueError
        _note = " ".join(str(_err).split())   # 记下报错原文（压成一行），待会儿展示

    mo.md(
        f"""
        **试试不规矩的嵌套**：`np.array([[1, 2], [3]])`

        ```
        ValueError: {_note}
        ```

        翻译成人话：**检测到不规则的形状**——第一行 2 个元素、第二行 1 个，拼不成矩形。

        踩坑高发场景：从文件或网页读来的"看着挺整齐"的数据，某一行偷偷缺了个值——
        转数组前先检查每行的长度，补齐或清洗之后再转。
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 2️⃣ 模具类：zeros / ones / full / empty

        数据不在手，只要一块"指定形状"的数组？模具类先定形状（一个元组），
        再往里灌内容。注意 `empty` 是个坑。
        """
    )
    return


@app.cell
def _(mo, np):
    # 模具类：形状用一个元组 (行数, 列数) 指定——所以会看到两层括号，不是写错
    mold_zeros = np.zeros((2, 3))       # 全 0：深度学习最常用的"空白画布"
    mold_ones = np.ones((2, 3))         # 全 1
    mold_full = np.full((2, 3), 7)      # 全 7：填什么值由第二个参数说了算
    mold_empty = np.empty((2, 3))       # 只分配内存，不填值！

    mo.vstack(
        [
            mo.md(
                "`np.zeros((2, 3))`——全 0（默认 float64，要整数可加 `dtype=int`，上一课的老朋友）："
            ),
            mo.plain_text(str(mold_zeros)),
            mo.md("`np.ones((2, 3))`——全 1："),
            mo.plain_text(str(mold_ones)),
            mo.md(
                f"`np.full((2, 3), 7)`——全 7（7 是整数，所以 dtype 是 `{mold_full.dtype}`）："
            ),
            mo.plain_text(str(mold_full)),
            mo.md(
                "`np.empty((2, 3))`——只分配内存、不填值，里面是\"上任住客\"留下的垃圾："
            ),
            mo.plain_text(str(mold_empty)),
            mo.callout(
                mo.md(
                    """
                    **empty：快，但脏。**

                    它跳过"填值"这一步，所以比 zeros 略快——代价是里面的值**完全不可预测**：
                    多运行几次这个单元，每次结果都不一样，甚至出现 e-316 这种"天文垃圾"。

                    用 empty 的前提是：**接下来每个位置都会被你亲自赋值**。
                    新手阶段拿不准就用 zeros——慢那么一点点，换一个确定的起点，值。
                    """
                ),
                kind="warn",
            ),
        ]
    )
    return (mold_zeros, mold_ones, mold_full, mold_empty)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 3️⃣ 序列类：arange 与 linspace

        按规律生成一串数。两个函数长得像，性格完全不同：
        arange 管"隔多远取一个"（步长），linspace 管"一共要几个"（个数）。
        """
    )
    return


@app.cell
def _(mo, np):
    # arange(起点, 终点, 步长)：等差数列，左闭右开——规则和 range 一模一样
    arange_5 = np.arange(5)             # 只给终点：从 0 开始 → [0, 1, 2, 3, 4]
    arange_2_10 = np.arange(2, 10)      # 给起点终点（步长默认 1）→ [2, 3, ..., 9]
    arange_desc = np.arange(10, 0, -2)  # 负步长：从 10 往回跳，每次 -2 → [10, 8, 6, 4, 2]

    mo.md(
        f"""
        | 写法 | 结果 | 说明 |
        |---|---|---|
        | `np.arange(5)` | `{arange_5}` | 只给终点，起点默认 0 |
        | `np.arange(2, 10)` | `{arange_2_10}` | **不含终点 10**（左闭右开） |
        | `np.arange(10, 0, -2)` | `{arange_desc}` | 负步长 = 倒着数，终点 0 同样不含 |

        整数步长的 arange 完全可靠，放心用。
        但步长一旦变成小数……请看下一个单元。👇
        """
    )
    return (arange_5, arange_2_10, arange_desc)


@app.cell
def _(mo, np):
    # ⚠️ 浮点步长的 arange：亲眼看看"不可靠"长什么样
    arange_float = np.arange(0, 1, 0.1)      # 本意：0.0, 0.1, ..., 0.9 共十个刻度
    arange_trap = np.arange(0.2, 0.8, 0.2)   # 本意：0.2, 0.4, 0.6 三个数（0.8 是终点，不该出现）
    _ls_fix = np.linspace(0.2, 0.8, 4)       # 同样这段路交给 linspace：要 4 个点，就给 4 个

    mo.md(
        f"""
        **第一个问题：值本身就不干净。** 把 arange(0, 1, 0.1) 的每个数原样打印（`.tolist()`）：

        ```
        {arange_float.tolist()}
        ```

        0.30000000000000004、0.7000000000000001……因为 0.1 在二进制里是无限循环小数
        （就像十进制的 1/3 = 0.333...），存进内存时被截断，误差随着累加越滚越大。

        **第二个问题更严重：点数会骗人。** arange(0.2, 0.8, 0.2) 的实际结果：

        ```
        {arange_trap.tolist()}
        ```

        说好的"右开"呢？终点 0.8 竟然混进了队伍——**4 个点，比理论多一个**。
        原因：NumPy 内部计算"该造几个点"时，(0.8 - 0.2) / 0.2 得到的不是干净的 3，
        而是 3.0000000000000004，向上取整就成了 4。

        **同样的活儿交给 linspace**——`np.linspace(0.2, 0.8, 4)`：

        ```
        {_ls_fix.tolist()}
        ```

        要 4 个点就给 4 个点，两端不多不少。中间那个 0.6000000000000001 说明
        二进制误差谁也躲不掉——但 linspace 把**最要紧的两件事（个数、两端）焊死了**。

        > 结论：**浮点等距采样，优先 linspace**；arange 留给整数步长。
        """
    )
    return (arange_float, arange_trap)


@app.cell
def _(mo, np):
    # linspace(起点, 终点, 个数)：在闭区间 [起点, 终点] 里均匀撒"个数"个点
    ls_0_1 = np.linspace(0, 1, 5)       # 0 到 1，要 5 个点 → 间距 0.25
    ls_0_10 = np.linspace(0, 10, 5)     # 0 到 10，要 5 个点 → 间距 2.5

    mo.vstack(
        [
            mo.md("`np.linspace(0, 1, 5)`："),
            mo.plain_text(str(ls_0_1)),
            mo.md("`np.linspace(0, 10, 5)`："),
            mo.plain_text(str(ls_0_10)),
            mo.md(
                """
                两端都在队伍里（0 和 1 / 0 和 10），中间均匀分布。

                **arange 与 linspace 对比小抄：**

                | | `arange` | `linspace` |
                |---|---|---|
                | 你来指定 | **步长**（隔多远取一个） | **个数**（一共要几个点） |
                | 终点 | 不包含（左闭右开） | **包含**（两端都有） |
                | 整数等差序列 | ✅ 首选 | 也能做，但别扭 |
                | 浮点等距采样 | ⚠️ 点数可能多/少一个 | ✅ 首选 |

                名字由来：arange = array + range（按 range 的规矩造数组）；
                linspace = linear space（把线段线性等分）。
                """
            ),
        ]
    )
    return (ls_0_1, ls_0_10)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 4️⃣ 矩阵类：eye

        单位矩阵：对角线全 1、其余全 0。`np.identity(n)` 是 `np.eye(n)` 的别名——
        两个名字，一个东西。
        """
    )
    return


@app.cell
def _(mo, np):
    # eye(N) 造 N×N 单位矩阵；eye(行数, 列数) 可以不方；k 参数让对角线"斜着挪"
    mat_eye = np.eye(3)                 # 3×3：对角线全 1，其余全 0
    mat_rect = np.eye(3, 4)             # 3 行 4 列：不方也行，对角线仍从左上角出发
    mat_shift = np.eye(3, 4, k=1)       # k=1：对角线整体向右平移一格

    mo.vstack(
        [
            mo.md("`np.eye(3)`——标准单位矩阵："),
            mo.plain_text(str(mat_eye)),
            mo.md("`np.eye(3, 4)`——长方形的（3 行 4 列），对角线仍从左上角出发："),
            mo.plain_text(str(mat_rect)),
            mo.md("`np.eye(3, 4, k=1)`——对角线右移一格（k 控制偏移，往左挪用负数）："),
            mo.plain_text(str(mat_shift)),
            mo.callout(
                mo.md(
                    """
                    一句话记住它：**单位矩阵就是矩阵世界里的数字 1**——
                    任何矩阵乘以它，都保持原样（3 × 1 = 3 的矩阵版）。

                    第 09 课讲线性代数时它是绝对主角，现在混个脸熟就够了。
                    """
                ),
                kind="info",
            ),
        ]
    )
    return (mat_eye, mat_rect, mat_shift)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 5️⃣ 综合演练：搭一个 5×5 棋盘

        国际象棋棋盘的规律：**(行号 + 列号) 是偶数 → 白格(0)，奇数 → 黑格(1)**。
        用今天学的函数，把这个规律变成一块数组。
        """
    )
    return


@app.cell
def _(mo, np):
    # 棋盘规律：(行号 + 列号) % 2 == 0 → 白格(0)，否则黑格(1)
    # % 是取余数：偶数 % 2 = 0，奇数 % 2 = 1（小学除法里的"余数"）
    #
    # 双重列表推导（先看骨架，再细看每一层）：
    #   外层 [ ... for r in range(5)]          → 造 5 行
    #   内层 [(r + c) % 2 for c in range(5)]   → 每行造 5 个格子，黑白由 (r+c)%2 决定
    # 得到"列表的列表"，再交给 np.array 转成 2 维数组——演示 1 的老朋友
    board = np.array([[(r + c) % 2 for c in range(5)] for r in range(5)])

    # 再配一条"列号坐标尺"：0 1 2 3 4——arange 出场
    ruler = np.arange(5)

    # 把数组画成棋盘（纯展示的小技巧，看不懂可以跳过）：
    # 逐行把 1 换成 🟫、0 换成 ⬜，再一行行拼起来
    _board_visual = "\n".join(
        "".join("🟫" if v else "⬜" for v in row) for row in board
    )

    mo.vstack(
        [
            mo.md(f"列号坐标尺（arange 造的）：`{ruler}`"),
            mo.md("棋盘本体（0 = 白格，1 = 黑格）："),
            mo.plain_text(str(board)),
            mo.md("画出来看看（🟫 = 黑格 1，⬜ = 白格 0）："),
            mo.plain_text(_board_visual),
            mo.md(
                f"""
                数一数这个小世界用了几种创建方式：**np.array（棋盘本体）+ arange（坐标尺）= 2 种**。

                再回头数数这一课集齐了多少：`array` `zeros` `ones` `full` `empty` `arange` `linspace` `eye`
                ——**8 个函数**，每一个你都已经亲手运行过。

                顺带一提：黑格占比 board.mean() = {board.mean():.0%}——向量化统计，第 01 课的老朋友。
                """
            ),
        ]
    )
    return (board, ruler)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 🎛️ 交互实验

        下面两个滑块控制一次 `linspace(0, stop, num)` 采样：

        - **采样点数 num**：3 ~ 21（步长 2，保持奇数——切出来的份数是偶数，间距更好看）
        - **区间终点 stop**：1 ~ 20

        拖动它们，观察点的个数与相邻点间距怎么联动。👇
        """
    )
    return


@app.cell
def _(mo):
    num_slider = mo.ui.slider(
        3, 21,
        value=5,
        step=2,                 # 步长 2：3, 5, 7, ..., 21——保持奇数
        label="🎛️ 采样点数 num（3 ~ 21 的奇数）",
    )
    stop_slider = mo.ui.slider(
        1, 20,
        value=10,
        step=1,
        label="🎯 区间终点 stop（1 ~ 20）",
    )
    mo.vstack([num_slider, stop_slider])
    return (num_slider, stop_slider)


@app.cell
def _(mo, np, num_slider, stop_slider):
    # 响应单元：滑块一动，这里自动重算——这就是 marimo 的响应式
    pts = np.linspace(0, stop_slider.value, num_slider.value)   # 两端都包含

    # 相邻点间距 = 后一个点 - 前一个点：
    #   pts[1:]  = 去掉第 0 个点（从第 2 个点到最后）
    #   pts[:-1] = 去掉最后一个点（从第 1 个点到倒数第 2 个）
    # 两个等长数组相减 = 对应位置逐个相减（第 01 课的向量化）。
    # 方括号这种"切片"写法是下一课的主角，现在先照抄这一行、看结果就好
    gaps = pts[1:] - pts[:-1]

    # 间距的理论值：区间长度 ÷ 间隔段数——n 个点把区间切成 n - 1 段
    _spacing_theory = stop_slider.value / (num_slider.value - 1)

    mo.vstack(
        [
            mo.md(
                f"当前：从 **0 到 {stop_slider.value}**，均匀取 **{num_slider.value}** 个点："
            ),
            mo.plain_text(str(pts)),
            mo.md(f"相邻点的间距（共 {num_slider.value - 1} 段）："),
            mo.plain_text(str(gaps)),
            mo.md(
                f"""
                - 理论间距 = 区间长 ÷ (点数 - 1) = {stop_slider.value} ÷ ({num_slider.value} - 1) ≈ **{_spacing_theory:.4g}**
                - **0 和 {stop_slider.value} 都在队伍里**——这就是 linspace 的"两端都包含"
                - 间距若是一串 6 或 7 结尾的小数（比如 0.16666...），那是二进制的正常现象，不是 bug

                两个观察作业：

                1. num 从 3 拖到 21：点越来越密，间距按 1/(num - 1) 的节奏变小
                2. stop 从 1 拖到 20：区间变长，同样的点数下间距成倍变大
                """
            ),
        ]
    )
    return (pts, gaps)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✏️ 动手练习（每课必做，本课 3 题）": mo.md(
                """
                练习代码就在下面几个单元里，改完直接运行（`Shift + Enter`）。
                做完——或实在做不出来——再展开参考答案对照。

                - **练习 1 · 全 7 矩阵**：用 `full` 和 `ones + 乘法`两种方式造 3×3 全 7 数组
                - **练习 2 · 倒序偶数**：生成 [20, 18, 16, ..., 2]
                - **练习 3 · 预测题**：先笔头预测 `linspace(0, 1, 4)`，再运行对照
                """
            )
        }
    )
    return


@app.cell
def _(np):
    # 练习 1：用两种方式创建 3×3 的"全 7"数组，验证它们等价
    #
    # 方式一：np.full——模具类，一步到位（提示：形状是元组 (3, 3)，填充值是 7）
    # 方式二：np.ones 先造全 1，再整体乘 7——"整体乘"是第 01 课的向量化思维
    #
    # 写好后运行，看最后一行的判断结果

    seven_full = None    # ← 方式一：把 None 换成 np.full(...)
    seven_ones = None    # ← 方式二：把 None 换成 np.ones(...) * 7

    if seven_full is None or seven_ones is None:
        ex1_msg = "（还有 None——把两个 None 都换成你的代码，再运行一次）"
    else:
        # np.array_equal(a, b)：逐元素比较，全部相同才返回 True（工具函数，知道即可）
        ex1_msg = f"两种方式的结果一样吗？→ {np.array_equal(seven_full, seven_ones)}"
    ex1_msg
    return (ex1_msg, seven_full, seven_ones)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 1 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                seven_full = np.full((3, 3), 7)
                seven_ones = np.ones((3, 3)) * 7
                ```

                值完全一样，都是 3×3 的全 7。但用 `.dtype` 查一下会发现小差别：

                - `np.full((3, 3), 7)` → int64（7 是整数，模具照着灌）
                - `np.ones((3, 3)) * 7` → float64（ones 默认造 float，乘完还是 float）

                两种思路都常见：full 直接；"先造模具再整体缩放"是深度学习里更惯用的思路
                （比如造一个全 0.1 的底板：`np.ones(shape) * 0.1`）。
                """
            )
        }
    )
    return


@app.cell
def _(np):
    # 练习 2：生成倒序偶数序列 [20, 18, 16, ..., 2]
    #
    # 提示（先在纸上推一遍再动手）：
    #   np.arange(起点, 终点, 步长)，倒序 = 负步长
    #   起点是 20，步长是 -2
    #   终点"不包含"：想让 2 留在队伍里，终点必须严格小于 2（写 0 或 1 都对；写 2 反而会漏掉 2）
    #
    # 挑战（可选）：再想一个"顺序生成再翻转"的思路——翻转要用切片 [::-1]，那是下一课的招

    even_desc = None   # ← 把 None 换成你的 np.arange(...)

    even_desc if even_desc is not None else "（还是 None——把 None 换成你的 np.arange(...)，再运行一次）"
    return (even_desc,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 2 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                even_desc = np.arange(20, 0, -2)
                ```

                结果：`[20 18 16 14 12 10 8 6 4 2]`，一共 10 个数。

                - 负步长时"左闭右开"依然成立，只是方向反了：从起点 20 往下走，一旦不比终点大就停
                - 终点写 0 和写 1 结果相同（2 比 0 和 1 都大，都能入队）；写 2 就漏掉了 2 本身（2 不比 2 大）
                - 另一个思路：`np.arange(2, 22, 2)[::-1]`——正着造 [2..20] 再整个翻转，
                  `[::-1]` 是下一课的切片招式，先记住有这回事
                """
            )
        }
    )
    return


@app.cell
def _(np):
    # 练习 3（预测题）：运行之前，先动笔！
    #
    # 第 1 步：不许先运行！在下面这行注释里写下你的预测（写错了才最有意思）：
    #   我的预测：np.linspace(0, 1, 4) = [                                      ]
    #
    # 第 2 步：把 None 换成 np.linspace(0, 1, 4)，运行，对照你写的
    #
    # 提示：linspace"两端都包含"；4 个点把区间切成了几段？

    pred_ls = None   # ← 把 None 换成 np.linspace(0, 1, 4)

    pred_ls if pred_ls is not None else "（还是 None——先在注释里写下预测，再换掉 None 运行对照）"
    return (pred_ls,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 3 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                pred_ls = np.linspace(0, 1, 4)
                ```

                结果：`[0.  0.33333333  0.66666667  1. ]`

                - 两端的 0 和 1 都在；中间均匀插 2 个点
                - 间距 = (1 - 0) / (4 - 1) = 1/3 ≈ 0.333...
                - 最常见的两个错判：以为终点不含 1（漏了 1.0）；以为间距是 1/4 = 0.25（把 4 个点当成了 4 段）

                记住公式：**n 个点，把区间切成 n - 1 段**。
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📝 本课小结（展开自查：6 条都答得上来吗？）": mo.md(
                """
                1. `np.array(列表)`：嵌套几层就是几维；必须"规矩"（每行等长），否则 ValueError
                2. dtype 自动推断：全整数 → int64，混入小数 → 全体 float64——同类型纪律的体现
                3. 模具类 `zeros` / `ones` / `full(形状, 值)`：先定形状再灌内容；
                   `empty` 只分配内存不填值（内存垃圾），新手阶段用 zeros 更稳
                4. `arange(起, 止, 步长)` 左闭右开（同 range）；`linspace(起, 止, 个数)` 两端都包含
                5. 浮点步长的 arange 连点数都不可靠（0.1 在二进制里存不准）——**浮点等距采样优先 linspace**
                6. `eye(n)` 单位矩阵 = 矩阵世界里的"1"，第 09 课的主角

                **选哪个函数？（小抄）**

                | 场景 | 用它 |
                |---|---|
                | 数据已经在手里（列表） | `np.array(数据)` |
                | 要一块全 0 的空白画布 | `np.zeros(形状)` |
                | 要全 1 的底板（乘法单位元） | `np.ones(形状)` |
                | 要全是指定值（全 7、全 -1...） | `np.full(形状, 值)` |
                | 整数等差、已知步长 | `np.arange(起, 止, 步长)` |
                | 已知要几个点（尤其浮点） | `np.linspace(起, 止, 个数)` |
                | 单位矩阵 | `np.eye(n)` |
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

        **下一课预告** — `04_索引与切片`：

        会造数组了，接下来学"取"：从一个大数组里精确取出你要的那一块——
        一个元素、一行、一列、一块小矩形，全靠方括号搞定。

        还要提前打预防针：切片里有**视图与副本**的大坑——多少人改了半天才发现
        自己改的是"影子"，debug 到半夜。那个坑，下一课带你亲眼看见、亲手跨过去。

        🧱 地基又厚了一层。去第 04 课！
        """
    )
    return


if __name__ == "__main__":
    app.run()
