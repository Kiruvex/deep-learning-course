import marimo

__generated_with = "0.25.2"
app = marimo.App(app_title="10 · 随机数与随机种子", width="medium")


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
        # 10 · 随机数与随机种子

        **从零到 Transformer · 手写深度学习** ｜ 第 1 部分 · NumPy 从零开始（10/10）
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📖 本课目标（先读我）": mo.md(
                """
                **这一课只解决一个问题：怎么让"随机"变得可信、可复现？**

                上一课的 `X @ W + b` 里其实藏着一个没回答的问题：**W 的初始值从哪来？**
                ——答案是：随机生成。线性代数是"确定"的数学，同样的输入永远得到同样的输出；
                这一课我们头一回主动邀请"不确定性"进场。

                深度学习离不开随机——权重要随机初始化、数据要随机打乱、dropout 要随机丢弃……
                但**失控的随机 = 没法调试的灾难**：跑一次好、再跑一次差，你分不清是代码的功劳还是运气的功劳。
                本课教你用"种子"给随机套上缰绳。

                学完你将能：

                - 解释**伪随机**：为什么同一个种子永远得到同一个序列——这不是缺陷，是超能力
                - 用现代 API `np.random.default_rng(42)` 生成均匀、正态、整数随机数
                - 用 `shuffle` 原位打乱数据、用 `choice` 无放回抽样——第 17 课划分数据集的两块基石
                - 说清为什么论文和教程的第一行总是 `seed = 42`

                本课只需要前九课的基础（会创建数组、会比较和求和），我们从零开始讲。
                顺便说：这是第 1 部分的**收官课**——文末有一场小小的毕业典礼等你。
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            '🎲 概念 1/3 · 伪随机：确定性的"随机"': mo.md(
                """
                计算机不会掷骰子——它没有手。你拿到的每一个"随机数"，
                都是**算法算出来的**：

                > 种子（seed）→ 一套确定的复杂变换 → 一串看起来杂乱无章的数

                关键性质：**同一个种子，永远得到同一个序列**。
                种子 42 配上这套算法，第一万次运行和第一次运行，吐出的数一模一样——
                所以它不是真随机，叫**伪随机**（pseudo-random）。

                真随机得去物理世界找：原子衰变的时刻、电路里的热噪声、大气涨落……
                那些才是"宇宙亲手掷的骰子"。

                **但这不是缺陷，是超能力。** 科学实验和调试最怕"每次都不一样"：
                结果一旦不可复现，改进就无从谈起、bug 就无从定位。
                伪随机给了你"**可复现的随机**"——随机的外观、确定的心，鱼和熊掌兼得。
                """
            ),
            "🔑 概念 2/3 · 现代 API：default_rng 与种子": mo.md(
                """
                NumPy 当代的用法是**两步走**：先造"发生器"，再问它要数。

                ```python
                rng = np.random.default_rng(42)   # ① 先造一个随机数发生器，42 是种子
                rng.random(5)                     # ② 再问它要数：5 个 [0, 1) 的均匀随机数
                ```

                `rng` 是 random number generator（随机数发生器）的缩写。
                它像一台内部带计数器的机器：**每问它要一次数，内部状态就前进一步**——
                所以同一个 rng 连续要两次，拿到的是序列里前后两段，各不相同；
                但只要用同一个种子**重建**一个 rng，整条序列就会从头**完全重演**。

                老代码里常见另一种风格：

                ```python
                np.random.seed(42)      # 设置全局种子
                np.random.rand(5)       # 从"全局水龙头"接水
                ```

                认得它就行，**新代码别学它**。理由：全局状态像**公共水龙头**——
                A 处拧一下阀门，B 处的水流跟着变，谁也不知道谁动了它，测试与复现都很痛苦；
                `rng` 像**专属水壶**——你的水壶只归你，别人倒空了也不影响你。
                本课程从今天起只用 rng 风格。
                """
            ),
            "🧰 概念 3/3 · 常用分布与深度学习用途对照表": mo.md(
                """
                | 函数 | 产出 | 深度学习里的用途 |
                |---|---|---|
                | `rng.integers(1, 7, size)` | 1~6 的整数（**左闭右开**，不含 7） | 数据增强、随机采样 |
                | `rng.random(n)` | n 个 [0, 1) 的均匀小数 | **dropout 的掩码**、初始化的原料 |
                | `rng.normal(μ, σ, shape)` | 均值 μ、标准差 σ 的正态分布 | **He / Xavier 权重初始化**（第 25 课） |
                | `rng.uniform(低, 高, shape)` | 指定区间内的均匀分布 | 数据增强的随机偏移、超参搜索 |
                | `rng.shuffle(arr)` | **原位**打乱（没有返回值！） | 每个 epoch 前打乱训练数据 |
                | `rng.choice(arr, size, replace=False)` | 无放回抽样 | **划分训练/验证集**、采 mini-batch |

                两个高频提醒：

                - `normal` 的第二个参数是**标准差 σ**，不是方差 σ²——写错会安静地错出好几倍
                - `integers` 和 `arange` 一个脾气：**左闭右开**——想掷骰子必须写 `integers(1, 7)`
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

        六个演示，从"魔术"一路玩到"深度学习为什么在乎"：

        1. **复现魔术**：同一个种子，两次生成，一个数都不差
        2. **状态前进**：同一个 rng 连续要两次数，为什么不同？（磁带与倒带键）
        3. **掷骰子**：`rng.integers`——又是左闭右开
        4. **身高直方图**：正态分布的钟形，用字符画亲眼看
        5. **shuffle 与 choice**：打乱与抽样，数据科学的日常
        6. **种子的分量**：初始化不同，训练轨迹可能完全不同

        每个单元都可以直接运行；**大胆改种子和参数**——看看随机世界怎么回应你。
        """
    )
    return


@app.cell
def _(mo, np):
    # ===== 演示 1 · 复现魔术：同一个种子 → 同一个序列 =====

    # 第 1 步：用种子 42 造一个"随机数发生器"（rng = random number generator）
    _rng_a = np.random.default_rng(42)

    # 问它要 5 个 [0, 1) 之间的均匀随机数
    _draw_a = _rng_a.random(5)                    # → [0.774 0.439 0.859 0.697 0.094]

    # 第 2 步：用种子 42 【重新】造一个全新的 rng
    #          它对刚才发生的事一无所知——初始状态和第一个 rng 完全相同
    _rng_b = np.random.default_rng(42)
    _draw_b = _rng_b.random(5)                    # → 一模一样的 5 个数！

    # 第 3 步：换个种子 43 再造一个——立刻换一批数
    _rng_c = np.random.default_rng(43)
    _draw_c = _rng_c.random(5)                    # → 完全不同的 5 个数

    # 铁证：np.array_equal 逐元素比较两个数组（第 04 课的老朋友）
    _same = np.array_equal(_draw_a, _draw_b)      # 同种子 → True
    _diff = not np.array_equal(_draw_a, _draw_c)  # 换种子 → True（意思是"确实不同"）

    # 把数组格式化成一串 3 位小数，方便放进下面的表格
    def _fmt(_arr):
        return "  ".join(f"{_v:.3f}" for _v in _arr)

    mo.vstack(
        [
            mo.md(
                f"""
                ### 演示 1 · 复现魔术：同一个种子 → 同一个序列

                | 实验 | 种子 | `rng.random(5)` 的结果 | 判决 |
                |---|---|---|---|
                | rng_a | 42 | {_fmt(_draw_a)} | 基准 |
                | rng_b（**重新造**一个） | 42 | {_fmt(_draw_b)} | 与 rng_a **一个数都不差** |
                | rng_c | 43 | {_fmt(_draw_c)} | 换了种子，**整批换新** |

                铁证（`np.array_equal` 逐元素比较）：

                - 种子 42 vs 种子 42 → **{_same}** ✅
                - 种子 42 vs 种子 43 → **{_diff}**（确实不同）
                """
            ),
            mo.callout(
                mo.md(
                    """
                    **这行代码，就是论文里那句 `seed = 42` 的全部含义。**

                    作者等于在告诉你：用同样的种子，你就能一步步"重放"他实验里出现过的
                    每一个随机数——初始化的权重、打乱的顺序、采样的索引……一个不差。
                    复现，从一行 `default_rng(42)` 开始。
                    """
                ),
                kind="success",
            ),
        ]
    )
    return


@app.cell
def _(mo, np):
    # ===== 演示 2 · rng 的内部状态：一台只往前走的放映机 =====

    # 造一个种子为 0 的 rng——此刻它的内部状态停在"序列开头"
    _rng = np.random.default_rng(0)

    # 第 1 次要 3 个数：吐出序列的第 1 段，然后内部状态前进一段
    _take1 = _rng.random(3)          # → [0.637 0.270 0.041]

    # 第 2 次再要 3 个数：从"新位置"继续吐——所以和第 1 段不同
    _take2 = _rng.random(3)          # → [0.017 0.813 0.913]

    # 对比实验：用种子 0 【重建】一个新 rng——相当于按下倒带键，状态回到开头
    _rng_rewind = np.random.default_rng(0)
    _take_rewind = _rng_rewind.random(3)     # → 又是 [0.637 0.270 0.041]！

    # 两个验证
    _moved_on = not np.array_equal(_take1, _take2)      # 同一个 rng 连续两次 → 不同
    _rewound = np.array_equal(_take1, _take_rewind)     # 重建同种子 → 回到第一段

    def _fmt(_arr):
        return "  ".join(f"{_v:.3f}" for _v in _arr)

    mo.vstack(
        [
            mo.md(
                f"""
                ### 演示 2 · 同一个 rng 的状态前进

                | 操作 | 结果 |
                |---|---|
                | `_rng = default_rng(0)` 后第 1 次 `random(3)` | {_fmt(_take1)} |
                | **同一个** `_rng` 第 2 次 `random(3)` | {_fmt(_take2)} ← 状态前进了，吐出下一段 |
                | **重建** `default_rng(0)` 后第 1 次 `random(3)` | {_fmt(_take_rewind)} ← 回到第 1 段！ |

                - 同一个 rng 连续要两次，结果不同 → **{_moved_on}**
                - 重建同种子的 rng，和第 1 次完全相同 → **{_rewound}**
                """
            ),
            mo.callout(
                mo.md(
                    """
                    **磁带类比**：一个 rng 像一盘**已经录好的磁带**——
                    整条序列在造出来那一刻就定死了；每问它要一次数，播放头就往前走一段；
                    **重建同种子的 rng = 按下倒带键**。

                    所以"随机数流"其实是"一条定死的序列"，种子只是这条序列的编号：
                    第 1 段、第 2 段……全都写死在磁带上——随机的外观，确定的心。
                    """
                ),
                kind="info",
            ),
        ]
    )
    return


@app.cell
def _(mo, np):
    # ===== 演示 3 · 掷骰子：integers，又是左闭右开 =====

    _rng_dice = np.random.default_rng(2024)

    # 掷 10 次骰子。注意：integers(1, 7) 生成的是 1~6 的整数——
    # 又是左闭右开！含 1、不含 7，和第 03 课的 arange(1, 7) 一个脾气
    _rolls10 = _rng_dice.integers(1, 7, size=10)      # → [2 5 1 2 2 2 6 5 6 6]

    # size 直接给元组 (3, 3) → 一次掷出一个 3×3 的骰子矩阵（创建二维数组的老套路）
    _grid = _rng_dice.integers(1, 7, size=(3, 3))

    # 加大样本：另起一个 rng，掷 60 次，数一数每个点数各出现几次
    _rng60 = np.random.default_rng(5)
    _rolls60 = _rng60.integers(1, 7, size=60)

    # 布尔计数法（第 05 课的招牌招式）：
    # _rolls60 == 点数 → 一串 True/False；.sum() 把 True 当 1 加起来 = 出现次数
    _counts = []                                    # 六个面的计数收集到这里
    for _face in range(1, 7):                       # 骰子的六个面：1~6
        _n_face = int((_rolls60 == _face).sum())    # 这一面出现了几次
        _counts.append(_n_face)

    # 顺手画个迷你条形图：每出现 2 次画 1 个 #
    _bars = "\n".join(
        f"  {_face} 点 | {'#' * (_counts[_face - 1] // 2)}  {_counts[_face - 1]} 次"
        for _face in range(1, 7)
    )

    # 多行的骰子矩阵和条形图上屏前，续行补 4 格缩进对齐这层 md（缩进最浅的一层）
    _grid_str = str(_grid).replace("\n", "\n" + " " * 4)
    _bars_str = _bars.replace("\n", "\n" + " " * 4)

    mo.md(
        f"""
        ### 演示 3 · 掷骰子：`rng.integers(1, 7)`

        ```
        掷 10 次：{_rolls10}

        3×3 骰子矩阵（size=(3, 3) 直接出二维）：
        {_grid_str}

        60 次里每个点数出现的次数（每 2 次画 1 个 #）：
        {_bars_str}
        ```

        **又是左闭右开！** `integers(1, 7)` 掷出的是 1~6：含 1、不含 7——
        想"掷一个 1 到 6 的骰子"就必须写 7，和第 03 课的 `arange(1, 7)` 一模一样的脾气。

        理论上每面应出现 60 ÷ 6 = **10 次**，实测是 {_counts}——
        5 点只有 7 次，1 点和 6 点却各有 12 次。**随机就是这样"不均匀"**；
        只有把次数加大到成千上万，每面的比例才会越来越接近 1/6
        （下面的交互实验里把样本量拖到 5000 亲自看看）。
        """
    )
    return


@app.cell
def _(mo, np):
    # ===== 演示 4 · 正态分布与文本直方图（本课视觉高潮）=====

    # 1000 个人的身高：均值 170cm、标准差 8cm 的正态分布
    # normal(均值, 标准差, size) —— 小心：第二个参数是标准差 σ，不是方差！
    _heights = np.random.default_rng(7).normal(170, 8, size=1000)

    # 实测均值 / 标准差：1000 个样本已经相当接近理论值 170 / 8
    _mean = _heights.mean()      # 实测 ≈ 169.42
    _std = _heights.std()        # 实测 ≈ 7.53

    # ---- 手工文本直方图：不用任何画图库，用布尔计数把 1000 个数分进 10 个格子 ----

    _lo = _heights.min()         # 最矮的人（≈ 144.0 cm）
    _hi = _heights.max()         # 最高的人（≈ 190.6 cm）
    _width = (_hi - _lo) / 10    # 每个格子覆盖的身高范围

    _rows = []                   # 每个格子的计数
    _labels = []                 # 每个格子的身高区间（显示用）
    for _i in range(10):
        _left = _lo + _i * _width      # 格子左边界（含）
        _right = _left + _width        # 格子右边界
        if _i < 9:
            # 前 9 格用左闭右开 [左, 右)——和切片、arange 同款习惯
            _mask = (_heights >= _left) & (_heights < _right)
        else:
            # 最后一格把最大值也收进来：[左, 右]
            _mask = (_heights >= _left) & (_heights <= _hi)
        _rows.append(int(_mask.sum()))              # 布尔掩码求和 = 这一格的人数
        _labels.append(f"{_left:.1f} ~ {_right:.1f}")

    # 每 20 个人画 1 个 #（不足 10 人的格子四舍五入后可能是零个，比如第一格）
    _bars = "\n".join(
        f"{_labels[_i]:>13} | {'#' * round(_rows[_i] / 20)}  {_rows[_i]} 人"
        for _i in range(10)
    )

    # 10 行直方图续行补 12 格缩进（这层 md 在 vstack 里，缩进更深）
    _bars_str = _bars.replace("\n", "\n" + " " * 12)

    mo.vstack(
        [
            mo.md(
                f"""
                ### 演示 4 · 正态分布：1000 个人的身高

                用 `normal(170, 8, size=1000)` 模拟 1000 个人的身高——均值 170cm、标准差 8cm：

                | | 理论 | 实测 |
                |---|---|---|
                | 均值 | 170 | {_mean:.2f} |
                | 标准差 | 8 | {_std:.2f} |

                ```
                身高区间 (cm)   | 人数直方图（每 20 人 1 个 #）
                {_bars_str}
                ```
                """
            ),
            mo.callout(
                mo.md(
                    """
                    **钟形出现了！** 中间高、两头低、左右大致对称——这就是**正态分布**
                    （高斯分布）的招牌轮廓。身高、考试成绩、测量误差……现实世界里大量现象都长这样。

                    更关键的是：**神经网络的初始权重，几乎总是用正态分布生成的**——
                    第 25 课的 He / Xavier 初始化，原料就是 `rng.normal(...)`。

                    注意我们**没有用任何画图库**：最小值/最大值线性切 10 格 +
                    布尔计数 + 一串 #，就是一张直方图。第 2 部分请出 matplotlib 之后，
                    同一个实验会画成一张真正的图。
                    """
                ),
                kind="success",
            ),
        ]
    )
    return


@app.cell
def _(mo, np):
    # ===== 演示 5 · shuffle 与 choice：数据科学的日常 =====

    # 一叠排好队的"卡片"：0~9（想象成 10 条训练数据）
    _data = np.arange(10)          # → [0 1 2 3 4 5 6 7 8 9]

    _rng2 = np.random.default_rng(3)

    # 原位打乱！划重点：shuffle【没有返回值】，它直接修改 _data 本身
    _rng2.shuffle(_data)           # 打乱后 → [9 6 0 2 1 4 7 5 3 8]

    # 无放回抽 3 个（replace=False）：抽走的不再放回，绝不重复——划分验证集就靠它
    _picked = _rng2.choice(_data, size=3, replace=False)      # → [4 1 2]

    # 有放回抽样（replace 默认 True）：抽完放回去，可能重复抽到同一个
    # choice 对字符串数组同样有效——实测确认 ✅
    _zoo = _rng2.choice(["猫", "狗", "鸟"], size=8)

    # 数一数三种动物各被抽到几次（布尔计数法再练一遍）
    _zoo_summary_parts = []
    for _name in ["猫", "狗", "鸟"]:
        _n_name = int((_zoo == _name).sum())      # 这种动物出现几次
        _zoo_summary_parts.append(f"**{_name}** × {_n_name}")
    _zoo_summary = "，".join(_zoo_summary_parts)

    mo.vstack(
        [
            mo.md(
                f"""
                ### 演示 5 · shuffle 与 choice：打乱与抽样

                | 操作 | 代码 | 结果 |
                |---|---|---|
                | 原位打乱 | `rng2.shuffle(data)` | `{_data}` |
                | 无放回抽 3 个 | `rng2.choice(data, size=3, replace=False)` | `{_picked}` |
                | 有放回抽 8 次 | `rng2.choice(["猫","狗","鸟"], size=8)` | `{_zoo}` |

                第三行的战果：{_zoo_summary}——"猫狗鸟"被有放回地抽了 8 次，
                有的被抽中 3 次、有的只有 2 次，这就是有放回的"重复"。

                打乱前 `data` 是整整齐齐的 `[0 1 2 3 4 5 6 7 8 9]`（`np.arange(10)`），
                shuffle 之后原数组**当场变乱序**——种子 3 定死了这个顺序，每次运行都一样。
                """
            ),
            mo.callout(
                mo.md(
                    """
                    **新手坑**：`shuffle` 是**原位**操作、返回 None——
                    如果你写 `data = rng.shuffle(data)`，`data` 会当场变成 None，后面全崩。
                    正确姿势就是让它单独一行（和 Python 列表的 `.sort()` 同款脾气）。

                    **预告**：第 17 课"过拟合与数据划分"就用这两招——
                    `shuffle` 打乱整个数据集，`choice(..., replace=False)` 抽出验证集；
                    mini-batch 的采样（第 25 课训练循环）也是 choice 的舞台。
                    """
                ),
                kind="warn",
            ),
        ]
    )
    return


@app.cell
def _(mo, np):
    # ===== 演示 6 · 为什么深度学习在乎种子 =====

    # 小实验：两个不同种子的 rng，各自生成一组 (4,) 的"初始权重"
    # （想象一个微型神经网络里 4 条连线的初始值，标准正态分布）
    _w_seed1 = np.random.default_rng(1).normal(0, 1, size=4)
    _w_seed2 = np.random.default_rng(2).normal(0, 1, size=4)

    # 逐元素差距有多大
    _gap_max = np.abs(_w_seed1 - _w_seed2).max()

    mo.vstack(
        [
            mo.md(
                f"""
                ### 演示 6 · 种子的分量：初始化不同，轨迹可能不同

                | 种子 | 初始权重（4 条连线） |
                |---|---|
                | 1 | `{np.round(_w_seed1, 3)}` |
                | 2 | `{np.round(_w_seed2, 3)}` |

                两组权重**完全不同**（逐元素差距最大 {_gap_max:.3f}）。
                而神经网络的训练，恰恰是从这样一组随机权重出发的：

                **起点不同，训练轨迹可能走向完全不同的地方**——同一段代码，
                种子 1 训出 97% 的准确率，种子 2 可能只有 95%，甚至干脆不收敛。
                你改进模型时涨的那 1 个点，到底是真的还是运气？——不固定种子，你永远说不清。
                """
            ),
            mo.callout(
                mo.md(
                    """
                    **实验三件套**（可复现实验的标准开场，后面的实验都这么写）：

                    ```python
                    SEED = 42                            # ① 种子写成一个常量，放在文件最顶上
                    rng = np.random.default_rng(SEED)    # ② 造一个专属 rng，一路传着用
                    w = rng.normal(0, 1, size=(4,))      # ③ 一切随机数都从 rng 拿，绝不碰全局随机
                    ```

                    复现论文的第一步，永远是固定种子；
                    第二步，是把 rng 当传家宝，从头到尾一路传下去。
                    """
                ),
                kind="info",
            ),
        ]
    )
    return


@app.cell
def _(mo):
    # 随机数实验台的四个旋钮
    seed_input = mo.ui.number(0, 9999, value=42, label="🎲 种子（0 ~ 9999，试试 42 和 43）")
    dist_dropdown = mo.ui.dropdown(
        options=["均匀 [0,1)", "正态 μ=0 σ=1", "正态 μ=170 σ=8", "整数 1~6（骰子）"],
        value="均匀 [0,1)",
        label="📊 分布",
    )
    n_slider = mo.ui.slider(
        10,
        5000,
        value=500,
        step=10,
        debounce=True,          # 拖动停止后才触发，避免边拖边算
        label="📈 样本量（10 ~ 5000）",
    )
    bump_switch = mo.ui.switch(value=False, label="➕ 换一个种子（+1）")

    mo.vstack([seed_input, dist_dropdown, n_slider, bump_switch])
    return (bump_switch, dist_dropdown, n_slider, seed_input)


@app.cell
def _(bump_switch, dist_dropdown, mo, n_slider, np, seed_input):
    # ===== 交互实验的响应单元：随机数实验台 =====

    # 种子 = 输入框的值 + 开关加成（开关打开 = 故意换成"下一个"种子）
    _seed_raw = seed_input.value
    _seed = (42 if _seed_raw is None else int(_seed_raw)) + (1 if bump_switch.value else 0)

    _n = int(n_slider.value)                     # 样本量
    _dist = dist_dropdown.value or "均匀 [0,1)"  # 当前分布（防手滑清空）

    # 用当前种子造 rng：种子一变，后面吐出的整条随机流立刻全部更新
    _rng = np.random.default_rng(_seed)

    # 按下拉框的选择生成样本，同时记下"理论均值 / 理论标准差"用于对照
    if _dist == "均匀 [0,1)":
        _samples = _rng.random(_n)               # [0, 1) 均匀分布
        _theo_mean, _theo_std = 0.5, 0.289       # 理论 std = 1/√12 ≈ 0.289
    elif _dist == "正态 μ=0 σ=1":
        _samples = _rng.normal(0, 1, _n)         # 标准正态
        _theo_mean, _theo_std = 0.0, 1.0
    elif _dist == "正态 μ=170 σ=8":
        _samples = _rng.normal(170, 8, _n)       # "身高"分布
        _theo_mean, _theo_std = 170.0, 8.0
    else:
        _samples = _rng.integers(1, 7, _n)       # 骰子：1~6 的整数（左闭右开）
        _theo_mean, _theo_std = 3.5, 1.708       # 理论 std = √(35/12) ≈ 1.708

    # 前 10 个数：浮点保留 3 位小数，整数直接显示
    if _samples.dtype.kind == "f":
        _first10 = "  ".join(f"{_v:.3f}" for _v in _samples[:10])
        _min_str, _max_str = f"{_samples.min():.3f}", f"{_samples.max():.3f}"
    else:
        _first10 = "  ".join(str(_v) for _v in _samples[:10])
        _min_str, _max_str = str(_samples.min()), str(_samples.max())

    # ---- 文本直方图：10 个 bin，最多的那个 bin 恰好 30 个 # ----
    _lo = float(_samples.min())
    _hi = float(_samples.max())
    if _hi <= _lo:                               # 兜底：万一所有样本相等（几乎不可能）
        _hi = _lo + 1.0

    _counts = []
    _labels = []
    for _i in range(10):
        _left = _lo + (_hi - _lo) * _i / 10      # 等宽切格子
        _right = _lo + (_hi - _lo) * (_i + 1) / 10
        if _i < 9:
            _mask = (_samples >= _left) & (_samples < _right)   # 左闭右开
        else:
            _mask = (_samples >= _left) & (_samples <= _hi)     # 末格收最大值
        _counts.append(int(_mask.sum()))
        _labels.append(f"{_left:.2f}~{_right:.2f}")

    _scale = max(_counts) / 30 if max(_counts) > 0 else 1.0   # 缩放系数：最多 30 个 #
    _bars = "\n".join(
        f"{_labels[_i]:>14} | {'#' * round(_counts[_i] / _scale)}  {_counts[_i]}"
        for _i in range(10)
    )

    # 同款缩进对齐：这层 md 在最外层，续行补 4 格
    _bars_str = _bars.replace("\n", "\n" + " " * 4)

    mo.md(
        f"""
        ## 🎛️ 交互实验 · 随机数实验台

        当前配置：**种子 = {_seed}**，**{_dist}**，**样本量 = {_n:,}**

        前 10 个数：`{_first10}`

        | 统计量 | 实测 | 理论 |
        |---|---|---|
        | 均值 mean | {_samples.mean():.3f} | {_theo_mean} |
        | 标准差 std | {_samples.std():.3f} | {_theo_std} |
        | 最小 min | {_min_str} | — |
        | 最大 max | {_max_str} | — |

        ```
        区间             | 直方图（最多的格子恰好 30 个 #）
        {_bars_str}
        ```

        **三件事值得玩出来：**

        1. **复现性**：同一种子、同一分布，无论刷新多少次，结果一个数都不变——
           这就是"可复现的随机"（演示 1 的魔术，这里随手验证）
        2. **多样性**：把种子 42 改成 43（或打开"+1"开关），整批数**立刻全部换新**——
           种子是随机宇宙的"编号"
        3. **大数定律**：把样本量从 10 慢慢拖到 5000，盯着"实测均值"一步步贴向理论值
           {_theo_mean}——样本越多，平均越稳（均匀分布理论均值 0.5，正态就是 μ，骰子是 3.5）
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✏️ 动手练习（每课必做，本课 3 题）": mo.md(
                """
                练习代码就在下面几个单元里，改完直接运行（`Shift + Enter`）。
                做完——或实在做不出来——再展开参考答案对照。

                - **练习 1 · 复现魔术**：亲手验证"同一种子 → 同一个矩阵"
                - **练习 2 · 双骰子分布**：掷 2 枚骰子 1000 次，画出"点数之和"的小山丘
                - **练习 3 · 思考题**：固定了种子，就真的万无一失了吗？
                """
            )
        }
    )
    return


@app.cell
def _(np):
    # 练习 1：复现魔术，亲手变一次
    #
    # 任务：用种子 2024 创建 rng，生成 3×3 的 [0,1) 均匀随机矩阵；
    #      再用种子 2024 【重新】创建一个 rng，再生成一次，验证两次结果完全相同。
    #
    # 提示（四步拼图）：
    #   ① rng_ex1 已经帮你造好（种子 2024）
    #   ② mat_ex1：问 rng 要一个 (3, 3) 的均匀矩阵——rng.random 的参数该传什么？
    #   ③ mat_again_ex1：np.random.default_rng(2024) 再造一个【新】rng，再 random((3, 3))
    #   ④ reproducible_ex1：用 np.array_equal 比较两个矩阵

    rng_ex1 = np.random.default_rng(2024)   # 第 ① 步已就位

    mat_ex1 = None           # ← ② 把 None 换成你的代码
    mat_again_ex1 = None     # ← ③ 把 None 换成你的代码
    reproducible_ex1 = None  # ← ④ 把 None 换成你的代码

    if mat_ex1 is None or mat_again_ex1 is None or reproducible_ex1 is None:
        ex1_msg = "（还是 None——把上面的 None 换成你的代码，再运行一次）"
    else:
        ex1_msg = f"两次的 3×3 矩阵完全相同吗？→ {reproducible_ex1}"
    ex1_msg
    return (ex1_msg, mat_again_ex1, mat_ex1, reproducible_ex1, rng_ex1)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 1 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                mat_ex1 = rng_ex1.random((3, 3))

                _rng_again = np.random.default_rng(2024)       # 用同一个种子 2024 重建
                mat_again_ex1 = _rng_again.random((3, 3))

                reproducible_ex1 = np.array_equal(mat_ex1, mat_again_ex1)   # → True
                ```

                分解动作：

                - `rng_ex1.random((3, 3))`：size 传元组 (3, 3)，直接得到 3×3 矩阵（第 03 课的老规矩）
                - 重建：**关键在"重新 default_rng(2024)"**——新 rng 的内部状态回到种子 2024 的起点
                - `np.array_equal`：逐元素比较，全部相同才返回 True

                结果是 **True**——只要你用同样的种子，在任何时间、任何机器上重跑，
                这 3×3 矩阵一个数都不会变。这就是论文里写一句 `seed=2024`，
                全世界就能复现他随机数的原因。
                """
            )
        }
    )
    return


@app.cell
def _(np):
    # 练习 2：掷 2 枚骰子 1000 次——"点数之和"的分布长什么样？
    #
    # 先做直觉预测：和的范围是 2~12；中间的 7 附近应该最多，两头的 2 和 12 最少
    # （因为 7 = 1+6 = 2+5 = 3+4 = ... 有 6 种组合凑法，而 2 只有 1+1 一种）
    #
    # 提示（两步拼图）：
    #   ① rolls_ex2：问 rng_ex2 要 1000 行 × 2 列的骰子点数——integers(1, 7, size=(1000, 2))
    #   ② sums_ex2：把每一行的两个点数加起来——想想第 08 课的心法：
    #      沿哪个 axis 求和，才能消灭"骰子"那一维、留下"次数"这一维？
    #
    # 做完运行，下面的代码会自动画出 2~12 的分布条形图

    rng_ex2 = np.random.default_rng(88)   # 固定种子：全班的结果一模一样

    rolls_ex2 = None   # ← ① 把 None 换成你的代码
    sums_ex2 = None    # ← ② 把 None 换成你的代码

    if sums_ex2 is None:
        ex2_msg = "（还是 None——把上面的 None 换成你的代码，再运行一次）"
    else:
        # np.unique：一次拿回"出现过哪些和 + 各出现几次"
        _faces, _cnts = np.unique(sums_ex2, return_counts=True)
        _lines = []
        for _f, _c in zip(_faces.tolist(), _cnts.tolist()):
            _lines.append(f"和为 {_f:>2} | {'#' * round(_c / 10)}  {_c} 次")   # 每 10 次 1 个 #
        ex2_msg = "\n".join(_lines)
    ex2_msg
    return (ex2_msg, rng_ex2, rolls_ex2, sums_ex2)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 2 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                rolls_ex2 = rng_ex2.integers(1, 7, size=(1000, 2))   # 1000 行 × 2 枚骰子
                sums_ex2 = rolls_ex2.sum(axis=1)                      # 沿 axis=1 求和
                ```

                分解动作：

                - `integers(1, 7, size=(1000, 2))`：shape (1000, 2)——每行是一次实验的两枚骰子
                - `.sum(axis=1)`：第 08 课心法——**axis=1 就是消灭第 1 个维度（列）**，
                  (1000, 2) → (1000,)，每行剩下的那个数就是两枚骰子的和
                - `np.unique(sums_ex2, return_counts=True)`：每种"和"出现几次，一次拿回

                种子 88 下的典型结果：**7 出现约 171 次（最高）**，6 和 8 次之，
                2 只有约 26 次、12 约 21 次——**中间高、两头低的小山丘**。

                道理：凑出 7 的组合有 6 种（1+6、2+5、……、6+1），凑出 2 的只有 1 种（1+1）。
                两个均匀的骰子相加，居然"卷"出了一座小钟形——这是"多个独立随机量相加、
                分布趋向钟形"的最早线索，第 2 部分聊正态分布时还会再遇到它。
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "🤔 练习 3 · 思考题（不用写代码）": mo.md(
                """
                深度学习代码里到处是 `torch.manual_seed(...)`、`np.random.seed(...)`：

                **问 1**：为什么大家这么执着于固定种子？（至少说出三个理由）

                **问 2**：如果两个人用**同一份代码、同一个种子、同一份数据**，
                他们能得到完全相同的结果吗？

                **问 3**：什么情况下，**仍然可能不同**？

                （提示：回想本课的"伪随机"——随机的外观、确定的心；再想想，
                除了随机数本身，还有哪些东西会影响计算结果。）
                """
            ),
            "💡 练习 3 参考答案（先想 3 分钟再看）": mo.md(
                """
                **问 1 · 为什么固定种子：**

                - **可复现**：bug 只有能复现才能定位——"同样的随机"是"同样的错误"的前提
                - **可调试**：改了一行代码，结果变了——是改动起效还是随机波动？固定种子才能归因
                - **公平比较**：对比两种方法时，初始化和数据打乱必须同起点，
                  否则就是"换着骰子比点数"，赢了也不算数

                **问 2 · 同代码 + 同种子 + 同数据：**

                **通常可以**得到完全相同的结果。伪随机的确定性保证了这一点——
                本课演示 1 的魔术，放大到整个实验依然成立。

                **问 3 · 但"通常"不是"一定"：**

                - **跨硬件**：CPU 与 GPU 的浮点运算顺序不同，累加顺序一变，末位小数就漂
                - **跨版本**：库升级后，随机数算法或底层实现可能变
                - **多线程 / 并行**：多个线程的执行顺序不确定，结果可能依赖调度
                - **某些 GPU 操作本身非确定**：比如并行累加的顺序不受控制

                在本课程的玩法里（CPU + NumPy + 专属 rng），你可以放心把"完全相同"当成默认；
                将来去了 GPU 的世界，"复现"会变成一个要专门花力气研究的课题。
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
                1. 计算机的随机数是**伪随机**：种子 → 确定性算法 → 一串貌似杂乱的数；
                   **同一个种子永远同一个序列**——可复现是超能力，不是缺陷
                2. 现代 API 两步走：`rng = np.random.default_rng(种子)` → 问 rng 要数；
                   **一切随机都从这个专属 rng 拿**，多个 rng 互不干扰
                3. rng 有**内部状态**：每要一次数前进一步，所以同一个 rng 连续两次结果不同；
                   **重建同种子的 rng，序列从头重演**（磁带与倒带键）
                4. 老式 `np.random.seed()` 是全局状态——公共水龙头，认得但别学；
                   `rng` 是专属水壶，新代码一律用它
                5. 常用伙伴：`integers`（**左闭右开**）、`random`（[0,1) 均匀）、
                   `normal(μ, σ, shape)`（**σ 是标准差不是方差**）、
                   `shuffle`（**原位**打乱、无返回值）、`choice(..., replace=False)`（无放回抽样）
                6. 深度学习第一行代码常是固定种子：**初始化不同，训练轨迹可能完全不同**——
                   实验三件套：固定 seed → 造专属 rng → 一切随机从 rng 拿
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
                ## 🎓 第 1 部分毕业典礼

                **从零到 Transformer · 手写深度学习** ｜ 第 1 部分 · NumPy 从零开始（10/10）——
                **全部完成！**

                十课之前，你可能还说不清"向量化"是什么；现在，你已经把 NumPy 的
                十八般武艺装进了工具箱。对着下面这份里程碑清单，逐行问自己一句：
                **"这一条，我现在做得到吗？"**——答不上来的，回去翻那一课，毕业不着急。
                """
            ),
            mo.md(
                """
                | 课 | 里程碑：你现在能做到什么 |
                |---|---|
                | **01** 为什么需要 NumPy | 说出数组比列表快的根本原因（同类型 + 连续内存 + 整块运算），写代码前先问"能向量化吗" |
                | **02** 解剖 ndarray | 拿到任何数组先看 `shape` / `dtype` / `ndim` / `size`，一眼估算内存账单 |
                | **03** 创建数组的多种方式 | 按需选用 `array` / `arange` / `linspace` / `zeros` / `ones` / `full` / `eye`，默写口诀"左闭右开" |
                | **04** 索引与切片 | 一刀切出行、列、块；知道切片是**视图**，改之前先想想要不要 `.copy()` |
                | **05** 布尔索引与花式索引 | 用 `grades[grades < 60]` 按条件筛选；用整数数组想挑哪几个就挑哪几个 |
                | **06** 广播机制 | 拿到两个 shape 就能心算"能不能算、广播成什么形状" |
                | **07** 形状变换 | `reshape` / `-1` 自动推算 / 转置 / `ravel` 自由变形，分得清"换形状"与"换排布" |
                | **08** axis 的真义 | 默念"**axis=i，消灭第 i 维**"，`sum` / `mean` / `argmax` 再也不用靠猜 |
                | **09** 线性代数操作 | 用 `@` 做矩阵乘法，`(n,k) @ (k,m) → (n,m)` 脱口而出 |
                | **10** 随机数与随机种子 | 用 `default_rng(种子)` 造出可复现的随机流，`shuffle` 打乱、`choice` 抽样信手拈来 |
                """
            ),
            mo.callout(
                mo.md(
                    """
                    这十项武功，是后面 38 课的全部地基：第 2 部分用它们**演数学**，
                    第 3 部分用它们**写出第一个机器学习模型**，第 7 部分用它们**手搭 Transformer**。
                    忘了哪一招，随时回来看一眼。
                    """
                ),
                kind="success",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(
                """
                ### 🏅 毕业自查：四道口诀 + 一道本课内功

                先不看答案，每题在心里答一遍，再点开对照——**五道全对，允许毕业**。
                """
            ),
            mo.accordion(
                {
                    "❓ 自查 1 · 广播口诀：`(5, 1) + (1, 3)` 能算吗？结果 shape 是？": mo.md(
                        """
                        **能。结果 shape 是 (5, 3)。**

                        口诀：**右对齐、逐维比，每维要么相等、要么有一个是 1**
                        （是 1 的那个维度会被拉伸成另一边的长度）。

                        - (5, 1) 与 (1, 3) 右对齐：最后维 1 vs 3 → 3；前一维 5 vs 1 → 5
                        - 于是 (5, 1) 那一列被"复印"成 3 份，(1, 3) 那一行被"复印"成 5 份
                        - 谁都不是 (5, 3)，却谁都不报错——这就是第 06 课的广播
                        """
                    ),
                    "❓ 自查 2 · axis 心法：shape (2, 3, 4) 的数组 `sum(axis=1)` 后，shape 变成？": mo.md(
                        """
                        **(2, 4)。**

                        心法：**axis=i 的运算，就是消灭第 i 个维度**——其余维度原样保留。

                        (2, 3, 4) 消灭第 1 维（那个 3）→ 剩下 (2, 4)。
                        换成 `sum(axis=0)` → (3, 4)；`sum(axis=2)` → (2, 3)。
                        忘了就回去翻第 08 课。
                        """
                    ),
                    "❓ 自查 3 · 视图 vs 副本：`b = a[::2]` 之后 `b[0] = 99`，`a` 会变吗？": mo.md(
                        """
                        **会。**

                        切片返回的是**视图**——原数组内存上开的一扇窗，
                        `b[0]` 和 `a[0]` 是同一格内存。想要"改 b 不动 a"，
                        取的时候就得复印：`b = a[::2].copy()`。

                        一句话：**窗口免费，复印明码标价**（第 04 课）。
                        """
                    ),
                    "❓ 自查 4 · `@` 的形状规则：`X @ W`，X 是 (100, 32)，想得到 (100, 10)，W 是什么形状？": mo.md(
                        """
                        **(32, 10)。**

                        规则：**(n, k) @ (k, m) → (n, m)**——中间的两个 k 必须咬合，
                        首尾的两个数决定结果的形状。

                        所以 (100, 32) @ (32, 10) → (100, 10)。深度学习里 `X @ W + b`
                        的 W，形状永远由"输入特征数 → 输出特征数"决定（第 09 课）。
                        """
                    ),
                    "❓ 自查 5 · 本课内功：两个都是种子 42 的 rng，各自 `random(5)`，结果相同吗？": mo.md(
                        """
                        **相同——这就是"可复现的随机"。**

                        种子决定 rng 的初始状态，初始状态决定整条随机数序列：
                        同种子的两个 rng 走的是同一条时间线；**换个种子（哪怕换成 43），
                        整条序列立刻全部换新**。

                        面试官爱追问："固定了种子就万事大吉了吗？"——跨硬件、跨库版本、
                        多线程并行时仍可能分叉（练习 3 里聊过）。
                        """
                    ),
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ---

        **下一课预告** — `11_向量的几何直觉`：

        恭喜——**第 1 部分 · NumPy 从零开始，十课功成！** 🎓

        第 2 部分"深度学习的数学"正式开幕。第一站：把 NumPy 里的向量变成几何世界里的**箭头**——
        点积变成夹角、范数变成长度、正交变成垂直。从此，你看到的每一个公式都有了画面：
        "梯度下降"不再是四个汉字，而是小球在山谷里滚动时选定的方向。

        下一课还会正式请出 **matplotlib**——深度学习最重要的画图武器，
        后面所有课的损失曲线、决策边界都靠它来画。

        十课攒下的 NumPy 内功，从现在开始要"演数学"了。

        🏔️ 带上你的数组，去第 2 部分见！
        """
    )
    return


if __name__ == "__main__":
    app.run()
