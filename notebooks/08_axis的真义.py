import marimo

__generated_with = "0.25.2"
app = marimo.App(app_title="08 · axis的真义", width="medium")


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
        # 08 · axis的真义

        **从零到 Transformer · 手写深度学习** ｜ 第 1 部分 · NumPy 从零开始（8/10）
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📖 本课目标（先读我）": mo.md(
                """
                **这一课只解决一个问题：axis 到底是什么？为什么它能难住几乎所有新手？**

                答案会出乎意料地简单：axis 不是玄学，它就是 shape 元组里一个清清楚楚的
                **位置编号**；而所有的困惑，都来自一个没有被说破的直觉——
                **聚合运算是"消灭"那个轴**。

                学完你将能：

                - 一句话说出 axis 的本义：shape (3, 4) 里，axis 0 管 3、axis 1 管 4；负数 axis 从右往左数
                - 用"消灭那个轴"的万能公式，**先写结果 shape、再算数**——任何聚合函数都难不倒你
                - 识破广播 + axis 联手的著名大坑，并用 `keepdims=True` 稳稳修好（本课高潮）
                - 在三维（批 × 高 × 宽）数组上自如地选轴——这正是神经网络里数据的真实形状

                本课只需要第 02、04、06、07 课的基础（shape、切片、广播、newaxis），我们从零开始讲。
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "🧭 概念 1/3 · axis 就是 shape 元组里的位置编号": mo.md(
                """
                问：`sum(axis=0)` 里的 0，到底指什么？

                答：**shape 元组里的第 0 个位置。** 就这么朴素，没有玄学。

                每个数组的 shape 是一个元组，比如 (3, 4)——3 行、4 列。
                **axis 就是给这个元组的每个位置编的号**：

                | shape 的位置 | 位置 0 | 位置 1 |
                |---|---|---|
                | 元组里的数 | 3 | 4 |
                | 对应的 axis | **axis 0** | **axis 1** |
                | 二维里的叫法 | 行方向 | 列方向 |
                | 成绩单例子 | 学生（3 个） | 科目（4 门） |

                **为什么从 0 开始数？** 和第 04 课的下标是同一个道理——那是"偏移量"的惯例：
                第一个维度偏移 0 步。数组的嵌套从外往里，一层一层依次是
                axis 0、axis 1、axis 2……所以最外层（比如"哪个学生"）永远是 axis 0，
                越往里编号越大。

                **负数 axis = 从右往左数。** -1 永远是最后一个维度、-2 是倒数第二个——
                和 `a[-1]` 取最后一个元素是同一套逻辑（演示 3 亲自验证）。

                **类比**：shape 是数组的"门牌号规则"（几单元、几层、几号室），
                axis 是每一段门牌的编号。报出 axis=k，就是在说"第 k 段门牌"。
                """
            ),
            "✂️ 概念 2/3 · 聚合运算是“消灭”那个轴（本课核心直觉）": mo.md(
                """
                新手最大的误解："`sum(axis=0)` 是对第 0 行求和"。**错，而且错得经典。**

                `sum(axis=0)` 的真正含义：**沿着 axis 0 的方向，把所有行叠压在一起**——
                像把一摞 3 张饼压成 1 张：3 行压没了，4 列原样保留。
                被压的那个轴，从**结果里彻底消失**。

                于是有一条万能公式（本课唯一需要背的东西）：

                > **结果 shape = 原 shape 删掉 axis 那个位置的数。**

                - (3, 4) --sum(axis=0)--> (4,)：删掉 3，每列剩一个总和
                - (3, 4) --sum(axis=1)--> (3,)：删掉 4，每行剩一个总和
                - (3, 4) --sum()--> 标量：不给 axis，全部轴一起压

                这正是第 04 课口诀的续集。那课说过：**整数索引砍维度**
                （`grades[0, :]` 行给整数，行维度消失）。聚合是同一件事的另一种砍法——
                只是"砍"的动作从"只留这一个"换成了"求和 / 求平均 / 取最大"。
                被砍的轴同样从 shape 里消失，规则完全一致。

                **判题心法：先写结果 shape，再算数。**

                拿到任何聚合题，第一步永远是在草稿上写下结果的 shape。
                shape 写对了，数值几乎不可能错；shape 写错了，数值碰对了也是蒙的。
                （演示 1 到演示 4，每一格都按这个心法走一遍。）
                """
            ),
            "📌 概念 3/3 · keepdims 与移动手指口诀": mo.md(
                """
                **keepdims=True：让被压掉的轴"留个牌位"。**

                (3, 4) --mean(axis=1)--> (3,)；但 --mean(axis=1, **keepdims=True**)--> (3, **1**)。
                轴没有消失，只是被压成了 1。多出来的这个 1 看似无用，实则专为**广播**而生：

                - 想每行减去自己的均值？(3,) 是"横着的行"，广播时对不上行维度——
                  轻则报错，重则**静默算错**（演示 2 亲眼见证）
                - (3, 1) 是"竖着的列"，广播时恰好对上行维度——一次成功

                所以记住：**凡是"聚合完还要减 / 除回去"的场合，先想 keepdims。**

                两套记忆口诀（二选一，都给你）：

                ① **移动手指**：axis=k，就想象第 k 根手指沿着数组滑动。
                  二维数组上，axis=0 的手指**竖着向下滑**、滑过所有行（把行叠压在一起）；
                  axis=1 的手指**横着向右滑**、滑过所有列。
                ② **二维速查**：**axis=0 压行、axis=1 压列**。日常二维够用，
                  但记住这只是速查版——三维以上必须回到本义"消灭第 k 个维度"。

                三维对照表（批次 × 高 × 宽 的图片——神经网络里数据的真实长相）：

                | axis | 消灭哪个维度 | 发生了什么 |
                |---|---|---|
                | 0 | 批（几张图） | 跨图逐像素平均，得到一张"平均图" |
                | 1 | 高（几行） | 每张图被纵向压扁 |
                | 2 | 宽（几列） | 每张图被横向压扁 |
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

        六组演示，把 axis 从二维打到三维：

        1. **核心对照实验**：同一张成绩单，三种压法
        2. **行向量与列向量陷阱**：广播 + axis 联手作案（本课高潮，著名大坑）
        3. **负数 axis**：-1 永远是最后一个维度
        4. **三维批量图片**：批 × 高 × 宽
        5. **keepdims 救场**：为广播而生的开关
        6. **argmax 彩蛋**：要位置，不要值

        每个单元都可以直接运行；**大胆改里面的数字**，坏了点撤销就行。
        """
    )
    return


@app.cell
def _(mo, np):
    # ===== 演示 1 · 核心对照实验：同一张成绩单，三种压法 =====

    # 成绩单 g：3 个学生 × 4 门科目
    #    行 = 学生（0~2），列 = 科目（0 语文、1 数学、2 英语、3 科学）
    #    shape = (3, 4)：axis 0 管“3”（学生方向），axis 1 管“4”（科目方向）
    g = np.array(
        [
            [85, 92, 78, 60],  # ← 学生 0
            [70, 88, 95, 65],  # ← 学生 1
            [90, 75, 82, 88],  # ← 学生 2
        ]
    )

    # ---- 先用心法推出结果 shape，再让 NumPy 算数 ----
    # (3, 4) 删掉 3 那一位 → (4,)：每科一个总分
    _sum_axis0 = g.sum(axis=0)   # 压掉 axis 0（学生）：3 行叠压在一起 → [245 255 255 213]
    # (3, 4) 删掉 4 那一位 → (3,)：每生一个总分
    _sum_axis1 = g.sum(axis=1)   # 压掉 axis 1（科目）：4 列叠压在一起 → [315 318 335]
    # 不给 axis：全部轴一起压 → 只剩 1 个数
    _sum_all = g.sum()           # 全班总分 → 968

    # ---- 换武器：mean / max，“消灭哪个轴”的规则一模一样 ----
    _mean_axis0 = g.mean(axis=0)   # 每科平均分（4 个数）→ [81.67 85. 85. 71.]
    _mean_axis1 = g.mean(axis=1)   # 每生平均分（3 个数）→ [78.75 79.5 83.75]
    _max_axis0 = g.max(axis=0)     # 每科最高分（4 个数）→ [90 92 95 88]
    _max_axis1 = g.max(axis=1)     # 每生最高分（3 个数）→ [92 95 90]

    mo.vstack(
        [
            mo.md(
                f"""
                ### 演示 1 · 核心对照实验：同一张成绩单，三种压法

                成绩单 `g`（行 = 学生 0~2，列 = 科目：语文、数学、英语、科学），
                shape = **{g.shape}**——axis 0 管第一个数 3（学生），axis 1 管第二个数 4（科目）。
                先在草稿上写好**结果 shape**，再对照 NumPy 的答案：
                """
            ),
            mo.md(f"```\n{g}\n```"),
            mo.md(
                f"""
                | 写法 | 原 shape | 消灭哪个轴 | 结果 shape | 业务含义（数值） |
                |---|---|---|---|---|
                | `g.sum(axis=0)` | {g.shape} | axis 0（学生） | {_sum_axis0.shape} | 每科总分：`{_sum_axis0}` |
                | `g.sum(axis=1)` | {g.shape} | axis 1（科目） | {_sum_axis1.shape} | 每生总分：`{_sum_axis1}` |
                | `g.sum()` | {g.shape} | 全部轴 | () | 全班总分：**{_sum_all}** |

                同一张表，武器换成 mean 和 max 再走一遍——**消灭哪个轴，和用什么武器无关**：

                | 写法 | 消灭的轴 | 结果 shape | 业务含义（数值） |
                |---|---|---|---|
                | `g.mean(axis=0)` | axis 0（学生） | {_mean_axis0.shape} | 每科平均分：`{np.round(_mean_axis0, 2)}` |
                | `g.mean(axis=1)` | axis 1（科目） | {_mean_axis1.shape} | 每生平均分：`{np.round(_mean_axis1, 2)}` |
                | `g.max(axis=0)` | axis 0（学生） | {_max_axis0.shape} | 每科最高分：`{_max_axis0}` |
                | `g.max(axis=1)` | axis 1（科目） | {_max_axis1.shape} | 每生最高分：`{_max_axis1}` |

                三个观察：

                1. **结果 shape = 原 shape 删掉那个位置的数**：(3, 4) 删掉 3 → (4,)；删掉 4 → (3,)
                2. `axis=0` 压掉“学生”→ 剩 4 个数，全是**按科目**的统计；
                   `axis=1` 压掉“科目”→ 剩 3 个数，全是**按学生**的统计
                3. sum 换成 mean、max、min、std……**“消灭哪个轴”的规则一模一样**——
                   学会一个，等于学会了全部聚合函数
                """
            ),
        ]
    )
    return (g,)


@app.cell
def _(g, mo, np):
    # ===== 演示 2 · 行向量与列向量陷阱：广播 + axis 联手作案 =====

    # 需求很常见（数据预处理天天写）：每一行减去“该行自己的均值”
    # ——行方向中心化。比如把每个学生的成绩换成“比自己的平均分高/低多少”

    # 第一步本来很顺：g.mean(axis=1) 消灭科目轴，得到 (3,)
    _row_means = g.mean(axis=1)   # (3,)：三个学生各自的平均分 [78.75 79.5 83.75]

    # ---- 尝试 1：想当然地直接减 g - _row_means ----
    # (3, 4) 对 (3,)：广播肯不肯放行？让 NumPy 自己表态
    try:
        g - _row_means            # 这一行会当场爆炸
        _crash_msg = "（居然没报错？不可能，别想了）"
    except ValueError as _exc:
        _crash_msg = str(_exc)    # 把 NumPy 的拒绝理由原样抄下来给大家看

    # ---- 尝试 2：更阴险——数组恰好是方阵时，同一句代码不报错了 ----
    _sq = g[:, :3]                # 只取前 3 门科目：shape (3, 3) 的方阵
    _silent = _sq - _sq.mean(axis=1)   # (3,3) - (3,)：广播“成功”了……但减错了对象
    _bad_sums = _silent.sum(axis=1)    # 中心化后每行之和本应全为 0，验证一下（结果 [3.33 1.33 -4.67]）

    # ---- 正确做法（演示 5 详细展开，这里先对照一眼）----
    _fixed = _sq - _sq.mean(axis=1, keepdims=True)   # (3,3) - (3,1)：每行减自己的均值
    _good_sums = _fixed.sum(axis=1)                  # 每行之和全是 0 ✅

    mo.vstack(
        [
            mo.md(
                f"""
                ### 演示 2 · 行向量与列向量陷阱：广播 + axis 联手作案

                目标：**每行减去该行自己的均值**。第一步 `g.mean(axis=1)` →
                `{np.round(_row_means, 2)}`，shape **{_row_means.shape}**——注意，它是个**横着的行**。

                **尝试 1 · 直接减**：`g - g.mean(axis=1)` → 💥 当场报错：
                """
            ),
            mo.md(f"```\nValueError: {_crash_msg}\n```"),
            mo.md(
                """
                为什么？广播的规矩（第 06 课）：形状**从最右边开始对齐**——

                ```
                (3, 4)   ← g
                (1, 3)   ← (3,) 先补成 (1, 3)，末尾的 3 撞上了 4 的位置
                     ↑ 3 ≠ 4，也不是 1 → NumPy 拒绝广播
                ```

                我们的本意是让 3 个均值**竖着**逐行减下去，需要的形状其实是 (3, 1)
                的“列”——(3,) 这个“行”差了一个方向。
                """
            ),
            mo.md(
                f"""
                **尝试 2 · 方阵陷阱（更危险）**：只取前 3 门科目，造一个 (3, 3) 的方阵 `_sq`，
                同一句 `_sq - _sq.mean(axis=1)`——这回**不报错了**：

                - 方阵时 (3, 3) 对 (3,)：(3,) 被右对齐补成 (1, 3)、再把 1 拉伸成 3，广播“成功”
                - 但每行减掉的是“[第 0 行的均值, 第 1 行的均值, 第 2 行的均值]”这个**错位向量**——
                  92 在第 0 行第 1 列，减掉的却是 84.33（**第 1 行**学生的均值），不是学生 0 自己的 85
                - 自检铁律：中心化后每行之和应为 0。错误版每行之和 =
                  `{np.round(_bad_sums, 2)}`——全不是 0，**静默算错，一声不吭**
                """
            ),
            mo.hstack(
                [
                    mo.md(
                        f"**错误版**（跑得通，但错位）\n\n```\n{np.round(_silent, 2)}\n```\n\n每行之和：`{np.round(_bad_sums, 2)}` ❌"
                    ),
                    mo.md(
                        f"**正确版**（keepdims=True）\n\n```\n{np.round(_fixed, 2)}\n```\n\n每行之和：`{np.round(_good_sums, 2)}` ✅"
                    ),
                ]
            ),
            mo.callout(
                mo.md(
                    """
                    **广播 + axis 联手的著名大坑**：`mean(axis=1)` 给出的 (3,) 会被广播规则
                    当作**横着的行**（右对齐）；想让它**竖着**起作用，必须先变成 (3, 1) 的**列**——
                    `keepdims=True` 或 `[:, None]`，二选一。

                    - 形状对不上（如 (3,4) 对 (3,)）：当场报错——算你**走运**，错误响亮
                    - 形状恰好对上（方阵 (3,3) 对 (3,)）：**静默错位**，一声不吭地算错——这才是灾难

                    所以看到“方阵数据 + 沿 axis 1 聚合 + 还要广播回去”，
                    请条件反射地想 keepdims。
                    """
                ),
                kind="warn",
            ),
        ]
    )
    return


@app.cell
def _(g, mo, np):
    # ===== 演示 3 · 负数 axis：从右往左数 =====

    # 第 04 课的负索引（a[-1] 是最后一个元素）在 axis 世界同样成立：
    # -1 永远指最后一个维度，-2 指倒数第二个
    _s_neg1 = g.sum(axis=-1)   # -1 = 最后一个维度，这里就是 axis 1（科目）→ [315 318 335]
    _s_neg2 = g.sum(axis=-2)   # -2 = 倒数第二个维度，这里就是 axis 0（学生）→ [245 255 255 213]

    # 严格验证：负数写法和正数写法算出的结果完全相等
    _eq_neg1 = np.array_equal(g.sum(axis=-1), g.sum(axis=1))
    _eq_neg2 = np.array_equal(g.sum(axis=-2), g.sum(axis=0))

    mo.md(
        f"""
        ### 演示 3 · 负数 axis：从右往左数

        | 写法 | 数法 | 等价于 | 结果 | 结果 shape |
        |---|---|---|---|---|
        | `g.sum(axis=-1)` | -1 = 最后一个维度 | `g.sum(axis=1)` | `{_s_neg1}` | {_s_neg1.shape} |
        | `g.sum(axis=-2)` | -2 = 倒数第二个 | `g.sum(axis=0)` | `{_s_neg2}` | {_s_neg2.shape} |

        严格验证：`np.array_equal(g.sum(axis=-1), g.sum(axis=1))` → **{_eq_neg1}** ✅，
        `np.array_equal(g.sum(axis=-2), g.sum(axis=0))` → **{_eq_neg2}** ✅

        **写代码时 `axis=-1` 往往比数下标更不容易错**：不用管数组是几维、
        前面还有几个轴，-1 永远指最后一个维度。深度学习的数据常常长成
        (批, 特征) 或 (批, 高, 宽)——“沿特征 / 宽度方向聚合”这种需求，
        闭着眼睛写 `axis=-1` 就对了。
        """
    )
    return


@app.cell
def _(mo, np):
    # ===== 演示 4 · 三维：批量图片 =====

    # imgs：2 张 3×4 的“图片”叠成一摞 → shape (2, 3, 4) = (批, 高, 宽)
    #    axis 0 = 批（2 那一位），axis 1 = 高（3 那一位），axis 2 = 宽（4 那一位）
    #    数值 0~23 依次填入，方便对照“哪些数被压到了一起”
    imgs = np.arange(24).reshape(2, 3, 4)

    _sum0 = imgs.sum(axis=0)   # 压掉“批”：两张图逐像素叠加 → (3, 4)，比如左上角 0+12=12
    _sum1 = imgs.sum(axis=1)   # 压掉“高”：每张图 3 行压成 1 行 → (2, 4)，比如 0+4+8=12
    _sum2 = imgs.sum(axis=2)   # 压掉“宽”：每行压成 1 个数 → (2, 3)，比如 0+1+2+3=6

    mo.vstack(
        [
            mo.md(
                """
                ### 演示 4 · 三维：批量图片

                `imgs = np.arange(24).reshape(2, 3, 4)`——2 张 3×4 的图，shape **(批, 高, 宽)**：
                """
            ),
            mo.md(f"```\n{imgs}\n```"),
            mo.md(
                f"""
                心法先走一遍：结果 shape = (2, 3, 4) **删掉对应位置的数**——

                | 写法 | 消灭什么 | 结果 shape | 业务解读 |
                |---|---|---|---|
                | `imgs.sum(axis=0)` | 批（2 那一位） | {_sum0.shape} | 跨图叠加：两张图同一位置的像素求和 |
                | `imgs.sum(axis=1)` | 高（3 那一位） | {_sum1.shape} | 每张图把 3 行压成 1 行（纵向压扁） |
                | `imgs.sum(axis=2)` | 宽（4 那一位） | {_sum2.shape} | 每行压成 1 个数（横向压扁） |
                """
            ),
            mo.hstack(
                [
                    mo.md(f"**axis=0** → {_sum0.shape}\n\n```\n{_sum0}\n```"),
                    mo.md(f"**axis=1** → {_sum1.shape}\n\n```\n{_sum1}\n```"),
                    mo.md(f"**axis=2** → {_sum2.shape}\n\n```\n{_sum2}\n```"),
                ]
            ),
            mo.md(
                """
                对照数值看“谁和谁被压到了一起”：axis=0 结果里的 12 = 0+12（两张图的左上角）、
                axis=1 结果里的 12 = 0+4+8（第一张图第一列纵向三个）、
                axis=2 结果里的 6 = 0+1+2+3（第一行横向四个）。

                二维的“压行 / 压列”口诀到这里就不够用了——**必须回到本义：
                axis=k 就是消灭 shape 的第 k 个维度。**维度再多也一样。
                """
            ),
        ]
    )
    return (imgs,)


@app.cell
def _(g, mo, np):
    # ===== 演示 5 · keepdims 救场：为广播而生的开关 =====

    # 演示 2 的正确做法，在这里完整展开
    _m = g.mean(axis=1, keepdims=True)   # keepdims=True：被压掉的轴保留为 1 → shape (3, 1)
    _centered = g - _m                   # (3, 4) - (3, 1)：广播成功，每行减自己的均值

    # 终极验证：中心化之后，每行之和应该全都是 0
    _row_sums = _centered.sum(axis=1)    # → [0. 0. 0.]

    # 另一种等价写法：[:, None] 把 (3,) 升维成 (3, 1)（第 07 课的主角之一）
    _alt = g - g.mean(axis=1)[:, None]
    _alt_same = np.array_equal(_alt, _centered)

    mo.vstack(
        [
            mo.md(
                f"""
                ### 演示 5 · keepdims 救场：为广播而生的开关

                ```
                _m = g.mean(axis=1, keepdims=True)   # (3,) → (3, 1)：被压掉的轴保留为 1
                _centered = g - _m                   # (3, 4) - (3, 1)：广播一次成功
                ```

                - `_m` 的 shape：**{_m.shape}**（注意！不是 (3,)）——一个**竖着的列**：
                """
            ),
            mo.md(f"```\n{_m}\n```"),
            mo.md(
                f"""
                - 每行减的都是自己的均值。拿第一行人工验算：
                  85 − 78.75 = **6.25**，92 − 78.75 = **13.25**，
                  78 − 78.75 = **−0.75**，60 − 78.75 = **−18.75**
                - 终极验证：中心化后每行之和 → `{_row_sums}` 全是 0 ✅
                """
            ),
            mo.md(f"```\n{_centered}\n```"),
            mo.md(
                f"""
                **keepdims 不是为了好看**：(3,) 是横着的行、(3, 1) 是竖着的列，
                广播从最右边对齐——只有“列”才能对上行维度。所以它的存在意义就一个：
                **聚合完还要广播回去的场合，形状别断线。**

                等价写法：`g - g.mean(axis=1)[:, None]`（用 None 升维，第 07 课讲过），
                验证两法相等 → **{_alt_same}** ✅。两种都行，keepdims 语义更直白。
                """
            ),
        ]
    )
    return


@app.cell
def _(g, mo):
    # ===== 演示 6 · 彩蛋：argmax —— 要位置，不要值 =====

    _best_subject = g.argmax(axis=1)   # 每个学生“最强科目”的编号（3 个数）→ [1 2 0]
    _best_score = g.max(axis=1)        # 对照组：每个学生最强的“分数”（3 个数）→ [92 95 90]

    mo.md(
        f"""
        ### 演示 6 · 彩蛋：argmax —— 要位置，不要值

        | 写法 | 返回什么 | 结果 |
        |---|---|---|
        | `g.argmax(axis=1)` | 每行最大值的**位置编号** | `{_best_subject}` |
        | `g.max(axis=1)` | 每行的**最大值本身** | `{_best_score}` |

        翻译成人话：学生 0 的最强科目是编号 **1**（数学，92 分）；
        学生 1 是编号 **2**（英语，95 分）；学生 2 是编号 **0**（语文，90 分）。
        科目编号对应第 0 列语文、第 1 列数学、第 2 列英语、第 3 列科学。

        一句话记住它：**argmax 不返回最大值，返回最大值的位置**——
        等到第 4 部分讲分类模型（softmax 取预测类别）时会天天用（“模型认为最可能是哪一类”，
        要的正是位置，不是分数）。
        """
    )
    return


@app.cell
def _(mo):
    # ===== 交互实验 · 三维压轴机 的控制台 =====

    # 三个滑块：三维数组三个维度的大小（批 b、高 h、宽 w）
    b_slider = mo.ui.slider(1, 4, value=2, step=1, debounce=True, label="🎛️ 批数 b（1 ~ 4）")
    h_slider = mo.ui.slider(1, 5, value=3, step=1, debounce=True, label="🎛️ 高度 h（1 ~ 5）")
    w_slider = mo.ui.slider(1, 5, value=4, step=1, debounce=True, label="🎛️ 宽度 w（1 ~ 5）")

    # 单选钮：沿哪个轴聚合（选项文字直接告诉你压掉什么）
    axis_radio = mo.ui.radio(
        options={
            "axis=0（压掉“批”）": "0",
            "axis=1（压掉“高”）": "1",
            "axis=2（压掉“宽”）": "2",
            "全压（不给 axis）": "all",
        },
        value="0",
        label="🎛️ 沿哪个轴求平均？",
    )

    # 开关：keepdims——被压掉的轴是否保留为 1
    keepdims_switch = mo.ui.switch(label="🎛️ keepdims=True（保留被压掉的轴）", value=False)

    mo.vstack([b_slider, h_slider, w_slider, axis_radio, keepdims_switch])
    return (axis_radio, b_slider, h_slider, keepdims_switch, w_slider)


@app.cell
def _(axis_radio, b_slider, h_slider, keepdims_switch, mo, np, w_slider):
    # ===== 交互实验的响应单元：三维压轴机 =====

    # 读取四个部件的当前值
    _b = b_slider.value          # 批数
    _h = h_slider.value          # 高度
    _w = w_slider.value          # 宽度
    _kd = keepdims_switch.value  # keepdims 开关

    # 造一个 (批, 高, 宽) 的三维数组：0, 1, 2, ... 依次填入
    _arr = np.arange(_b * _h * _w).reshape(_b, _h, _w)

    # 按单选钮的选择执行 mean（“全压” = 不给 axis）
    if axis_radio.value == "all":
        _result = np.asarray(_arr.mean(keepdims=_kd))   # 全部轴一起压
        _expr = "mean(keepdims=True)" if _kd else "mean()"
        _read = f"全压：{_b}×{_h}×{_w} 个数一起求平均，只剩一个总平均数"
    else:
        _ax = int(axis_radio.value)                     # 选中的轴编号
        _result = np.asarray(_arr.mean(axis=_ax, keepdims=_kd))
        _expr = f"mean(axis={_ax}, keepdims=True)" if _kd else f"mean(axis={_ax})"
        _reads = {
            0: f"跨图平均：{_b} 张图同一位置的像素互相平均——“批”被压掉，高、宽原样保留",
            1: f"每张图内部沿高度求平均：{_h} 行压成 1 行——“高”被压掉，批、宽保留",
            2: f"每张图内部沿宽度求平均：每行压成 1 个数——“宽”被压掉，批、高保留",
        }
        _read = _reads[_ax]

    # 组装输出：标题 + 箭头图 + 结果数组 + 一句话解读
    _items = [
        mo.md(
            f"""
            ## 🎛️ 交互实验 · 三维压轴机

            你拖出了一组 **({_b}, {_h}, {_w})** 的数据（{_b} 张 {_h}×{_w} 的“图”），
            压轴机正在执行：
            """
        ),
        mo.md(f"```\n{_arr.shape} --{_expr}--> {_result.shape}\n```"),
        mo.md(f"**结果数组**（shape {_result.shape}）："),
        mo.md(f"```\n{_result}\n```"),
        mo.md(f"**一句话解读**：{_read}。"),
    ]

    # keepdims 打开时追加一条提示，把开关和“为广播而生”再绑一次
    if _kd:
        _items.append(
            mo.callout(
                mo.md(
                    """
                    **keepdims=True 生效**：被压掉的轴没有消失，而是保留为 1——
                    形状信息没丢，后续可以直接广播回去。这就是它存在的全部意义。
                    """
                ),
                kind="info",
            )
        )

    mo.vstack(_items)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✏️ 动手练习（每课必做，本课 3 题）": mo.md(
                """
                练习代码就在下面几个单元里，改完直接运行（`Shift + Enter`）。
                做完——或实在做不出来——再展开参考答案对照。

                - **练习 1 · 每科最高分**：对演示 1 的 g 求每个科目的最高分（4 个数）
                - **练习 2 · 城市气温中心化**：每行减去该行的均值——keepdims 的实战
                - **练习 3 · 预测题**：`a.sum(axis=(0, 2))` 的 shape 是什么？先预测，再运行验证
                """
            )
        }
    )
    return


@app.cell
def _(g):
    # 练习 1：对演示 1 的成绩单 g，求“每个科目的最高分”（应该得到 4 个数）
    #
    # 提示（三步拼图）：
    #   ① “每个科目” = 沿学生方向、把同一列的 3 个分数叠压在一起 → 消灭哪个轴？
    #   ② 聚合武器用 max
    #   ③ 先在草稿上写出结果 shape，再动手写代码（判题心法）
    #
    # 期望结果：[90 92 95 88]

    subject_best = None   # ← 把 None 换成你的实现（一行就够）

    subject_best if subject_best is not None else "（还是 None——先想清楚消灭哪个轴，再运行一次）"
    return (subject_best,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 1 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                subject_best = g.max(axis=0)
                ```

                分解动作：

                - “每个科目的最高分” = 沿学生方向、把同一列的 3 个分数叠压 → **消灭 axis 0（学生）**
                - 结果 shape：(3, 4) 删掉 3 那一位 → (4,)，4 个数 ✓
                - 结果：`[90 92 95 88]`——语文 90（学生 2）、数学 92（学生 0）、
                  英语 95（学生 1）、科学 88（学生 2）

                错一次才记得牢：写成 `g.max(axis=1)` 会得到 (3,) 的 `[92 95 90]`——
                那是“每个学生的最高分”。**消灭的轴不同，回答的问题就不同**——
                所以第一步永远是问自己：我要消灭谁？
                """
            )
        }
    )
    return


@app.cell
def _(np):
    # 练习 2：3 个城市 × 3 个时刻的气温（行 = 城市，列 = 时刻）
    #
    # 任务：每个城市减去“该城市三个时刻的均值”（行中心化）
    # —— 演示 2 的翻车现场换了个数据，keepdims 的实战
    temps = np.array(
        [
            [12.5, 13.0, 11.8],  # ← 城市 0
            [22.3, 25.1, 24.0],  # ← 城市 1
            [18.7, 18.2, 17.9],  # ← 城市 2
        ]
    )

    # 提示：先看看 temps 的 shape——它和演示 2 里的方阵有什么共同点？
    #       再想想“直接减”会发生什么，最后写出安全版本

    centered_temps = None   # ← 把 None 换成你的实现（一行就够）

    centered_temps if centered_temps is not None else "（还是 None——想想为什么这里必须 keepdims）"
    return (centered_temps, temps)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 2 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                centered_temps = temps - temps.mean(axis=1, keepdims=True)
                ```

                结果（每行之和全为 0，浮点误差量级）：

                ```
                [[ 0.0667  0.5667 -0.6333]
                 [-1.5     1.3     0.2  ]
                 [ 0.4333 -0.0667 -0.3667]]
                ```

                为什么 keepdims 在这里**不是可选项、是必需品**？

                - `temps.mean(axis=1)` 的 shape 是 (3,)——而 temps 恰好是 (3, **3**) 的**方阵**
                - 方阵意味着 (3, 3) - (3,) 广播**恰好能算通**：(3,) 被右对齐补成 (1, 3)、
                  再把 1 拉成 3，每行减掉的是“[城市 0 的均值, 城市 1 的均值, 城市 2 的均值]”
                  这个错位向量——结果悄悄全错、一声不吭
                  （错误版的每行之和是 [-17.2 16.9 0.3]，根本不是 0）
                - `keepdims=True` 把均值变成 (3, 1) 的“列”，广播时对上行维度，
                  每行减的才是**自己**的均值

                这正是演示 2 的方阵陷阱换了件马甲。非方阵时通常当场报错（走运），
                **方阵最危险：它能“成功地”算错。**
                """
            )
        }
    )
    return


@app.cell
def _(mo, np):
    # 练习 3 · 预测题：tuple 形式的 axis——一次压掉多个轴
    #
    # a_ex3：2 张 3×3 的图，shape (2, 3, 3) = (批, 高, 宽)
    a_ex3 = np.arange(18).reshape(2, 3, 3)

    # ✍️ 先别运行！把预测写进下面两行注释里：
    #    我预测 a_ex3.sum(axis=(0, 2)) 的 shape 是：______
    #    我预测它的含义是：______
    #
    # （提示：axis 不一定是单个数，还可以是一个 tuple，一次压掉好几个轴；
    #   心法不变——结果 shape = 原 shape 删掉那些位置的数）

    # 写好预测，再把 None 换成 a_ex3.sum(axis=(0, 2))，运行对照：
    sum_axes_ex3 = None

    _result_msg = (
        f"运行结果：shape = **{sum_axes_ex3.shape}**，数组 = `{sum_axes_ex3}`"
        if sum_axes_ex3 is not None
        else "⏳ 还没填答案：先在上面两行注释里写下你的预测，再把 None 换成 a_ex3.sum(axis=(0, 2))，运行对照。"
    )

    mo.md(
        f"""
        {_result_msg}

        和你的预测对上了吗？对上了就去展开参考答案核对“含义”；
        没对上，回到概念 2 的万能公式再推一遍。
        """
    )
    return (a_ex3, sum_axes_ex3)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 3 参考答案（先自己写过再看！）": mo.md(
                """
                答案：shape 是 **(3,)**——同时压掉批（axis 0）和宽（axis 2），只剩下高。

                分解动作：

                - `axis=(0, 2)`：axis 不一定是单个数，还可以是 **tuple**，一次压掉多个轴
                - 结果 shape = (2, 3, 3) 删掉位置 0 和位置 2 → **(3,)**
                - 数值含义：对每一个“高度位置”，把所有图、所有宽度上的数全部加起来
                  （[33 51 69]：高 = 0 那一层的 0~2 和 9~11 全加起来是 33，以此类推）

                验证直觉的等价写法——一次压多个轴 = 一个一个压（顺序无所谓）：

                ```python
                a_ex3.sum(axis=(0, 2))          # (3,)
                a_ex3.sum(axis=2).sum(axis=0)   # 先压宽、再压批，殊途同归 → (3,)
                ```

                两种写法结果完全相等。以后遇到“批和特征一起统计”的需求，一个 tuple 就搞定。
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
                1. axis 就是 **shape 元组里的位置编号**：shape (3, 4) → axis 0 管 3（行）、axis 1 管 4（列）；负数 axis 从右往左数，-1 永远是最后一个维度
                2. 聚合 = **消灭那个轴**：`sum(axis=0)` 沿 axis 0 方向叠压、该轴从结果中消失；结果 shape = 原 shape 删掉那个位置的数
                3. 不给 axis（如 `g.sum()`）= 全部轴一起压，只剩一个标量
                4. **判题心法**：先写结果 shape，再算数——shape 对了数值几乎不会错，shape 错了数值对了也是蒙的
                5. `keepdims=True` 让被压掉的轴保留为 1（(3,4) → (3,1)），**专为广播而生**；“每行减自己的均值”必须用它或 `[:, None]`——否则轻则报错、重则静默错位（方阵最危险）
                6. `argmax` 返回最大值的**位置**而不是值——softmax 取预测类别时的标配
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

        **下一课预告** — `09_线性代数操作`：

        到今天为止，NumPy 数组的“语法”你已经全部学完：怎么造、怎么取、怎么广播、
        怎么变形、怎么聚合。下一课进入“数组的**代数**”——**点积、矩阵乘法 `@`、范数**。

        别被名字吓住：这三个词，就是神经网络前向传播的**全部运算**——
        一层网络本质上不过是一次矩阵乘法加一次偏移。学完下一课，
        你手里就有了搭第一个“线性层”的全部零件。

        第 1 部分只剩最后两课。翻过今天这座山，冲顶见！🏔️
        """
    )
    return


if __name__ == "__main__":
    app.run()
