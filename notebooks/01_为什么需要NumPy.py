import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="01 · 为什么需要 NumPy")


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
    from time import perf_counter

    return (perf_counter,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # 01 · 为什么需要 NumPy？

    **从零到 Transformer · 手写深度学习** ｜ 第 1 部分 · NumPy 从零开始（1/10）
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📖 本课目标（先读我）": mo.md(
                """
                **这一课只解决一个问题：为什么深度学习离不开 NumPy？**

                学完你将能：

                - 亲手测出纯 Python 与 NumPy 的速度差距（通常是**上百倍**的量级）
                - 说清楚这个差距的真正原因——不是"因为它是库"，而是**内存布局**的本质区别
                - 理解"向量化"这个词，并从此开始用向量化的思维写代码

                本课不需要任何 NumPy 基础，我们从零开始。
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "🐍 概念 1/3 · Python 列表：为灵活性付出了速度": mo.md(
                """
                你早就和列表打过交道：`[1, 2, 3]`。它极其灵活——可以随时增删，还可以混装任意类型：
                `[42, "你好", 3.14, [又一个列表]]` 完全合法。

                但这份灵活有代价。在最常用的 Python 实现（CPython）里，列表存的其实是一串**指针**，
                每个指针指向一个飘在内存别处的**完整对象**。哪怕只是一个整数 3，也是一个约 28 字节的
                对象——除了数值本身，还背着类型标签、引用计数等"管理信息"。

                于是 `x * x` 这样看似"一步"的运算，背后是：跟着指针找到对象 → 读类型标签确认它是
                int → 取出数值 → 计算 → 再包一个新对象。循环一百万次，这套流程就走一百万遍，
                而**每一步都是 Python 解释器在慢悠悠地执行字节码**。
                """
            ),
            "🚀 概念 2/3 · NumPy 的答案：ndarray 的三条纪律": mo.md(
                """
                NumPy 的核心数据结构叫 **ndarray**（n 维数组）。它给自己立了三条纪律：

                1. **类型统一**：一个数组里所有元素必须是同一种类型（这个类型叫 `dtype`，下一课细讲）
                2. **内存连续**：所有元素像火车车厢一样首尾相连，排在一整块内存里
                3. **整块运算**：对数组的操作一次性交给编译好的 C 代码去执行

                这三条共同造就了速度（马上用计时赛验证），也解释了为什么数组不能像列表那样混装。

                **类比**：列表像出租车——招手就停、哪都能去，但一次只运你一个；
                数组像地铁——只能沿固定轨道跑，但一次运走一整车厢的人。
                深度学习恰恰是"每天通勤几十亿人次"的场景，所以必须修地铁。
                """
            ),
            "🔤 概念 3/3 · 核心词汇：向量化（vectorization）": mo.md(
                """
                **向量化 = 用"对整个数组的运算"代替"对每个元素的显式循环"。**

                ```python
                # 非向量化：肉眼可见的循环
                ys = []
                for x in xs:
                    ys.append(3 * x + 5)

                # 向量化：循环消失了，被"整块运算"取代
                ys = 3 * xs + 5        # xs 是 NumPy 数组
                ```

                向量化三连收益：

                - **快**：循环在 C 层飞驰
                - **短**：一行顶四行
                - **像数学**：代码和公式 y = 3x + 5 长得一模一样——这对后面读懂神经网络的公式至关重要

                从今天起，每次想写 `for` 循环处理数字时，先停一秒问自己：
                **"这件事能向量化吗？"**
                """
            ),
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    n_race = 1_000_000  # 比赛规模：一百万个数
    mo.md(
        f"""
        ## 🏁 代码演示 · 一场计时赛

        任务很简单：**给 {n_race:,} 个整数（0 到 {n_race - 1:,}）逐个求平方**。

        三位选手：

        - 选手 1：最朴素的 `for` 循环 + `append`
        - 选手 2：列表推导式（Python 中更优雅的循环写法）
        - 选手 3：NumPy 向量化

        计时工具是 `perf_counter()`：它返回一个高精度时钟读数，
        **调用两次、相减**，就是中间那段代码的耗时（单位：秒）。

        依次运行下面三个单元，然后看成成绩公布。👉
        """
    )
    return (n_race,)


@app.cell
def _(mo, n_race, perf_counter):
    # 选手 1：纯 Python 循环
    #
    # 思路：准备一个空列表，循环一百万次，每次把 x*x 追加进去。

    _t0 = perf_counter()                    # ⏱️ 按下秒表

    _squares_list = []                      # 空列表，用来收集结果
    for _x in range(n_race):                # 循环一百万次
        _squares_list.append(_x * _x)       # 每轮：取数 → 乘法 → append（一次函数调用）

    py_loop_time = perf_counter() - _t0     # ⏱️ 停表：两次读数相减 = 耗时

    mo.md(f"**选手 1 · 纯 Python 循环**：{py_loop_time:.4f} 秒")
    return (py_loop_time,)


@app.cell
def _(mo, n_race, perf_counter):
    # 选手 2：列表推导式
    #
    # [表达式 for 变量 in 可迭代对象] —— 一行写完"循环 + 收集"。
    # 语法更漂亮，也确实比 append 快一点，但本质仍是"逐个处理"。

    _t0 = perf_counter()
    _squares_lc = [_x * _x for _x in range(n_race)]   # 语法糖：同样的循环，浓缩成一行
    listcomp_time = perf_counter() - _t0

    mo.md(f"**选手 2 · 列表推导式**：{listcomp_time:.4f} 秒")
    return (listcomp_time,)


@app.cell
def _(mo, n_race, np, perf_counter):
    # 选手 3：NumPy 向量化
    #
    # np.arange(n_race) 生成 [0, 1, 2, ...] 这样的数组（创建方式下一课细讲）。
    # 关键在 ** 2：它作用在"整个数组"上，一百万次平方在 C 层一口气完成。

    _t0 = perf_counter()
    _squares_arr = np.arange(n_race) ** 2   # 整个数组一起求平方，没有可见的循环
    np_time = perf_counter() - _t0

    # 顺手抽查前 5 个结果，确认三位选手算的是同一件事：
    _check = [x * x for x in range(5)]      # 用最笨的办法算出 [0, 1, 4, 9, 16]

    mo.md(
        f"""
        **选手 3 · NumPy 向量化**：{np_time:.4f} 秒

        结果抽查（前 5 个平方数）：`{_check}` ↔ `{_squares_arr[:5].tolist()}` ✅
        """
    )
    return (np_time,)


@app.cell(hide_code=True)
def _(listcomp_time, mo, np_time, py_loop_time):
    _speedup = py_loop_time / np_time   # 选手 3 比选手 1 快多少倍

    mo.vstack(
        [
            mo.md(
                f"""
                ## 🏁 成绩公布

                | 选手 | 耗时 | 比最慢的快 |
                |---|---|---|
                | 1 · 纯 Python 循环 | {py_loop_time:.4f} 秒 | 1.0 倍（基准） |
                | 2 · 列表推导式 | {listcomp_time:.4f} 秒 | {py_loop_time / listcomp_time:.1f} 倍 |
                | 3 · NumPy 向量化 | {np_time:.4f} 秒 | **{_speedup:,.0f} 倍** 🚀 |
                """
            ),
            mo.callout(
                mo.md(
                    """
                    **两个观察：**

                    1. 列表推导式比普通循环略快（少了每轮一次的函数调用查找），但**仍是同一量级**——
                       它没有跳出"逐个处理"的框架
                    2. NumPy 快了两个数量级。它并没有让 Python 循环变快，而是**把整件事打包交给了
                       编译好的 C 代码**——循环依然存在，只是在 C 里转，你感知不到

                    每次运行的数字都会有波动，不同机器也不一样——**看数量级就好**。
                    """
                ),
                kind="success",
            ),
        ]
    )
    return


@app.cell
def _(mo, np):
    import sys as _sys

    # 同样的一百万个整数，分别用列表和数组存放，看看各自的"内存账单"
    _nums_list = list(range(1_000_000))     # Python 列表版
    _nums_arr = np.arange(1_000_000)        # NumPy 数组版

    # 列表的账单分两部分：
    _list_container = _sys.getsizeof(_nums_list)   # ① 列表本体：只存放"指针"（一列门牌号）
    _int_object = _sys.getsizeof(1_000_000)        # ② 单个 int 对象自身的大小（约 28 字节）
    _list_total = _list_container + _int_object * len(_nums_list)

    _arr_bytes = _nums_arr.nbytes                  # NumPy：一整块连续内存，账单只有一份

    mo.md(
        f"""
        ## 🔬 但是，为什么这么快？

        同样存放 0 ~ 999,999 这一百万个整数：

        | | Python 列表 | NumPy 数组 |
        |---|---|---|
        | 内存占用（约） | {_list_total / 1e6:.0f} MB | {_arr_bytes / 1e6:.0f} MB |
        | 元素存放方式 | 散落在内存各处的独立对象 | 连续排布的一整块数字 |
        | 取第 i 个元素 | 先读指针，再跳到对象所在位置 | 直接算地址：起点 + i × 8 字节 |

        列表像**一串氢气球**——列表手里只攥着一根系绳（指针），每个数字都是飘在别处的独立气球；
        数组像**一列火车**——车厢首尾相连，报出车厢号就能直达。

        连续内存还附赠两个隐藏福利：CPU 缓存可以整块预读（访问下一个元素几乎免费），
        并且能用上 CPU 的 **SIMD 指令**（一条指令同时算好几个数）。这些不需要背，记住结论即可：

        > **同类型 + 连续内存 + 整块运算 = 快**
        """
    )
    return


@app.cell
def _(mo, np):
    # 向量化不只是快，还让代码读起来像数学公式
    x_demo = np.arange(1, 6)        # 生成 [1, 2, 3, 4, 5]
    y_demo = 3 * x_demo + 5         # 对整个数组"乘 3 加 5"，一步到位

    # 注意 3 * x_demo：3 是一个单独的数（标量），数组里每个元素都乘上了它
    # ——这个行为在第 06 课有个正式名字，叫"广播"，现在先记住现象

    mo.hstack(
        [
            mo.md(f"x = `{x_demo}`"),
            mo.md(f"y = 3x + 5 = `{y_demo}`"),
        ]
    )
    return


@app.cell
def _(mo, np):
    # 再来一发"一行代码做统计"——8 位同学的期末成绩
    scores = np.array([62, 88, 55, 94, 73, 81, 47, 90])

    mo.md(
        f"""
        成绩单：`{scores}`

        | 想知道的事 | NumPy 写法 | 结果 |
        |---|---|---|
        | 平均分 | `scores.mean()` | {scores.mean():.1f} |
        | 最高分 | `scores.max()` | {scores.max()} |
        | 最低分 | `scores.min()` | {scores.min()} |
        | 及格率 | `(scores > 60).mean()` | {(scores > 60).mean():.0%} |

        最后一行是不是有点神奇？——`scores > 60` 先生成一串 True/False，
        `.mean()` 把 True 当 1、False 当 0 求平均，恰好就是及格人数的比例。

        这套**布尔索引**玩法是第 05 课的主角，这里先混个脸熟。
        """
    )
    return


@app.cell
def _(mo):
    size_slider = mo.ui.slider(
        1,
        7,
        value=6,
        step=1,
        debounce=True,           # 拖动停止后才触发，避免边拖边算
        label="🎛️ 数据规模：10 的几次方？（拖到 7 要等一两秒）",
    )
    size_slider
    return (size_slider,)


@app.cell
def _(mo, np, perf_counter, size_slider):
    n_interact = 10**size_slider.value   # 真实的数据量

    # 选手 A：纯 Python 循环
    _t0 = perf_counter()
    _result_list = []
    for _i in range(n_interact):
        _result_list.append(_i * _i)
    _py_secs = perf_counter() - _t0

    # 选手 B：NumPy 向量化
    _t0 = perf_counter()
    _result_arr = np.arange(n_interact) ** 2
    _np_secs = perf_counter() - _t0

    _speedup_live = _py_secs / _np_secs

    mo.md(
        f"""
        ## 🎛️ 交互实验 · 把规模拖大看看

        当前 n = **{n_interact:,}**：

        | 选手 | 耗时 |
        |---|---|
        | 纯 Python 循环 | {_py_secs:.4f} 秒 |
        | NumPy 向量化 | {_np_secs:.4f} 秒 |

        **NumPy 快了约 {_speedup_live:,.0f} 倍。**

        拖动滑块从 1 到 7，观察两件事：

        1. 两个选手的耗时都在增长，但 NumPy 的增长"温和"得多
        2. 规模越大，倍数差距通常越明显——因为启动 NumPy 的固定开销被摊薄了
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

                - **练习 1 · 调和级数**：向量化计算 1/1 + 1/2 + ... + 1/1,000,000
                - **练习 2 · 正弦对决**：`math.sin` 循环 vs `np.sin`，亲手测出倍数
                - **练习 3 · 思考题**：为什么 NumPy 一定要求数组元素类型一致？
                """
            )
        }
    )
    return


@app.cell
def _():
    # 练习 1：计算调和级数前 n 项的和：1/1 + 1/2 + 1/3 + ... + 1/n
    # 要求：用"向量化"写法，不许写 for 循环！
    #
    # 提示（三步拼图）：
    #   ① np.arange(1, n_ex1 + 1)        → 生成 [1, 2, 3, ..., n]
    #      （注意：arange 不含终点，所以要 +1，下一课会细讲）
    #   ② 1.0 / 上面的数组               → 每一项都取倒数
    #   ③ 对结果调用 .sum()              → 全部加起来

    n_ex1 = 1_000_000

    harmonic_sum = None   # ← 把 None 换成你的向量化实现（一行就够）

    harmonic_sum if harmonic_sum is not None else "（还是 None——把上面的 None 换成你的代码，再运行一次）"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 1 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                harmonic_sum = (1.0 / np.arange(1, n_ex1 + 1)).sum()
                ```

                分解动作：

                - `np.arange(1, n_ex1 + 1)` → `[1, 2, 3, ..., 1000000]`（终点要多取 1）
                - `1.0 / 数组` → 每一项都取倒数：`[1.0, 0.5, 0.333..., ...]`
                - `.sum()` → 全部加起来

                n = 1,000,000 时，结果约为 **14.3927**。

                顺带一提：调和级数增长极其缓慢，它近似等于 ln(n) + 0.5772。
                n 从 100 涨到一百万（一万倍），这个和只增加了约 9.2。
                """
            )
        }
    )
    return


@app.cell
def _(perf_counter):
    # 练习 2：正弦对决——给一百万个数计算 sin 值，比比谁快
    import math as _math   # Python 标准库里的数学函数（逐个计算型选手）

    n_ex2 = 1_000_000

    # 选手 A：math.sin + 列表推导（已经帮你写好了）
    _t0 = perf_counter()
    sines_python = [_math.sin(v / 100) for v in range(n_ex2)]
    py_sin_time = perf_counter() - _t0

    # 选手 B：NumPy——请把 None 换成你的向量化实现
    # 提示：np.arange(n_ex2) 生成 [0, 1, ..., n-1]，
    #       除以 100 之后整体喂给 np.sin(...)，一行搞定
    _t0 = perf_counter()
    sines_numpy = None
    np_sin_time = perf_counter() - _t0

    if sines_numpy is None:
        ex2_msg = "👆 先把 sines_numpy = None 换成你的 NumPy 实现，然后重新运行"
    else:
        ex2_msg = f"选手 A：{py_sin_time:.4f} 秒 ｜ 选手 B：{np_sin_time:.4f} 秒"
    ex2_msg
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 2 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                sines_numpy = np.sin(np.arange(n_ex2) / 100)
                ```

                典型结果：`math.sin` 列表推导约 0.1 秒，`np.sin` 约 0.01 秒，
                大约快 **10~30 倍**（视机器而定）。

                注意：倍数比"求平方"那一局小——因为 sin 本身计算量大，
                "逐个调用"的开销占比就相对小了。**向量化不是魔法倍率，赢在省掉逐个调度的开销**，
                所以操作越轻量、规模越大，优势越夸张。
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
                Python 列表可以混装：`[1, "两", 3.0, [4, 5]]` 完全合法；
                但 NumPy 数组要求所有元素**同一种类型**。

                **问：这条"纪律"和它的快，有什么必然联系？**

                （提示：回想"火车"的比喻，以及"直接算地址：起点 + i × 8 字节"这句话。）
                """
            ),
            "💡 练习 3 参考答案（先想 3 分钟再看）": mo.md(
                """
                因为"直接算地址"这条捷径，**前提是每个元素的大小完全一样**：
                已知起点、已知每节车厢的长度，第 i 节的位置一算就出，不用从头数。

                如果元素类型可以不同（有的 28 字节、有的一条 40 字节、有的是字符串），
                就没法用乘法定位，只能像列表那样挨个存指针、挨个跳转——快就无从谈起。

                一句话：**类型统一是连续内存的前提，连续内存是速度的前提。**

                "灵活"与"快"常常不可兼得，NumPy 选择了快。深度学习每天要做几十亿次
                同一种类型的运算，自然站 NumPy 这边。
                """
            ),
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📝 本课小结（展开自查：5 条都答得上来吗？）": mo.md(
                """
                1. Python 列表慢的根源：每个元素是**独立对象**、散落在内存各处，运算时逐个"找对象、拆包装"
                2. NumPy 数组快的三板斧：**同类型 + 连续内存 + 把循环交给 C**
                3. **向量化** = 用"对整个数组的运算"代替"对元素的显式循环"，又快、又短、又像数学公式
                4. `perf_counter()` 两次调用相减，可以给任意代码段计时——后面课程会反复用到
                5. 深度学习 = 海量同类型数值计算，所以 NumPy 是整门课的地基
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---

    **下一课预告** — `02_解剖ndarray`：

    今天我们把 NumPy 数组当成了一辆黑盒跑车——只知道它快。
    下一课要打开引擎盖：`shape`、`dtype`、`ndim`、`size`、`nbytes`……
    这些属性决定了数组"长什么样、占多少内存"。

    搞懂它们，以后遇到任何形状（shape）相关的报错，你都能自己看懂、自己修——
    那可是深度学习里最高频的报错类型。

    🚗 发车，去第 02 课！
    """)
    return


if __name__ == "__main__":
    app.run()
