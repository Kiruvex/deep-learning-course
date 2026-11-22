import marimo

__generated_with = "0.25.2"
app = marimo.App(app_title="02 · 解剖ndarray", width="medium")


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
        # 02 · 解剖ndarray

        **从零到 Transformer · 手写深度学习** ｜ 第 1 部分 · NumPy 从零开始（2/10）
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📖 本课目标（先读我）": mo.md(
                """
                **这一课只解决一个问题：拿到任何一个数组，能不能立刻"读"出它的全部身份信息？**

                上一课我们把数组当成一辆黑盒跑车——只知道它快。这一课打开引擎盖：
                `shape`、`dtype`、`ndim`、`size`、`nbytes` 这五个属性，
                决定了数组"长什么样、住着什么、占多少内存"。

                学完你将能：

                - 一眼读出上面五个字段，并用"一栋楼"的比喻向别人解释它们
                - 说清楚深度学习为什么偏爱 float32、数字图片为什么常用 uint8
                - 亲眼见识整数溢出（255 + 1 = 0）和 float16 的精度丢失，从此对 dtype 心存敬畏

                本课不需要新知识——只需要上一课那句"数组 = 同类型 + 连续内存"的印象。
                """
            ),
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "🪪 概念 1/3 · 数组的身份证：五个字段": mo.md(
                """
                把一个 ndarray 想象成**一栋楼**，它的身份证上有五个字段：

                | 字段 | 问的问题 | 一栋楼的比喻 |
                |---|---|---|
                | `shape` | 每个方向上各有多少个元素（从外往里数） | 这栋楼有几层、每层几间房 |
                | `dtype` | 元素是什么类型（全楼同一种） | 每间房里住的"是什么人" |
                | `ndim` | 有几个方向 | 身份证上写了几项——恒等于 `len(shape)` |
                | `size` | 元素总个数 | 全楼总共多少间房 |
                | `itemsize` / `nbytes` | 每个元素 / 全部元素占多少字节 | 每间房多大 / 全楼总面积 |

                三个换算关系，记住一个就等于记住三个：

                - `ndim == len(shape)`：shape 是 (4, 5, 3)，ndim 就是 3
                - `size == 各维相乘`：4 × 5 × 3 = 60
                - `nbytes == size × itemsize`：60 个元素 × 每个 8 字节 = 480 字节

                读 shape 的口诀（整个 NumPy 生涯都靠它）：**从外往里数，一层一个数**。
                (4, 5, 3) 就是：最外层 4 个"行"，每行 5 个"像素"，每个像素 3 个"数"。

                以后遇到任何形状（shape）相关的报错——深度学习里最高频的一类——
                第一反应就是把这五个字段打印出来对账。这一课就是在练这个反射。
                """
            ),
            "🔢 概念 2/3 · dtype：数组的血液类型": mo.md(
                """
                dtype（data type）规定了一个数组里**所有元素**的类型与宽度。
                上一课说过"类型统一"是数组的三条纪律之首——dtype 就是这条纪律的名字。

                | 家族 | 成员 | 每个元素占 |
                |---|---|---|
                | 有符号整数 | int8 / int16 / int32 / int64 | 1 / 2 / 4 / 8 字节 |
                | 无符号整数 | uint8 / uint16 / uint32 / uint64 | 1 / 2 / 4 / 8 字节（只装非负数） |
                | 浮点数 | float16 / float32 / float64 | 2 / 4 / 8 字节 |
                | 布尔 | bool | 1 字节（只存 0 或 1） |

                同一个数字，住在不同 dtype 里，房租完全不同：数字 5 用 int64 存要 8 字节，
                用 int8 存只要 1 字节。**数字没变，变的是"房型"。**

                深度学习几乎只用 **float32**，三个理由：

                1. **内存减半**：比 float64 省一半——模型一胖，省的就是真金白银
                2. **硬件加速**：CPU/GPU 的向量化指令对 float32 支持最充分
                3. **精度够用**：float32 有约 7 位有效数字，而神经网络天生容忍误差
                   （梯度本来就是"大概方向"），完全够训练

                类型转换用 `arr.astype(新类型)`——注意它**返回一个新数组**，原数组纹丝不动。
                而且要警告在先：**float 转 int 不是四舍五入，是"向零截断"**（砍掉小数部分），
                马上在演示里见识。
                """
            ),
            "🕸️ 概念 3/3 · 维度直觉：从 0 维到 4 维": mo.md(
                """
                别背数学定义，用生活画面记：

                | 维度 | 数学名字 | 生活画面 | 例子 |
                |---|---|---|---|
                | 0 维 | 标量 | 就一个数 | 此刻气温：25.0 |
                | 1 维 | 向量 | 一行数 | 一天 24 小时的气温表，shape (24,) |
                | 2 维 | 矩阵 | 一张表格 | 成绩单（学生 × 科目）、灰度图（高 × 宽） |
                | 3 维 | —— | 一张彩色图片 | 高 × 宽 × RGB 三个通道 |
                | 4 维 | —— | 一批图片 | 批量 × 高 × 宽 × 通道 |

                最值得建立的是 **3 维图片直觉**：彩色图不是"一张平面"，而是三张灰度图叠在一起——
                R、G、B 各占一层，每层记录"这种颜色有多亮"。
                所以 (高, 宽, 3) 里那个 3，就是三张叠在一起的"透明胶片"。

                4 维就是再往上摞一层"批"：网络训练时一次不只看一张图，而是一次看一批（比如 32 张），
                shape 变成 (32, 高, 宽, 3)。训练动辄几万张图，按"批"打包喂给 GPU 才划算——
                这就是深度学习里天天挂在嘴边的 **batch（批）** 的来源。

                一句话总结：**每多一个维度，就是多套一层"容器"。**
                容器里装的是数（0 维）还是又一 层容器（更高维），仅此而已。
                """
            ),
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 🖥️ 代码演示 · 五张身份证

        接下来抓五个"嫌疑人"来验明正身：

        1. 一张假想的彩色小图（3 维数组的身份证）
        2. 一维成绩单 vs 二维成绩单（同样的数据，不同的 shape）
        3. 四种 dtype 的内存账单 + `.astype()` 截断现场 + float16 精度丢失
        4. 整数溢出现场（本课最震撼的一幕）
        5. 深度学习的算账视角：dtype 就是预算

        每个单元都值得逐行读注释——注释讲的是"为什么"。
        """
    )
    return


@app.cell
def _(mo, np):
    # 演示 1：给一张"假彩色图片"办身份证
    #
    # np.arange(60) 生成 [0, 1, 2, ..., 59] 共 60 个整数（造数组的方式下一课细讲）
    # .reshape(4, 5, 3) 把这 60 个数装进一个三层嵌套的盒子：
    #   4 个"行"，每行 5 个"像素"，每个像素 3 个通道（R/G/B）
    # ——想象一张 4 像素高、5 像素宽的彩色小图，数字本身只是道具
    _fake_img = np.arange(60).reshape(4, 5, 3)

    # 五个字段逐个"问诊"。注意它们都是【属性】，访问时不加括号：
    #   shape    每个方向上有多少个元素（一个元组，从外往里数）
    #   dtype    元素的类型（全数组同一种——上一课的"纪律一"）
    #   ndim     维度个数，永远等于 len(shape)
    #   size     元素总个数，永远等于各维相乘
    #   itemsize 每个元素占多少字节
    #   nbytes   整个数组占多少字节 = size × itemsize

    mo.md(
        f"""
        数组内容（只展示前两行像素）：`{_fake_img[:2].tolist()}`

        | 身份证字段 | 值 | 怎么读 |
        |---|---|---|
        | `shape` | `{_fake_img.shape}` | 4 行 × 每行 5 像素 × 每像素 3 通道 |
        | `dtype` | `{_fake_img.dtype}` | 64 位有符号整数，每个元素 8 字节 |
        | `ndim` | `{_fake_img.ndim}` | 三个维度（= len(shape)） |
        | `size` | `{_fake_img.size}` | 4 × 5 × 3 = 60 个元素 |
        | `itemsize` | `{_fake_img.itemsize}` 字节 | int64 → 每个元素 8 字节 |
        | `nbytes` | `{_fake_img.nbytes}` 字节 | size × itemsize = 60 × 8 = 480 |

        换算关系自查：`ndim == len(shape)` ✅ ／ `size == 各维相乘` ✅ ／ `nbytes == size × itemsize` ✅

        一句话读出这张身份证："这是一个 3 维、共 60 个 int64 元素、
        住着 480 字节连续内存的数组。"——上一课说数组住"一整块连续内存"，
        nbytes 就是那块地的面积。
        """
    )
    return


@app.cell
def _(mo, np):
    # 演示 2：同一份数据，可以有不同的"维度长相"
    #
    # 场景 A：8 位同学的数学成绩——最朴素的 1 维数组（向量）
    _scores_1d = np.array([62, 88, 55, 94, 73, 81, 47, 90])

    # 场景 B：3 位学生 × 4 门科目的成绩单——2 维数组（矩阵）
    # 写法就是"列表的列表"：外层 3 个内层列表 = 3 行，每个内层列表 4 个数 = 4 列
    _grade_2d = np.array(
        [
            [92, 88, 79, 96],   # 小明的 语·数·英·科
            [78, 95, 84, 88],   # 小红的
            [85, 76, 92, 90],   # 小刚的
        ]
    )

    mo.md(
        f"""
        **1 维 · 一列成绩**：`{_scores_1d.tolist()}`

        - shape = `{_scores_1d.shape}` → 元组里只有一个数：一个方向上排了 8 个元素
        - ndim = {_scores_1d.ndim}，size = {_scores_1d.size}

        **2 维 · 一张成绩单**：`{_grade_2d.tolist()}`

        - shape = `{_grade_2d.shape}` → 两个数：外层 3 个列表（3 个学生）× 内层各 4 个数（4 门课）
        - ndim = {_grade_2d.ndim}，size = {_grade_2d.size}（3 × 4 = 12，没有魔法）

        口诀再念一遍：**从外往里数，一层一个数**。
        (3, 4) 读作"3 行 4 列"——前面的数管外层（行），后面的数管内层（列）。

        数据还是那点数据，但"装进几个维度"是可以选的——
        第 07 课的 `reshape` 干的就是这件事：size 不变，换个 shape。
        """
    )
    return


@app.cell
def _(mo, np):
    # 演示 3：dtype 深入——同样的数字，不同的"血液类型"
    #
    # 用同一份列表分别造四个数组，唯一不同的是 dtype：
    _same = [10, 20, 30, 40, 50]

    _a_int32 = np.array(_same, dtype=np.int32)    # 32 位整数：每个 4 字节
    _a_int64 = np.array(_same, dtype=np.int64)    # 64 位整数：每个 8 字节
    _a_f32 = np.array(_same, dtype=np.float32)    # 32 位浮点：每个 4 字节
    _a_f64 = np.array(_same, dtype=np.float64)    # 64 位浮点：每个 8 字节

    # 观察重点：四个数组【数值一模一样】，账单却差一倍——
    # dtype 名字里的数字（32/64）说的是"每个元素用几个比特存"，
    # 房间越大，能装的数越大越精细，但房租也越贵
    mo.md(
        f"""
        | dtype | 能耐（大致） | itemsize | 5 个元素共 nbytes |
        |---|---|---|---|
        | `int32` | 数到 ±21 亿 | {_a_int32.itemsize} 字节 | {_a_int32.nbytes} 字节 |
        | `int64` | 数到 ±9.2 × 10^18 | {_a_int64.itemsize} 字节 | {_a_int64.nbytes} 字节 |
        | `float32` | 约 7 位有效数字 | {_a_f32.itemsize} 字节 | {_a_f32.nbytes} 字节 |
        | `float64` | 约 16 位有效数字 | {_a_f64.itemsize} 字节 | {_a_f64.nbytes} 字节 |

        选 dtype = 在"能耐"和"房租"之间做交易。

        顺带两个默认值（Python 里 `np.array([10, 20])` 不指定 dtype 时）：
        整数默认 `int64`、带小数点的数默认 `float64`——都是"最大房型"，
        安全但费内存，所以深度学习代码里几乎总能看到显式写出的 `dtype=np.float32`。
        """
    )
    return


@app.cell
def _(mo, np):
    # 演示 3 续 · 之一：.astype() 类型转换——是"砍"，不是"舍"
    #
    # 两个要点：
    #   ① astype 返回【新数组】，原数组纹丝不动（下面会打印原数组 dtype 验证）
    #   ② float → int 是"向零截断"：小数部分直接扔掉，取离 0 更近的那个整数
    _h = np.array([3.7, -3.7, 2.99, -2.99])    # 4 个带小数的数
    _h_int = _h.astype(np.int32)               # float64 → int32

    # 演示 3 续 · 之二：float16 的精度天花板
    #
    # float16 每个元素只有 2 字节 = 16 比特，其中只有 10 比特存有效数字，
    # 于是它能【精确】表示的整数最大只到 2048——再大就会被"挤"到最近的合法值
    _f16 = np.array([2047, 2048, 2049, 4097], dtype=np.float16)

    mo.md(
        f"""
        **截断现场**：`{_h.tolist()}` → astype(int32) → `{_h_int.tolist()}`

        - 3.7 → 3（不是 4）；-3.7 → -3（不是 -4）：**朝 0 的方向砍**，和四舍五入无关
        - 转换后原数组 dtype 仍是 `{_h.dtype}`——想留住新数组，必须接住 astype 的返回值

        **float16 精度丢失现场**：`[2047, 2048, 2049, 4097]` 存成 float16 后变成 `{_f16.tolist()}`

        - 2049 → **2048**：不是 bug，是 float16 根本存不下 2049，只能取最近的合法值
        - 4097 → 4096：数越大，"能存的值"之间隔得越开（2048 以上每格隔 2，4096 以上每格隔 4）

        float16 平时用得少，但"混合精度训练"（用一半内存训大模型）里它是主角，
        到第 4 部分会再遇到。现在记住一句话就行：**位数少 = 便宜但粗糙。**
        """
    )
    return


@app.cell
def _(mo, np):
    # 演示 4：整数溢出——本课最震撼的一幕
    #
    # uint8 = 无符号 8 位整数：只有 8 个比特，只能数 0 ~ 255 共 256 个值
    # int8  = 有符号 8 位整数：同样 8 个比特，要分一半给负数，范围 -128 ~ 127
    #
    # 问题来了：uint8 的 255 再 +1，会发生什么？
    _brightness = np.array([250, 254, 255], dtype=np.uint8)   # 一组像素亮度

    # 数组 + Python 整数：结果 dtype 仍是 uint8，装不下的部分会"回绕"（wrap around）
    _brightness_plus = _brightness + 1     # 255 + 1 → 0：像里程表转到头归零

    # int8 更戏剧化：127 + 1 直接从最大正数穿越到最小负数
    _i8 = np.array([126, 127], dtype=np.int8)
    _i8_plus = _i8 + 1                     # 127 + 1 → -128

    # NumPy 2.x 的贴心之处：两个 uint8【标量】相加溢出时，会明确发出运行时警告。
    # 用 catch_warnings 把警告"录下来"，一会儿展示在下方
    import warnings as _warnings

    with _warnings.catch_warnings(record=True) as _caught:
        _warnings.simplefilter("always")
        _scalar_overflow = np.uint8(255) + np.uint8(1)   # 结果 0，外加一条警告
    _overflow_note = str(_caught[0].message) if _caught else "（这个 NumPy 版本没有发警告）"

    # 还有一道安全网：如果 Python 整数【本身】就超出 dtype 能装的范围，会当场报错拒绝。
    # 1 在 uint8 范围内，所以上面 _brightness + 1 安静地回绕；300 连门都进不去：
    try:
        _rejected = _brightness + 300
        _reject_note = "居然没报错？"
    except OverflowError as _e:
        _reject_note = f"OverflowError: {_e}"

    mo.vstack(
        [
            mo.md(
                f"""
                | 算式 | 数学上的答案 | NumPy 的答案 |
                |---|---|---|
                | `uint8 数组 [250, 254, 255] + 1` | `[251, 255, 256]` | `{_brightness_plus.tolist()}` |
                | `int8 数组 [126, 127] + 1` | `[127, 128]` | `{_i8_plus.tolist()}` |
                | `np.uint8(255) + np.uint8(1)` | `256` | `{_scalar_overflow}`（附一条警告） |
                """
            ),
            mo.callout(
                mo.md(
                    f"""
                    **这叫"回绕"（wrap around）。不是 bug，是规则。**

                    uint8 只有 8 个比特。255 的二进制是 `11111111`（已经装满），+1 本应是
                    9 位的 `100000000`，但只有 8 位可住——**最高位的进位被丢弃**，
                    剩下 `00000000`，也就是 0。int8 的 127（`01111111`）+1 变成 `10000000`，
                    按"有符号"的读法恰好是 -128。

                    NumPy 2.x 的三条行为规则（本机实测）：

                    1. **数组运算是静默回绕**：`{_brightness.tolist()} + 1 = {_brightness_plus.tolist()}`，没有任何提示
                    2. **标量运算会发警告**：警告内容是 "{_overflow_note}"
                    3. **本身超界的 Python 整数直接报错**：{_reject_note}

                    **教训**：dtype 决定"能装多大"。图片亮度天然在 0~255，所以 uint8 安全；
                    但累加计数、大数编号千万别用小整数类型。写代码前先问自己：
                    **我的数最大会到多少？**
                    """
                ),
                kind="warn",
            ),
        ]
    )
    return


@app.cell
def _(mo, np):
    # 演示 5：深度学习视角——dtype 直接换算成真金白银
    #
    # 假设一个模型有 1000 万个参数（对现代模型来说只是零头），
    # 分别用三种浮点类型把它们存起来，看看内存账单：
    _n_params = 10_000_000

    # np.dtype(...).itemsize：查出某种类型每个元素占几字节——不用真的造大数组
    _bill_f16 = _n_params * np.dtype(np.float16).itemsize   # 2 字节/参数
    _bill_f32 = _n_params * np.dtype(np.float32).itemsize   # 4 字节/参数
    _bill_f64 = _n_params * np.dtype(np.float64).itemsize   # 8 字节/参数

    # 用"100 万参数"的小号版本做一次真实抽查
    # （np.zeros：全 0 占位数组，下一课的主角之一，先混个脸熟）
    _probe_f32 = np.zeros(1_000_000, dtype=np.float32).nbytes
    _probe_f64 = np.zeros(1_000_000, dtype=np.float64).nbytes

    mo.md(
        f"""
        ## 💸 dtype 就是预算

        同一个"1000 万参数的模型"，只换 dtype，账单变成：

        | dtype | 每个参数 | 1000 万参数合计 | 换算 |
        |---|---|---|---|
        | `float16` | 2 字节 | {_bill_f16:,} 字节 | {_bill_f16 / 1e6:.0f} MB |
        | `float32` | 4 字节 | {_bill_f32:,} 字节 | {_bill_f32 / 1e6:.0f} MB |
        | `float64` | 8 字节 | {_bill_f64:,} 字节 | {_bill_f64 / 1e6:.0f} MB |

        真实抽查（各造一个 100 万参数的数组读 nbytes）：
        float32 → {_probe_f32:,} 字节 = 4 MB ／ float64 → {_probe_f64:,} 字节 = 8 MB ✅

        而且这只是"参数"一项：训练时**梯度**和**优化器状态**还要各存一份同尺寸数组，
        中间结果更是一批一批地冒出来（第 25 课细讲）——dtype 每大一号，整张账单等比例膨胀。

        这就是"深度学习偏爱 float32"最直白的原因：
        **在"精度够用"和"内存可控"之间，它刚好站在甜点上。**
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 🎛️ 交互实验 · 亲手拼一张图片

        三个滑块决定一张"虚拟图片"的身材：高度 H、宽度 W、通道数 C。

        拖动它们，盯着下面实时刷新的"身份证"——
        重点观察：**shape 里哪个数跟着哪个滑块变？nbytes 又是怎么算出来的？**
        """
    )
    return


@app.cell
def _(mo):
    # 三个滑块各自控制图片的一个维度
    # mo.ui.slider(起点, 终点, value=默认值, step=步长, label=给人看的中文标签)
    h_slider = mo.ui.slider(1, 8, value=4, step=1, label="图片高度 H（行数）")
    w_slider = mo.ui.slider(1, 8, value=5, step=1, label="图片宽度 W（每行的像素数）")
    c_slider = mo.ui.slider(1, 4, value=3, step=1, label="通道数 C（3 = RGB 彩色，1 = 灰度）")

    mo.vstack([h_slider, w_slider, c_slider])
    return (c_slider, h_slider, w_slider)


@app.cell
def _(c_slider, h_slider, mo, np, w_slider):
    _H = h_slider.value   # 当前高度
    _W = w_slider.value   # 当前宽度
    _C = c_slider.value   # 当前通道数

    # 真的造出这张图：全 0 的 float32 数组（内容不重要，"身材"才重要）。
    # 深度学习的惯例：图片进入网络前先转 float32——从这节课就开始养成这个手感
    _my_img = np.zeros((_H, _W, _C), dtype=np.float32)

    _kind = (
        "彩色图片（RGB 三通道）"
        if _C == 3
        else ("灰度图片（单通道）" if _C == 1 else f"{_C} 通道图片")
    )

    mo.md(
        f"""
        你现在拼出的是一张 **{_H} × {_W}** 的{_kind}，身份证如下：

        | 字段 | 值 | 怎么读 |
        |---|---|---|
        | `shape` | `{_my_img.shape}` | {_H} 行 × {_W} 列 × 每像素 {_C} 个通道 |
        | `ndim` | {_my_img.ndim} | 恒等于 len(shape) |
        | `size` | {_my_img.size} | {_H} × {_W} × {_C} = {_my_img.size} |
        | `dtype` | `{_my_img.dtype}` | 造数组时亲手指定的 |
        | `itemsize` | {_my_img.itemsize} 字节 | float32 → 每元素 4 字节 |
        | `nbytes` | {_my_img.nbytes} 字节 | ≈ {_my_img.nbytes / 1024:.2f} KB |

        shape = ({_H}, {_W}, {_C}) 的读法：
        **{_H} 个"行"，每行 {_W} 个"像素"，每个像素 {_C} 个"通道"**。

        三个观察题（先拖滑块验证，再看自己想得对不对）：

        1. 只把高度 +1：shape 的**第一个数**变，nbytes 增加 {_W} × {_C} × 4 = {_W * _C * 4} 字节
        2. 通道从 3 拖到 1（彩色变灰度）：nbytes 直接缩为原来的**三分之一**
        3. 无论怎么拖：size 永远等于三个数相乘——这就是"各维相乘"的含义

        到第 31 课加载 MNIST 手写数字时，图片会以 (批量, 高, 宽, 通道) 的 **4 维**数组
        进入网络——比这里再多套一层"批"的容器。到那天你会想起这个实验。
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

                - **练习 1 · 全 1 矩阵**：造一个 (3, 4) 的 float32 全 1 数组，并算出它占多少字节
                - **练习 2 · 预测截断**：把 4 个浮点数转成 int32——先预测，再运行对照
                - **练习 3 · 思考题**：为什么数字图片几乎都用 uint8 存储？
                """
            ),
        }
    )
    return


@app.cell
def _(mo, np):
    # 练习 1：造一个 shape 为 (3, 4)、dtype 为 float32 的"全 1"数组
    #
    # 两步走：
    #   ① 把 None 换成一行代码。提示：np.ones(形状, dtype=...)
    #      其中"形状"要写成一个元组 (3, 4)——括号和逗号都不能少
    #   ② 运行后读输出，回答：这个数组总共占多少字节？
    #      两种验证方式（两个数应该相同）：
    #        - 属性验证：ones_ex1.nbytes
    #        - 手算验证：元素个数 3 × 4 = 12 个，每个 float32 占 4 字节

    ones_ex1 = None   # ← 把 None 换成你的代码

    if ones_ex1 is None:
        ex1_msg = mo.md("（还是 None——把上面的 None 换成你的代码，再运行一次）")
    else:
        ex1_msg = mo.md(
            f"""
            shape = `{ones_ex1.shape}` ｜ dtype = `{ones_ex1.dtype}`

            - 属性验证：nbytes = **{ones_ex1.nbytes} 字节**
            - 手算验证：3 × 4 × 4 = **{3 * 4 * 4} 字节**

            两个数对得上，说明 shape 和 dtype 都写对了 ✅（对不上就回头检查哪一步走偏了）
            """
        )
    ex1_msg
    return (ex1_msg, ones_ex1)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 1 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                ones_ex1 = np.ones((3, 4), dtype=np.float32)
                ```

                分解动作：

                - `(3, 4)` 是一个元组：3 行 × 4 列。常见笔误是写成 `np.ones(3, 4)`——
                  NumPy 会把 4 当成 dtype 参数，然后一脸困惑地报错
                - `dtype=np.float32`：每个元素 4 字节。不写的话默认 float64（每元素 8 字节），
                  账单直接翻倍

                答案：**48 字节**。两种算法殊途同归：

                - 属性验证：`ones_ex1.nbytes` → 48
                - 手算验证：size × itemsize = (3 × 4) × 4 = 12 × 4 = 48
                """
            ),
        }
    )
    return


@app.cell
def _(mo, np):
    # 练习 2：预测截断——先动笔，再运行！
    #
    # 下面的 raw_ex2 是一组 float64。三步走：
    #   ① 先在纸上（或脑子里）写下你预测的 4 个转换结果
    #   ② 把 None 换成你的 astype 代码（提示：raw_ex2.astype(...)）
    #   ③ 运行对照——如果错了，回看演示 3 的"向零截断"
    raw_ex2 = np.array([1.9, -2.9, 3.5, -0.5])

    ints_ex2 = None   # ← 把 None 换成你的代码

    if ints_ex2 is None:
        ex2_msg = mo.md("👆 先写下你的预测，再把 None 换成你的代码，然后重新运行")
    else:
        ex2_msg = mo.md(
            f"""
            转换前：`{raw_ex2.tolist()}`（dtype：`{raw_ex2.dtype}`）

            转换后：`{ints_ex2.tolist()}`（dtype：`{ints_ex2.dtype}`）

            注意原数组 raw_ex2 仍是 `{raw_ex2.dtype}`——astype 只交出**新数组**，不动原件。
            预测全对了吗？错在哪一个数上，往往就暴露了对哪条规则的误解。
            """
        )
    ex2_msg
    return (ex2_msg, ints_ex2, raw_ex2)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 2 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                ints_ex2 = raw_ex2.astype(np.int32)
                ```

                结果：`[1.9, -2.9, 3.5, -0.5]` → `[1, -2, 3, 0]`

                逐个对照"向零截断"（砍掉小数，朝 0 的方向靠）：

                - 1.9 → **1**（不是 2：0.9 直接扔掉）
                - -2.9 → **-2**（不是 -3：负数也是朝 0 砍）
                - 3.5 → **3**（.5 没有任何"进位"特权）
                - -0.5 → **0**

                想要真正的四舍五入，可以写 `np.round(raw_ex2).astype(np.int32)`
                （注意 round 对 .5 用"向偶数取"的规则：3.5 → 4，但 2.5 → 2）。

                更重要的习惯：**需要整数语义时，显式写 astype，别依赖"自动转换"**——
                隐式转换静默发生，错了极难查。
                """
            ),
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "🤔 练习 3 · 思考题（不用写代码）": mo.md(
                """
                数字图片的每个像素存的是"亮度"：0 = 纯黑，255 = 纯白，
                中间恰好 **256 级**灰度——不多不少。

                **问：为什么全世界的图片格式几乎都用 uint8 存像素，而不用 float32？**

                （提示一：256 = 2 的 8 次方，uint8 正好 8 个比特，一位不浪费；
                提示二：算一算，一张 1200 万像素的照片，两种类型各占多少内存？）
                """
            ),
            "💡 练习 3 参考答案（先想 3 分钟再看）": mo.md(
                """
                三个理由刚好凑成"天作之合"：

                1. **范围刚好**：亮度天然就是 0~255 这 256 级，uint8 的 8 个比特一位不浪费
                2. **人眼够用**：人眼分辨不出比 256 级更细的灰度——存得更精细是白花钱
                3. **省内存**：uint8 每像素 1 字节，float32 要 4 字节。一张 1200 万像素的照片，
                   uint8 约 12 MB，float32 直接 48 MB——每张图差 4 倍，
                   图库有几万张时，差距是灾难性的

                一句话：**选 dtype = 在"数值范围/精度需求"和"内存开销"之间找平衡。**

                顺带剧透：深度学习的惯例是**存储用 uint8、计算前转 float32**——
                第 31 课加载 MNIST 图片时，你会亲手写下这行转换。
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
                1. 身份证五字段：`shape`（每个方向几个元素，从外往里数）、`dtype`（元素类型）、
                   `ndim = len(shape)`、`size = 各维相乘`、`nbytes = size × itemsize`
                2. 读 shape 的口诀：**从外往里数，一层一个数**——
                   (4, 5, 3) = 4 行 × 每行 5 像素 × 每像素 3 通道
                3. dtype 是"血液类型"：同样的数，int8 和 int64 内存差 8 倍；
                   深度学习偏爱 float32——内存减半、硬件加速、约 7 位有效数字对训练够用
                4. `.astype()` 返回**新数组**；float → int 是**向零截断**
                   （3.7 → 3，-3.7 → -3），不是四舍五入
                5. 整数会**回绕**：uint8 的 255 + 1 = 0，int8 的 127 + 1 = -128，
                   而且数组运算是静默的——选 dtype 前先问"我的数最大会到多少"
                6. 维度直觉：0 维标量（一个数）→ 1 维向量（一行数）→ 2 维矩阵（表格/灰度图）→
                   3 维彩图（高×宽×通道）→ 4 维一批图（批量×高×宽×通道，batch 的由来）
                """
            ),
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ---

        **下一课预告** — `03_创建数组的多种方式`：

        今天我们学会了"读"数组——五个字段，一眼看穿它的身材与身份。
        下一课学"造"数组：`zeros`、`ones`、`arange`、`linspace`……
        各种一键生成的姿势，想造什么形状就造什么形状，
        还会正式揭开 `reshape` 的真面目（今天已经偷偷用过两次了）。

        一句鼓励：**能脱口而出 shape 和 dtype 的人，已经赢下了深度学习里一半的报错。**

        🛠️ 上路前的小仪式：回到上面的交互实验，先闭眼说出当前 shape 里三个数各自管什么，
        再拖动滑块验证自己说得对不对——全对了，就发车去第 03 课！🚀
        """
    )
    return


if __name__ == "__main__":
    app.run()
