import marimo

__generated_with = "0.25.2"
app = marimo.App(app_title="07 · 形状变换", width="medium")


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
        # 07 · 形状变换

        **从零到 Transformer · 手写深度学习** ｜ 第 1 部分 · NumPy 从零开始（7/10）
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📖 本课目标（先读我）": mo.md(
                """
                **这一课只解决一个问题：shape 不合适时，怎么把它变过来？**

                数据在手里是一维的、模型要二维的；图片是 (高, 宽, 通道)、框架要 (通道, 高, 宽)；
                两个向量想广播成一张表……在深度学习里，"变形状"和"算数"一样高频。

                学完你将能：

                - 用 `reshape` 把数组摆成任意形状，并说清 **C 序（行优先）**的填充规则
                - 用 `-1` 让 NumPy 自己算某一维，并一眼读懂"除不尽"的报错
                - 用 `.T` / `transpose` 完成二维翻折与三维换轴（HWC ↔ CHW 不再神秘）
                - 用 `[:, None]` / `[None, :]` 凭空长出维度，一行写出广播加法表
                - 分清 `ravel` 与 `flatten`：谁返回视图、谁返回副本

                本课站在第 02~06 课的肩膀上（shape、视图、广播）；哪块不熟，回去翻一翻再上路。
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "🧩 概念 1/4 · reshape：同一批数字，换个格子放": mo.md(
                """
                手里的数组 shape 不合适（模型要 (批量, 特征)，你手里却是 (特征,) 的散装数据），
                第一反应就是 **reshape**：把同一批数字搬进新格子。

                两条铁律：

                1. **总元素数不变（size 守恒）**：3 × 4 = 12 = 2 × 6 = 12 × 1——
                   新形状各维相乘必须等于原来的元素总数，一个数不能丢、也不能多
                2. **按 C 序（行优先）填格子**：先填满第一行、再填第二行，
                   和读书的顺序一样——数字的巡访次序永远不变，变的只是"每行放几个"

                还有一条要绷紧的弦（第 04 课的老朋友）：**reshape 大多数情况返回视图**——
                它通常不搬运数据，只是换一套下标读同一块内存，**改它会动原数组**。
                （个别情况如"转置之后再 reshape"，内存不连续时会悄悄复印一份；
                但默认按"视图"警惕它，永远不会错。）
                """
            ),
            "🪄 概念 2/4 · -1：让 NumPy 自己算": mo.md(
                """
                某一维懒得心算？写 **-1**，意思是"这一维你自己算"：

                - `a.reshape(3, -1)`：12 个元素、给了 3 行 → 每行必然 4 个 → (3, 4)
                - `a.reshape(-1, 6)`：给了 6 列 → 必然 2 行 → (2, 6)

                两条规矩：

                1. **一次最多一个 -1**：写两个 -1 就有无穷多组解，NumPy 不猜谜
                2. **总数必须除得尽**：12 个元素写 `reshape(5, -1)`，12 ÷ 5 = 2.4，
                   当场报错：`cannot reshape array of size 12 into shape (5,newaxis)`

                第二种错是**新手高频报错**（批大小、通道数对不上时天天见）。
                要学会读它：数字全写在脸上——数一数元素总数、检查能否被另一维整除，
                答案就在报错原文里（演示 2 拆解给你看）。
                """
            ),
            "🔄 概念 3/4 · transpose：行变列": mo.md(
                """
                **二维转置 = 沿对角线翻折。** 第 i 行第 j 列的数，翻到第 j 行第 i 列：

                ```
                m = [[0, 1, 2],        m.T = [[0, 3],
                     [3, 4, 5]]               [1, 4],
                                             [2, 5]]
                ```

                shape (2, 3) → (3, 2)。写法两种：`m.T`（属性，最顺手）或 `m.transpose()`。

                **三维及以上**要靠 `transpose(axes)` 参数指定新顺序，比如
                `img.transpose(2, 0, 1)` 的意思是——**新数组的第 0 维，用旧的第 2 维**
                （新的第 k 维 = 旧的第 axes[k] 维，其余依此类推）。最经典的场景是图片格式：

                - **HWC**：(高, 宽, 通道)——TensorFlow 一派的默认
                - **CHW**：(通道, 高, 宽)——PyTorch 一派的默认

                深度学习两大阵营各用一种，模型对接时**换轴全靠 transpose**。

                最后：`.T` 和 `transpose` 都返回**视图**——不搬运数据，只换下标读法，
                和 reshape 同款脾气（第 04 课的结论继续适用）。
                """
            ),
            "🆕 概念 4/4 · newaxis：凭空长出一个维度": mo.md(
                """
                有时不需要重排数字，只需要**多出一个维度**来配合广播——这就是 **newaxis**：
                在索引里写 `None`，那个位置就凭空长出一个长度为 1 的维度。
                **shape 变了，数据一个没动。**

                - `x[:, None]`：(3,) → **(3, 1)**——None 在第 1 个位置，第 1 维长出 1（竖起来）
                - `x[None, :]`：(3,) → **(1, 3)**——None 在第 0 个位置，第 0 维长出 1（横过去）

                口诀：**None 在哪个位置，哪个位置长出 1。**

                它是上一课广播的最佳搭档：想用 (4,1) + (1,4) 生成一张加法表？
                用 newaxis 写就是 `x[:, None] + y[None, :]`——
                比 `x.reshape(4, 1) + y.reshape(1, 4)` 更短，还不用数元素个数。

                （`np.newaxis` 是 `None` 的正式全名，两者完全等价，平时写 None 就行。）
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

        六组演示，把"整形三件套"挨个上手：

        1. **reshape 与 C 序**：亲眼看 12 个数怎么流进新格子
        2. **-1 自动补维**：让 NumPy 自己算，学会读"除不尽"的报错
        3. **视图验证**：reshape 改的到底是谁的内存
        4. **transpose**：二维沿对角线翻折 + 三维 axes 换轴
        5. **newaxis 四连**：`[:, None]` 摆姿势，一行写出加法表
        6. **ravel vs flatten**：摊平的一对孪生兄弟，一个视图一个副本

        每个单元都可以直接运行；**大胆改里面的数字**，坏了点撤销就行。
        """
    )
    return


@app.cell
def _(mo, np):
    # ===== 演示 1 · reshape 与 C 序：12 个数，换个格子放 =====

    # 本课的主角 a：0~11 躺成一排的一维数组
    a = np.arange(12)

    # 四种 reshape：数字一个没换，只是格子换了
    _r34 = a.reshape(3, 4)      # 3 行 4 列（最常见）
    _r26 = a.reshape(2, 6)      # 2 行 6 列（更扁长）
    _r12_1 = a.reshape(12, 1)   # 12 行 1 列（一根竖柱）
    _r1_12 = a.reshape(1, 12)   # 1 行 12 列（一根横条）

    # 渲染器：把二维数组画成文字网格（行内空格分隔、行间换行、个位数对齐）
    def _as_grid(_arr):
        return "\n".join(" ".join(f"{_x:2d}" for _x in _row) for _row in _arr)

    mo.vstack(
        [
            mo.md(
                f"""
                ### 演示 1 · reshape 与 C 序：同一批数字，换个格子放

                `a = np.arange(12)`——0~11 本来躺成一排。reshape **不动任何数字**，
                只是把它们按固定顺序重新流进新格子：

                | 写法 | shape | 形象说法 |
                |---|---|---|
                | `a.reshape(3, 4)` | {_r34.shape} | 3 行 4 列的表格 |
                | `a.reshape(2, 6)` | {_r26.shape} | 更扁长的表格 |
                | `a.reshape(12, 1)` | {_r12_1.shape} | 一根竖柱 |
                | `a.reshape(1, 12)` | {_r1_12.shape} | 一根横条 |

                关键是看数字**怎么流**：NumPy 按 **C 序（行优先）**填格子——
                先填满第一行、再填第二行，和读书的顺序一模一样：
                """
            ),
            mo.hstack(
                [
                    mo.md(f"**`a.reshape(3, 4)`**\n\n```\n{_as_grid(_r34)}\n```"),
                    mo.md(f"**`a.reshape(2, 6)`**\n\n```\n{_as_grid(_r26)}\n```"),
                ]
            ),
            mo.hstack(
                [
                    mo.md(f"**`a.reshape(12, 1)`** · 竖柱\n\n```\n{_as_grid(_r12_1)}\n```"),
                    mo.md(f"**`a.reshape(1, 12)`** · 横条\n\n```\n{_as_grid(_r1_12)}\n```"),
                ]
            ),
            mo.md(
                f"""
                盯着数字看：**0 永远第一个进格子**（左上角），**11 永远最后一个**（右下角）。
                四种摆法的巡访顺序完全相同，不同的只是"每行放几个"。

                **size 是守恒的**：`a.size` = {a.size}，四种新形状的 `.size` 也都是
                {_r34.size}——因为 3 × 4 = 2 × 6 = 12 × 1 = 1 × 12 = **12**。
                这就是 reshape 的铁律：**新形状各维相乘，必须等于元素总数**。
                """
            ),
        ]
    )
    return (a,)


@app.cell
def _(a, mo):
    # ===== 演示 2 · -1：让 NumPy 自己算，以及"除不尽"的报错怎么读 =====

    # -1 的含义："这一维我不说，你自己算"
    _r3 = a.reshape(3, -1)   # 定了 3 行：12 ÷ 3 = 4 → 每行 4 个
    _c6 = a.reshape(-1, 6)   # 定了 6 列：12 ÷ 6 = 2 → 共 2 行

    # 故意踩坑：12 个数想摆 5 行，12 ÷ 5 = 2.4 除不尽 → 报错
    # 用 try/except 把报错原文抓下来（不抓的话整个单元会变红）
    try:
        a.reshape(5, -1)
        _err_text = "（咦，居然没报错？）"
    except ValueError as _e:
        _err_text = str(_e)

    mo.vstack(
        [
            mo.md(
                f"""
                ### 演示 2 · -1：让 NumPy 自己算

                总共 12 个数，只要另一维定了，剩下那维**不用你心算**，写 -1 让 NumPy 算：

                | 写法 | 人话翻译 | NumPy 的算式 | 结果 shape |
                |---|---|---|---|
                | `a.reshape(3, -1)` | "要 3 行，每行几个你算" | 12 ÷ 3 = 4 | {_r3.shape} |
                | `a.reshape(-1, 6)` | "每行 6 个，几行你算" | 12 ÷ 6 = 2 | {_c6.shape} |

                规矩：**一次最多一个 -1**——写两个 -1 就有无穷多组解，NumPy 不猜谜。
                """
            ),
            mo.callout(
                mo.md(
                    f"""
                    **新手高频报错：除不尽。** 下面是 `a.reshape(5, -1)` 的真实报错原文：

                    ```
                    ValueError: {_err_text}
                    ```

                    逐段读它：

                    - **array of size 12** —— 你手里有 12 个元素
                    - **into shape (5, newaxis)** —— 你却要 5 行（-1 被还原成了 newaxis）
                    - 12 ÷ 5 = 2.4，**没有任何整数列数能让 5 × ? = 12**

                    以后看到这个报错别慌——它是最友好的一类报错，数字全写在脸上：
                    **数一数元素总数，检查它能不能被另一维整除**，问题在哪一眼便知。
                    """
                ),
                kind="warn",
            ),
        ]
    )
    return


@app.cell
def _(a, mo):
    # ===== 演示 3 · reshape 是视图：改"新"数组，动的是旧内存 =====

    # 第 1 步：reshape 出一个"看起来全新"的数组 _b
    # 关键认知：大多数情况下这一步【没有搬运任何数据】——_b 只是换了副眼镜看同一块内存
    _b = a.reshape(3, 4)

    # 第 2 步：只改 _b 的第一个元素
    _b[0, 0] = 99

    # 第 3 步：回头看 a 的第 0 个元素——也变成 99 了！我们没直接碰过 a
    _a0_after = a[0]

    # 第 4 步：一行鉴定（第 04 课的老朋友 .base）：视图的背后是原数组
    _is_view = _b.base is a

    # 第 5 步：修复 a，后面的演示和练习还要用干净的它
    a[0] = 0
    _a0_fixed = a[0]

    mo.vstack(
        [
            mo.md(
                f"""
                ### 演示 3 · reshape 是视图：改"新"数组 = 改原数组

                ```
                _b = a.reshape(3, 4)   # 看起来"新建"了一个数组
                _b[0, 0] = 99          # 只改了 _b，没碰过 a
                ```

                然后看 `a[0]`：变成了 **{_a0_after}**。

                鉴定（`.base` 大法）：`_b.base is a` → **{_is_view}**——
                reshape 和切片一样，默认返回**视图**：不搬运数据，只是换一套下标读同一块内存。
                第 04 课的结论原样适用：**改它会动原数组；想要独立的副本，得显式 `.copy()`**。
                """
            ),
            mo.callout(
                mo.md(
                    f"""
                    为什么这么设计？还是那个理由：**省内存省时间**——
                    reshape 一块 1 GB 的数据如果每次都复印一份，程序就不用跑了。

                    （a 的第 0 个元素已修复回 **{_a0_fixed}**，后面的单元不受影响。）
                    """
                ),
                kind="info",
            ),
        ]
    )
    return


@app.cell
def _(mo, np):
    # ===== 演示 4 · transpose：二维行变列，三维换轴 =====

    # 二维主角 m：2 行 3 列
    m = np.arange(6).reshape(2, 3)
    mt = m.T   # 转置：沿对角线翻折，(2, 3) → (3, 2)

    # 渲染器：数组 → 文字网格
    def _as_grid(_arr):
        return "\n".join(" ".join(f"{_x:2d}" for _x in _row) for _row in _arr)

    # 挑三组位置，验证"第 i 行第 j 列 ↔ 第 j 行第 i 列"
    # 多行表格行插进 f-string 前，续行要补 12 格缩进对齐这层 md（不然公共缩进出栈会散架）
    _pair_rows = ("\n" + " " * 12).join(
        f"| ({_i}, {_j}) | **{m[_i, _j]}** | ({_j}, {_i}) | **{mt[_j, _i]}** |"
        for _i, _j in ((0, 1), (1, 2), (1, 0))
    )

    # 三维主角 img：2 批 × 每批 3 行 × 每行 4 列
    img = np.arange(24).reshape(2, 3, 4)
    img_t = img.transpose(2, 0, 1)   # axes=(2, 0, 1)：新数组的第 0 维用旧的第 2 维（其余类推）

    mo.vstack(
        [
            mo.md(
                """
                ### 演示 4 · transpose：行变列

                `m = np.arange(6).reshape(2, 3)` 与它的转置 `m.T` 并排看：
                """
            ),
            mo.hstack(
                [
                    mo.md(f"**m** · shape {m.shape}\n\n```\n{_as_grid(m)}\n```"),
                    mo.md(f"**m.T** · shape {mt.shape}\n\n```\n{_as_grid(mt)}\n```"),
                ]
            ),
            mo.md(
                f"""
                转置 = **沿对角线翻折**：每个数字的门牌号行列互换——
                **第 i 行第 j 列 ↔ 第 j 行第 i 列**：

                | 在 m 的位置 | 值 | 搬到 m.T 的位置 | 值 |
                |---|---|---|---|
                {_pair_rows}

                值一个没变，变的是"门牌号"。`.T` 是 `transpose()` 的简写，
                两者都是**视图**（不搬运数据，同 reshape 的脾气）。
                """
            ),
            mo.md(
                f"""
                **三维：用 axes 指定新顺序**

                ```
                img = np.arange(24).reshape(2, 3, 4)   # 2 批 × 每批 3 行 × 每行 4 列
                img_t = img.transpose(2, 0, 1)
                ```

                | | shape | 各维怎么读 |
                |---|---|---|
                | `img` | {img.shape} | 2 批 × 每批 3 行 × 每行 4 个 |
                | `img.transpose(2, 0, 1)` | {img_t.shape} | 新第 0 维 = 旧第 2 维（4 个）、新第 1 维 = 旧第 0 维（2 批）、新第 2 维 = 旧第 1 维（3 行） |

                一句话读法：**axes=(2, 0, 1) 意思是新数组的第 0 维用旧的第 2 维**，后面依此类推。
                深度学习最经典的换轴场景：图片 **HWC (高, 宽, 通道) ↔ CHW (通道, 高, 宽)**——
                TensorFlow 和 PyTorch 两大阵营各用一种，对接时换轴全靠 transpose。
                """
            ),
        ]
    )
    return (img, img_t, m, mt)


@app.cell
def _(mo, np):
    # ===== 演示 5 · newaxis 四连：shape 变了，数据一个没动 =====

    v = np.arange(4)   # 一维：(4,)

    # 同一个 v，三种"拍照姿势"——只是 shape 在变，数据一个没动
    _v_col = v[:, None]   # None 站在第 1 个位置（列的位置）→ (4, 1)：竖起来
    _v_row = v[None, :]   # None 站在第 0 个位置（行的位置）→ (1, 4)：横过去

    # newaxis 的主场：和上一课的广播搭档，一行写出 4×4 加法表
    add_table = v[:, None] + v[None, :]   # (4, 1) + (1, 4) → 广播咬合成 (4, 4)

    # np.newaxis 就是 None 的"正式全名"，两者是同一个东西
    _same = np.newaxis is None

    # 渲染器：数组 → 文字网格
    def _as_grid(_arr):
        return "\n".join(" ".join(f"{_x:2d}" for _x in _row) for _row in _arr)

    # 多行网格上屏前先给续行补 12 格缩进，对齐下面 md 的行
    _table_grid = _as_grid(add_table).replace("\n", "\n" + " " * 12)

    mo.vstack(
        [
            mo.md(
                f"""
                ### 演示 5 · newaxis：凭空长出一个维度

                | 写法 | shape | 直觉 |
                |---|---|---|
                | `v` | {v.shape} | 躺平的一排 |
                | `v[:, None]` | {_v_col.shape} | 竖起来（列向量） |
                | `v[None, :]` | {_v_row.shape} | 横过去（行向量） |
                | `v[:, None] + v[None, :]` | {add_table.shape} | 广播加法表（下面见） |

                口诀：**None 在哪个位置，哪个位置长出 1**——
                `[:, None]` 的 None 在第 1 个位置，第 1 维变成 1；
                `[None, :]` 的 None 在第 0 个位置，第 0 维变成 1。
                数据一个没动，动的只是 shape（它也是视图）。
                """
            ),
            mo.md(
                f"""
                加法表 `add_table = v[:, None] + v[None, :]`（第 i 行第 j 列 = i + j）：

                ```
                {_table_grid}
                ```

                上一课（第 06 课）演示 3 里，同款加法表（3×4 版）是用
                `np.arange(3).reshape(3, 1) + np.arange(4).reshape(1, 4)` 写的——
                换成 `v[:, None] + v[None, :]` **更短也更稳**：不用数元素个数，长短自动对上。

                顺带一提：`np.newaxis is None` → **{_same}**——
                `np.newaxis` 就是 None 的正式全名，平时写 None 就够了。
                """
            ),
        ]
    )
    return (add_table, v)


@app.cell
def _(mo, np):
    # ===== 演示 6 · ravel vs flatten：摊平的一对孪生兄弟 =====

    # 小设计：m2 用字面量 np.array 直接创建，让它自己拥有内存（base 为空）——
    # 这样下面的 .base 鉴定才会指向 m2 本尊，结果干净利落
    m2 = np.array([[0, 1, 2], [3, 4, 5]])   # 2 行 3 列，值仍是 0~5

    # 三个都把 (2, 3) 摊回一维，排布全是行优先的 0~5
    _flat_r = m2.ravel()          # ravel：视图（不复印）
    _flat_f = m2.flatten()        # flatten：副本（真的复印一份）
    _flat_alias = m2.reshape(-1)  # reshape(-1)：ravel 的别名写法（也是视图）

    # .base 鉴定：视图背后是 m2，副本背后空无一物
    _ravel_view = _flat_r.base is m2       # True
    _flat_view = _flat_f.base is m2        # False
    _alias_view = _flat_alias.base is m2   # True

    # 亲眼验证 ravel 是视图：只改 _flat_r，看 m2 动没动
    _flat_r[0] = 99
    _m2_00_after = m2[0, 0]   # 99 —— m2 跟着变了！
    _flat_r[0] = 0            # 修复

    mo.md(
        f"""
        ### 演示 6 · ravel vs flatten：摊平的孪生兄弟

        `m2 = np.array([[0, 1, 2], [3, 4, 5]])`——2 行 3 列（这次用字面量创建，
        让 m2 自己拥有内存，`.base` 鉴定才指向它本尊）。三种摊平写法：

        | 写法 | 摊平结果 | 是视图吗（`.base is m2`） | 一句话定位 |
        |---|---|---|---|
        | `m2.ravel()` | `{_flat_r}` | {_ravel_view} | 快速看一眼 |
        | `m2.flatten()` | `{_flat_f}` | {_flat_view} | 要安全修改 |
        | `m2.reshape(-1)` | `{_flat_alias}` | {_alias_view} | ravel 的别名 |

        结果一模一样，差别全在**是不是视图**：

        刚才只执行了 `_flat_r[0] = 99`——`m2[0, 0]` 就变成了 **{_m2_00_after}**
        （随后已修复回 0）。ravel 没有搬运数据，和 m2 读的是同一块内存；
        flatten 是真复印了一份新内存，怎么改都不惊动 m2。

        **怎么选：快速看一眼用 ravel，要安全修改用 flatten。**
        深度学习里 flatten 出场率更高——卷积特征图接全连接层之前先 flatten，
        是神经网络的老传统（第 5 部分会亲眼见）。
        """
    )
    return (m2,)


@app.cell
def _(mo):
    # reshape 游乐场的两个部件：
    # ① 下拉框：12 的全部因数组合（1×12、2×6、3×4、4×3、6×2、12×1）
    shape_dropdown = mo.ui.dropdown(
        options=["(1, 12)", "(2, 6)", "(3, 4)", "(4, 3)", "(6, 2)", "(12, 1)"],
        value="(3, 4)",
        label="🎛️ 选一个形状（12 的因数组合）",
    )

    # ② 开关：要不要把结果转置一下
    transpose_switch = mo.ui.switch(label="转置结果（.T）", value=False)

    mo.vstack([shape_dropdown, transpose_switch])
    return (shape_dropdown, transpose_switch)


@app.cell
def _(mo, np, shape_dropdown, transpose_switch):
    # ===== 交互实验的响应单元：reshape 游乐场 =====

    # 读取下拉框选中的形状字符串（如 "(3, 4)"）；万一被清空就兜底回默认
    _shape_str = shape_dropdown.value or "(3, 4)"

    # 解析出两个整数：剥掉括号、按逗号切开
    _r, _c = [int(_s) for _s in _shape_str.strip("()").split(",")]

    # 同样的 12 个数，流进所选的格子
    _arr = np.arange(12).reshape(_r, _c)

    # 开关打开就再转置一次（行变列）
    if transpose_switch.value:
        _arr = _arr.T

    # 数组 → 文字网格（行内空格分隔、行间换行、个位数对齐）
    _grid = "\n".join(" ".join(f"{_x:2d}" for _x in _row) for _row in _arr)

    # 开关状态的中文
    _sw_txt = "开 ✅" if transpose_switch.value else "关"

    mo.vstack(
        [
            mo.md(
                f"""
                ## 🎛️ 交互实验 · reshape 游乐场

                **同样的 12 个数，行优先地流进不同格子。**
                换形状、开转置，盯着任何一个数字看它的"搬家路线"。
                """
            ),
            mo.hstack(
                [
                    mo.md(
                        f"""
                        ```
                        {_grid}
                        ```
                        """
                    ),
                    mo.md(
                        f"""
                        - 下拉框选中：**{_shape_str}**
                        - 转置开关：**{_sw_txt}**
                        - 实际 shape：**{_arr.shape}**
                        - size：**{_arr.size}**（永远守恒）
                        - 每行 **{_arr.shape[1]}** 个 ｜ 共 **{_arr.shape[0]}** 行
                        """
                    ),
                ]
            ),
            mo.md(
                """
                玩法建议：

                1. 换不同形状：`0` 永远第一个进格子（左上角）、`11` 永远最后一个（右下角）——C 序雷打不动
                2. 选 `(3, 4)` 并打开转置 → 变 `(4, 3)`：每个数从"第 i 行第 j 列"搬去"第 j 行第 i 列"
                3. 选 `(12, 1)`（竖柱）并打开转置 → 变 `(1, 12)`（横条）——两根"棍子"互相变
                """
            ),
        ]
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

                - **练习 1 · 变形二连**：0~11 先变 4 行 3 列、再转置成 3 行 4 列
                - **练习 2 · 还原小图**：把 18 个数还原成两幅 3×3 小图，并预测其中一个元素
                - **练习 3 · 思考题**：不许用 reshape，只用 newaxis 造出 5×5 加法表
                """
            )
        }
    )
    return


@app.cell
def _(np):
    # 练习 1：把 0~11 的一维数组，先变成 4 行 3 列，再转置成 3 行 4 列
    #
    # 提示（两步拼图，可以链式串成一行）：
    #   ① reshape(4, 3)：行数写在前——4 行、每行 3 个
    #   ② 再 .T：行变列 → 3 行、每行 4 个
    #
    # 期望结果：shape (3, 4)，且 result_ex1[0] 是 [0 3 6 9]

    arr_ex1 = np.arange(12)

    result_ex1 = None   # ← 把 None 换成你的实现（一行就够）

    result_ex1 if result_ex1 is not None else "（还是 None——提示：reshape 完直接接 .T）"
    return (arr_ex1, result_ex1)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 1 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                result_ex1 = arr_ex1.reshape(4, 3).T
                ```

                分解动作：

                - `reshape(4, 3)`：12 个数按 C 序流进 4 行 3 列
                - `.T`：沿对角线翻折，(4, 3) → (3, 4)
                - 结果第 0 行是 `[0 3 6 9]`——原数组每一列的"列头"排成了一行

                小心方向：`reshape(3, 4).T` 得到的是 **(4, 3)**——和 `reshape(4, 3)` 形状一样，
                **内容却完全不同**：行优先的填充次序不同，数字落位两样（不妨亲手验证）。
                记住两句话：**reshape 行数写在前；转置行列互换**，两步各自管各自的。
                """
            )
        }
    )
    return


@app.cell
def _(mo, np):
    # 练习 2：img_flat 是 18 个数，把它们还原成 (2, 3, 3)——"两幅 3×3 小图"
    #
    # ✍️ 运行之前先写预测（第 2 幅图的第 1 行、第 3 列）：
    #    我的预测：img_ex2[1, 0, 2] = ________
    #
    # 提示：18 = 2 × 3 × 3，一个 reshape 的事；
    #       行优先意味着前 9 个数流进第 1 幅、后 9 个流进第 2 幅

    img_flat = np.arange(18)

    img_ex2 = None   # ← 把 None 换成你的 reshape

    if img_ex2 is None:
        _msg_ex2 = "（还是 None——把上面的 None 换成你的 reshape，再运行一次）"
    else:
        _grid1 = "\n".join(" ".join(f"{_x:2d}" for _x in _row) for _row in img_ex2[1])
        _msg_ex2 = mo.md(
            f"""
            shape = **{img_ex2.shape}**。第 2 幅 `img_ex2[1]` 长这样：

            ```
            {_grid1}
            ```

            **img_ex2[1, 0, 2] = {img_ex2[1, 0, 2]}** —— 和你注释里的预测对上了吗？
            """
        )
    _msg_ex2
    return (img_ex2, img_flat)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 2 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                img_ex2 = img_flat.reshape(2, 3, 3)
                ```

                - `reshape(2, 3, 3)`：18 = 2 × 3 × 3，size 守恒
                - 行优先填充：**前 9 个数流进第 1 幅**（0~8），**后 9 个流进第 2 幅**（9~17）
                - 第 1 幅 `img_ex2[0]`：`[[0, 1, 2], [3, 4, 5], [6, 7, 8]]`
                - 第 2 幅 `img_ex2[1]`：`[[9, 10, 11], [12, 13, 14], [15, 16, 17]]`

                **img_ex2[1, 0, 2] = 11**（第 2 幅、第 1 行、第 3 列）。

                不看数组也能算出来——三维的行优先有个扁平位置公式：
                **每一维下标 × 它右边所有维度的乘积，再全部相加**：

                ```
                [1, 0, 2] → 1×(3×3) + 0×(3) + 2×(1) = 9 + 0 + 2 = 11
                ```

                （如果你预测的是 15：那是 `[1, 2, 0]` 的位置——第 2 幅、**第 3 行**、第 1 列。
                下标顺序 (批, 行, 列) 一晕就会算错，reshape 完先默念一遍每一维是谁。）
                """
            )
        }
    )
    return


@app.cell
def _(np):
    # 练习 3 · 思考题：不许用 reshape！
    #
    # x_ex3 是 0~4。只允许用 newaxis（[None, :] 或 [:, None]）和加法，
    # 让 x_ex3 和它自己相加，得到 5×5 加法表：table_ex3[i, j] == i + j
    #
    # 提示：想想演示 5——谁竖着、谁横着，广播才能咬合成 (5, 5)？
    # 期待结果：shape (5, 5)，对角线是 0, 2, 4, 6, 8

    x_ex3 = np.arange(5)

    table_ex3 = None   # ← 把 None 换成你的实现（不许出现 reshape！）

    table_ex3 if table_ex3 is not None else "（还是 None——提示：一个 [:, None] 加一个 [None, :]）"
    return (table_ex3, x_ex3)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 3 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                table_ex3 = x_ex3[:, None] + x_ex3[None, :]
                ```

                两个 None 各自在哪个位置长的维度：

                - `x_ex3[:, None]`——None 站在**第 1 个位置（列的位置）**，
                  于是第 1 维长出 1：(5,) → **(5, 1)**，数字竖着站
                - `x_ex3[None, :]`——None 站在**第 0 个位置（行的位置）**，
                  于是第 0 维长出 1：(5,) → **(1, 5)**，数字横着躺

                然后广播接力（上一课的主场）：**(5, 1) + (1, 5)** →
                两边互相朝对方的形状扩张 → **(5, 5)**，
                第 i 行第 j 列 = 竖着的第 i 个 + 横着的第 j 个 = **i + j**。

                口诀复习：**None 在哪个位置，哪个位置长出 1。**
                newaxis 不搬运任何数据（它也是视图），是"为了广播而摆姿势"的最短写法——
                后面写注意力机制时，它会天天出场。
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
                1. **reshape 换格子不换数字**：新形状各维相乘必须等于 size（3×4 = 2×6 = 12×1 = 12）；填充按 **C 序（行优先）**——先填满一行再填下一行，巡访次序永远不变
                2. **-1 = "这一维你自己算"**：一次最多一个；**除不尽当场报错**（`cannot reshape array of size 12 into shape (5,newaxis)`），见到它就去检查整除
                3. **reshape 默认返回视图**：改"新"数组会动原数组；`.base` 一行鉴定（第 04 课的老朋友），要独立副本就显式 `.copy()`
                4. **transpose 行变列**：第 i 行第 j 列 ↔ 第 j 行第 i 列；三维用 `transpose(axes)`——**新数组第 k 维 = 旧数组第 axes[k] 维**；图片 HWC ↔ CHW 全靠它；它也是视图
                5. **newaxis 凭空长维度**：`x[:, None]` → (n, 1)、`x[None, :]` → (1, n)，数据一个没动；口诀"**None 在哪个位置，哪个位置长出 1**"；广播的最佳搭档
                6. **ravel / flatten / reshape(-1)**：都摊平成一维；ravel 与 reshape(-1) 是视图、flatten 是副本——快速看一眼用 ravel，要安全修改用 flatten
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

        **下一课预告** — `08_axis的真义`：

        reshape、transpose 玩到这一课，你其实一直在和**维度**打交道：
        `.sum(axis=0)` 把行消掉、`.sum(axis=1)` 把列消掉……axis 到底是什么？
        为什么一会儿是行、一会儿是列，还能是负数？

        下一课把 axis 讲透——**它是读懂所有 NumPy 聚合运算的钥匙，
        也是新手最大的困惑源**。搞懂它，sum / mean / max 从此不再靠猜。

        🧭 三把整形刀已到手，去第 08 课解剖维度！
        """
    )
    return


if __name__ == "__main__":
    app.run()
