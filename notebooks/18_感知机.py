import marimo

__generated_with = "0.25.2"
app = marimo.App(app_title="18 · 感知机", width="medium")


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
    # matplotlib 默认字体不含汉字，中文会显示成方框 □□□
    # 修法：给 font.sans-serif 一张"候选字体名单"，从左到右挑第一个装了的：
    #   PingFang SC / Microsoft YaHei / SimHei    → macOS / Windows 自带
    #   Noto Sans SC / WenQuanYi Zen Hei          → Linux 常见"思源黑体 / 文泉驿"
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Microsoft YaHei", "SimHei",
        "Noto Sans SC", "WenQuanYi Zen Hei",
    ]
    # 中文字体常缺"数学负号"字形，不设这行负数会变方框
    plt.rcParams["axes.unicode_minus"] = False
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # 18 · 感知机：一个神经元的诞生

        **从零到 Transformer · 手写深度学习** ｜ 第 4 部分 · 神经网络（1/8）
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📖 本课目标（先读我）": mo.md(
                """
                **这一课只解决一个问题：一个"神经元"长什么样？它怎么自己学会分类？**

                第 17 课结尾的预告还热着：热身结束，**主角登场**。第 4 部分要把"一层"
                摞成"多层"、把直线掰弯成任意曲线——而这一切的起点，是 1958 年发明的一个
                小零件：**感知机**（perceptron），史上第一个人工神经元。

                打法依然稳扎稳打：先把神经元拆成零件看清楚（加权投票 + 门槛 + 开关），
                再亲手做一次"纠正错点"的手术，然后放它自己学——看它 8 次更新就学会了
                一整张卷子；最后领它去撞一堵著名的墙（XOR），撞出"为什么要多层"的答案。

                学完你将能：

                - 画出**感知机**的三件零件：权重 w（每票的分量）、偏置 b（门槛）、
                  sign（开关），并说出它和 16 课逻辑回归的异同
                - 写出**感知机学习规则**：`w ← w + lr·y·x`——错一个点，修一次参数，
                  对了就一动不动
                - 亲手训练感知机，看它**几轮收敛**，也能看懂它"修好一个、碰歪另一个"的
                  小脾气
                - 讲出 **XOR 之墙**的故事：为什么一条直线永远分不开它，
                  以及这堵墙如何逼出了后来的多层网络

                前置：第 16 课的决策边界（z = 0 的那条线）、第 12 课的矩阵几何
                （w 是边界的法线方向）、第 15 课的训练循环直觉（本课的循环和四步曲
                神似，但第 ③ 步不是梯度——一个有趣的例外）。

                一个诚实的提醒：感知机**不是**梯度下降，它是一套更古老的"纠错"规则；
                但它训练循环的骨架、决策边界的几何，和你已经学的一切严丝合缝。
                摞多层、装激活函数、上梯度——那是 19、20、22 课的事，先把第一个
                神经元捏明白。
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "🧠 概念 1/4 · 从大脑到数学：一个神经元的零件表": mo.md(
                """
                你大脑里有大约 860 亿个神经元。每个干的事出奇地简单：

                - **接收**：从其他神经元收一堆电信号，每个信号乘上突触的强度再相加
                - **判断**：总刺激超过某个**阈值**，神经元就放电；否则保持安静

                1943 年，McCulloch 和 Pitts 把这件事翻译成数学；1958 年，
                Rosenblatt 造出了能**自动学习权重**的版本——感知机。
                零件只有三件，一张表看完：

                | 生物原型 | 数学零件 | 符号 |
                |---|---|---|
                | 突触强度（这条线多重要） | 权重 | w |
                | 放电阈值（攒多少才动） | 偏置（门槛） | b |
                | 放电 / 安静 | 开关函数 | sign |

                拼起来就是一个公式：

                ```
                z = w·x + b          （加权投票：每个特征投一票，w 是票的分量，b 是门槛）
                ŷ = sign(z)          （开关：z > 0 输出 +1，否则 -1）
                ```

                眼熟吗？**前半句就是 16 课的打分公式** `z = w·x + b`——只换了包装。
                16 课把 z 喂给 sigmoid 得到概率，讲究；感知机更糙：直接 sign 砍成
                正负两半，**只要判决，不要概率**。

                | | 逻辑回归（16 课） | 感知机（本课） |
                |---|---|---|
                | 打分 | z = w·x + b | z = w·x + b（同一个） |
                | 输出 | σ(z) ∈ (0, 1)，概率 | sign(z) ∈ {+1, -1}，硬判决 |
                | 边界 | z = 0 那条直线 | z = 0 那条直线（同一条） |
                | 学习 | 梯度下降 + 交叉熵 | 纠错规则（概念 3） |

                同一条直线、两种性格——这个对照先记下，演示 5 会让他们同台比一场。
                """
            ),
            "📏 概念 2/4 · 几何视角：还是那条直线": mo.md(
                """
                把 z = w·x + b = 0 画在平面上，还是 16 课那条**决策边界**。
                12 课的几何在此回收（这已是它第三次登场）：

                - **w 指向正类的领地**：法线方向垂直于边界，顺着 w 走，z 越来越大
                - **b 负责平移**：b = 0 时边界过原点；b 越大，边界整体往负方向挪
                - **‖w‖ 是刻度**：w 拉长一倍，z 也翻倍——但 sign(z) 不变，
                  所以**边界的位置只由 w 的方向和 b/‖w‖ 决定**

                本课把 16 课的 0/1 标签换成对称的 **+1 / -1**（感知机的传统记法）。
                好处立竿见影——"判没判对"一行搞定：

                ```
                y · z > 0    对了（标签和打分同号）
                y · z ≤ 0    错了（异号，或压线 z = 0——我们约定压线算错）
                ```

                乘法比 if-else 诚实：对，就是同号；错，就是异号。这个 y·z 后面叫
                **间隔打分**——它不光说对错，还说"对得多理直气壮"（|y·z| 越大，
                离边界越远越有底）。演示 5 会用它量出感知机的小毛病。
                """
            ),
            "🔁 概念 3/4 · 学习规则：错了才动": mo.md(
                """
                感知机怎么学？一句话：**扫一遍数据，谁错修谁，全对收工**。

                ```
                对每个样本 (x, y)：
                    若 y · (w·x + b) ≤ 0（错了）：
                        w ← w + lr · y · x        （把 w 往 x 的方向拽一把）
                        b ← b + lr · y            （门槛也跟着挪）
                重复扫描，直到一整轮一个错都没有
                ```

                为什么"往 x 方向拽"是对的？分两种情况看：

                - **正类（y = +1）被判错**（z 太小）：w += x 之后，再遇到这个 x，
                  新打分 = (w + x)·x + (b + 1) = 旧 z + (x·x + 1)——**变大了一截**，
                  朝对的方向修正
                - **负类（y = -1）被判错**（z 太大）：w -= x，新打分 = 旧 z - (x·x + 1)，
                  **变小一截**——同样朝对的方向修正

                两种情况一个公式全包，这就是 y 乘在里面的妙处。
                而判对的样本呢？**一动不动**——感知机的损失天生为零，
                "没错就别折腾"是它和逻辑回归最大的性格差异（16 课的逻辑回归
                对每个点都永远在微调）。

                一张历史保证书：**Novikoff 收敛定理（1962）**——只要数据线性可分，
                感知机最多 (R/γ)² 次更新必收敛（R 是数据的半径，γ 是最优边界的间隔；
                  两个都和 lr 无关）。反过来，**数据若不可分，它永远不收敛**——
                错点数会永远震荡。这不是 bug，是一台灵敏的"可分性检测器"：
                演示 4 就靠这个性质给 XOR 定罪。

                一个彩蛋式的冷知识：lr 在感知机里几乎是个摆设——把 w 和 b 一起等比例
                放大，边界一毫米都不动，所以 lr 只改变"步子的大小"，不改变
                "往哪修"。真正的梯度下降（15 课）里 lr 是生死攸关的超参数，
                这里它只是个音量旋钮。
                """
            ),
            "🚧 概念 4/4 · XOR：1969 年的那堵墙": mo.md(
                """
                1969 年，Minsky 和 Papert 出版了《Perceptrons》一书，用数学证明了一件
                让全行业心凉的事：**单层感知机连 XOR（异或）都学不会**。

                XOR 的四类点摆出来是这样（+1 和 -1 各占对角）：

                ```
                (0,0) → -1      (1,0) → +1
                (0,1) → +1      (1,1) → -1
                ```

                画在纸上：负类在左下、右上两个角，正类在另外两个角。想用**一条**直线
                把两个 +1 和两个 -1 分开？做不到——你需要**两条**线围出一个角落。
                而单层感知机手里只有一条。

                这件事的破坏力不在 XOR 本身，而在它是一整类问题的代表：任何
                "非线性可分"的数据，感知机都无能为力。书出版后经费退潮，
                史称第一次 AI 寒冬——直到近 20 年后，**多层网络 + 反向传播**重新点火，
                才把墙变成门。

                出路其实就写在墙上：

                1. **摞多层**：先用两个神经元各画一条线（比如 x1 > 0.5 一条、
                   x1 + x2 < 1.5 一条），再用第三个神经元把两条线的结论**与**起来
                   ——两条线围出那个角。这是 20 课的戏份
                2. **弯折的激活函数**：摞起来的层之间得有非线性"关节"，
                   否则摞一百层还是一条直线。这是 19 课的戏份

                本课演示 4 会亲手把感知机按在 XOR 上摩擦 30 轮，让你看着错点数
                震荡不休、准确率死活上不去——墙，是真的硬。
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

        五组演示，认识第一位主角：

        - **演示 1** · 造数据：两团点、+1/-1 标签、sign 判决
        - **演示 2** · 手动挡：从一条歪边界出发，亲手纠正一个错点
        - **演示 3** · 完整训练：放它自己学——8 次更新、4 轮收敛
        - **演示 4** · 撞墙：XOR 上挣扎 30 轮的完整记录
        - **演示 5** · 同题对照：感知机 vs 逻辑回归，及格交卷 vs 稳中求进
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 1 · 造数据：两团点与 sign 判决

        老配方（第 16 课）：两团高斯点，种子固定 13。这次把标签换成
        **+1 / -1**（感知机的传统记号，概念 2 讲过它省 if-else 的好处）。

        顺便把 sign 函数画出来看一眼：一个台阶。**z > 0 跳到 +1，
        z ≤ 0 蹲在 -1**——就是这么一个不讲情面的开关。注意它和 16 课
        sigmoid 的关系：sigmoid 是这个台阶的"平滑版"，拐点也在 z = 0、
        也在那里取 0.5——一个冷面判官，一个温柔顾问。
        """
    )
    return


@app.cell
def _(np, plt):
    # 演示 1 · 造两团线性可分的点（+1 / -1 标签）
    _rng = np.random.default_rng(13)               # 专属 rng，种子 13（10 课老规矩）
    X = np.vstack([
        _rng.normal((1.0, 1.0), 0.85, size=(35, 2)),    # 正类：中心 (1,1) 的云
        _rng.normal((-1.0, -1.0), 0.85, size=(35, 2)),  # 负类：中心 (-1,-1) 的云
    ])                                             # (70, 2)：每行一个样本（12 课行样本约定）
    y = np.array([1] * 35 + [-1] * 35)             # 前 35 个 +1，后 35 个 -1

    # ---- 左图：两团点；右图：sign 台阶 ----
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4.2))

    _axes[0].scatter(X[y == 1, 0], X[y == 1, 1], s=30, color="tab:blue", label="+1 类（35 点）")
    _axes[0].scatter(X[y == -1, 0], X[y == -1, 1], s=30, color="tab:red", marker="s",
                     label="-1 类（35 点）")
    _axes[0].set_title("两团点：肉眼可见能被一条直线分开")
    _axes[0].set_xlabel("x1")
    _axes[0].set_ylabel("x2")
    _axes[0].legend(loc="upper left")
    _axes[0].grid(True, alpha=0.3)

    _zz = np.linspace(-3, 3, 400)                  # 采样点（03 课 linspace）
    _axes[1].plot(_zz, np.sign(_zz), color="tab:green", lw=2.5, label="sign(z)")
    _axes[1].plot(_zz, 1 / (1 + np.exp(-_zz)), color="gray", lw=1.5, ls="--",
                  alpha=0.8, label="sigmoid(z)（16 课，对照）")
    _axes[1].axhline(0, color="gray", lw=0.8, alpha=0.5)
    _axes[1].set_title("sign：不讲情面的台阶开关")
    _axes[1].set_xlabel("z = w·x + b")
    _axes[1].set_ylabel("输出")
    _axes[1].set_ylim(-1.6, 1.6)
    _axes[1].legend(loc="upper left")
    _axes[1].grid(True, alpha=0.3)

    _fig.tight_layout()
    _fig
    return (X, y)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 2 · 手动挡：亲手纠正一个错点

        感知机的学习规则说：错了才动。那我们先动一次看看。

        从一条**手摆的歪边界**出发：w = (1, -0.6)、b = 0.3（故意摆歪，
        模拟"没学之前的胡乱猜测"）。这一版参数下 70 个点里错了 **27** 个
        （左图红圈套住的）。找到数据里**第一个**错点，做一次更新——

        ```
        w ← w + y·x        （lr 取 1：只改步子大小，不改方向，概念 3 讲过）
        b ← b + y
        ```

        右图是更新后的世界。盯住两个变化：**边界转了**（w 是法线，w 一动
        边界就跟着转——12 课的几何在眼前兑现）；**红圈从 27 个掉到 11 个**。
        一次手术，砍掉 16 个错点。剩下 11 个？留给下一轮扫描继续修。
        """
    )
    return


@app.cell
def _(X, np, plt, y):
    # 演示 2 · 手动挡：从歪边界出发，亲手做一次感知机更新

    # ---- 画边界的通用小工具：给定 (w, b)，在 ax 上画直线 w·x + b = 0 ----
    # 原理（12 课）：边界上离原点最近的点是 -b·w/‖w‖²，走向沿 (-w2, w1)
    def draw_boundary(ax, w, b, color, label, ls="-", lw=2.0):
        _wn = np.asarray(w, dtype=float)
        _p0 = -b * _wn / (_wn @ _wn)               # 边界上离原点最近的点
        _d = np.array([-_wn[1], _wn[0]]) / np.linalg.norm(_wn)   # 边界的方向向量
        _t = 6.0                                   # 沿走向伸 6 个单位，足够横穿画面
        _p1, _p2 = _p0 - _t * _d, _p0 + _t * _d    # 边界的两个端点
        return ax.plot([_p1[0], _p2[0]], [_p1[1], _p2[1]], color=color,
                       ls=ls, lw=lw, label=label)[0]

    _w_bad = np.array([1.0, -0.6])                 # 手摆的歪权重（法线朝右下，故意摆歪）
    _b_bad = 0.3                                   # 歪门槛
    _z_all = X @ _w_bad + _b_bad                   # 70 个打分（09 课矩阵乘一把梭）
    _wrong_before = y * _z_all <= 0                # 布尔掩码：谁错了（05 课布尔索引）
    _i_fix = int(np.argmax(_wrong_before))         # 第一个错点下标（08 课 argmax 找 True）

    # ---- 一次感知机更新（概念 3 的公式，lr = 1）----
    _lr = 1.0
    _w_new = _w_bad + _lr * y[_i_fix] * X[_i_fix]  # w ← w + lr·y·x：往错点方向拽一把
    _b_new = _b_bad + _lr * y[_i_fix]              # b ← b + lr·y：门槛跟着挪
    _wrong_after = y * (X @ _w_new + _b_new) <= 0  # 更新后还有谁错

    # ---- 双面板：更新前 vs 更新后（sharex/sharey：同一把尺子才谈得上比较）----
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4.4), sharex=True, sharey=True)
    for _ax, _w, _b, _wrong, _ttl in [
        (_axes[0], _w_bad, _b_bad, _wrong_before, f"更新前：{_wrong_before.sum()} 个错点"),
        (_axes[1], _w_new, _b_new, _wrong_after, f"一次更新后：{_wrong_after.sum()} 个错点"),
    ]:
        _ax.scatter(X[y == 1, 0], X[y == 1, 1], s=26, color="tab:blue")
        _ax.scatter(X[y == -1, 0], X[y == -1, 1], s=26, color="tab:red", marker="s")
        _ax.scatter(X[_wrong, 0], X[_wrong, 1], s=140, facecolors="none",
                    edgecolors="red", linewidths=1.8, zorder=3)     # 错点套红圈
        draw_boundary(_ax, _w, _b, "tab:green", "边界 w·x + b = 0")
        _ax.set_title(_ttl)
        _ax.set_xlabel("x1")
        _ax.set_ylabel("x2")
        _ax.set_xlim(-3.2, 3.2)
        _ax.set_ylim(-3.2, 3.2)
        _ax.grid(True, alpha=0.3)

    # 高亮"引发这次更新的点"（左图橙色大星）
    _axes[0].scatter(X[_i_fix, 0], X[_i_fix, 1], s=220, color="tab:orange",
                     marker="*", zorder=4, label="第一个错点（被纠正）")
    _axes[0].legend(loc="upper left")
    _z_fix = X[_i_fix] @ _w_bad + _b_bad
    _axes[0].annotate(f"z = {_z_fix:.2f}：差一点点就及格",
                      xy=tuple(X[_i_fix]), xytext=(-2.9, 2.45), fontsize=9,
                      arrowprops=dict(arrowstyle="->", color="tab:orange"))
    _fig
    return (draw_boundary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        读一遍数：被纠正的是 **4 号样本**——坐标 (0.56, 1.49)、标签 +1，
        旧参数给它的打分 z = **-0.03**：差 0.03 就翻正，可惜压线算错。
        更新后 w 从 (1.00, -0.60) 转到 **(1.56, 0.89)**、b 从 0.30 抬到 **1.30**，
        27 个错点掉到 **11** 个。

        值得盯一眼的细节：一次修正只保证"被纠正的那个点翻正"，
        不保证全体变好（27 → 11 是大赚，但下一组演示你会看到也有越修越歪的瞬间）。
        感知机没有全局的眼光，它只有"眼前这个点错了"的条件反射。
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 3 · 完整训练：8 次更新，4 轮收敛

        手动挡踩过一脚，挂自动挡：把"扫描一圈、错一个修一个"包成函数，
        让它自己跑到一整轮零错点为止。

        先看函数本体——四步曲的骨架还在（15 课），只是第 ③ 步换了人：

        | 四步曲（15 课） | 感知机版 |
        |---|---|
        | ① 前向 | 算 z = w·x + b（原样） |
        | ② 损失 | 不算！错 = 1、对 = 0，只数个数 |
        | ③ 梯度 | y·x（只有错点贡献，对点为零） |
        | ④ 更新 | w ← w + lr·y·x（还是"往对的方向挪"） |

        然后放它跑，记两本账：**每次更新前全场的错点数**（看它修得怎么样）
        和**每一轮的错点数**（看收敛的节奏）。
        """
    )
    return


@app.cell
def _(np):
    # 感知机训练函数：扫到一整轮零错点为止（后面几组演示反复要用，先包好）
    def perceptron_train(X, y, lr=1.0, max_epochs=100):
        """返回 (w, b, 每次更新前的全场错点数列表, 每轮错点数列表, 实际轮数)"""
        _w = np.zeros(X.shape[1])                  # 从 w = 0 起步（一张白纸）
        _b = 0.0
        _err_before = []                           # 每次更新前：全场错点数
        _per_epoch = []                            # 每一轮：错点数
        for _epoch in range(max_epochs):
            _n_wrong = 0
            for _i in range(len(y)):
                if y[_i] * (X[_i] @ _w + _b) <= 0:         # 错了（含压线 z = 0）
                    _err_before.append(int(np.sum(y * (X @ _w + _b) <= 0)))
                    _w = _w + lr * y[_i] * X[_i]           # 概念 3 的纠错更新
                    _b = _b + lr * y[_i]
                    _n_wrong += 1
            _per_epoch.append(_n_wrong)
            if _n_wrong == 0:                      # 一整轮零错点 = 收工
                break
        return _w, _b, _err_before, _per_epoch, _epoch + 1
    return (perceptron_train,)


@app.cell
def _(X, draw_boundary, mo, np, perceptron_train, plt, y):
    # 演示 3 · 完整训练：放它自己学，记录每一次更新
    w_final, b_final, _err_before, _per_epoch, _n_epochs = perceptron_train(X, y)
    n_updates = len(_err_before)                   # 总更新次数（后面练习要对答案）

    # ---- 重放一遍，抓几个边界快照（确定性过程，重算成本忽略不计）----
    _snap_at = [1, 2, 4, 8]                        # 想看的档位：第 1/2/4/8 次更新后
    _snap = {}
    _w2, _b2 = np.zeros(2), 0.0
    _count = 0
    for _ in range(_n_epochs):
        for _i in range(len(y)):
            if y[_i] * (X[_i] @ _w2 + _b2) <= 0:
                _w2 = _w2 + y[_i] * X[_i]
                _b2 = _b2 + y[_i]
                _count += 1
                if _count in _snap_at:
                    _snap[_count] = (_w2.copy(), _b2)

    # ---- 左图：边界快照四连拍（颜色由浅到深 = 时间从早到晚）----
    _fig, _axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    _ax = _axes[0]
    _ax.scatter(X[y == 1, 0], X[y == 1, 1], s=18, color="tab:blue", alpha=0.65)
    _ax.scatter(X[y == -1, 0], X[y == -1, 1], s=18, color="tab:red", marker="s", alpha=0.65)
    _shades = ["#c6dbef", "#6baed6", "#2171b5", "#08306b"]     # 越晚越深
    for _shade, _step in zip(_shades, _snap_at):
        _ws, _bs = _snap[_step]
        draw_boundary(_ax, _ws, _bs, _shade, f"更新 {_step} 次后")
    draw_boundary(_ax, w_final, b_final, "tab:green", "最终边界", lw=2.5)
    _ax.set_title(f"边界进化：{n_updates} 次更新走完一生")
    _ax.set_xlabel("x1")
    _ax.set_ylabel("x2")
    _ax.set_xlim(-3.2, 3.2)
    _ax.set_ylim(-3.2, 3.2)
    _ax.legend(fontsize=8.5, loc="upper left")
    _ax.grid(True, alpha=0.3)

    # ---- 右图：每次更新后的全场错点数（注意那个回升）----
    _ax2 = _axes[1]
    _err_after = _err_before[1:] + [0]             # 更新后 = 下一次更新前；最后一次更新后归零
    _ax2.plot(range(0, n_updates + 1), [_err_before[0]] + _err_after,
              "o-", color="tab:purple", label="每次更新后的错点数")
    _ax2.axhline(0, color="gray", lw=0.8)
    _ax2.annotate("第 5 次更新后：2 → 3\n（修好一个，碰歪另一个）",
                  xy=(5, 3), xytext=(5.8, 20), fontsize=9, color="tab:red",
                  arrowprops=dict(arrowstyle="->", color="tab:red"))
    _ax2.set_title(f"错点数：从 70 一路修到 0（{n_updates} 次更新、{_n_epochs} 轮）")
    _ax2.set_xlabel("更新次数")
    _ax2.set_ylabel("错点数")
    _ax2.set_xticks(range(0, n_updates + 1))
    _ax2.legend()
    _ax2.grid(True, alpha=0.3)

    mo.vstack([
        _fig,
        mo.md(
            f"""
            | 读数 | 值 |
            |---|---|
            | 总更新次数 | **{n_updates}** |
            | 收敛轮数 | **{_n_epochs}**（每轮错点 {_per_epoch}） |
            | 最终 w | **({w_final[0]:.2f}, {w_final[1]:.2f})**，b = **{b_final:.2f}** |
            | 最终错点数 | **0**（训练集 100% 分开） |
            """
        ),
    ])
    return (b_final, n_updates, w_final)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        读图（左图四条由浅到深的线，是边界进化的四个瞬间）：

        - **w 从全零起步**：第一轮开局 70 个点全体错判（z 全是 0，压线算错）——
          白纸开局，谁都错
        - **8 次更新、4 轮收敛**：错点数 70 → 28 → 14 → 4 → 2 → 3 → 2 → 2 → 0，
          每轮错点 [4, 2, 2, 0]——收敛定理兑现，而且远比理论上界宽松
        - **右图那个刺眼的回升**：第 5 次更新把错点数从 2 修成了 **3**。
          感知机只对"当前这个点"负责：把它翻正的同时，可能把旁边一个
          原本对的点挤过线。没有全局损失函数的世界，就是这么跌跌撞撞
        - **45 号和 8 号样本各被纠正了两次**：它们贴着边界住，边界每动一次
          它们就换个阵营——边界附近的点，永远是最后安顿下来的

        一个值得回味的对比：15 课的梯度下降每一步都在"整体损失"上下坡，
        感知机每一步只在意"眼下的错点"。两条哲学都能到罗马——
        但只有前者能升级成多层网络（22 课见分晓）。
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 4 · 撞墙：XOR 的 30 轮挣扎

        现在把同一台感知机按到 XOR 上。数据用四团小云（概念 4 的四个角，
        每角 10 个带噪声的点），**线性不可分**——收敛定理的前提被撕毁，
        我们亲眼看它挣扎：

        右图是 30 轮的**每轮错点数**：3、5、5、5、4、4、7、6、8、6……
        永远在 4~8 之间震荡，一个都降不下去。30 轮后准确率停在 **52.5%**——
        比抛硬币强不了多少。

        左图还画了一条**随机撒线找出的最优直线**（撒 2000 条、取准确率最高的）：
        也只能对 **75%**。这不是运气差，是几何判决——XOR 的两个 +1 在对角线上，
        任何半平面只要装下两个 +1，连接它们的对角线就整个落进来，
        而这条对角线把两个 -1 隔在两侧，于是至少一个 -1 被捎带进来。
        **75% 是所有直线的天花板**，撒再多线也只是逼近它。

        墙，是真的硬。出路只有摞层——20 课见。
        """
    )
    return


@app.cell
def _(draw_boundary, mo, np, perceptron_train, plt):
    # 演示 4 · XOR：四团点（对角同号），感知机 30 轮学不动
    _rng = np.random.default_rng(7)                # 换个种子换批数据（10 课：一课一台 rng 也行）
    _centers = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])   # 四个角
    _labs = np.array([-1, -1, 1, 1])               # 对角同号 = XOR
    _Xx = np.vstack([_rng.normal(_c, 0.18, size=(10, 2)) for _c in _centers])
    _yx = np.repeat(_labs, 10)                     # (40,) 标签

    _wx, _bx, _, _per_epoch_x, _ep = perceptron_train(_Xx, _yx, max_epochs=30)
    _acc = np.mean(np.sign(_Xx @ _wx + _bx) == _yx)          # 30 轮后的准确率

    # 随机撒 2000 条直线，看"直线的天花板"（纯几何对照，不是训练）
    _best_acc, _best_line = 0.0, None
    for _ in range(2000):
        _cand_w = _rng.uniform(-3, 3, 2)           # 随机法线方向
        _cand_b = _rng.uniform(-2, 2)              # 随机平移
        _a = np.mean(np.sign(_Xx @ _cand_w + _cand_b) == _yx)
        if _a > _best_acc:
            _best_acc, _best_line = _a, (_cand_w, _cand_b)

    # ---- 左图：数据 + 感知机 30 轮后的边界 + 随机最优直线 ----
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4.4))
    _ax = _axes[0]
    _ax.scatter(_Xx[_yx == 1, 0], _Xx[_yx == 1, 1], s=30, color="tab:blue", label="+1（一条对角）")
    _ax.scatter(_Xx[_yx == -1, 0], _Xx[_yx == -1, 1], s=30, color="tab:red", marker="s",
                label="-1（另一条对角）")
    draw_boundary(_ax, _wx, _bx, "tab:purple", f"感知机 30 轮后（准确率 {_acc:.1%}）")
    draw_boundary(_ax, _best_line[0], _best_line[1], "tab:green",
                  f"随机 2000 条里最优（{_best_acc:.0%}）", ls="--")
    _ax.set_title("XOR：一条直线怎么摆都不对")
    _ax.set_xlabel("x1")
    _ax.set_ylabel("x2")
    _ax.legend(fontsize=9, loc="upper right")
    _ax.grid(True, alpha=0.3)

    # ---- 右图：30 轮的每轮错点数 ----
    _axes[1].plot(range(1, 31), _per_epoch_x, "o-", color="tab:red", ms=4)
    _axes[1].set_title("每轮错点数：在 4~8 之间震荡，永不收敛")
    _axes[1].set_xlabel("轮数（epoch）")
    _axes[1].set_ylabel("错点数")
    _axes[1].set_ylim(0, 12)
    _axes[1].grid(True, alpha=0.3)

    mo.vstack([
        _fig,
        mo.md(
            f"""
            | 读数 | 值 |
            |---|---|
            | 30 轮后准确率 | **{_acc:.1%}**（瞎猜基线 50%） |
            | 每轮错点数 | **永不归零**（最后 5 轮：{_per_epoch_x[-5:]}） |
            | 随机撒线的最优 | **{_best_acc:.0%}**——所有直线的几何天花板 |
            """
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 5 · 同题对照：感知机 vs 逻辑回归

        概念 1 立的 flag 来兑现：同一条直线、两种性格，同台比一场。
        一边是本课训练的感知机，一边是 16 课配方训练的逻辑回归
        （lr = 0.5、2000 步梯度下降，老代码一行不改）。

        两个都把训练集分到了 100%。但看**边界站的位置**：量一量每个点到
        边界的最近距离（几何间隔 = min|z| / ‖w‖，把概念 2 的 y·z 换算成尺子）——

        感知机的边界**贴着点过**，像个刚过及格线就交卷的学生；
        逻辑回归的边界站在两团点的**正中间**，因为它的交叉熵损失在**每一个点**上
        都永远有梯度、永远在把边界往"两边都离远点"的方向推——
        这就是 16 课那位"温柔顾问"的长期主义。
        """
    )
    return


@app.cell
def _(X, b_final, draw_boundary, mo, np, plt, w_final, y):
    # 演示 5 · 感知机 vs 逻辑回归：同样 100% 分开，边界站位天差地别
    def _sigmoid(_z):
        return 1 / (1 + np.exp(-_z))               # 16 课的老朋友

    _y01 = (y + 1) / 2                             # +1/-1 换回 0/1（16 课的标签习惯）
    _wl = np.zeros(2)                              # 逻辑回归：16 课配方原样照抄
    _bl = 0.0
    for _ in range(2000):                          # 2000 步梯度下降
        _p = _sigmoid(X @ _wl + _bl)
        _wl = _wl - 0.5 * (X.T @ (_p - _y01)) / len(y)     # 16 课推过的梯度
        _bl = _bl - 0.5 * np.mean(_p - _y01)

    # 几何间隔：全场点到边界的最近距离 = min|z| / ‖w‖
    _gap_p = np.min(np.abs(X @ w_final + b_final)) / np.linalg.norm(w_final)
    _gap_l = np.min(np.abs(X @ _wl + _bl)) / np.linalg.norm(_wl)
    _ratio = _gap_l / _gap_p

    # ---- 画图：两条边界同框 ----
    _fig, _ax = plt.subplots(figsize=(7.8, 4.8))
    _ax.scatter(X[y == 1, 0], X[y == 1, 1], s=22, color="tab:blue", alpha=0.7)
    _ax.scatter(X[y == -1, 0], X[y == -1, 1], s=22, color="tab:red", marker="s", alpha=0.7)
    draw_boundary(_ax, w_final, b_final, "tab:purple",
                  f"感知机（间隔 {_gap_p:.3f}：贴着点过）")
    draw_boundary(_ax, _wl, _bl, "tab:green",
                  f"逻辑回归（间隔 {_gap_l:.3f}：站正中间）")
    # 各自"离边界最近的点"套圈标出
    _i_p = int(np.argmin(np.abs(X @ w_final + b_final)))
    _i_l = int(np.argmin(np.abs(X @ _wl + _bl)))
    _ax.scatter(X[_i_p, 0], X[_i_p, 1], s=180, facecolors="none",
                edgecolors="tab:purple", linewidths=2)
    _ax.scatter(X[_i_l, 0], X[_i_l, 1], s=180, facecolors="none",
                edgecolors="tab:green", linewidths=2)
    _ax.set_title("及格就交卷 vs 稳中求进：两条边界的品味")
    _ax.set_xlabel("x1")
    _ax.set_ylabel("x2")
    _ax.legend(fontsize=9)
    _ax.grid(True, alpha=0.3)

    mo.vstack([
        _fig,
        mo.md(
            f"""
            | 量 | 感知机 | 逻辑回归 |
            |---|---|---|
            | 训练准确率 | 100% | 100% |
            | 法线方向 | ({w_final[0]/np.linalg.norm(w_final):.2f}, {w_final[1]/np.linalg.norm(w_final):.2f}) | ({_wl[0]/np.linalg.norm(_wl):.2f}, {_wl[1]/np.linalg.norm(_wl):.2f}) |
            | 几何间隔 | **{_gap_p:.3f}** | **{_gap_l:.3f}**（{_ratio:.0f} 倍宽） |
            | 停下的理由 | 零错点，纠错项消失 | 你叫停的（损失还在降，只是很慢） |

            两条边界的**方向**几乎一样（法线都朝右上一带），差的只是**站位**。
            哪个更好？训练集说了不算——如果新来的点恰好落在那条窄缝附近，
            贴着过的边界会当场翻车（17 课"背题代价"的边界版）。
            感知机对"分开之后"毫无追求，这个性格缺陷，深度学习用更聪明的
            损失函数来治——21 课的故事。
            """
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 🎛️ 交互实验

        **成长录像机**——把演示 3 的 8 次更新做成逐帧快进。拖动滑块（0~8），
        每一档都是感知机生涯的一个瞬间：当前边界、引发这次纠错的点（橙色星）、
        更新后还剩的错点数，一屏全给你。

        两种玩法：

        1. 从 0 拖到 8：看边界从"白纸"一步步进化到"交卷"，注意第 5 档
           错点数会从 2 反弹到 3（演示 3 右图那个刺眼的回升，在这里亲手摸到）
        2. 拖回第 3、4 档盯边界转动的方向——每次都朝着"被纠正的点"
           所在的一侧翻过去，概念 3 的"往 x 方向拽"看得见摸得着

        第 0 档没有边界：w = (0, 0) 时 z 恒为 0，压线算错——70 个点全体蒙冤。
        这就是"白纸开局"的真正含义。
        """
    )
    return


@app.cell
def _(mo):
    # 交互控制：成长录像机（debounce=True：松手才重算，画面不闪烁）
    step_slider = mo.ui.slider(0, 8, value=0, step=1, debounce=True,
                               label="🎞️ 快进到第几次更新？（0~8）")
    step_slider
    return (step_slider,)


@app.cell
def _(X, draw_boundary, mo, np, plt, step_slider, y):
    # ===== 交互实验 · 成长录像机（响应单元）=====
    _k = step_slider.value                        # 当前档位 = 已完成的更新次数

    # ---- 重放感知机的一生，记下每一档的状态（确定性，成本忽略不计）----
    _w2, _b2 = np.zeros(2), 0.0
    _states = [(_w2.copy(), _b2, -1, int(np.sum(y * (X @ _w2 + _b2) <= 0)))]   # 第 0 档
    for _ in range(100):
        _n_wrong = 0
        for _i in range(len(y)):
            if y[_i] * (X[_i] @ _w2 + _b2) <= 0:
                _w2 = _w2 + y[_i] * X[_i]
                _b2 = _b2 + y[_i]
                _n_wrong += 1
                _states.append((_w2.copy(), _b2, _i,
                                int(np.sum(y * (X @ _w2 + _b2) <= 0))))       # 第 k 档
        if _n_wrong == 0:
            break
    _n_total = len(_states) - 1                  # 总更新次数（应为 8）

    _wk, _bk, _ik, _errk = _states[_k]           # 当前档位的四件套

    # ---- 画图：当前边界 + 被纠正的点 ----
    _fig, _ax = plt.subplots(figsize=(7.8, 4.6))
    _ax.scatter(X[y == 1, 0], X[y == 1, 1], s=26, color="tab:blue")
    _ax.scatter(X[y == -1, 0], X[y == -1, 1], s=26, color="tab:red", marker="s")
    _wrong_now = y * (X @ _wk + _bk) <= 0        # 当前还错的点
    _ax.scatter(X[_wrong_now, 0], X[_wrong_now, 1], s=140, facecolors="none",
                edgecolors="red", linewidths=1.8, zorder=3)
    if _k >= 1:
        draw_boundary(_ax, _wk, _bk, "tab:green", "当前边界")
        _ax.scatter(X[_ik, 0], X[_ik, 1], s=230, color="tab:orange",
                    marker="*", zorder=4, label=f"第 {_k} 次更新纠正的样本 {_ik}")
        _ax.legend(loc="upper left")
    else:
        _ax.set_xlim(-3.2, 3.2)                  # w = 0 画不出边界，只摆视野
    _ax.set_ylim(-3.2, 3.2)
    _ax.set_xlim(-3.2, 3.2)
    _ax.set_title(f"第 {_k} 次更新后的世界（还剩 {_errk} 个错点）")
    _ax.set_xlabel("x1")
    _ax.set_ylabel("x2")
    _ax.grid(True, alpha=0.3)

    # ---- 判词：现在处于生涯的哪一段 ----
    if _k == 0:
        _verdict = "🎬 白纸开局：w = 0，z 恒为 0，70 个点全体蒙冤——拖一下滑块，看第一次纠错"
    elif _k < _n_total:
        _verdict = f"⏩ 已完成 {_k} 次更新，还剩 {_errk} 个错点——继续往右拖"
    else:
        _verdict = f"🏁 收敛！零错点——{_n_total} 次更新、边界交卷（演示 3 同款结局）"

    mo.vstack([
        _fig,
        mo.md(
            f"""
            | 量 | 值 |
            |---|---|
            | 已更新次数 | **{_k}** |
            | 当前 w | **({_wk[0]:.2f}, {_wk[1]:.2f})** |
            | 当前 b | **{_bk:.2f}** |
            | 剩余错点 | **{_errk}** / 70 |
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
                **练习 1 · 手写判决函数**：补全 `predict(x, w, b)`——
                就一行 `np.sign`，但这是你亲手写下的第一个神经元。

                **练习 2 · 补全训练循环**：`perceptron_train_ex2` 的更新两行被抽走了，
                补回来，跑出演示 3 的 8 次更新。

                **练习 3 · 挑战 · AND 与 XOR 的命运**：单位正方形的四个角，
                AND 标签（三个 +1 一个 -1）和 XOR 标签（对角同号）——
                同一台感知机，两种命运。各跑一遍，填上更新次数。

                规则不变：先把 `None` 换成你的答案再运行；写完再看参考答案。
                """
            )
        }
    )
    return


@app.cell
def _(b_final, mo, np, w_final):
    # ===== 练习 1 · 手写判决函数 predict =====
    def predict(x, w, b):
        """输入单个样本 x（长度 2）、权重 w、偏置 b，输出 +1 或 -1"""
        _z = x @ w + b                   # ① 打分（这行不变）
        return None                      # ← ② 补这行：np.sign(_z)，把打分砍成正负号

    predict_ex1 = None                  # ← ③ 补完后，把这行换成 predict（就写函数名）

    if predict_ex1 is not None:
        try:
            _test_pts = np.array([[2.0, 2.0], [-2.0, -2.0], [0.5, -2.0]])   # 三个考点
            _preds = [int(predict_ex1(_p, w_final, b_final)) for _p in _test_pts]
            if _preds == [1, -1, -1]:
                _msg = f"三个考点的判决 {_preds}——✅ 全对，第一个神经元出厂"
            else:
                _msg = f"三个考点的判决 {_preds}（期望 [1, -1, -1]）——❌ 检查 sign 的方向"
        except TypeError:
            _msg = "⏳ 函数跑起来了但吐出 None——② 的 return 补了吗？"
    else:
        _msg = "⏳ 先补 ② 的 return，再把 predict_ex1 换成 predict"
    mo.md(_msg)
    return (predict, predict_ex1)


@app.cell
def _(X, mo, n_updates, np, y):
    # ===== 练习 2 · 补全感知机训练循环 =====
    def perceptron_train_ex2(X, y, lr=1.0, max_epochs=100):
        """和演示 3 的 perceptron_train 一模一样，只是更新那两行被抽走了"""
        _w = np.zeros(X.shape[1])
        _b = 0.0
        _n_upd = 0
        for _ in range(max_epochs):
            _n_wrong = 0
            for _i in range(len(y)):
                if y[_i] * (X[_i] @ _w + _b) <= 0:
                    # ← ① 补两行（概念 3 的纠错规则）：
                    #    _w = _w + lr * y[_i] * X[_i]
                    #    _b = _b + lr * y[_i]
                    _n_wrong += 1
                    _n_upd += 1
            if _n_wrong == 0:
                break
        return _w, _b, _n_upd

    n_updates_ex2 = None    # ← ② 跑一遍 perceptron_train_ex2(X, y)，把更新次数填进来

    if n_updates_ex2 is not None:
        if n_updates_ex2 == n_updates:
            _msg = f"✅ {n_updates_ex2} 次——和演示 3 分毫不差，你已经会从零训练一个神经元了"
        elif n_updates_ex2 == 7000:
            _msg = ("❌ 7000 次 = 70 错 × 100 轮全在空转：① 的两行没生效，"
                    "w 从头到尾没动过——每个错点每轮都被重新数一遍")
        else:
            _msg = f"❌ 演示 3 是 {n_updates} 次，你是 {n_updates_ex2} 次——对照概念 3 检查那两行"
    else:
        _msg = "⏳ 先补 ① 的两行更新，再跑函数把 ② 填上"
    mo.md(_msg)
    return (n_updates_ex2,)


@app.cell
def _(mo, np, perceptron_train):
    # ===== 练习 3 · 挑战：AND 收敛，XOR 卡死 =====
    _pts = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)   # 单位正方形四角
    _and_y = np.array([-1, -1, -1, 1])   # AND：只有 (1,1) 是 +1（线性可分）
    _xor_y = np.array([-1, 1, 1, -1])    # XOR：对角同号（线性不可分）

    # 提示：第一个返回值是 w、第二个是 b；更新次数 = len(_err)
    # （_err 是"每次更新前的全场错点数"列表，它的长度就是更新次数）
    #   _w, _b, _err, _pe, _ep = perceptron_train(_pts, _and_y)

    and_updates_ex3 = None    # ← ① AND 收敛总共用了几次更新？
    xor_updates_ex3 = None    # ← ② XOR 跑满 50 轮（max_epochs=50），累计更新了多少次？

    if and_updates_ex3 is not None and xor_updates_ex3 is not None:
        _msg = (f"你的答案：AND **{and_updates_ex3}** 次，XOR **{xor_updates_ex3}** 次。"
                + ("✅ 一个收敛、一个空转——两种命运都亲手摸过了"
                   if (and_updates_ex3, xor_updates_ex3) == (18, 200) else
                   "❌ 再跑一遍：AND 用默认参数，XOR 记得 max_epochs=50"))
    else:
        _msg = "⏳ 两个数字都跑出来再对答案（提示：XOR 的更新次数是 4 的倍数）"
    mo.md(_msg)
    return (and_updates_ex3, xor_updates_ex3)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 1 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                return np.sign(_z)      # 就这一行：把打分砍成正负号
                ```

                三个考点的判决：**(2, 2) → +1**（z = 10.6，正类大本营）、
                **(-2, -2) → -1**（z = -10.6）、**(0.5, -2) → -1**（z = -4.4，
                在边界的另一侧）。

                一个冷细节：`np.sign(0)` 返回 **0**，不是 -1。所以严格的感知机判决
                应该写 `1 if _z > 0 else -1`（压线判负，和本课"y·z ≤ 0 算错"的约定
                完全一致）。两个写法在考点上都对——因为考点不压线；但在边界正中央，
                它们会给出不同答案。**工程上的教训：先想清楚边界上的点归谁。**
                """
            ),
            "✅ 练习 2 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                _w = _w + lr * y[_i] * X[_i]      # ① 把 w 往错点方向拽一把
                _b = _b + lr * y[_i]              # ② 门槛跟着挪
                ```

                补完后跑 `perceptron_train_ex2(X, y)`，更新次数 = **8**，和演示 3 一致。

                两个值得咀嚼的细节：

                - **如果不补这两行**：w 永远是 0，每一轮都数出 70 个错点，
                  100 轮后 n_upd = 7000——"更新次数暴增"往往是"参数没动"的信号，
                  这个直觉以后调试神经网络时天天用
                - **lr 换成 0.1 结果还是 8**：概念 3 的彩蛋在这里兑现——
                  lr 只把 w、b 整体缩小 10 倍，所有 z 也跟着缩小 10 倍，
                  正负号一个都不变，纠错序列就一个都不变。
                  感知机的 lr 是音量旋钮，不是方向盘
                """
            ),
            "✅ 练习 3 参考答案（先自己写过再看！）": mo.md(
                """
                答案：**AND 18 次更新、9 轮收敛**（每轮错点 [2, 3, 3, 2, 2, 3, 2, 1, 0]，
                最终 w = (3, 2)、b = -4，边界 3x1 + 2x2 = 4——把 (1,1) 单独圈进
                正类，其余三个角挡在外面）；**XOR 50 轮 200 次更新**，
                每轮恒定 4 次、一次不省。

                XOR 那 200 次里藏着一个漂亮的死循环：四个点的更新
                （+、+、−、−）每一轮恰好**互相抵消**——w 绕一圈回到 (0, 0)，
                下一轮从头再来。**西西弗斯式训练**：不是不努力，是地形决定了
                努力白费。

                顺手做个对照：把标签换成 OR（`np.array([-1, 1, 1, 1])`，
                只有 (0,0) 是 -1）——同样线性可分，9 次更新、6 轮就收敛。
                AND、OR 都过，XOR 独死：**可分与不可分的分水岭，不靠努力靠几何**。

                再留意一个反直觉的小事实：AND 只有 4 个点、参数只有 3 个，
                却更新了 18 次——更新次数和"问题大小"没有直接关系，
                只和数据的**间隔**干不干净有关（Novikoff 上界 (R/γ)² 里
                根本没有样本数的事，只有间隔的事）。
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
                1. **神经元零件表**：加权投票（w）+ 门槛（b）+ 开关（sign）；
                   打分公式和 16 课逻辑回归同一个 z = w·x + b，
                   区别只在输出：sigmoid 给概率，sign 给硬判决
                2. **学习规则**：错了才动——w ← w + lr·y·x、b ← b + lr·y；
                   判对的样本零贡献。y·z > 0 即正确，|y·z| 是"对得多有底气"
                3. **收敛保证与死刑判决**：线性可分 → 有限步必收敛（Novikoff 定理）；
                   不可分 → 永不收敛，错点数永远震荡——这反而是它当
                   "可分性检测器"的用法（演示 4 就是这么给 XOR 定罪的）
                4. **训练骨架神似四步曲，但第 ③ 步不是梯度**：感知机的"梯度"
                   只有错点贡献，没有全局损失函数——所以错点数会非单调
                   （演示 3 的 2 → 3 回升），它只对眼下的点负责
                5. **XOR 之墙**：一条直线至多分对 XOR 的 75%（几何天花板）；
                   1969 年这堵墙引发第一次 AI 寒冬。出路 = 摞多层（20 课）
                   + 非线性激活（19 课）
                6. **及格就交卷**：感知机零错点即停，边界贴着点过（间隔 0.003）；
                   逻辑回归被交叉熵持续推动，站在正中间（0.024，8 倍宽）——
                   "分开"和"站得稳"是两回事，21 课用损失函数把后者也管起来
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

        **下一课预告** — `19_激活函数`：

        感知机出征归来，也把 sign 的两个毛病带回了台面：**它不可导**
        （z = 0 处突然跳变，14 课的链式法则没法从它身上穿过去）；
        **它只给判决不给概率**（错了多离谱、对了多勉强，一概不说）。

        下一课请出把直线掰弯的那只手——**激活函数**：sigmoid、tanh、ReLU
        三位选手同台，先讲一个细思极恐的事实：**线性摞线性还是线性**，
        摞一百层矩阵乘法等于一层（12 课的复合变换早埋好了伏笔）——
        没有非线性关节，深度就是一句空话。

        你还会亲手用两个 ReLU 拼出一条折线，提前看见"万能逼近"的雏形。

        🎆 第一个神经元出厂了，去给它装上会弯的关节！
        """
    )
    return


if __name__ == "__main__":
    app.run()
