import marimo

__generated_with = "0.25.2"
app = marimo.App(app_title="16 · 逻辑回归与交叉熵", width="medium")


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
        # 16 · 逻辑回归与交叉熵

        **从零到 Transformer · 手写深度学习** ｜ 第 3 部分 · 经典机器学习热身（2/3）
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📖 本课目标（先读我）": mo.md(
                """
                **这一课只换两件装备：sigmoid 上岗，交叉熵接班。**

                第 15 课结尾留了一句预告：回归预测的是连续数值，但世界的另一半问题是
                "是/否"——这封邮件是垃圾吗？图片里是猫还是狗？把直线掰弯成概率需要
                sigmoid，把 MSE 换成交叉熵，**同一个四步曲，新装备上阵**。今天兑现承诺。

                学完你将能：

                - 用 **sigmoid** 把任意打分压进 (0, 1)，说出它的两条性质：
                  关于 (0, 0.5) 中心对称、导数 σ'(z) = σ(z)·(1-σ(z))
                - 读懂**交叉熵**："说错话的重罚"——自信地错，比犹犹豫豫地错罚得狠得多
                - 画出**决策边界**：15 课的直线换了身份，成为两类"领地"的分界线
                - 用**同一个四步曲**训练逻辑回归，并记住那份干净得惊人的梯度
                  mean((p - y)·x)
                - 拧一拧**判决门槛**：训练给概率，业务定门槛——垃圾邮件和癌症筛查
                  拧的方向刚好相反

                前置：第 15 课的四步曲与损失曲线（本课的地基）、第 14 课链式法则
                （梯度推导的引擎）、第 13 课 meshgrid/contour（概率地图的画笔）、
                第 11 课散点图与 set_aspect（老朋友了）。

                整个第 3 部分都在练一件事：**同一个训练骨架，装不同的零件**。
                和 15 课摆在一起看，本课只改了三行字：

                | | 15 课 · 线性回归 | 本课 · 逻辑回归 |
                |---|---|---|
                | 问题 | 预测连续数值（房价） | 判别属于哪类（垃圾邮件？） |
                | 前向 | y = wx + b | p = σ(wx + b) |
                | 损失 | MSE（差的平方） | 交叉熵（说错话的重罚） |
                | 梯度 | mean(2(ŷ - y)·x) | mean((p - y)·x) |

                看出来了吗？去掉一个可吸进学习率的常数因子 2，连梯度都长一个模样。
                这就是"稳扎稳打"的复利。
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "🎯 概念 1/4 · 分类问题与概率": mo.md(
                """
                第 15 课的线性回归预测**连续数值**：房价、气温、销量——一条直线拟合一个数。
                但世界的另一半问题是**"是/否"**：这封邮件是垃圾吗？图里是猫还是狗？
                明天会下雨吗？这类问题的答案不再是数，而是**属于哪一类**——
                最自然的语言是**概率** p ∈ (0, 1)。

                麻烦来了：15 课的打分 z = wx + b 是任意实数（可以是 -7.3，也可以是 42），
                怎么变成 (0, 1) 里的概率？请出本课第一位主角——**sigmoid**（逻辑函数）：

                ```
                σ(z) = 1 / (1 + e^(-z))
                ```

                它是一位**概率翻译官**：输入任意实数，输出 (0, 1) 之间的概率。
                z = 0 → σ = 0.5（五五开）；z 越大 → 越接近 1；z 越小 → 越接近 0。
                试试手感：z = 2 → 0.8808，z = -2 → 0.1192，z = ±6 时已经贴到
                0.9975 / 0.0025 的渐近线上——sigmoid 压得再狠，也永远不说 0 或 1 的死话。
                （为什么叫"逻辑回归"这么别扭的名字？历史遗留：当年它确实顶着
                "回归"的名号出生，后来大家发现它其实是分类器，名字却改不动了。）

                两条性质，先记住形式（后续马上要用）：

                - **中心对称**：曲线关于 (0, 0.5) 中心对称，代数说法是
                  σ(-z) = 1 - σ(z)——所以 σ(2) + σ(-2) 恰好等于 1
                - **导数特别乖**：σ'(z) = σ(z)·(1-σ(z))——自己的导数，用自己的值
                  就能拼出来（演示 1 右图亲眼验证）
                """
            ),
            "⚖️ 概念 2/4 · 交叉熵损失：说错话的重罚": mo.md(
                """
                有了概率，损失函数也要跟着换班——分类世界的标配是**交叉熵**
                （cross-entropy）：

                ```
                L = -mean( y·log(p) + (1-y)·log(1-p) )
                ```

                y 是 0/1 真值，p 是预测概率。别被公式吓住，它其实只在问一句话：
                **"真值那边，你押了多少？"**——y=1 的样本只算 -log(p)，
                y=0 的样本只算 -log(1-p)，另一项乘上 0 自动消失（公式自带分拣）。

                一张"罚单表"（真值 y=1 时，押注 p 与罚金 -log(p) 的关系）：

                | 押注 p | 罚金 -log(p) | 解读 |
                |---|---|---|
                | 0.99 | 0.010 | 自信答对，几乎不罚 |
                | 0.90 | 0.105 | 答对，轻罚 |
                | 0.50 | 0.693 | 犹豫不决（= 瞎猜的代价） |
                | 0.10 | 2.303 | 自信答错，重罚 |
                | 0.01 | 4.605 | 错得振振有词，往死里罚 |

                这就是**"说错话的重罚"**：真值 y=1 时预测 p=0.01，罚 4.6；
                预测 p=0.99，只罚 0.01——自信地错与自信地对，罚金差出约 460 倍。
                **"自信地错"比"犹犹豫豫地错"罚得狠得多**——这正是我们想训练出的性格：
                不懂就承认五五开（罚 0.693），懂了就大声说出来。

                最外层的 mean 也别小看：取平均让罚单不随样本数变胖——
                80 个样本和 800 个样本的损失能在同一把尺子上量（15 课的 MSE
                也套了同一件外套 mean，骨架从头到尾都是统一的）。

                顺带回答一个悬念：为什么分类不用 15 课的 MSE？因为 MSE 配 sigmoid
                会得一种病——错得离谱时（p 贴着 0 或 1），sigmoid 两端太平、导数趋近 0，
                模型"知道自己错了却迈不动腿"。交叉熵配 sigmoid 恰好把这块平地铲平——
                概念 4 会看到那份干净得惊人的梯度。
                """
            ),
            "🧮 概念 3/4 · 模型与决策边界：领地的分界线": mo.md(
                """
                "逻辑回归"的"回归"二字是历史遗留——它其实是个**分类器**。
                模型只有两步：先打分（15 课的老本行），再翻译成概率（本课新装备）：

                ```
                p = sigmoid(w·x + b)                一维特征版
                p = sigmoid(w1·x1 + w2·x2 + b)      二维特征版（本课主场）
                ```

                判决规则：**p ≥ 0.5 判 1 类，否则判 0 类**。而 sigmoid 单调递增，
                σ(z) ≥ 0.5 恰好等价于 z ≥ 0（0.5 对应输入 0），于是：

                ```
                判 1 类  ⟺  w1·x1 + w2·x2 + b ≥ 0
                边界线：   w1·x1 + w2·x2 + b = 0     （一条直线！）
                ```

                15 课的直线在这里换了身份：**不再是预测值，而是两类"领地"的分界线**——
                边界一侧是 0 类的地盘（p < 0.5），另一侧是 1 类的地盘（p > 0.5），
                线上每个点五五开、两不管。

                用 11 课的几何眼睛再看一眼：w 是平面上的一支箭头，边界线与 w **垂直**——
                w 的箭头指向"1 类腹地"（沿它走 z 越来越大），而且 |w| 越长，
                边界两侧的概率翻转越猛烈。第 12 课"矩阵即变换"的直觉，在这里悄悄回来了。

                还有个常被问到的问题：**二维都这么费劲，一百维怎么办？** 放心，
                公式一个字都不用改——w 变成一百维向量，X 变成 (n, 100) 的矩阵，
                边界从直线升级成"超平面"（一百维空间里那张平的分隔面）。
                NumPy 的 X @ w 照单全收，这正是 09 课矩阵乘法埋好的伏笔。
                """
            ),
            "🔁 概念 4/4 · 同一个四步曲：骨架不变，只换装备": mo.md(
                """
                15 课的四步曲骨架，本课**原封不动**再跑一遍，只换两件装备
                （sigmoid 上岗、MSE 换成交叉熵）：

                ```
                ① 前向：z = X @ w + b  →  p = σ(z)               （打分 + 翻译成概率）
                ② 损失：L = -mean(y·log(p) + (1-y)·log(1-p))     （交叉熵罚单）
                ③ 梯度：dL/dw = mean((p - y)·x)，dL/db = mean(p - y)
                ④ 更新：w ← w - lr · 梯度                          （15 课同款下坡）
                ```

                ③ 的形式漂亮得**惊人**：预测减真值，乘特征，求平均——
                没有 log、没有 e、没有分式，比 MSE 的梯度还干净。

                这不是巧合，是 14 课链式法则的**三层应用**（sigmoid → 交叉熵 → 对 w）：
                交叉熵求导带来的分母，恰好被 sigmoid 导数里的同类项约掉——
                两层"难看的导数"相乘，噪声互相抵消，留下干干净净的 (p - y)。
                正文先给结果：演示 4 用它训练，练习 3 让你用**数值梯度**亲手验真；
                完整的三层推导放在练习 3 的参考答案里，敢挑战就去看。

                **骨架不变是本课的暗线**：你已经悄悄从"会训练一个模型"进化到
                "会训练一类模型"——只要换掉 ① 前向和 ② 损失，四步曲就是一台通用训练机。
                后面的多层网络、甚至 Transformer，骨子里都是这四步。
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

        五组演示，从"认识新装备"一路走到"完整训练 + 概率地图"：

        - **演示 1** · 认识 sigmoid：左看曲线，右看导数
        - **演示 2** · 造两团数据：80 个点，两团各据一方
        - **演示 3** · 手算一轮交叉熵：4 个样本，逐个开罚单
        - **演示 4** · 完整训练逻辑回归：四步曲 + 损失曲线 + 决策边界
        - **演示 5** · 概率地图：meshgrid 铺满特征平面，contourf 上色
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 1 · 认识 sigmoid

        z 从 -6 走到 6。左图画 σ(z) 本体——标出分界点 (0, 0.5)、两条渐近线 y=0 与 y=1
        （"渐近线"= 曲线无限贴近但永远碰不到的线：z 跑得越远，σ 离 0 或 1 越近，
        但 (0, 1) 是开区间，死话不说满），顺手钉上 σ(2) ≈ 0.88 和 σ(-2) ≈ 0.12
        两个点（练习 1 要手算它们）。

        右图画导数 σ'(z) = σ(z)·(1-σ(z))——概念 1 里"先记住形式"的公式，
        这里让你亲眼验证：**钟形、处处为正、峰值 0.25 恰好在 z=0**。
        导数最高才 0.25，这个数字后面讲深层网络时还会回来找我们。
        """
    )
    return


@app.cell
def _(np, plt):
    # 演示 1 · 认识 sigmoid：左看本体，右看导数
    def sigmoid(z):
        """概率翻译官：把任意实数打分 z 翻译成概率 p ∈ (0, 1)。第 3 部分全程在岗"""
        return 1.0 / (1.0 + np.exp(-z))          # e^(-z)：np.exp 今天首次亮相（求 e 的幂）

    _z = np.linspace(-6, 6, 400)                 # 打分范围 -6 ~ 6，均匀采 400 个点（03 课）
    _p = sigmoid(_z)                             # 翻译成概率：一条 S 形曲线
    _dp = _p * (1 - _p)                          # 导数 σ'(z) = σ(z)·(1-σ(z))——用 σ 的值直接拼

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # ---- 左图：sigmoid 本体 ----
    _ax1.plot(_z, _p, color="tab:blue", lw=2.5, label="σ(z) = 1/(1+e^(-z))")
    _ax1.axhline(0, color="gray", ls="--", lw=1, alpha=0.7)   # 渐近线 y=0：z→-∞ 时贴着它躺平
    _ax1.axhline(1, color="gray", ls="--", lw=1, alpha=0.7)   # 渐近线 y=1：z→+∞ 时贴着它
    _ax1.scatter([0], [0.5], s=70, color="tab:red", zorder=3)  # 分界点 (0, 0.5)：打平手，五五开
    _ax1.annotate("σ(0) = 0.5\n（打平手，五五开）", xy=(0, 0.5), xytext=(0.8, 0.30),
                  fontsize=10, color="tab:red",
                  arrowprops=dict(arrowstyle="->", color="tab:red"))
    # 顺手钉两个点：σ(2) 与 σ(-2)，一高一低恰好关于 (0, 0.5) 对称
    _ax1.scatter([2, -2], [sigmoid(2), sigmoid(-2)], s=45, color="tab:orange", zorder=3)
    _ax1.text(2.6, 0.78, "σ(2) ≈ 0.88", fontsize=9, color="tab:orange")
    _ax1.text(-5.8, 0.28, "σ(-2) ≈ 0.12", fontsize=9, color="tab:orange")
    _ax1.set_title("sigmoid：把任意打分压进 (0, 1)")
    _ax1.set_xlabel("打分 z（任意实数）")
    _ax1.set_ylabel("概率 σ(z)")
    _ax1.legend(loc="upper left")
    _ax1.grid(True, alpha=0.3)

    # ---- 右图：导数 σ'(z) = σ(z)·(1-σ(z)) ----
    _ax2.plot(_z, _dp, color="tab:green", lw=2.5, label="σ'(z) = σ(z)·(1-σ(z))")
    _ax2.scatter([0], [0.25], s=70, color="tab:red", zorder=3)  # 峰值：z=0 处导数最大，也只有 0.25
    _ax2.annotate("峰值 0.25 在 z=0\n（越接近五五开，学得越快）", xy=(0, 0.25),
                  xytext=(1.6, 0.20), fontsize=10, color="tab:red",
                  arrowprops=dict(arrowstyle="->", color="tab:red"))
    _ax2.set_title("导数：钟形，处处为正，最高才 0.25")
    _ax2.set_xlabel("打分 z")
    _ax2.set_ylabel("σ'(z)")
    _ax2.legend(loc="upper right")
    _ax2.grid(True, alpha=0.3)

    _fig
    return (sigmoid,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 2 · 造两团数据

        分类要有分类的数据。`rng.normal((2.0, 2.0), 0.8, size=(40, 2))` 一次生成
        40 个点：中心 (2, 2)、两个坐标各自独立地撒 0.8 标准差的高斯噪声——
        10 课的 default_rng 老配方，种子固定 7，每次运行长出同一批点。

        两团各 40 个点，拼成 **X_clf (80, 2)**（每行一个坐标）和
        **y_clf (80,)**（每行一个 0/1 真值）。这份数据是本课的练武场：
        演示 4 训练它、演示 5 给它上色、交互实验拧门槛，全用这一份。
        """
    )
    return


@app.cell
def _(np, plt):
    # 演示 2 · 造两团数据：分类问题的练武场
    _rng = np.random.default_rng(7)              # 固定种子 7：每次运行同一批点（10 课的老规矩）

    # 类 0：40 个点，中心 (2, 2)。loc=(2, 2) 会广播到两列——每个坐标独立撒 0.8 标准差噪声
    _X0 = _rng.normal((2.0, 2.0), 0.8, size=(40, 2))
    # 类 1：40 个点，中心 (5, 5)，同样的散布（两队人马，装备对等）
    _X1 = _rng.normal((5.0, 5.0), 0.8, size=(40, 2))

    # 拼成训练数据：X (80, 2) 每行一个点的坐标 (x1, x2)；y (80,) 每行一个真值 0/1
    X_clf = np.vstack([_X0, _X1])                # 上下摞起来：(40,2) + (40,2) → (80,2)
    y_clf = np.concatenate([np.zeros(40), np.ones(40)])   # 前 40 行真值 0，后 40 行真值 1
    # 自检一下形状再往下走（写 NumPy 的好习惯：shape 不对，后面全白搭）
    assert X_clf.shape == (80, 2) and y_clf.shape == (80,)

    _fig, _ax = plt.subplots(figsize=(6.5, 6))
    # 两类各画各的：布尔索引先挑行、再取列（05 课 X[mask, 0] 的老手势）
    _ax.scatter(X_clf[y_clf == 0, 0], X_clf[y_clf == 0, 1],
                s=45, color="tab:blue", alpha=0.8, label="类别 0")
    _ax.scatter(X_clf[y_clf == 1, 0], X_clf[y_clf == 1, 1],
                s=45, color="tab:red", alpha=0.8, label="类别 1")
    # 两团的"司令部"：中心点打星标，一眼看出阵地在哪
    _ax.scatter([2, 5], [2, 5], marker="*", s=320, color="gold",
                edgecolor="k", zorder=3, label="两团中心 (2,2) 与 (5,5)")

    _ax.set_xlim(-0.5, 7)
    _ax.set_ylim(-0.5, 7.5)
    _ax.set_aspect("equal")                      # 等比例：几何图的生命线（11 课的叮嘱）
    _ax.grid(True, alpha=0.3)
    _ax.legend(loc="upper left")
    _ax.set_xlabel("特征 x1")
    _ax.set_ylabel("特征 x2")
    _ax.set_title("两团各 40 个点：一刀能分开吗？")
    _fig
    return (X_clf, y_clf)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        读图：两团中心相距 √(3² + 3²) ≈ 4.24，各自的标准差只有 0.8——
        **一条斜线就能把两团分开**（术语：线性可分）。

        但两团边缘并不规整：0.8 的标准差让零星点探出主阵地，
        尤其沿着对角线方向（那正是两类贴脸的方向）——这就是后面
        "门槛一低就有误伤"的伏笔。星标是两团的中心 (2, 2) 与 (5, 5)。

        顺手回答一个小疑问：这 80 个点**没有打散**（前 40 行全是 0 类，后 40 行全是 1 类），
        要紧吗？不要紧——我们的梯度是"全批"下降：每步都对 80 个样本取平均，
        加法不讲究顺序，平均出来的梯度一模一样。（第 17 课切训练/验证/测试集时，
        打乱就变成生死攸关的事了——到时候细说。）
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 3 · 手算一轮交叉熵

        4 个样本：2 个真值 y=1、2 个真值 y=0，配 4 个预测概率
        （0.90 / 0.05 / 0.30 / 0.05）。上图是两条**罚单曲线**：
        蓝线 -log(p)（y=1 的样本走这条）、红虚线 -log(1-p)（y=0 的样本走这条）——
        两条线关于 p=0.5 镜像对称，越往两端越陡（往死里罚）。
        4 个样本各踩在哪条线上、挨多重的罚，一眼看清；
        竖直的点线是 p=0.05——**同一个押注、两种命运**的分界见证线。

        下方的表格逐样本对账，最后 mean 收尾——公式
        L = -mean(y·log(p) + (1-y)·log(1-p)) 从抽象符号变成手里的算盘。
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # 演示 3 · 手算一轮交叉熵：4 个样本，逐个开罚单
    _y4 = np.array([1.0, 1.0, 0.0, 0.0])         # 真值：前两个样本是 1 类，后两个是 0 类
    _p4 = np.array([0.90, 0.05, 0.30, 0.05])     # 模型给出的预测概率（手工编排，先练手感）

    # 完整公式逐样本展开：y=1 的样本只剩 -log(p) 项，y=0 的样本只剩 -log(1-p) 项
    # （另一项乘上 y 或 (1-y) = 0，自动消失——公式自带"分拣"）
    _per = -(_y4 * np.log(_p4) + (1 - _y4) * np.log(1 - _p4))   # 4 张罚单
    _mean4 = _per.mean()                          # 交叉熵 = 罚单的平均值

    # ---- 罚单曲线：两条 -log，一条管 y=1、一条管 y=0 ----
    _pq = np.linspace(0.01, 0.99, 400)            # 扫一遍预测概率（端点 0/1 处 log 会炸，避开）
    _fig, _ax = plt.subplots(figsize=(7.2, 5))
    _ax.plot(_pq, -np.log(_pq), color="tab:blue", lw=2.2,
             label="-log(p)：真值 y=1 走这条")
    _ax.plot(_pq, -np.log(1 - _pq), color="tab:red", lw=2.2, ls="--",
             label="-log(1-p)：真值 y=0 走这条")
    _ax.axvline(0.05, color="gray", ls=":", lw=1.2)          # 同一个 p=0.05，两种命运
    # 把 4 个样本钉到各自的曲线上（前两个在蓝线上，后两个在红线上）
    _ax.scatter([0.90, 0.05], [-np.log(0.90), -np.log(0.05)], s=65,
                color="tab:blue", zorder=3)
    _ax.scatter([0.30, 0.05], [-np.log(0.70), -np.log(0.95)], s=65,
                color="tab:red", zorder=3)
    _ax.annotate("y=1, p=0.05\n罚 3.0（自信答错，往死里罚）", xy=(0.05, -np.log(0.05)),
                 xytext=(0.20, 3.75), fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="tab:blue"))
    _ax.annotate("y=0, p=0.05\n罚 0.05（自信答对，几乎白给）", xy=(0.05, -np.log(0.95)),
                 xytext=(0.16, 0.62), fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="tab:red"))
    _ax.set_ylim(0, 5)
    _ax.set_xlabel("预测概率 p")
    _ax.set_ylabel("单样本罚金")
    _ax.set_title("罚单曲线：同一个 p=0.05，两种命运")
    _ax.legend(loc="upper center")
    _ax.grid(True, alpha=0.3)

    # ---- 逐样本对账表（多行插值，markdown 顶格写才能正确渲染成表格）----
    _notes = ["自信答对，轻罚", "自信答错，重罚", "偏对了但不敢肯定", "自信答对，几乎白给"]
    _rows = "\n".join(
        f"| {_i + 1} | {int(_y4[_i])} | {_p4[_i]:.2f} | **{_per[_i]:.3f}** | {_notes[_i]} |"
        for _i in range(4)
    )

    mo.vstack([
        _fig,
        mo.md(
            f"""
| # | 真值 y | 预测 p | 罚金（损失） | 判词 |
|---|---|---|---|---|
{_rows}

四张罚单取平均：L = **{_mean4:.3f}**——这一轮预测总共挨了多少罚。
盯住第 2、4 行：**同样是押 p=0.05，真值是 1 罚 2.996，真值是 0 只罚 0.051**——
交叉熵只问一句话：真值那边，你押了多少？
            """
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 4 · 完整训练逻辑回归

        四步曲全功率运转：80 个样本、w 从 (0, 0) 出发、b 从 0 出发、
        学习率 0.1（沿 15 课"每份数据单独调"的方法试出：交叉熵配 sigmoid，
        梯度尺度更温和，安全区比 15 课宽）、走 600 步。
        循环体内四步注释标号 ①②③④——和 15 课逐行对照，你会发现
        **只有 ① 前向和 ② 损失换了装备**，③④ 一字不改。

        唯一的"新数学"是 ③ 里那行 `X_clf.T @ (_p - y_clf) / _n`——
        概念 4 的漂亮公式 mean((p-y)·x) 的矩阵写法：残差 (p-y) 是 (80,) 向量，
        X 转置 (2, 80) 乘它得 (2,)——一次乘法同时算出两个权重的梯度
        （09 课 X.T @ v 的老配方）。推导看练习 3 参考答案，这里先享受结果。

        左图看损失曲线：起点 0.693 = ln 2——这不是巧合，w 全 0 时人人 p = 0.5，
        交叉熵恰好罚出"瞎猜的代价" log 2（演示 3 罚单表中间那行）；
        之后一路平滑下滑到 0.127。右图看战果：两团散点 + 黑色决策边界，
        标题报告准确率——**80 个点，全部判对**。
        """
    )
    return


@app.cell
def _(mo, np, plt, sigmoid, X_clf, y_clf):
    # 演示 4 · 完整训练逻辑回归：15 课的四步曲，原封不动再跑一遍
    _n = X_clf.shape[0]                          # 样本数 80（02 课 shape）
    w_final = np.zeros(2)                        # 权重从 (0, 0) 出发：白纸起步
    b_final = 0.0                                # 偏置从 0 出发
    _lr = 0.1                                    # 学习率 0.1：沿 15 课"单独调"的方法试出，交叉熵梯度温和
    _steps = 600                                 # 走 600 步：80 个点的小场子，秒级跑完

    _losses = []                                 # 每一步的损失都记下来（15 课的损失曲线配方）
    for _step in range(_steps):
        # ① 前向：先打分 z = X @ w + b，再翻译成概率 p = σ(z)（09 课 X @ w 一次算 80 个）
        _p = sigmoid(X_clf @ w_final + b_final)  # (80,) 每个样本一个概率

        # ② 损失：交叉熵——对 80 张罚单取平均
        _losses.append(-np.mean(y_clf * np.log(_p) + (1 - y_clf) * np.log(1 - _p)))

        # ③ 梯度：概念 4 的漂亮公式 dL/dw = mean((p-y)·x)、dL/db = mean(p-y)
        _grad_w = X_clf.T @ (_p - y_clf) / _n    # (2,) 一行搞定：X 转置乘残差再平均（09 课）
        _grad_b = np.mean(_p - y_clf)            # 标量

        # ④ 更新：沿梯度反方向挪一小步（10 课埋种子，15 课发芽，本课继续长个）
        w_final = w_final - _lr * _grad_w
        b_final = b_final - _lr * _grad_b

    # ---- 训练完毕，验收 ----
    _p_fin = sigmoid(X_clf @ w_final + b_final)  # 最终概率 (80,)
    _pred = (_p_fin >= 0.5).astype(float)        # 判决规则：p ≥ 0.5 判 1 类（概念 3）
    _acc = np.mean(_pred == y_clf)               # 准确率 = 判对的比例（bool 数组求均值）

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # ---- 左图：损失曲线 ----
    _ax1.plot(_losses, color="tab:blue", lw=2)
    _ax1.scatter([0, _steps - 1], [_losses[0], _losses[-1]], s=55, color="tab:red", zorder=3)
    _ax1.text(12, 0.715, "起点 0.693 = ln 2\n（w 全 0 → 人人 p=0.5，等于瞎猜）", fontsize=9)
    _ax1.text(588, 0.165, f"终点 {_losses[-1]:.3f}", fontsize=9, ha="right")
    _ax1.set_ylim(0, 0.8)
    _ax1.set_xlabel("训练步数")
    _ax1.set_ylabel("交叉熵损失")
    _ax1.set_title("损失曲线：平滑下降（15 课同款走势）")
    _ax1.grid(True, alpha=0.3)

    # ---- 右图：训练好的决策边界 ----
    _ax2.scatter(X_clf[y_clf == 0, 0], X_clf[y_clf == 0, 1],
                 s=40, color="tab:blue", alpha=0.8, label="类别 0")
    _ax2.scatter(X_clf[y_clf == 1, 0], X_clf[y_clf == 1, 1],
                 s=40, color="tab:red", alpha=0.8, label="类别 1")
    # 决策边界 w1·x1 + w2·x2 + b = 0：移项解出 x2 = -(w1·x1 + b) / w2，连成一条直线
    _x1_line = np.linspace(-0.5, 7, 60)
    _x2_line = -(w_final[0] * _x1_line + b_final) / w_final[1]
    _ax2.plot(_x1_line, _x2_line, color="k", lw=2.2, label="决策边界（p = 0.5）")
    _ax2.set_xlim(-0.5, 7)
    _ax2.set_ylim(-0.5, 7.5)
    _ax2.set_aspect("equal")
    _ax2.grid(True, alpha=0.3)
    _ax2.legend(loc="lower right")
    _ax2.set_xlabel("特征 x1")
    _ax2.set_ylabel("特征 x2")
    _ax2.set_title(f"训练 600 步后：准确率 {_acc:.0%}")

    mo.vstack([
        _fig,
        mo.md(
            f"""
            读图：左——损失从 0.693（= ln 2，瞎猜的代价）一路滑到 **{_losses[-1]:.3f}**，
            前 100 步就降到 0.38，第 300 步约 0.21，全程没有震荡：学习率 0.1 配 80 个点，
            步子迈得刚刚好（15 课的调参直觉直接迁移）。

            右——学到的参数 w = **({w_final[0]:.2f}, {w_final[1]:.2f})**、
            b = **{b_final:.2f}**，边界线恰好从两团中间穿过
            （与对角线的交点约在 (3.09, 3.09)），**80 个点全部判对**。
            """
        ),
    ])
    return (b_final, w_final)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 5 · 概率地图

        决策边界只是一条线，但边界两侧其实在**处处报价**：特征平面上每个点
        都可以问一句"这里有多像 1 类？"。13 课的 meshgrid 铺一张 220×220 的网格，
        每个格点算 p = σ(w1·x1 + w2·x2 + b)，再让 contourf 按
        [0, 0.25, 0.5, 0.75, 1] 四档上色（RdYlGn 配色：越绿越像 1 类）。

        看图三件事：底色从左下的红渐变到右上的绿；**加粗黑线（p = 0.5 等值线）
        就是演示 4 的决策边界**——概率场这座"山"的山脊线；右侧色条是
        颜色↔数值的对照表（11 课 colorbar 的老朋友）。如果说 11 课的 imshow
        是把矩阵画成格子，contourf 就是把一整片概率场画成等高线农场。

        再盯一处细节：靠近边界的点颜色暧昧（泡在 0.25~0.75 的过渡带里），
        深入腹地的点才染上浓红浓绿——模型对"边境居民"留了余地
        （练习 3 的梯度接近 0 但不是 0，也是同一个"余地"）。
        这个余地，正是交互实验里拧门槛的空间。
        """
    )
    return


@app.cell
def _(b_final, np, plt, sigmoid, w_final, X_clf, y_clf):
    # 演示 5 · 概率地图：边界两侧，处处报价
    _gx = np.linspace(-0.5, 7, 220)              # x1 方向 220 个格子（03 课 linspace）
    _gy = np.linspace(-0.5, 7.5, 220)            # x2 方向 220 个格子
    _XX, _YY = np.meshgrid(_gx, _gy)             # 13 课 meshgrid：铺一张 220×220 的网格

    # 每个格点算一次概率：p = σ(w1·x1 + w2·x2 + b)——广播一次算完 48400 个格子（06 课功力）
    _PP = sigmoid(w_final[0] * _XX + w_final[1] * _YY + b_final)

    _fig, _ax = plt.subplots(figsize=(6.8, 6.2))
    # 四档概率底色：0~0.25 / 0.25~0.5 / 0.5~0.75 / 0.75~1，淡扫一层（alpha=0.25）
    _cf = _ax.contourf(_XX, _YY, _PP, levels=[0, 0.25, 0.5, 0.75, 1],
                       cmap="RdYlGn", alpha=0.25)
    _fig.colorbar(_cf, ax=_ax, label="预测概率 p")          # 色条：颜色↔数值对照表
    # 0.5 等值线描黑加粗——它就是决策边界（呼应 13 课的 contour）
    _ax.contour(_XX, _YY, _PP, levels=[0.5], colors="k", linewidths=2)

    # 两团散点盖回地图上：谁站在红区、谁站在绿区，一目了然
    _ax.scatter(X_clf[y_clf == 0, 0], X_clf[y_clf == 0, 1],
                s=38, color="tab:blue", alpha=0.9, label="类别 0", zorder=3)
    _ax.scatter(X_clf[y_clf == 1, 0], X_clf[y_clf == 1, 1],
                s=38, color="tab:red", alpha=0.9, label="类别 1", zorder=3)

    # 两块"地契"：哪一侧归谁
    _ax.text(0.30, 0.40, "这一侧 p < 0.5\n判 0 类", fontsize=10, color="tab:blue",
             ha="center")
    _ax.text(4.30, 6.80, "这一侧 p > 0.5\n判 1 类", fontsize=10, color="tab:red",
             ha="center")
    _ax.set_xlim(-0.5, 7)
    _ax.set_ylim(-0.5, 7.5)
    _ax.set_aspect("equal")                      # 等比例：格子是正方形，地图不失真
    _ax.grid(True, alpha=0.3)
    _ax.legend(loc="lower right")
    _ax.set_xlabel("特征 x1")
    _ax.set_ylabel("特征 x2")
    _ax.set_title("概率地图：0.5 等值线就是决策边界")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 🎛️ 交互实验

        **判决门槛**——模型已经训练完毕（演示 4 的 w_final、b_final 原封不动），
        现在把最后一道工序交给你：**p ≥ 多少才判 1 类？** 默认 0.5，你可以从
        0.10 一路拧到 0.90。盯三处变化：

        1. **判错的点**（黑色大 X）：门槛越低，越多 0 类点被"误伤"（错判成 1）；
           门槛越高，越多 1 类点"漏网"（漏成 0）
        2. **紫色虚线**：p = 门槛 的等值线随手门移动——黑色实线（p = 0.5）纹丝不动，
           动的是你的判决标准，不是模型
        3. **数值表**：准确率、误伤数、漏网数实时刷新

        两个极端预告（先算好）：门槛拧到 0.10 → **26 个 0 类点被误伤**，
        准确率掉到 67.5%；拧到 0.90 → **8 个 1 类点漏网**，准确率 90%。
        而在 0.40 ~ 0.60 的舒适区里，两类一个不错——先想想为什么中间这么宽？
        （提示：看演示 5 的地图，两团泡在哪个色带里。）

        **教学点：门槛是业务旋钮，不是训练出来的。** 训练只负责给出概率 p；
        把 p 折成"是/否"的那一刀，由业务拍板。垃圾邮件过滤可以"宁可错杀"——
        门槛调低，多拦几封正常邮件的代价不过是去垃圾箱翻一翻，放过一封钓鱼邮件
        的代价可能是一整台电脑；人群癌症初筛则常被主张"宁可放过"——门槛调高，
        把健康人错判成"疑似"会带来过度检查与虚惊，而初筛之后还有复诊兜底。
        （这对错误各有学名：误伤 = 假阳性，漏网 = 假阴性——以后聊评估指标时再会。）

        建议按三步玩：

        1. 拧到 **0.10**：看误伤大军——紫色虚线压进 0 类腹地，"判 1"的地盘扩大，
           蓝色点成片被判红、挂上 X
        2. 拧到 **0.90**：看漏网——紫色虚线缩回 1 类腹地，"判 1"的地盘缩小，
           红色点被放走一片
        3. 回到 **0.50** 附近：找到最宽的"安全带"，并对照数值表确认两类全对
        """
    )
    return


@app.cell
def _(mo):
    # 交互控制：判决门槛（debounce=True：松手才重算，散点 + 等值线重画不闪烁）
    threshold_slider = mo.ui.slider(0.1, 0.9, value=0.5, step=0.05, debounce=True,
                                    label="🎛️ 判决门槛：p ≥ 多少才判 1 类？（0.10 ~ 0.90）")
    threshold_slider
    return (threshold_slider,)


@app.cell
def _(b_final, mo, np, plt, sigmoid, threshold_slider, w_final, X_clf, y_clf):
    # ===== 交互实验 · 判决门槛（响应单元）=====
    _thr = threshold_slider.value                 # 当前门槛（0.10 ~ 0.90）

    _p = sigmoid(X_clf @ w_final + b_final)       # 概率场纹丝不动——模型早已训练完毕
    _pred = (_p >= _thr).astype(float)            # 用新门槛重新判决：p ≥ 门槛 才判 1 类

    _n_correct = int(np.sum(_pred == y_clf))      # 判对个数
    _acc = _n_correct / len(y_clf)                # 准确率
    _miss0 = int(np.sum((_pred == 1) & (y_clf == 0)))   # 误伤：0 类被硬判成 1
    _miss1 = int(np.sum((_pred == 0) & (y_clf == 1)))   # 漏网：1 类被放走成 0

    # ---- 画图：按"新预测"着色；判错的点打黑色大 X ----
    _fig, _ax = plt.subplots(figsize=(6.8, 6))
    _ax.scatter(X_clf[_pred == 0, 0], X_clf[_pred == 0, 1],
                s=42, color="tab:blue", alpha=0.8, label="判为 0 类")
    _ax.scatter(X_clf[_pred == 1, 0], X_clf[_pred == 1, 1],
                s=42, color="tab:red", alpha=0.8, label="判为 1 类")
    _wrong = _pred != y_clf                       # 判错的位置（布尔索引，05 课）
    if _wrong.any():                              # 有错才画 X（全对时不占图例）
        _ax.scatter(X_clf[_wrong, 0], X_clf[_wrong, 1],
                    marker="x", s=170, color="k", lw=2.5, zorder=4, label="判错的点")

    # 两条参考线：p=0.5 决策边界（实线，不动）+ p=门槛 等值线（虚线，随手门走）
    _gx = np.linspace(-0.5, 7, 200)               # 13 课 meshgrid 再就业
    _gy = np.linspace(-0.5, 7.5, 200)
    _XX, _YY = np.meshgrid(_gx, _gy)
    _PP = sigmoid(w_final[0] * _XX + w_final[1] * _YY + b_final)
    _ax.contour(_XX, _YY, _PP, levels=[0.5], colors="k", linewidths=2)
    _ax.contour(_XX, _YY, _PP, levels=[_thr], colors="purple",
                linewidths=2, linestyles="--")

    from matplotlib.lines import Line2D as _Line2D   # 图例替身：让两条等值线也进图例
    _ax.legend(handles=[
        _Line2D([0], [0], color="k", lw=2, label="p=0.5 决策边界（不动）"),
        _Line2D([0], [0], color="purple", lw=2, ls="--",
                label=f"p={_thr:.2f} 等值线（随手门移动）"),
        *_ax.get_legend_handles_labels()[0],      # 再接上散点们的图例
    ], loc="lower right")

    _ax.set_xlim(-0.5, 7)
    _ax.set_ylim(-0.5, 7.5)
    _ax.set_aspect("equal")
    _ax.grid(True, alpha=0.3)
    _ax.set_xlabel("特征 x1")
    _ax.set_ylabel("特征 x2")
    _ax.set_title(f"门槛 = {_thr:.2f}：准确率 {_acc:.1%}")

    # ---- 判词：门槛的三种典型状态（按实际错误数裁决，不冤枉好人）----
    _verdict = (
        f"门槛太低：大面积误伤 0 类——{_miss0} 个 0 类点被硬判成 1"
        if _miss0 > _miss1
        else f"门槛太高：1 类漏网——{_miss1} 个 1 类点被放走"
        if _miss1 > _miss0
        else "0.5 附近最平衡：两类各自守土，一个不伤"
    )

    mo.vstack([
        _fig,
        mo.md(
            f"""
            | 指标 | 值 |
            |---|---|
            | 当前门槛 | **{_thr:.2f}** |
            | 准确率 | **{_acc:.1%}**（80 个点判对 {_n_correct} 个） |
            | 误伤：0 类被错判成 1 | **{_miss0} 个** |
            | 漏网：1 类被错判成 0 | **{_miss1} 个** |
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
                **练习 1 · sigmoid 手算**：σ(0) = ?、σ(2) ≈ ?、σ(-2) ≈ ?
                σ(0) 代公式就能出；σ(2) 按一下计算器（e^(-2) ≈ 0.135）；
                σ(-2) **不许按计算器**——用对称性 σ(-z) = 1 - σ(z) 白拿。
                填完看验证行：σ(2) + σ(-2) 应该恰好 = 1。

                **练习 2 · 交叉熵手算**：同一个预测 p = 0.7，两个样本真值不同——
                甲：y = 1 → 损失 -log(0.7) ≈ ?；乙：y = 0 → 损失 -log(0.3) ≈ ?
                算完体会一句话："同一个 0.7，对不同真值，罚度天差地别"。

                **练习 3 · 挑战 · 数值梯度验证**：概念 4 的梯度公式
                mean((p - y)·x) 漂亮得让人不敢信。对训练好的 w_final 的第 0 维，
                用 13 课的**对称差分**（h = 1e-5）算数值梯度，与公式对照
                np.allclose——三层复合（sigmoid → 交叉熵 → 对 w）链式法则的实战验收。
                做对了你会拿到一个令人安心的 True。

                规则不变：先把 `None` 换成你的答案再运行；写完再看参考答案。
                """
            )
        }
    )
    return


@app.cell
def _(mo, np, sigmoid):
    # ===== 练习 1 · sigmoid 手算 =====
    # 三问：σ(0) = ?、σ(2) ≈ ?、σ(-2) ≈ ?
    # 提示 1：σ(0)——e^0 = 1，分母 = 1 + 1 = 2，所以 σ(0) = 1/2
    # 提示 2：σ(2)——e^(-2) ≈ 0.1353，1/1.1353 ≈ ?
    # 提示 3：σ(-2) 不用重算！对称性 σ(-z) = 1 - σ(z)：算出 σ(2) 就白拿 σ(-2)

    sig_ex1 = None    # ← 填 np.array([σ(0), σ(2), σ(-2)])，三个数按此顺序

    _ref = np.array([sigmoid(0.0), sigmoid(2.0), sigmoid(-2.0)])   # 代码算的标准答案
    _msg = (
        f"你的答案：σ(0) = **{sig_ex1[0]:.4f}**、σ(2) = **{sig_ex1[1]:.4f}**、"
        f"σ(-2) = **{sig_ex1[2]:.4f}**\n\n"
        f"对称性对账：σ(2) + σ(-2) = **{sig_ex1[1] + sig_ex1[2]:.4f}**（应恰好 = 1）；"
        f"标准答案对账 np.allclose → **{np.allclose(sig_ex1, _ref)}**"
        if sig_ex1 is not None
        else "⏳ 先在纸上手算三个值，再把 sig_ex1 的 None 换成 np.array([...])"
    )
    mo.md(_msg)
    return (sig_ex1,)


@app.cell
def _(mo, np):
    # ===== 练习 2 · 交叉熵手算 =====
    # 同一个预测 p = 0.7，两个样本真值不同：
    #   甲样本：y = 1，模型押 p = 0.7 → 损失 = -log(0.7) ≈ ?
    #   乙样本：y = 0，模型押 p = 0.7 → 损失 = -log(1 - 0.7) = -log(0.3) ≈ ?
    # 提示：np.log(0.7) ≈ -0.3567，np.log(0.3) ≈ -1.2040（演示 3 的罚单曲线上找位置）

    ce_ex2 = None    # ← 填 np.array([甲的损失, 乙的损失])

    _ref = np.array([-np.log(0.7), -np.log(0.3)])   # 标准答案 ≈ [0.357, 1.204]
    _msg = (
        f"甲（y=1，p=0.7）：罚 **{ce_ex2[0]:.4f}**；"
        f"乙（y=0，p=0.7）：罚 **{ce_ex2[1]:.4f}**\n\n"
        f"同一个预测 0.7，乙比甲重 **{ce_ex2[1] / ce_ex2[0]:.1f} 倍**——"
        f"对账 np.allclose → **{np.allclose(ce_ex2, _ref)}**"
        if ce_ex2 is not None
        else "⏳ 手算两张罚单，再把 ce_ex2 的 None 换成 np.array([甲的损失, 乙的损失])"
    )
    mo.md(_msg)
    return (ce_ex2,)


@app.cell
def _(b_final, mo, np, sigmoid, w_final, X_clf, y_clf):
    # ===== 练习 3 · 挑战：数值梯度验证 =====
    # 概念 4 说 dL/dw = mean((p - y)·x)——不信公式？用 13 课的对称差分亲手验一个：
    #
    #   数值梯度 = ( L(w0 + h) - L(w0 - h) ) / (2h)，h = 1e-5
    #   公式梯度 = mean((p - y) · X[:, 0])
    #
    # 提示：把损失写成一个小函数 _loss(w0_test)——只动 w 的第 0 维，其余原封不动

    def _loss(_w0_test):
        _w_try = w_final.copy()                   # 复制一份参数，别动原模型
        _w_try[0] = _w0_test                      # 只改第 0 维
        _p_try = sigmoid(X_clf @ _w_try + b_final)         # 前向：打分 + 翻译
        return -np.mean(y_clf * np.log(_p_try) + (1 - y_clf) * np.log(1 - _p_try))   # 交叉熵

    _h = 1e-5                                     # 扰动步长：太小浮点 noise，太大不近似
    num_grad_ex3 = None    # ← 对称差分：(_loss(w_final[0] + _h) - _loss(w_final[0] - _h)) / (2 * _h)

    _ana = np.mean((sigmoid(X_clf @ w_final + b_final) - y_clf) * X_clf[:, 0])   # 公式梯度
    _msg = (
        f"数值梯度 = **{num_grad_ex3:.6f}**，公式梯度 = **{_ana:.6f}**\n\n"
        f"np.allclose 对账：**{np.allclose(num_grad_ex3, _ana)}**"
        if num_grad_ex3 is not None
        else "⏳ 把 num_grad_ex3 的 None 换成对称差分公式（提示都在上面的注释里）"
    )
    mo.md(_msg)
    return (num_grad_ex3,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 1 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                sig_ex1 = np.array([0.5, 0.8808, 0.1192])
                # 或者干脆现算：np.array([sigmoid(0), sigmoid(2), sigmoid(-2)])
                ```

                手算过程：

                - σ(0)：e^0 = 1 → 1/(1+1) = **0.5**（打平手，五五开）
                - σ(2)：e^(-2) ≈ 0.1353 → 1/1.1353 ≈ **0.8808**
                - σ(-2)：不许按计算器！σ(-2) = 1 - σ(2) = 1 - 0.8808 = **0.1192**

                第三问就是"中心对称"的实战：σ(2) + σ(-2) = 1 恰好成立——
                曲线关于 (0, 0.5) 中心对称的代数说法，就是 σ(-z) = 1 - σ(z)。
                以后见到 σ(-3) 这种，先算 σ(3) 再用 1 减，又快又不容易错。

                再送一个口算技巧：σ(z) 与赔率（odds）是亲戚——
                σ(2) ≈ 0.88 意味着"押 1 类的机会大约是押 0 类的 7 倍"
                （0.88/0.12 ≈ 7.3）。z 每增加 1，赔率就翻约 e ≈ 2.7 倍——
                这就是"打分每涨一分，底气翻一倍多"的直观含义。
                """
            ),
            "✅ 练习 2 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                ce_ex2 = np.array([-np.log(0.7), -np.log(0.3)])   # ≈ [0.357, 1.204]
                ```

                - 甲（y=1，p=0.7）：走 -log(p) = -log(0.7) ≈ **0.357**——
                  方向对但不敢肯定，小罚
                - 乙（y=0，p=0.7）：走 -log(1-p) = -log(0.3) ≈ **1.204**——
                  方向错还押了大注，罚 3.4 倍

                **同一个预测 0.7，对不同真值，罚度天差地别。** 公式里的 y 和 (1-y)
                是两个开关：y=1 只留 -log(p) 项，y=0 只留 -log(1-p) 项。
                回忆演示 3 的罚单曲线：p=0.7 在蓝线上很浅（0.357）、在红线上已经爬坡
                （1.204）——预测离真值越远，你站的那条曲线罚得越狠。
                """
            ),
            "✅ 练习 3 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                _h = 1e-5
                num_grad_ex3 = (_loss(w_final[0] + _h) - _loss(w_final[0] - _h)) / (2 * _h)
                # ≈ -0.006269，与公式 mean((p - y) · X[:, 0]) 分毫不差，allclose = True
                ```

                数值梯度对上了——但为什么两个数都这么接近 0？训练 600 步后，
                模型已经躺在"谷底附近"：两团数据本来就分得开，损失还能再降一点点，
                只是坡度几乎没了。（一个小真相：对线性可分的数据，交叉熵真正的谷底
                在 w→∞ 的无穷远处——模型会越走越"自信"。这团火气，
                正是下一课"过拟合"话题的引子之一。）

                敢挑战推导的看这里——14 课链式法则的三层总装（单样本版，
                z = w·x + b、p = σ(z)）：

                ```
                第一层  交叉熵对 p：   dL/dp = -y/p + (1-y)/(1-p)
                第二层  sigmoid 对 z：  dp/dz = p·(1-p)
                相乘：(-y/p + (1-y)/(1-p)) · p(1-p) = -y(1-p) + (1-y)·p = p - y
                                              ↑ 分母恰好被约掉！
                第三层  z 对 w：       dz/dw = x
                总装：dL/dw = mean((p - y) · x)
                ```

                两层"难看的导数"（带分式的交叉熵 + 带指数的 sigmoid）相乘，
                居然约得只剩 (p - y)——这不是运气，是交叉熵与 sigmoid 的"天作之合"
                （概念 2 末尾提过：MSE 配 sigmoid 就没这种好事）。
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
                1. **分类输出概率**：答案是"属于哪一类"，最自然的语言是 p ∈ (0, 1)；
                   sigmoid(z) = 1/(1+e^(-z)) 把任意打分压进 (0, 1)——
                   z=0 → 0.5，z 越大越接近 1，但永远不说 0/1 的死话
                2. **sigmoid 的形状与性质**：S 形曲线、渐近线 y=0 与 y=1；
                   σ(-z) = 1-σ(z)（关于 (0, 0.5) 中心对称）；
                   导数 σ'(z) = σ(z)·(1-σ(z))，峰值 0.25 在 z=0
                3. **交叉熵 = "说错话的重罚"**：L = -mean(y·log(p) + (1-y)·log(1-p))；
                   -log 罚单表要刻在脑子里：押 0.99 罚 0.010、押 0.5 罚 0.693、
                   押 0.01 罚 4.605——自信地错，罚得最狠
                4. **决策边界是一条直线**：p ≥ 0.5 ⟺ w1·x1 + w2·x2 + b ≥ 0，
                   边界即 w·x + b = 0；15 课的直线换了身份——不再是预测值，
                   而是两类领地的分界线（且与 w 垂直，w 指向 1 类腹地）
                5. **梯度惊人简洁**：dL/dw = mean((p-y)·x)——链式法则三层
                   （交叉熵 → sigmoid → w）相乘、分母恰好约掉；
                   练习 3 的数值梯度已经替你验过真
                6. **四步曲骨架原封不动**——这就是框架的力量，从线性回归到
                   Transformer 都逃不出这四步；而**阈值是业务旋钮**：
                   训练给概率，业务定门槛（垃圾邮件宁可错杀，癌症初筛宁可放过）
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

        **下一课预告** — `17_过拟合与数据划分`：

        到今天为止，你手里已经有了两台能跑的模型：线性回归（15 课）和逻辑回归（本课）。
        但一个幽灵正在逼近：**训练集上 100 分、新数据上翻车**——它叫**过拟合**。

        下一课，把数据切成三份（训练 / 验证 / 测试），亲眼看一台 9 次多项式
        如何把训练点"背"下来——连噪声都一字不落地背！——以及怎么在它疯掉之前
        踩下刹车。你会学到深度学习工程里最常说的一句话：
        **"先看验证集，再说话。"**

        🧭 概率翻译官已上岗，去把数据切开，看看真相！
        """
    )
    return


if __name__ == "__main__":
    app.run()
