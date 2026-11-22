import marimo

__generated_with = "0.25.2"
app = marimo.App(app_title="15 · 线性回归", width="medium")


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
        # 15 · 线性回归

        **从零到 Transformer · 手写深度学习** ｜ 第 3 部分 · 经典机器学习热身（1/3）
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📖 本课目标（先读我）": mo.md(
                """
                **这一课只解决一个问题：怎么让一个模型"学会"一件事？**

                第 2 部分收官时我们预告过：数学三件套已经集齐——几何画面（11~12 课）、
                微积分方向（13 课）、链式传动（14 课）。第 3 部分真刀真枪：把这些零件
                组装成全课程**第一个完整的训练循环**，亲眼看损失一路下降、
                直线自己学会贴上数据——亲手养第一个会学习的模型。

                换句话说：前 14 课你造好了所有零件，今天装出第一台能跑的机器。
                **这是里程碑的一课。**

                学完你将能：

                - 用 `default_rng` 造一批"真值 + 噪声"的仿真数据（10 课技能回收）
                - 说出 MSE 损失为什么要平方，并把损失想象成 (w, b) 头顶的一只"碗"（13 课画面回收）
                - 用 14 课的链式法则两行推出 dL/dw 与 dL/db——本课唯一的手推公式
                - 独立写出"**前向 → 损失 → 梯度 → 更新**"四步曲训练循环——后面 30 课的骨架
                - 看懂损失曲线（模型的心电图），理解学习率 = 步长：太小磨蹭、太大崩盘

                提前打个预防针：**今天几乎不学新数学**。碗是 13 课的碗，链式法则
                是 14 课的法则，画图是 11 课的手艺——新东西只有"组装"本身。
                而组装，恰恰是从"懂数学"到"会训练"之间最宽的那条沟。

                前置：第 13 课（梯度下降与对称差分）、第 14 课（链式法则）、
                第 10 课（default_rng / 种子 / normal）、第 11 课（matplotlib 散点与折线）。
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "🎯 概念 1/4 · 机器学习的第一性问题：从数据里找规律": mo.md(
                """
                给一堆成对的数据点 (x, y)，找出 y 随 x 变化的规律——这是机器学习的
                **第一性问题**。房价随面积、气温随日期、销售额随广告费……本质都是它。

                最简模型：**y = w·x + b**——一条直线，只有两个参数：

                - **w**（weight，权重）：直线的斜率，x 每涨 1，y 涨 w
                - **b**（bias，偏置）：直线在 y 轴上的起跳高度

                为什么从它开始？因为它是最小可用的"有参数模型"：参数只有两个，
                一眼看穿；但"学习"的全部本质已经齐了——**学习 = 不断微调 w、b，
                让直线的输出尽量贴合数据**。Transformer 有上亿个参数，
                做的事和这俩一模一样，只是拧的旋钮多了几千万倍。

                顺带解惑：明明是拟合直线，为什么叫"回归"（regression）？
                词源是 19 世纪高尔顿研究父母身高与子女身高：高个子父母的孩子
                往往矮一点、矮个子父母的孩子往往高一点——身高会"回归"平均值。
                名字留下了，含义早已泛化成"从数据里估计规律"。

                一个术语先立此存照：w、b 是模型**自己学**出来的，叫**参数**；
                稍后登场的"学习率"是**你替它定**的，叫**超参数**——所谓"调参调参"，
                调的从来是超参数；参数是训练出来的，不是拍脑袋写的。
                """
            ),
            "📏 概念 2/4 · 损失函数 MSE：错多少、罚多少": mo.md(
                """
                模型好不好，得有个数来打分。回归问题的默认打分器是
                **MSE（均方误差，Mean Squared Error）**：

                ```
                L = mean( (w·x + b - y)² )        （对 60 个点取平均）
                ```

                对每个点算"预测 - 真实"的误差，平方，再取平均。
                **为什么要平方？** 三个理由，一个比一个实在：

                - **大错重罚**：误差 4 的惩罚（16）是误差 2（4）的四倍——离谱的预测被重点盯防
                - **处处可导**：|误差| 在 0 点有个尖角不可导（练习 2 会亲身体验），
                  平方处处光滑，14 课的求导工具随便用
                - **好算**：平方、平均都是 NumPy 一行——"好算"在深度学习里是硬通货

                两个阅读提示：MSE 的单位是 y 的**平方**（本课是"y 的平方"），
                单看数值没意义，要看相对变化（480 → 2.26 才是故事）；另外，
                **损失为 0 未必是好事**——那往往意味着连噪声都背下来了，
                17 课（过拟合）会专门讲这个甜蜜的陷阱。

                损失越小，直线越贴。把损失看成一个**碗**（13 课的画面正式回收）：
                **w 和 b 是地板上的两个坐标，损失 L 是头顶的高度**。
                我们这只碗，碗口高度约 480（w = b = 0 处），碗底约 2.24——
                梯度下降就是让小球从碗口滚向碗底。
                """
            ),
            "🏂 概念 3/4 · 梯度下降：13 课的预告今天兑现": mo.md(
                """
                13 课的更新公式 x ← x - lr·∇f，今天正式上岗——只是 x 换成了参数 (w, b)：

                ```
                w ← w - lr · dL/dw        （lr = learning rate，学习率）
                b ← b - lr · dL/db
                ```

                两个梯度从哪来？14 课链式法则的**两环推导**，一眼看完。
                记误差 u = w·x + b - y，则 L = mean(u²)：

                - **外环（平方）**：d(u²)/du = 2u
                - **内环（线性）**：u 对 w 的导数是 x，对 b 的导数是 1

                两环相乘，再对 60 个点取平均：

                ```
                dL/dw = mean( 2x·(w·x + b - y) )
                dL/db = mean( 2·(w·x + b - y) )
                ```

                这两行是全课程出场率最高的公式之一，建议背下来。

                为什么是**减**号？13 课的旧账：梯度指向"上坡最快"的方向，
                我们要下坡，所以反着走——负号就是"往低处走"的数学写法。
                为什么不一步跳到谷底？因为我们**不知道**谷底在哪——只知道
                脚下的坡度。看一步走一步，每步都变好，就是梯度下降的全部哲学。

                lr 是**学习率**——每步沿反梯度方向挪多远，是超参数。
                本课经验值 0.02：这批数据安全区内偏上沿的顺手值（爆炸线在 0.03 附近，演示 5 见分晓）。
                """
            ),
            "🔁 概念 4/4 · 训练循环四步曲：全课程的骨架": mo.md(
                """
                把上面的零件按顺序组装，就是深度学习唯一的训练流程——**四步曲**：

                **① 前向**：用当前参数算预测（y_pred = w·x + b）
                **② 损失**：算预测错多少（MSE）
                **③ 梯度**：算参数该往哪修（两个偏导）
                **④ 更新**：修一点点（w ← w - lr·dL/dw，b 同理）

                然后回到 ①，循环几百上千次。

                再预发两个术语（后面课程天天见）：①② 合称**前向传播**——
                数据从输入流到损失；③ 的批量计算版叫**反向传播**——
                梯度从损失流回参数。四步曲是骨架，这两个词是骨架各段的官名。

                请把这张骨架**背下来**：后面 30 课——MLP、CNN、RNN、Transformer——
                从头到尾都是这四步，一步不多一步不少；变的只有"前向"越来越深
                （一条直线 → 万层网络），以及"梯度"越来越长（链式法则越串越长）。
                今天你手写的 20 行循环，就是未来所有大模型训练循环的微缩模型。
                从下一节开始，把这四拍在心里默念着读代码——直到它变成呼吸。
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

        五组演示，正好是"养一个模型"的完整生命周期：

        - **演示 1** · 造数据：60 个带噪声的点（真值 y = 3x + 5）
        - **演示 2** · 第一轮手动挡：w = 0、b = 0 起步，亲手踩一脚油门
        - **演示 3** · 完整训练 300 步：四步曲转起来——全课程第一张损失下降图
        - **演示 4** · 学习快进：四联快照，看直线从平躺到贴上数据
        - **演示 5** · 学习率的三种命运：太慢 / 正好 / 崩盘

        一条观看主线：演示 2 是"一步"，演示 3 是"三百步"，演示 4 是"回放"，
        演示 5 是"换挡"——四者共用同一份数据、同一个模型、同一套梯度公式。
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 1 · 造数据：60 个带噪声的点

        真实世界不会把真值给你，但教学可以"作弊"：自己造一批**已知真值**的数据。
        配方三行——x 在 0~10 均匀撒 60 个点；真值直线 y = 3x + 5；
        再叠加 σ = 2 的正态噪声（10 课的 `rng.normal`）。

        为什么要加噪声？噪声让数据"不完美"，模型才有东西可学——
        没有噪声，"拟合"就退化成解方程，一丝悬念都没有。
        σ = 2 也刻意选得不小：散点会明显散在真值线两侧，肉眼能看出
        "规律 + 扰动"的两层结构——待会儿的损失地板就藏在 σ 里。

        为什么造数据也用种子？10 课的老规矩：固定 `default_rng(42)`，
        所有人看到**同一批点**，讲义里的每个数字（480、93、2.26……）
        你都能亲手复现——新手课，数字对不上是灾难。

        这批 `x_data`、`y_data` 后面每个单元都要复用（跨单元变量）。
        """
    )
    return


@app.cell
def _(np, plt):
    # 演示 1 · 造数据：真值 y = 3x + 5，噪声 σ = 2，60 个点
    # 全课程的"训练数据"第一次登场——后面 4 个演示、交互实验、3 道练习，
    # 全都踩在这 60 个点上
    _rng = np.random.default_rng(42)             # 固定种子（10 课：复现的开关）
    x_data = _rng.uniform(0, 10, 60)             # 60 个 x：0~10 均匀撒点，shape (60,)
    y_data = 3 * x_data + 5 + _rng.normal(0, 2, 60)   # 真值直线 + 正态噪声，shape (60,)
    # 注意两行都向同一个 _rng 要数：先 uniform 后 normal，顺序也是"数据"的一部分
    # ——换个顺序，噪声就换了一批（10 课：发生器按调用顺序吐数）

    _fig, _ax = plt.subplots(figsize=(7, 4.5))
    _ax.scatter(x_data, y_data, s=28, alpha=0.75, color="tab:blue",
                label="数据点（60 个，含噪声）")     # 11 课的散点图
    _x_line = np.linspace(0, 10, 50)             # 画直线用的采点（03 课 linspace）
    _ax.plot(_x_line, 3 * _x_line + 5, "--", color="gray", lw=1.5,
             label="真值直线 y = 3x + 5")          # 灰虚线：答案先亮出来
    _ax.set_xlabel("x")
    _ax.set_ylabel("y")
    _ax.set_title("造数据：真值 y = 3x + 5 + 噪声（σ = 2）")
    _ax.legend()
    _ax.grid(True, alpha=0.3)
    _fig
    return (x_data, y_data)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 2 · 第一轮手动挡：从零踩一脚油门

        先不当司机，下车站着推一次车，感受动力：**w = 0、b = 0 起步**——
        直线平躺在 y = 0 上，60 个预测全错。走一遍四步曲的 ①②③ 步
        算出损失和梯度，然后只踩一脚油门（更新一次），看新直线长什么样。

        剧透三个数（都能对上）：起步损失 ≈ 480；梯度 dL/dw ≈ -257
        （负的——碗往"加大 w"的方向低）；一脚油门后 w 从 0 冲到 ≈ 5.1——
        **冲过头了**（真值才 3）！但损失确实掉到 ≈ 93。

        梯度下降从不指望一步到位，它靠"每步都往低处走"慢慢逼近。
        手动挡的意义在于：**下一节把这几行套上 for，就是全部的深度学习**。
        现在去代码里数一数四步曲的注释。
        """
    )
    return


@app.cell
def _(mo, np, plt, x_data, y_data):
    # 演示 2 · 手动挡：从 w = 0, b = 0 出发，只更新一次
    _w, _b = 0.0, 0.0                     # 起步参数：直线 y = 0·x + 0（平躺在 x 轴上）
    _lr = 0.02                            # 学习率（本课经验值）

    # ===== 第 1 步 · 前向：算 60 个预测（此刻全错）=====
    _pred = _w * x_data + _b              # 60 个 0——直线还平躺着

    # ===== 第 2 步 · 损失：MSE =====
    _loss0 = np.mean((_pred - y_data) ** 2)            # ≈ 480.17

    # ===== 第 3 步 · 梯度：两个偏导（14 课链式法则的成品）=====
    _dw = np.mean(2 * x_data * (_pred - y_data))       # ≈ -257.09
    _db = np.mean(2 * (_pred - y_data))                # ≈ -40.36

    # ===== 第 4 步 · 更新：踩一脚油门 =====
    _w1 = _w - _lr * _dw                  # 0 - 0.02×(-257.09) ≈ 5.14（冲过头！）
    _b1 = _b - _lr * _db                  # 0 - 0.02×(-40.36)  ≈ 0.81
    _loss1 = np.mean((_w1 * x_data + _b1 - y_data) ** 2)   # ≈ 92.97
    # 为什么会冲过头？负梯度 × 学习率 = 步长 ≈ 5.14，而最优的 w 是 3——
    # 步子比"到最优的距离"还大。没关系：下一步的梯度会自动变号往回拉，
    # 这正是"一步比一步准"的机制（演示 3 里你会看到它来回修正、越荡越小）。

    _fig, _ax = plt.subplots(figsize=(7, 4.5))
    _ax.scatter(x_data, y_data, s=28, alpha=0.75, color="tab:blue", label="数据点")
    _x_line = np.linspace(0, 10, 50)
    _ax.plot(_x_line, np.zeros_like(_x_line), "--", color="gray", lw=2,
             label="第 0 步：y = 0（起步，全错）")        # w=0,b=0 → 恒为 0 的水平线
    _ax.plot(_x_line, _w1 * _x_line + _b1, color="tab:orange", lw=2.5,
             label=f"更新 1 次后：y = {_w1:.2f}x + {_b1:.2f}")   # 一脚油门后的直线
    _ax.set_xlabel("x")
    _ax.set_ylabel("y")
    _ax.set_title("手动挡第一轮：一步就有进步")
    _ax.legend()
    _ax.grid(True, alpha=0.3)

    mo.vstack([
        _fig,
        mo.md(
            f"""
            | 量 | 值 | 备注 |
            |---|---|---|
            | 起步损失（w=0, b=0） | **{_loss0:.2f}** | 碗口高度 |
            | 梯度 dL/dw | **{_dw:.2f}** | 负 → 该往"加大 w"的方向走 |
            | 梯度 dL/db | **{_db:.2f}** | 同上 |
            | 一脚油门后的 w | **{_w1:.2f}** | 0 + 0.02×257.09，冲过了真值 3 |
            | 一脚油门后的 b | **{_b1:.2f}** | |
            | 一脚油门后的损失 | **{_loss1:.2f}** | 480 → 93，一步就有进步 |
            """
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 3 · 完整训练 300 步：名场面来了

        手动挡踩过一脚，现在挂自动挡：把四步曲放进 `for` 循环，转 300 圈。
        每一步的损失记进列表——画出来就是**全课程第一张损失下降图**。

        从今天起，判断"模型好不好"你都会先看这张图：它是模型的心电图。
        同时把每一步的 w、b 也存进历史列表（演示 4 要回放）。

        数据规模刻意小（60 个点、300 步）：眨眼跑完，注意力留给循环本身。
        读代码的方法：先找到 `for` 循环，再数注释块——**四块，一块不多，
        一块不少**；300 步里的每一步，都在原地重演这四拍。
        """
    )
    return


@app.cell
def _(mo, np, plt, x_data, y_data):
    # 演示 3 · 完整训练循环：四步曲 × 300 步（全课程第一次完整运行）
    # 和演示 2 唯一的区别：外面套了一层 for——深度学习的全部秘密就这么多
    _w, _b = 0.0, 0.0        # 起步参数（和演示 2 一样，从零开始）
    _lr = 0.02               # 学习率：本课经验值（演示 5 讲它为什么金贵）

    losses = []              # losses[k] = 第 k+1 步开头的损失（即更新 k 次后的损失）
    w_hist = [_w]            # w_hist[k] = 更新 k 次后的 w（演示 4 回放用）
    b_hist = [_b]            # b_hist 同理

    for _step in range(300):
        # ===== 第 1 步 · 前向：用当前 w、b 算 60 个预测 =====
        _pred = _w * x_data + _b

        # ===== 第 2 步 · 损失：MSE（预测离数据有多远）=====
        _loss = np.mean((_pred - y_data) ** 2)
        losses.append(_loss)

        # ===== 第 3 步 · 梯度：损失对两个参数的偏导（14 课链式法则成品）=====
        _dw = np.mean(2 * x_data * (_pred - y_data))
        _db = np.mean(2 * (_pred - y_data))

        # ===== 第 4 步 · 更新：沿反梯度方向挪一小步（13 课的心跳）=====
        _w = _w - _lr * _dw
        _b = _b - _lr * _db
        w_hist.append(_w)    # 存快照：每一步的参数都留底
        b_hist.append(_b)

    w_final = _w             # 训练结束的参数（练习 1 要在这个点上验梯度）
    b_final = _b
    _final_loss = np.mean((w_final * x_data + b_final - y_data) ** 2)   # ≈ 2.26

    # ---- 左图：全课程第一张损失下降图 ----
    # 线性轴：头 10 步的悬崖式俯冲是这个量级里最震撼的画面
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    _ax1.plot(losses, color="tab:blue", lw=2)
    _ax1.scatter([0, len(losses) - 1], [losses[0], losses[-1]],
                 color="tab:red", zorder=3, s=30)          # 标出起点和终点
    _ax1.annotate(f"起点 {losses[0]:.1f}", xy=(0, losses[0]), xytext=(35, 430),
                  arrowprops=dict(arrowstyle="->", color="gray"))
    _ax1.annotate(f"终点 {losses[-1]:.2f}", xy=(299, losses[-1]), xytext=(180, 90),
                  arrowprops=dict(arrowstyle="->", color="gray"))
    _ax1.set_xlabel("训练步数")
    _ax1.set_ylabel("损失（MSE）")
    _ax1.set_title("损失下降曲线（全课程第一张，名场面）")
    _ax1.grid(True, alpha=0.3)

    # ---- 右图：最终拟合 vs 真值 ----
    # 橙线是学到的，灰虚线是出题答案——肉眼看它们几乎重合的那一刻，
    # 就是"模型学会了"四个字的画面版
    _ax2.scatter(x_data, y_data, s=24, alpha=0.7, color="tab:blue", label="数据点")
    _x_line = np.linspace(0, 10, 50)
    _ax2.plot(_x_line, w_final * _x_line + b_final, color="tab:orange", lw=2.5,
              label=f"学到：y = {w_final:.2f}x + {b_final:.2f}")
    _ax2.plot(_x_line, 3 * _x_line + 5, "--", color="gray", lw=1.5,
              label="真值：y = 3x + 5")
    _ax2.set_xlabel("x")
    _ax2.set_ylabel("y")
    _ax2.set_title("300 步后的拟合结果")
    _ax2.legend()
    _ax2.grid(True, alpha=0.3)
    _fig.tight_layout()

    mo.vstack([
        _fig,
        mo.md(
            f"""
            | 量 | 训练前（第 0 步） | 训练后（第 300 步） | 真值 |
            |---|---|---|---|
            | w | 0 | **{w_final:.3f}** | 3 |
            | b | 0 | **{b_final:.3f}** | 5 |
            | 损失 | 480.17 | **{_final_loss:.3f}** | —（地板 ≈ 2.24） |
            """
        ),
    ])
    return (b_final, b_hist, losses, w_final, w_hist)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        读图：左边的损失曲线是深度学习的"标准脸"——**头 10 步悬崖式俯冲**
        （480 → 5.5），**之后 290 步一条长尾巴贴地慢磨**（5.5 → 2.26）。
        往后 30 课你会认出这张脸几百次：先陡后缓，越来越慢，渐入佳境。

        心电图的三看法，今天一并教给你：一看**走势**（一路向下 = 训练健康；
        往上翘 = 出事了）；二看**地板**（降不动了 = 到了这批数据的极限）；
        三看**陡缓**（太缓 = 学习率可能偏小，太抖 = 可能偏大）。

        右边：学到的直线 y = 3.03x + 4.36 和真值 3x + 5 肉眼几乎重合。
        w 只差 0.03；b 差 0.64——**b 学得慢不是意外**：b 的梯度天生比 w 温和
        （少了 x 的放大——x 平均 5.2），步子小，自然到得晚。练习 3 会看到
        它 1000 步之后能到多近。

        一个伏笔：300 步后损失 2.26，离"地板 2.24"只差 0.02。这个地板是
        **数据里的噪声**决定的，模型再练也压不下去——练习 3 揭晓为什么。
        （顺势记住一对术语：训练到 2.26 叫"收敛慢一点"；想冲破 2.24 才停
        叫"过拟合"——现在只需混个眼熟。）
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 4 · 学习快进：四联快照

        损失曲线是"心电图"，四联快照是"成长录像"：演示 3 已经把每一步的
        w、b 存进了 `w_hist`、`b_hist`，现在回放第 1 / 30 / 120 / 300 步的直线——
        看它从平躺（损失 93）、到立起来（4.5）、到贴上数据（2.7）、
        到几乎与真值重合（2.26）。

        特别留意**斜率先到位、截距后到位**：第 30 步 w 已经 3.46（很准了），
        b 才 1.5（还差得远）——梯度大的参数先收敛，梯度小的参数慢慢赶。
        这个现象在深度学习里极为常见，后来甚至有专门的自适应学习率方法
        （Adam，第 4 部分的优化器一课）来平衡快慢参数。

        "训练时存快照、事后回放"这个手法本身也值得学：后面课程画
        特征演化、注意力漂移，用的都是同一招。
        """
    )
    return


@app.cell
def _(b_hist, np, plt, w_hist, x_data, y_data):
    # 演示 4 · 学习快进：回放第 1 / 30 / 120 / 300 步的直线
    # 快照语义：第 k 步 = 更新 k 次之后——和 w_hist[k] 的下标严格对应
    _snap_steps = [1, 30, 120, 300]        # 四个回放时刻（= 更新次数）
    _x_line = np.linspace(0, 10, 50)

    _fig, _axes = plt.subplots(2, 2, figsize=(11, 8))
    for _ax, _k in zip(_axes.flat, _snap_steps):
        _wk = w_hist[_k]                   # 第 k 次更新后的 w（演示 3 存的历史）
        _bk = b_hist[_k]
        _lk = np.mean((_wk * x_data + _bk - y_data) ** 2)   # 当步损失（现场重算）
        _ax.scatter(x_data, y_data, s=18, alpha=0.6, color="tab:blue")
        _ax.plot(_x_line, _wk * _x_line + _bk, color="tab:orange", lw=2.5)
        # 标题即成绩单：斜率、截距、损失，一步到位
        _ax.set_title(f"第 {_k} 步：w = {_wk:.2f}，b = {_bk:.2f}，损失 = {_lk:.2f}")
        _ax.set_xlabel("x")
        _ax.set_ylabel("y")
        _ax.grid(True, alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 5 · 学习率的三种命运

        四步曲里唯一"你说了算"的数就是 lr。同一份数据、同一个模型、同样 300 步，
        只换 lr，命运分三岔（数值都已替你跑好，可对照）：

        - **lr = 0.002（太慢）**：损失也降，但磨蹭——300 步还悬在 4.5，
          离地板 2.24 差一倍；要追上 0.02 的进度得练约 3000 步
        - **lr = 0.02（正好）**：平滑俯冲，300 步贴到 2.26——本课经验值
        - **lr = 0.08（崩盘）**：第 2 步损失就破万，第 5 步冲破 10⁹——**发散**，
          参数像在碗里越弹越高的皮球

        为什么太大就崩？直觉：步子跨过了碗底，跳到对面更高的坡壁上，
        下一步更狠地弹回来——误差越弹越大，直到数值爆炸。
        这批数据的爆炸线在 0.03 附近（交互实验里可以亲手找到它）。

        （三张图都用**对数轴**：发散时数字横跨好几个数量级，
        线性轴会把前两种命运糊成一条线。）
        """
    )
    return


@app.cell
def _(mo, np, plt, x_data, y_data):
    # 演示 5 · 学习率的三种命运：0.002 / 0.02 / 0.08 各跑 300 步
    _lr_list = [0.002, 0.02, 0.08]        # 同一份数据、同一个模型，只换步长
    _fate_names = ["太慢", "正好", "崩盘"]
    _curves = []                                 # 三条损失曲线

    for _lr in _lr_list:
        _w, _b = 0.0, 0.0
        _curve = []
        for _step in range(300):
            _pred = _w * x_data + _b                              # ① 前向
            _loss = np.mean((_pred - y_data) ** 2)                # ② 损失
            _curve.append(_loss)
            if (not np.isfinite(_loss)) or (_loss > 1e9):         # 护栏：爆炸就刹车
                break                                             #   （否则会溢出到 inf）
            _dw = np.mean(2 * x_data * (_pred - y_data))          # ③ 梯度
            _db = np.mean(2 * (_pred - y_data))
            _w = _w - _lr * _dw                                   # ④ 更新
            _b = _b - _lr * _db
        _curves.append(_curve)

    _fig, _axes = plt.subplots(1, 3, figsize=(12, 4.2))
    # 三张图同一套坐标逻辑（对数轴），方便横向对比；
    # 灰色点线是地板 2.24——第一张够不着、第二张贴住、第三张飞上天
    for _ax, _curve, _lr, _name in zip(_axes, _curves, _lr_list, _fate_names):
        _ax.plot(_curve, color="tab:blue", lw=2)
        _ax.set_yscale("log")                    # 对数轴：横跨数量级才看得清
        _ax.axhline(2.24, color="gray", ls=":", lw=1.2)          # 噪声地板（伏笔）
        _ax.set_title(f"lr = {_lr}（{_name}）")
        _ax.set_xlabel("步数")
        _ax.set_ylabel("损失（对数轴）")
        _ax.grid(True, alpha=0.3)
    _axes[0].legend(["损失", "地板 ≈ 2.24"], fontsize=9)
    _axes[2].text(0.42, 0.18, f"第 {len(_curves[2]) - 1} 步爆表\n护栏拦停",
                  transform=_axes[2].transAxes, ha="center", color="tab:red",
                  fontsize=11)
    _fig.tight_layout()

    mo.vstack([
        _fig,
        mo.md(
            f"""
            | 学习率 | 300 步后的损失 | 命运 |
            |---|---|---|
            | 0.002 | **{_curves[0][-1]:.2f}** | 磨蹭：离地板还差一倍，要约 3000 步才追上 |
            | 0.02 | **{_curves[1][-1]:.2f}** | 正好：平滑下降，贴到地板 |
            | 0.08 | 第 {len(_curves[2]) - 1} 步冲破 10⁹ | 崩盘：发散，护栏拦停 💥 |

            一句话：**步子太小磨蹭、步子正好稳降、步子太大崩盘**。
            lr 没有万能值——换一份数据（尺度不同）安全区就挪窝，
            所以每次实战都要重新调。实战调法也送你：**从大往小试**——
            先挑一个让损失明显下降的值，再减半微调；一旦见到损失上翘，
            立刻回退。这是 15 课最重要的一节实践课。
            """
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 🎛️ 交互实验

        **学习率调音台**——同一份数据，你来当调音师。两个旋钮：

        - **学习率**（0.001 ~ 0.1）与**训练步数**（50 ~ 600）

        响应单元会用当前设置**现场重跑训练**（毫秒级），给出损失曲线 +
        拟合直线 + 成绩单 + 一句判词。四个任务：

        1. 从 0.001 慢慢往上拧：找到"刚好开始发散"的临界值
           （对照演示 5 的结论：爆炸线在 0.03 附近）
        2. 在安全区里微调：多小的 lr 会吃到"太保守"的判词？
        3. 固定 lr = 0.02，把步数从 50 拉到 600：长尾巴还能再挤出多少？
        4. 综合冠军赛：在"不发散"的前提下，哪组 (lr, 步数) 能最快贴到地板？

        发散时损失会爆到天文数字——代码里装了护栏（损失 > 10⁹ 提前刹车，
        防止溢出到 inf/nan 把图和表弄坏），图上会标注"被护栏拦下"。

        两个滑块都开了 **debounce**（11 课交互的老规矩）：拖动过程中不重算，
        松手才触发——不然每一格都跑一遍训练，画面会闪成幻灯片。
        """
    )
    return


@app.cell
def _(mo):
    # 交互控制：学习率 + 步数（debounce=True：松手才重算，画面不闪烁）
    # 学习率用线性步进 0.001：范围只有 0.001~0.1，线性足够细，直觉更好拧
    lr_slider = mo.ui.slider(0.001, 0.1, value=0.02, step=0.001, debounce=True,
                             label="🎛️ 学习率 lr（0.001 ~ 0.1）")
    steps_slider = mo.ui.slider(50, 600, value=300, step=50, debounce=True,
                                label="🎛️ 训练步数（50 ~ 600，步进 50）")
    mo.vstack([lr_slider, steps_slider])
    return (lr_slider, steps_slider)


@app.cell
def _(lr_slider, mo, np, plt, steps_slider, x_data, y_data):
    # ===== 交互实验 · 学习率调音台（响应单元）=====
    _lr = lr_slider.value                    # 当前学习率
    _steps = steps_slider.value              # 当前步数

    _w, _b = 0.0, 0.0
    _losses = []                             # 本轮的损失曲线（临时变量，_ 前缀）
    _diverged = False
    for _step in range(_steps):
        _pred = _w * x_data + _b                               # ① 前向
        _loss = np.mean((_pred - y_data) ** 2)                 # ② 损失
        _losses.append(_loss)
        if (not np.isfinite(_loss)) or (_loss > 1e9):          # 护栏：爆炸就刹车
            _diverged = True
            break
        _dw = np.mean(2 * x_data * (_pred - y_data))           # ③ 梯度
        _db = np.mean(2 * (_pred - y_data))
        _w = _w - _lr * _dw                                    # ④ 更新
        _b = _b - _lr * _db

    _final_loss = _losses[-1]
    _loss_str = "爆表（> 10⁹，护栏拦停）💥" if _diverged else f"{_final_loss:.3f}"
    # 判词的门槛参考：地板 2.24（这批数据的极限），演示 5 的三种命运各落一档

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    _ax1.plot(_losses, color="tab:blue", lw=2)
    _ax1.set_yscale("log")                   # 对数轴：发散时也能看清走势
    _ax1.axhline(2.24, color="gray", ls=":", lw=1.2, label="地板 ≈ 2.24")
    _ax1.set_xlabel("训练步数")
    _ax1.set_ylabel("损失（对数轴）")
    _ax1.set_title(f"损失曲线（lr = {_lr}，{_steps} 步）")
    _ax1.grid(True, alpha=0.3)
    _ax1.legend(fontsize=9)
    if _diverged:
        _ax1.text(0.5, 0.2, "损失在爆涨——发散！", transform=_ax1.transAxes,
                  ha="center", color="tab:red", fontsize=12)

    _ax2.scatter(x_data, y_data, s=24, alpha=0.7, color="tab:blue", label="数据点")
    _x_line = np.linspace(0, 10, 50)
    if not _diverged:
        _ax2.plot(_x_line, _w * _x_line + _b, color="tab:orange", lw=2.5,
                  label=f"y = {_w:.2f}x + {_b:.2f}")
        _ax2.set_title("拟合结果")
    else:
        _ax2.text(0.5, 0.5, "损失爆表\n参数已飞出画面", transform=_ax2.transAxes,
                  ha="center", va="center", color="tab:red", fontsize=13)
        _ax2.set_title("拟合结果（发散）")
    _ax2.set_xlabel("x")
    _ax2.set_ylabel("y")
    _ax2.legend()
    _ax2.grid(True, alpha=0.3)
    _fig.tight_layout()

    # 一句话判词（四档）：发散 / 贴地板 / 在下降 / 太保守
    # 门槛不是铁律，是给新手一个"下一步往哪调"的方向感
    if _diverged:
        _verdict = "发散了！损失在爆涨——步子太大，快把学习率调小"
        _kind = "danger"
    elif _final_loss <= 2.8:
        _verdict = "稳步下降，几乎贴到噪声地板（≈ 2.24）——这组设置很健康"
        _kind = "success"
    elif _final_loss <= 4.0:
        _verdict = "在下降，但还没贴到地板——试试大一点的学习率，或多练几百步"
        _kind = "info"
    else:
        _verdict = "太保守了，损失下不去——把学习率往上调一档试试"
        _kind = "warn"

    mo.vstack([
        _fig,
        mo.md(
            f"""
            | 量 | 值 |
            |---|---|
            | 学习率 lr | **{_lr}** |
            | 训练步数 | **{_steps}** |
            | 最终损失 | **{_loss_str}** |
            | 最终 w | **{_w:.3f}**（真值 3，差 {abs(_w - 3):.3f}） |
            | 最终 b | **{_b:.3f}**（真值 5，差 {abs(_b - 5):.3f}） |
            """
        ),
        mo.callout(mo.md(f"**实时判词：{_verdict}**"), kind=_kind),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✏️ 动手练习（每课必做，本课 3 题）": mo.md(
                """
                **练习 1 · 数值梯度验证（13 课技能回收）**：训练结束后，
                用对称差分 (L(w+h, b) - L(w-h, b)) / 2h 算出 dL/dw 的数值版，
                与公式 mean(2x·(wx+b-y)) 对照 `np.allclose`——亲手确认我们
                手推的梯度没有错。提示：h = 1e-4。

                **练习 2 · 换损失：MSE → MAE**：把损失换成平均绝对误差
                mean|wx+b-y|，梯度会变成什么样？把 `mae_grad_w_ex2` 补完
                再重训 300 步，观察收敛风格和 MSE 版有什么不同。提示：d|e|/de = sign(e)。

                **练习 3 · 探索：1000 步能离真值多近？**：自选学习率练 1000 步，
                把最好的 w、b 填进 `best_w_ex3` / `best_b_ex3`。它们能到 3 和 5 吗？
                为什么永远差一点点？（这道题的答案，是 17 课的伏笔。）

                三道题的定位：第 1 题验证"梯度的对"，第 2 题体验"损失的选"，
                第 3 题看清"极限在哪"——恰好是训练循环的三个灵魂拷问。

                规则不变：先把 `None` 换成你的答案再运行；写完再看参考答案。
                """
            )
        }
    )
    return


@app.cell
def _(b_final, mo, np, w_final, x_data, y_data):
    # ===== 练习 1 · 数值梯度验证 =====
    _h = 1e-4     # 扰动步长：太小会被浮点噪声淹没，太大截断误差高（13 课的老经验）

    # 公式版（解析解，14 课链式法则推出）——在训练终点 (w_final, b_final) 处取值：
    _ana_dw = np.mean(2 * x_data * (w_final * x_data + b_final - y_data))

    # 你的任务：对称差分算数值版 dL/dw
    #   L(w, b) = mean((w·x + b - y)²)   ← 就是训练用的 MSE
    #   数值梯度 = ( L(w_final + h, b_final) - L(w_final - h, b_final) ) / (2h)
    #   写法自由：可以先用两个临时变量存两次 L，再相减除以 2h
    num_grad_w_ex1 = None    # ← 把算出来的数填这里（一个 float）

    _msg = (
        f"公式版 dL/dw = **{_ana_dw:.6f}**，你的数值版 = **{num_grad_w_ex1:.6f}**，"
        f"np.allclose 判决：**{np.allclose(num_grad_w_ex1, _ana_dw)}** ✅ 两边一致，梯度手推无误"
        if num_grad_w_ex1 is not None
        else f"⏳ 公式版 dL/dw = **{_ana_dw:.6f}**（训练终点处梯度已接近 0——谷底几乎平了）。"
             f"用对称差分把 num_grad_w_ex1 的 None 补上（提示：L 要算两次，h = 1e-4）"
    )
    mo.md(_msg)
    return (num_grad_w_ex1,)


@app.cell
def _(mo, np, x_data, y_data):
    # ===== 练习 2 · 换损失：MSE → MAE =====
    def mae_grad_w_ex2(w, b, x, y):
        """MAE 损失 mean|w·x + b - y| 对 w 的梯度（返回一个数）。
        提示：d|e|/de = sign(e)（e≠0 时），再乘链式法则内层的 x，对 n 个点取 mean。"""
        return None    # ← 一行搞定（想想 sign 在 NumPy 里怎么写）

    # 训练循环已备好：同款四步曲，只是损失/梯度换成 MAE 版（lr 仍 0.02，300 步）
    # 注意：循环里没有第②步——更新只需要梯度，不需要损失本身；
    # 损失只在最后"打分汇报"时才算（训练循环里的损失常常只为监控而存）
    _probe = mae_grad_w_ex2(0.0, 0.0, x_data, y_data)   # 先探一下：None 说明还没填
    if _probe is not None:
        _w2, _b2 = 0.0, 0.0
        for _step in range(300):
            _pred = _w2 * x_data + _b2                             # ① 前向
            _dw = mae_grad_w_ex2(_w2, _b2, x_data, y_data)         # ③ 梯度（w 版——你写的）
            _db = np.mean(np.sign(_pred - y_data))                  #    b 版如法炮制，送给你
            _w2 = _w2 - 0.02 * _dw                                 # ④ 更新
            _b2 = _b2 - 0.02 * _db
        _mae_final = np.mean(np.abs(_w2 * x_data + _b2 - y_data))   # 用 MAE 打分
        _mse_final = np.mean((_w2 * x_data + _b2 - y_data) ** 2)    # 顺便看 MSE
        _msg = (
            f"补完运行后：w = **{_w2:.3f}**，b = **{_b2:.3f}**，最终 MAE = **{_mae_final:.3f}**；\n"
            f"按 MSE 打分则是 **{_mse_final:.3f}**（演示 3 的 MSE 训练是 2.26）——\n"
            f"换了损失就是换了目标，练出来的模型自然不一样。"
        )
    else:
        _msg = "⏳ 先把 mae_grad_w_ex2 里的 None 补上（提示：np.sign），再回来运行"
    mo.md(_msg)
    return (mae_grad_w_ex2,)


@app.cell
def _(mo, np, x_data, y_data):
    # ===== 练习 3 · 探索：1000 步能离真值多近？=====
    # 玩法：把演示 3 的循环复制一份（新单元或草稿），步数改成 1000，
    # 学习率自选（lr = 0.02 就够），跑完把最好的结果填进来。
    # 小建议：顺手把每步的损失也存下来画个图——看看 2.26 之后还能不能再降：
    best_w_ex3 = None    # ← 你 1000 步训出来的 w
    best_b_ex3 = None    # ← 你 1000 步训出来的 b

    # 参照系（填完后对照）：
    #   真值：              w = 3        b = 5      （造数据时亲手写的）
    #   这批数据的解析最优：w ≈ 2.990    b ≈ 4.640  （最小二乘精确解——梯度下降的极限）
    if best_w_ex3 is not None and best_b_ex3 is not None:
        _loss3 = np.mean((best_w_ex3 * x_data + best_b_ex3 - y_data) ** 2)
        _msg = f"""
            | 量 | 你的 1000 步 | 真值 | 解析最优 |
            |---|---|---|---|
            | w | **{best_w_ex3:.4f}** | 3 | 2.9896 |
            | b | **{best_b_ex3:.4f}** | 5 | 4.6400 |
            | 损失 | **{_loss3:.4f}** | — | 地板 2.2369 |
            """
    else:
        _msg = "⏳ 跑完你的 1000 步，把 best_w_ex3 / best_b_ex3 填上再看对照表"
    mo.md(_msg)
    return (best_b_ex3, best_w_ex3)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 1 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                num_grad_w_ex1 = (
                    np.mean(((w_final + 1e-4) * x_data + b_final - y_data) ** 2)
                    - np.mean(((w_final - 1e-4) * x_data + b_final - y_data) ** 2)
                ) / (2 * 1e-4)
                ```

                数值版 ≈ **0.018839**，公式版 ≈ **0.018839**——`np.allclose` 返回 **True**。

                如果你的 allclose 是 False，先查两处：是不是把 (w+h) 写成了
                (w+h·x)（括号位置错了）；是不是用了前向差分 (L(w+h)-L(w)) / h
                ——它也能对上前两三位，但精度差一截，对称差分才是正解。

                三个值得咂摸的细节：

                - 训练终点处梯度只有 ≈ 0.019，而起点处是 -257：**收敛 = 谷底坡度趋近 0**。
                  再练意义不大——这就是"够了"的数学信号
                - h = 1e-4 是甜点位：h 太小，两个 L 只差 4×10⁻⁶ 量级，浮点噪声（09 课）
                  会把差值淹没；h 太大，截断误差抬头——13 课试过的教训，这里正好回收
                - "数值梯度 vs 公式梯度"的对照是真实的工程调试手段：以后课程手写
                  反向传播，写错了就是这么抓出来的——这个技巧值得练到肌肉记忆
                """
            ),
            "✅ 练习 2 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                return np.mean(np.sign(w * x + b - y) * x)
                ```

                MAE 对 w 的梯度 = sign(误差)·x 的平均：**不看误差大小，只看方向**——
                错 0.1 和错 10，惩罚的推力一样大。跑满 300 步：w ≈ **3.41**、
                b ≈ **1.90**、MAE ≈ **1.63**；按 MSE 打分是 **3.95**，
                明显不如 MSE 训练的 2.26。两个观察：

                - **收敛风格不同**：MSE 的梯度含误差本身，越接近谷底推力越小，
                  像减震器慢慢刹住；MAE 推力恒定，像匀速步进的士兵——
                  看 w 的轨迹会发现它在 3.41 附近小幅"来回踱步"
                - **0 点不可导**：|e| 在 e = 0 有个尖角。NumPy 的 sign(0) = 0
                  恰好无害，训练照样能跑——工程上"几乎处处可导"就够用

                换损失 = 换目标：模型努力成为"MAE 意义上最好"的直线，而不是
                "MSE 意义上最好的"。**损失函数是你对"什么是错"的定义权**——
                同一份数据，换个打分器，就练出另一条"最优"直线；
                16 课换交叉熵，还是同一件事。
                """
            ),
            "✅ 练习 3 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                w, b = 0.0, 0.0
                for step in range(1000):                    # 只改这一处：300 → 1000
                    pred = w * x_data + b
                    dw = np.mean(2 * x_data * (pred - y_data))
                    db = np.mean(2 * (pred - y_data))
                    w -= 0.02 * dw
                    b -= 0.02 * db
                ```

                结果：w ≈ **2.9897**、b ≈ **4.6394**、损失 ≈ **2.2369**。
                和"解析最优"（最小二乘精确解 w* ≈ 2.9896、b* ≈ 4.6400，
                地板 2.2369）几乎重合——梯度下降 1000 步已经贴到了这批数据的极限。

                **但它们都到不了 3 和 5。** 为什么？两层天花板：

                - 这 60 个点里的噪声把"**这批数据的最优直线**"本身推离了真值：
                  解析最优 b* = 4.64 ≠ 5。巧的是，这批噪声的样本均值 ≈ -0.41，
                  正好把截距往下拽了约 0.4——你看到的差距不是训练不够，
                  是数据里的噪声在"说谎"
                - 损失地板 ≈ 2.24：就算 w、b 精确等于解析最优，平均平方误差
                  也是 2.24——**模型能学"规律"，学不会"噪声"**。噪声是天花板

                训练再久也只能逼近解析最优；解析最优离真值的偏差，是数据的
                固有局限，不是算法的锅。怎么区分"模型不行"和"数据到顶了"——
                17 课（过拟合与数据划分）正式回答。在那之前先记住这句话：
                **损失压不下去，未必是训练的错。**

                彩蛋：换一批种子（比如 default_rng(7)）重造数据，解析最优就会
                落在另一个位置——有时离 3、5 更近，有时更远。60 个点对付 σ=2 的
                噪声，就是这个精度量级；想要更准？多给数据（这也是 17 课的话头）。
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
                1. 模型 **y = w·x + b**：两个参数的世界；**学习 = 不断微调 w、b
                   让直线贴数据**；w、b 是参数（学出来的），学习率是超参数（你定的）
                2. **MSE = mean((wx+b-y)²)**：平方三连——大错重罚、处处可导、好算；
                   损失是 (w, b) 头顶的一只碗（13 课画面），本课的碗从 480 滚到 2.26
                3. 两个梯度公式请背：**dL/dw = mean(2x·(wx+b-y))**，
                   **dL/db = mean(2·(wx+b-y))**——14 课链式法则两环相乘的成品
                4. 四步曲骨架：**前向 → 损失 → 梯度 → 更新**，循环。后面 30 课
                   （MLP、CNN、RNN、Transformer）一步不多一步不少，只是前向越来越深
                5. **损失曲线是模型的心电图**：头几十步悬崖俯冲 + 长尾贴地；
                   三看法——看走势（健康与否）、看地板（到没到极限）、
                   看陡缓（步子大小合不合适）；梯度趋近 0 = 收敛 = "够了"的信号
                6. **学习率 = 步长**：太小磨蹭（0.002 三百步还悬在 4.5）、
                   太大崩盘（0.08 五步爆破 10⁹）；0.02 是我们这份数据的经验值，
                   换一份数据要重新调——实战口诀：从大往小试，见上翘立刻回退
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

        **下一课预告** — `16_逻辑回归与交叉熵`：

        今天，你的模型学会了预测**连续的数值**——房价、温度、销量。
        但世界的另一半问题是"**是 / 否**"：这封邮件是不是垃圾？
        这张图里是猫还是狗？用户会不会点击这个商品？

        直线输出的是 -∞ 到 +∞ 的任意数，而"是 / 否"需要的是 0 到 1 的**概率**。
        办法已在前方等你：把直线掰弯成 S 形（**sigmoid**），把 MSE 换成
        更配概率的**交叉熵**——名字全新，骨架照旧：还是那四步曲，
        前向、损失、梯度、更新，一件不多一件不少。你会亲眼看到：
        换了装备的四步曲，一样转得飞起。而那个 S 形曲线，正是
        神经网络里"神经元"的标准长相——你已经在通往 Transformer 的路上了。

        🏅 恭喜！你亲手养活了第一个会学习的模型。从一条直线到 Transformer，
        中间隔的只是"更深的前向"——四步曲，你已经会了。
        """
    )
    return


if __name__ == "__main__":
    app.run()
