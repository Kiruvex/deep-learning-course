import marimo

__generated_with = "0.25.2"
app = marimo.App(app_title="17 · 过拟合与数据划分", width="medium")


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
        # 17 · 过拟合与数据划分

        **从零到 Transformer · 手写深度学习** ｜ 第 3 部分 · 经典机器学习热身（3/3）
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📖 本课目标（先读我）": mo.md(
                """
                **这一课只解决一个问题：模型"训练 100 分、新数据翻车"，怎么办？**

                第 16 课结尾的预告还热着：一个幽灵正在逼近——训练集 100 分、新数据翻车，
                它叫过拟合；这一课要把数据切成三份，亲眼看 9 次多项式如何"背题"，
                以及怎么在疯掉之前踩刹车。

                今天，正面迎击这个幽灵：**过拟合**（overfitting）。
                打法依然稳扎稳打：先造一份"可以作弊也没人发现"的小数据，
                把过拟合亲手养出来；再学怎么抓住它（验证集）、怎么给它定性
                （诊断图）、怎么治它（正则化）。

                这是第 3 部分的收官课——好消息是，你手里的零件一件不缺：
                造数据要 rng（第 10 课）、算误差要 MSE（第 15 课）、
                评模型要看没见过的数据（本课主题）、解方程要 solve、
                量权重要范数（第 09 课）。本课只是把它们组装成一台"模型体检机"。

                学完你将能：

                - 说出**过拟合**是什么：模型不学规律、改背题——以及为什么容量越大越有背题的本钱
                - 把数据**洗牌后切成三份**（train / val / test），讲清"上课 / 模考 / 高考"各自的身份与纪律
                - 画出**次数-误差曲线**：训练误差一路降、验证误差先降后升，谷底就是刚刚好
                - 用 **L2 正则化**给权重上"紧箍咒"，亲手把 9 次疯魔曲线驯服成乖巧模样

                前置：第 15 课的 MSE 与"训练"、第 16 课的模型评估、第 10 课的 rng 与洗牌、
                第 09 课的范数与浮点噪声（solve 是 linalg 家族今天新认识的一位，比先求逆更稳）。

                收官之后，第 4 部分"神经网络"开门——本课的诊断图和刹车片，往后每一课都要用。
                把这一课吃透，你就拿到了进入主角篇章的门票。
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "👻 概念 1/4 · 过拟合 = 背题": mo.md(
                """
                一个学生备考，有两种活法：

                - **理解解题方法**：换一张卷子照样会做
                - **把往年题的答案抄进小抄**：原卷 100 分，新卷当场翻车

                模型也一样：

                - **学规律**：抓住"x 变大、y 大致跟着 sin 走"这类真实结构，新数据照样准
                - **背题**：把训练集里**每个点的噪声**（造数据时叠的那 0.25 随机抖动）也一并
                  背下来——训练误差漂亮得不像话，一换数据就露馅

                对照着翻译一下：

                | 学生世界 | 模型世界 |
                |---|---|
                | 解题方法 | 数据里的真实规律（sin 的形状） |
                | 小抄 | 把噪声也编进去的巨大权重 |
                | 原卷 100 分 | 训练误差极低 |
                | 新卷翻车 | 验证 / 测试误差爆炸 |

                后者就是**过拟合**。它有个孪生兄弟叫**欠拟合**（underfitting）：
                连规律都没学够，训练误差本身就下不来——一条直线硬拟 sin，怎么摆都不像。

                关键旋钮是**容量**（capacity）：模型能表达的复杂程度。
                多项式的次数就是容量旋钮——次数越高，能扭出的花样越多，**背题的本钱越足**：
                1 次多项式想背题都没门路（它只能是一条直线）；
                9 次多项式有 10 个自由参数，14 个训练点的弯弯绕绕它绰绰有余
                （连噪声都能背下一大半）。
                演示 2 会让你亲眼看到这两极。
                """
            ),
            "✂️ 概念 2/4 · 三分天下：train / val / test": mo.md(
                """
                既然"用训练集打分"永远查不出背题（小抄上当然全对），那就**换卷子考它**。
                标准做法：把数据切成三份，各司其职——

                | 集合 | 身份 | 用途 | 纪律 |
                |---|---|---|---|
                | 训练集 train（约 60%） | 上课 | 学参数（拟合） | 随便用 |
                | 验证集 val（约 20%） | 模考 | 选模型、调超参 | 可以反复看，但别用它学参数 |
                | 测试集 test（约 20%） | 高考 | 最终评估 | **只许碰一次**，碰完不许再调 |

                比例是惯例不是法律：60/20/20 或 80/10/10 都常见；数据越多，
                val / test 的占比可以越小。24 个点太小，本课切成 14 / 5 / 5。

                一个常见陷阱要提前打过疫苗：**拿测试集调模型 = 高考作弊**。
                "用 test 试了 9 次，第 5 次分数最好"——那个分数已经不可信了：
                你等于对着高考答案反复修改自己的解法，报出来的分是虚的，
                上线（真来了新数据）立刻现形。所以纪律才这么苛刻。

                还有一条铁律：**切分之前必须先洗牌**（第 10 课的 rng 回归）。
                如果数据是按类别排好序的（前一半全是 A 类、后一半全是 B 类），
                不洗牌直接切，训练集可能全是一类——模型连"另一类长什么样"都没见过，
                后面的一切都白搭。
                """
            ),
            "📈 概念 3/4 · 次数-误差曲线（本课灵魂图）": mo.md(
                """
                把容量旋钮从 1 拧到 9，每个次数都拟合一次，分别记下训练 / 验证 MSE，
                画在同一张图上——你会看到两条走势截然不同的曲线：

                - **训练误差**：随容量**单调下降**。容量越大越能贴着训练点扭，
                  把噪声一个个背下来，训练分数只会越来越好
                - **验证误差**：**先降后升**（U 型）。左端容量不够、规律没学够（欠拟合）；
                  右端开始背题、把噪声也背进去了（过拟合）；**谷底 = 刚刚好**

                数值预告（演示 3 会画出来）：本课的数据上，训练 MSE 从 0.158 一路降到
                0.038；验证 MSE 从 0.335 俯冲到 0.081（第 3 次，谷底），再抬升到 0.14 上下。

                这张图是以后所有"选模型"动作的原型：横轴不只是多项式次数——
                换成神经网络层数、训练轮数、λ 大小，故事完全一样。

                用它选模型的标准三步：**扫一遍候选值 → 看验证线的谷底 → 取谷底对应的容量**。
                看到自己的验证曲线抬头，就是该踩刹车的信号。
                """
            ),
            "🛡️ 概念 4/4 · 三件防御武器": mo.md(
                """
                确认过拟合之后，有三件常规武器：

                **① 早停（early stopping）**：训练过程中盯着验证误差，一旦开始回升立刻停手——
                "在疯掉之前踩刹车"。第 25 课训练神经网络时细讲。

                **② 正则化（L2）**：在损失里加一项 **λ·‖w‖²**（第 09 课范数的第二次登场！），
                意思是"权重别长太大"。加完之后总损失变成"MSE + λ·‖w‖²"——
                模型从此活在两头拉扯里：一头想拟合得更准，一头想权重更小，
                λ 决定哪头的嗓门大。多项式疯魔的本质是高次项系数巨大、
                曲线被几个大系数甩来甩去；给所有权重套上**紧箍咒**，
                它们就扭不出那么夸张的花样。λ 是松紧旋钮：λ=0 不设防，
                λ 太大把模型勒成一条平线。本课演示 4 亲手拧这个旋钮。

                **③ 更多数据**：最朴素也最有效——背题之所以可能，是因为题少：
                14 个训练点喂 10 个参数，模型有的是余力背；2400 个点喂 10 个参数，
                它就只能老老实实学规律了。

                三件武器不互斥，实战常常一起上。
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

        五组演示，组装一台完整的"模型体检机"：

        - **演示 1** · 造数据并三分：24 个带噪声的点，洗牌切成 14 / 5 / 5
        - **演示 2** · 名场面：1 次 / 3 次 / 9 次同台——欠拟合、刚刚好、疯魔
        - **演示 3** · 灵魂图：次数-误差曲线，谷底在哪一目了然
        - **演示 4** · L2 正则化：紧箍咒 λ 亲手拧，9 次曲线从疯魔到平滑
        - **演示 5** · 高考揭榜：谷底模型 vs 从未见过的测试集
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 1 · 造数据，洗牌，三分

        老配方（第 10 课）：`rng = np.random.default_rng(3)` 造一台专属随机数发生器，
        `uniform(0, 6, 24)` 在 0~6 之间撒 24 个 x，真值取 sin(x)，
        再叠一层 σ=0.25 的高斯噪声——**这层噪声就是本课的"反派"**：
        模型的正经任务是学 sin，不是把这 0.25 的抖动也背下来。

        切分用 `rng.permutation(24)`：返回 0~23 的一个随机排列，
        拿它当花式索引（第 05 课）去捞数据——前 14 个给训练、中间 5 个给验证、
        最后 5 个给测试。它和"先 arange 再 shuffle"效果一样但少一步、更稳（第 10 课）。

        顺手算笔账：24×0.6 = 14.4、24×0.2 = 4.8——除不尽就用 round 取整，
        得到 14 / 5 / 5；三段拼回来正好 24，不多不少。比例是近似，纪律是精确的。
        """
    )
    return


@app.cell
def _(np, plt):
    # 演示 1 · 造 24 个带噪声的点，洗牌后切成 14/5/5
    # 数据配方（10 课老三样）：均匀撒 x → 算真值 → 叠高斯噪声
    _rng = np.random.default_rng(3)                # 专属 rng，种子 3（10 课：一切随机从它拿，可复现）
    x = _rng.uniform(0, 6, 24)                     # 24 个 x，均匀撒在 0~6
    _y_true = np.sin(x)                            # 真值：sin 曲线（模型的"标准答案"）
    y = _y_true + _rng.normal(0, 0.25, 24)         # 叠加 σ=0.25 的噪声——本课反派登场
    # 刻意只造 24 个点：点越少、模型相对越肥，过拟合的疯魔效果越足

    # ---- 三分天下：先洗牌，再切片 ----
    _idx = _rng.permutation(24)                    # 0~23 的随机排列（10 课 permutation，返回新数组）
    _n_train = int(round(24 * 0.6))                # 24×0.6 = 14.4 → 14 个训练点
    _n_val = int(round(24 * 0.2))                  # 24×0.2 = 4.8 → 5 个验证点，剩下的 5 个做测试
    _i_tr = _idx[:_n_train]                        # 乱序索引的前一段 → 训练集
    _i_va = _idx[_n_train:_n_train + _n_val]       # 中间一段 → 验证集
    _i_te = _idx[_n_train + _n_val:]               # 剩下一段 → 测试集（先封存，演示 5 才拆）

    x_train, y_train = x[_i_tr], y[_i_tr]          # 花式索引捞数据（05 课）
    x_val, y_val = x[_i_va], y[_i_va]              # 验证集：模考卷
    x_test, y_test = x[_i_te], y[_i_te]            # 测试集：高考卷
    # 自查：三份切的是同一根乱序索引的不同段——天然不重叠、不遗漏，14+5+5=24

    # ---- 画图：三种身份，三种记号（圆点 / 方块 / 三角，一眼分清）----
    _fig, _ax = plt.subplots(figsize=(7.5, 4.2))
    _xx = np.linspace(0, 6, 300)                   # 采细一点的 x，画真曲线用（03 课）
    _ax.plot(_xx, np.sin(_xx), color="gray", lw=1.5, alpha=0.6, label="sin(x) 真曲线")
    _ax.scatter(x_train, y_train, s=45, color="tab:blue", marker="o",
                label=f"训练集（上课，{_n_train} 点）")
    _ax.scatter(x_val, y_val, s=55, color="tab:orange", marker="s",
                label=f"验证集（模考，{_n_val} 点）")
    _ax.scatter(x_test, y_test, s=55, color="tab:green", marker="^",
                label=f"测试集（高考，{len(_i_te)} 点）")
    _ax.set_title("三分天下：同一个 sin，三种身份的 24 个点")
    _ax.set_xlabel("x")
    _ax.set_ylabel("y")
    _ax.legend()
    _ax.grid(True, alpha=0.3)
    _fig
    return (x, y, x_test, x_train, x_val, y_test, y_train, y_val)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 2 · 名场面：1 次 / 3 次 / 9 次同台

        同一个训练集，只拧"容量旋钮"：分别用 1 次、3 次、9 次多项式拟合。

        求解器用 `np.polyfit(x, y, 次数)`——先亮明身份：**最小二乘的库函数版**。
        第 15 课你手写过"梯度下降一步步找直线"；polyfit 干的是同一件事
        （最小化 MSE），只是它用公式一步解出、不用迭代。本课的主角是**过拟合现象**
        而不是求解器，所以放心把求解交给它——判断模型好不好，
        看的是它在**没见过的数据**上的表现，跟它怎么被解出来的无关。
        （第 15 课的梯度下降和今天的 polyfit，是同一个目标的两条求解路线。）

        三块面板用**同一把 y 尺子**（-4.5 ~ 2.0），谁也没作弊。
        重点盯第三块：曲线贴着训练点扭动，却在最左端（训练点最稀疏的地方）一头扎下悬崖。
        """
    )
    return


@app.cell
def _(np, plt, x_train, x_val, y_train, y_val):
    # 演示 2 · 同一训练集，三种容量：1 次（欠）、3 次（刚好）、9 次（过）
    _degs_show = [1, 3, 9]                         # 三位选手：容量旋钮的三个档位
    _verdicts = ["欠拟合：一条直线硬掰",             # 1 次：没弯的本事
                 "刚刚好：贴着 sin 走",              # 3 次：容量刚好装下规律
                 "过拟合：贴点狂舞，空档疯魔"]        # 9 次：连噪声一起背
    _xx = np.linspace(0, 6, 300)                   # 拟合曲线的采样点（03 课）

    # sharex / sharey：三块面板共用一把坐标尺——要比较，尺子先统一
    _fig, _axes = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True, sharey=True)
    for _ax, _d, _v in zip(_axes, _degs_show, _verdicts):
        _c = np.polyfit(x_train, y_train, _d)      # 最小二乘解出 _d 次多项式（系数高次在前）
        _mse_tr = float(np.mean((np.polyval(_c, x_train) - y_train) ** 2))   # 上课分（15 课 MSE）
        _mse_va = float(np.mean((np.polyval(_c, x_val) - y_val) ** 2))       # 模考分（16 课）
        # 每块面板三件套：真曲线（淡）+ 拟合曲线（红）+ 训练点（蓝）
        # val / test 一律不画——狠下心，只给"训练视角"
        _ax.plot(_xx, np.sin(_xx), color="gray", lw=1.2, alpha=0.55)         # 真曲线（淡）
        _ax.plot(_xx, np.polyval(_c, _xx), color="tab:red", lw=2)            # 拟合曲线：polyval 按系数求值
        _ax.scatter(x_train, y_train, s=28, color="tab:blue", zorder=3)      # 训练点（只画训练！）
        _ax.set_title(f"{_d} 次 · {_v}\ntrain MSE = {_mse_tr:.3f} ｜ val MSE = {_mse_va:.3f}",
                      fontsize=10)
        _ax.set_xlabel("x")
        _ax.grid(True, alpha=0.3)
    _axes[0].set_ylabel("y")
    _axes[0].set_ylim(-4.5, 2.0)                   # 三块面板共用一把 y 尺：9 次的悬崖不作弊地露出来

    # 给第三块的悬崖加个标注：曲线在 x→0 处冲出画面底部
    _axes[2].annotate("左端悬崖：x→0 冲到 -15.9\n（冲出画面了）",
                      xy=(0.16, -4.3), xytext=(1.2, -3.4),
                      fontsize=9, color="tab:red",
                      arrowprops=dict(arrowstyle="->", color="tab:red"))
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        读图（三块面板，三个名次）：

        - **1 次**：一条直线硬掰 sin——train MSE **0.158**，连训练集的弯都没学会（欠拟合）。
          直线没有弯的本事，容量不够；换一批数据它还是这个水平
        - **3 次**：贴着 sin 走，train **0.066** / val **0.081**，两边都低——学到了规律本身
        - **9 次**：train 掉到 **0.038**（三人中最低！），但曲线在最左端一头扎下悬崖：
          x=0 处低到 **-15.9**，而 sin(0) = 0。最左端 0~0.5 一带一个训练点都没有，
          10 个自由参数在无人监督的空档里尽兴发挥——这就是**背题**的样子：
          分数是训练点上的，疯魔是空档里的

        一个量化的小尺子：噪声 σ=0.25，方差 σ²=**0.0625**——这是"误差地板"，
        任何模型在**没见过的数据**上都不可能低于它（随机就是随机，预测不了）。
        3 次的训练 MSE 0.066 ≈ 0.0625：规律学尽、噪声剩下，刚刚好；
        9 次的训练 MSE 0.038 **低于地板**——多出来的部分，只能是背下来的噪声。

        注意那个反直觉的事实：**9 次的训练误差是三人中最低的**。
        如果只看训练分数选模型，你会兴高采烈选最疯的那个——
        这就是为什么需要验证集，以及为什么要把整个旋钮扫一遍（下一组演示）。

        最后品一下这三种"输法"：1 次输在**学不够**（训练、验证都高），
        9 次输在**学过头**（训练最低、验证抬头）——方向正好相反；
        只有 3 次把两头都压住了。以后拿到任何模型的成绩单，
        先问一句：它输在哪一边？答案决定你该加容量、减容量，还是上正则。
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 3 · 灵魂图：次数-误差曲线

        演示 2 只看了三个档位，现在把整个旋钮拧一遍：次数 1~9 逐个拟合，
        每个都记下 train / val 两份 MSE，画在同一张图上。
        蓝线（训练）只会低头，橙线（验证）先俯冲后抬头——**谷底落在第 3 次**。

        图里还画了一条**噪声地板**（σ² = 0.0625 的水平虚线）：
        训练曲线穿到地板之下之时，就是模型开始背噪声之刻。

        看图三步：先找橙线的**最低点**（谷底 = 该选的容量）；
        再看蓝线**有没有穿过噪声地板**（穿过 = 开始背噪声）；
        最后看谷底右边橙线**抬头的斜率**（抬头越猛，背题越狠）。

        这张图就是"选模型"的标准动作本身：
        以后选神经网络深度、选训练轮数、选 λ，画的都是它的亲戚。
        """
    )
    return


@app.cell
def _(np, plt, x_train, x_val, y_train, y_val):
    # 演示 3 · 灵魂图：容量从 1 拧到 9，train / val 两份 MSE 各记一笔
    degs = np.arange(1, 10)                        # 候选次数 1~9（03 课 arange，左闭右开）
    mses_train = np.zeros(9)                       # 每个次数的训练 MSE（先开好房间再填）
    mses_val = np.zeros(9)                         # 每个次数的验证 MSE

    # 这九次拟合互相独立——不存在"在第 3 次的基础上接着练"，每一档都从头拟
    for _k, _d in enumerate(degs):                 # _k 是数组下标（0~8），_d 是次数（1~9）
        _c = np.polyfit(x_train, y_train, _d)      # 拟合当前次数
        mses_train[_k] = np.mean((np.polyval(_c, x_train) - y_train) ** 2)   # 上课分
        mses_val[_k] = np.mean((np.polyval(_c, x_val) - y_val) ** 2)         # 模考分

    best_deg = int(degs[np.argmin(mses_val)])      # 谷底次数：argmin 给下标，degs[下标] 才是次数（08 课）

    # ---- 画图：两条误差曲线 + 噪声地板 + 谷底红圈 ----
    _fig, _ax = plt.subplots(figsize=(7.5, 4.4))
    _ax.plot(degs, mses_train, "o-", color="tab:blue", label="训练 MSE（上课分）")
    _ax.plot(degs, mses_val, "s-", color="tab:orange", label="验证 MSE（模考分）")
    _ax.axhline(0.0625, color="gray", ls=":", lw=1, label="噪声地板（σ² = 0.0625）")
    _ax.scatter([best_deg], [mses_val.min()], s=150, facecolors="none",
                edgecolors="tab:red", linewidths=2, zorder=4)          # 红圈套住谷底
    # 圈出谷底并标注——选模型的落点，全图唯一的大红圈
    _ax.annotate(f"谷底：{best_deg} 次\nval = {mses_val.min():.3f}",
                 xy=(best_deg, mses_val.min()), xytext=(4.4, 0.19),
                 fontsize=10, arrowprops=dict(arrowstyle="->", color="tab:red"))
    _ax.text(1.05, 0.372, "← 欠拟合区：容量不够", fontsize=9, color="gray")
    _ax.text(6.9, 0.235, "过拟合区：开始背题 →", fontsize=9, color="gray", ha="center")
    _ax.set_ylim(0, 0.42)
    _ax.set_xticks(degs)
    _ax.set_xlabel("多项式次数（容量旋钮）")
    _ax.set_ylabel("MSE")
    _ax.set_title("灵魂图：训练只降不升，验证先降后升")
    _ax.legend(loc="upper right")
    _ax.grid(True, alpha=0.3)
    _fig
    return (best_deg, degs, mses_train, mses_val)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 4 · L2 正则化：给 9 次多项式上紧箍咒

        9 次疯魔的病根：**高次项系数长得太大**，曲线被几个大系数甩来甩去。
        L2 正则化的药方：解方程时在对角线上加一个 λ——相当于在损失里加
        **λ·‖w‖²**（第 09 课范数的第二次登场），谁想长太大就罚谁的分。

        手写"多项式版岭回归"需要一件新工具：**Vandermonde 矩阵**。
        `np.vander(x, 10, increasing=True)` 造一个 (点数, 10) 的矩阵，
        **第 k 列是 x 的 k 次幂**——第 0 列全是 1（截距项），第 1 列是 x 本身，
        第 9 列是 x⁹。举个例子：x/6 = 0.5 的那一行就是
        [1, 0.5, 0.25, 0.125, …, 0.5⁹]——每往右一列，就多乘一次 x。
        于是"拟合多项式" = "解 V·w = y 的最小二乘"，
        正规方程一步出解（np.linalg.solve——linalg 家族的新成员，比先求逆再乘更稳）：

        ```
        (VᵀV + λI)·w = Vᵀy        （λ=0 就是普通最小二乘；λ>0 就是岭回归）
        ```

        一个必须先做的工程处理：**把 x 归一化到 0~1**（除以 6）。
        0~6 的 9 次幂会冲到 10⁷ 量级，VᵀV 的条件数大到浮点扛不住
        （第 09 课的浮点噪声，这里直接决定解可不可信）；除以 6 之后每列都在 0~1，
        数值立刻安分。代价只是系数 w 换了单位（针对 x/6 而非 x）——
        画图时把采样点也除以 6 再喂给同一个 V，曲线形状一分不少。

        顺便说个名字的来历：对角线加 λ 之后，病态的 VᵀV 像被垫上了一道
        **山脊**（ridge），解重新变得可靠——所以这套方法叫**岭回归**。
        深度学习里它换个马甲叫 **weight decay**（权重衰减），第 25 课还会遇到。
        """
    )
    return


@app.cell
def _(mo, np, plt, x_train, x_val, y_train, y_val):
    # 演示 4 · 岭回归驯服 9 次多项式：λ ∈ {0, 0.01, 0.1, 1} 四连拍
    _lams = [0.0, 0.01, 0.1, 1.0]                  # 紧箍咒的四档松紧

    _xn_tr = x_train / 6.0                         # 归一化到 0~1：x⁹ 从 10⁷ 量级压回 1 以内（数值稳定）
    _xn_va = x_val / 6.0                           # 验证集走同一套变换
    _V_tr = np.vander(_xn_tr, 10, increasing=True)  # Vandermonde (14, 10)：第 k 列 = (x/6) 的 k 次幂
    _V_va = np.vander(_xn_va, 10, increasing=True)  # 验证集同款特征矩阵
    _I = np.eye(10)                                # 10×10 单位阵（03 课 eye）
    _xx = np.linspace(0, 6, 300)
    _V_xx = np.vander(_xx / 6.0, 10, increasing=True)   # 画曲线：采样点也走同一套特征

    _ms_va_list = []                               # 记下四档的验证 MSE
    _w_norm_list = []                              # 记下四档的 ‖w‖（紧箍咒读数）

    # 同一个 9 次模型、同一份数据，只有 λ 在变——控制变量，才能看清 λ 的作用
    _fig, _axes = plt.subplots(1, 4, figsize=(13, 3.6), sharex=True, sharey=True)
    for _ax, _lam in zip(_axes, _lams):
        _A = _V_tr.T @ _V_tr + _lam * _I           # 正规方程左边：VᵀV + λI——对角线上的 λ 就是紧箍咒
        _b = _V_tr.T @ y_train                     # 右边：Vᵀy
        _w = np.linalg.solve(_A, _b)               # 一步解出 10 个权重（linalg 新成员 solve）
        _mse_va = float(np.mean((_V_va @ _w - y_val) ** 2))     # 模考分
        _nw = float(np.linalg.norm(_w))            # ‖w‖：09 课的范数，正则松紧的读数
        _ms_va_list.append(_mse_va)
        _w_norm_list.append(_nw)

        _ax.plot(_xx, np.sin(_xx), color="gray", lw=1.2, alpha=0.55)      # 真曲线（淡）
        _ax.plot(_xx, _V_xx @ _w, color="tab:red", lw=2)                 # 拟合曲线：矩阵×权重一把梭
        _ax.scatter(x_train, y_train, s=24, color="tab:blue", zorder=3)  # 训练点
        _nw_str = f"{_nw:,.0f}" if _nw >= 100 else f"{_nw:.1f}"          # 百万级用千分位，个位留一位小数
        _ax.set_title(f"λ = {_lam}｜val MSE = {_mse_va:.3f}｜‖w‖ = {_nw_str}", fontsize=10)
        _ax.set_ylim(-2.0, 2.0)                    # 四块共用一把 y 尺
        _ax.set_xlabel("x")
        _ax.grid(True, alpha=0.3)

    mo.vstack([
        _fig,
        mo.md(
            f"""
            读图（从左到右，紧箍咒越勒越紧）：

            - **λ = 0**：不设防。曲线和演示 2 的 9 次一样疯，而且解出来的 ‖w‖ 冲到
              **{_w_norm_list[0]:,.0f}**——正常该是个位数！不设防的解连数值都在走钢丝
              （VᵀV 病态到什么程度，09 课的浮点噪声提前打过预防针）
            - **λ = 0.01 / 0.1**：曲线明显收敛，‖w‖ 缩到 **{_w_norm_list[1]:.1f}** 和
              **{_w_norm_list[2]:.1f}**——权重被罚小，高次项扭不动了
            - **λ = 1**：勒过头。曲线被压平，训练误差反而不如一条直线——
              紧箍咒是防疯的，不是防本事的

            一个诚实的观察：这四档的 val MSE（{_ms_va_list[0]:.3f} → {_ms_va_list[1]:.3f} →
            {_ms_va_list[2]:.3f} → {_ms_va_list[3]:.3f}）单调变差。两个原因：
            其一，验证集只有 5 个点、x 全落在 1.7~4.4 的中段，一个都没踩进
            x < 0.5 的悬崖区——λ=0 的疯没被模考抓到（小验证集的盲区，现实就是这样）；
            其二，9 次对 14 个训练点本来就太肥，勒紧箍咒救不回容量错误。
            真正的甜点在 0 和 0.01 之间——练习 3 的 λ=0.001 把 val 压到 **0.108**，
            曲线照样乖。**先选容量（演示 3），再调正则（练习 3），顺序别颠倒。**
            """
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 5 · 高考揭榜：测试集的最终审判

        先把演示 3 那张灵魂图读完整（以后天天要读）：

        - **左端**（1~2 次）：两条线都很高——欠拟合，容量不够
        - **谷底**（3 次）：验证 MSE **0.081**，几乎贴着噪声地板 0.0625——规律学尽、噪声剩下
        - **右端**（4~9 次）：训练线继续低头（背题分数越来越好），验证线整体抬到
          0.10~0.14——过拟合（第 8 格有个小抖动：验证集只有 5 个点，现实就是会抖；
          验证集越大，U 型越干净）

        谷底选出了 **3 次多项式**——注意这个决定只用了训练集和验证集，
        测试集从头到尾没让它看过。现在拆封高考卷：用训练集拟合 3 次
        （和演示 3 里那次完全相同），在 5 个**从未见过的测试点**上算 MSE，
        与 train / val 并排出表。

        纪律重申：**测试集只许碰这一次**。看完揭榜又回头改参数再考一次，
        高考就退化成第二次模考——分数不再可信。顺手做个反面对照：
        如果当初被漂亮的训练分数骗去选 9 次，同一张高考卷它能考成什么样。
        """
    )
    return


@app.cell
def _(best_deg, mo, np, plt, x_test, x_train, x_val, y_test, y_train, y_val):
    # 演示 5 · 测试集审判：谷底模型上考场
    # 纪律就体现在"一次"上：只在 test 上算这一次分，算完就封卷，不许回头调
    _c_best = np.polyfit(x_train, y_train, best_deg)      # 用选出的最佳次数拟合（与演示 3 同一款）
    _mse_tr = float(np.mean((np.polyval(_c_best, x_train) - y_train) ** 2))   # 上课分
    _mse_va = float(np.mean((np.polyval(_c_best, x_val) - y_val) ** 2))       # 模考分
    _mse_te = float(np.mean((np.polyval(_c_best, x_test) - y_test) ** 2))     # 高考分：第一次也是最后一次

    # 反面对照：如果当初选了 9 次，同一张高考卷……
    # （同一个训练集、同一个测试集，只换容量——控制变量，差距才说明问题）
    _c_bad = np.polyfit(x_train, y_train, 9)
    _mse_te_bad = float(np.mean((np.polyval(_c_bad, x_test) - y_test) ** 2))
    _ratio = _mse_te_bad / _mse_te                        # 两个模型的高考差距（倍数）

    _xx = np.linspace(0, 6, 300)
    _fig, _ax = plt.subplots(figsize=(7.5, 4.4))
    _ax.plot(_xx, np.sin(_xx), color="gray", lw=1.2, alpha=0.55, label="sin(x) 真曲线")
    _ax.plot(_xx, np.polyval(_c_best, _xx), color="tab:red", lw=2,
             label=f"{best_deg} 次拟合曲线")
    _ax.scatter(x_train, y_train, s=24, color="tab:blue", alpha=0.5, label="训练点（淡出）")
    _ax.scatter(x_test, y_test, s=90, color="tab:green", marker="^", zorder=4,
                label="测试点（高考！）")
    _ax.set_ylim(-1.8, 1.8)
    _ax.set_xlabel("x")
    _ax.set_ylabel("y")
    _ax.legend()
    _ax.grid(True, alpha=0.3)
    _ax.set_title(f"高考揭榜：{best_deg} 次多项式 vs 5 个从未见过的点")

    mo.vstack([
        _fig,
        mo.md(
            f"""
            | 卷子 | MSE | 身份 |
            |---|---|---|
            | 训练集（上课） | **{_mse_tr:.3f}** | 学参数用的 |
            | 验证集（模考） | **{_mse_va:.3f}** | 选模型用的 |
            | **测试集（高考）** | **{_mse_te:.3f}** | 只碰这一次 |
            """
        ),
        mo.callout(
            mo.md(
                f"""
                **揭榜**：高考 {_mse_te:.3f} ≈ 模考 {_mse_va:.3f}——它真的会做题，不是背题。

                反面教材：9 次多项式在同一张高考卷上考出 **{_mse_te_bad:.1f}**，
                差 **{_ratio:.0f} 倍**。它的疯魔集中在训练点的空档里（左端悬崖），
                模考的 5 个点恰好没踩进去、只涨了七成；高考的 5 个点里恰好有一个
                落在悬崖正下方——**背题的代价，全押在没见过的点上**。这就是 val 与 test
                都不能省的原因：一个帮你选模型，一个替你揭老底。
                """
            ),
            kind="success",
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 🎛️ 交互实验

        **容量旋钮**——把演示 2 的三连拍变成连续剧。拖动次数滑块（1~9），
        看拟合曲线、训练 / 验证 MSE、"验证 ÷ 训练"的倍数如何实时变化，
        最后让判词告诉你现在站在 U 型曲线的哪一段。

        两种玩法：

        1. 从 9 开始往回拧，找到验证 MSE 的最低点，再拧过头到 1——
           欠拟合、刚刚好、过拟合三种状态各站一次，过拟合的"手感"就有了
        2. 盯住"验证 ÷ 训练"的倍数：过了谷底，容量越大倍数越大——训练分和新数据分
           之间的落差，就是"背题指数"

        注意高次时 y 轴会被疯魔撑大——轴的刻度本身就是警报。
        """
    )
    return


@app.cell
def _(mo):
    # 交互控制：容量旋钮（debounce=True：松手才重算，画面不闪烁）
    degree_slider = mo.ui.slider(1, 9, value=9, step=1, debounce=True,
                                 label="🎛️ 多项式次数（容量旋钮，1~9）")
    degree_slider
    return (degree_slider,)


@app.cell
def _(degree_slider, mo, np, plt, x_train, x_val, y_train, y_val):
    # ===== 交互实验 · 容量旋钮（响应单元）=====
    # 滑块一动这里就重算：marimo 自动追踪依赖，控制单元变了响应单元就跑
    _d = degree_slider.value                        # 当前次数（滑块值）

    # 先把 1~9 全部拟一遍，找到 U 型谷底做参照（9 次 polyfit 也就几十毫秒，放心算）
    _val_all = []
    for _dd in range(1, 10):
        _cc = np.polyfit(x_train, y_train, _dd)
        _val_all.append(np.mean((np.polyval(_cc, x_val) - y_val) ** 2))
    _best_d = int(np.argmin(_val_all)) + 1          # 谷底次数（argmin 给下标，+1 才是次数）
    _min_val = float(_val_all[_best_d - 1])         # 谷底的验证 MSE

    # 当前次数的拟合与两份误差
    _c = np.polyfit(x_train, y_train, _d)
    _mse_tr = float(np.mean((np.polyval(_c, x_train) - y_train) ** 2))   # 上课分
    _mse_va = float(np.mean((np.polyval(_c, x_val) - y_val) ** 2))       # 模考分
    _ratio = _mse_va / max(_mse_tr, 1e-12)         # 验证是训练的几倍（max 兜底防除零）

    # 画图：训练点 + 验证点 + 拟合曲线（测试点一个都不画——纪律！）
    _xx = np.linspace(0, 6, 300)
    _yy_curve = np.polyval(_c, _xx)
    _fig, _ax = plt.subplots(figsize=(7.5, 4.6))
    _ax.plot(_xx, np.sin(_xx), color="gray", lw=1.2, alpha=0.55, label="sin(x) 真曲线")
    _ax.plot(_xx, _yy_curve, color="tab:red", lw=2, label=f"{_d} 次拟合曲线")
    _ax.scatter(x_train, y_train, s=30, color="tab:blue", zorder=3, label="训练点")
    _ax.scatter(x_val, y_val, s=55, color="tab:orange", marker="s", zorder=3,
                label="验证点（它没见过）")
    _ymin = min(-1.8, float(_yy_curve.min()) - 0.4)   # 疯魔时 y 轴自动放宽——刻度本身就是警报
    _ymax = max(1.8, float(_yy_curve.max()) + 0.4)
    _ax.set_ylim(_ymin, _ymax)
    _stretch = "（注意：y 轴被撑大了）" if (_ymin < -2 or _ymax > 2) else ""
    _ax.set_title(f"容量旋钮：{_d} 次多项式{_stretch}")
    _ax.set_xlabel("x")
    _ax.set_ylabel("y")
    _ax.legend()
    _ax.grid(True, alpha=0.3)

    # 判词：现在站在 U 型曲线的哪一段？
    # 判词规则：拿当前次数和谷底比——左边欠、右边过、谷底刚好
    if _d < _best_d:
        _verdict = (f"🥶 欠拟合：训练 {_mse_tr:.3f} / 验证 {_mse_va:.3f} 双双居高——"
                    f"规律本身没学够，容量不足，往右拧")
    elif _d == _best_d:
        _verdict = (f"✅ 刚刚好：验证 {_mse_va:.3f} 是全场最低"
                    f"（第 {_best_d} 次就是 U 型谷底）——停在这里")
    else:
        _verdict = (f"👻 过拟合警报：训练 {_mse_tr:.3f} 还在降，验证却抬到 {_mse_va:.3f}"
                    f"（谷底的 {_mse_va / _min_val:.1f} 倍）——它在背题！")

    mo.vstack([
        _fig,
        mo.md(
            f"""
            | 量 | 值 |
            |---|---|
            | 训练 MSE（上课分） | **{_mse_tr:.3f}** |
            | 验证 MSE（模考分） | **{_mse_va:.3f}** |
            | 验证 ÷ 训练（背题指数） | **{_ratio:.1f} 倍** |
            | 全场谷底 | 第 **{_best_d}** 次（val = {_min_val:.3f}） |
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
                **练习 1 · 手写三分函数**：把演示 1 的切分逻辑包成
                `train_val_test_split(x, y, seed, ratio)`——函数骨架已给出，
                你只需补上"permutation 拿乱序索引"那一行，再用演示 1 的数据验证。

                **练习 2 · 找出最佳次数**：循环拟合 1~9 次，找出验证 MSE 最小的次数——
                这就是"超参数选择"的完整流程，演示 3 的代码化。

                **练习 3 · 调 λ**：固定 9 次与归一化，λ ∈ {0.001, 0.01, 0.1, 1}，
                哪个 λ 的验证 MSE 最低？（提示：比演示 4 多了一个更小的 0.001。）

                小提示：练习 1 的函数自带 seed 参数——它内部造自己的 rng，
                和演示 1 的互不干扰（10 课：一个种子一台 rng）。

                规则不变：先把 `None` 换成你的答案再运行；写完再看参考答案。
                """
            )
        }
    )
    return


@app.cell
def _(mo, np, x, y):
    # ===== 练习 1 · 手写 train_val_test_split =====
    def train_val_test_split(x, y, seed, ratio=(0.6, 0.2, 0.2)):
        """洗牌后按比例切三份。返回 (x_train, y_train, x_val, y_val, x_test, y_test)"""
        _rng = np.random.default_rng(seed)         # 专属 rng（10 课：一切随机从它拿）
        _n = len(x)                                # 总点数
        _idx = None                                # ← ① 补这行：_rng.permutation(_n)，拿到打乱的索引
        _n_tr = int(round(_n * ratio[0]))          # 训练集点数（四舍五入成整数）
        _n_va = int(round(_n * ratio[1]))          # 验证集点数（剩下的全给测试）
        _i_tr = _idx[:_n_tr]                       # 乱序索引的前一段 → 训练
        _i_va = _idx[_n_tr:_n_tr + _n_va]         # 中间一段 → 验证
        _i_te = _idx[_n_tr + _n_va:]               # 剩下一段 → 测试
        return x[_i_tr], y[_i_tr], x[_i_va], y[_i_va], x[_i_te], y[_i_te]

    split_fn_ex1 = None    # ← ② 补完函数后，把这行换成 train_val_test_split（就写函数名）

    if split_fn_ex1 is not None:
        try:
            _xt, _yt, _xv, _yv, _xe, _ye = split_fn_ex1(x, y, seed=7)     # 用演示 1 的数据 + 种子 7
            _n_all = len(np.unique(np.concatenate([_xt, _xv, _xe])))     # 三份拼回去应该还是 24 个
            _msg = (
                f"切分结果：训练 {len(_xt)} / 验证 {len(_xv)} / 测试 {len(_xe)} 点；"
                f"三份拼回去共 {_n_all} 个不同的 x（应等于 24：不重叠、不遗漏）✅"
            )
        except TypeError:
            _msg = "⏳ 函数里的 _idx 还是 None（None 没法切片）——先把 ① 补完"
    else:
        _msg = "⏳ 先补 _idx 那一行，再把 split_fn_ex1 换成 train_val_test_split"
    mo.md(_msg)
    return (split_fn_ex1, train_val_test_split)


@app.cell
def _(best_deg, mo, np, x_train, x_val, y_train, y_val):
    # ===== 练习 2 · 找最佳次数（超参数选择的完整流程）=====
    # 提示：把演示 3 的循环搬过来——
    #   for _d in range(1, 10):
    #       _c = np.polyfit(x_train, y_train, _d)
    #       _val = np.mean((np.polyval(_c, x_val) - y_val) ** 2)
    #       ...把每个 _val 记下来，最后 np.argmin 找最小值的位置（08 课）
    #   注意：argmin 给的是数组下标，+1（或 degs[下标]）才是次数
    #   记录可以用列表 append，也可以像演示 3 那样先 np.zeros(9) 再按下标填

    best_deg_ex2 = None    # ← 验证 MSE 最小的次数（1~9 中的某个整数）
    best_val_ex2 = None    # ← 对应的最小验证 MSE（选填，方便对答案）

    if best_deg_ex2 is not None:
        _val_part = f"{best_val_ex2:.3f}" if best_val_ex2 is not None else "（未填）"
        _match = (
            f"✅ 和演示 3 的谷底（第 {best_deg} 次）一致——这就是选模型的标准动作"
            if best_deg_ex2 == best_deg
            else f"❌ 演示 3 的谷底在第 {best_deg} 次；检查一下 argmin 的下标要不要 +1"
        )
        _msg = f"你的答案：最佳次数 = **{best_deg_ex2}**，val MSE = **{_val_part}**。{_match}"
    else:
        _msg = "⏳ 三步走：循环 1~9 拟合 → 记录每个次数的验证 MSE → argmin 找谷底"
    mo.md(_msg)
    return (best_deg_ex2, best_val_ex2)


@app.cell
def _(mo, np, x_train, x_val, y_train, y_val):
    # ===== 练习 3 · 调 λ：9 次 + 归一化，λ 扫描 =====
    lam_list_ex3 = [0.001, 0.01, 0.1, 1.0]        # 四档紧箍咒（比演示 4 多了更小的 0.001）

    # 提示骨架（把演示 4 的核心四行搬进循环）：
    #   _xn_tr, _xn_va = x_train / 6, x_val / 6        # 归一化别忘了
    #   _V_tr = np.vander(_xn_tr, 10, increasing=True)
    #   _V_va = np.vander(_xn_va, 10, increasing=True)
    #   for _lam in lam_list_ex3:
    #       _w = np.linalg.solve(_V_tr.T @ _V_tr + _lam * np.eye(10), _V_tr.T @ y_train)
    #       print(_lam, np.mean((_V_va @ _w - y_val) ** 2))   # 每档的验证 MSE
    # 注意：归一化（÷6）和 increasing=True 都别改——和演示 4 保持一致才能对上数

    best_lam_ex3 = None    # ← 验证 MSE 最低的 λ（填 lam_list_ex3 里的某一个）

    if best_lam_ex3 is not None:
        _match = (
            "✅ 正是它——最小的 λ 刚好掐住疯魔，又不至于勒死模型"
            if best_lam_ex3 == 0.001
            else "❌ 再跑一遍循环：四个 λ 里验证 MSE 最小的是哪一个？"
        )
        _msg = f"你的答案：best λ = **{best_lam_ex3}**。{_match}"
    else:
        _msg = "⏳ 照着提示骨架把循环写出来，看四个 λ 各自的验证 MSE"
    mo.md(_msg)
    return (best_lam_ex3, lam_list_ex3)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 1 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                _idx = _rng.permutation(_n)      # 就这一行：10 课的 permutation，返回打乱的索引
                ```

                切分逻辑三件套：**permutation 拿乱序索引 → round 算三段长度 → 切片捞数据**。
                补完后 `split_fn_ex1 = train_val_test_split`，验证输出应为
                14 / 5 / 5，且三份拼回去恰好 24 个不同的 x（不重叠、不遗漏）。

                三个细节值得咀嚼：

                - 为什么不用 `rng.shuffle`？shuffle 是**原位**打乱（10 课），
                  得先造 `np.arange(24)` 再洗；permutation 一步返回新数组，切分场景更顺手——
                  两者共享同一套随机数机制
                - 换个 seed（比如 7）切出来的三份和演示 1 的不同——**划分本身也是一次随机选择**，
                  所以严肃的实验会固定 seed，并在报告里写清"我们是怎么切的"
                - 顺带一问：为什么用 `round` 不用 `int`？`int` 是直接砍尾巴：
                  24×0.2 = 4.8 → 4；`round` 四舍五入 → 5。两种都能用
                  （只要三段拼起来等于 24），round 切出的比例更接近 6 : 2 : 2
                """
            ),
            "✅ 练习 2 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                _degs = np.arange(1, 10)
                _vals = np.zeros(9)
                for _k, _d in enumerate(_degs):
                    _c = np.polyfit(x_train, y_train, _d)
                    _vals[_k] = np.mean((np.polyval(_c, x_val) - y_val) ** 2)
                best_deg_ex2 = int(_degs[np.argmin(_vals)])    # argmin 给下标，degs[下标] 才是次数
                best_val_ex2 = float(_vals.min())
                ```

                答案：**3 次**，val MSE ≈ **0.081**——和演示 3 的谷底分毫不差。

                这段代码就是"**超参数选择**"的完整流程：次数不是梯度下降学出来的，
                是**人扫出来的**——候选值逐个试、验证集打分、取最优。
                以后选神经网络层数、学习率、λ，全是这一个模板。
                另一个冷知识：换个切分（另一个 seed），谷底可能挪到 4——
                超参数的选择依赖验证集，这也是为什么论文要报告"数据怎么切的"。
                切记：**全程不碰测试集**——测试集留给最后一步揭榜（演示 5）。
                """
            ),
            "✅ 练习 3 参考答案（先自己写过再看！）": mo.md(
                """
                四个 λ 的成绩单（9 次、归一化、同一切分）：

                | λ | val MSE | ‖w‖ | 状态 |
                |---|---|---|---|
                | 0.001 | **0.108** | 13.8 | 最优：刚掐住疯魔 |
                | 0.01 | 0.167 | 3.8 | 略紧 |
                | 0.1 | 0.259 | 1.9 | 偏紧 |
                | 1 | 0.457 | 0.9 | 勒死：训练 MSE 0.189，比一条直线（0.158）还差 |

                答案是 **λ = 0.001**。三个体会：

                - **λ 是超参数**：不是梯度学出来的，是人挑的——用**验证集**挑
                  （练习 2 的模板换一组候选值再跑一遍而已）
                - **λ → 0 退化回过拟合**：演示 4 里 λ=0 的疯魔和百万级 ‖w‖ 还历历在目
                - **λ 太大就欠拟合**：所有权重被死死压小、曲线摊平，
                  连 1 次多项式都不如——紧箍咒是防疯的，不是防本事的

                还有一个冷静的观察：0.108 依然不如 3 次多项式的 0.081——
                **容量选错，正则救不回全部**。先选容量（练习 2），再调正则（本练习），
                顺序别颠倒。
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
                1. **过拟合 = 背题 ≠ 学习**：把训练集的噪声也背下来，训练 100 分、
                   新数据翻车；它的对偶是欠拟合（规律都没学够）——两者都不是"学"
                2. **容量旋钮**：多项式次数 = 模型容量。容量越大越有背题的本钱——
                   9 次多项式有 10 个自由参数，24 个点随便它扭；噪声方差 σ² 是误差地板，
                   训练误差一旦低于它，多出来的必是背下来的噪声
                3. **三分天下与各自身份**：train 上课学参数、val 模考选模型、
                   test 高考只碰一次；切分前必须 shuffle（10 课），
                   否则按序切分可能训练集全是一类
                4. **次数-误差曲线 = 诊断图**：训练误差单调降、验证误差 U 型，
                   谷底 = 刚刚好（本课数据：3 次，val 0.081）；左端欠拟合、右端过拟合——
                   训练线穿过噪声地板（σ²）即背噪声实锤；
                   以后选层数、选轮数、选 λ，画的都是它的亲戚
                5. **三件防御武器**：早停（验证回升就停，25 课细讲）、L2 正则（λ·‖w‖²）、
                   更多数据（最朴素也最有效）；λ 也有自己的 U 型——太小不设防、太大勒成平线
                6. **λ·‖w‖² 是 09 课范数的第二次登场**：给权重上"别长太大"的紧箍咒；
                   λ 本身是超参数，用验证集挑——先选容量，再调正则
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
                ## 🎓 第 3 部分收官：经典机器学习热身（3/3）

                三个热身模型打完——第 15 课手写训练循环、第 16 课概率输出与新损失、
                本课诊断与防御过拟合。对着清单清点行囊：

                | 课 | 里程碑：你现在能做到什么 |
                |---|---|
                | **15** 线性回归 | 手写完整训练循环四步曲（前向 → 算损失 → 求梯度 → 更新），MSE 是你人生第一个损失函数 |
                | **16** 逻辑回归 | 把回归掰成分类：概率输出 + 新损失，学会用没见过的数据给模型打分 |
                | **17** 过拟合与数据划分 | 三分天下、次数-误差曲线诊断、三件防御武器——会训练，更会"体检" |

                训练、评估、诊断防御——**经典机器学习的三板斧齐了**。

                用一句话回顾第 3 部分的剧情：第 15 课学会"让它学"（训练循环），
                第 16 课学会"让它说概率"（分类与评估），本课学会"让它别学过头"
                （诊断与防御）——一台完整的"模型养成流水线"，到手了。
                """
            ),
            mo.callout(
                mo.md(
                    """
                    从"会写一个模型"到"会诊断一个模型"，这一步比大多数人想象的重要：
                    往后 30 多课里，你要不断画损失曲线、盯验证集、判断该不该踩刹车——
                    全是本课练出的手感。带着它去见主角。
                    """
                ),
                kind="success",
            ),
            mo.md(
                """
                ---

                **下一课预告** — `18_感知机`（第 4 部分 · 神经网络）：

                热身结束，**主角登场**。第 4 部分把"一层"摞成"多层"、把直线掰弯成任意曲线——
                从感知机起步：一个自己长出决策边界的小神经元；然后是激活函数
                （把直线掰弯的那只手）、MLP（层层摞起来）、反向传播（让梯度穿过每一层）。

                你将从"**调一个模型**"走向"**造一种模型**"。训练循环四步曲一步不变，
                但前向会越来越深——第 14 课的链式法则，终于要火力全开。

                🎆 三个热身模型打完，去见神经元！
                """
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
