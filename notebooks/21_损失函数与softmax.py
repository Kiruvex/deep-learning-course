import marimo

__generated_with = "0.25.2"
app = marimo.App(app_title="21 · 损失函数与softmax", width="medium")


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
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Microsoft YaHei", "SimHei",
        "Noto Sans SC", "WenQuanYi Zen Hei",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # 21 · 损失函数与 softmax：给网络装上计分板

        **从零到 Transformer · 手写深度学习** ｜ 第 4 部分 · 神经网络（4/8）
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📖 本课目标（先读我）": mo.md(
                """
                **这一课只解决一个问题：多分类的"计分板"怎么造？**

                20 课把整机装好了，结尾留了话：输出层后面接什么激活、
                配什么损失，是一对的，不能乱点鸳鸯谱。前两对你们都处过：
                15 课的 MSE（配回归）、16 课的交叉熵（配 sigmoid 二分类）。
                今天补上最后一对、也是现代深度学习出镜率最高的一对——
                **softmax + 交叉熵**，从此网络能回答"十个候选里挑哪个"。

                打法依然稳扎稳打：先看**配对表**（什么输出配什么损失、
                为什么）；再拆 softmax 的三步计算和它的"锐度"性格；
                然后算几笔交叉熵的账（答对、犹豫、自信地错，各罚多少）；
                接着围观一场**数值事故**——logits 冲到 1000，softmax 当场
                溢出成 nan，学会 log-sum-exp 这剂退烧药（09 课浮点噪声
                的第二次清算）；最后验证一个漂亮到不像话的结论：
                **softmax 配交叉熵，梯度恰好是 p − y**——22 课反向传播
                的最后一块拼图。

                学完你将能：

                - 背出**输出层-损失配对表**：恒等+MSE、sigmoid+BCE、
                  softmax+CE，并说出配错的下场
                - 手算 **softmax**（指数、求和、归一三步），
                  并解释"锐度"（温度）怎么由 logits 的尺度决定
                - 手算**多分类交叉熵**，记住瞎猜基线 log K
                  和"自信地错"的重罚
                - 写出**数值稳定**的 softmax / log-sum-exp，
                  知道减 max 这一步在救谁的命
                - 亲手验证 **∂L/∂z = p − y**（对称差分，13 课验钞机）

                前置：第 16 课的交叉熵与"说错话的重罚"、第 15 课的 MSE、
                第 09 课的浮点噪声（今天正式结账）、第 13 课的对称差分、
                第 20 课的输出层与维度账本。

                本课依然没有训练——计分板装好，导航仪（22 课反向传播）
                就差最后一颗螺丝。
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "🎯 概念 1/4 · 配对表：输出层与损失是一对的": mo.md(
                """
                网络的最后一层决定它**说什么话**，损失函数决定**说错话罚多少**。
                两者必须配套，错配会出怪病。三对官配一张表：

                | 任务 | 输出层激活 | 网络在说什么 | 损失 | 出处 |
                |---|---|---|---|---|
                | 回归（预测数值） | 恒等（不接激活） | "我认为答案是 3.2" | MSE | 15 课 |
                | 二分类 | sigmoid | "我押 73% 是正类" | 二元交叉熵 | 16 课 |
                | **多分类（K 选 1）** | **softmax** | "我押：猫 66%、狗 24%、鸟 10%" | **交叉熵** | **本课** |

                配错的下场举两个例子：

                - 回归任务接了 sigmoid：输出被压进 (0, 1)，预测房价 320 万
                  直接没戏——**输出层把网络能说的话限死了**
                - 多分类用 MSE：MSE 假设"差 1 罚 1 分"，但类别 2 和类别 7
                  之间没有"差 5"这回事——**它不懂类别之间的距离是哑的**
                  （16 课讲过 MSE 配 sigmoid 迈不动腿的病，同族并发症）

                一个工程口诀先挂墙上：**softmax 的输入叫 logits**
                （打分，还没归一的那一排数）；以后读论文、看代码，
                `logits` 指的就是它——"网络打出的原始分，还没换成概率"。
                """
            ),
            "🎲 概念 2/4 · softmax：十票归一": mo.md(
                """
                K 个输出神经元各打一个分 z₁…z_K（logits），softmax 把它们
                变成**一套加起来等于 1 的概率**。三步，一步都不难：

                ```
                ① 全部取指数：    e^{z₁}, e^{z₂}, …, e^{z_K}     （全变成正数）
                ② 求和：          S = e^{z₁} + … + e^{z_K}
                ③ 各自除以 S：    p_k = e^{z_k} / S               （归一，总和 = 1）
                ```

                为什么必须是指数？三个理由合起来看：

                - **全正**：概率不许是负数，e^z 恰好恒正
                - **放大差距**：指数是"越大的越拉爆"——z 差 1 分，
                  概率差约 2.7 倍；差 5 分，差 148 倍。**大者恒大**，
                  奖励自信（演示 1 的缩放阶梯看得最清楚）
                - **可导且脾气好**：处处光滑，导数好算（22 课的 p−y
                  之所以能那么漂亮，指数功不可没）

                logits 的**尺度**决定分布的**锐度**（行业黑话叫"温度"）：
                把 z 整体除以 T 再 softmax——T 小（打分放大）→ 分布尖锐，
                一家独大；T 大（打分压扁）→ 分布平坦，人人差不多。
                训练的完整故事就是：**网络从"接近均匀的瞎猜"（z 全 0，
                p = 1/K）一步步走向"自信的正确"（正确类的 z 拉开差距）**。

                一个漂亮的冷知识：K = 2 时 softmax **就是** sigmoid
                （把两类的分差喂进去，出来的概率分毫不差）——
                softmax 是 sigmoid 的多类扩展，不是新发明。演示 1 验给你看。
                """
            ),
            "⚖️ 概念 3/4 · 交叉熵的多分类版：说错话的重罚（扩编）": mo.md(
                """
                16 课的交叉熵只有两个选项（是/否），今天扩编成 K 个。

                标签先用 **one-hot** 编码：真值是第 3 类，y 就写成
                (0, 0, 0, 1, 0, …, 0)——只有正确类的位置是 1，其余全 0。
                交叉熵：

                ```
                L = − Σ_k  y_k · log p_k
                ```

                one-hot 的妙处：求和号里**只有正确类那一项活下来**（其余
                项乘 0 阵亡），所以实际算的永远是：

                ```
                L = − log p_正确类
                ```

                三个锚点数字（演示 3 亲手算，先记住方向）：

                - **瞎猜**（p = 1/K 均匀）：L = log K。三类是 **1.099**，
                  十类是 2.303——这是"什么都不学"的起点，训练的第一步
                  损失就从这里出发（16 课的 ln 2 是它的二类版）
                - **自信地错**：正确类只押了 5% → L = −log 0.05 ≈ **3.0**，
                  比瞎猜还疼两倍——"说错话的重罚"，多分类版原样继承
                - **完美预测**：p_正确 → 1，L → 0。罚分只认"正确类押了多少"，
                  别的类怎么分剩余的概率，它一概不管
                """
            ),
            "🧯 概念 4/4 · 数值安全与 p − y：两件传家宝": mo.md(
                """
                **传家宝一：减 max（log-sum-exp 技巧）。**

                float64 能装的最大数约 1.8×10³⁰⁸，而 e^710 就已经爆表。
                logits 冲到 1000（深度网络里真的会发生），第①步
                `np.exp(z)` 直接吐 inf，第③步 inf/inf = **nan**——
                整个网络当场死机（09 课的浮点噪声，结账的时候到了）。

                解药简单得离谱：**先给所有 logits 减去最大值再做 softmax**。

                ```
                p_k = e^{z_k − m} / Σ e^{z_j − m}        （m = max(z)）
                ```

                减同一个数不改变 softmax 的结果（分子分母同乘 e^{−m}，
                约掉了），但把最大的指数压回 e⁰ = 1——**一步退烧**。
                它的 log 版本叫 **log-sum-exp**：
                log Σ e^{z_j} = m + log Σ e^{z_j − m}，
                概率里最常用的一块积木（45 课位置编码还会遇到它的亲戚）。

                **传家宝二：∂L/∂z = p − y。**

                softmax 配交叉熵，损失对 **logits** 的梯度是：

                ```
                ∂L/∂z_k = p_k − y_k        （预测概率减 one-hot 真值）
                ```

                漂亮得不像话：**预测对了几成，梯度就剩几成的错误**。
                押 90% 押对了 → 梯度只剩 0.1，轻轻修；押 5% 押错了 →
                梯度 −0.95，往死里修。这个"错多少、学多少"的比例阀，
                是 22 课反向传播穿过输出层的标准动作——先在这课用
                对称差分验明正身（演示 5），22 课直接上岗。
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

        五组演示，把计分板装上：

        - **演示 1** · softmax 手算：三步走一遍 + 缩放阶梯 + sigmoid 亲戚
        - **演示 2** · 锐度旋钮：同一组打分，从平坦到尖锐的五连拍
        - **演示 3** · 交叉熵三案例：答对、犹豫、自信地错，各罚多少
        - **演示 4** · 事故现场：logits 1000 的溢出与退烧药
        - **演示 5** · p − y 验明正身：对称差分 vs 公式，一行不差
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 1 · softmax 手算：三步走一遍

        logits = (2.0, 1.0, 0.1)——比如"猫 2 分、狗 1 分、鸟 0.1 分"。
        三步：取指数、求和、归一。先手算再看代码对账：

        | 候选 | z | e^z | ÷ S = p |
        |---|---|---|---|
        | 猫 | 2.0 | 7.389 | **0.659** |
        | 狗 | 1.0 | 2.718 | **0.242** |
        | 鸟 | 0.1 | 1.105 | **0.099** |
        | 合计 | — | S = 11.212 | 1.000 |

        再验那个冷知识：**两类的 softmax 就是 sigmoid**——
        softmax((z, 0)) 的第一个分量，和 σ(z) 在 z = 0、±2、−1.5 处
        分毫不差（代码里逐一核对）。16 课的老朋友，原来一直是
        softmax 的化身。
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # 演示 1 · softmax 三步手算 + 两类=sigmoid 验证
    _z = np.array([2.0, 1.0, 0.1])                 # logits：猫 / 狗 / 鸟
    _e = np.exp(_z)                                # ① 指数：全变正
    _S = _e.sum()                                  # ② 求和
    _p = _e / _S                                   # ③ 归一：总和 = 1

    # 两类 softmax = sigmoid 的验证
    _sig = lambda _v: 1 / (1 + np.exp(-_v))
    _pairs = []
    for _zv in [0.0, 2.0, -1.5]:
        _p2 = np.exp(np.array([_zv, 0.0])) / np.exp(np.array([_zv, 0.0])).sum()
        _pairs.append((_zv, _p2[0], _sig(_zv)))

    _fig, _ax = plt.subplots(figsize=(6.8, 3.8))
    _bars = _ax.bar(["猫 (z=2.0)", "狗 (z=1.0)", "鸟 (z=0.1)"], _p,
                    color=["tab:blue", "tab:orange", "tab:green"], alpha=0.88)
    for _b, _pv in zip(_bars, _p):
        _ax.text(_b.get_x() + _b.get_width() / 2, _pv + 0.02, f"{_pv:.3f}",
                 ha="center", fontsize=11, weight="bold")
    _ax.axhline(1/3, color="gray", ls=":", lw=1.2)
    _ax.text(2.42, 0.345, "瞎猜线 1/3", fontsize=9, color="gray", ha="center")
    _ax.set_ylabel("概率 p")
    _ax.set_ylim(0, 0.8)
    _ax.set_title("softmax((2.0, 1.0, 0.1))：一套加和为 1 的押注")
    _ax.grid(True, alpha=0.3, axis="y")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 2 · 锐度旋钮：从平坦到尖锐

        同一组打分 (2.0, 1.0, 0.1)，整体**乘一个系数 c** 再 softmax
        （等价于除以温度 T = 1/c）。五连拍从 c = 0.25 到 c = 10：

        | 打分放大倍数 c | 概率分布 | 性格 |
        |---|---|---|
        | 0.25（T=4，发蔫） | (0.42, 0.32, 0.26) | 接近瞎猜，三成对三成 |
        | 0.5 | (0.50, 0.30, 0.19) | 略有倾向 |
        | 1（原样） | (0.66, 0.24, 0.10) | 有主见 |
        | 2 | (0.86, 0.12, 0.02) | 相当自信 |
        | 10（T=0.1，刀尖） | (1.00, 0.00, 0.00) | 一家独大，其余归零 |

        这张表就是**训练进度条**：网络初始化时输出层打分接近全 0
        （p = 均匀的 1/K，损失 = log K 的起点）；随着训练，
        正确类的 z 被逐步拉开——**分布从平坦长出尖峰**。
        45 课讲 GPT 采样时，"温度"这个旋钮还会回来（低温度 = 尖 = 保守，
        高温度 = 平 = 敢说骚话）。
        """
    )
    return


@app.cell
def _(np, plt):
    # 演示 2 · 锐度五连拍：同一 logits 乘不同系数后的 softmax
    _z = np.array([2.0, 1.0, 0.1])
    _cs = [0.25, 0.5, 1.0, 2.0, 10.0]              # 打分放大倍数（温度 T = 1/c）
    _labels = ["猫", "狗", "鸟"]
    _colors = ["tab:blue", "tab:orange", "tab:green"]

    _fig, _axes = plt.subplots(1, 5, figsize=(13.2, 3.4), sharey=True)
    for _ax, _c in zip(_axes, _cs):
        _zz = _z * _c                              # 打分整体放大/压扁
        _p = np.exp(_zz) / np.exp(_zz).sum()       # softmax（这组数不会溢出，先裸写）
        _ax.bar(_labels, _p, color=_colors, alpha=0.88)
        _ax.set_title(f"c = {_c:g}\n(T = {1/_c:g})", fontsize=10)
        _ax.set_ylim(0, 1.05)
        _ax.grid(True, alpha=0.3, axis="y")
        if _c == _cs[0]:
            _ax.set_ylabel("概率 p")
    _fig.suptitle("锐度旋钮：打分放大 → 分布长出尖峰", fontsize=12.5)
    _fig.tight_layout(rect=(0, 0, 1, 0.93))
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 3 · 交叉熵三案例：答对、犹豫、自信地错

        真值固定为**猫**（one-hot = (1, 0, 0)），看三种押注各罚多少：

        | 押注 | p | L = −log p_猫 | 点评 |
        |---|---|---|---|
        | 答对且自信 | (0.70, 0.20, 0.10) | **0.357** | 押对七成，罚 0.36 |
        | 犹豫不决 | (0.34, 0.33, 0.33) | **1.079** | 几乎瞎猜（基线 log 3 = 1.099） |
        | 自信地错 | (0.05, 0.90, 0.05) | **2.996** | 把宝押在狗身上，罚到瞎猜的 2.7 倍 |

        注意两个方向：**从 0.357 到 1.079**，"少押对的"每少一成，
        罚分平稳上涨（log 的缓坡）；**从 1.079 到 2.996**，押错方向
        再自信，罚分继续拉高——交叉熵从不封顶，**越自信地错越疼**
        （16 课的祖训，K 类版原样继承）。

        还有个工程细节：只罚"正确类押了多少"，**不管剩下的概率
        怎么分**——押猫 0.7 之后，狗 0.2 鸟 0.1 还是狗 0.1 鸟 0.2，
        罚分一模一样。交叉熵眼里只有一件事：**给真类留了多少席位**。
        """
    )
    return


@app.cell
def _(np, plt):
    # 演示 3 · 交叉熵三案例（真值 = 猫）
    _cases = [
        ("答对且自信\n(0.70, 0.20, 0.10)", 0.3567, "tab:blue"),
        ("犹豫不决\n(0.34, 0.33, 0.33)", 1.0788, "tab:gray"),
        ("自信地错\n(0.05, 0.90, 0.05)", 2.9957, "tab:red"),
    ]

    _fig, _ax = plt.subplots(figsize=(7.6, 4.2))
    _xs = np.arange(len(_cases))
    _vals = [_c[1] for _c in _cases]
    _bars = _ax.bar(_xs, _vals, color=[_c[2] for _c in _cases], alpha=0.88, width=0.55)
    for _b, _v in zip(_bars, _vals):
        _ax.text(_b.get_x() + _b.get_width() / 2, _v + 0.06, f"{_v:.3f}",
                 ha="center", fontsize=11, weight="bold")
    _ax.axhline(np.log(3), color="gray", ls=":", lw=1.4)
    _ax.text(2.32, np.log(3) + 0.05, "瞎猜基线 log 3 = 1.099",
             fontsize=9, color="gray", ha="center")
    _ax.set_xticks(_xs)
    _ax.set_xticklabels([_c[0] for _c in _cases], fontsize=9.5)
    _ax.set_ylabel("交叉熵损失 L（真值 = 猫）")
    _ax.set_ylim(0, 3.5)
    _ax.set_title("说错话的重罚：越自信地错，越疼")
    _ax.grid(True, alpha=0.3, axis="y")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 4 · 事故现场：logits 冲到 1000

        左图是**案发现场**：e^z 这条曲线在 z ≈ 709 处冲破 float64 的
        天花板（约 1.8×10³⁰⁸），再往前一步就是 inf。

        右边是**验尸报告**：logits = (1000, 1000.1)——模型极度自信地
        认为第二类略好。三行对照：

        | 做法 | 结果 | 死因 |
        |---|---|---|
        | 裸写 softmax：exp(1000) 先算 | **inf → nan** | e^1000 ≈ 10⁴³⁴，float64 装不下 |
        | 先减 max 再 softmax | **(0.475, 0.525)** | 最大的指数被压回 e⁰ = 1，退烧成功 |
        | log-sum-exp：m + log Σ e^{z−m} | **1000.744** | 同一剂药的水合版本 |

        妙处在于：**减 max 一点都不改变答案**——分子分母同乘 e^{−m}
        约得干干净净，概率一个数都不变，变的只是"数值还活着"。

        顺带一提另一起连环事故：交叉熵要算 −log p，而 softmax 输出的 p
        可能下溢到 0（比如 1e-400）——log(0) = −inf，损失当场炸成 inf。
        工程解法是**根本不单独算 p**：把 log(softmax) 合并成
        `z − logsumexp(z)` 一步到位（log 里永远是负得有限的书）。
        本课的迷你版先记口诀：**减 max，再取指，log 里面别单独算 p**。
                """
    )
    return


@app.cell
def _(mo, np, plt):
    # 演示 4 · 溢出现场：e^z 曲线冲破 float64 天花板 + 三行验尸
    _z_bad = np.array([1000.0, 1000.1])

    with np.errstate(over="ignore"):               # 预期会溢出，让它溢个明白
        _e_bad = np.exp(_z_bad)                    # → (inf, inf)
        _p_naive = _e_bad / _e_bad.sum()           # inf/inf → (nan, nan)

    _p_stable = np.exp(_z_bad - _z_bad.max())      # 先减 max（退烧药）
    _p_stable = _p_stable / _p_stable.sum()
    _lse = _z_bad.max() + np.log(np.exp(_z_bad - _z_bad.max()).sum())   # log-sum-exp

    # ---- 左图：e^z 曲线冲出画面（x 只画到 40，曲线在 z≈30 冲破画面顶）----
    _zz = np.linspace(0, 40, 400)
    _ee = np.exp(_zz)                             # 40 以内还炸不了，放心算
    _fig, _ax = plt.subplots(figsize=(7.6, 4.3))
    _ax.semilogy(_zz, _ee, color="tab:red", lw=2.6, label="e^z（指数爆炸）")
    _ax.annotate("z ≈ 30：曲线冲出画面顶\n（已是 10¹³）",
                 xy=(30, 1e13), xytext=(4, 3e8), fontsize=10, color="tab:red",
                 arrowprops=dict(arrowstyle="->", color="tab:red"))
    _ax.text(31, 30, "曲线继续右上：z = 710 处\n冲破 float64 天花板\n1.8×10³⁰⁸ → inf",
             fontsize=10, color="tab:red", ha="left")
    _ax.set_xlim(0, 40)
    _ax.set_ylim(1e-1, 1e13)
    _ax.set_xlabel("z（logit）")
    _ax.set_ylabel("e^z（log 尺）")
    _ax.set_title("案发现场：指数曲线一骑绝尘")
    _ax.legend(fontsize=9, loc="upper left")
    _ax.grid(True, alpha=0.3, which="both")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 5 · p − y 验明正身

        概念 4 的传家宝二，按 13 课的规矩过验钞机。
        logits = (2.0, 1.0, 0.5)，真值 = 第 0 类（one-hot (1, 0, 0)）。

        先算概率：p = (0.629, 0.231, 0.140)。然后两路对账：

        - **数值梯度**：对每个 z_i 单独拧动 ±10⁻⁶（对称差分），
          算损失变化率——每个要跑两次 softmax + 交叉熵
        - **公式 p − y**：一行减法

        | 分量 | 数值梯度 | p − y | 差 |
        |---|---|---|---|
        | ∂L/∂z₁ | −0.37146828 | −0.37146828 | < 10⁻¹⁰ |
        | ∂L/∂z₂ | +0.23122390 | +0.23122390 | < 10⁻¹⁰ |
        | ∂L/∂z₃ | +0.14024438 | +0.14024438 | < 10⁻¹⁰ |

        读一眼数字的含义：z₁ 是正确类，押了 0.629 → 梯度 −0.371
        （"再多押 37 个百分点就对了"）；z₂、z₃ 是错误类 → 梯度恰好是
        它们各自抢走的概率（"抢了多少，退回多少"）。**三行梯度加起来
        恰好是 0**——概率往一处加，别处就得减，softmax 的守恒律
        直接写进了梯度里。

        22 课的反向传播走到输出层，抄的就是这一行。
        """
    )
    return


@app.cell
def _(mo, np):
    # 演示 5 · p − y 数值验证（13 课对称差分）
    def _softmax(_zz):
        _sh = _zz - _zz.max()                      # 退烧药（演示 4 刚学）
        _e = np.exp(_sh)
        return _e / _e.sum()

    _z = np.array([2.0, 1.0, 0.5])
    _y = np.array([1.0, 0.0, 0.0])                 # 真值：第 0 类
    _p = _softmax(_z)

    def _ce(_zz):                                  # 损失：−log 正确类概率
        return -np.log(_softmax(_zz)[_y == 1][0] + 1e-300)

    _h = 1e-6
    _num = np.zeros(3)
    for _i in range(3):                            # 逐分量对称差分（13 课）
        _zp, _zm = _z.copy(), _z.copy()
        _zp[_i] += _h
        _zm[_i] -= _h
        _num[_i] = (_ce(_zp) - _ce(_zm)) / (2 * _h)

    _ana = _p - _y                                 # 公式：p − y
    _diff = float(np.max(np.abs(_num - _ana)))

    mo.md(
        f"""
        | 量 | 值 |
        |---|---|
        | 概率 p | **({', '.join(f'{_v:.4f}' for _v in _p)})** |
        | 数值梯度 | ({', '.join(f'{_v:+.6f}' for _v in _num)}) |
        | p − y | ({', '.join(f'{_v:+.6f}' for _v in _ana)}) |
        | 最大误差 | **{_diff:.1e}**（对称差分与公式的差距） |
        | 梯度之和 | **{_num.sum():+.1e}**（softmax 的守恒律：加起来是 0） |
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 🎛️ 交互实验

        **押注台**——三个滑块调 logits（三类候选的打分），下拉框选真值，
        实时看概率怎么分、交叉熵怎么跳。玩法建议：

        1. 先把真值设为"猫"、三个滑块摆成 (2, 1, 0)——答对且自信的
           标准姿势，损失 0.35 上下
        2. 把 z_猫 拖到 −3：**自信地错**，损失冲上 3+——看着那个红字
           想想"反向传播会怎么修它"（p − y 的方向）
        3. 三个滑块全归零：**均匀瞎猜**，损失精确停在 log 3 = 1.099
           ——训练的起跑线，亲手踩一次
        4. 把三个滑块同时加 2：分布**一点不变**——softmax 只看**差距**
           不看绝对值（给所有候选加分等于谁都没加）
        """
    )
    return


@app.cell
def _(mo):
    # 交互控制：三个 logits + 真值下拉（debounce：松手才重算）
    z1_slider = mo.ui.slider(-5.0, 5.0, value=2.0, step=0.1, debounce=True,
                             label="🎛️ z₁（猫的打分）")
    z2_slider = mo.ui.slider(-5.0, 5.0, value=1.0, step=0.1, debounce=True,
                             label="🎛️ z₂（狗的打分）")
    z3_slider = mo.ui.slider(-5.0, 5.0, value=0.0, step=0.1, debounce=True,
                             label="🎛️ z₃（鸟的打分）")
    true_dropdown = mo.ui.dropdown(options=["猫", "狗", "鸟"], value="猫",
                                    label="🎯 真值是哪一类？")
    mo.vstack([z1_slider, z2_slider, z3_slider, true_dropdown])
    return (true_dropdown, z1_slider, z2_slider, z3_slider)


@app.cell
def _(mo, np, plt, true_dropdown, z1_slider, z2_slider, z3_slider):
    # ===== 交互实验 · 押注台（响应单元）=====
    _z = np.array([z1_slider.value, z2_slider.value, z3_slider.value])
    _names = ["猫", "狗", "鸟"]
    _k_true = _names.index(true_dropdown.value or "猫")

    _sh = _z - _z.max()                            # 减 max（演示 4 的退烧药，顺手用上）
    _e = np.exp(_sh)
    _p = _e / _e.sum()
    _L = -np.log(_p[_k_true] + 1e-300)             # 交叉熵：−log 正确类概率

    _fig, _ax = plt.subplots(figsize=(7.0, 4.0))
    _colors = ["tab:blue", "tab:orange", "tab:green"]
    _bars = _ax.bar(_names, _p, color=_colors, alpha=0.88)
    _bars[_k_true].set_edgecolor("black")          # 真值候选描黑边
    _bars[_k_true].set_linewidth(2.5)
    for _b, _pv in zip(_bars, _p):
        _ax.text(_b.get_x() + _b.get_width() / 2, _pv + 0.02, f"{_pv:.3f}",
                 ha="center", fontsize=11, weight="bold")
    _ax.axhline(1 / 3, color="gray", ls=":", lw=1.2)
    _ax.text(2.45, 0.345, "瞎猜线 1/3", fontsize=9, color="gray", ha="center")
    _ax.set_ylim(0, 1.02)
    _ax.set_ylabel("概率 p")
    _ax.set_title(f"押注台：真值 = {_names[_k_true]}（黑框），损失 L = {_L:.3f}")
    _ax.grid(True, alpha=0.3, axis="y")
    _fig.tight_layout()
    _fig

    # ---- 判词 ----
    if _p[_k_true] > 0.6:
        _verdict = (f"✅ 答对且自信：押了 {_p[_k_true]:.0%}，损失 {_L:.2f}——"
                    f"反向传播只轻推（p−y = {_p[_k_true] - 1:+.2f}）")
    elif _p[_k_true] < 0.15:
        _verdict = (f"🚨 自信地错（或快了）：真值只押 {_p[_k_true]:.0%}，损失 {_L:.2f}"
                    f"——p−y = {_p[_k_true] - 1:+.2f}，往死里修")
    else:
        _verdict = f"🤔 犹豫区间：押 {_p[_k_true]:.0%}，损失 {_L:.2f}（瞎猜基线 log 3 = 1.099）"

    _callout = mo.callout(mo.md(f"**实时判词：{_verdict}**"), kind="info")
    mo.vstack([_fig, _callout])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✏️ 动手练习（每课必做，本课 3 题）": mo.md(
                """
                **练习 1 · 手写稳定版 softmax**：骨架已给，只差"减 max"
                那一行——写完用两组 logits 验收，一组正常、一组会溢出的。

                **练习 2 · 手写交叉熵**：one-hot 版公式一行流，
                用演示 3 的两个案例对账。

                **练习 3 · 挑战 · 亲手验证 p − y**：对 logits = (2, 1, 0.5)、
                真值 = 第 0 类，用对称差分算出三个数值梯度，
                填"数值梯度与 p − y 的最大绝对差"——演示 5 的代码化。

                规则不变：先把 `None` 换成你的答案再运行；写完再看参考答案。
                """
            )
        }
    )
    return


@app.cell
def _(mo, np):
    # ===== 练习 1 · 手写稳定版 softmax =====
    def softmax_ex1(z):
        z = np.asarray(z, dtype=float)
        z_shifted = None        # ← ① 补这行：z - z.max()（退烧药）
        e = np.exp(z_shifted)
        return e / e.sum()

    softmax_ready_ex1 = None    # ← ② 补完后，把这行换成 softmax_ex1（就写函数名）

    if softmax_ready_ex1 is not None:
        try:
            _ok_a = np.allclose(softmax_ready_ex1([2.0, 1.0, 0.5]),
                                [0.628532, 0.231224, 0.140244])
            _ok_b = np.allclose(softmax_ready_ex1([1000.0, 1000.1]),
                                [0.475021, 0.524979])
            if _ok_a and _ok_b:
                _msg = "✅ 两组验收全过——包括 logits=1000 那组：减 max 救了命"
            elif _ok_a and not _ok_b:
                _msg = "❌ 普通组对、溢出组炸了（nan）——① 的减 max 补了吗？"
            else:
                _msg = "❌ 普通组就对不上——三步顺序：指数 → 求和 → 归一"
        except TypeError:
            _msg = "⏳ ① 那行还是 None，np.exp(None) 会炸——先补上"
    else:
        _msg = "⏳ 先补 ① 的减 max，再把 softmax_ready_ex1 换成 softmax_ex1"
    mo.md(_msg)
    return (softmax_ex1, softmax_ready_ex1)


@app.cell
def _(mo, np):
    # ===== 练习 2 · 手写交叉熵 =====
    def cross_entropy_ex2(p, y):
        """p：概率向量；y：one-hot 真值。返回交叉熵（一个数）"""
        return None    # ← ① 补：-np.sum(y * np.log(p))（本题的 p 都远离 0，先不管下溢）

    ce_ready_ex2 = None    # ← ② 补完后，把这行换成 cross_entropy_ex2

    if ce_ready_ex2 is not None:
        try:
            _v1 = float(ce_ready_ex2(np.array([0.7, 0.2, 0.1]), np.array([1.0, 0.0, 0.0])))
            _v2 = float(ce_ready_ex2(np.array([0.05, 0.9, 0.05]), np.array([1.0, 0.0, 0.0])))
            if abs(_v1 - 0.3567) < 1e-3 and abs(_v2 - 2.9957) < 1e-3:
                _msg = f"✅ 两笔账都对：答对罚 {_v1:.3f}，自信错罚 {_v2:.3f}"
            else:
                _msg = f"❌ 你算出 {_v1:.3f} 和 {_v2:.3f}（期望 0.357 和 2.996）——注意 log 的底是 e"
        except TypeError:
            _msg = "⏳ 返回值还是 None——① 的公式补了吗？"
    else:
        _msg = "⏳ 先补 ①，再把 ce_ready_ex2 换成 cross_entropy_ex2"
    mo.md(_msg)
    return (ce_ready_ex2, cross_entropy_ex2)


@app.cell
def _(mo, np):
    # ===== 练习 3 · 挑战：验证 p − y =====
    _z = np.array([2.0, 1.0, 0.5])                 # logits
    _y = np.array([1.0, 0.0, 0.0])                 # 真值：第 0 类
    _h = 1e-6                                      # 扰动量（13 课：1e-5 ~ 1e-7 的甜点区）

    # 提示骨架：
    #   def _ce(zz):                                # 损失（softmax + 交叉熵合并）
    #       _sh = zz - zz.max()                     # 别忘了退烧药
    #       _p = np.exp(_sh) / np.exp(_sh).sum()
    #       return -np.log(_p[0] + 1e-300)          # 真值是第 0 类
    #   _num = np.zeros(3)
    #   for i in range(3):
    #       _zp = _z.copy(); _zp[i] += _h           # 只拧动第 i 个分量
    #       _zm = _z.copy(); _zm[i] -= _h
    #       _num[i] = (_ce(_zp) - _ce(_zm)) / (2 * _h)
    #   _ana = softmax(_z) - _y                     # 公式 p − y
    #   max_diff_ex3 = 最大绝对差

    max_diff_ex3 = None    # ← 数值梯度与 p − y 的最大绝对差（应远小于 1e-6）

    if max_diff_ex3 is not None:
        if float(max_diff_ex3) < 1e-6:
            _msg = f"✅ 最大误差 {float(max_diff_ex3):.1e}——p − y 验明正身，22 课直接上岗"
        else:
            _msg = (f"❌ 误差 {float(max_diff_ex3):.1e} 太大——检查：h 是不是 1e-6 量级？"
                    "退烧药（减 max）加了吗？")
    else:
        _msg = "⏳ 照提示骨架把对称差分写出来，填上最大误差"
    mo.md(_msg)
    return (max_diff_ex3,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 1 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                z_shifted = z - z.max()      # 就这一行：先把最大的指数压回 e⁰
                ```

                验收：softmax((2, 1, 0.5)) = (0.6285, 0.2312, 0.1402)；
                softmax((1000, 1000.1)) = (0.4750, 0.5250)——后者不减 max
                直接 nan。

                两个细节值得咀嚼：

                - **为什么结果不变**：softmax 的分子分母同乘 e^{−m}，
                  约分后一个数都不动——变的只是"数值活没活着"。
                  数学等价、数值不等价，这是浮点世界的常见剧情（09 课）
                - **为什么是减 max 不是减第一个**：减谁都能活，
                  但减 max 保证所有指数 ≤ e⁰ = 1，连"最小的那个爆不爆"
                  都不用想——一劳永逸的写法
                """
            ),
            "✅ 练习 2 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                return -np.sum(y * np.log(p))
                ```

                两笔账：答对罚 **0.357**（−log 0.7）、自信错罚 **2.996**
                （−log 0.05）。

                三个细节：

                - one-hot 相乘就是"**只留正确类**"——其余项乘 0 阵亡，
                  所以这行代码在算 −log p_正确，求和号只是礼貌
                - log 的底必须是 **e**（自然对数）——瞎猜基线才是 log K
                  （三类 1.099、十类 2.303）。用 log10 的话基线变成
                  log₁₀K，整条损失曲线被换了刻度，超参数也得跟着重调
                - 生产代码里别这么裸写：p 一旦下溢到 0，log(0) = −inf。
                  稳妥做法是 log(softmax) 合并计算（演示 4 的口诀），
                  或者像本课一样加一个 1e-300 的保底
                """
            ),
            "✅ 练习 3 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                def _ce(zz):
                    _sh = zz - zz.max()
                    _p = np.exp(_sh) / np.exp(_sh).sum()
                    return -np.log(_p[0] + 1e-300)

                _num = np.zeros(3)
                for i in range(3):
                    _zp, _zm = _z.copy(), _z.copy()
                    _zp[i] += _h
                    _zm[i] -= _h
                    _num[i] = (_ce(_zp) - _ce(_zm)) / (2 * _h)

                _p = np.exp(_z - _z.max()); _p = _p / _p.sum()
                max_diff_ex3 = float(np.max(np.abs(_num - (_p - _y))))   # ≈ 1e-10
                ```

                典型误差 **10⁻¹⁰ ~ 10⁻¹¹**（h = 10⁻⁶ 时）——对称差分的
                精度极限（13 课演示 2 的 U 型：h 再小，浮点噪声就上台了）。

                这套"**手写梯度 → 数值对账**"的流程，是 22、23 两课的
                标准动作：反向传播每写一层，就对一层账。工业界叫
                **gradient check**，字面意思：梯度也要查账。
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
                1. **配对表**：恒等+MSE（回归）、sigmoid+BCE（二分类）、
                   softmax+CE（多分类）——输出层决定网络说什么话，
                   损失决定说错话罚多少；错配会出怪病
                   （sigmoid 压死回归输出、MSE 不懂类别距离）
                2. **softmax 三步**：指数 → 求和 → 归一；
                   全正、和为 1、大者恒大（指数放大差距）。
                   两类 softmax = sigmoid（K=2 的化身）；logits 是
                   "还没归一的打分"
                3. **锐度（温度）**：z 整体缩放改变分布尖锐度——
                   c=0.25 接近瞎猜 (0.42, 0.32, 0.26)，c=10 一家独大
                   (1, 0, 0)；训练 = 从平坦长出尖峰（45 课采样温度再见）
                4. **交叉熵的锚点**：瞎猜基线 log K（三类 1.099、十类 2.303）、
                   自信错 ≈ 3.0（押 0.05）、完美 → 0；
                   只罚"真类留了多少席位"，不管剩余概率怎么分
                5. **减 max 退烧药**：e^z 在 z≈710 爆 float64（1.8×10³⁰⁸），
                   logits=1000 时裸写 softmax → nan；
                   减 max 数学等价、数值救命；log-sum-exp 是它的水合版本
                6. **∂L/∂z = p − y**：softmax 配交叉熵对 logits 的梯度
                   就是"预测减真值"——错多少学多少，梯度之和为 0
                   （softmax 的守恒律）；数值验证误差 10⁻¹⁰，
                   22 课输出层的标准动作
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

        **下一课预告** — `22_反向传播上`：

        计分板装好了，万事俱备——**只差梯度**。下一课，全课程最重要
        的一课，正式开讲**反向传播**（backpropagation）。

        好消息是：你早就见过它的海报。14 课画过的那张计算图
        （x →×w₁→ u₁ →×w₂→ u₂ →平方→ L）就是预告片——这次把它放成
        正片：**前向算值、反向传梯度**，每个节点只干一件事
        （把自己的局部导数乘上游传来的数，往左递）。

        你会从一个**标量版微型网络**起步（5 个参数，手推每一步），
        把每条梯度都对完对称差分的账，然后亲眼看着它被 200 步
        梯度下降训练到损失 0.01 以下——**第一次，参数是自己学会的**。

        15 课的四步曲、14 课的链式法则、19 课的导数、21 课的 p−y，
        全部在这一课合龙。深呼吸，去见深度学习的心脏。

        🎆 计分板亮灯，导航仪启动！
        """
    )
    return


if __name__ == "__main__":
    app.run()
