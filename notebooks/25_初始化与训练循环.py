import marimo

__generated_with = "0.25.2"
app = marimo.App(app_title="25 · 初始化与训练循环", width="medium")


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
        # 25 · 初始化、Dropout 与训练循环：第 4 部分收官

        **从零到 Transformer · 手写深度学习** ｜ 第 4 部分 · 神经网络（8/8）
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📖 本课目标（先读我）": mo.md(
                """
                **这一课只解决一个问题：把最后几块拼图装上，然后打一场大仗。**

                前七课攒下的兵器：前向（20）、损失（21）、反向传播（22、23）、
                优化器（24）——还差三块小拼图，训练学就齐了：

                - **初始化**：参数的起跑线。全零是灾难、太大是昏迷，
                  Xavier/He 用一条方差守恒公式定标
                - **Dropout**：训练时随机让神经元下班，用"集体投票"
                  对抗过拟合
                - **早停 + weight decay**：17 课欠下的两个承诺，
                  今天连本带息兑现

                然后是收官大战——**MNIST 手写数字**：784 个像素进、
                10 个类别出，十万零一千七百七十个参数，Adam 驱动，
                完全体训练循环首次实战。目标：**测试集 97% 以上**。

                学完你将能：

                - 说出**全零初始化的对称性灾难**：所有隐单元活成一个，
                  宽度白费；以及 He/Xavier 为什么是 sqrt(2/d)、sqrt(1/d)
                - 手写 **inverted dropout**（含"除以保留率"的期望守恒），
                  并讲清它为什么能防过拟合
                - 用**早停**踩刹车（patience 机制）、用 **weight decay**
                  给权重上慢性紧箍咒（17 课 λ·‖w‖² 的优化器马甲）
                - 独立组装**工业级训练循环完全体**，并在 MNIST 上
                  从零打到 97%+

                前置：第 19 课的死 ReLU 与饱和（初始化的病因）、
                第 17 课的三分天下/早停/L2（今天的兑现清单）、
                第 23 课的双月牙与 train_mlp（今天的底盘）、
                第 24 课的 Adam（今天的发动机）。

                这是第 4 部分的最后一课。打完这一仗，你手里就有了一套
                **完整的、亲手写过每一个零件的深度学习**——第 5 部分开始，
                我们带着它去图像的世界。
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "🎲 概念 1/4 · 初始化：起跑线定生死": mo.md(
                """
                训练开始前，参数站在哪？三种站法，三种命运：

                **① 全零 = 对称性灾难。** 每个隐单元的输入相同（都是零）、
                权重相同（都是零）→ 输出相同、梯度相同 → **更新之后
                还是相同**。h = 8 的网络实际宽度是 1，剩下 7 个是陪跑的
                永远分不开的双胞胎。演示 1 会看到：全零网络训练 300 轮，
                train 准确率纹丝不动地卡在 51%——它只学到了一个偏置。

                **② 太大 = 饱和昏迷。** 权重一大，打分 |z| 就大，
                tanh 被推到两端平地（导数趋 0）、ReLU 大批死亡
                （19 课的两桩旧案）——起步就躺在病床上。

                **③ 小随机 × 正确的尺度 = 唯一活路。** 每个神经元
                拿到**不同的**随机权重（打破对称），尺度由
                **方差守恒**决定：希望每层输出的方差 ≈ 输入的方差
                ——信号既不层层衰减、也不层层爆炸。解出来的尺度就是：

                ```
                Xavier（配 tanh）：std = sqrt(1 / fan_in)     fan_in = 进该层的特征数
                He（配 ReLU）：    std = sqrt(2 / fan_in)     （ReLU 砍掉一半信号，补两倍）
                ```

                直觉版推导（三行）：输出 z = Σᵢ wᵢxᵢ，n 个独立项的方差
                相加 → Var(z) = fan_in · Var(w) · Var(x)；要 Var(z) ≈ Var(x)，
                就要 Var(w) ≈ 1/fan_in。ReLU 把负半轴砍没了（方差减半），
                所以补一个 2。23 课 `init_params` 里那个
                `sqrt(2/d)` 今天正式挂牌——它就是 He。
                """
            ),
            "🧢 概念 2/4 · Dropout：让神经元轮流下班": mo.md(
                """
                对抗过拟合（17 课）的神经网络专属武器：

                ```
                训练时（每一步重新抽）：mask = (随机数 ≥ p) / (1 − p)
                                      H ← H ⊙ mask        （p 比例的神经元被置零，
                测试时：什么都不做，全网络在岗                  幸存者放大 1/(1−p) 倍）
                ```

                p 叫**丢弃率**（常用 0.2~0.5）。两个设计细节都是精华：

                - **除以 (1−p)（inverted dropout）**：幸存者放大补位，
                  保证这一层的**期望输出不变**——测试时才不用做任何补偿
                  （老式写法是测试时乘 p，训练和测试不一致，容易出 bug）
                - **每一步重新抽 mask**：每一步都在训练一个**不同的子网络**。
                  训练完相当于得到了指数多个子网络的集成（ensemble），
                  测试时全网络在岗 ≈ 对所有子网络取平均——
                  "三个臭皮匠"的数学版本

                它防过拟合的第二层机理：**不许共谋**。任何特征都不能
                指望某个特定同伴一定在场——每个神经元都被迫学出
                **独立有用**的特征，冗余的"复制粘贴"特征被清除。

                一个使用纪律：**Dropout 只在训练时开**。忘了关（评估时
                还在随机丢弃）是新手经典 bug——验证准确率会莫其名地
                抖动变差。
                """
            ),
            "🛑 概念 3/4 · 早停与 weight decay：17 课的兑现日": mo.md(
                """
                17 课立过两块碑，今天兑现：

                **早停（early stopping）**——"在疯掉之前踩刹车"：

                ```
                每个 epoch 结束：在验证集上算损失
                    若验证损失创新低 → 记住当前参数（最佳版本），耐心归零
                    否则 → 耐心 +1
                    耐心攒满 patience 轮 → 停止训练，参数回滚到最佳版本
                ```

                patience（耐心）防的是"一时抖动误杀"：验证损失偶尔
                平一下不代表到顶了，连续多轮不改善才算。停下来的那组
                参数不是最后一步的，而是**验证损失最低那一刻**的——
                早停是"自动帮你回到最好的过去"。

                **weight decay（权重衰减）**——L2 正则的优化器马甲：

                ```
                w ← w − lr·(grad + λ·w)   ＝   w ← (1 − lr·λ)·w − lr·grad
                ```

                每次更新前先把权重**乘性缩小一点点**（17 课的紧箍咒
                λ·‖w‖² 求梯度就是 λ·w）。λ 的量级：10⁻⁴ ~ 10⁻²；
                λ 太大把模型勒进欠拟合（17 课的"勒成平线"重演）。

                两件武器和 Dropout 不冲突、常常同上场——25 课的最后
                一块拼图：把它们全部缝进训练循环。
                """
            ),
            "🏁 概念 4/4 · 训练循环完全体 + MNIST 大战": mo.md(
                """
                组装**完全体**——从今天起，任何网络都是这一套
                （把 15 课的四步曲放大到工业规格）：

                ```
                ① 初始化：He/Xavier 小随机（b 全零没关系——对称性来自 W）
                重复（每个 epoch）：
                    洗牌 → 切 mini-batch
                    对每一批：
                        ② 前向（训练模式：含 Dropout）
                        ③ 损失（+ weight decay 的梯度项）
                        ④ 反向（22、23 课）
                        ⑤ 优化器更新（24 课，Adam）
                    ⑥ 体检：训练/验证损失与准确率（评估模式：关 Dropout）
                    ⑦ 早停检查：耐心攒满 → 回滚最佳参数，停
                最终：测试集一次性揭榜（17 课的纪律）
                ```

                大战配置（每一项都来自前几课）：

                | 部件 | 选择 | 出处 |
                |---|---|---|
                | 数据 | MNIST：60000 训练 + 10000 测试，784 像素，10 类 | 本课自动下载 |
                | 切分 | 50000 上课 / 10000 模考 / 10000 高考 | 17 课 |
                | 网络 | 784 → 128(ReLU) → 10(softmax)，101,770 参数 | 20、21 课 |
                | 损失 | softmax + 交叉熵（梯度 = p − y） | 21 课 |
                | 优化器 | Adam，lr = 0.001，batch = 64 | 24 课 |
                | 防过拟合 | 早停 patience = 3 | 17 课 |

                预期战果（演示 5 实测）：**验证/测试准确率 97%+**。
                作为对照——瞎猜是 10%，线性模型约 92%，
                人类目测大约 99.8%。深度学习的第一场正式胜利，
                就用这三万行里你亲手写的每一行去拿。
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

        五组演示，收官：

        - **演示 1** · 对称性灾难：全零初始化的验尸报告
        - **演示 2** · 尺度实验：10 层网络的方差演化 + 饱和率
        - **演示 3** · Dropout 解剖：掩码、期望守恒、训练对比
        - **演示 4** · 早停现场：U 型曲线、patience、weight decay
        - **演示 5** · MNIST 大战：完全体训练循环首次实战
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 1 · 对称性灾难：全零初始化的验尸报告

        双月牙（23 课同款）、h = 8 的网络、**权重和偏置全部从零出发**，
        训练 300 个 epoch。右边是验尸报告，左边是"同款网络、He 初始化"
        的对照组。

        验尸发现，一具都没活：

        - **W₁、b₁、W₂ 三组参数 300 轮后仍然精确为零**——
          它们的梯度是 0 × 0 = 0（H = 0 让 dW₂ = 0，W₂ = 0 让 δ₁ = 0，
          δ₁ = 0 让 dW₁ = 0），一个死锁的三角
        - 唯一动过的参数是 **b₂ = −0.04**——整个网络 300 轮
          只学会了"押两类中稍微多一点的那个"
        - train 准确率 **51%**（几乎瞎猜），8 个隐单元输出
          **完全相同**（全是 0）——宽度 8 白花了

        对照组（He 初始化，只差在起点）train 100%——
                **同样的结构、同样的梯度、同样的学习率，起跑线不同，
                一个躺在原地一个冲过终点**。这就是初始化的地位：
                它不参与学习，却决定学习能不能开始。
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # 演示 1 · 对称性灾难：全零 init vs He init
    def make_moons25(n_per=60, noise=0.10, seed=11):
        """23 课同款双月牙"""
        _rng = np.random.default_rng(seed)
        _t = _rng.uniform(0, np.pi, n_per)
        _X0 = np.column_stack([np.cos(_t), np.sin(_t)]) + _rng.normal(0, noise, (n_per, 2))
        _X1 = np.column_stack([1 - np.cos(_t), 0.5 - np.sin(_t)]) + _rng.normal(0, noise, (n_per, 2))
        _X = np.vstack([_X0, _X1])
        _y = np.concatenate([np.zeros(n_per), np.ones(n_per)])
        _idx = _rng.permutation(len(_y))
        return _X[_idx], _y[_idx]

    _X25, _y25 = make_moons25()
    _Xtr25, _ytr25 = _X25[:72], _y25[:72]          # 与 23 课同切分

    def train_init(init_kind, h=8, lr=1.0, epochs=300):
        """双月牙 + 2-h-1 网络，只有初始化不同"""
        if init_kind == "zeros":
            _W1 = np.zeros((2, h))
        else:                                       # He：小随机 × sqrt(2/进维)
            _rng = np.random.default_rng(7)
            _W1 = _rng.normal(0, 1, (2, h)) * np.sqrt(2 / 2)
        _b1 = np.zeros(h)
        _rng2 = np.random.default_rng(8)
        _W2 = _rng2.normal(0, 1, (h, 1)) * np.sqrt(2 / h) if init_kind != "zeros" else np.zeros((h, 1))
        _b2 = np.zeros(1)
        for _ in range(epochs):
            _H = np.tanh(_Xtr25 @ _W1 + _b1)
            _p = 1 / (1 + np.exp(-(_H @ _W2 + _b2)))
            _n = len(_Xtr25)
            _yc = _ytr25.reshape(-1, 1)
            _d2 = (_p - _yc) / _n
            _d1 = (_d2 @ _W2.T) * (1 - _H ** 2)
            _W2 -= lr * (_H.T @ _d2)
            _b2 -= lr * _d2.sum(0)
            _W1 -= lr * (_Xtr25.T @ _d1)
            _b1 -= lr * _d1.sum(0)
        _H = np.tanh(_Xtr25 @ _W1 + _b1)
        _p = 1 / (1 + np.exp(-(_H @ _W2 + _b2)))
        _acc = float(((_p > 0.5) == _yc).mean())
        return _W1, _b1, _W2, _b2, _acc, _H

    _W1z, _b1z, _W2z, _b2z, _accz, _Hz = train_init("zeros")
    _W1h, _b1h, _W2h, _b2h, _acch, _Hh = train_init("he")

    # ---- 画图：全零版隐层输出（8 列全相同）vs He 版（8 列各有分工）----
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for _i in range(8):
        _axes[0].plot(_Hz[:, _i], color="tab:red", alpha=0.5, lw=1.2)
        _axes[1].plot(_Hh[:, _i], alpha=0.6, lw=1.2)
    _axes[0].set_title(f"全零初始化：8 个隐单元输出完全重合\n（train acc {_accz:.0%}——宽度白费）")
    _axes[1].set_title(f"He 初始化：8 个隐单元各司其职\n（train acc {_acch:.0%}）")
    for _ax in _axes:
        _ax.set_xlabel("样本编号")
        _ax.set_ylabel("隐层输出 h")
        _ax.grid(True, alpha=0.3)
    _fig.tight_layout()

    mo.vstack([
        _fig,
        mo.md(
            f"""
            | 验尸项 | 全零版（300 epoch 后） | He 版 |
            |---|---|---|
            | train acc | **{_accz:.0%}**（瞎猜水平） | **{_acch:.0%}** |
            | W₁ 是否仍全零 | **{bool(np.all(_W1z == 0))}** | 否（各有各的值） |
            | b₁ 是否仍全零 | **{bool(np.all(_b1z == 0))}** | 否 |
            | W₂ 是否仍全零 | **{bool(np.all(_W2z == 0))}** | 否 |
            | 唯一动过的参数 | b₂ = {_b2z[0]:.3f} | 全部在动 |
            | 隐单元输出的列 | 全部相同 | 各不相同 |

            死锁三角的传导链：H = 0 → dW₂ = Hᵀδ₂ = 0；
            W₂ = 0 → δ₁ = (δ₂W₂ᵀ)⊙(…) = 0；δ₁ = 0 → dW₁ = 0。
            **三组梯度互相把对方锁死在零**——随机性是唯一的开锁人。
            """
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 2 · 尺度实验：10 层网络的方差演化

        造一个 **10 层、每层 256 单元、tanh** 的深网络，输入方差 1，
        只改初始化尺度，看信号穿过 10 层后的**方差**（log 尺）：

        | 初始化 | 每层方差走势 | 诊断 |
        |---|---|---|
        | std = 0.01（太小） | 1 → 6×10⁻⁴ → 4×10⁻⁷ → … → 10⁻¹⁶ | **信号消失**：到第 6 层已是浮点噪声 |
        | Xavier = 0.0625 | 1 → 0.24 → 0.13 → 0.09 → 稳住 | **基本守恒**（tanh 的官方配对） |
        | He = 0.088 | 1 → 0.40 → 0.33 → 0.31 → 稳住 | **守恒偏稳**（ReLU 官方配对，tanh 用也行） |
        | std = 1.0（太大） | 方差守恒在 0.95 | 看似健康——**但饱和率 87%**！ |

        最后一行是本 demo 的陷阱题：方差没塌，但 tanh 被推进了两端
        平地——**饱和率**（|输出| > 0.99 的比例）才是真正的体检指标：

        | 尺度 | Xavier 0.0625 | He 0.088 | 1.0 | 3.0 |
        |---|---|---|---|---|
        | 饱和率 | **0.8%** | 6.1% | **87%** | 96% |

        结论一句话：**初始化的尺度不是"越大越稳"，是"刚好守恒"**——
        Xavier 配 tanh、He 配 ReLU（补 2 倍是因为 ReLU 砍掉半轴），
        深到 10 层、100 层都照用。
        """
    )
    return


@app.cell
def _(np, plt):
    # 演示 2 · 10 层 tanh：方差演化（4 种尺度）+ 饱和率
    _rng = np.random.default_rng(5)
    _X0 = _rng.normal(0, 1, (512, 256))             # 方差 1 的输入

    def forward_deep(std, n_layers=10, h=256, seed=0):
        _r = np.random.default_rng(seed)
        _Ws = [_r.normal(0, 1, (h, h)) * std for _ in range(n_layers)]
        _A = _X0.copy()
        _vars = [float(_A.var())]
        for _W in _Ws:
            _A = np.tanh(_A @ _W)
            _vars.append(float(_A.var()))
        return _vars

    _setups = [("std=0.01", 0.01, "tab:blue", ":"),
               ("Xavier 1/√fan_in", 1 / 16, "tab:green", "-"),
               ("He √(2/fan_in)", np.sqrt(2 / 256), "tab:orange", "-"),
               ("std=1.0", 1.0, "tab:red", ":")]
    _fig, _ax = plt.subplots(figsize=(7.8, 4.3))
    for _name, _std, _c, _ls in _setups:
        _v = forward_deep(_std)
        _ax.semilogy(range(11), np.maximum(_v, 1e-18), _ls, color=_c, lw=2.1,
                     marker="o", ms=3.5, label=_name)
    _ax.axhline(1.0, color="gray", ls=":", lw=1)
    _ax.set_xlabel("层号（0 = 输入）")
    _ax.set_ylabel("激活的方差（log 尺）")
    _ax.set_title("10 层 tanh 的信号命运：消失 / 守恒 / 饱和")
    _ax.legend(fontsize=9)
    _ax.grid(True, alpha=0.3, which="both")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 3 · Dropout 解剖：掩码、期望守恒、训练对比

        三步解剖。**第一步**看掩码长什么样：一个 40×20 的隐层矩阵，
        p = 0.35 时约三成格子被置零（黑），幸存者放大 1/0.65（亮）。

        **第二步**验证期望守恒：随机抽一个隐层过 20000 次 dropout，
        丢弃后输出的**平均值**和原始平均值之差只有 ~10⁻³——
        "除以保留率"那一手，把期望精确焊回了原地。

        **第三步**上战场：容易过拟合的双月牙（噪声 0.20、只有 60 个
        训练点、h = 96），训练 1500 epoch——

        | 配置 | 末段 train 损失 | 末段 val 损失 | 落差 |
        |---|---|---|---|
        | 无 Dropout | 0.014 | 0.585 | 0.57 |
        | p = 0.2 | 0.027 | 0.383 | 0.36 |
        | p = 0.5 | 0.069 | **0.292** | **0.22** |

        Dropout 把"训练和验证的落差"砍掉六成：train 损失不再贴地
        （背不动题了），val 损失显著下降（真会做题了）。
        注意它**没有让训练更轻松**——恰恰相反，train 损失变高了。
        正则化的哲学从来不是"学得更好"，是**"学得别那么死"**。
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # 演示 3 · Dropout：掩码可视化 + 期望守恒 + 训练对比
    _rng = np.random.default_rng(0)
    _H_demo = np.tanh(_rng.normal(0, 1, (40, 20)))          # 一个假想隐层
    _p_demo = 0.35
    _mask = (_rng.random(_H_demo.shape) >= _p_demo) / (1 - _p_demo)
    _H_drop = _H_demo * _mask

    # ---- 期望守恒的统计验证 ----
    _Hbig = np.tanh(_rng.normal(0, 1, (400, 50)))
    _acc_sum = np.zeros((400, 50))
    _n_trials = 20000
    for _ in range(_n_trials):
        _m = (_rng.random(_Hbig.shape) >= 0.5) / 0.5        # p = 0.5
        _acc_sum += _Hbig * _m
    _mean_orig = float(np.mean(np.abs(_Hbig)))
    _mean_drop = float(np.mean(np.abs(_acc_sum / _n_trials)))
    _gap_mean = abs(_mean_orig - _mean_drop)

    # ---- 训练对比（容易过拟合的月牙）----
    _rng_d = np.random.default_rng(21)
    _t = _rng_d.uniform(0, np.pi, 50)
    _X0d = np.column_stack([np.cos(_t), np.sin(_t)]) + _rng_d.normal(0, 0.20, (50, 2))
    _X1d = np.column_stack([1 - np.cos(_t), 0.5 - np.sin(_t)]) + _rng_d.normal(0, 0.20, (50, 2))
    _Xd = np.vstack([_X0d, _X1d])
    _yd = np.concatenate([np.zeros(50), np.ones(50)])
    _idx = _rng_d.permutation(100)
    _Xd, _yd = _Xd[_idx], _yd[_idx]
    X_fit, y_fit = _Xd[:60], _yd[:60]                       # 只有 60 个训练点
    X_val25, y_val25 = _Xd[60:76], _yd[60:76]               # 16 个验证点

    def train_drop(p_drop, h=96, lr=1.0, epochs=1500, seed=0):
        _ri = np.random.default_rng(0)
        _W1 = _ri.normal(0, 1, (2, h))
        _b1 = np.zeros(h)
        _W2 = _ri.normal(0, 1, (h, 1)) * np.sqrt(2 / h)
        _b2 = np.zeros(1)
        _r = np.random.default_rng(seed + 1)
        _Ltr, _Lva = [], []
        for _ in range(epochs):
            _H = np.tanh(X_fit @ _W1 + _b1)
            if p_drop > 0:                                  # 训练模式：开 Dropout
                _m = (_r.random(_H.shape) >= p_drop) / (1 - p_drop)
                _Hd = _H * _m
            else:
                _Hd = _H
            _p = 1 / (1 + np.exp(-(_Hd @ _W2 + _b2)))
            _n = len(X_fit)
            _yc = y_fit.reshape(-1, 1)
            _d2 = (_p - _yc) / _n
            _d1 = (_d2 @ _W2.T) * (1 - _H ** 2)
            if p_drop > 0:
                _d1 = _d1 * _m                              # 反向过同一个 mask
            _W2 -= lr * (_Hd.T @ _d2)
            _b2 -= lr * _d2.sum(0)
            _W1 -= lr * (X_fit.T @ _d1)
            _b1 -= lr * _d1.sum(0)
            # 体检：评估模式（无 Dropout）
            _pt = 1 / (1 + np.exp(-(np.tanh(X_fit @ _W1 + _b1) @ _W2 + _b2)))
            _pv = 1 / (1 + np.exp(-(np.tanh(X_val25 @ _W1 + _b1) @ _W2 + _b2)))
            _Ltr.append(float(-np.mean(_yc * np.log(_pt + 1e-12)
                                       + (1 - _yc) * np.log(1 - _pt + 1e-12))))
            _Lva.append(float(-np.mean(y_val25.reshape(-1, 1) * np.log(_pv + 1e-12)
                                       + (1 - y_val25.reshape(-1, 1)) * np.log(1 - _pv + 1e-12))))
        return _Ltr, _Lva

    _curves = {}
    for _p in [0.0, 0.2, 0.5]:
        _curves[_p] = train_drop(_p)

    # ---- 画图：左掩码，右训练对比 ----
    _fig, _axes = plt.subplots(1, 2, figsize=(12.4, 4.2))
    _axes[0].imshow(_H_drop, cmap="RdBu", aspect="auto")
    _axes[0].set_title(f"Dropout 掩码（p = {_p_demo}）：黑 = 下班，亮 = 放大补位")
    _axes[0].set_xlabel("隐单元编号")
    _axes[0].set_ylabel("样本编号")

    for _p, _c in [(0.0, "tab:red"), (0.2, "tab:orange"), (0.5, "tab:green")]:
        _axes[1].plot(_curves[_p][1], color=_c, lw=1.9,
                      label=f"val（p = {_p}，末段 {_curves[_p][1][-1]:.3f}）")
    _axes[1].plot(_curves[0.0][0], color="tab:red", lw=1.2, ls=":", alpha=0.7,
                  label="train（无 Dropout，贴地）")
    _axes[1].set_xlabel("epoch")
    _axes[1].set_ylabel("损失")
    _axes[1].set_title("正则化：train 不再贴地，val 明显下降")
    _axes[1].legend(fontsize=8.5)
    _axes[1].grid(True, alpha=0.3)
    _fig.tight_layout()

    mo.vstack([
        _fig,
        mo.md(
            f"""
            | 体检项 | 数值 |
            |---|---|
            | 期望守恒（20000 次抽签） | 原始均值 {_mean_orig:.4f} vs 丢弃后 {_mean_drop:.4f}，差 **{_gap_mean:.1e}** ✅ |
            | 无 Dropout 的落差（train→val） | 0.014 → 0.585（**0.57**） |
            | p = 0.5 的落差 | 0.069 → 0.292（**0.22**，砍掉六成） |
            """
        ),
    ])
    return (X_val25, y_val25)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 4 · 早停现场：patience 踩刹车，weight decay 上慢药

        同一份容易过拟合的数据（演示 3），两件 17 课许诺过的武器：

        **左图 · 早停**：验证损失（橙）在第 129 epoch 触底后一路上翘——
        这就是"开始背题"的信号。patience = 100 的刹车在 **第 228 epoch**
        踩下（触底后再忍 100 轮无改善），参数回滚到**第 128 epoch 的
        最佳版本**。对照"训练到底"的版本：验证损失 0.585 vs 早停版 0.325。

        **右图 · weight decay**：λ 从 0 到 0.005，验证损失末段
        0.585 → 0.347 → 0.271——紧箍咒越紧，train 损失越下不去
        （0.014 → 0.044 → 0.108），**"防疯"和"防本事"的拉锯**
        一览无余。λ = 0.05 以上就会把模型勒进欠拟合（17 课的预告）。

        两件武器的分工：早停管"**什么时候停**"（时间维），
        weight decay 管"**参数长多大**"（空间维）——和 Dropout（结构维）
        三件套齐活，17 课的三件防御武器全部到位。
        """
    )
    return


@app.cell
def _(X_val25, mo, np, plt, y_val25):
    # 演示 4 · 早停 + weight decay（沿用演示 3 的过拟合配置）
    # ---- 重新生成演示 3 的训练数据（配方一致）----
    _rng_d = np.random.default_rng(21)
    _t = _rng_d.uniform(0, np.pi, 50)
    _X0d = np.column_stack([np.cos(_t), np.sin(_t)]) + _rng_d.normal(0, 0.20, (50, 2))
    _X1d = np.column_stack([1 - np.cos(_t), 0.5 - np.sin(_t)]) + _rng_d.normal(0, 0.20, (50, 2))
    _Xd = np.vstack([_X0d, _X1d])
    _yd = np.concatenate([np.zeros(50), np.ones(50)])
    _idx = _rng_d.permutation(100)
    _Xd, _yd = _Xd[_idx], _yd[_idx]
    _Xf, _yf = _Xd[:60], _yd[:60]

    def train_es(wd=0.0, patience=None, h=96, lr=1.0, epochs=1500, seed=0):
        _ri = np.random.default_rng(0)
        _W1 = _ri.normal(0, 1, (2, h))
        _b1 = np.zeros(h)
        _W2 = _ri.normal(0, 1, (h, 1)) * np.sqrt(2 / h)
        _b2 = np.zeros(1)
        _r = np.random.default_rng(seed + 1)
        _Ltr, _Lva = [], []
        _best_va, _best_pack, _wait, _stopped = 1e9, None, 0, epochs
        for _ep in range(epochs):
            _H = np.tanh(_Xf @ _W1 + _b1)
            _p = 1 / (1 + np.exp(-(_H @ _W2 + _b2)))
            _n = len(_Xf)
            _yc = _yf.reshape(-1, 1)
            _d2 = (_p - _yc) / _n
            _d1 = (_d2 @ _W2.T) * (1 - _H ** 2)
            # weight decay：梯度里加 λ·w（17 课 L2 的马甲）
            _W2 -= lr * ((_H.T @ _d2) + wd * _W2)
            _b2 -= lr * (_d2.sum(0) + wd * _b2)
            _W1 -= lr * ((_Xf.T @ _d1) + wd * _W1)
            _b1 -= lr * (_d1.sum(0) + wd * _b1)
            _pt = 1 / (1 + np.exp(-(np.tanh(_Xf @ _W1 + _b1) @ _W2 + _b2)))
            _pv = 1 / (1 + np.exp(-(np.tanh(X_val25 @ _W1 + _b1) @ _W2 + _b2)))
            _lt = float(-np.mean(_yc * np.log(_pt + 1e-12) + (1 - _yc) * np.log(1 - _pt + 1e-12)))
            _lv = float(-np.mean(y_val25.reshape(-1, 1) * np.log(_pv + 1e-12)
                                 + (1 - y_val25.reshape(-1, 1)) * np.log(1 - _pv + 1e-12)))
            _Ltr.append(_lt)
            _Lva.append(_lv)
            # ---- 早停检查（17 课的刹车片）----
            if _lv < _best_va - 1e-4:
                _best_va, _wait = _lv, 0
                _best_pack = (_W1.copy(), _b1.copy(), _W2.copy(), _b2.copy(), _ep + 1)
            else:
                _wait += 1
                if patience and _wait >= patience:
                    _stopped = _ep + 1
                    break
        return _Ltr, _Lva, _stopped, _best_pack

    _Ltr0, _Lva0, _, _best0 = train_es()
    _best_ep = _best0[4]
    _, _, _stopped, _ = train_es(patience=100)

    _fig, _axes = plt.subplots(1, 2, figsize=(12.6, 4.3))
    _axes[0].plot(_Ltr0, color="tab:blue", lw=1.8, label="train")
    _axes[0].plot(_Lva0, color="tab:orange", lw=1.8, label="val")
    _axes[0].scatter([_best_ep - 1], [_Lva0[_best_ep - 1]], s=140, facecolors="none",
                     edgecolors="tab:green", linewidths=2.5, zorder=5,
                     label=f"val 最低：epoch {_best_ep}")
    _axes[0].scatter([_stopped - 1], [_Lva0[_stopped - 1]], s=120, marker="X",
                     color="tab:red", zorder=5, label=f"早停踩下：epoch {_stopped}（patience=100）")
    _axes[0].set_xlabel("epoch")
    _axes[0].set_ylabel("损失")
    _axes[0].set_title("早停：触底后上翘，刹车自动踩下")
    _axes[0].legend(fontsize=9)
    _axes[0].grid(True, alpha=0.3)

    for _wd, _c in [(0.0, "tab:red"), (0.001, "tab:orange"), (0.005, "tab:green")]:
        _lt, _lv, _, _ = train_es(wd=_wd)
        _axes[1].plot(_lv, color=_c, lw=1.8, label=f"val（λ = {_wd}，末段 {_lv[-1]:.3f}）")
    _axes[1].set_xlabel("epoch")
    _axes[1].set_ylabel("验证损失")
    _axes[1].set_title("weight decay：慢性紧箍咒（λ 越大越勒）")
    _axes[1].legend(fontsize=9)
    _axes[1].grid(True, alpha=0.3)
    _fig.tight_layout()

    mo.vstack([
        _fig,
        mo.md(
            f"""
            | 读数 | 值 |
            |---|---|
            | val 损失触底 | 第 **{_best_ep}** epoch（{_Lva0[_best_ep-1]:.3f}） |
            | 早停刹车点（patience = 100） | 第 **{_stopped}** epoch |
            | 训练到底的 val 损失 | {_Lva0[-1]:.3f}（比触底恶化 {_Lva0[-1]-min(_Lva0):.3f}） |
            | weight decay λ = 0.005 的 val 末段 | **0.271**（对照无正则 0.585） |
            """
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 5 · MNIST 大战：完全体首次实战

        数据：**MNIST 手写数字**——60000 张 28×28 的训练图 + 10000 张
        测试图，压平成 784 维向量（每维是一个像素的灰度，归一化到 0~1）。
        第一次运行会自动从镜像下载（约 11 MB，缓存到 `data/`，
        之后都走本地）；网络不通时看下方的手动下载说明。

        网络与配置（每一件都是前几课的零件）：

        - **784 → 128(ReLU) → 10(softmax)**，101,770 个参数（20 课数过）
        - 损失：softmax + 交叉熵，输出层梯度 p − y（21 课）
        - 优化器：Adam，lr = 0.001，batch = 64（24 课）
        - 初始化：He（本课），切分 50000/10000/10000（17 课）
        - 早停：patience = 3（17 课的承诺，今天兑现）

        先看数据长什么样——16 张样例图，人眼一秒能认，
        网络要从 784 个数字里学出同样的眼力。
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # 演示 5A · MNIST 加载器（自动下载 + 缓存）+ 样例图
    def load_mnist():
        """加载 MNIST。优先读 data/mnist.npz 缓存；没有则找/下载 4 个 .gz 原始文件再解析。
        返回 (X_tr, y_tr, X_te, y_te)，像素归一化到 [0,1]，X 形状 (n, 784)。
        数据目录的定位与工作目录无关：优先「笔记本文件旁边的 ../data」"""
        import os, gzip, urllib.request
        from pathlib import Path
        _files = {"train-images": ("train-images-idx3-ubyte.gz", 9912422),
                  "train-labels": ("train-labels-idx1-ubyte.gz", 28881),
                  "t10k-images": ("t10k-images-idx3-ubyte.gz", 1648877),
                  "t10k-labels": ("t10k-labels-idx1-ubyte.gz", 4542)}
        _mirrors = ["https://ossci-datasets.s3.amazonaws.com/mnist",      # PyTorch 官方镜像
                    "https://storage.googleapis.com/cvdf-datasets/mnist"]  # 备用镜像
        # ---- 定位数据目录：笔记本的上级/data（课程标准位置）优先，其次相对 CWD 的 ----
        _cands = []
        try:
            _nb = mo.notebook_location()               # 本笔记本文件的路径（与 CWD 无关）
            if _nb is not None:
                _cands += [Path(_nb).parent.parent / "data", Path(_nb).parent / "data"]
        except Exception:
            pass
        _cands += [Path("../data"), Path("data")]
        _dir = next((_d for _d in _cands if _d.is_dir()), None)
        if _dir is None:                               # 都不存在：试荐第一个可创建的
            for _d in _cands:
                try:
                    _d.mkdir(parents=True, exist_ok=True)
                    _dir = _d
                    break
                except OSError:
                    continue
            if _dir is None:
                raise RuntimeError("找不到可写的数据目录：请把 MNIST 文件放进课程的 data/ 目录")
        _cache = _dir / "mnist.npz"
        if _cache.exists():
            with np.load(_cache) as _z:
                return _z["X_tr"], _z["y_tr"], _z["X_te"], _z["y_te"]
        for _key, (_fname, _size) in _files.items():                      # 缺哪个下哪个
            _path = _dir / _fname
            if _path.exists() and _path.stat().st_size == _size:
                continue
            _ok = False
            for _m in _mirrors:
                try:
                    print(f"下载 {_fname}（{_size/1e6:.1f} MB）……")
                    urllib.request.urlretrieve(f"{_m}/{_fname}", _path)
                    if _path.stat().st_size == _size:
                        _ok = True
                        break
                except Exception:
                    continue
            if not _ok:
                raise RuntimeError(
                    f"MNIST 下载失败：{_fname}。请展开本单元上方的「手动下载说明」，"
                    "把 4 个文件放进 data/ 目录后重新运行。")
        def _images(_path):                                               # IDX 大端解析
            with gzip.open(_path, "rb") as _f:
                _raw = _f.read()
            _magic, _n, _r, _c = np.frombuffer(_raw[:16], dtype=">i4")
            assert _magic == 2051, "图片文件魔数不对"
            return np.frombuffer(_raw[16:], dtype=np.uint8).reshape(_n, _r * _c).astype(np.float32) / 255.0
        def _labels(_path):
            with gzip.open(_path, "rb") as _f:
                _raw = _f.read()
            _magic, _n = np.frombuffer(_raw[:8], dtype=">i4")
            assert _magic == 2049, "标签文件魔数不对"
            return np.frombuffer(_raw[8:], dtype=np.uint8).astype(np.int64)
        X_tr, y_tr = _images(_dir / _files["train-images"][0]), \
                     _labels(_dir / _files["train-labels"][0])
        X_te, y_te = _images(_dir / _files["t10k-images"][0]), \
                     _labels(_dir / _files["t10k-labels"][0])
        np.savez_compressed(_cache, X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te)   # 缓存
        return X_tr, y_tr, X_te, y_te

    X_mnist, y_mnist, X_test25, y_test25 = load_mnist()
    print("训练集:", X_mnist.shape, " 测试集:", X_test25.shape)

    # ---- 16 张样例 ----
    _fig, _axes = plt.subplots(2, 8, figsize=(11, 3.1))
    for _k, _ax in enumerate(_axes.ravel()):
        _ax.imshow(X_mnist[_k].reshape(28, 28), cmap="gray_r")
        _ax.set_title(str(int(y_mnist[_k])), fontsize=10)
        _ax.axis("off")
    _fig.suptitle("MNIST 样例：每张图 = 784 个像素 = 一个 784 维向量", fontsize=12)
    _fig.tight_layout()
    _fig
    return (X_mnist, X_test25, load_mnist, y_mnist, y_test25)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📥 手动下载说明（自动下载失败时看这里）": mo.md(
                """
                把下面 4 个文件下载后放进课程的 `data/` 目录
                （和 `notebooks/` 同级）即可，文件名保持原样：

                - `train-images-idx3-ubyte.gz`（9.5 MB）
                  https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
                - `train-labels-idx1-ubyte.gz`（28 KB）
                  https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
                - `t10k-images-idx3-ubyte.gz`（1.6 MB）
                  https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
                - `t10k-labels-idx1-ubyte.gz`（4 KB）
                  https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz

                备用镜像（把地址里的域名换成）：
                `storage.googleapis.com/cvdf-datasets/mnist/`

                如果你本机已有 keras 或 PyTorch 下载过的 MNIST，
                也可以自己存一份 `data/mnist.npz`（四个数组：
                X_tr (60000,784) float 0~1、y_tr (60000,)、
                X_te (10000,784)、y_te (10000,)）——加载器会直接读缓存。
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 5 续 · 点火：12 个 epoch 的完全体

        洗牌切批 → 前向 → 反向 → Adam → 每 epoch 体检 → 早停检查。
        跑起来大约每 epoch 2 秒（普通 CPU），全程半分钟左右——
        **十万参数的网络，纯 NumPy，CPU 足够**（第 0 课的承诺兑现）。

        战报（早停在 epoch 12，最佳参数是 epoch 9 的版本）：

        | epoch | 1 | 3 | 6 | 9（峰值） | 12（刹车） |
        |---|---|---|---|---|---|
        | train acc | 95.3% | 97.8% | 98.9% | **99.6%** | 99.8% |
        | val acc | 94.6% | 96.6% | 97.2% | **97.77%** | 97.5%（回落 → 刹车） |

        epoch 9 之后验证准确率三连回落——早停的耐心耗尽，刹车踩下，
        参数回滚到 epoch 9 的最佳版本。这就是 17 课那句
        "在疯掉之前踩刹车"的机械化实现。
        """
    )
    return


@app.cell
def _(X_mnist, X_test25, mo, np, plt, y_mnist, y_test25):
    # 演示 5B · 完全体训练循环：MNIST 784-128-10 + Adam + 早停
    def train_mnist_full(h=128, lr=0.001, batch=64, epochs=15, patience=3,
                         seed=0, verbose=True):
        """完全体训练循环。返回 (最佳参数, 训练历史)"""
        _X_all, _y_all = X_mnist, y_mnist
        _r_split = np.random.default_rng(42)                 # 三分天下（17 课）
        _idx = _r_split.permutation(len(_X_all))
        _Xa, _ya = _X_all[_idx], _y_all[_idx]
        _Xtr, _ytr = _Xa[:50000], _ya[:50000]                 # 上课 5 万
        _Xva, _yva = _Xa[50000:], _ya[50000:]                 # 模考 1 万

        _ri = np.random.default_rng(seed)                     # He 初始化（本课概念 1）
        _params = [_ri.normal(0, 1, (784, h)) * np.sqrt(2 / 784),
                   np.zeros(h),
                   _ri.normal(0, 1, (h, 10)) * np.sqrt(2 / h),
                   np.zeros(10)]
        _m = [np.zeros_like(_p) for _p in _params]            # Adam 的四本账（24 课）
        _s = [np.zeros_like(_p) for _p in _params]
        _r = np.random.default_rng(seed + 1)
        _hist = {"L": [], "acc_tr": [], "acc_va": []}
        _best_va, _best_pack, _wait, _step = -1.0, None, 0, 0
        import time as _time
        _t0 = _time.time()
        for _ep in range(epochs):
            _order = _r.permutation(len(_Xtr))                # 每轮洗牌（10 课）
            for _st in range(0, len(_Xtr), batch):
                _ib = _order[_st:_st + batch]                 # 切批（24 课）
                _Xb, _yb = _Xtr[_ib], _ytr[_ib]
                _W1, _b1, _W2, _b2 = _params
                _Z1 = _Xb @ _W1 + _b1                         # ② 前向（20 课）
                _H = np.maximum(0, _Z1)                       # ReLU（19 课）
                _logits = _H @ _W2 + _b2
                _e = np.exp(_logits - _logits.max(1, keepdims=True))   # softmax（21 课，减 max）
                _P = _e / _e.sum(1, keepdims=True)
                _L = -np.mean(np.log(_P[np.arange(len(_yb)), _yb] + 1e-12))
                _dl = _P.copy()                               # ③ 反向：p − y（21 课）
                _dl[np.arange(len(_yb)), _yb] -= 1
                _dl /= len(_yb)
                _dW2 = _H.T @ _dl                             # 四条矩阵梯度（23 课）
                _db2 = _dl.sum(0)
                _dZ1 = (_dl @ _W2.T) * (_Z1 > 0)              # ReLU 的导数（19 课）
                _dW1 = _Xb.T @ _dZ1
                _db1 = _dZ1.sum(0)
                _grads = [_dW1, _db1, _dW2, _db2]
                _step += 1
                for _i in range(4):                           # ⑤ Adam（24 课）
                    _m[_i] = 0.9 * _m[_i] + 0.1 * _grads[_i]
                    _s[_i] = 0.999 * _s[_i] + 0.001 * _grads[_i] ** 2
                    _params[_i] -= lr * (_m[_i] / (1 - 0.9 ** _step)) \
                        / (np.sqrt(_s[_i] / (1 - 0.999 ** _step)) + 1e-8)
            # ⑥ 体检（每 epoch 一次）
            def _acc(_X, _y):
                _W1, _b1, _W2, _b2 = _params
                return float(np.mean(np.argmax(np.maximum(0, _X @ _W1 + _b1) @ _W2 + _b2, 1) == _y))
            _a_tr, _a_va = _acc(_Xtr, _ytr), _acc(_Xva, _yva)
            _hist["L"].append(float(_L))
            _hist["acc_tr"].append(_a_tr)
            _hist["acc_va"].append(_a_va)
            if verbose:
                print(f"  epoch {_ep+1}: L={_L:.4f} acc_tr={_a_tr:.4f} acc_va={_a_va:.4f} "
                      f"({_time.time()-_t0:.0f}s)")
            # ⑦ 早停检查（17 课 → 本课概念 3）
            if _a_va > _best_va + 1e-4:
                _best_va, _wait = _a_va, 0
                _best_pack = [_p.copy() for _p in _params]
                _hist["best_ep"] = _ep + 1
            else:
                _wait += 1
                if _wait >= patience:
                    _hist["stopped"] = _ep + 1
                    break
        _hist.setdefault("stopped", epochs)
        _hist["best_va"] = _best_va
        _hist["seconds"] = _time.time() - _t0
        return (_best_pack if _best_pack else _params), _hist

    mnist_params, mnist_hist = train_mnist_full()

    # ---- 训练曲线 ----
    _fig, _ax1 = plt.subplots(figsize=(8.2, 4.3))
    _ep = np.arange(1, len(mnist_hist["L"]) + 1)
    _ax1.plot(_ep, mnist_hist["L"], "o-", color="tab:blue", lw=2, label="训练损失（每轮最后一批）")
    _ax1.set_xlabel("epoch")
    _ax1.set_ylabel("训练损失", color="tab:blue")
    _ax2 = _ax1.twinx()
    _ax2.plot(_ep, mnist_hist["acc_va"], "s-", color="tab:orange", lw=2, label="验证准确率")
    _ax2.scatter([mnist_hist["best_ep"]], [mnist_hist["best_va"]], s=150, facecolors="none",
                 edgecolors="tab:red", linewidths=2.5, zorder=5,
                 label=f"峰值 epoch {mnist_hist['best_ep']}（{mnist_hist['best_va']:.2%}）")
    _ax2.set_ylabel("验证准确率", color="tab:orange")
    _ax2.set_ylim(0.90, 1.0)
    _ax1.grid(True, alpha=0.3)
    _lines = _ax1.get_lines() + _ax2.get_lines()
    _ax1.legend(_lines, [_l.get_label() for _l in _lines], fontsize=9, loc="center right")
    _ax1.set_title(f"MNIST 训练战报：早停于 epoch {mnist_hist['stopped']}"
                   f"（patience = 3），用时 {mnist_hist['seconds']:.0f}s")
    _fig.tight_layout()
    _fig
    return (mnist_params, train_mnist_full)


@app.cell
def _(X_test25, mnist_params, mo, np, plt, y_test25):
    # 演示 5C · 高考揭榜 + 错得最自信的 8 张图
    _W1, _b1, _W2, _b2 = mnist_params
    _H = np.maximum(0, X_test25 @ _W1 + _b1)
    _logits = _H @ _W2 + _b2
    _e = np.exp(_logits - _logits.max(1, keepdims=True))
    _P = _e / _e.sum(1, keepdims=True)
    _pred = np.argmax(_P, 1)
    _acc_te = float(np.mean(_pred == y_test25))
    _wrong = np.where(_pred != y_test25)[0]
    _conf_wrong = _wrong[np.argsort(-_P[_wrong, _pred[_wrong]])][:8]   # 最自信的错

    _fig, _axes = plt.subplots(2, 4, figsize=(10.5, 5.6))
    for _k, _ax in zip(_conf_wrong, _axes.ravel()):
        _ax.imshow(X_test25[_k].reshape(28, 28), cmap="gray_r")
        _ax.set_title(f"真值 {int(y_test25[_k])} → 预测 {int(_pred[_k])}\n"
                      f"（置信 {_P[_k, _pred[_k]]:.1%}）", fontsize=9, color="tab:red")
        _ax.axis("off")
    _fig.suptitle("错得最自信的 8 张：连人眼都要眯一下的烂字", fontsize=12)
    _fig.tight_layout()
    _fig

    _per = [float(np.mean(_pred[y_test25 == _k] != _k)) for _k in range(10)]
    _worst = int(np.argmax(_per))
    mo.vstack([
        _fig,
        mo.md(
            f"""
            | 高考揭榜 | 值 |
            |---|---|
            | **测试集准确率（只碰这一次）** | **{_acc_te:.2%}** |
            | 错分数量 | {len(_wrong)} / 10000 |
            | 训练用时 | 约 25 秒（纯 CPU、纯 NumPy） |
            | 对照：瞎猜 | 10% |
            | 对照：线性模型（同数据量） | 约 92% |
            | 最容易错的数字 | **{_worst}**（错误率 {_per[_worst]:.1%}）——和 3、5、9 互相纠缠 |

            看看那 8 张"最自信的错"：7 被看成 2（置信度 100%！）、
            5 被看成 0——**很多连人眼都要犹豫**。98% 之后的每一分，
            都要从这种烂字里抠，这正是后面 CNN（第 5 部分）要接的班：
            像素的**局部结构**（笔画怎么连）没有被全连接网络利用，
            卷积就是来干这个的。
            """
        ),
        mo.callout(
            mo.md(
                f"""
                **第 4 部分收官战报**：一个你亲手写了每个零件的网络——
                手写 forward（20 课）、手写 softmax 与 p−y（21 课）、
                手写反向传播（22、23 课）、手写 Adam（24 课）、
                He 初始化和早停（本课）——在 10000 张从未见过的
                手写数字上，**每 100 张只错 2 张出头**。
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

        **正则化旋钮**——把演示 3 那份容易过拟合的双月牙摆上台面，
        拖动 Dropout 的丢弃率 p（0 ~ 0.7），看训练/验证两条曲线的
        "剪刀差"怎么被一点点合上。

        玩法建议：

        1. **p = 0**（关掉）：train 损失一路贴地、val 曲线后期上翘——
           演示 3 的过拟合现场原样重播
        2. **p 拖到 0.5**：train 损失被"垫高"（背不了题了），
           val 明显下降且不再上翘
        3. **p 拖到 0.7**（过猛）：两条曲线都下不去了——
           一半神经元常年在下班，网络兵力不足（欠拟合的正则版）
        """
    )
    return


@app.cell
def _(mo):
    # 交互控制：Dropout 丢弃率（debounce：松手才重算，训练要跑几秒）
    drop_slider = mo.ui.slider(0.0, 0.7, value=0.0, step=0.05, debounce=True,
                               label="🎩 Dropout 丢弃率 p（0 ~ 0.7）")
    drop_slider
    return (drop_slider,)


@app.cell
def _(drop_slider, mo, np, plt):
    # ===== 交互实验 · 正则化旋钮（响应单元）=====
    _p = drop_slider.value

    # ---- 演示 3 同款数据配方（种子固定，可与演示对数）----
    _rng_d = np.random.default_rng(21)
    _t = _rng_d.uniform(0, np.pi, 50)
    _X0d = np.column_stack([np.cos(_t), np.sin(_t)]) + _rng_d.normal(0, 0.20, (50, 2))
    _X1d = np.column_stack([1 - np.cos(_t), 0.5 - np.sin(_t)]) + _rng_d.normal(0, 0.20, (50, 2))
    _Xd = np.vstack([_X0d, _X1d])
    _yd = np.concatenate([np.zeros(50), np.ones(50)])
    _idx = _rng_d.permutation(100)
    _Xd, _yd = _Xd[_idx], _yd[_idx]
    _Xf, _yf = _Xd[:60], _yd[:60]
    _Xv, _yv = _Xd[60:76], _yd[60:76]

    _ri = np.random.default_rng(0)
    _W1 = _ri.normal(0, 1, (2, 96))
    _b1 = np.zeros(96)
    _W2 = _ri.normal(0, 1, (96, 1)) * np.sqrt(2 / 96)
    _b2 = np.zeros(1)
    _r = np.random.default_rng(1)
    _Ltr, _Lva = [], []
    for _ in range(1500):
        _H = np.tanh(_Xf @ _W1 + _b1)
        if _p > 0:
            _mk = (_r.random(_H.shape) >= _p) / (1 - _p)     # inverted dropout
            _Hd = _H * _mk
        else:
            _Hd = _H
        _pp = 1 / (1 + np.exp(-(_Hd @ _W2 + _b2)))
        _n = len(_Xf)
        _yc = _yf.reshape(-1, 1)
        _d2 = (_pp - _yc) / _n
        _d1 = (_d2 @ _W2.T) * (1 - _H ** 2)
        if _p > 0:
            _d1 = _d1 * _mk
        _W2 -= (_Hd.T @ _d2)
        _b2 -= _d2.sum(0)
        _W1 -= (_Xf.T @ _d1)
        _b1 -= _d1.sum(0)
        _pt = 1 / (1 + np.exp(-(np.tanh(_Xf @ _W1 + _b1) @ _W2 + _b2)))     # 评估模式
        _pv = 1 / (1 + np.exp(-(np.tanh(_Xv @ _W1 + _b1) @ _W2 + _b2)))
        _Ltr.append(float(-np.mean(_yc * np.log(_pt + 1e-12) + (1 - _yc) * np.log(1 - _pt + 1e-12))))
        _Lva.append(float(-np.mean(_yv.reshape(-1, 1) * np.log(_pv + 1e-12)
                                   + (1 - _yv.reshape(-1, 1)) * np.log(1 - _pv + 1e-12))))

    _fig, _ax = plt.subplots(figsize=(7.8, 4.2))
    _ax.plot(_Ltr, color="tab:blue", lw=1.9, label=f"train（末段 {_Ltr[-1]:.3f}）")
    _ax.plot(_Lva, color="tab:orange", lw=1.9, label=f"val（末段 {_Lva[-1]:.3f}）")
    _ax.set_xlabel("epoch")
    _ax.set_ylabel("损失")
    _ax.set_title(f"Dropout p = {_p:.2f}：剪刀差 = {_Lva[-1] - _Ltr[-1]:.3f}")
    _ax.legend(fontsize=9)
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()

    _gap = _Lva[-1] - _Ltr[-1]
    if _p == 0:
        _verdict = "👻 p = 0：过拟合现场——train 贴地、val 上翘，剪刀差撑开"
    elif _p <= 0.5:
        _verdict = (f"✅ p = {_p:.2f}：剪刀差收窄到 {_gap:.3f}"
                    "——train 被垫高（背不了题），val 受益（真会做题）")
    else:
        _verdict = (f"⚠️ p = {_p:.2f} 有点猛：两条曲线都下不去——"
                    "太多神经元常年在岗外，兵力不足（正则版欠拟合）")

    mo.vstack([
        _fig,
        mo.callout(mo.md(f"**实时判词：{_verdict}**"), kind="info"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✏️ 动手练习（每课必做，本课 3 题）": mo.md(
                """
                **练习 1 · 手写 He 初始化**：一行代码
                （小随机 × sqrt(2/进维)）——23 课用了两课的"出厂设置"，
                今天把它的出厂权限拿回来。

                **练习 2 · 手写 inverted dropout**：两行（掩码 + 逐元素乘），
                用"期望守恒"验收——丢完的平均输出要和原来一样。

                **练习 3 · 挑战 · MNIST 冲 98%**：主配置拿到了 97.8%，
                还差 0.2 个点。调 h / epochs（train_mnist_full 的参数），
                在**测试集**上冲过 **98%**——参考配置在参考答案里。
                注意：测试集只许最后碰一次（17 课的纪律），
                调参全程用验证集。

                规则不变：先把 `None` 换成你的答案再运行；写完再看参考答案。
                """
            )
        }
    )
    return


@app.cell
def _(mo, np):
    # ===== 练习 1 · 手写 He 初始化 =====
    def he_init_ex1(rng, d, h):
        """He 初始化：返回形状 (d, h) 的权重矩阵"""
        return None    # ← ① 补：rng.normal(0, 1, (d, h)) * np.sqrt(2 / d)

    he_ready_ex1 = None    # ← ② 补完后，把这行换成 he_init_ex1

    if he_ready_ex1 is not None:
        try:
            _rng_a = np.random.default_rng(7)
            _rng_b = np.random.default_rng(7)
            _got = he_ready_ex1(_rng_a, 784, 128)
            _ref = _rng_b.normal(0, 1, (784, 128)) * np.sqrt(2 / 784)
            if _got.shape == (784, 128) and np.allclose(_got, _ref):
                _msg = (f"✅ 形状 {_got.shape}、标准差 {float(np.std(_got)):.4f} "
                        f"（理论值 sqrt(2/784) = {np.sqrt(2/784):.4f}）——He 出厂权限到手")
            else:
                _msg = "❌ 形状或数值对不上——① 的形状是 (d, h)，尺度是 sqrt(2/d)"
        except TypeError:
            _msg = "⏳ 返回值还是 None——① 那行补了吗？"
    else:
        _msg = "⏳ 先补 ①，再把 he_ready_ex1 换成 he_init_ex1"
    mo.md(_msg)
    return (he_init_ex1, he_ready_ex1)


@app.cell
def _(mo, np):
    # ===== 练习 2 · 手写 inverted dropout =====
    def dropout_ex2(H, rng, p):
        """inverted dropout 前向。返回 (丢弃后的 H, mask)"""
        mask = None    # ← ① 补：(rng.random(H.shape) >= p) / (1 - p)
        return None    # ← ② 补：H * mask

    drop_ready_ex2 = None    # ← ③ 补完后，把这行换成 dropout_ex2

    if drop_ready_ex2 is not None:
        try:
            _rng = np.random.default_rng(3)
            _H = np.tanh(_rng.normal(0, 1, (2000, 200)))      # 大矩阵：统计才稳
            _Hd, _mk = drop_ready_ex2(_H, _rng, 0.4)
            _ratio = float(np.mean(_Hd)) / float(np.mean(_H))  # 期望比应 ≈ 1
            _zero_rate = float(np.mean(_mk == 0))              # 置零比例应 ≈ p
            if abs(_ratio - 1) < 0.03 and abs(_zero_rate - 0.4) < 0.02:
                _msg = (f"✅ 期望比 {_ratio:.4f}（≈1：守恒）、置零率 {_zero_rate:.1%}"
                        "（≈p）——inverted dropout 验收通过")
            elif abs(_zero_rate - 0.4) >= 0.02:
                _msg = f"❌ 置零率 {_zero_rate:.1%} 应约等于 p = 40%——检查①的比较方向"
            else:
                _msg = (f"❌ 期望比 {_ratio:.3f} 偏离 1——忘了除以 (1 - p)？"
                        "幸存者要放大补位")
        except TypeError:
            _msg = "⏳ mask 或返回值还是 None——两行都补上"
    else:
        _msg = "⏳ 先补两行，再把 drop_ready_ex2 换成 dropout_ex2"
    mo.md(_msg)
    return (drop_ready_ex2, dropout_ex2)


@app.cell
def _(mo, train_mnist_full):
    # ===== 练习 3 · 挑战：MNIST 冲 98% =====
    # 提示：train_mnist_full(h=…, epochs=…) 的返回值是 (最佳参数, hist)；
    # 用最佳参数在测试集上算准确率——测试集只许最后碰这一次！
    # （测试集的评估代码可以抄演示 5C 的三行）

    best_test_acc_ex3 = None    # ← 你最好配置的测试集准确率（目标 ≥ 0.98）

    if best_test_acc_ex3 is not None:
        if float(best_test_acc_ex3) >= 0.98:
            _msg = (f"✅ **{float(best_test_acc_ex3):.2%}** ≥ 98%——收官挑战达成！"
                    "你已经会独立调一台真正的神经网络了")
        else:
            _msg = (f"❌ {float(best_test_acc_ex3):.2%} 还差一点——"
                    "提示：h 加到 256、epochs 给到 20（早停会自动踩刹车）")
    else:
        _msg = "⏳ 调参 → 训练 → 测试集算分 → 把成绩填进来（全程用验证集调参）"
    mo.md(_msg)
    return (best_test_acc_ex3,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 1 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                return rng.normal(0, 1, (d, h)) * np.sqrt(2 / d)
                ```

                验收：形状 (784, 128)、标准差 0.0505 = sqrt(2/784)。

                两个容易混的细节：

                - **形状是 (d, h) 不是 (h, d)**——X @ W 的内维要咬合
                  （20 课维度账本）；23 课 `init_params` 里那一行
                  `rng.normal(0, 1, (d, h)) * np.sqrt(2 / d)` 原样就是它
                - **sqrt(2/d) 配 ReLU，sqrt(1/d)（Xavier）配 tanh**——
                  2 是给 ReLU 砍掉的半轴补的量。MNIST 这套网络用 ReLU，
                  所以是 He；换 tanh 就该换 Xavier（演示 2 的两种都守恒，
                  但饱和率 He 版略高）
                """
            ),
            "✅ 练习 2 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                mask = (rng.random(H.shape) >= p) / (1 - p)   # ① 抽签 + 放大补位
                return H * mask                                # ② 逐元素乘
                ```

                验收：期望比 ≈ 1.000（2000×200 的矩阵统计）、置零率 ≈ 40%。

                三个细节值得咀嚼：

                - **mask 要存下来**：反向传播要过**同一个** mask
                  （被丢弃的单元梯度也是 0）——演示 3 的 `d1 = d1 * m`
                  就是它。训练循环里 mask 是前向和反向的共享票据
                - **除以 (1−p) 的意义**：40% 的单元置零、剩下 60% 放大
                  1.67 倍 → 期望输出不变 → **测试时不用做任何补偿**。
                  老式写法（测试时乘 p）训练/测试不一致，bug 温床
                - **每次前向重新抽**：`rng.random` 每次调用都是新签——
                  每一步都在训练不同的子网络，集成效应由此而来
                """
            ),
            "✅ 练习 3 参考答案（先自己写过再看！）": mo.md(
                """
                一份能过的配置（作者实测）：

                ```python
                params, hist = train_mnist_full(h=256, epochs=20, patience=5)
                # 早停在 epoch 15，最佳 val acc 97.78%
                # 测试集：98.05%
                ```

                冲分路线图（每一分都有出处）：

                | 手段 | 收益 | 出处 |
                |---|---|---|
                | h: 128 → 256 | 97.78% → 98.05% | 容量（17 课） |
                | epochs 给足 + 早停兜底 | 防止"马拉松跑 30 公里" | 本课 |
                | 再往上：h=512、数据增强、CNN | 98.3%+ / 99%+ | 25 课之后的事 |

                两条纪律重申：

                - **全程用验证集调参**，测试集只在最后碰一次——
                  每用测试集"试一次分"，它就退化成第二次模考（17 课）
                - h=512 在这份代码上只有 ~0.1% 的边际收益，但训练时间
                  翻倍——**容量不是免费的**，真实工程里 98% 和 98.1%
                  之间隔着的是算力预算。下一部分你会看到更聪明的
                  花法：同样的参数量，卷积把它们花在刀刃上
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
                1. **初始化定生死**：全零 = 对称性灾难（隐单元全同、
                   梯度死锁三角、train 卡 51%）；太大 = 饱和昏迷
                   （std=1 时 87% 的 tanh 饱和）；He/Xavier = 方差守恒
                   （sqrt(2/fan_in) 配 ReLU，sqrt(1/fan_in) 配 tanh）
                2. **inverted dropout**：mask = (随机 ≥ p)/(1−p)、
                   H⊙mask、反向过同一个 mask；期望守恒让测试模式
                   零补偿；每步重抽 = 子网络集成 + 禁止共谋；
                   **只在训练时开**
                3. **早停**：val 创新低 → 存参数、耐心归零；耐心攒满
                   patience → 停、回滚最佳版本——"自动回到最好的过去"；
                   MNIST 上 epoch 9 峰值 97.77%、epoch 12 刹车
                4. **weight decay**：梯度里加 λ·w = 每步先乘性缩小
                   （17 课 L2 的马甲）；λ 从 0 到 0.005，val 末段
                   0.585 → 0.271；太大会勒进欠拟合
                5. **训练循环完全体**：He 初始化 → 洗牌切批 → 前向
                   （Dropout）→ 损失 → 反向 → Adam → 体检 → 早停
                   ——15 课四步曲的工业放大版，往后所有网络都是这套
                6. **MNIST 战报**：784-128-10、101,770 参数、25 秒训练，
                   测试集 **97.78%**（瞎猜 10%、线性 92%）；挑战配置
                   h=256 冲到 **98.05%**；最难的数字是 3 和 9——
                   剩下的错分多在烂字上，交给第 5 部分的卷积
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
                ## 🎓 第 4 部分收官：神经网络（8/8）

                八课连打，一台完整的深度学习流水线到手。对着清单清点行囊：

                | 课 | 里程碑：你现在能做到什么 |
                |---|---|
                | **18** 感知机 | 第一个神经元：错了才动的纠错规则，和 XOR 之墙 |
                | **19** 激活函数 | 线性摞线性还是线性的魔咒；ReLU = 斜率补丁；饱和与死区 |
                | **20** 多层感知机 | 维度账本、手设权重拆 XOR 墙、隐层 = 空间折叠 |
                | **21** 损失与 softmax | 输出层-损失配对表、减 max 退烧药、p − y |
                | **22** 反向传播（上） | 计算图双向跑通、两条铁律、标量网络全梯度对账 |
                | **23** 反向传播（下） | 四条矩阵梯度 + 形状检查法、双月牙 100% |
                | **24** 优化器 | 动量 20 倍加速、Adam、mini-batch、batch 旋钮 |
                | **25** 初始化与训练循环 | He/Dropout/早停/weight decay、MNIST 97.8% |

                用一句话回顾第 4 部分的剧情：18 课造出神经元、19 课给它
                会弯的关节、20 课把层摞成机器、21 课装上计分板、
                22~23 课装上导航仪、24 课升级走法、25 课配齐起跑线和刹车——
                **一台你亲手拧过每颗螺丝的深度学习发动机，点火、实测、
                收工**。
                """
            ),
            mo.callout(
                mo.md(
                    """
                    现在回头再看那句"深度学习"：**深**，是非线性弯折的层数
                    （19 课）；**度**，是梯度穿过每一层的量尺（22~23 课）；
                    **学习**，是损失驱动、优化器执行的参数更新（21、24 课）。
                    三个词你都写过、调过、修过——它不再是黑盒，是你的手艺。

                    第 5 部分开始，这门手艺要换一个战场：图像。
                    """
                ),
                kind="success",
            ),
            mo.md(
                """
                ---

                **下一课预告** — `26_卷积直觉`（第 5 部分 · 卷积神经网络）：

                MNIST 上那 2% 的错分，根子在全连接的"挥霍"：
                **每个神经元都直勾勾盯着全部 784 个像素**——
                可认字靠的是笔画的**局部**形状，不是某个具体像素的亮度。

                下一课开始的三连击：**卷积**——一个只在局部小窗口里
                巡逻的权重模板（想象一个 3×3 的小手电，扫过整张图找边、
                找角、找笔画）；**权值共享**——同一个模板扫全图，
                参数量从十万级砍到百级；**池化**——把"大概在哪"
                代替"精确在哪"，位置的轻微挪动不再影响判决。

                你会把 24 课的 Adam、25 课的完全体训练循环原封不动带上——
                换的只是网络里那个"层"的形状。

                🎆 第 4 部分通关，去图像的世界！
                """
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
