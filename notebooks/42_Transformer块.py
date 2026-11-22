import marimo

__generated_with = "0.25.2"
app = marimo.App(app_title="42 · Transformer块", width="medium")


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
        # 42 · Transformer 块：把零件组装成一层楼

        **从零到 Transformer · 手写深度学习** ｜ 第 7 部分 · Transformer（5/8）
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📖 本课目标（先读我）": mo.md(
                """
                **零件全齐了：多头注意力（40）、位置编码
                （41）、FFN 的原型你在 20 课就写过（两层
                MLP）。这一课组装成一层楼——Transformer
                Block，43 课叠 N 层就是完整编码器。**

                三个新零件都不难，难的是"为什么非它们
                不可"：

                - **残差连接** x + F(x)：梯度的高速公路。
                  35 课见过连乘消失的病，这里是结构级
                  解药——30 层深塔实测：无残差梯度掉到
                  10⁻⁷，有残差稳在 20 上下（**差一亿倍**）
                - **层归一化** LayerNorm：每个位置自己的
                  特征向量内部归一。为什么不用 25 课的
                  BatchNorm？（序列变长、batch 统计不稳、
                  推理要挪参数——三宗罪）
                - **前馈网络** FFN：d → 4d → d，参数 8d²
                  ——**是注意力（4d²）的两倍**！注意力管
                  "看谁"，FFN 管"看完之后想什么"
                - **Pre-LN vs Post-LN**：归一化放残差里面
                  还是外面——一个决定训练要不要 warmup
                  的工程抉择

                学完你将能：

                - 用**深塔梯度实验**量化残差的价值（不是
                  听说，是亲手测出 10⁻⁷ vs 20）
                - 手写 **LayerNorm**（前向 + 反向，反向
                  严格对过数值梯度）并说清它和 BatchNorm
                  的分工
                - 画出 **Pre-LN / Post-LN 塔的激活与梯度
                  轨迹**，解释为什么 GPT-2 之后全员 Pre-LN
                - 数清一个 Block 的参数账：**12d²**

                前置：35 课梯度消失、25 课 BatchNorm 与
                训练循环、20 课 MLP、40 课多头注意力。

                本课的反向传播是全课程至今最立体的一课
                ——LN 的反向公式值得你手推一遍。
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "🛣️ 概念 1/3 · 残差连接：梯度的高速公路": mo.md(
                """
                一行代码的零件：

                ```
                y = x + F(x)      # F 是任意子层（注意力/FFN）
                ```

                前向的直觉：每个子层只学"**增量**"——
                别推翻 x，只往上面添砖。反向才是精髓：

                ```
                ∂y/∂x = 1 + ∂F/∂x
                ```

                那个 **1** 是恒等项：梯度回传时，不管 F
                内部多么崎岖（∂F/∂x 再小、再饱和），
                梯度至少有一条**原样通过**的路。深塔里
                无残差的梯度是 L 个 ∂F/∂x 的**连乘**
                （35 课的病根），有残差就变成 L 项
                **(1 + ∂F/∂x) 的连乘**——每项至少为 1，
                想消失都难。

                这就是"ResNet（2015）让深度学习走进百层
                时代"的那个 idea，Transformer 生在 ResNet
                之后，把它当默认配置。残差流（residual
                stream）有个更诗意的名字：**梯度高速公路**。
                """
            ),
            "📊 概念 2/3 · LayerNorm：每个位置自己管自己": mo.md(
                """
                LayerNorm 对**每个位置的特征向量**（每个
                词的 d 维表示）独立归一：

                ```
                LN(x) = γ ⊙ (x − μ) / √(σ² + ε) + β
                μ, σ：这一个位置自己的均值和标准差
                γ, β：可学习的缩放和偏移（d 维）
                ```

                为什么不用 BatchNorm（25 课）？三个理由：

                - **序列变长**：一个 batch 里句子长短不齐，
                  BN 沿 batch 统计的"同一特征位"在序列里
                  根本不是同一概念
                - **统计不稳**：batch 小、分布随训练漂移，
                  BN 的移动平均在 NLP 里很难伺候
                - **位置间独立**：LN 对每个位置独立计算，
                  一个句子的第 3 个词和第 30 个词互不干扰
                  ——天然适配变长输入

                一句话分工：**CV 用 BN（图像网格同构），
                NLP 用 LN（序列位置异质）**。顺带一提：
                LN 的反向公式是全课程最值得手推的一个
                （演示 2 的代码里藏着，练习 2 让你补前向）。
                """
            ),
            "🏗️ 概念 3/3 · 组装：Pre-LN 块与参数账": mo.md(
                """
                一层楼的两半（原始论文 Post-LN 在左，
                主流 Pre-LN 在右）：

                ```
                Post-LN（原始 2017）          Pre-LN（GPT-2 起）
                h = LN(x + Attn(x))           h = x + Attn(LN(x))
                y = LN(h + FFN(h))            y = h + FFN(LN(h))
                ```

                FFN 的样子（20 课的老朋友穿新衣）：

                ```
                FFN(x) = GELU(x·W₁ + b₁)·W₂ + b₂
                W₁: d×4d，W₂: 4d×d    →  参数 8d²
                ```

                先膨胀 4 倍再压回来：注意力收集的线索
                在 4d 维的宽广场里充分组合，再浓缩成
                对每个位置的"感想"。GELU 是 ReLU 的
                平滑表亲（GPT 系标配）。

                参数账（d=512 的一层 Block）：

                | 部件 | 参数 | 占比 |
                |---|---|---|
                | 多头注意力（含 W_o） | 4d² = 105 万 | 1/3 |
                | FFN（d→4d→d） | 8d² = 210 万 | **2/3** |
                | 合计 | **12d² = 315 万** | |

                惊讶吗？**FFN 是注意力的两倍大**——
                "看完之后想什么"比"看谁"更费参数。
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

        ## 🎬 演示
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 1 · 高速公路实测：30 层深塔的梯度

        造一座 L 层的塔（每层 tanh(x·W)，W 用小尺度
        初始化让每层增益 < 1——35 课的消失场景），
        末端施加单位扰动，用**解析反向传播**算输入端
        的梯度范数（实现严格对过数值梯度）。

        本机实测的账本：

        | 层数 L | 无残差 | 有残差 | 差距 |
        |---|---|---|---|
        | 5 | 1.97×10⁻¹ | 8.5 | 43 倍 |
        | 10 | 6.6×10⁻³ | 14.8 | 2 千倍 |
        | 20 | 1.0×10⁻⁵ | 13.6 | **130 万倍** |
        | 30 | 1.1×10⁻⁷ | 20.3 | **1.9 亿倍** |

        无残差：每过一层乘个 0.03，30 层下来归零；
        有残差：恒等项保底，梯度健康——而且**不随深度
        恶化**。这就是百层网络能训练的原因。
        """
    )
    return


@app.cell
def _(np, plt):
    # 演示 1 · 深塔梯度实验（解析反向，实现已对数值梯度校验）
    def tower_grad42(L, residual=True, d=32, seed=0, w_std=0.1):
        """L 层 tanh 塔的输入端梯度范数。"""
        _r = np.random.default_rng(seed)
        _Ws = [_r.normal(0, w_std, (d, d)) for _ in range(L)]
        _X = _r.normal(0, 1, (d,))
        _pres = []
        _h = _X
        for _W in _Ws:
            _p = _h @ _W
            _F = np.tanh(_p)
            _h = _h + _F if residual else _F
            _pres.append(_p)
        _g = np.ones(d)                              # 末端单位扰动
        for _i in reversed(range(L)):
            _sech2 = 1 - np.tanh(_pres[_i]) ** 2
            if residual:
                _g = _g + (_g * _sech2) @ _Ws[_i].T
            else:
                _g = (_g * _sech2) @ _Ws[_i].T
        return np.linalg.norm(_g)

    _Ls = [5, 10, 20, 30]
    _g_res = [tower_grad42(_L, True) for _L in _Ls]
    _g_no = [tower_grad42(_L, False) for _L in _Ls]
    print("L    无残差        有残差")
    for _L, _a, _b in zip(_Ls, _g_no, _g_res):
        print(f"{_L:3d}  {_a:.2e}     {_b:6.2f}   （差 {_b / max(_a, 1e-300):.0e} 倍）")

    _fig, _ax = plt.subplots(figsize=(8.6, 3.9))
    _ax.semilogy(_Ls, _g_no, "s--", color="tab:red", label="无残差：连乘消失")
    _ax.semilogy(_Ls, _g_res, "o-", color="tab:green", label="有残差：高速公路保底")
    _ax.set_xlabel("塔深 L（层数）")
    _ax.set_ylabel("输入端梯度范数")
    _ax.set_title("30 层深塔：残差连接让梯度相差一亿倍", fontsize=11.5)
    _ax.legend()
    _ax.grid(True, which="both", alpha=0.3)
    _fig.tight_layout()
    _fig
    return (tower_grad42,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 2 · LayerNorm：手写前向，对比 BatchNorm

        随机造一个 (6 位置 × 8 维) 的矩阵（均值 2、
        方差 3——故意的脏数据），分别做 LayerNorm
        （**沿行**：每个位置自己归一）和 BatchNorm
        （**沿列**：跨样本统计）。

        看逐行统计：LN 之后**每一行**的均值都是 0、
        方差都是 1.000——逐位置精确归一；BN 之后行内
        统计五花八门（+0.32/−1.15/+0.30…）——它归一
        的是列，行（= 单个样本）没人管。序列任务要的
        是前者：**每个词位置的表示各自规范**。
        """
    )
    return


@app.cell
def _(np, plt):
    # 演示 2 · LayerNorm vs BatchNorm 逐行统计
    def layernorm42(X, gamma=None, beta=None, eps=1e-5):
        """LN：沿最后一维（每个位置自己）。gamma/beta 可学习。"""
        _mu = X.mean(-1, keepdims=True)
        _var = X.var(-1, keepdims=True)
        _xhat = (X - _mu) / np.sqrt(_var + eps)
        if gamma is not None:
            return _xhat * gamma + beta
        return _xhat

    _rng = np.random.default_rng(0)
    _X = _rng.normal(2.0, 3.0, (6, 8))               # 6 位置 × 8 维，故意又偏又散
    _Y = layernorm42(_X)
    _Yb = (_X - _X.mean(0, keepdims=True)) / np.sqrt(_X.var(0, keepdims=True) + 1e-5)

    print(f"原始数据：行均值 {np.round(_X.mean(-1), 2)}")
    print("\n逐行统计对比（前 4 行）：")
    print("  行 |      LayerNorm        |      BatchNorm")
    for _i in range(4):
        print(f"  {_i}  | 均值 {_Y[_i].mean():+.1e}  方差 {_Y[_i].var():.3f}"
              f" | 均值 {_Yb[_i].mean():+.3f}  方差 {_Yb[_i].var():.3f}")
    print("\nLN：每行（每个位置）精确归一 → 序列的刚需")
    print("BN：归一的是列，行内统计没人管 → 适合同构网格（图像），不适合变长序列")

    _fig, _axes = plt.subplots(1, 2, figsize=(11.0, 3.9))
    _axes[0].imshow(_Y, cmap="RdBu_r", vmin=-2, vmax=2)
    _axes[0].set_title("LayerNorm 后：每行均值 0、方差 1", fontsize=11)
    _axes[0].set_ylabel("位置（样本）")
    _axes[0].set_xlabel("特征维")
    _im = _axes[1].imshow(_Yb, cmap="RdBu_r", vmin=-2, vmax=2)
    _axes[1].set_title("BatchNorm 后：每列均值 0（行内乱），方差不一", fontsize=11)
    _axes[1].set_xlabel("特征维")
    _fig.tight_layout()
    _fig
    return (layernorm42,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 3 · Pre-LN vs Post-LN：一座 20 层塔的体检

        两种块各造一座 20 层塔（Xavier 初始化，不
        训练——只看**随机初始化下的健康状况**），
        末端接一个随机读出向量，反向到输入端：

        - **激活轨迹（左）**：Pre-LN 的残差流像雪球，
          范数 8.6 → 26 一路涨——每层往高速路上**加**
          东西；Post-LN 每层出口都被 LN 钉死在 √d=8，
          残差流被反复"清零重来"
        - **梯度轨迹（右）**：Pre-LN 单调平滑（8.4 →
          24，健康）；Post-LN 起伏震荡（10.6 → 13.4 →
          9.4 → 9.8），每过一层 LN 都要钻一次窄门
          （除以该层残差流的方差）

        历史注脚：原始 Transformer（Post-LN）必须配
        **学习率预热**（warmup：前几千步从小学习率
        爬升）才训得稳；GPT-2 改用 Pre-LN 后训练
        皮实得多，从此 Pre-LN 成为主流。我们 43 课
        的编码器直接用 Pre-LN。
        """
    )
    return


@app.cell
def _(np, plt):
    # 演示 3 · Pre/Post-LN 塔的激活与梯度轨迹（随机初始化体检）
    def ln42(z, eps=1e-5):
        _mu = z.mean(-1, keepdims=True)
        return (z - _mu) / np.sqrt(z.var(-1, keepdims=True) + eps)

    def ln_backward42(gy, z_in, eps=1e-5):
        """LN 反向（单向量）：gz = [(gy−mean) − y·mean(gy·y)] / s。已对数值梯度校验。"""
        _mu = z_in.mean()
        _var = z_in.var() + eps
        _s = np.sqrt(_var)
        _y = (z_in - _mu) / _s
        return ((gy - gy.mean()) - _y * (gy @ _y) / len(z_in)) / _s

    def tower42(L, mode, d=64, seed=1):
        """Pre/Post-LN 塔：返回 (各层激活范数轨迹, 各深度梯度范数轨迹)。"""
        _w_std = 1.0 / np.sqrt(d)
        _r = np.random.default_rng(seed)
        _Ws = [_r.normal(0, _w_std, (d, d)) for _ in range(L)]
        _X = _r.normal(0, 1, (d,))
        _w_out = _r.normal(0, 1, (d,))
        _pres, _lns_in, _hs = [], [], [_X]
        _h = _X
        for _W in _Ws:
            if mode == "pre":
                _p = ln42(_h) @ _W
                _lns_in.append(_h)
                _h = _h + np.tanh(_p)
            else:
                _p = _h @ _W
                _z_in = _h + np.tanh(_p)
                _lns_in.append(_z_in)
                _h = ln42(_z_in)
            _pres.append(_p)
            _hs.append(_h)
        _g = _w_out.copy()
        _gnorms = [np.linalg.norm(_g)]
        for _i in reversed(range(L)):
            _sech2 = 1 - np.tanh(_pres[_i]) ** 2
            if mode == "pre":
                _dz = (_g * _sech2) @ _Ws[_i].T
                _g = _g + ln_backward42(_dz, _lns_in[_i])
            else:
                _gz = ln_backward42(_g, _lns_in[_i])
                _g = _gz + (_gz * _sech2) @ _Ws[_i].T
            _gnorms.append(np.linalg.norm(_g))
        return (np.array([np.linalg.norm(_h) for _h in _hs]),
                np.array(_gnorms))

    L_tower42 = 20
    act_pre42, g_pre42 = tower42(L_tower42, "pre")
    act_post42, g_post42 = tower42(L_tower42, "post")
    print(f"20 层塔体检（随机初始化 + 随机读出）：")
    print(f"  激活范数：Pre 8.6 → {act_pre42[-1]:.1f}（雪球）"
          f" ｜ Post 8.6 → {act_post42[-1]:.1f}（钉死 √64=8）")
    print(f"  输入端梯度：Pre {g_pre42[0]:.1f} → {g_pre42[-1]:.1f}（平滑上扬）"
          f" ｜ Post {g_post42[0]:.1f} → {g_post42[-1]:.1f}（震荡）")

    _fig, _axes = plt.subplots(1, 2, figsize=(11.4, 4.0))
    _axes[0].plot(act_pre42, "o-", ms=4, color="tab:green", label="Pre-LN（GPT-2 起）")
    _axes[0].plot(act_post42, "s-", ms=4, color="tab:orange", label="Post-LN（原始 2017）")
    _axes[0].set_xlabel("层数")
    _axes[0].set_ylabel("残差流的范数")
    _axes[0].set_title("激活轨迹：雪球 vs 每层清零", fontsize=11)
    _axes[0].legend(fontsize=9)
    _axes[0].grid(True, alpha=0.3)
    _axes[1].plot(g_pre42, "o-", ms=4, color="tab:green", label="Pre-LN")
    _axes[1].plot(g_post42, "s-", ms=4, color="tab:orange", label="Post-LN")
    _axes[1].set_xlabel("反向走过的层数（0=末端）")
    _axes[1].set_ylabel("梯度范数")
    _axes[1].set_title("梯度轨迹：平滑 vs 震荡（窄门效应）", fontsize=11)
    _axes[1].legend(fontsize=9)
    _axes[1].grid(True, alpha=0.3)
    _fig.tight_layout()
    _fig
    return (g_post42, g_pre42, ln42, ln_backward42, tower42)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ---

        ## 🎛️ 交互实验
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 交互实验 · 建塔师

        你来建塔：选**结构**（Pre-LN / Post-LN）和
        **层数**（5~40），现场体检——激活轨迹和梯度
        轨迹两张图实时刷新。

        玩法建议：

        - Post-LN + 40 层：看梯度震荡怎么随深度加剧
        - Pre-LN + 40 层：对比之下的平滑
        - 同一结构换层数，看末端激活：Pre 的雪球
          越滚越大（残差流能量随深度累积），Post
          永远钉在 8——两种哲学
        - 想一想：为什么 Pre-LN 的雪球不炸？（后面
          每层的 LN 先把输入归一再进子层——雪球
          再大，子层看到的都是标准尺寸）
        """
    )
    return


@app.cell
def _(mo):
    # 交互控制：结构 + 层数
    mode_dd42 = mo.ui.dropdown(options=["Pre-LN（GPT-2 起）", "Post-LN（原始 2017）"],
                               value="Pre-LN（GPT-2 起）", label="🏗️ 块结构")
    depth_slider42 = mo.ui.slider(5, 40, value=20, step=1, debounce=True,
                                  label="🔢 层数")
    mo.hstack([mode_dd42, depth_slider42])
    return (depth_slider42, mode_dd42)


@app.cell
def _(depth_slider42, mode_dd42, mo, plt, tower42):
    # ===== 交互实验 · 建塔师（响应单元）=====
    _mode = "pre" if "Pre" in mode_dd42.value else "post"
    _L = depth_slider42.value
    _acts, _gs = tower42(_L, _mode)

    _fig, _axes = plt.subplots(1, 2, figsize=(11.4, 3.9))
    _axes[0].plot(_acts, "o-", ms=4, color="tab:green" if _mode == "pre" else "tab:orange")
    _axes[0].axhline(8, color="gray", ls=":", lw=1)
    _axes[0].text(_L * 0.98, 8.6, "√d = 8", fontsize=8, color="gray", ha="right")
    _axes[0].set_xlabel("层数")
    _axes[0].set_ylabel("残差流范数")
    _axes[0].set_title(f"{_L} 层 {_mode.upper()} 塔的激活轨迹", fontsize=11)
    _axes[0].grid(True, alpha=0.3)
    _axes[1].plot(_gs, "o-", ms=4, color="tab:green" if _mode == "pre" else "tab:orange")
    _axes[1].set_xlabel("反向走过的层数")
    _axes[1].set_ylabel("梯度范数")
    _axes[1].set_title("梯度轨迹（末端 → 输入端）", fontsize=11)
    _axes[1].grid(True, alpha=0.3)
    _fig.tight_layout()

    _ratio = _gs[-1] / _gs[0]
    if _mode == "pre":
        _verdict = (f"**Pre-LN · {_L} 层**：激活范数 {_acts[0]:.1f} → {_acts[-1]:.1f}"
                    f"（雪球），输入端梯度是末端的 {_ratio:.1f} 倍——单调平滑。\n\n"
                    "LN 在子层入口：残差流再大，子层看到的都是归一后的标准尺寸；"
                    "梯度沿恒等路直通，每层只叠加小小的增量。训练皮实，不需要精心的 warmup。")
    else:
        _verdict = (f"**Post-LN · {_L} 层**：激活范数被每层出口的 LN 钉死在 √d={8}"
                    f"，输入端梯度是末端的 {_ratio:.2f} 倍——震荡起伏。\n\n"
                    "LN 在残差之外：梯度每过一层都要除以该层残差流的标准差（钻窄门），"
                    "轨迹忽高忽低。原始 Transformer 靠学习率预热驯服它；"
                    "GPT-2 之后大家干脆改用 Pre-LN。")
    mo.vstack([
        _fig,
        mo.callout(mo.md(_verdict), kind="info"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✏️ 动手练习（本课 3 题）": mo.md(
                """
                **练习 1 · 手算残差梯度**：某子层的局部
                导数 ∂F/∂x = 0.1（很饱和）。无残差时
                10 层连乘后梯度缩到多少？有残差时每层
                的 (1 + ∂F/∂x) 连乘后又是多少？

                **练习 2 · 补全 LayerNorm**：把 LN 前向
                的两行补完——先归一（减均值除标准差），
                再仿射（乘 γ 加 β）。

                **练习 3 · 数参数**：d=512 的一个完整
                Transformer 块（注意力 4d² + FFN 8d²），
                总参数多少（整数）？GPT-2 小号有 12 层，
                这些块合计多少参数？
                """
            )
        }
    )
    return


@app.cell
def _(mo):
    # ===== 练习 1 · 手算残差梯度 =====
    print("∂F/∂x = 0.1（子层很饱和），塔深 10 层")

    g_no_ex1_42 = None                               # ← ① 无残差：0.1^10 = ?（科学计数或小数）
    g_res_ex1_42 = None                              # ← ② 有残差：(1+0.1)^10 ≈ ?（两位小数）

    if g_no_ex1_42 is not None and g_res_ex1_42 is not None:
        _a, _b = 0.1 ** 10, 1.1 ** 10
        if abs(g_no_ex1_42 - _a) < 1e-10 and abs(g_res_ex1_42 - _b) < 0.05:
            _msg = (f"✅ 正确：{_a:.1e} vs {_b:.2f}。无残差连乘 0.1，10 层归零"
                    "（35 课的病）；有残差连乘 1.1，10 层反而涨到 2.6 倍——梯度"
                    "非但不消失，还缓慢增长（演示 1 里 Pre-LN 轨迹上扬的同款原因）。"
                    "残差的代价是可能梯度偏大，但「偏大」好治（clip，35 课），"
                    "「消失」难救——这笔交换稳赚。")
        else:
            _msg = f"❌ 参考：0.1^10 = 1e-10；（1+0.1）^10 ≈ 2.59"
    else:
        _msg = "⏳ 两空都填上（一个是 10 次方连乘）"
    mo.md(_msg)
    return (g_no_ex1_42, g_res_ex1_42)


@app.cell
def _(mo, np):
    # ===== 练习 2 · 补全 LayerNorm =====
    def ln_ex2_42(X, gamma, beta, eps=1e-5):
        """LayerNorm：X 最后一维归一后仿射。"""
        _mu = X.mean(-1, keepdims=True)
        _var = X.var(-1, keepdims=True)
        _xhat = None                                 # ← ① 补：归一化（减均值除标准差）
        return None                                  # ← ② 补：仿射（乘 gamma 加 beta）

    ln_ready_ex2_42 = None                           # ← 补完后，换成 ln_ex2_42

    if ln_ready_ex2_42 is not None:
        try:
            _X = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 40.0]])
            _g = np.array([2.0, 0.5, 1.0])
            _b = np.array([0.0, 1.0, -1.0])
            _out = ln_ready_ex2_42(_X, _g, _b)
            _xh = (_X - _X.mean(-1, keepdims=True)) / np.sqrt(_X.var(-1, keepdims=True) + 1e-5)
            _ref = _xh * _g + _b
            if _out is not None and np.allclose(_out, _ref, atol=1e-9):
                _msg = ("✅ 正确。两个细节：① eps 加在方差上（根号里）——方差"
                        "接近 0 时防止除零；② gamma/beta 是逐维（d 个）参数，"
                        "不是标量——归一化「抹平」了分布，仿射再把有用的尺度"
                        "学回来。LN 的反向公式比前向难一档（练习里不要求），"
                        "但前向这两行是 43 课编码器的现成零件。")
            else:
                _msg = "❌ 输出与参考不一致（检查 eps 的位置和 gamma/beta 的形状）"
        except (TypeError, ValueError) as _e:
            _msg = f"❌ 运行出错：{_e}——_xhat=(X−μ)/√(σ²+eps)；return _xhat*gamma+beta"
    else:
        _msg = "⏳ 补两行：归一 → 仿射"
    mo.md(_msg)
    return (ln_ex2_42, ln_ready_ex2_42)


@app.cell
def _(mo):
    # ===== 练习 3 · 数参数 =====
    print("d = 512 的一层 Transformer 块；GPT-2 小号共 12 层")

    block_params_ex3_42 = None                       # ← ① 一层的参数量（整数）
    twelve_params_ex3_42 = None                      # ← ② 12 层合计（整数）

    if block_params_ex3_42 is not None and twelve_params_ex3_42 is not None:
        _n1, _n12 = 12 * 512 ** 2, 12 * 12 * 512 ** 2
        if abs(block_params_ex3_42 - _n1) < 10000 and abs(twelve_params_ex3_42 - _n12) < 100000:
            _msg = (f"✅ 正确：一层 12d² = {_n1:,} ≈ 315 万；12 层 {_n12:,} ≈ 3770 万。"
                    "这还没算词嵌入（50257×768 ≈ 3860 万）——GPT-2 小号总参数"
                    " 1.24 亿，块占三成，嵌入占三成，剩下的在输出头。"
                    "注意 FFN 占每块的 2/3：大模型里「想」的参数比「看」的多——"
                    "这也是 MoE（混合专家）潮流的动机：让 FFN 稀疏化，只在需要时"
                    "激活部分专家。")
        else:
            _msg = f"❌ 参考：12×512² = {_n1:,}；12 层 = {_n12:,}"
    else:
        _msg = "⏳ 两空都填上（一层 = 4d² + 8d²）"
    mo.md(_msg)
    return (block_params_ex3_42, twelve_params_ex3_42)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 1 参考答案（先自己写过再看！）": mo.md(
                """
                **无残差：0.1¹⁰ = 10⁻¹⁰；
                有残差：1.1¹⁰ ≈ 2.59。**

                一个反直觉的好消息：残差不但防消失，
                还让梯度**随深度缓慢增长**（每层至少
                乘 1）——这就是演示 1 里 Pre-LN 梯度
                轨迹上扬、演示 3 里输入端梯度比末端大
                的原因。梯度偏大好治（clip 一下，35 课
                的手艺），梯度消失难救（信息没了就是
                没了）——残差这笔交易稳赚。

                更深一层：y = x + F(x) 迭代 L 次后，
                输出 = x + F₁ + F₂ + … + F_L——**每层
                的贡献是并列的加法项**，不是嵌套的乘法
                项。加法结构是深度网络的可训练性
                （trainability）之源。
                """
            ),
            "✅ 练习 2 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                _xhat = (X - _mu) / np.sqrt(_var + eps)
                return _xhat * gamma + beta
                ```

                三件容易被忽略的事：

                - **eps 在根号里、加在方差上**：防除零；
                  常取 1e-5
                - **gamma/beta 形状是 (d,)**：逐特征维的
                  仿射，广播到每个位置——归一化统一规格，
                  仿射学回有用的尺度
                - **keepdims=True**：让 (T, d) 的每行各自
                  统计——去掉它就变成全矩阵统计（错）

                PyTorch 里 `nn.LayerNorm(d)` 一行等价；
                但你现在知道它每一行在干什么——43 课
                直接把这两个手写零件用上。
                """
            ),
            "✅ 练习 3 参考答案（先自己写过再看！）": mo.md(
                """
                **一层 12d² = 3,145,728 ≈ 315 万；
                12 层 = 37,748,736 ≈ 3770 万。**

                | 部件 | 公式 | d=512 | 占比 |
                |---|---|---|---|
                | 注意力（3W+W_o） | 4d² | 105 万 | 33% |
                | FFN（W₁+W₂） | 8d² | 210 万 | 67% |
                | 一层合计 | 12d² | 315 万 | |
                | GPT-2 小号 12 层 | 144d² | 3770 万 | |

                FFN 占 2/3 这个事实影响深远：MoE（混合
                专家）把每层的 FFN 换成多个「专家」、
                每次只激活少数——用稀疏性把参数量和
                计算量解耦。GPT-4 被广泛猜测是 MoE
                架构——「会想的层」太大，只好请专家
                轮流值班。
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
                1. **残差连接**：y = x + F(x)，反向多出
                   恒等项 1——梯度高速公路；30 层塔实测
                   10⁻⁷ vs 20（差一亿倍），且不随深度恶化
                2. **加法结构**：深层输出 = x + ΣFᵢ，
                   各层贡献并列相加而非嵌套相乘——
                   可训练性之源；副作用是梯度缓涨（好治）
                3. **LayerNorm**：每个位置的特征向量
                   自己归一（μ、σ 是行内的）+ 可学习
                   γβ 仿射；逐行精确 0 均值 1 方差
                4. **LN vs BN**：BN 沿 batch 归一列
                   （适合图像同构网格）；LN 沿特征归一行
                   （适合变长序列、小 batch、推理无依赖）
                5. **FFN**：d→4d→d 两层 MLP（GELU），
                   参数 8d²——注意力的两倍，「看完之后
                   想什么」比「看谁」更贵
                6. **Pre-LN vs Post-LN**：Pre 的残差流
                   是雪球（子层入口先归一，不怕大），
                   梯度平滑；Post 每层清零、梯度震荡
                   （窄门效应）——原始论文要 warmup，
                   GPT-2 起 Pre-LN 一统江湖；一层 Block
                   合计 12d²
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
                ---

                **下一课预告** — `43_编码器实战`（第 7 部分 · Transformer）：

                图纸齐了：注意力（38-40）、位置编码
                （41）、Block（42）。下一课动工——**用
                NumPy 造一台完整的 Transformer 编码器
                并真刀真枪训练它**：

                - 词嵌入 + 正弦位置编码 + N 个 Pre-LN 块
                  叠塔，全部手写、全部可反向传播
                - 任务选**序列分类**（句子情感二分类的
                  玩具版）：T 个词进去，一个标签出来——
                  读池化（mean pooling）怎么把序列压成
                  一个向量
                - 训练循环还是 25 课那套骨架，但模型
                  换成 Transformer——亲眼看它比同尺寸
                  MLP 快多少、准多少
                - **注意力可视化收尾**：训好的模型里，
                  [CLS] 位置的注意力到底在看哪些词？

                这是第 7 部分的主战役——冲线之后，
                解码器（44）只是加一张掩码。
                """
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
