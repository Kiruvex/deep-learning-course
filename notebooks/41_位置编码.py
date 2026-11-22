import marimo

__generated_with = "0.25.2"
app = marimo.App(app_title="41 · 位置编码", width="medium")


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
        # 41 · 位置编码：给每个位置发一张身份证

        **从零到 Transformer · 手写深度学习** ｜ 第 7 部分 · Transformer（4/8）
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📖 本课目标（先读我）": mo.md(
                """
                **一个被悄悄回避了三课的问题：注意力根本
                不知道"顺序"为何物。**

                把一句话的词序打乱喂给注意力，输出只是
                相应地重排一遍——一丁点都不觉得别扭。
                这叫**排列不变性**，是注意力的数学天性
                （它把输入当**集合**处理，不是序列）。
                但语言显然在乎顺序："狗咬人"和"人咬狗"
                用词完全相同。40 课我们偷偷给每个位置塞了
                one-hot 位置特征蒙混过关——今天把这个
                问题正面解决。

                方案：给每个位置发一张**身份证**，直接加在
                词向量上。三代身份证一课讲完：

                - **one-hot 位置**（40 课的权宜之计）：
                  直白，但维度绑死序列长度、无外推
                - **正弦位置编码**（原论文的选择）：不同
                  频率的 sin/cos 波叠加，最漂亮的性质是
                  **相对位置只需一次点积就能读出**
                - **旋转位置编码 RoPE**（今天大模型标配）：
                  概念预告——把"相对性"从点积副产品升级
                  为设计目标

                学完你将能：

                - 用**排列不变性实验**亲眼确认注意力是
                  "顺序盲"（打乱输入，输出只是跟着重排，
                  误差精确到机器精度）
                - 写出**正弦位置编码**并解释多尺度直觉：
                  低频波量"大概在哪"，高频波量"精确在哪"
                  ——一把游标卡尺
                - 推导并验证**相对位置性质**：
                  <PE_t, PE_{t+k}> 只依赖距离 k，与 t
                  无关（和角公式一行搞定）
                - 演示**外推**：位置 100 的编码照样合法
                  ——one-hot 做不到这一点

                前置：40 课多头注意力（用过 one-hot 位置）、
                11 课点积与三角函数、39 课缩放。

                本课没有训练，全是"编码的物理学"——
                把身份证的性质摸透，43 课组编码器时
                它就是一颗拧好的螺丝。
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "🔄 概念 1/3 · 排列不变性：注意力是顺序盲": mo.md(
                """
                数学一句话：设 P 是打乱行的置换矩阵，则

                ```
                Attention(P·X) = P·Attention(X)
                ```

                为什么？看三步流程：Q、K、V 逐行变换
                （打乱只是同步换行）；打分 Q·Kᵀ 的行列
                同步置换，**每行做完 softmax 还是原来那
                行**（softmax 沿行算，行的内容没变，
                只是搬了位置）；加权汇总同样跟着换行。
                全程没有任何一步用到"谁在前谁在后"。

                这不是 bug，是天性：注意力天生把输入当
                **集合**（bag）处理。对集合任务（"句子里
                有没有负面词"）这甚至是优点；但语言的核心
                结构——语序、层次、因果——全在排列里。

                解法只有一个：**把顺序写进向量本身**。
                位置编码（positional encoding, PE）就是
                干这个的：X' = X + PE，让每个词的表示
                从一开始就带着"我在第几位"的信息。
                """
            ),
            "🌊 概念 2/3 · 正弦编码：一把多尺度游标卡尺": mo.md(
                """
                原论文的公式（d 是编码维度，i 是维度对
                下标，t 是位置）：

                ```
                PE(t, 2i)   = sin(t / 10000^(2i/d))
                PE(t, 2i+1) = cos(t / 10000^(2i/d))
                ```

                每个维度对是一对同频率的正余弦波，频率
                从高到低几何级数递减（相邻对波长乘
                10000^(2/d) 倍）。d=64 时实测周期：

                | 维度对 | 周期（位置数） |
                |---|---|
                | 0 | ≈ 6 |
                | 2 | ≈ 11 |
                | 8 | ≈ 63 |
                | 15 | ≈ 471 |

                直觉：**一把游标卡尺**。高频维度对（周期 6）
                一格一格地数，分清"隔壁"和"隔两个"；
                低频维度对（周期 471）看大势，分清"开头"
                和"很远"。两个尺度拼起来，粗定位 + 细定位
                都有了——和十进制数数一个道理（个位管
                细，十位百位管粗）。

                另一个隐藏福利：每个位置编码的**范数恒定**
                （每对 sin²+cos²=1，共 d/2 对 → 范数 √(d/2)）
                ——不会因为位置靠后就越长，加到词向量上
                不会喧宾夺主。
                """
            ),
            "📐 概念 3/3 · 相对位置：点积读出距离": mo.md(
                """
                正弦编码最漂亮的性质，一行和角公式：

                ```
                <PE_t, PE_{t+k}> 的第 i 对
                  = sin(tθᵢ)·sin((t+k)θᵢ) + cos(tθᵢ)·cos((t+k)θᵢ)
                  = cos(k·θᵢ)          ← 与 t 无关！
                ```

                两个位置编码的点积**只依赖距离 k**，与它们
                具体在哪无关——"你在我前面 3 格"天然可读，
                "你是第 47 位我是第 50 位"反而不可读。这正
                好对上语言的直觉：相对位置（隔壁、上一段）
                比绝对位置（第 47 个词）更有语言学意义。

                每个维度对给出一个 cos(kθᵢ) 读数，32 个
                读数拼成距离的**条形码**：高频对在近距离
                上灵敏（k=1 就从 1 掉到 0.54），低频对在
                远距离上才有分辨力（k=32 仍接近 1）。
                注意力想学"看前一个词"，只需让 Q/K 的
                投影对准这条条形码的某几个频道。

                外推是免费的副产品：sin/cos 对任意 t 有
                定义，训练时最长 48，推理时来个位置 100
                照样编码（演示 3 验证）。one-hot 做不到：
                位置 100 需要一条新的维度，序列一长参数
                就得重造。
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
        ### 演示 1 · 顺序盲实验：打乱一句话试试

        8 个词向量做一次注意力（39 课的三步），然后
        随机打乱行序再做一次。如果注意力有任何一点
        "顺序感"，两次输出之间就该出现不可预测的
        差异；实际结果是——**打乱后的输出恰好等于
        原输出的相应重排，误差 2×10⁻¹⁶（机器精度）**。

        注意力对顺序的无感是**精确的数学事实**，
        不是近似。词序信息必须来自别处——位置编码。
        """
    )
    return


@app.cell
def _(np, plt):
    # 演示 1 · 排列不变性实验
    _rng = np.random.default_rng(0)
    _T, _d = 8, 16
    _X = _rng.normal(0, 1, (_T, _d))
    _Wq = _rng.normal(0, 0.3, (_d, _d))
    _Wk = _rng.normal(0, 0.3, (_d, _d))
    _Wv = _rng.normal(0, 0.3, (_d, _d))

    def attn41(X):
        _Q, _K, _V = X @ _Wq, X @ _Wk, X @ _Wv
        _S = _Q @ _K.T / np.sqrt(_d)
        _E = np.exp(_S - _S.max(-1, keepdims=True))
        _A = _E / _E.sum(-1, keepdims=True)
        return _A @ _V

    _O1 = attn41(_X)
    _perm = _rng.permutation(_T)
    _O2 = attn41(_X[_perm])
    _diff = np.abs(_O2 - _O1[_perm]).max()
    print(f"随机排列：{_perm}（第 i 个数 = 原句第 i 个词的新位置）")
    print(f"打乱后输出 vs 原输出重排：最大差异 {_diff:.1e}（机器精度 ≈ 2.2e-16）")
    print("→ 注意力精确地不在乎顺序：输出只是行跟着词搬家")

    _fig, _axes = plt.subplots(1, 2, figsize=(10.8, 3.6))
    _axes[0].imshow(_O1, cmap="RdBu_r", vmin=-1.5, vmax=1.5)
    _axes[0].set_title("原顺序的注意力输出", fontsize=11)
    _axes[0].set_ylabel("词的位置")
    _axes[1].imshow(_O2, cmap="RdBu_r", vmin=-1.5, vmax=1.5)
    _axes[1].set_title(f"打乱后（排列 {_perm[:3]}…）", fontsize=11)
    _fig.suptitle("两张图互为行重排——注意力是顺序盲的", fontsize=12)
    _fig.tight_layout()
    _fig
    return (attn41,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 2 · 正弦编码长什么样：波纹与条纹

        两张图看同一份编码（12 个位置 × 64 维）：

        - **左边（编码热力图）**：横轴维度、纵轴位置。
          高频维度对（左边）波纹密，低频对（右边）
          波纹疏——游标卡尺的粗细刻度肉眼可见
        - **右边（点积矩阵）**：G = PE·PEᵀ，12×12。
          亮带沿对角线平行排布——**G 只依赖 |t−u|**：
          距离 1 的点积恒为 30.9、距离 2 恒为 28.3、
          距离 5 恒为 23.5，换个 t 丝毫不变

        右图就是"相对位置可读"的可视化：想找"前一个
        词"的注意力头，只需认出这条条纹里的某一带。
        """
    )
    return


@app.cell
def _(np, plt):
    # 演示 2 · 正弦编码热力图 + 点积条纹
    def pe_sin41(pos, d, base=10000.0):
        """正弦位置编码：pos 标量或数组，返回 (len(pos), d)。"""
        _pos = np.atleast_1d(np.asarray(pos, np.float64))
        _pe = np.zeros((len(_pos), d))
        for _i in range(d // 2):
            _ang = _pos / (base ** (2 * _i / d))
            _pe[:, 2 * _i] = np.sin(_ang)
            _pe[:, 2 * _i + 1] = np.cos(_ang)
        return _pe

    d_pe41 = 64
    PE41 = pe_sin41(np.arange(12), d_pe41)
    G41 = PE41 @ PE41.T

    print("各维度对的周期（d=64）：")
    for _i in (0, 2, 8, 15):
        _per = 2 * np.pi * 10000 ** (2 * _i / d_pe41)
        print(f"  维度对 {_i:2d}（第 {2*_i}/{2*_i+1} 维）：周期 ≈ {_per:,.0f} 个位置")
    print(f"\n点积只看距离：距离1 → {G41[1, 0]:.1f}｜距离2 → {G41[2, 0]:.1f}"
          f"｜距离5 → {G41[5, 0]:.1f}｜距离11 → {G41[11, 0]:.1f}")
    print(f"每个位置的编码范数：{np.linalg.norm(PE41[0]):.3f}"
          f"（= √(d/2) = √32——恒定，不随位置变长）")

    _fig, _axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
    _im0 = _axes[0].imshow(PE41, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    _axes[0].set_xlabel("维度（左高频 ↔ 右低频）")
    _axes[0].set_ylabel("位置 t")
    _axes[0].set_title("正弦编码：多尺度波纹（游标卡尺）", fontsize=11)
    _fig.colorbar(_im0, ax=_axes[0], shrink=0.8)
    _im1 = _axes[1].imshow(G41, cmap="viridis")
    _axes[1].set_xlabel("位置 u")
    _axes[1].set_ylabel("位置 t")
    _axes[1].set_title("点积矩阵 G = PE·PE^T：条纹只依赖 |t−u|", fontsize=11)
    _fig.colorbar(_im1, ax=_axes[1], shrink=0.8)
    _fig.tight_layout()
    _fig
    return (G41, PE41, d_pe41, pe_sin41)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 3 · 条形码与外推：位置 100 照样合法

        两件事收尾：

        - **距离条形码**：固定起点 0，看每个维度对的
          读数 cos(k·θᵢ) 随距离 k 怎么变。高频对
          （维度对 0）近距离就剧变：k=1 时 0.54、
          k=2 时 −0.42；低频对（维度对 15）到 k=32
          还是 0.91——粗尺和细尺各司其职
        - **外推**：把编码算到位置 111（本课其他实验
          只用过 0~11）——位置 100 与 95 的点积 =
          23.504 = 位置 5 与 0 的点积。**相对性质在
          训练范围之外原样成立**。one-hot 在这里已经
          哑火（位置 100 需要新维度）
        """
    )
    return


@app.cell
def _(d_pe41, np, pe_sin41, plt):
    # 演示 3 · 距离条形码 + 外推
    _PE_ext = pe_sin41(np.arange(112), d_pe41)
    _G2 = _PE_ext @ _PE_ext.T

    print("距离条形码（各维度对读数 = cos(k·θᵢ)）：")
    print(f"{'距离 k':>6} | {'维度对0':>7} | {'维度对1':>7} | {'维度对5':>7} | {'维度对15':>7}")
    print("-" * 44)
    for _k in (0, 1, 2, 4, 8, 16, 32):
        _vals = [round(float(np.cos(_k / 10000 ** (2 * _i / d_pe41))), 2)
                 for _i in (0, 1, 5, 15)]
        print(f"{_k:>6} | {_vals[0]:>7} | {_vals[1]:>7} | {_vals[2]:>7} | {_vals[3]:>7}")

    print(f"\n外推检验（编码只算到过位置 11）：")
    print(f"  <PE₁₀₀, PE₉₅> = {_G2[100, 95]:.3f}")
    print(f"  <PE₅, PE₀>    = {_G2[5, 0]:.3f}（同为距离 5，分毫不差）")
    print(f"  位置 50/100 的编码范数：{np.linalg.norm(_PE_ext[50]):.3f} / "
          f"{np.linalg.norm(_PE_ext[100]):.3f}（恒定）")

    _fig, _ax = plt.subplots(figsize=(8.8, 3.8))
    _ks = np.arange(0, 48)
    _ax.plot(_ks, [_G2[_k, 0] for _k in _ks], "o-", ms=3, lw=1.4,
             label="总点积 <PE(0), PE(k)>（32 对读数之和）")
    _ax.axhline(0, color="gray", lw=0.8)
    _ax.set_xlabel("距离 k")
    _ax.set_ylabel("点积")
    _ax.set_title("距离的指纹：总点积随 k 单调衰减、轻微震荡（每对周期不同）", fontsize=11)
    _ax.legend(fontsize=9)
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()
    _fig
    return


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
        ### 交互实验 · 身份证检查站

        两个旋钮：

        - **维度对 i**：看这一对 sin/cos 波长什么样
          （左图）——选 0 是细尺，选 31 是粗尺
        - **距离 k**：看条形码在这一格的读数
          cos(k·θᵢ)（右图柱状，全部 32 对并排）——
          细尺在近距离就上下翻飞，粗尺纹丝不动

        玩法建议：

        - i=0 + k=1：细尺一格就走远（0.54）
        - i=31 + k=32：粗尺几乎没动（0.99）
        - 想想：要设计"看前一个词"的头，该对准哪些
          频道？（答案：细尺——近距离分辨力）
        """
    )
    return


@app.cell
def _(mo):
    # 交互控制：维度对 + 距离
    ipair_dd41 = mo.ui.dropdown(options=["0（最细尺）", "1", "5", "15", "31（最粗尺）"],
                                value="0（最细尺）", label="📏 维度对 i（尺的粗细）")
    k_slider41 = mo.ui.slider(0, 40, value=1, step=1, debounce=True,
                              label="🎯 距离 k（条形码读数）")
    mo.hstack([ipair_dd41, k_slider41])
    return (ipair_dd41, k_slider41)


@app.cell
def _(d_pe41, ipair_dd41, k_slider41, mo, np, plt):
    # ===== 交互实验 · 身份证检查站（响应单元）=====
    _i = int(ipair_dd41.value.split("（")[0])
    _k = k_slider41.value
    _theta = 1.0 / 10000 ** (2 * _i / d_pe41)
    _period = 2 * np.pi / _theta
    _read = np.cos(_k * _theta)

    _pos = np.arange(0, 48, 0.5)
    _fig, _axes = plt.subplots(1, 2, figsize=(11.4, 3.8))
    _axes[0].plot(_pos, np.sin(_pos * _theta), lw=1.6, label="sin(t·θ)")
    _axes[0].plot(_pos, np.cos(_pos * _theta), lw=1.6, ls="--", label="cos(t·θ)")
    _axes[0].axvline(_k, color="tab:red", ls=":", lw=1.2)
    _axes[0].set_xlabel("位置 t")
    _axes[0].set_title(f"维度对 {_i} 的波形（周期 ≈ {_period:,.0f} 个位置）", fontsize=10.5)
    _axes[0].legend(fontsize=9)
    _axes[0].grid(True, alpha=0.3)

    _pairs = np.arange(d_pe41 // 2)
    _reads = np.cos(_k / 10000 ** (2 * _pairs / d_pe41))
    _colors = ["tab:red" if _p == _i else "tab:blue" for _p in _pairs]
    _axes[1].bar(_pairs, _reads, color=_colors)
    _axes[1].axhline(0, color="gray", lw=0.8)
    _axes[1].set_xlabel("维度对 i（左细尺 ↔ 右粗尺）")
    _axes[1].set_ylabel("读数 cos(k·θ)")
    _axes[1].set_title(f"距离 k={_k} 的条形码（红 = 当前维度对）", fontsize=10.5)
    _axes[1].grid(True, axis="y", alpha=0.3)
    _fig.tight_layout()

    if _period < 20:
        _verdict = (f"**维度对 {_i} · 细尺**（周期 ≈ {_period:,.0f}）：k={_k} 时读数 "
                    f"{_read:.2f}——近距离灵敏，远距离绕圈（cos 是周期函数，"
                    "k 大了会折返）。细尺负责「隔壁还是隔俩」的精细分辨。\n\n"
                    "40 课学「看前一词」的头，最需要对准这种高频频道。")
    elif _period < 500:
        _verdict = (f"**维度对 {_i} · 中尺**（周期 ≈ {_period:,.0f}）：k={_k} 时读数 "
                    f"{_read:.2f}——中距离的主力。")
    else:
        _verdict = (f"**维度对 {_i} · 粗尺**（周期 ≈ {_period:,.0f}）：k={_k} 时读数 "
                    f"{_read:.2f}——几乎没动。粗尺负责「开头还是很远」的大势判断，"
                    "近距离全靠细尺。")
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
                **练习 1 · 手算编码**：d=64、base=10000，
                位置 t=3 的**第 0 维** PE(3, 0) = sin(3/1)
                和**第 1 维** PE(3, 1) = cos(3) 各是多少
                （两位小数）？——拿计算器或查 sin(3)≈0.141、
                cos(3)≈−0.990。

                **练习 2 · 补全编码函数**：正弦位置编码的
                核心两行——第 2i 维写 sin、第 2i+1 维写
                cos，角度是 pos 除以 10000^(2i/d)。

                **练习 3 · 两代身份证的账**：序列长 1000，
                用 one-hot 位置编码需要多少维？用正弦
                编码需要多少维（d=512 固定）？
                """
            )
        }
    )
    return


@app.cell
def _(mo, np):
    # ===== 练习 1 · 手算编码 =====
    print("d=64，base=10000，位置 t=3")
    print("  PE(3, 0) = sin(3 / 10000^0) = sin(3) = ?")
    print("  PE(3, 1) = cos(3 / 10000^0) = cos(3) = ?")

    pe0_ex1_41 = None                                 # ← ① PE(3, 0)（两位小数）
    pe1_ex1_41 = None                                 # ← ② PE(3, 1)（两位小数）

    if pe0_ex1_41 is not None and pe1_ex1_41 is not None:
        _r0, _r1 = np.sin(3.0), np.cos(3.0)
        if abs(pe0_ex1_41 - _r0) < 0.02 and abs(pe1_ex1_41 - _r1) < 0.02:
            _msg = (f"✅ 正确：{_r0:.2f} / {_r1:.2f}。第 0 个维度对频率最高（θ=1），"
                    "位置 3 的角度就是 3 弧度——sin/cos 直接吃弧度。这也是为什么"
                    "维度对 0 的周期是 2π≈6 个位置：角度每走 2π，波形转一圈。"
                    "低维对的 θ 越来越小（10000 的负幂），波长越来越长。")
        else:
            _msg = f"❌ 参考：sin(3)≈0.14、cos(3)≈−0.99（弧度制！）"
    else:
        _msg = "⏳ 两空都填上（注意是弧度制）"
    mo.md(_msg)
    return (pe0_ex1_41, pe1_ex1_41)


@app.cell
def _(mo, np):
    # ===== 练习 2 · 补全编码函数 =====
    def pe_ex2_41(pos, d, base=10000.0):
        """正弦位置编码（只补核心两行）。"""
        _pos = np.atleast_1d(np.asarray(pos, np.float64))
        _pe = np.zeros((len(_pos), d))
        for _i in range(d // 2):
            _ang = _pos / (base ** (2 * _i / d))
            _pe[:, 2 * _i] = None                    # ← ① 补：第 2i 维
            _pe[:, 2 * _i + 1] = None                # ← ② 补：第 2i+1 维
        return _pe

    pe_ready_ex2_41 = None                           # ← 补完后，换成 pe_ex2_41

    if pe_ready_ex2_41 is not None:
        try:
            _out = pe_ready_ex2_41(3, 64)
            _ref = np.zeros((1, 64))
            for _i in range(32):
                _a = 3.0 / 10000 ** (2 * _i / 64)
                _ref[0, 2 * _i] = np.sin(_a)
                _ref[0, 2 * _i + 1] = np.cos(_a)
            if _out is not None and _out.shape == (1, 64) and np.allclose(_out, _ref, atol=1e-9):
                _msg = ("✅ 正确。sin 进偶数维、cos 进奇数维，角度随维度对指数衰减"
                        "——这张表就是演示 2 左图那些波纹。原论文还证明了任意"
                        "有限位置的编码线性无关（PE 矩阵满秩）：每个位置的身份证"
                        "都不重号。")
            else:
                _msg = "❌ 输出应为 (1, 64) 且与参考一致（检查 sin/cos 的位置和角度）"
        except (TypeError, ValueError) as _e:
            _msg = f"❌ 运行出错：{_e}——_pe[:, 2i] = sin(角)；_pe[:, 2i+1] = cos(角)"
    else:
        _msg = "⏳ 补两行：sin 进偶数维，cos 进奇数维"
    mo.md(_msg)
    return (pe_ex2_41, pe_ready_ex2_41)


@app.cell
def _(mo):
    # ===== 练习 3 · 两代身份证的账 =====
    print("序列长 T=1000 的位置编码维度")

    dim_onehot_ex3_41 = None                         # ← ① one-hot 位置编码需要的维度（整数）
    dim_sin_ex3_41 = None                            # ← ② 正弦编码的维度（d=512 固定）

    if dim_onehot_ex3_41 is not None and dim_sin_ex3_41 is not None:
        _a, _b = 1000, 512
        if dim_onehot_ex3_41 == _a and dim_sin_ex3_41 == _b:
            _msg = ("✅ 正确：one-hot 要 1000 维（每个位置独占一维，维度绑死序列"
                    "长度——序列涨到 2000 就得重造 2000 维）；正弦编码恒 512 维"
                    "（= d），位置 1000、10000、任意大都只是换个角度代入 sin/cos。"
                    "这就是「外推」的参数化本质：**用函数代替查表**。今天的 "
                    "RoPE、ALiBi 把这条路线走到极致——外推性成了长上下文模型"
                    "的硬指标。")
        else:
            _msg = "❌ 参考：1000 维 / 512 维（正弦编码维度 = d，与序列长度无关）"
    else:
        _msg = "⏳ 两空都填上（一个跟着 T 走，一个恒等于 d）"
    mo.md(_msg)
    return (dim_onehot_ex3_41, dim_sin_ex3_41)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 1 参考答案（先自己写过再看！）": mo.md(
                """
                **PE(3, 0) = sin(3) ≈ 0.14；
                PE(3, 1) = cos(3) ≈ −0.99。**

                第 0 个维度对的频率参数 θ₀ = 10000⁰ = 1，
                位置 3 的角度就是 3 弧度。注意两件事：
                ① 三角函数吃的是**弧度**不是角度；
                ② sin 和 cos 永远成对出现——正是一对
                同频正余弦，才能在点积里用和角公式
                消掉绝对位置 t（概念 3 的关键）。

                顺带一算：PE(6, 0) = sin(6) ≈ −0.28，
                PE(9, 0) = sin(9) ≈ 0.41——第 0 维确实
                在以 2π ≈ 6.28 个位置为周期打转。
                """
            ),
            "✅ 练习 2 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                _pe[:, 2 * _i] = np.sin(_ang)      # 偶数维：sin
                _pe[:, 2 * _i + 1] = np.cos(_ang)  # 奇数维：cos
                ```

                为什么 sin 和 cos 要**交错**放而不是
                前一半 sin 后一半 cos？没有深刻理由，
                原论文就这么写的（工程惯例）；真正重要
                的是**成对**——和角公式需要同一频率的
                sin 和 cos 同时在场才能消掉 t。

                工程注记：真实实现用向量化的
                `pos[:, None] / base ** (2i/d)[None, :]`
                一次算完全部角度，避免 Python 循环；
                结果与本练习逐对循环完全一致。
                """
            ),
            "✅ 练习 3 参考答案（先自己写过再看！）": mo.md(
                """
                **one-hot：1000 维；正弦：512 维。**

                | 方案 | 维度 | 随 T 变？ | 外推 | 可学习？ |
                |---|---|---|---|---|
                | one-hot 位置 | T | 是 | 否（新位置=新维度） | 否（固定） |
                | 正弦编码 | d | 否 | 是（函数定义） | 否（固定） |
                | 可学习位置 | d | 否 | 否（查表） | 是（GPT-2 用） |
                | RoPE | d | 否 | 是 | 否（旋转） |

                有趣的历史：GPT-2 放弃正弦改用**可学习**
                位置嵌入（训练时学一张 (max_len, d) 的
                表）——外推没了，但学得更贴任务；今天
                的长上下文模型又回归**函数式**（RoPE、
                ALiBi）——外推重新变成硬指标。工程没有
                终极答案，只有权衡。
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
                1. **排列不变性**：Attention(P·X) =
                   P·Attention(X)——打乱输入，输出只是
                   相应重排（实测误差 2×10⁻¹⁶）；注意力
                   天生把输入当集合
                2. **解法**：位置信息必须写进向量——
                   X' = X + PE；40 课的 one-hot 位置是
                   权宜之计（维度绑死 T、无外推）
                3. **正弦编码**：每维度对一对同频
                   sin/cos，频率几何递减——游标卡尺：
                   高频量"隔壁还是隔俩"，低频量"开头
                   还是很远"
                4. **范数恒定**：每对 sin²+cos²=1 →
                   范数恒为 √(d/2)，位置不会越靠后
                   越喧宾夺主
                5. **相对性质**：<PE_t, PE_{t+k}> 的第 i 对
                   = cos(k·θᵢ)（和角公式）——只依赖距离；
                   32 个读数拼成距离的条形码
                6. **外推免费**：sin/cos 对任意位置有
                   定义（位置 100−95 的点积 = 5−0 的，
                   分毫不差）；one-hot 查表做不到——
                   今天 RoPE/ALiBi 把函数式路线走到极致
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

                **下一课预告** — `42_Transformer块`（第 7 部分 · Transformer）：

                零件齐了：多头注意力（40）、位置编码（41）、
                FFN？——还没讲，但你在 20 课就写过它的
                原型（两层 MLP + 激活）。下一课把零件
                组装成**一层楼**：

                - **残差连接**：x + F(x)——梯度的高速公路
                  （35 课的老朋友，但这次是结构标配）
                - **层归一化** LayerNorm：每层出口统一
                  规格，训练稳定的基石（和 25 课的
                  BatchNorm 是表亲）
                - **前馈网络** FFN：先膨胀 4 倍再压回——
                  注意力管"看谁"，FFN 管"看完之后想什么"
                - **Pre-LN vs Post-LN**：归一化放在残差
                  里还是外——一个改变训练稳定性的
                  工程抉择

                组好这一层，43 课就是把它叠 N 层、配上
                训练循环——完整编码器冲线。
                """
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
