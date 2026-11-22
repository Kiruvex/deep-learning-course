import marimo

__generated_with = "0.25.2"
app = marimo.App(app_title="39 · QKV三角色", width="medium")


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
        # 39 · QKV 三角色：把打分函数升级成可学习的

        **从零到 Transformer · 手写深度学习** ｜ 第 7 部分 · Transformer（2/8）
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📖 本课目标（先读我）": mo.md(
                """
                **38 课的注意力有一个隐藏的偷懒：查询、键、值
                全都是词向量本尊——一个向量身兼三职。**
                这一课把它拆开，装上三张可学习的矩阵。

                为什么必须拆？先看一个尴尬的实验：让一句话
                里每个词用**裸点积**看所有词（Q=K=V=X），
                你会发现**每个词最像的词是自己**（x·x=|x|²，
                柯西-施瓦茨不等式的亲儿子）——信息基本
                不流动，注意力成了回音室。

                拆法朴素得漂亮：给每个词发三顶帽子——

                - **Q**（query，查询）：我想找什么
                - **K**（key，键）：我可以被怎样找到
                - **V**（value，值）：找到我之后，我交出什么

                各乘一张可学习的矩阵（W_q、W_k、W_v），
                38 课的三步就升级成正式公式：

                ```
                Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V
                ```

                学完你将能：

                - 用**自恋实验**说清"身兼三职"的问题所在
                - 亲手**训练** W_q/W_k/W_v 完成一个检索任务
                  （"重"→取出书、"大"→取出箱子），看角色
                  分离怎么被梯度学出来
                - 推导**√d_k 缩放**的方差账本（38 课埋的针，
                  这里拔掉）：为什么维度越大越要除
                - 画出一次前向传播的**形状流水线**，数清
                  注意力层的参数量（3d²）

                前置：38 课三步注意力、21 课 softmax 与
                数值稳定、22-23 课反向传播、24 课 Adam。

                本课是 Transformer 的**心脏手术课**：
                43 课的完整编码器，用的就是今天这块积木。
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "🎩 概念 1/3 · 三顶帽子：一个向量为什么不能身兼三职": mo.md(
                """
                想象你在图书馆找书。你脑子里有个**查询**（"讲
                注意力的、中文的"）；每本书书脊上有**键**
                （分类标签，供比对）；翻开书读到的是**值**
                （真正的内容）。三个角色天然不同：**用什么
                找**（Q）、**被怎么找到**（K）、**找到后给
                什么**（V）——强行让一个东西干三件事，就会
                顾此失彼。

                反映到向量上：**同一个词，作为查询、作为
                键、作为值，需要暴露的信息不同。**"书"被
                "重"查询时，键应该突出"我重 0.9"；但作为
                值被取出时，应该交出完整身份。让 W_k 和
                W_v 两个矩阵各自学各自的投影，就是给每个
                词配了三张会变形的脸。

                更要命的是自恋问题（演示 1 现场演示）：
                裸点积下 x·x = |x|² 恒不小于 x·y——**自己
                永远是自己的最优匹配**，注意力退化为
                照镜子。W_q ≠ W_k 之后，"我的查询"和
                "我的键"不再同一副面孔，镜子被拆掉，
                目光才能落到别人身上。
                """
            ),
            "📏 概念 2/3 · √d 缩放：一笔方差账": mo.md(
                """
                38 课练习 1 的预防针今天拔掉。设查询和键的
                每一维都服从标准正态、互相独立，点积是 d
                项之和：

                ```
                q·k = q₁k₁ + q₂k₂ + … + q_d k_d
                E[q·k] = 0，Var(q·k) = d，std = √d
                ```

                维度一大，分数天然膨胀：d=256 时随机点积
                的标准差就有 16——**什么都没相关，分数先
                涨到 ±16**。softmax 一看分数差 30，直接
                饱和成 one-hot（21 课：指数级放大差距），
                权重非 1 即 0，梯度归零——38 课练过手算的
                那个极限情形。

                解药一行：**除以 √d**，方差从 d 拉回 1，
                分数回到 softmax 的舒适区。注意是 √d 不是
                d：我们要的是方差归一（std 是 √d），不是
                范数归一。这一除写在原论文公式里，成了
                "scaled dot-product attention" 名字里的
                "scaled"。
                """
            ),
            "🔧 概念 3/3 · 完整公式与形状流水线": mo.md(
                """
                三步齐了，正式登场：

                ```
                Q = X·W_q    (T×d)(d×d) → (T×d)   发查询
                K = X·W_k    (T×d)(d×d) → (T×d)   挂键
                V = X·W_v    (T×d)(d×d) → (T×d)   备值
                S = Q·Kᵀ/√d  (T×d)(d×T) → (T×T)   打分
                A = softmax(S)          → (T×T)   归一
                O = A·V      (T×T)(T×d) → (T×d)   汇总
                ```

                三个观察：

                - **X 只出现一次**：每个词的原始向量乘三张
                  不同的矩阵，得到三副面孔——"戴帽子"的
                  正式写法
                - **参数量 3d²**：与序列长度 T 无关（33 课
                  RNN 的同款美德）；d=512 时约 79 万参数
                - **T×T 是唯一的平方项**：练习 3 会数这笔账，
                  它就是 38 课"平方的代价"的全部来源

                一个容易混的点：d_k 是**键的维度**，不一定
                等于词向量维度 d——40 课多头注意力会故意
                把 d_k 缩小（d/头数），这里先按下不表。
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
        ### 演示 1 · 自恋实验 + 角色分离训练

        前半场（自恋）：还是 38 课的句子，让每个词用
        **裸点积**看所有词（Q=K=V=X）。看热力图的对角线：
        每个词看自己的权重（均值 0.21）是看别人的
        （0.10）的两倍——**"重"照镜子，"书"在角落**，
        信息不流动。

        后半场（拆分 + 训练）：给三张矩阵 W_q/W_k/W_v
        初始化成 **单位阵 + 小噪声**（起点恰好就是裸
        点积），用 Adam 训练一个检索任务——查询"重"
        应取出"书"的向量、查询"大"应取出"箱子"的
        向量（损失 = 输出与目标的 MSE）。看训练后：
        注意力翻向正确的词、输出与目标余弦达到
        **1.000**——三顶帽子各司其职，镜子拆了。
        """
    )
    return


@app.cell
def _(np, plt):
    # 演示 1A · 自恋实验：裸点积的注意力热力图
    sent39 = ["小明", "把", "书", "放进", "箱子", "，", "很", "重"]
    feat39 = {                                   # 4 维：[人物, 物品, 体积, 重量]
        "小明": [1.0, 0.0, 0.3, 0.3], "把": [0, 0, 0, 0],
        "书":   [0.0, 1.0, 0.2, 0.9], "放进": [0.0, 0.3, 0.1, 0.1],
        "箱子": [0.0, 1.0, 0.9, 0.2], "，": [0, 0, 0, 0],
        "很":   [0, 0, 0, 0], "重": [0.0, 0.2, 0.1, 1.0],
    }
    X39 = np.array([feat39[w] for w in sent39], np.float64)

    _S = X39 @ X39.T                             # 裸点积：Q=K=V=X
    _E = np.exp(_S - _S.max(-1, keepdims=True))
    A_raw39 = _E / _E.sum(-1, keepdims=True)

    _diag = np.diag(A_raw39)
    _off = A_raw39.copy(); np.fill_diagonal(_off, 0)
    print(f"裸点积注意力：看自己（对角线）均值 {_diag.mean():.2f}"
          f" ｜ 看别人（非对角）均值 {_off.sum() / 56:.2f}")
    print("「重」的目光分布：", {sent39[i]: round(float(A_raw39[7, i]), 2) for i in (2, 4, 7)})

    _fig, _ax = plt.subplots(figsize=(7.2, 5.6))
    _im = _ax.imshow(A_raw39, cmap="Blues", vmin=0, vmax=0.5)
    _ax.set_xticks(range(8), sent39, rotation=45)
    _ax.set_yticks(range(8), sent39)
    for _i in range(8):
        for _j in range(8):
            if A_raw39[_i, _j] > 0.18:
                _ax.text(_j, _i, f"{A_raw39[_i, _j]:.2f}", ha="center",
                         va="center", fontsize=8,
                         color="white" if A_raw39[_i, _j] > 0.3 else "black")
    _ax.set_title("裸点积注意力矩阵（Q=K=V=X）：对角线发亮——人人照镜子", fontsize=11)
    _fig.colorbar(_im, shrink=0.8)
    _fig.tight_layout()
    _fig
    return (X39, feat39, sent39)


@app.cell
def _(X39, np, plt, sent39):
    # 演示 1B · 训练三顶帽子：检索任务（重→书、大→箱子）
    _d = 4
    qs39 = np.array([X39[7], [0.0, 0.2, 1.0, 0.1]], np.float64)   # 查询词：重 / 大
    tg39 = X39[[2, 4]]                                             # 目标：书 / 箱子

    def train_qkv39(X, qs, tg, steps=4000, lr=0.02, seed=1):
        """W 初始化为单位阵+噪声，Adam 训练检索任务。返回 (Wq, Wk, Wv, 曲线)。"""
        _r = np.random.default_rng(seed)
        _d = X.shape[1]
        _Ws = [np.eye(_d) + _r.normal(0, 0.08, (_d, _d)) for _ in range(3)]
        _ms = [np.zeros_like(_w) for _w in _Ws]
        _vs = [np.zeros_like(_w) for _w in _Ws]
        _hist = []
        for _step in range(1, steps + 1):
            _Wq, _Wk, _Wv = _Ws
            _Q, _K, _V = qs @ _Wq, X @ _Wk, X @ _Wv
            _S = _Q @ _K.T / np.sqrt(_d)
            _E = np.exp(_S - _S.max(-1, keepdims=True))
            _A = _E / _E.sum(-1, keepdims=True)
            _O = _A @ _V
            _diff = _O - tg
            _hist.append((_diff ** 2).mean() / 2)
            # ---- 反向传播（22-23 课手艺）----
            _dO = _diff / _diff.size
            _dA = _dO @ _V.T
            _dV = _A.T @ _dO
            _dS = _A * (_dA - (_dA * _A).sum(-1, keepdims=True))
            _dS /= np.sqrt(_d)
            _dQ = _dS @ _K
            _dK = _dS.T @ _Q
            _grads = [qs.T @ _dQ, X.T @ _dK, X.T @ _dV]
            # ---- Adam（24 课配方）----
            for _i in range(3):
                _ms[_i] = 0.9 * _ms[_i] + 0.1 * _grads[_i]
                _vs[_i] = 0.999 * _vs[_i] + 0.001 * _grads[_i] ** 2
                _Ws[_i] -= lr * (_ms[_i] / (1 - 0.9 ** _step)) / (
                    np.sqrt(_vs[_i] / (1 - 0.999 ** _step)) + 1e-8)
        return _Ws[0], _Ws[1], _Ws[2], np.array(_hist)

    Wq39, Wk39, Wv39, hist39 = train_qkv39(X39, qs39, tg39)
    print(f"检索任务训练：loss {hist39[0]:.4f} → {hist39[-1]:.6f}（Adam，4000 步）")

    # 训练后的注意力分布
    _Q, _K, _V = qs39 @ Wq39, X39 @ Wk39, X39 @ Wv39
    _S = _Q @ _K.T / 2.0
    _E = np.exp(_S - _S.max(-1, keepdims=True))
    A_learned39 = _E / _E.sum(-1, keepdims=True)
    _O = A_learned39 @ _V
    for _qi, _qn in enumerate(["重", "大"]):
        _top = np.argsort(A_learned39[_qi])[::-1][:3]
        print(f"查询「{_qn}」→ " + "  ".join(f"{sent39[i]}({A_learned39[_qi, i]:.2f})"
                                             for i in _top))
    _cos1 = _O[0] @ X39[2] / (np.linalg.norm(_O[0]) * np.linalg.norm(X39[2]))
    _cos2 = _O[1] @ X39[4] / (np.linalg.norm(_O[1]) * np.linalg.norm(X39[4]))
    print(f"输出 vs 目标余弦：重→书 {_cos1:.3f} ｜ 大→箱子 {_cos2:.3f}")

    _fig, _axes = plt.subplots(1, 2, figsize=(11.2, 4.0))
    _axes[0].plot(hist39, lw=1.5)
    _axes[0].set_yscale("log")
    _axes[0].set_xlabel("训练步数")
    _axes[0].set_ylabel("检索损失（MSE）")
    _axes[0].set_title("三顶帽子的训练曲线：0.097 → 0.000", fontsize=11)
    _axes[0].grid(True, alpha=0.3)
    _pos = np.arange(8)
    _axes[1].bar(_pos - 0.2, A_learned39[0], 0.38, label="查询「重」", color="tab:red")
    _axes[1].bar(_pos + 0.2, A_learned39[1], 0.38, label="查询「大」", color="tab:blue")
    _axes[1].set_xticks(_pos, sent39, rotation=45)
    _axes[1].set_ylabel("注意力权重")
    _axes[1].set_title("训练后的目光：重→书、大→箱子（镜子拆了）", fontsize=11)
    _axes[1].legend(fontsize=9)
    _axes[1].grid(True, axis="y", alpha=0.3)
    _fig.tight_layout()
    _fig
    return (A_learned39, Wq39, Wk39, Wv39, hist39, train_qkv39)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 2 · 缩放实验：一笔方差账

        随机造 2000 对查询/键（每维标准正态），看点积
        分布怎么随维度膨胀；再造 10 个候选，看 softmax
        的最大权重在**不缩放**时怎么滑向 1（饱和），
        **缩放后**怎么回到温和区。

        本机实测的账本（你的数字会几乎一样——这是
        数学，不是运气）：

        | d | 点积 std（理论 √d） | 10 候选最大权重：不缩放 → 缩放后 |
        |---|---|---|
        | 4 | 2.0（2.0） | 0.23 → 0.06 |
        | 16 | 4.0（4.0） | 0.29 → 0.07 |
        | 64 | 8.0（8.0） | 0.49 → 0.05 |
        | 256 | 15.5（16.0） | **0.90 → 0.05** |

        d=256 不缩放时，**随机分数就能让 softmax 输出
        0.90**——什么都没学到，先饱和了。一除 √d，
        各维度全部回到 0.05~0.07：起点公平，才有得学。
        """
    )
    return


@app.cell
def _(np, plt):
    # 演示 2 · 缩放实验：点积方差 ∝ d，softmax 饱和救火
    _rng = np.random.default_rng(0)
    _ds = [4, 16, 64, 256]
    _rows = []
    for _d in _ds:
        _q = _rng.normal(0, 1, (2000, _d))
        _k = _rng.normal(0, 1, (2000, _d))
        _s = (_q * _k).sum(1)
        _qq = _rng.normal(0, 1, (10, _d))
        _kk = _rng.normal(0, 1, (10, _d))
        _sc = _qq @ _kk.T
        _wr = np.exp(_sc - _sc.max()); _wr /= _wr.sum()
        _sc2 = _sc / np.sqrt(_d)
        _ws = np.exp(_sc2 - _sc2.max()); _ws /= _ws.sum()
        _rows.append((_d, _s.std(), _s.std() / np.sqrt(_d), _wr.max(), _ws.max()))
        print(f"d={_d:3d}：点积 std={_s.std():5.2f}（√d={np.sqrt(_d):4.1f}）"
              f"｜缩放后 std={(_s / np.sqrt(_d)).std():.2f}"
              f"｜10 候选最大权重：不缩放 {_wr.max():.3f} → 缩放后 {_ws.max():.3f}")

    _fig, _axes = plt.subplots(1, 2, figsize=(11.2, 4.0))
    _axes[0].bar([str(d) for d in _ds], [r[1] for r in _rows], 0.55,
                 color="tab:orange", label="实测 std")
    _axes[0].plot([str(d) for d in _ds], [np.sqrt(d) for d in _ds], "kx--",
                  ms=9, label="理论 √d")
    _axes[0].set_xlabel("维度 d")
    _axes[0].set_ylabel("随机点积的标准差")
    _axes[0].set_title("方差账本：std = √d，维度越大分数越炸", fontsize=11)
    _axes[0].legend(fontsize=9)
    _axes[0].grid(True, axis="y", alpha=0.3)
    _x = np.arange(4)
    _axes[1].bar(_x - 0.19, [r[3] for r in _rows], 0.38, color="tab:red", label="不缩放")
    _axes[1].bar(_x + 0.19, [r[4] for r in _rows], 0.38, color="tab:green", label="除以 √d")
    _axes[1].axhline(0.1, color="gray", ls=":", lw=1)
    _axes[1].text(3.4, 0.12, "均匀线 0.1", fontsize=8, color="gray", ha="right")
    _axes[1].set_xticks(_x, [f"d={d}" for d in _ds])
    _axes[1].set_ylabel("10 候选中的最大权重")
    _axes[1].set_title("softmax 饱度：d=256 不缩放已 0.90（假 one-hot）", fontsize=11)
    _axes[1].legend(fontsize=9)
    _axes[1].grid(True, axis="y", alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 3 · 形状流水线与参数账本

        把公式跑一遍，盯住每一步的形状——T 个词进去，
        T 个词出来，中间只有一次 T×T 的平方项：

        ```
        X(T×d) → Q/K/V(T×d) → S(T×T) → A(T×T) → O(T×d)
        ```

        参数账本：W_q、W_k、W_v 各 d×d，合计 **3d²，
        与 T 无关**。d=512（GPT-2 小号的四分之一）时
        是 78.6 万参数；对比 37 课的 LSTM 层
        （4×(d·d + d·V)）量级相当——但注意力一步
        并行算完所有位置，LSTM 要走 T 步。
        """
    )
    return


@app.cell
def _(np):
    # 演示 3 · 形状流水线（打印每步形状）+ 参数账本
    _rng = np.random.default_rng(0)
    _T, _d = 8, 4
    _X = _rng.normal(0, 1, (_T, _d))
    _Wq = _rng.normal(0, 0.3, (_d, _d))
    _Wk = _rng.normal(0, 0.3, (_d, _d))
    _Wv = _rng.normal(0, 0.3, (_d, _d))
    _Q, _K, _V = _X @ _Wq, _X @ _Wk, _X @ _Wv
    _S = _Q @ _K.T / np.sqrt(_d)
    _A = np.exp(_S - _S.max(-1, keepdims=True)); _A /= _A.sum(-1, keepdims=True)
    _O = _A @ _V
    print("形状流水线（T=8 词，d=4 维）：")
    for _nm, _arr in [("X（输入）", _X), ("Q = X·Wq", _Q), ("K = X·Wk", _K),
                      ("V = X·Wv", _V), ("S = Q·Kᵀ/√d", _S), ("A = softmax(S)", _A),
                      ("O = A·V（输出）", _O)]:
        print(f"  {_nm:14s} → {_arr.shape}")
    print(f"\n参数账本：3 × d² = 3 × {_d}² = {3 * _d ** 2}（本例）")
    print(f"d=512 时：3 × 512² = {3 * 512 ** 2:,} ≈ 79 万（与 T 无关）")
    print(f"d=4096（GPT-3 级）：3 × 4096² = {3 * 4096 ** 2:,} ≈ 5000 万")
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
        ### 交互实验 · 缩放实验室

        你来调**维度 d**（4~256）：左边看随机点积的
        分布直方图（±√d 两条理论线），右边看 10 个候选
        的 softmax 最大权重——不缩放 vs 除以 √d。

        玩法建议：

        - d=4：分布窄窄的，缩不缩放差别不大——小维度
          的奢侈
        - d=64：不缩放的最大权重已爬到 0.4~0.6，
          softmax 开始"装自信"
        - d=256：不缩放逼近 0.9+（假 one-hot），缩放后
          稳在 0.1 以下——**√d 是 softmax 的安全带**
        - 思考：如果分布图变宽 2 倍，最大权重的刻度
          会怎么变？（指数放大——21 课的旧知识）
        """
    )
    return


@app.cell
def _(mo):
    # 交互控制：维度 d
    d_slider39 = mo.ui.slider(4, 256, value=64, step=4, debounce=True,
                              label="🎚️ 维度 d（查询/键向量的长度）")
    d_slider39
    return (d_slider39,)


@app.cell
def _(d_slider39, mo, np, plt):
    # ===== 交互实验 · 缩放实验室（响应单元）=====
    _d = d_slider39.value
    _rng = np.random.default_rng(42)
    _q = _rng.normal(0, 1, (3000, _d))
    _k = _rng.normal(0, 1, (3000, _d))
    _s = (_q * _k).sum(1)
    _qq = _rng.normal(0, 1, (10, _d))
    _kk = _rng.normal(0, 1, (10, _d))
    _sc = _qq @ _kk.T
    _wr = np.exp(_sc - _sc.max()); _wr /= _wr.sum()
    _sc2 = _sc / np.sqrt(_d)
    _ws = np.exp(_sc2 - _sc2.max()); _ws /= _ws.sum()

    _fig, _axes = plt.subplots(1, 2, figsize=(11.2, 3.9))
    _axes[0].hist(_s, bins=48, color="tab:orange", alpha=0.85)
    _axes[0].axvline(np.sqrt(_d), color="k", ls="--", lw=1.2)
    _axes[0].axvline(-np.sqrt(_d), color="k", ls="--", lw=1.2)
    _axes[0].set_xlabel("随机点积 q·k 的取值")
    _axes[0].set_ylabel("频数")
    _axes[0].set_title(f"点积分布（d={_d}）：std={_s.std():.1f}，虚线=±√d={np.sqrt(_d):.0f}",
                       fontsize=10.5)
    _axes[0].grid(True, alpha=0.3)
    _axes[1].bar([0, 1], [_wr.max(), _ws.max()], 0.5,
                 color=["tab:red", "tab:green"])
    _axes[1].axhline(0.1, color="gray", ls=":", lw=1)
    _axes[1].text(1.35, 0.115, "均匀线 0.1", fontsize=8, color="gray", ha="center")
    _axes[1].set_xticks([0, 1], ["不缩放", "除以 √d"])
    _axes[1].set_ylim(0, 1.05)
    _axes[1].set_ylabel("10 候选最大权重")
    _axes[1].set_title(f"softmax 饱度：{_wr.max():.2f} vs {_ws.max():.2f}", fontsize=10.5)
    _axes[1].grid(True, axis="y", alpha=0.3)
    _fig.tight_layout()

    if _wr.max() > 0.7:
        _verdict = (f"**d={_d} · 危险区**：随机分数就能把最大权重推到 {_wr.max():.2f}——"
                    "还没学任何东西，softmax 已经装出笃定的样子。梯度被压平"
                    "（21 课：饱和区导数≈0），训练等于白跑。除以 √d 后"
                    f"回到 {_ws.max():.2f}：分数回吐到温和区，梯度活过来了。")
    elif _wr.max() > 0.35:
        _verdict = (f"**d={_d} · 观察区**：不缩放的最大权重 {_wr.max():.2f}，"
                    "已明显偏离均匀线 0.1——分布尾部开始支配 softmax。"
                    "缩放后 {_ws.max():.2f}，安全。")
    else:
        _verdict = (f"**d={_d} · 舒适区**：最大权重 {_wr.max():.2f}，"
                    "分布还窄，缩放与否差别不大——小维度的奢侈。"
                    "但真实模型 d 动辄 64~4096，安全带必须常系。")
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
                **练习 1 · 手算方差账**：查询和键每维独立
                标准正态，d=100 时随机点积的标准差约是
                多少？除以 √100 后又是多少？

                **练习 2 · 补全注意力**：把缩放点积注意力
                的三行核心补完——输入 Q（t×d）、K（n×d）、
                V（n×d），返回输出（t×d）。打分要缩放，
                softmax 要数值稳定。

                **练习 3 · 数参数**：d=512 的注意力层，
                W_q/W_k/W_v 三张矩阵合计多少参数（整数）？
                再算：序列长度从 128 涨到 512，参数量
                变不变？
                """
            )
        }
    )
    return


@app.cell
def _(mo, np):
    # ===== 练习 1 · 手算方差账 =====
    print("查询、键每维独立标准正态，d = 100")
    print("  Var(q·k) = d = 100 → std = ?")
    print("  除以 √100 = 10 之后 → std = ?")

    std_raw_ex1_39 = None                             # ← ① 不缩放时的 std（整数）
    std_scaled_ex1_39 = None                          # ← ② 缩放后的 std（整数）

    if std_raw_ex1_39 is not None and std_scaled_ex1_39 is not None:
        _r1, _r2 = 10.0, 1.0
        if abs(std_raw_ex1_39 - _r1) < 0.5 and abs(std_scaled_ex1_39 - _r2) < 0.1:
            _msg = ("✅ 正确：10 → 1。d 项独立乘积求和，方差累加成 d；"
                    "除以 √d 恰好把方差除回 1（(√d)²=d）。这就是缩放选 √d "
                    "而不是 d 的原因：目标是方差归一。d=4096 时这条安全带"
                    "把 std 从 64 拉回 1——没有它，softmax 天生饱和。")
        else:
            _msg = "❌ 参考：10 和 1（std=√d；除以 √d 后 std=1）"
    else:
        _msg = "⏳ 两空都填上（提示：std = √d）"
    mo.md(_msg)
    return (std_raw_ex1_39, std_scaled_ex1_39)


@app.cell
def _(mo, np):
    # ===== 练习 2 · 补全注意力 =====
    def attention_ex2_39(Q, K, V):
        """缩放点积注意力：Q(t,d)、K(n,d)、V(n,d) → 输出(t,d)。"""
        _S = None                                      # ← ① 补：打分（点积 + √d 缩放）
        _A = None                                      # ← ② 补：softmax（数值稳定版）
        return None                                    # ← ③ 补：加权汇总

    attention_ready_ex2_39 = None                      # ← 补完后，换成 attention_ex2_39

    if attention_ready_ex2_39 is not None:
        try:
            _rng = np.random.default_rng(0)
            _Q = _rng.normal(0, 1, (3, 8))
            _K = _rng.normal(0, 1, (5, 8))
            _V = _rng.normal(0, 1, (5, 8))
            _out = attention_ready_ex2_39(_Q, _K, _V)
            _S = _Q @ _K.T / np.sqrt(8)
            _E = np.exp(_S - _S.max(-1, keepdims=True))
            _ref = (_E / _E.sum(-1, keepdims=True)) @ _V
            if _out is not None and _out.shape == (3, 8) and np.allclose(_out, _ref, atol=1e-8):
                _msg = ("✅ 形状与数值全对。这就是 Attention(Q,K,V)=softmax(QKᵀ/√d)V "
                        "的完整实现——三个矩阵乘、一次 softmax，零循环。"
                        "43 课把它原封不动搬进编码器；你现在写出的这三行，"
                        "和 PyTorch 里 nn.MultiheadAttention 的核心是同一件事。")
            else:
                _msg = "❌ 输出形状应为 (3, 8) 且与参考一致（检查缩放用的是 √d_k）"
        except (TypeError, ValueError) as _e:
            _msg = f"❌ 运行出错：{_e}——三行：S=Q@K.T/√d；softmax 减 max；A@V"
    else:
        _msg = "⏳ 补三行：打分 → softmax → 汇总"
    mo.md(_msg)
    return (attention_ex2_39, attention_ready_ex2_39)


@app.cell
def _(mo):
    # ===== 练习 3 · 数参数 =====
    print("d = 512（GPT-2 小号的四分之一）")

    n_params_ex3_39 = None                            # ← ① Wq/Wk/Wv 合计参数量（整数）
    t_change_ex3_39 = None                            # ← ② T 从 128 → 512，参数量变化（填 0 表示不变，1 表示变）

    if n_params_ex3_39 is not None and t_change_ex3_39 is not None:
        _n = 3 * 512 ** 2
        if abs(n_params_ex3_39 - _n) < 100 and t_change_ex3_39 == 0:
            _msg = (f"✅ 正确：3×512² = {_n:,} ≈ 79 万，且与 T 无关。参数只由 d 决定——"
                    "这是 33 课 RNN「参数与序列长度无关」美德的延续。"
                    "T 变化改变的只有计算量（T×T 的打分矩阵）和显存，"
                    "不动参数。GPT 系列「同一模型、不同上下文长度」"
                    "的灵活性正源于此。")
        elif abs(n_params_ex3_39 - _n) < 100:
            _msg = f"❌ 参数量对，但 T 变化不影响参数——只影响 T×T 计算量"
        else:
            _msg = f"❌ 参考：3 × 512² = {_n:,}（约 79 万）；T 变化参数不变（填 0）"
    else:
        _msg = "⏳ 两空都填上（参数量 = 3d²）"
    mo.md(_msg)
    return (n_params_ex3_39, t_change_ex3_39)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 1 参考答案（先自己写过再看！）": mo.md(
                """
                **std = 10；除以 √100 后 std = 1。**

                ```
                Var(q·k) = Var(Σ qᵢkᵢ) = Σ Var(qᵢkᵢ) = d
                std(q·k) = √d = √100 = 10
                除以 √d：Var 变成 d/(√d)² = 1，std = 1
                ```

                关键一步是独立性让方差直接相加（每项
                Var(qᵢkᵢ)=E[qᵢ²]E[kᵢ²]=1）。缩放的本质是
                **方差归一化**：无论 d 是 64 还是 4096，
                分数的波动范围恒定，softmax 永远工作在
                有梯度的区间。原论文里一句话带过的
                "/√d_k"，背后就是这笔账。
                """
            ),
            "✅ 练习 2 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                _S = Q @ K.T / np.sqrt(Q.shape[-1])   # 打分 + 缩放
                _E = np.exp(_S - _S.max(-1, keepdims=True))  # 数值稳定
                _A = _E / _E.sum(-1, keepdims=True)   # softmax
                return _A @ V                          # 加权汇总
                ```

                四个细节：① 缩放取 √d_k（Q 最后一维），
                不是 √d_model——40 课多头会看到它们的
                区别；② 减最大值防溢出（21 课老坑）；
                ③ softmax 沿**最后一个轴**（候选维）归一；
                ④ 输出形状 (t, d)：每个查询得到一份
                候选的加权平均。这三行就是 Transformer
                的全部核心——剩下的课都是给它搭台子。
                """
            ),
            "✅ 练习 3 参考答案（先自己写过再看！）": mo.md(
                """
                **3 × 512² = 786,432 ≈ 79 万；T 变化，
                参数不变。**

                | 部件 | 参数量 | 随 T 变？ |
                |---|---|---|
                | W_q, W_k, W_v | 3d² = 78.6 万 | 否 |
                | 打分矩阵 S | 0（无参数） | 计算量 T² |
                | 40 课的 W_o | +d² = 26.2 万 | 否 |
                | 合计（完整层） | 4d² ≈ 105 万 | 否 |

                对比记忆：37 课 LSTM 层参数 4(d·d + d·V)，
                注意力层 4d²（算上 W_o）——量级相同。真正的
                差别在**计算图**：LSTM 是 T 步串行链，
                注意力是一层稠密矩阵乘。前者训练慢但
                计算省（O(T)），后者训练快但计算贵
                （O(T²)）——深度学习没有免费的午餐，
                只有换着法子付账。
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
                1. **三顶帽子**：Q（我想找什么）、K（我可被
                   怎样找到）、V（找到后交出什么）——同一
                   个词向量乘三张不同的矩阵，三种身份
                2. **自恋问题**：裸点积下 x·x=|x|² 恒占优，
                   每个词最像自己（实测对角 0.21 vs 非对角
                   0.10）——W_q≠W_k 拆掉镜子
                3. **角色分离可学**：检索任务上训 W_q/W_k/W_v
                   （初始 I+噪声），loss 0.097→0.000，
                   注意力翻向正确目标、输出余弦 1.000
                4. **√d 缩放**：随机点积方差=d、std=√d；
                   d=256 不缩放时 softmax 最大权重 0.90
                   （假饱和）；除以 √d 把方差拉回 1——
                   方差归一，不是范数归一
                5. **完整公式**：softmax(QKᵀ/√d)·V——三个
                   矩阵乘 + 一次 softmax，零循环，T 个位置
                   一次算完
                6. **参数账本**：3d² 与 T 无关（d=512 约
                   79 万）；唯一的平方项是 T×T 打分矩阵——
                   38 课"平方的代价"的全部来源
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

                **下一课预告** — `40_多头注意力`（第 7 部分 · Transformer）：

                一套 QKV 只能问一种问题："谁跟我相关？"
                但一句话里的关系是多元的：指代（它→书）、
                修饰（很→重）、句法（放进→箱子）……一个
                头问不过来。下一课把 d 维空间**切成 h 份**，
                每份独立跑一套 QKV——h 个"头"各自学各自
                的关系，最后拼接投影。你会亲手训练出两个
                分工明确的头：一个学会**盯前一个词**、
                另一个学会**盯前两个词**——分工是损失函数
                逼出来的，没人安排；还会撞上一个真实大模型
                训练也有的难题：**头的冷启动**（不是每个头
                都自己醒过来）。

                顺带解开本课按下的伏笔：多头里 d_k = d/h
                （比如 512/8=64）——**总计算量不变，
                换来 h 个视角**。
                """
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
