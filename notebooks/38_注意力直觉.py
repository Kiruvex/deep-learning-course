import marimo

__generated_with = "0.25.2"
app = marimo.App(app_title="38 · 注意力直觉", width="medium")


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
        # 38 · 注意力直觉：让每个词直接看到每个词

        **从零到 Transformer · 手写深度学习** ｜ 第 7 部分 · Transformer（1/8 · 开篇）
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📖 本课目标（先读我）": mo.md(
                """
                **第 7 部分开门第一课，只回答一个问题：
                如果每个位置能"一步直达"所有位置，会发生什么？**

                第 6 部分收官时留了两道伤口：**记忆要接力**
                （信息从第 1 个字传到第 50 个字，要走 49 步，
                每一步都有损耗——35 课的连乘病根）；**计算要
                排队**（读完第 1 个字才能读第 2 个，梯度沿
                整条时间轴回传）。LSTM 缓解了前者，治不了
                后者。

                这一课用一个具体到不能再具体的任务开场：
                **"小明把书放进箱子，它很重"——"它"是谁？**
                你扫一眼就有了答案，因为你的眼睛能**直接
                对比**"它"和前文所有词。今天就用最朴素的
                两个老朋友实现这份"直接对比"：**点积打分**
                （11 课：向量点积大 = 方向近 = 相关）+
                **softmax 加权平均**（21 课：分数变权重）。

                学完你将能：

                - 用"**打分 → softmax → 加权平均**"三步讲清
                  注意力的全部计算（一行循环都没有）
                - 亲手做**指代消解实验**：属性词从"重"换成
                  "大"，注意力权重从"书"翻向"箱子——上下文
                  改变查询，查询改变看法
                - 解释注意力输出为什么是**温和的加权平均**
                  而不是硬选择（可导，才能训练）
                - 用计时实验看清**串行 vs 并联**的结构差距
                  （T 步循环 vs 一次矩阵乘），以及 O(T²)
                  代价的初次现身

                前置：11 课点积与几何直觉、21 课 softmax、
                33 课 RNN 展开、35 课梯度消失。

                本课全程手工向量、零训练——先把直觉焊牢，
                39 课再给打分函数装上可学习的参数。
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "🔍 概念 1/3 · 注意力 = 打分 + 加权平均": mo.md(
                """
                注意力的全部计算，只有三步：

                ```
                ① 打分：s_i = query · key_i          （相关度）
                ② 归一：w = softmax(s)               （分数 → 权重）
                ③ 汇总：新表示 = Σ w_i × value_i     （加权平均）
                ```

                一个比喻：你在聚会里找人（query 是你的
                "找谁"），屋里每个人都举着名牌（key 是
                "我是谁"），你和每个名牌**比对一下**得到
                相似度分数，softmax 把分数变成权重，最后
                把所有人的信息**按权重调配**成一份给你
                用的汇总（value 是"我携带的信息"）。

                关键性质：**这是加权平均，不是硬选择**。
                softmax 永远不会输出 1 和 0，只会输出
                0.33 和 0.24——注意力"看重"谁，但不"独占"
                谁。听起来很怂？恰恰相反：**加权平均可导**
                （21 课的老道理），梯度能顺着权重流回打分
                函数；如果换成 argmax 硬选择，梯度到处为
                零，整个结构学不了。深度学习里"温和"往往
                不是缺点，是生存策略。
                """
            ),
            "📐 概念 2/3 · 点积：为什么能当打分器": mo.md(
                """
                打分函数为什么选点积？11 课的几何直觉直接
                搬过来：两个向量点积大，意味着**方向靠近**
                （夹角小）；点积接近零，意味着**正交无关**
                （一个在说重量，一个在说语法，互不相干）。

                ```
                query · key = |query| |key| cos θ
                ```

                把每个词的语义压缩成向量后，"它很重"里的
                "重"字分量就会和"书"的重量维度对上——点积
                自然变大。**语义相关 = 向量同向**，这是
                词向量世界（第 11 课埋的种子）的中心假设。

                一个本课先按下不表的隐患：维度 d 越大，
                点积是 d 项求和，方差随 d 线性增长——
                随机向量对的点积也能大到 ±√d。分数一
                大，softmax 就饱和成"赢家通吃"，梯度消失。
                解药是除以 √d 缩放——本课代码先悄悄用上，
                39 课正式讲透。
                """
            ),
            "⚡ 概念 3/3 · 从接力到直达：并行的结构本质": mo.md(
                """
                RNN 处理 T 个词：第 t+1 步的输入是第 t 步
                的隐状态——**串行链**，一步等一步，GPU 上
                万个核心排着队看一个核心干活。注意力处理
                T 个词：一个 (T×d)·(d×T) 的矩阵乘法，
                **所有词对所有词的相关度同时算完**——
                没有任何一步需要"等上一步"。

                ```
                RNN   ：h₁ → h₂ → h₃ → … → h_T     （T 次串行）
                注意力：A = softmax(X Xᵀ) → 一次矩阵乘 （0 次串行）
                ```

                但天下没有免费的午餐。RNN 的计算量随 T
                **线性**增长；注意力的相关度矩阵是 T×T
                ——**平方**增长。T=1024 时就是一百万个
                格子。这就是今天大模型"上下文长度"贵的
                原因：每翻一倍长度，注意力计算翻四倍。

                一句话总结本课的世界观：**用平方的代价，
                买零串行的自由**。短序列血赚，长序列要
                精打细算——第 7 部分结束前你会看到这个
                交易的全部账本。
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
        ### 演示 1 · 「它」指的是谁：一场指代消解

        句子：**小明把书放进箱子，它很重。**

        "它"的候选指代对象在前半句：小明、书、箱子
        （虚词"把""放进"、标点也列进去凑热闹）。我们给
        每个词手工构造 4 维语义向量——**[人物, 物品,
        体积, 重量]**——比如"书"是 [0, 1, 0.2, 0.9]
        （是物品、偏小、偏重），"箱子"是 [0, 1, 0.9,
        0.2]（是物品、大、轻）。

        查询怎么来？**"它"的向量 + 属性词的向量**：
        "重"给查询的重量维度充电，"大"给体积维度充电。
        然后对每个候选点积打分、softmax、看权重分布。

        看点：把"重"换成"大"，同一句话、同一个"它"，
        权重**从书翻向箱子**——上下文改变查询，查询
        改变看法。这就是注意力最迷人的性质：**看法
        是算出来的，不是存出来的**。
        """
    )
    return


@app.cell
def _(np, plt):
    # 演示 1 · 指代消解：手工语义向量 + 点积打分 + softmax
    cands38 = ["小明", "把", "书", "放进", "箱子", "，"]   # 候选（前半句全部）
    feat38 = {                                            # 4 维：[人物, 物品, 体积, 重量]
        "小明": [1.0, 0.0, 0.3, 0.3],
        "把":   [0.0, 0.0, 0.0, 0.0],
        "书":   [0.0, 1.0, 0.2, 0.9],
        "放进": [0.0, 0.3, 0.1, 0.1],
        "箱子": [0.0, 1.0, 0.9, 0.2],
        "，":   [0.0, 0.0, 0.0, 0.0],
    }
    it_vec38 = np.array([0.1, 0.9, 0.5, 0.5], np.float32)  # 「它」：指向物品，属性未知
    attr_vecs38 = {                                       # 属性词：给查询的对应维度充电
        "重": np.array([0.0, 0.2, 0.1, 1.0], np.float32),
        "大": np.array([0.0, 0.2, 1.0, 0.1], np.float32),
        "旧": np.array([0.0, 0.5, 0.1, 0.1], np.float32),
        "贵": np.array([0.0, 0.3, 0.2, 0.2], np.float32),
    }
    K38 = np.stack([feat38[w] for w in cands38]).astype(np.float32)

    def attend38(query, K, scale=True):
        """三步注意力：点积打分 → softmax → 返回权重。"""
        _s = K @ query
        if scale:
            _s = _s / np.sqrt(query.shape[0])             # 缩放：39 课正式讲，先当仪式
        _e = np.exp(_s - _s.max())
        return _e / _e.sum()

    _fig, _axes = plt.subplots(1, 2, figsize=(10.4, 3.7), sharey=True)
    for _ax, _attr in zip(_axes, ["重", "大"]):
        _q = it_vec38 + attr_vecs38[_attr]
        _w = attend38(_q, K38)
        _best = int(np.argmax(_w))
        _ax.bar(cands38, _w,
                color=["tab:red" if _i == _best else "tab:blue" for _i in range(len(cands38))])
        for _i, _v in enumerate(_w):
            _ax.text(_i, _v + 0.008, f"{_v:.2f}", ha="center", fontsize=9)
        _ax.set_title(f"「它」很{_attr} → 最像：{cands38[_best]}", fontsize=12)
        _ax.grid(True, axis="y", alpha=0.3)
    _axes[0].set_ylabel("注意力权重")
    _fig.suptitle("指代消解：属性词一换，「它」的目光就换了方向", fontsize=12.5)
    _fig.tight_layout()
    _fig

    # 「旧」「贵」：不提供区分度的属性
    for _attr in ["旧", "贵"]:
        _w = attend38(it_vec38 + attr_vecs38[_attr], K38)
        print(f"它很{_attr}：书 {_w[2]:.2f} vs 箱子 {_w[4]:.2f} —— 平分秋色，模型也拿不准")
    return (K38, attr_vecs38, attend38, cands38, feat38, it_vec38)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 2 · 吸收之后：「它」的新表示

        注意力的输出不是"选一个词"，而是**把候选词的
        向量按权重调配成一份新表示**。"它很重"的查询下，
        "它"的新向量 = 0.33×书 + 0.24×箱子 + 0.13×小明
        + ……——书占大头，但混着别人的成分。

        三个观察角度：

        - **方向**：新"它"与书的余弦相似度升到前排，
          与箱子的差距拉开——"它"的语义向"书"漂移
        - **温和**：即使书是最佳候选，权重也只有
          0.33——加权平均不独断，永远留有余地
        - **可导**：正因为温和，"它 → 书"的每一步
          都能求导，打分函数才能被训练（39 课的主题）
        """
    )
    return


@app.cell
def _(K38, attend38, cands38, it_vec38, np, plt):
    # 演示 2 · 「它」的新表示 = 候选向量的加权平均
    _q = it_vec38 + np.array([0.0, 0.2, 0.1, 1.0], np.float32)   # 「它 + 重」
    _w = attend38(_q, K38)
    new_it38 = _w @ K38                                          # 加权平均：一行矩阵乘

    _cos_book = float(new_it38 @ K38[2] / (np.linalg.norm(new_it38) * np.linalg.norm(K38[2])))
    _cos_box = float(new_it38 @ K38[4] / (np.linalg.norm(new_it38) * np.linalg.norm(K38[4])))
    print(f"「它」原向量：{np.round(it_vec38, 2)}")
    print(f"「它」新向量：{np.round(new_it38, 2)}（书 {_w[2]:.2f}份 + 箱子 {_w[4]:.2f}份 + 其余 {1 - _w[2] - _w[4]:.2f}份）")
    print(f"与「书」余弦相似度：{_cos_book:.3f} ｜ 与「箱子」：{_cos_box:.3f}")

    _fig, _ax = plt.subplots(figsize=(8.8, 3.6))
    _dims = ["人物", "物品", "体积", "重量"]
    _x = np.arange(4)
    _ax.bar(_x - 0.22, it_vec38, 0.42, label="「它」原来", color="tab:gray", alpha=0.75)
    _ax.bar(_x + 0.22, new_it38, 0.42, label="「它」吸收上下文后", color="tab:red")
    _ax.bar(_x + 0.22, K38[2] - new_it38, 0.42, bottom=new_it38,
            label="距「书」还差的部分", color="tab:red", alpha=0.25)
    _ax.set_xticks(_x, _dims)
    _ax.legend(fontsize=9)
    _ax.set_title("「它」的新表示向「书」漂移（重量维 0.5 → 0.39，混入了小明的 0.13 份）", fontsize=10.5)
    _ax.grid(True, axis="y", alpha=0.3)
    _fig.tight_layout()
    _fig
    return (new_it38,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 3 · 串行 vs 并联：一场计时实验

        同样处理一条 T 个词、d=512 维的序列（真实模型
        的工作尺寸），两边计时：

        - **RNN**：T 步 Python 循环，每步两次小矩阵乘，
          步与步环环相扣——想快也快不起来
        - **注意力**：一次 (T×d)@(d×T) 矩阵乘 + 一次
          softmax + 一次加权汇总，全部位置同时完成

        本机实测（你的数字会不同，形状一样）：短序列上
        注意力**快出好几倍**；T=512 仍有明显优势；到
        T=1024 倍数明显收窄——**O(T²) 的账单开始压过
        并行的红利**。同时盯住那个数字：**T=1024 的
        注意力矩阵有一百万个格子**，而 RNN 全程只
        需要 1024 个隐状态。并行的自由，是拿平方的
        存储和计算换的。
        """
    )
    return


@app.cell
def _(np, plt):
    # 演示 3 · 计时实验：T 步串行 vs 一次矩阵乘
    import time

    _rng = np.random.default_rng(0)
    _d = 512
    _rows = []
    for _T in (128, 512, 1024):
        _X = _rng.normal(0, 1, (_T, _d)).astype(np.float32)
        _W = _rng.normal(0, 0.04, (_d, _d)).astype(np.float32)
        _U = _rng.normal(0, 0.04, (_d, _d)).astype(np.float32)
        # ---- RNN：T 步串行 ----
        _t0 = time.perf_counter()
        _h = np.zeros(_d, np.float32)
        for _t in range(_T):
            _h = np.tanh(_X[_t] @ _W + _h @ _U)
        _t1 = time.perf_counter()
        # ---- 注意力：一次到位 ----
        _t2 = time.perf_counter()
        _S = _X @ _X.T / np.sqrt(_d)
        _A = np.exp(_S - _S.max(-1, keepdims=True))
        _A /= _A.sum(-1, keepdims=True)
        _out = _A @ _X
        _t3 = time.perf_counter()
        _rows.append((_T, (_t1 - _t0) * 1000, (_t3 - _t2) * 1000))
        print(f"T={_T:5d}：RNN {_T} 步串行 {(_t1-_t0)*1000:7.1f} ms ｜ "
              f"注意力 1 次矩阵乘 {(_t3-_t2)*1000:6.1f} ms ｜ "
              f"加速 {(_t1-_t0)/(_t3-_t2):4.1f}x ｜ 注意力矩阵 {_T}×{_T}={_T*_T:,} 格")

    _Ts = [r[0] for r in _rows]
    _fig, _ax = plt.subplots(figsize=(8.6, 3.8))
    _ax.plot(_Ts, [r[1] for r in _rows], "o-", color="tab:red", label="RNN：T 步串行")
    _ax.plot(_Ts, [r[2] for r in _rows], "s-", color="tab:blue", label="注意力：一次矩阵乘")
    for _T, _r1, _r2 in _rows:
        _ax.annotate(f"{_r1/_r2:.1f}x", xy=(_T, (_r1 + _r2) / 2), fontsize=10,
                     ha="center", color="gray")
    _ax.set_xlabel("序列长度 T（词数）")
    _ax.set_ylabel("耗时（毫秒）")
    _ax.set_title("串行 vs 并联：短序列碾压式加速，长序列被 O(T²) 追平", fontsize=11.5)
    _ax.legend()
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
        ### 交互实验 · 指代消解直播间

        你来出题：选**属性词**（重 / 大 / 旧 / 贵），
        调**温度 τ**（softmax 前给分数除以 τ），现场看
        "它"的目光分布。

        玩法建议：

        - "重" + τ=1：书领先——看领先幅度
        - "大" + τ=1：箱子反超——同一句话，结论反过来了
        - "旧"或"贵" + 任何温度：书和箱子五五开——
          属性词不提供区分度时，注意力也**诚实地拿不准**
        - τ 拉到 0.5：分布变尖，微弱优势被放大成"赢家"；
          τ 拉到 3：分布变平，回到"雨露均沾"——
          37 课温度采样 的同一枚旋钮，这里转到了
          "看法的锐度"上
        """
    )
    return


@app.cell
def _(mo):
    # 交互控制：属性词 + 温度
    attr_dd38 = mo.ui.dropdown(options=["重", "大", "旧", "贵"], value="重",
                               label="🏷️ 属性词（它很__）")
    tau_slider38 = mo.ui.slider(0.5, 3.0, value=1.0, step=0.1, debounce=True,
                                label="🌡️ 温度 τ（看法的锐度）")
    mo.hstack([attr_dd38, tau_slider38])
    return (attr_dd38, tau_slider38)


@app.cell
def _(K38, attr_dd38, attr_vecs38, attend38, cands38, it_vec38, mo, np, plt,
      tau_slider38):
    # ===== 交互实验 · 指代消解直播间（响应单元）=====
    _attr = attr_dd38.value or "重"
    _tau = tau_slider38.value
    _s = (K38 @ (it_vec38 + attr_vecs38[_attr])) / 2.0 / _tau       # 打分 + 缩放 + 温度
    _e = np.exp(_s - _s.max())
    _w = _e / _e.sum()
    _best = int(np.argmax(_w))

    _fig, _ax = plt.subplots(figsize=(8.8, 3.5))
    _ax.bar(cands38, _w,
            color=["tab:red" if _i == _best else "tab:blue" for _i in range(len(cands38))])
    for _i, _v in enumerate(_w):
        _ax.text(_i, _v + 0.008, f"{_v:.2f}", ha="center", fontsize=9)
    _ax.set_ylim(0, max(0.55, _w.max() + 0.07))
    _ax.set_ylabel("注意力权重")
    _ax.grid(True, axis="y", alpha=0.3)
    _ax.set_title(f"「它」很{_attr} · τ={_tau:.1f} → 最像：{cands38[_best]}", fontsize=12)
    _fig.tight_layout()

    _gap = abs(_w[2] - _w[4])
    if _attr in ("旧", "贵"):
        _verdict = (f"**「它很{_attr}」· 书 {_w[2]:.2f} vs 箱子 {_w[4]:.2f}**——"
                    "属性词没有区分度：两个候选打分几乎相同，注意力诚实地五五开。\n\n"
                    "人也一样：「那本书旧还是箱子旧？」——上下文里没有答案，"
                    "任何「独断」都是幻觉。软分布的另一面：**它不会假装确定**。")
    elif _gap < 0.06:
        _verdict = (f"**「它很{_attr}」· 书 {_w[2]:.2f} vs 箱子 {_w[4]:.2f}**——"
                    f"温度 τ={_tau:.1f} 把差距压平了：明明有倾向（重→书、大→箱子），"
                    "高温却让看法变钝。\n\n"
                    "τ 是注意力的「锐度旋钮」：τ→0 逼近 argmax（硬选择），"
                    "τ→大 回到平均主义。训练时 τ=1，推理生成时才像 37 课那样调它。")
    else:
        _verdict = (f"**「它很{_attr}」· 书 {_w[2]:.2f} vs 箱子 {_w[4]:.2f}**——"
                    f"「{_attr}」给查询的对应维度充电，目光清晰倒向 {cands38[_best]}。\n\n"
                    "同一句话、同一个「它」，只换一个属性词，权重分布就翻了——"
                    "**看法是现场算出来的，不是存在词里的**。这就是注意力 "
                    "（attention）名字里「注意」二字的含义：看哪里、看多深，"
                    "由当前查询决定。")
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
                **练习 1 · 手算权重**：两个候选的打分
                （缩放后）s = [2, 0]，softmax 后权重各是
                多少（两位小数）？——e²≈7.39，e⁰=1。

                **练习 2 · 补全加权平均**：把注意力第三步
                补完——输入权重 `w`（长度 n）和候选矩阵
                `V`（n×d），返回加权平均（一行代码的
                事件，矩阵乘法）。

                **练习 3 · 平方的账单**：T=1024 的注意力
                相关度矩阵有多少个格子（整数）？RNN 同样
                长度全程只需要多少个隐状态（整数）？——
                算完你对"上下文越长越贵"会有体感。
                """
            )
        }
    )
    return


@app.cell
def _(mo, np):
    # ===== 练习 1 · 手算权重 =====
    print("两个候选，缩放后打分 s = [2, 0]")
    print("  w₁ = e² / (e² + e⁰) = 7.39 / 8.39 = ?")
    print("  w₂ = e⁰ / (e² + e⁰) = 1.00 / 8.39 = ?")

    w1_ex1_38 = None                                 # ← ① 候选 1 的权重（两位小数）
    w2_ex1_38 = None                                 # ← ② 候选 2 的权重（两位小数）

    if w1_ex1_38 is not None and w2_ex1_38 is not None:
        _z = np.array([2.0, 0.0])
        _ref = np.exp(_z) / np.exp(_z).sum()
        if abs(w1_ex1_38 - _ref[0]) < 0.02 and abs(w2_ex1_38 - _ref[1]) < 0.02:
            _msg = (f"✅ 正确：{_ref[0]:.2f} / {_ref[1]:.2f}。2 分的差距（e² 倍 ≈ 7.4 倍的"
                    "优势）换来 0.88 的权重——softmax 对差距敏感，但永远不给零。"
                    "想想：如果打分是 [20, 0] 呢？（权重 0.999999998——饱和，"
                    "这正是 39 课要除以 √d 的原因）")
        else:
            _msg = f"❌ 参考：{_ref[0]:.2f} / {_ref[1]:.2f}（分母是两个 e 值之和）"
    else:
        _msg = "⏳ 两空都填上（和为 1.00）"
    mo.md(_msg)
    return (w1_ex1_38, w2_ex1_38)


@app.cell
def _(mo, np):
    # ===== 练习 2 · 补全加权平均 =====
    def weighted_avg_ex2_38(w, V):
        """注意力第三步：w 是长度 n 的权重，V 是 n×d 的候选矩阵。"""
        return None                                   # ← 补：一行，返回 d 维加权平均

    weighted_ready_ex2_38 = None                      # ← 补完后，把这行换成 weighted_avg_ex2_38

    if weighted_ready_ex2_38 is not None:
        try:
            _w = np.array([0.7, 0.2, 0.1])
            _V = np.array([[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0]])
            _out = weighted_ready_ex2_38(_w, _V)
            _ref = _w @ _V
            if _out is not None and np.allclose(_out, _ref, atol=1e-8):
                _msg = (f"✅ 正确：输出 {np.round(_out, 2)}——0.7 份第一个候选 + 0.2 份第二个"
                        "+ 0.1 份第三个。w @ V 就地完成「Σ wᵢvᵢ」：行向量乘矩阵，"
                        "n 个 d 维向量调配成 1 个。演示 2 里「它」的新表示，"
                        "用的就是这一行。")
            else:
                _msg = f"❌ 输出应为 {np.round(_ref, 2)}（w @ V）"
        except (TypeError, ValueError) as _e:
            _msg = f"❌ 运行出错：{_e}——提示：w @ V（形状 (n,) @ (n,d) → (d,)）"
    else:
        _msg = "⏳ 补一行：w @ V"
    mo.md(_msg)
    return (weighted_avg_ex2_38, weighted_ready_ex2_38)


@app.cell
def _(mo):
    # ===== 练习 3 · 平方的账单 =====
    print("T = 1024（GPT-2 的上下文长度）")

    n_cells_ex3_38 = None                             # ← ① 注意力矩阵的格子数（整数）
    n_hidden_ex3_38 = None                            # ← ② RNN 全程的隐状态个数（整数）

    if n_cells_ex3_38 is not None and n_hidden_ex3_38 is not None:
        _c, _h = 1024 * 1024, 1024
        if abs(n_cells_ex3_38 - _c) <= 1024 and abs(n_hidden_ex3_38 - _h) <= 1:
            _msg = (f"✅ 正确：{_c:,}（约 105 万） vs {_h}。注意力的相关度矩阵 T×T 按"
                    "平方膨胀：长度翻倍，格子翻四倍；RNN 的隐状态永远只存 T 份。"
                    "这就是为什么大模型的「长上下文」是卖点——每 1 倍长度，"
                    "注意力要多付 4 倍计算。回头看演示 3 的 T=1024 行："
                    "加速倍数被追平，O(T²) 开始收租。")
        else:
            _msg = f"❌ 参考：{_c:,} / {_h}（格子数 = T×T；隐状态 = 每步 1 个 × T 步）"
    else:
        _msg = "⏳ 两空都填上（格子数是 T 平方）"
    mo.md(_msg)
    return (n_cells_ex3_38, n_hidden_ex3_38)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 1 参考答案（先自己写过再看！）": mo.md(
                """
                **w₁ = 0.88，w₂ = 0.12。**

                ```
                分母 = e² + e⁰ = 7.39 + 1.00 = 8.39
                w₁   = 7.39 / 8.39 ≈ 0.88
                w₂   = 1.00 / 8.39 ≈ 0.12
                ```

                值得咂摸的两点：① 打分差 2，权重比就是
                e² ≈ 7.4 倍——softmax 把线性差距变成
                **指数级**的权重比；② 但只要分数有限，
                权重永远不为零（0.12 不是 0）——梯度
                总有路可走。当分数差拉到 20 以上，
                权重比变成 e²⁰ ≈ 5 亿倍，softmax 实际
                死亡（梯度为零）——预防针：39 课的
                √d 缩放就是防这个的。
                """
            ),
            "✅ 练习 2 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                return w @ V        # (n,) @ (n, d) → (d,)
                ```

                注意力的第三步浓缩成一次矩阵乘。把 T 个
                查询摞成矩阵 Q（T×d），第三步就变成
                **W @ V**（T×T 乘 T×d → T×d）——T 个词
                的新表示**一次算完**，没有任何循环。
                38 课"并联"的全部秘密，就是这三步都能
                写成矩阵乘：打分 Q·Kᵀ、归一 softmax、
                汇总 W·V。39 课把 Q、K、V 三个名字
                正式颁发给它们。
                """
            ),
            "✅ 练习 3 参考答案（先自己写过再看！）": mo.md(
                """
                **格子数 = 1024² = 1,048,576（约 105 万）；
                隐状态 = 1024 个。**

                | T | 注意力格子 T² | RNN 隐状态 T |
                |---|---|---|
                | 128 | 1.6 万 | 128 |
                | 512 | 26 万 | 512 |
                | 1024 | **105 万** | 1024 |
                | 128K（GPT-4 级） | 170 亿 | 13 万 |

                注意力让每个词看到所有词，代价是**每对
                关系都要存一个分数**。长度 12.8 万的上下文
                需要 170 亿个格子——这是今天"长上下文"
                研究的全部动机（稀疏注意力、滑动窗口、
                FlashAttention……都在给这张表瘦身，
                但没有一张敢完全丢掉它：**看得全，才叫
                注意力**）。
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
                1. **注意力三步**：点积打分（相关度）→
                   softmax（分数变权重）→ 加权平均
                   （权重调配信息）——全部矩阵乘，零循环
                2. **看法是算出来的**：查询由上下文现场
                   构成（"它"+"重"），同一句话换个属性词，
                   权重分布就翻向——不是存在词里的死知识
                3. **软选择是生存策略**：加权平均不独断
                   （最高权重 0.33），但处处可导——argmax
                   硬选择不可导，学不了
                4. **点积 = 语义相关度**：向量同向 = 相关
                   （11 课几何直觉的直接应用）；隐患：维度
                   大时分数方差涨，softmax 饱和——√d 缩放
                   是解药（39 课讲透）
                5. **串行 vs 并联**：RNN 是 T 步接力，
                   注意力一次矩阵乘——实测 T=128 快约
                   7 倍；但相关度矩阵 T×T 平方膨胀
                6. **平方的账单**：T=1024 就是 105 万个
                   格子——长度翻倍、计算翻四倍，大模型
                   "长上下文"贵的根本原因
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

                **下一课预告** — `39_QKV三角色`（第 7 部分 · Transformer）：

                本课的打分是"裸点积"：查询直接拿词向量
                去点。但"它"找指代、动词找主语、形容词
                找名词——**不同的任务需要不同的比对方式**。
                与其手工设计，不如让网络自己学：给每个词
                发三顶帽子——**Q**（查询：我想找什么）、
                **K**（键：我可被怎样找到）、**V**（值：
                我携带什么信息），各自乘一个可学习的矩阵
                再做本课的三步。

                你会发现缩放 √d 的正式理由（今天埋的针），
                以及注意力公式完整形态的每一块是怎么从
                本课的朴素版一步步长出来的——**Attention
                is all you need** 的标题句，39 课见分晓。
                """
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
