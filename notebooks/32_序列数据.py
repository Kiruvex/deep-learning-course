import marimo

__generated_with = "0.25.2"
app = marimo.App(app_title="32 · 序列数据", width="medium")


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
        # 32 · 序列数据：喂网络喝"时间"

        **从零到 Transformer · 手写深度学习** ｜ 第 6 部分 · 循环神经网络（1/6）
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📖 本课目标（先读我）": mo.md(
                """
                **这一课只回答一个问题：为什么"预测下一个"这么简单的
                事，我们 31 课攒下的工具箱就是接不住？**

                到现在为止，网络吃过的所有输入都是**定长**的：784 个
                像素、一张 28×28 的图。可语言、语音、股价、代码——
                这些数据是**一串一串**的：长度不固定、顺序不能换、
                而且下一个元素由**上文**决定。这种数据叫**序列**。

                第 5 部分开课时我们审判过"全连接看图"的三宗罪；今天
                会看到同一场审判在序列世界**原样重演**——位置绑定、
                参数挥霍、变长装不下。老祖宗的滑窗法（把窗口拉平喂
                MLP）能临时翻墙，但有个天花板：**窗口之外的事，
                一概不知。**

                学完你将能：

                - 说出序列数据的三特征：**变长、有序、上文决定下文**
                - 用"三堵墙"讲清全连接在序列上的水土不服
                  （26 课三宗罪的时间版）
                - 亲手做滑窗实验，亲眼看见**依赖长度超过窗宽时，
                  训练 90% 也是背题**（17 课的教训在序列上重演）
                - 理解出路的方向：**让"处理每个元素的规则"沿时间
                  共享**——权值共享的时间版，你在 26 课刚见过它的
                  空间版

                前置：26 课的三宗罪与权值共享、17 课的过拟合与
                数据划分、21 课的 one-hot。

                本课代码不重（一个微型 MLP 复用 25 课的训练配方），
                但它是第 6 部分的**开题报告**：33 课的隐状态、35 课
                的梯度消失，都是为了回答今天提出的问题。
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "🧓 概念 1/3 · 序列长什么样：三个特征": mo.md(
                """
                **序列**就是按顺序排成一列的数据：x₁, x₂, …, x_T。
                和向量、图像比，它有三个鲜明特征：

                **特征一 · 变长。** 一句话三个字，下一句三十个字；
                一段音频 0.5 秒，下一段 5 分钟。数据的"长度"本身
                就是变化的——而我们的网络在建造那一刻就把输入
                维度焊死了（784 就是 784）。

                **特征二 · 有序。** "猫追狗"和"狗追猫"用的字一模
                一样，意思完全相反。**顺序本身携带信息**——这是
                序列与"一堆数的集合"的本质区别。图像其实也有序
                （空间顺序），但序列的顺序是**因果**的：先来后到，
                不可颠倒。

                **特征三 · 上文决定下文。** 看到"床前明月"，你
                已经能猜到下一个字大概率是"光"。预测序列的下一个
                元素，**依据全在历史里**——这正是"语言模型"做的
                事，也是第 6 部分六课要攻下的山头。

                身边的序列：文本、语音波形、股价、DNA 碱基、
                用户点击流、代码、天气记录……你会发现它们比图像
                更常见。
                """
            ),
            "🧱 概念 2/3 · 全连接的三堵墙（26 课三宗罪重演）": mo.md(
                """
                拿全连接网络处理序列，会撞上三堵墙——结构上和
                26 课"看图三宗罪"一一对应：

                **墙一 · 位置绑定（平移罪的时间版）。** 全连接把
                "输入第 i 个格子"和"权重第 i 列"焊死在一起。把
                "猫追狗"换成"快猫追狗"，猫从第 1 格挪到第 2 格
                ——语义几乎没变，输入向量却**整个换了一批格子**。
                26 课演示 2 里"数字挪 2 像素、响应从 +2.99 翻到
                −2.49"的惨案，在序列上是家常便饭：**词挪一个位置，
                之前学的全部作废。**

                **墙二 · 参数挥霍。** 想在位置 3 学会"追"的后面
                常跟"狗"？这套知识存在位置 3 的权重里。同一个
                "追"出现在位置 7，网络得**从头再学一遍**——因为
                位置 7 是另一套权重。同一个规律，学 T 遍。

                **墙三 · 变长装不下。** 3 个词的句子拼成 15 维
                向量，4 个词的句子拼成 20 维——**维度都对不上**，
                一个固定输入维度的网络连"吃进去"都做不到。

                演示 2 会把三堵墙各撞一遍给你看。
                """
            ),
            "🪟 概念 3/3 · 滑窗：权宜之计与它的天花板": mo.md(
                """
                工程界最早的解法很朴素：**滑窗（sliding window）**。

                挑一个窗宽 W，只取"最近 W 个元素"拼成定长向量喂
                MLP，预测下一个。三堵墙瞬间都翻过去了：窗口内
                位置固定（墙一）、参数一套（墙二）、长度 W 写死
                （墙三）。 predicting 股价的经典 baseline、
                n-gram 语言模型，都是这个思路。

                但天花板也是结构性的：**窗口之外的事，一概不知。**

                ```
                依赖长度 L = 10：第 t 个字符由第 t−10 个字符决定
                窗宽   W =  4：只看得见 t−4 … t−1
                            → 决定性的字符在视野之外 → 只能瞎猜
                ```

                演示 3 会造一条"依赖长度精确可控"的合成序列，
                让你亲眼看到：**W < L 时，网络训练能上 90%——但
                那是在背题；换一批新序列立刻打回 33%（瞎猜线）。**

                出路在哪？与其每次都回头重看 W 个字符，不如让网络
                **边读边记**：把"读过的一切"压缩进一个不断更新的
                向量——**隐状态**；并且让"处理每个元素的那套权重"
                **沿时间共享**（26 课权值共享的时间版）。这正是
                33 课 RNN 的两个关键词，今天先把问题立起来。
                """
            ),
        }
    )
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 🎬 演示
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 1 · 人脑怎么猜"下一个"

        四条小序列，先自己猜猜问号是什么（答案在代码输出里）：

        你刚才做的事，就是**语言模型**在做的事：从上文找规则，
        推测下一个元素。三条序列各不相同——周期、递增、交替——
        但有一个共同点：**答案完全由上文决定**。

        下面的图把每条序列画在时间轴上；最下面一行是**打乱顺序后**
        的样子：字符还是那些字符，但"下一个是谁"再也无从谈起——
        这就是**特征二（有序）**的直观意义：打乱即毁约。
        """
    )
    return


@app.cell
def _(np, plt):
    # 演示 1 · 时间轴上的序列 vs 打乱后的"字符口袋"
    seqs_demo1 = [
        ("周期", list("abcbabcba")),
        ("递增×2", [1, 2, 4, 8, 16, 32, 64, 128, 256]),
        ("交替嵌套", list("1a2b3c4d5e")),
        ("打乱后（周期那条）", list("abcbabcba")),
    ]
    _shuffled = list(seqs_demo1[0][1])
    _rng32 = np.random.default_rng(7)
    _rng32.shuffle(_shuffled)
    seqs_demo1[3] = ("打乱后（还是那些字符）", _shuffled)

    print("猜猜下一个：")
    for _name, _s in seqs_demo1[:3]:
        print(f"  {list(map(str, _s))} … → ?   （{_name}）")
    print(f"\n打乱后：{seqs_demo1[3][1]}\n→ 上文还是那些字符，但规则没了——顺序本身携带信息")

    _fig, _axes = plt.subplots(4, 1, figsize=(11.5, 6.2))
    for _r, (_name, _s) in enumerate(seqs_demo1):
        _axes[_r].set_xlim(-0.8, len(_s) - 0.2)
        _axes[_r].set_ylim(-1.2, 1.2)
        for _k, _ch in enumerate(_s):
            _axes[_r].add_patch(plt.Circle((_k, 0), 0.34, facecolor="#dceefb",
                                            edgecolor="#2b6cb0", lw=1.4))
            _axes[_r].text(_k, 0, str(_ch), ha="center", va="center",
                           fontsize=11, fontweight="bold")
            if _k < len(_s) - 1:
                _axes[_r].annotate("", xy=(_k + 0.66, 0), xytext=(_k + 0.36, 0),
                                   arrowprops=dict(arrowstyle="->", color="gray", lw=1.2))
        _axes[_r].set_title(_name, fontsize=10.5, loc="left")
        _axes[_r].axis("off")
    _axes[3].set_title("打乱后（还是那些字符）：箭头方向不再代表任何规则",
                       fontsize=10.5, loc="left", color="tab:red")
    _fig.suptitle("序列 = 有序的元素流；打乱顺序，信息即毁灭", fontsize=12)
    _fig.tight_layout()
    _fig
    return (seqs_demo1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 2 · 三堵墙，各撞一遍

        词表 {猫, 追, 狗, 咬, 快}，one-hot 5 维/词（21 课的老朋友）。

        **撞墙一（位置绑定）**：`快猫追狗` vs `猫追狗快`——语义
        相近的两句，四个位置**全部换人**，拼接向量的欧氏距离直接
        拉到最大值 √8 ≈ 2.83。位置敏感到了"语义无关"的程度。

        **撞墙二（参数挥霍）**：句长 T 变化时，拼接 MLP 第一层的
        参数量线性膨胀——而且每个位置一套权重，"追"的搭配知识
        要在每个位置各学一遍。

        **撞墙三（变长装不下）**：3 词句是 15 维，4 词句是 20 维，
        给 15 维输入建的权重矩阵，遇到 20 维输入**当场报错**——
        这次是真的让 NumPy 报给你看。
        """
    )
    return


@app.cell
def _(np, plt):
    # 演示 2 · 三堵墙现场
    VOCAB32 = ["猫", "追", "狗", "咬", "快"]

    def one_hot_sent32(sent):
        """把词句编码成拼接 one-hot 向量（每词 5 维）。"""
        _idx = [VOCAB32.index(_w) for _w in sent]
        return np.eye(len(VOCAB32))[_idx].ravel()

    def bow_sent32(sent):
        """词袋（bag of words）：只数出现次数，无视顺序。"""
        _v = np.zeros(len(VOCAB32))
        for _w in sent:
            _v[VOCAB32.index(_w)] += 1
        return _v

    # ---- 墙一：位置绑定 ----
    sa32, sb32 = ["快", "猫", "追", "狗"], ["猫", "追", "狗", "快"]
    va32, vb32 = one_hot_sent32(sa32), one_hot_sent32(sb32)
    d_concat32 = float(np.linalg.norm(va32 - vb32))
    d_bow32 = float(np.linalg.norm(bow_sent32(sa32) - bow_sent32(sb32)))
    print("【墙一 · 位置绑定】")
    print(f"  「{''.join(sa32)}」vs「{''.join(sb32)}」：语义相近（都是猫追狗，快字强调）")
    print(f"  拼接向量距离 = {d_concat32:.3f}（4 词最大可能 = {np.sqrt(8):.3f}，直接拉满）")
    print(f"  词袋距离     = {d_bow32:.3f}（顺序信息被完全丢弃）")
    print("  → 拼接表示对位置敏感到语义无关；词袋表示对语义钝感到主宾不分")

    # ---- 墙二：参数账 ----
    print("\n【墙二 · 参数挥霍】拼接 MLP（隐层 64）第一层参数：")
    for _T in (3, 10, 30):
        _d = _T * len(VOCAB32)
        print(f"  句长 T={_T:2d} → 输入维 {_d:3d} → 第一层参数 {_d * 64 + 64:,}"
              f"（且每个位置一套权重，知识不共享）")

    # ---- 墙三：变长 ----
    print("\n【墙三 · 变长装不下】")
    W_fix32 = np.ones((15, 64))          # 为 3 词句（15 维）建好的权重
    x3_32, x4_32 = one_hot_sent32(["猫", "追", "狗"]), one_hot_sent32(["快", "猫", "追", "狗"])
    try:
        x4_32 @ W_fix32
    except ValueError as _e:
        print(f"  3 词句（{x3_32.shape[0]} 维）能吃；4 词句（{x4_32.shape[0]} 维）喂进来：")
        print(f"  ValueError: {_e}")

    # ---- 图：拼接 vs 词袋 ----
    _fig, _axes = plt.subplots(2, 2, figsize=(10.6, 5.4),
                               gridspec_kw={"width_ratios": [1, 1]})
    _axes[0, 0].imshow(va32.reshape(4, 5), cmap="Blues", vmin=0, vmax=1)
    _axes[0, 0].set_title(f"「{''.join(sa32)}」拼接向量\n（4 词 × 5 维 = 20 格）", fontsize=10)
    _axes[0, 1].imshow(vb32.reshape(4, 5), cmap="Blues", vmin=0, vmax=1)
    _axes[0, 1].set_title(f"「{''.join(sb32)}」拼接向量\n距离 {d_concat32:.2f} / 最大 {np.sqrt(8):.2f}", fontsize=10)
    for _ax in _axes[0]:
        _ax.set_xticks(range(5), VOCAB32, fontsize=8)
        _ax.set_yticks(range(4), ["位置1", "位置2", "位置3", "位置4"], fontsize=8)
    _axes[1, 0].bar(range(5), bow_sent32(sa32), color="#4c78a8")
    _axes[1, 0].set_title(f"「{''.join(sa32)}」词袋：两袋一模一样\n（距离 {d_bow32:.1f}，但主宾已互换）", fontsize=10)
    _axes[1, 1].bar(["实际拼接距离", "理论最大\n（每位置全换）"], [d_concat32, np.sqrt(8)],
                    color=["tab:red", "lightgray"], width=0.5)
    _axes[1, 1].set_title("拼接距离 vs 4 词最大距离\n（拉满 = 四个位置全换人）", fontsize=10)
    _axes[1, 1].set_ylim(0, 3.4)
    for _k, _v in enumerate([d_concat32, np.sqrt(8)]):
        _axes[1, 1].text(_k, _v + 0.08, f"{_v:.2f}", ha="center", fontsize=11)
    _fig.suptitle("三堵墙现场：位置敏感（上）vs 顺序失忆（下左）", fontsize=12)
    _fig.tight_layout()
    _fig
    return (VOCAB32, W_fix32, bow_sent32, d_concat32, one_hot_sent32, sa32, sb32)

@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 3 · 滑窗 MLP：依赖长度就是天花板

        造一条**依赖长度精确可控**的合成序列，规则只有一条：

        ```
        s[t] = (s[t−L] + 1) mod 3     （下一个字符 = L 步前的字符 + 1）
        ```

        字母表 {a,b,c}，每条序列的前 L 个字符随机，之后按规则生成。
        **关键设计**：语料 = 40 条**不同随机开头**的序列（32 条训练、
        8 条考试）——就像真实语料由许多独立片段组成，同一个开头
        后面可以接不同内容。

        用 25 课同款训练配方（MLP + Adam）预测下一个字符，换三个
        窗宽看成绩单（L = 8）：

        | 窗宽 | 训练 | 考试（新序列） | 诊断 |
        |---|---|---|---|
        | W=4 | ~48% | ~37% | 窗口离锚太远，只能背一小半 |
        | W=7 | **~90%** | **~33%** | **背题现场**：训练语料里每个窗口值出现次数少，背住了；新序列现出原形 |
        | W=8 | 100% | 100% | 窗口刚好够到锚字符，规则在手 |

        W=7 那一行是本课的**案发现场**：训练 90% 看起来像学会了，
        考试 33% 暴露了它只是在背 32 条序列的具体模式——**17 课
        的过拟合教训，在序列世界里原样重演**。而 W=8 只多看一步，
        就从瞎猜直接跳到满分：决定性字符进没进窗口，天壤之别。
        """
    )
    return


@app.cell
def _(np, plt):
    # 演示 3 · 滑窗 MLP × 依赖长度（合成语料：40 条独立序列）
    V32 = 3                                     # 字母表大小 {a,b,c}

    def make_seq32(L, n, seed):
        """s[t] = (s[t-L]+1) % 3，前 L 个随机；L = 依赖长度。"""
        _r = np.random.default_rng(seed)
        _s = list(_r.integers(0, V32, size=L))
        for _t in range(L, n):
            _s.append((_s[_t - L] + 1) % V32)
        return np.array(_s)

    def make_corpus32(L, n_seq, seq_len, seed0):
        """n_seq 条独立随机开头的序列（seed 各不相同）。"""
        return [make_seq32(L, seq_len, seed0 + _i) for _i in range(n_seq)]

    def windows_from32(seqs, W):
        """所有 (窗口 one-hot 拼接, 下一字符) 样本。"""
        _X, _y = [], []
        for _s in seqs:
            for _t in range(W, len(_s)):
                _X.append(np.eye(V32)[_s[_t - W:_t]].ravel())
                _y.append(_s[_t])
        return np.array(_X), np.array(_y)

    def train_mlp32(Xtr, ytr, Xte, yte, epochs=300, lr=0.02, seed=0):
        """25 课同款：tanh 隐层 + softmax + Adam（复用第 25 课配方）。"""
        _n, _d = Xtr.shape
        _r = np.random.default_rng(seed)
        _W1 = _r.normal(0, 0.3, (_d, 24)); _b1 = np.zeros(24)
        _W2 = _r.normal(0, 0.3, (24, V32)); _b2 = np.zeros(V32)
        _m = [np.zeros_like(_p) for _p in (_W1, _b1, _W2, _b2)]
        _v = list(_m)
        for _step in range(1, epochs + 1):
            _h = np.tanh(Xtr @ _W1 + _b1)
            _logits = _h @ _W2 + _b2
            _logits -= _logits.max(axis=1, keepdims=True)
            _p = np.exp(_logits); _p /= _p.sum(axis=1, keepdims=True)
            _dlog = _p.copy(); _dlog[np.arange(_n), ytr] -= 1; _dlog /= _n
            _dW2 = _h.T @ _dlog; _db2 = _dlog.sum(0)
            _dh = _dlog @ _W2.T * (1 - _h * _h)
            _dW1 = Xtr.T @ _dh; _db1 = _dh.sum(0)
            for _i, (_g, _par) in enumerate(zip((_dW1, _db1, _dW2, _db2),
                                                (_W1, _b1, _W2, _b2))):
                _m[_i] = 0.9 * _m[_i] + 0.1 * _g
                _v[_i] = 0.999 * _v[_i] + 0.001 * _g * _g
                _mh = _m[_i] / (1 - 0.9 ** _step)
                _vh = _v[_i] / (1 - 0.999 ** _step)
                _par -= lr * _mh / (np.sqrt(_vh) + 1e-8)
        _acc = lambda _X, _y: float((((np.tanh(_X @ _W1 + _b1)) @ _W2 + _b2)
                                     .argmax(1) == _y).mean())
        return _acc(Xtr, ytr), _acc(Xte, yte)

    L32 = 8                                                    # 依赖长度
    corpus_tr32 = make_corpus32(L32, 32, L32 + 16, seed0=0)    # 32 条训练序列
    corpus_te32 = make_corpus32(L32, 8, L32 + 16, seed0=1000)  # 8 条考试序列
    CH32 = ["a", "b", "c"]
    print(f"依赖长度 L = {L32}，规则 s[t] = (s[t-L]+1) % 3")
    print(f"训练语料第一条：{''.join(CH32[c] for c in corpus_tr32[0])}")
    print(f"考试语料第一条：{''.join(CH32[c] for c in corpus_te32[0])}（不同的随机开头）\n")

    demo3_results32 = []
    for _W in (4, 7, 8):
        _Xtr, _ytr = windows_from32(corpus_tr32, _W)
        _Xte, _yte = windows_from32(corpus_te32, _W)
        _a_tr, _a_te = train_mlp32(_Xtr, _ytr, _Xte, _yte)
        demo3_results32.append((_W, _a_tr, _a_te))
        print(f"W={_W} | 训练 {_a_tr:6.1%} | 考试 {_a_te:6.1%}")

    _fig, _ax = plt.subplots(figsize=(8.6, 4.4))
    _ws = [f"W={_r[0]}" for _r in demo3_results32]
    _tr = [_r[1] for _r in demo3_results32]
    _te = [_r[2] for _r in demo3_results32]
    _xb = np.arange(3)
    _ax.bar(_xb - 0.18, _tr, 0.34, label="训练语料", color="#4c78a8")
    _ax.bar(_xb + 0.18, _te, 0.34, label="考试语料（新序列）", color="#f58518")
    _ax.axhline(1 / 3, color="tab:red", ls="--", lw=1.4)
    _ax.text(2.42, 1 / 3 + 0.015, "瞎猜线 33%", color="tab:red", fontsize=9, ha="right")
    _ax.axhline(1.0, color="gray", ls=":", lw=1.2)
    for _k in range(3):
        _ax.text(_k - 0.18, _tr[_k] + 0.015, f"{_tr[_k]:.0%}", ha="center", fontsize=10)
        _ax.text(_k + 0.18, _te[_k] + 0.015, f"{_te[_k]:.0%}", ha="center", fontsize=10)
    _ax.set_xticks(_xb, _ws)
    _ax.set_ylim(0, 1.12)
    _ax.set_ylabel("预测下一字符的准确率")
    _ax.set_title(f"滑窗 MLP 成绩单（依赖长度 L = {L32}）：W=7 训练 90% 是背题，考试打回瞎猜线")
    _ax.legend(loc="center right", fontsize=9)
    _ax.grid(True, axis="y", alpha=0.3)
    _fig.tight_layout()
    _fig
    return (CH32, L32, corpus_te32, corpus_tr32, demo3_results32, make_corpus32,
            make_seq32, train_mlp32, windows_from32)

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
        ### 交互实验 1 · 词序交换台

        从下拉框选两句（词表 {猫, 追, 狗, 咬, 快}），看两种表示
        给出的"距离"：

        - **拼接 one-hot**：保序，但位置一换就天翻地覆
        - **词袋**：无视顺序，主宾互换它毫无察觉

        试试把 A 选 `猫追狗`、B 选 `狗追猫`（主宾互换，语义反转，
        两种距离各有多离谱）；再试试 A `快猫追狗`、B `猫追狗快`
        （四个位置全换）；最后选一对**长度不同**的句子，看第三堵墙。
        """
    )
    return


@app.cell
def _(mo):
    # 交互控制：选两句参加比较
    sents_lab32 = ["猫追狗", "狗追猫", "猫咬狗", "狗咬猫", "追猫狗", "快猫追狗", "猫追狗快"]
    sentA_dd32 = mo.ui.dropdown(options=sents_lab32, value="猫追狗", label="📝 句子 A")
    sentB_dd32 = mo.ui.dropdown(options=sents_lab32, value="狗追猫", label="📝 句子 B")
    mo.hstack([sentA_dd32, sentB_dd32])
    return (sentA_dd32, sentB_dd32, sents_lab32)


@app.cell
def _(VOCAB32, bow_sent32, mo, np, one_hot_sent32, plt, sentA_dd32, sentB_dd32):
    # ===== 交互实验 1 · 词序交换台（响应单元）=====
    _a32, _b32 = sentA_dd32.value, sentB_dd32.value
    _sem32 = {"猫追狗": "主谓宾：猫在追狗",
              "狗追猫": "主宾对调：狗在追猫（语义反转）",
              "猫咬狗": "主谓宾：动作换成咬",
              "狗咬猫": "主宾对调 + 换动作",
              "追猫狗": "乱序：不成句",
              "快猫追狗": "加了状语「快」，核心仍是猫追狗",
              "猫追狗快": "同样的字打乱，不成句"}
    _va, _vb = one_hot_sent32(list(_a32)), one_hot_sent32(list(_b32))
    _same_len = len(_a32) == len(_b32)

    _fig, _axes = plt.subplots(1, 3, figsize=(11.6, 3.8),
                               gridspec_kw={"width_ratios": [1.1, 1.1, 1.0]})
    _axes[0].imshow(_va.reshape(len(_a32), 5), cmap="Blues", vmin=0, vmax=1)
    _axes[0].set_title(f"A「{_a32}」：{_sem32[_a32]}", fontsize=9.5)
    _axes[0].set_xticks(range(5), VOCAB32, fontsize=8)
    _axes[0].set_yticks(range(len(_a32)), [f"位置{_i+1}" for _i in range(len(_a32))], fontsize=8)
    _axes[1].imshow(_vb.reshape(len(_b32), 5), cmap="Oranges", vmin=0, vmax=1)
    _axes[1].set_title(f"B「{_b32}」：{_sem32[_b32]}", fontsize=9.5)
    _axes[1].set_xticks(range(5), VOCAB32, fontsize=8)
    _axes[1].set_yticks(range(len(_b32)), [f"位置{_i+1}" for _i in range(len(_b32))], fontsize=8)
    _axes[2].axis("off")

    if _same_len:
        _d_c = float(np.linalg.norm(_va - _vb))
        _d_max = float(np.sqrt(2 * len(_a32)))
        _d_b = float(np.linalg.norm(bow_sent32(list(_a32)) - bow_sent32(list(_b32))))
        _axes[2].bar([0, 1], [_d_c, _d_b], color=["tab:red", "tab:blue"], width=0.5)
        _axes[2].axhline(_d_max, color="gray", ls="--", lw=1.2)
        _axes[2].text(1.32, _d_max, f"拼接最大 {_d_max:.2f}", fontsize=8.5,
                      color="gray", ha="right", va="bottom")
        _axes[2].set_xticks([0, 1], ["拼接距离\n（保序）", "词袋距离\n（无序）"], fontsize=9)
        for _k, _v in enumerate([_d_c, _d_b]):
            _axes[2].text(_k, _v + 0.05, f"{_v:.2f}", ha="center", fontsize=11)
        _axes[2].set_ylim(0, max(_d_max, _d_c) * 1.3)
        _axes[2].set_title("两种「距离」的成绩", fontsize=10)
        if _d_c > 0 and _d_b == 0:
            _verdict = (f"**拼接距离 {_d_c:.2f}**（vs 词袋 **0**）——词袋眼里两句**一模一样**，"
                        f"但 A 是「{_sem32[_a32]}」、B 是「{_sem32[_b32]}」。"
                        "保序的表示对位置敏感，无序的表示对语义失明：**各有各的瞎法**，"
                        "这就是第 6 部分要造新表示的原因")
        elif _d_c >= _d_max - 1e-9:
            _verdict = (f"**拼接距离 {_d_c:.2f} = 最大值拉满**——每个位置都换了人。"
                        f"可 A「{_a32}」和 B「{_b32}」说的是相近的事。"
                        "位置绑定让「语义距离」和「向量距离」彻底脱钩（墙一）")
        else:
            _verdict = (f"拼接距离 **{_d_c:.2f}**、词袋距离 **{_d_b:.2f}**——"
                        f"A「{_sem32[_a32]}」、B「{_sem32[_b32]}」。"
                        "两个数字都给不出「语义差多远」的答案")
    else:
        _axes[2].text(0.5, 0.55, f"墙三现场\n\nA 是 {len(_a32)} 词 = {len(_a32)*5} 维\n"
                                  f"B 是 {len(_b32)} 词 = {len(_b32)*5} 维\n\n"
                                  "维度都对不上，\n连「喂给同一个网络」都做不到",
                      ha="center", va="center", fontsize=12, color="tab:red",
                      transform=_axes[2].transAxes)
        _verdict = (f"**变长装不下（墙三）**：「{_a32}」（{len(_a32)*5} 维）和"
                    f"「{_b32}」（{len(_b32)*5} 维）无法共用一个输入层——"
                    "滑窗法就是靠「只取固定 W 个」绕过这堵墙的")
    _fig.suptitle("词序交换台：拼接保序但位置敏感，词袋稳定但顺序失明", fontsize=11.5)
    _fig.tight_layout()

    mo.vstack([
        _fig,
        mo.callout(mo.md(_verdict), kind="info"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 交互实验 2 · 滑窗实验室：窗口能不能看见"决定性字符"

        两条滑块：**依赖长度 L**（决定下一字符的字符在多远之前）、
        **窗宽 W**（网络能回看多远）。每拖一次都现场训练一个小 MLP
        （300 轮，不到 0.1 秒），并画出**视野图**：

        - 上：时间轴。灰格 = 窗口覆盖的位置，⭐ = 决定性字符
          （t−L），❓ = 要预测的 t。星星落在灰格里 = 看得见
        - 下：训练/考试准确率，红线是 33% 瞎猜线

        建议：固定 L=8，把 W 从 4 拖到 8，看考试准确率什么时候
        跳变；再试试 L=12、W=11——训练分很高，考试呢？
        """
    )
    return


@app.cell
def _(mo):
    # 交互控制：依赖长度 L × 窗宽 W
    depL_slider32 = mo.ui.slider(3, 12, value=8, step=1, debounce=True,
                                 label="📏 依赖长度 L（决定性字符在 L 步之前）")
    winW_slider32 = mo.ui.slider(2, 14, value=5, step=1, debounce=True,
                                 label="🪟 窗宽 W（网络能回看的步数）")
    mo.hstack([depL_slider32, winW_slider32])
    return (depL_slider32, winW_slider32)


@app.cell
def _(depL_slider32, make_corpus32, mo, plt, train_mlp32, winW_slider32, windows_from32):
    # ===== 交互实验 2 · 滑窗实验室（响应单元）=====
    _L, _W = depL_slider32.value, winW_slider32.value
    _ctr = make_corpus32(_L, 32, _L + 16, seed0=0)
    _cte = make_corpus32(_L, 8, _L + 16, seed0=1000)
    _Xtr, _ytr = windows_from32(_ctr, _W)
    _Xte, _yte = windows_from32(_cte, _W)
    _a_tr, _a_te = train_mlp32(_Xtr, _ytr, _Xte, _yte)

    _fig, _axes = plt.subplots(2, 1, figsize=(11.2, 5.0),
                               gridspec_kw={"height_ratios": [1.0, 1.3]})
    # ---- 视野图 ----
    _n_ax = max(_W, _L) + 3
    _axes[0].set_xlim(-0.7, _n_ax)
    _axes[0].set_ylim(-0.6, 1.35)
    for _k in range(_n_ax):
        _pos_t = _k - (_n_ax - 1)              # 时间轴：最右是 t=0（要预测的位置）
        _is_win = -_W <= _pos_t <= -1
        _fc = "#cfe3f7" if _is_win else "#f0f0f0"
        _axes[0].add_patch(plt.Rectangle((_k - 0.42, 0.18), 0.84, 0.62,
                                         facecolor=_fc, edgecolor="gray", lw=0.8))
        _axes[0].text(_k, 0.49, f"{_pos_t}", ha="center", va="center", fontsize=9)
        if _pos_t == -_L:
            _axes[0].text(_k, 1.12, "★", ha="center", fontsize=15, color="tab:green")
        if _pos_t == 0:
            _axes[0].text(_k, 1.12, "?", ha="center", fontsize=14,
                          color="tab:red", fontweight="bold")
    _axes[0].text(-0.55, -0.38, f"灰色 = 窗口（回看 {_W} 步）　"
                                f"★ = 决定性字符（{_L} 步前）　? = 要预测的位置",
                  fontsize=9, color="gray")
    _axes[0].set_title("视野图：★ 在不在灰格里？", fontsize=11)
    _axes[0].axis("off")
    # ---- 成绩条 ----
    _axes[1].bar([0, 1], [_a_tr, _a_te], 0.45,
                 color=["#4c78a8", "#f58518"])
    _axes[1].axhline(1 / 3, color="tab:red", ls="--", lw=1.4)
    _axes[1].text(1.46, 1 / 3 + 0.015, "瞎猜线 33%", color="tab:red",
                  fontsize=9, ha="right")
    for _k, _v in enumerate([_a_tr, _a_te]):
        _axes[1].text(_k, _v + 0.015, f"{_v:.0%}", ha="center", fontsize=11)
    _axes[1].set_xticks([0, 1], ["训练语料", "考试语料（新序列）"])
    _axes[1].set_ylim(0, 1.1)
    _axes[1].set_ylabel("准确率")
    _axes[1].set_title(f"L = {_L}，W = {_W}")
    _axes[1].grid(True, axis="y", alpha=0.3)
    _fig.tight_layout()

    if _a_te > 0.95:
        _verdict = (f"**规则在手**（考试 {_a_te:.0%}）：W = {_W} ≥ L = {_L}，"
                    "决定性字符在窗口里，MLP 学到了真规则「回看 L 步、加 1 取模」，"
                    "新序列上照样满分——泛化，来自看见真正的依据")
    elif _a_te < 0.45:
        if _a_tr > 0.7:
            _verdict = (f"**背题现场**：训练 {_a_tr:.0%} 看着像学会了，考试 {_a_te:.0%} "
                        f"打回瞎猜线——W = {_W} < L = {_L}，★ 在窗口外，网络背的是"
                        "32 条序列的具体模式（17 课：训练分高不代表学会了）")
        else:
            _verdict = (f"**瞎猜**（考试 {_a_te:.0%}）：W = {_W} < L = {_L}，"
                        "决定性字符在视野之外——窗口里的一切与答案统计独立，"
                        "换任何模型来都只能 33%")
    else:
        _verdict = (f"**半开眼界**（考试 {_a_te:.0%}）：窗口边缘擦到了一些线索"
                    f"（W = {_W} vs L = {_L}），把 W 拖到 ≥ {_L} 看看跳变")
    mo.vstack([
        _fig,
        mo.callout(mo.md(_verdict), kind="info"),
    ])
    return

@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✏️ 动手练习（每课必做，本课 3 题）": mo.md(
                """
                **练习 1 · 上文长度侦探**：三条序列，每条最少要看
                **几个上文元素**才能唯一确定下一个？（这就是序列的
                "记忆深度"——33 课隐状态要背的正是它）

                **练习 2 · 亲手实现 one_hot**：补全 2 行，把 token
                列表编码成 one-hot 矩阵，与 np.eye 对账（21 课旧知识，
                37 课语言模型还要用它）。

                **练习 3 · 窗口够不够**：依赖长度 L=6 的序列，窗宽
                W=4 的滑窗 MLP 学得会吗？最少要多宽？

                规则不变：先把 `None` 换成你的答案再运行；写完再看参考答案。
                """
            )
        }
    )
    return


@app.cell
def _(mo, np):
    # ===== 练习 1 · 上文长度侦探 =====
    seqs_ex1 = {
        "A": list("010101010101"),
        "B": [1, 2, 1, 3, 1, 4, 1, 5, 1, 6, 1, 7],
        "C": list("abcababcababcab"),
    }
    for _k, _s in seqs_ex1.items():
        print(f"序列 {_k}: {_s[:11]} …")

    ans_ex1 = [None, None, None]              # ← ①②③ A/B/C 各需要几个上文（整数）

    def min_context_ex1(seq, k_max=6):
        """暴力验证：长度 k 的后缀是否唯一确定下一个元素。"""
        for _k in range(1, k_max + 1):
            _seen = {}
            _ok = True
            for _t in range(_k, len(seq)):
                _key = tuple(map(str, seq[_t - _k:_t]))
                if _key in _seen and _seen[_key] != seq[_t]:
                    _ok = False
                    break
                _seen[_key] = seq[_t]
            if _ok:
                return _k
        return None

    if all(_a is not None for _a in ans_ex1):
        _truth = [min_context_ex1(seqs_ex1[k]) for k in ("A", "B", "C")]
        if list(ans_ex1) == _truth:
            _msg = (f"✅ 全对！A/B/C 的记忆深度 = {_truth}。"
                    "A 看 1 个就够（0→1、1→0）；B 看到 1 不够（后面跟过 2/3/4/…），"
                    "得再看前一个（x 1 → x+1）；C 的「ab」后跟过 c 也跟过 a，"
                    "要 3 个后缀才唯一——**不同序列需要的记忆深度天差地别，"
                    "这就是为什么 33 课要让网络自己学着「记多少、记什么」**")
        else:
            _msg = (f"❌ 参考：{_truth}。提示：对每条序列问「只看最后 1 个够吗？"
                    "最后 2 个呢？」——把短后缀相同、下一位不同的位置找出来")
    else:
        _msg = "⏳ 三个深度都填上（A 只看最后 1 个元素就能定下一个吗？）"
    mo.md(_msg)
    return (ans_ex1, seqs_ex1)


@app.cell
def _(mo, np):
    # ===== 练习 2 · 亲手实现 one_hot =====
    def one_hot_ex2(vocab, tokens):
        """把 token 列表编码成 (n_tokens, len(vocab)) 的 one-hot 矩阵。"""
        _idx = None                              # ← ① 补：每个 token 在词表里的下标
        return None                              # ← ② 补：用 np.eye 造矩阵并取行

    onehot_ready_ex2 = None                      # ← 补完后，把这行换成 one_hot_ex2

    _vocab = ["猫", "追", "狗"]
    if onehot_ready_ex2 is not None:
        try:
            _got = onehot_ready_ex2(_vocab, ["狗", "猫", "猫", "追"])
            _ref = np.eye(3)[[2, 0, 0, 1]]
            if _got is not None and _got.shape == (4, 3) and np.allclose(_got, _ref):
                _msg = ("✅ 对账通过！每行一个词、每列一个词表位，行和为 1——"
                        "21 课 softmax 前的这套编码，37 课语言模型会原样复用"
                        "（那边叫「词表 → one-hot」）")
            else:
                _msg = "❌ 形状或数值不对：期望 (4, 3)，与 np.eye 参考矩阵逐元素一致"
        except (TypeError, ValueError, IndexError) as _e:
            _msg = f"❌ 运行出错：{_e}——两行分别用到 list.index 和 np.eye(len(vocab))"
    else:
        _msg = "⏳ 补两行：① [vocab.index(t) for t in tokens]；② np.eye(len(vocab))[_idx]"
    mo.md(_msg)
    return (one_hot_ex2, onehot_ready_ex2)


@app.cell
def _(mo):
    # ===== 练习 3 · 窗口够不够 =====
    L_ex3 = 6                                    # 依赖长度：s[t] 由 s[t-6] 决定
    W_ex3 = 4                                    # 滑窗 MLP 的窗宽

    can_learn_ex3 = None                         # ← ① W=4 学得会吗？（True / False）
    W_min_ex3 = None                             # ← ② 最少要多宽？（整数）

    if can_learn_ex3 is not None and W_min_ex3 is not None:
        _truth_can, _truth_w = (W_ex3 >= L_ex3), L_ex3
        if can_learn_ex3 == _truth_can and W_min_ex3 == _truth_w:
            _msg = (f"✅ 正确：W = {W_ex3} < L = {L_ex3}，决定性字符在窗口外，学不会；"
                    f"最少 W = {L_ex3}（窗口 [t−{L_ex3}, t−1] 恰好含住 s[t−{L_ex3}]）。"
                    "注意是「≥ L」而不是「≥ L+1」：窗口包含 t−L 那个位置本身")
        else:
            _msg = (f"❌ 参考：{'能' if _truth_can else '不能'}、最少 W = {_truth_w}。"
                    "把交互实验 2 的 L 拖到 6、W 拖到 6 亲眼验证")
    else:
        _msg = "⏳ 两空都填上（提示：视野图里 ★ 要落进灰格子）"
    mo.md(_msg)
    return (L_ex3, W_ex3, can_learn_ex3, W_min_ex3)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 1 参考答案（先自己写过再看！）": mo.md(
                """
                **A = 1，B = 2，C = 3。**

                - **A（0 1 0 1 …）**：看到 0 就知道下一个是 1，
                  看到 1 就知道是 0——当前值就是一个双向映射，
                  **1 个上文够**
                - **B（1 2 1 3 1 4 …）**：看到 1 不够（1 后面跟过
                  2、3、4、5……），但「x 1」后必是 x+1、
                  「1 x」后必是 1——**2 个上文够**
                - **C（a b c a b 循环）**：「a b」后跟过 c（周期
                  中段）也跟过 a（周期接缝），必须看到 3 个
                  后缀（如 b c a → b）才唯一——**3 个上文**

                「最少要看几个上文」有个学名：序列的**记忆深度**。
                现实语言里这个深度没有上限——一句话开头的人名，
                  隔三十个字还要被代词引用。固定窗口注定捉襟见肘，
                33 课的隐状态就是来回答这个问题的。
                """
            ),
            "✅ 练习 2 参考答案（先自己写过再看！）": mo.md(
                """
                两行分别是：

                ```python
                _idx = [vocab.index(t) for t in tokens]
                return np.eye(len(vocab))[_idx]
                ```

                - 列表推导式（03 课）查出每个 token 的词表下标
                - `np.eye(n)` 造 n×n 单位阵（09 课），按 `_idx`
                  **花式索引**（05 课）取出对应行——行 i 就是
                  词表第 i 个词的 one-hot

                三个旧零件拼出一个新编码器。顺便一提：
                `vocab.index` 是 O(词表) 的查找，词表大时换成
                字典 `{词: 下标}`（37 课就是这么做的）。
                """
            ),
            "✅ 练习 3 参考答案（先自己写过再看！）": mo.md(
                """
                **学不会（False），最少 W = 6。**

                窗口是 [t−W, t−1] 共 W 个位置，含住锚位置 t−L
                的条件是 t−W ≤ t−L，即 **W ≥ L**。W=4、L=6 时
                锚字符在视野外——窗口内的字符与答案**统计独立**
                （初始字符独立随机），任何模型都只能 33%。

                演示 3 的 W=7、L=8 是同一个局面：训练 90% 全靠
                背题。想「学得会」和「看起来学得会」是两回事——
                **17 课的教训，序列版。**
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
                1. **序列三特征**：变长（3 字句和 30 字句共存）、
                   有序（打乱即毁约）、上文决定下文（预测的依据
                   全在历史里）
                2. **三堵墙**：位置绑定（词挪一位，向量面目全非——
                   26 课平移罪的时间版）、参数挥霍（同一规律每个
                   位置重学一遍）、变长装不下（15 维 vs 20 维，
                   连喂都喂不进去）
                3. **拼接 vs 词袋**：拼接保序但对位置过敏（快猫追狗
                   vs 猫追狗快 距离拉满）；词袋稳定但顺序失明
                   （猫追狗 vs 狗追猫 距离为 0）——各有各的瞎法
                4. **滑窗法**：只看最近 W 个字符拼定长向量，三堵墙
                   都绕过了，但天花板是结构性的：**窗口之外，
                   一概不知**
                5. **W < L = 瞎**：决定性字符在窗外时，窗口信息与
                   答案统计独立——训练分再高（W=7 时 90%）也是
                   背题，新序列打回 33% 瞎猜线
                6. **出路**：让网络边读边记（隐状态），让处理规则
                   **沿时间共享**（权值共享的时间版）——33 课
                   把这两个想法装进一个公式
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
                ## 🎓 下一站

                今天的结论可以压成一句话：**固定窗口注定失明，
                网络需要的是"记忆"。**

                但"记忆"具体长什么样？总不能开辟一块无限大的
                硬盘把读过的字符全存下来——那是字典查找，不是
                学习。真正可行的方案朴素得惊人：

                - 用一个固定长度的向量 **h** 当"工作记忆"
                - 每读一个新字符，就把 h **更新一次**：
                  新记忆 = f(旧记忆, 新字符)
                - 处理第 1 个字符和第 100 个字符，用的是
                  **同一组权重**（26 课权值共享的时间版）

                序列再长，参数不变；该记什么、忘什么，全部
                交给 f 去学。
                """
            ),
            mo.callout(
                mo.md(
                    """
                    **下一课预告** — `33_RNN单元与展开`：

                    把上面三条变成一个只有一行核心公式的零件：
                    **hₜ = tanh(W·xₜ + U·hₜ₋₁ + b)**。两条输入
                    （当前字符 + 上次的记忆）、一套共享权重、
                    沿时间展开成一串相同的单元——你会亲手用
                    NumPy 实现它，喂一句话，看隐状态如何随着
                    每个字符**流动、沉淀**；再算一笔参数账：
                    为什么序列长 10 还是长 1000，参数都一样多。

                    演示里的随机权重 RNN 会露出一个可爱的性质：
                    没训练过的它也有"惯性"——h 变化得很慢，
                    像一条惯性滑行的小船。34 课再给这条小船
                    装上方向盘（BPTT）。

                    🌊 序列的海，跳吧！
                    """
                ),
                kind="success",
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
