import marimo

__generated_with = "0.25.2"
app = marimo.App(app_title="37 · 字符级语言模型", width="medium")


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
        # 37 · 字符级语言模型：让网络学会"说话"

        **从零到 Transformer · 手写深度学习** ｜ 第 6 部分 · 循环神经网络（6/6 · 收官）
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📖 本课目标（先读我）": mo.md(
                """
                **第 6 部分的终局之战：把 32-36 课攒的全部家当
                ——one-hot、RNN/LSTM、BPTT、梯度裁剪——装进一台
                真正的机器：字符级语言模型。**

                任务一句话讲完：读一段中文文本，每读一个字符
                就预测**下一个字符**的概率分布。别小看这个
                任务——ChatGPT 的本质就是它的放大版（预测
                下一个 token）。

                语料是课程自写的科普长文《一堂看懂的深度学习》
                （约 1.2 万字符）——让模型学的和学的对象都是
                深度学习，闭环了。

                学完你将能：

                - 亲手写完整的语言模型训练循环（采样窗口、
                  截断 BPTT、梯度裁剪，全套工业件）
                - 解释**温度**怎么把同一个模型调出三种性格
                  （保守 / 平衡 / 天马行空）
                - 读懂语言模型的成绩单：**困惑度**（perplexity）
                - 亲眼见证 **RNN vs LSTM 终局对决**：小语料上
                  双双过拟合（17 课的老朋友），早停点定胜负
                  ——LSTM 困惑度 106 vs RNN 125
                - 用训练好的模型**逐字符生成**文本，看它学到
                  了哪些"文法"

                前置：34 课 BPTT、35 课梯度裁剪、36 课 LSTM、
                21 课 softmax、17 课过拟合与早停。

                预训练好的模型权重已放在 `data/charlm37.npz`
                （训练约 1 分钟），交互生成秒出结果；训练循环
                的完整代码在演示 2 现场跑给你看。
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📖 概念 1/3 · 语言模型：下一个字符的概率": mo.md(
                """
                语言模型（language model）回答一个问题：

                ```
                P(下一个字符 | 上文)
                ```

                给它"梯度下"三个字，它输出一整张概率表：
                "降" 0.62、"度" 0.21、"是" 0.05……字符级
                模型的词表就是语料里出现过的所有字符（这门
                课的语料约 600 个，含标点和换行）。

                训练方法你已经全会了：

                - 输入：每个时间步喂一个字符的 one-hot（21 课）
                - 结构：RNN 或 LSTM（33/36 课）逐字符更新记忆
                - 输出：每步一个 V 维 logits，softmax 成分布
                - 损失：**每个位置都算交叉熵**——目标就是
                  下一个字符（21 课）
                - 反向：BPTT（34 课）+ 梯度裁剪（35 课）

                一个工程细节：**词表压缩**。语料里一半的字符
                只出现一两次（生僻字、英文术语的字母）——
                模型永远学不会它们。处理办法：只保留出现
                ≥3 次的字符进词表，其余统一映射成一个
                ◇（UNK，未知符），生成时屏蔽。这是 NLP 的
                祖传手艺。
                """
            ),
            "🌡️ 概念 2/3 · 采样与温度：给模型调性格": mo.md(
                """
                训练完的模型每步给一个分布，怎么选下一个字符？

                - **贪心**：永远选概率最高的——很快掉进复读机
                  循环（"的学习的学习的学习"）
                - **采样**：按概率抽签——文本多样，但可能
                  抽到离谱的低概率字符

                工业界标准方案是**带温度的采样**。温度 τ 是
                softmax 的除数：

                ```
                P(char) ∝ exp(logit / τ)
                ```

                - **τ → 0**：分布越来越尖，逼近贪心——保守、
                  循规蹈矩、容易复读
                - **τ = 1**：模型的真实分布
                - **τ → 大**：分布越来越平，逼近均匀抽签——
                  天马行空、错字连篇

                直觉：温度是采样时的**胆量旋钮**。小语料上的
                模型本来就没学透，τ=0.7 附近通常最耐看。
                """
            ),
            "📊 概念 3/3 · 工程三件套：困惑度、窗口采样、早停": mo.md(
                """
                **困惑度（perplexity）**——语言模型的标准成绩单：

                ```
                ppl = exp(交叉熵损失)
                ```

                直觉：模型每次平均在**多少个字符里犹豫**。
                ppl=1 完美笃定；ppl=600（词表大小）= 均匀瞎猜。
                这份语料上：早停点的 LSTM ppl=106，RNN 125——
                离瞎猜很远，离人话也远，这正是小语料小模型的
                真实水位。

                **窗口采样（截断 BPTT 的取数据方式）**——长文本
                切成随机窗口：每次随机挑一个起点，取连续 64
                个字符做输入、错一位的 64 个做目标。梯度只在
                窗口内回传（截断 BPTT）——无限长的文档因此
                可以分批训练。

                **早停**——小语料 + 大词表，过拟合必来（17 课
                的规律在语言模型上原样重演）：训练损失一路降，
                验证损失降到某个点开始回升。**在验证最低点
                保存参数**，这是本课两个模型共同的操作——
                演示 3 的曲线会清楚标出这两个早停点。
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
        ### 演示 1 · 语料巡礼：模型要学的是什么样的文本

        《一堂看懂的深度学习》，课程自写，约 1.2 万字符：
        概念篇（网络、梯度、卷积、循环、注意力）+ 动手篇
        （五步训练流程）+ 问答篇（十个新手问题）+ 小词典
        （三十个术语，格式高度统一——字符模型的最爱）。

        看三个统计：**字符频率排行**（"的"断层第一，逗号
        句号高频——标点结构是可学的强信号）、**换行分布**
        （段落结构）、**最常出现的双字组合**（"梯度""网络"
        "学习""损失"——这些高频词是模型最先学会的"词感"）。
        """
    )
    return


@app.cell
def _(np, plt):
    # 演示 1 · 语料加载与巡礼
    def load_corpus37():
        """加载语料（与笔记本同目录约定：notebooks/../data 或 data/）。"""
        import os
        _cands = []
        try:
            _nb = __file__
            _cands += [os.path.join(os.path.dirname(_nb), "..", "data")]
        except NameError:
            pass
        _cands += ["../data", "data", "/home/z/my-project/download/deep-learning-course/data"]
        for _d in _cands:
            _p = os.path.join(_d, "charlm37.npz")
            if os.path.exists(_p):
                return _p
        raise RuntimeError("找不到 data/charlm37.npz——请看本课开头的说明")

    import collections

    def corpus_text37():
        """从缓存里取语料原文（按词表反查）。"""
        _z = np.load(load_corpus37(), allow_pickle=True)
        _vocab = list(_z["vocab"])
        _ids = _z["ids"]
        return "".join(_vocab[i] for i in _ids), _vocab, _ids

    text37, vocab37, ids37 = corpus_text37()
    _z37 = np.load(load_corpus37(), allow_pickle=True)
    n_tr37 = int(_z37["n_tr"])
    print(f"语料长度：{len(text37)} 字符｜词表（含◇UNK）：{len(vocab37)}｜"
          f"训练/验证切分：{n_tr37} / {len(text37) - n_tr37}")

    _cnt = collections.Counter(text37)
    print("\n字符频率 Top 12：")
    _top = _cnt.most_common(12)
    _fig, _ax = plt.subplots(figsize=(9.6, 3.6))
    _ax.bar([f"{c}" for c, _ in _top], [n for _, n in _top],
            color=["tab:red" if c in "，。\n" else "tab:blue" for c, _ in _top])
    for _k, (_, _n) in enumerate(_top):
        _ax.text(_k, _n + 60, str(_n), ha="center", fontsize=9)
    _ax.set_title("语料字符频率 Top 12（红色 = 标点/换行——结构信号）", fontsize=11)
    _ax.grid(True, axis="y", alpha=0.3)
    _fig.tight_layout()
    _fig

    # 高频双字组合
    _bigrams = collections.Counter(text37[i:i+2] for i in range(len(text37) - 1))
    print("\n双字组合 Top 10（模型最先学会的「词感」）：")
    print("  " + "  ".join(f"{w}({n})" for w, n in _bigrams.most_common(10)))
    return (ids37, load_corpus37, n_tr37, text37, vocab37)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 2 · 训练循环：全套工业件现场跑

        完整的语言模型训练循环，每一件都是真家伙：

        - **gather 优化**：one-hot × 矩阵 = 直接取列
          （`W[:, x]`）——比真乘 one-hot 快百倍，笔记本里
          秒级可训
        - **窗口采样**：每步随机起点切 48 字符窗口
        - **截断 BPTT**：梯度只在窗口内回传（34 课骨架）
        - **梯度裁剪**：max_norm=5（35 课的三行）
        - **Adam**：24 课配方

        现场跑 **150 步小规模训练**（d_h=64，约 10 秒），看
        训练损失从 ln(601)≈6.4 一路跳水。正式对决（演示 3）
        用预训练的完整版——同样的代码，只是跑得更久、
        加了验证评估和早停。
        """
    )
    return


@app.cell
def _(ids37, np, plt, vocab37):
    # 演示 2 · 现场训练循环（小规模：d_h=64, 150 步）
    def train_lm_live37(ids, V, d_h=64, steps=150, T=48, batch=16, lr=0.01, seed=0):
        """现场版训练循环（RNN）。返回 loss 曲线。"""
        _r = np.random.default_rng(seed)
        _Wx = _r.normal(0, 0.08, (d_h, V)).astype(np.float32)
        _Wh = _r.normal(0, 1 / np.sqrt(d_h), (d_h, d_h)).astype(np.float32)
        _bh = np.zeros(d_h, np.float32)
        _Wy = _r.normal(0, 0.08, (V, d_h)).astype(np.float32)
        _by = np.zeros(V, np.float32)
        _ms = [np.zeros_like(_p) for _p in (_Wx, _Wh, _bh, _Wy, _by)]
        _vs = list(_ms)
        _losses = []
        for _step in range(1, steps + 1):
            # ---- 窗口采样：随机起点，输入/目标错一位 ----
            _starts = _r.integers(0, len(ids) - T - 1, batch)
            _X = np.stack([ids[_s:_s + T] for _s in _starts])
            _Y = np.stack([ids[_s + 1:_s + T + 1] for _s in _starts])
            # ---- 前向（gather：Wx[:, x] 直接取列）----
            _h = np.zeros((batch, d_h), np.float32)
            _hs = np.zeros((batch, T, d_h), np.float32)
            _logits = np.zeros((batch, T, V), np.float32)
            for _t in range(T):
                _h = np.tanh(_Wx[:, _X[:, _t]].T + _h @ _Wh.T + _bh)
                _hs[:, _t] = _h
                _logits[:, _t] = _h @ _Wy.T + _by
            _Pr = _logits.astype(np.float64)
            _Pr = np.exp(_Pr - _Pr.max(-1, keepdims=True))
            _Pr /= _Pr.sum(-1, keepdims=True)
            _n, _T2 = _X.shape
            _loss = -np.log(_Pr[np.arange(_n)[:, None], np.arange(_T2)[None, :], _Y]
                            + 1e-12).mean()
            # ---- 反向（BPTT + 裁剪 + Adam）----
            _dlog = _Pr.astype(np.float32)
            _dlog[np.arange(_n)[:, None], np.arange(_T2)[None, :], _Y] -= 1
            _dlog /= (_n * _T2)
            _dWy = _dlog.reshape(-1, V).T @ _hs.reshape(-1, d_h)
            _dby = _dlog.sum((0, 1))
            _dh_next = np.zeros((batch, d_h), np.float32)
            _dWx = np.zeros_like(_Wx); _dWh = np.zeros_like(_Wh); _dbh = np.zeros_like(_bh)
            for _t in reversed(range(T)):
                _dh = _dlog[:, _t] @ _Wy + _dh_next
                _dh = _dh * (1 - _hs[:, _t] ** 2)
                np.add.at(_dWx, (slice(None), _X[:, _t]), _dh.T)
                _hp = _hs[:, _t - 1] if _t > 0 else np.zeros((batch, d_h), np.float32)
                _dWh += _dh.T @ _hp
                _dbh += _dh.sum(0)
                _dh_next = _dh @ _Wh.T
            _gl = [_dWx, _dWh, _dbh, _dWy, _dby]
            _tot = np.sqrt(sum((_g.astype(np.float64) ** 2).sum() for _g in _gl))
            if _tot > 5.0:                                     # 梯度裁剪
                _sc = 5.0 / (_tot + 1e-12)
                _gl = [(_g * _sc).astype(np.float32) for _g in _gl]
            for _i, (_g, _p) in enumerate(zip(_gl, (_Wx, _Wh, _bh, _Wy, _by))):
                _ms[_i] = 0.9 * _ms[_i] + 0.1 * _g
                _vs[_i] = 0.999 * _vs[_i] + 0.001 * _g * _g
                _p -= (lr * (_ms[_i] / (1 - 0.9 ** _step)) /
                       (np.sqrt(_vs[_i] / (1 - 0.999 ** _step)) + 1e-8)).astype(np.float32)
            _losses.append(_loss)
        return np.array(_losses)

    losses_live37 = train_lm_live37(ids37, len(vocab37))
    print(f"现场训练 150 步：loss {losses_live37[0]:.2f}（瞎猜线 ln{len(vocab37)}≈{np.log(len(vocab37)):.1f}）"
          f" → {losses_live37[-1]:.2f}")
    print("正式版（预训练 600 步 + 早停）在演示 3 加载——同样的循环，多了验证评估")

    _fig, _ax = plt.subplots(figsize=(8.6, 3.8))
    _ax.plot(losses_live37, color="tab:blue", lw=1.8)
    _ax.axhline(np.log(len(vocab37)), color="gray", ls="--", lw=1.1)
    _ax.text(2, np.log(len(vocab37)) - 0.28, "瞎猜线", fontsize=9, color="gray")
    _ax.set_xlabel("训练步数")
    _ax.set_ylabel("交叉熵损失（当前批）")
    _ax.set_title("现场版：语言模型训练循环跑起来（RNN，d_h=64）", fontsize=11)
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()
    _fig
    return (losses_live37, train_lm_live37)

@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 3 · 终局对决：RNN vs LSTM（预训练加载）

        `data/charlm37.npz` 里存着两个模型各自 **600 步训练的
        完整记录**（d_h=96，每 25 步评估一次验证损失，全程
        保存早停点参数）。两条曲线并排看：

        - **形状**：双双先降后升——训练损失一路俯冲，验证
          损失在 125 步附近触底反弹。17 课的过拟合规律，
          在语言模型上原样重演（1.2 万字符的语料养不起
          十几万参数的模型，背书是必然）
        - **深度**：LSTM 的验证谷更深（**4.66 vs 4.83**），
          且下降段更持久——长记忆带来的真实优势
        - **早停点**：两个模型都在验证最低点保留参数（竖线
          标注）——后续生成用的就是这份

        困惑度换算：LSTM **106** vs RNN **125**（瞎猜是 601，
        均匀线 ln601≈6.4）。同配置同预算，35 课的理论优势
        在真实语料上兑现。
        """
    )
    return


@app.cell
def _(load_corpus37, np, plt, text37, vocab37):
    # 演示 3 · 预训练对决：加载 + 曲线 + 困惑度
    _z = np.load(load_corpus37(), allow_pickle=True)
    va_rnn37, va_lstm37 = _z["va_rnn"], _z["va_lstm"]
    tr_rnn37, tr_lstm37 = _z["tr_rnn"], _z["tr_lstm"]
    bs_rnn37, bs_lstm37 = int(_z["best_step_rnn"]), int(_z["best_step_lstm"])
    bv_rnn37, bv_lstm37 = float(_z["best_val_rnn"]), float(_z["best_val_lstm"])
    _steps = np.arange(25, 25 * (len(va_rnn37) + 1), 25)

    print(f"词表 {len(vocab37)}（瞎猜线 ln V ≈ {np.log(len(vocab37)):.2f}，ppl 601）")
    print(f"RNN ：验证最低 {bv_rnn37:.3f} @ step {bs_rnn37} → 困惑度 {np.exp(bv_rnn37):.0f}")
    print(f"LSTM：验证最低 {bv_lstm37:.3f} @ step {bs_lstm37} → 困惑度 {np.exp(bv_lstm37):.0f}")
    print(f"训练末端：RNN {tr_rnn37[-1]:.2f} / LSTM {tr_lstm37[-1]:.2f}"
          "（背书背得炉火纯青——但考试要看验证线）")

    _fig, _axes = plt.subplots(1, 2, figsize=(11.6, 4.3), sharey=True)
    for _ax, (_tr, _va, _bs, _nm, _c) in zip(_axes, [
            (tr_rnn37, va_rnn37, bs_rnn37, "RNN", "tab:red"),
            (tr_lstm37, va_lstm37, bs_lstm37, "LSTM", "tab:green")]):
        _ax.plot(_steps, _tr, color=_c, lw=1.5, alpha=0.55, label="训练损失")
        _ax.plot(_steps, _va, color=_c, lw=2.2, label="验证损失")
        _ax.axvline(_bs, color="gray", ls="--", lw=1.3)
        _ax.scatter([_bs], [_va[(_bs // 25) - 1]], s=70, color=_c, zorder=5)
        _ax.annotate(f"早停点 @ {_bs}\nppl {np.exp(_va[(_bs // 25) - 1]):.0f}",
                     xy=(_bs, _va[(_bs // 25) - 1]), xytext=(_bs + 40, _va[0] * 0.86),
                     fontsize=9, arrowprops=dict(arrowstyle="->", color="gray"))
        _ax.axhline(np.log(len(vocab37)), color="lightgray", ls=":", lw=1.1)
        _ax.set_xlabel("训练步数")
        _ax.set_title(_nm, fontsize=12)
        _ax.legend(fontsize=9)
        _ax.grid(True, alpha=0.3)
    _axes[0].set_ylabel("交叉熵损失")
    _fig.suptitle("终局对决：LSTM 的验证谷更深（4.66 vs 4.83），两边都过拟合、都靠早停", fontsize=11.5)
    _fig.tight_layout()
    _fig
    return (bs_lstm37, bs_rnn37, tr_lstm37, tr_rnn37, va_lstm37, va_rnn37)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 4 · 开麦：两个模型的生成对决 + 温度三档

        加载早停点参数，从"深度学习"四个字开始逐字符采样
        （生成时屏蔽◇UNK）。三个观察角度：

        - **RNN（τ=0.7）**：能拼出局部词汇（"模型""数据"
          "梯度"），但句子东一榔头西一棒——三五个字换一次
          话题，长句组织不起来
        - **LSTM（τ=0.7）**：注意**段落结构**——它学会了
          "换行 + 空行"的节奏、冒号引出的词条格式（小词典
          的功劳）。词级通顺未必强多少，但**篇章级的骨架**
          学到了：这是长记忆的直接证据
        - **温度扫描（LSTM）**：τ=0.2 循规蹈矩爱复读，
          τ=0.7 最耐看，τ=1.2 开始出现没见过的搭配——
          同一个模型，三种性格
        """
    )
    return


@app.cell
def _(load_corpus37, np, vocab37):
    # 演示 4 · 加载权重 + 采样函数 + 三档温度生成
    _z = np.load(load_corpus37(), allow_pickle=True)
    V37 = len(vocab37)
    vocab_list37 = list(_z["vocab"])
    ch2id37 = {c: i for i, c in enumerate(vocab_list37)}
    _unk_id = ch2id37["◇"]
    D_H37 = 96

    def load_params37(which):
        _P = {}
        for _k, _v in zip(list(_z[f"P_{which}_keys"]), list(_z[f"P_{which}"])):
            _P[_k] = _v
        return _P

    P_rnn37 = load_params37("rnn")
    P_lstm37 = load_params37("lstm")

    def sigmoid37(z):
        return 1.0 / (1.0 + np.exp(-z))

    def sample37(P, model, seed_text, length, temp, rng_seed=0):
        """从 seed_text 起逐字符采样（屏蔽 UNK）。"""
        _rng = np.random.default_rng(rng_seed)
        _out = list(seed_text)
        _h = np.zeros(D_H37, np.float32); _c = np.zeros(D_H37, np.float32)
        def _step(x):
            nonlocal _h, _c
            if model == "rnn":
                _h = np.tanh(P["Wx"][:, x] + P["Wh"] @ _h + P["bh"])
            else:
                _i = sigmoid37(P["Wxi"][:, x] + P["Whi"] @ _h + P["bi"])
                _f = sigmoid37(P["Wxf"][:, x] + P["Whf"] @ _h + P["bf"])
                _o = sigmoid37(P["Wxo"][:, x] + P["Who"] @ _h + P["bo"])
                _g = np.tanh(P["Wxg"][:, x] + P["Whg"] @ _h + P["bg"])
                _c = _f * _c + _i * _g
                _h = _o * np.tanh(_c)
        for _ch in seed_text:
            if _ch in ch2id37:
                _step(ch2id37[_ch])
        for _ in range(length):
            _z2 = (P["Wy"] @ _h + P["by"]).astype(np.float64) / temp
            _p = np.exp(_z2 - _z2.max())
            _p /= _p.sum()
            _p[_unk_id] = 0.0                       # 屏蔽 UNK
            _p /= _p.sum()
            _ch = vocab_list37[int(_rng.choice(V37, p=_p))]
            _out.append(_ch)
            _step(ch2id37[_ch])
        return "".join(_out)

    print("═" * 46, " RNN · τ=0.7 ", "═" * 46)
    print(sample37(P_rnn37, "rnn", "深度学习", 90, 0.7))
    print()
    print("═" * 46, " LSTM · τ=0.7 ", "═" * 46)
    print(sample37(P_lstm37, "lstm", "深度学习", 90, 0.7))
    print()
    for _t in (0.2, 1.2):
        print("═" * 44, f" LSTM · τ={_t} ", "═" * 44)
        print(sample37(P_lstm37, "lstm", "深度学习", 90, _t))
        print()
    return (P_lstm37, P_rnn37, V37, ch2id37, sample37, vocab_list37)


@app.cell
def _(P_lstm37, P_rnn37, ch2id37, np, vocab_list37):
    # 演示 4 附 · 量化对比：给上文，看下一字符分布的前二名
    def greedy_top2(P, model, ctx):
        """返回 ctx 后下一字符分布的前二名（贪心近似）。"""
        _h = np.zeros(96, np.float32); _c = np.zeros(96, np.float32)
        for _ch in ctx:
            _x = ch2id37.get(_ch, None)
            if _x is None:
                continue
            if model == "rnn":
                _h = np.tanh(P["Wx"][:, _x] + P["Wh"] @ _h + P["bh"])
            else:
                _i = 1 / (1 + np.exp(-(P["Wxi"][:, _x] + P["Whi"] @ _h + P["bi"])))
                _f = 1 / (1 + np.exp(-(P["Wxf"][:, _x] + P["Whf"] @ _h + P["bf"])))
                _o = 1 / (1 + np.exp(-(P["Wxo"][:, _x] + P["Who"] @ _h + P["bo"])))
                _g = np.tanh(P["Wxg"][:, _x] + P["Whg"] @ _h + P["bg"])
                _c = _f * _c + _i * _g
                _h = _o * np.tanh(_c)
        _z = (P["Wy"] @ _h + P["by"]).astype(float)
        _p = np.exp(_z - _z.max()); _p /= _p.sum()
        _o2 = _p.argsort()[::-1]
        return f"{vocab_list37[_o2[0]]} {_p[_o2[0]]:.2f} ｜ {vocab_list37[_o2[1]]} {_p[_o2[1]]:.2f}"

    print("考题：给上文，看下一字符分布的前两名（谁更像「学会了」？）")
    for _ctx in ("梯度下", "损失函", "神经网", "训练损失", "换句话说"):
        print(f"  「{_ctx}」→  RNN: {greedy_top2(P_rnn37, 'rnn', _ctx):<26}"
              f"  LSTM: {greedy_top2(P_lstm37, 'lstm', _ctx)}")
    print("\n两个模型都学到了「梯度下→降」「损失函→数」这类高频搭配——")
    print("差异藏在长上文的把握和篇章结构里（演示 4 的生成样本）")
    return (greedy_top2,)

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
        ### 交互实验 · 开麦直播间

        你的直播间你做主：选**模型**（RNN / LSTM）、给**开头**
        （会被过滤到词表内）、调**温度**和**长度**——现场
        逐字符生成（用早停点的权重，秒出）。

        玩法建议：

        - 同一个开头、τ=0.7，RNN 和 LSTM 各生成一次，比
          谁的**段落结构**更清楚
        - LSTM + τ=0.2：看复读机模式（高频词循环）
        - τ=1.2：看小概率字符怎么把句子带偏
        - 开头试试"问题"、"梯度"、"卷积"——语料里高密度
          出现的词，生成质量最好
        """
    )
    return


@app.cell
def _(mo):
    # 交互控制：模型 + 开头 + 温度 + 长度
    model_dd37 = mo.ui.dropdown(options=["LSTM（早停点）", "RNN（早停点）"],
                                value="LSTM（早停点）", label="🤖 模型")
    seed_text37 = mo.ui.text(value="深度学习", label="✍️ 开头（1~8 个字）")
    temp_slider37 = mo.ui.slider(0.1, 1.5, value=0.7, step=0.05, debounce=True,
                                 label="🌡️ 温度 τ")
    len_slider37 = mo.ui.slider(30, 240, value=120, step=10, debounce=True,
                                label="📏 生成长度（字符）")
    mo.vstack([
        mo.hstack([model_dd37, temp_slider37]),
        mo.hstack([seed_text37, len_slider37]),
    ])
    return (len_slider37, model_dd37, seed_text37, temp_slider37)


@app.cell
def _(P_lstm37, P_rnn37, V37, ch2id37, len_slider37, mo, model_dd37, sample37,
      seed_text37, temp_slider37):
    # ===== 交互实验 · 开麦直播间（响应单元）=====
    _which = "lstm" if "LSTM" in model_dd37.value else "rnn"
    _P = P_lstm37 if _which == "lstm" else P_rnn37
    _raw = (seed_text37.value or "").strip()[:8]
    _seed = "".join(c for c in _raw if c in ch2id37)
    if not _seed:
        _seed = "深度学习"
    _temp, _n = temp_slider37.value, len_slider37.value
    _text = sample37(_P, _which, _seed, _n, _temp, rng_seed=0)

    _has_para = ("\n\n" in _text) or ("\n" in _text)
    _vocab_hits = sum(1 for w in ("梯度", "网络", "模型", "数据", "训练", "损失", "学习")
                      if w in _text)
    _verdict = (f"**{_which.upper()} · τ={_temp:.2f} · {_n} 字**："
                f"高频术语命中 {_vocab_hits}/7，"
                f"{'有' if _has_para else '无'}段落结构。\n\n")
    if _temp < 0.35:
        _verdict += ("低温区：分布被压尖，模型只敢说最有把握的话——"
                     "看看是不是在复读高频词？")
    elif _temp <= 0.95:
        _verdict += ("舒适区：分布保真。LSTM 注意看换行节奏和冒号词条格式"
                     "（小词典的印记）；RNN 更多是词级拼贴，篇章感弱一档")
    else:
        _verdict += ("高温区：小概率字符频繁出场，句子被带向没见过的搭配——"
                     "分布越平，模型的「个性」越少、随机性越多")
    mo.vstack([
        mo.plain(f"「{_seed}」+ {_n} 字："),
        mo.callout(mo.md(f"```\n{_text}\n```"), kind="none"),
        mo.callout(mo.md(_verdict), kind="info"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✏️ 动手练习（本课 3 题）": mo.md(
                """
                **练习 1 · 手算温度**：logits = [2, 1, 0]，
                笔算 τ=1 和 τ=0.5 时 softmax 的**第一类概率**
                （保留两位）——算完你会记住"温度减半，尖峰
                翻倍"的手感。

                **练习 2 · 补全采样**：带温度采样函数的核心
                三行（logits 除温度、softmax、按概率抽签）。

                **练习 3 · 困惑度换算**：交叉熵 3.0 对应多少
                困惑度？它的直觉解释是什么？
                """
            )
        }
    )
    return


@app.cell
def _(mo, np):
    # ===== 练习 1 · 手算温度 =====
    print("logits = [2, 1, 0]（三类）")
    print("  τ = 1.0：P(第一类) = e² / (e²+e¹+e⁰) = ?")
    print("  τ = 0.5：P(第一类) = e⁴ / (e⁴+e²+e⁰) = ?")
    print("（e²≈7.39，e⁴≈54.6，供笔算用）")

    p_t1_ex1_37 = None                             # ← ① τ=1 时第一类概率（两位小数）
    p_t05_ex1_37 = None                            # ← ② τ=0.5 时第一类概率（两位小数）

    if p_t1_ex1_37 is not None and p_t05_ex1_37 is not None:
        _z1 = np.array([2.0, 1.0, 0.0])
        _t1 = float(np.exp(_z1[0]) / np.exp(_z1).sum())
        _z2 = _z1 / 0.5
        _t2 = float(np.exp(_z2[0]) / np.exp(_z2).sum())
        if abs(p_t1_ex1_37 - _t1) < 0.02 and abs(p_t05_ex1_37 - _t2) < 0.02:
            _msg = (f"✅ 正确：{_t1:.2f} → {_t2:.2f}。温度减半，领先者的概率从 "
                    f"{_t1:.0%} 涨到 {_t2:.0%}——低温把差距放大（除以 τ 等于把 "
                    "logits 的差距按 1/τ 倍拉大），分布变尖；高温反之")
        else:
            _msg = (f"❌ 参考：{_t1:.2f} / {_t2:.2f}。先把 logits 除以 τ 再 softmax；"
                    "分母是三类 e 值之和")
    else:
        _msg = "⏳ 两空都填上（τ=0.5 时 logits 变 [4, 2, 0]）"
    mo.md(_msg)
    return (p_t05_ex1_37, p_t1_ex1_37)


@app.cell
def _(mo, np):
    # ===== 练习 2 · 补全采样 =====
    def sample_step_ex2_37(logits, temp, rng):
        """带温度采一步：返回按概率抽中的类别下标。"""
        _z = None                                     # ← ① 补：logits 除以温度
        _p = None                                     # ← ② 补：softmax 概率（数值稳定版）
        return None                                   # ← ③ 补：rng.choice 按概率抽签

    sample_ready_ex2_37 = None                       # ← 补完后，把这行换成 sample_step_ex2_37

    if sample_ready_ex2_37 is not None:
        try:
            _rng = np.random.default_rng(0)
            _z0 = np.array([2.0, 1.0, 0.0])
            _cnt = [0, 0, 0]
            for _ in range(6000):
                _cnt[int(sample_ready_ex2_37(_z0, 1.0, _rng))] += 1
            _ref = np.exp(_z0) / np.exp(_z0).sum()
            _emp = np.array(_cnt) / 6000
            if np.allclose(_emp, _ref, atol=0.03):
                _msg = (f"✅ 6000 次抽签频率 {np.round(_emp, 3)} ≈ 理论 {np.round(_ref, 3)}——"
                        "数值稳定版 softmax 记得先减最大值（21 课），"
                        "rng.choice 的 p 参数按概率抽签")
            else:
                _msg = f"❌ 抽签频率 {np.round(_emp, 3)} 偏离理论 {np.round(_ref, 3)}"
        except (TypeError, ValueError, ZeroDivisionError) as _e:
            _msg = (f"❌ 运行出错：{_e}——三行：_z = logits / temp；"
                    "_p 减 max 后 exp 归一；int(rng.choice(len(_p), p=_p))")
    else:
        _msg = "⏳ 补三行：除温度 → softmax → choice"
    mo.md(_msg)
    return (sample_ready_ex2_37, sample_step_ex2_37)


@app.cell
def _(mo, np):
    # ===== 练习 3 · 困惑度换算 =====
    print("某语言模型在验证集上的交叉熵损失 = 3.0")

    ppl_ex3_37 = None                               # ← ① 困惑度（整数）

    if ppl_ex3_37 is not None:
        _t = int(round(np.exp(3.0)))
        if abs(ppl_ex3_37 - _t) <= 1:
            _msg = (f"✅ 正确：ppl = e³ ≈ {_t}。直觉：模型每个位置平均在 "
                    f"约 {_t} 个字符里犹豫（词表 600 的瞎猜是 600）。"
                    "本课的 LSTM 早停点 ppl≈106、RNN≈125——离瞎猜很远，"
                    "离人话也远，这就是 1.2 万字符小语料的真实水位。"
                    "大模型把这个数字压到个位数靠的是亿万倍的数据")
        else:
            _msg = f"❌ 参考：e³ ≈ {_t}（ppl = exp(交叉熵)）"
    else:
        _msg = "⏳ 填一个整数（ppl = e^loss）"
    mo.md(_msg)
    return (ppl_ex3_37,)

@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 1 参考答案（先自己写过再看！）": mo.md(
                """
                **τ=1：0.67；τ=0.5：0.87。**

                ```
                τ=1  ：e²/(e²+e¹+e⁰) = 7.39/11.10 ≈ 0.67
                τ=0.5：e⁴/(e⁴+e²+e⁰) = 54.6/63.0  ≈ 0.87
                ```

                温度除在 logits 上，等于把差距**按 1/τ 倍
                放大**：τ=0.5 时 logits 差距翻倍，赢家通吃；
                τ=2 时差距减半，分布趋平。采样时"胆量"的
                数学含义就在这一除。
                """
            ),
            "✅ 练习 2 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                _z = logits / temp
                _e = np.exp(_z - _z.max())          # 数值稳定：先减最大值
                _p = _e / _e.sum()
                return int(rng.choice(len(_p), p=_p))
                ```

                两个细节：① **减最大值**不是可选美化——
                logits 上百时直接 exp 会溢出成 inf（21 课的
                老坑）；② `rng.choice` 的 `p` 参数要求概率
                和恰为 1（归一化后浮点误差通常在容差内）。

                工业界还常在采样前加 **top-k / top-p 截断**
                （只在前 k 个或累积概率 p 内抽）——把最离谱
                的长尾直接掐掉，是温度之外的另一套胆量
                旋钮。
                """
            ),
            "✅ 练习 3 参考答案（先自己写过再看！）": mo.md(
                """
                **ppl = e³ ≈ 20。**

                困惑度的直觉：模型每个位置平均在约 20 个
                字符里犹豫。几个锚点帮你校准手感：

                | ppl | 意味着 |
                |---|---|
                | 1 | 完美笃定（过拟合背书的典型） |
                | 20 | 高频搭配基本掌握（本练习） |
                | 106 | 本课 LSTM 的早停点 |
                | 601 | 词表级瞎猜（本课语料） |

                大语言模型在大语料上能压到个位数——靠的
                不是结构奇迹（还是那套 softmax + 反向传播），
                是**数据量乘以算力**。公式没变过，规模变了。
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
                1. **语言模型**：P(下一字符 | 上文)——每读一个
                   字符输出一张 V 维概率表；训练 = 每个位置
                   算交叉熵 + BPTT（34 课全套直接上岗）
                2. **词表压缩**：只保留出现 ≥3 次的字符，其余
                   →◇（UNK），生成时屏蔽——生僻字是学不会
                   的噪声，NLP 祖传手艺
                3. **温度**：logits ÷ τ 再 softmax——τ 小分布
                   尖（保守复读）、τ 大分布平（天马行空）；
                   0.7 附近最耐看
                4. **困惑度**：ppl = exp(交叉熵)——"每次平均
                   在几个字符里犹豫"；本课 LSTM 106 vs
                   RNN 125 vs 瞎猜 601
                5. **小语料的宿命**：1.2 万字符养十几万参数，
                   过拟合必来——训练损失俯冲、验证损失触底
                   反弹；**早停点参数**是对决的公平裁判
                6. **生成质量**：RNN 学到词级搭配（"梯度下→降"），
                   LSTM 额外学到**篇章结构**（换行节奏、
                   词条格式）——长记忆在生成上的直接证据
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
                ## 🎓 第 6 部分收官：循环神经网络（6/6）

                六课连打，你现在拥有一台**亲手实现、亲手训练、
                亲耳听过它说话**的序列模型。清点行囊：

                | 课 | 里程碑：你现在能做到什么 |
                |---|---|
                | **32** 序列数据 | 三堵墙、滑窗天花板（W<L 必瞎）、背题 vs 泛化 |
                | **33** RNN 单元 | hₜ = tanh(W·xₜ + U·hₜ₋₁ + b)、展开图、参数与 T 无关 |
                | **34** BPTT | 倒序单循环、数值对账 10⁻⁸、梯度时间分解 |
                | **35** 消失与爆炸 | 连乘公式、clip 急救（治炸不治聋）、gap=20 审判 |
                | **36** LSTM | 加法传送带、三道门、b_f=2 先验、出院实验 91% |
                | **37** 语言模型 | 完整训练循环、温度采样、困惑度、RNN vs LSTM 终局 |

                用一句话回顾第 6 部分：序列的规律是**沿时间
                传递的**——RNN 把"同一套权重扫过时间"焊进
                结构（32-33），梯度沿原路回传（34），连乘的
                命运拆成两课（35-36），最后全部组装成一台
                会说话的机器（37）。
                """
            ),
            mo.callout(
                mo.md(
                    """
                    但第 6 部分留了一个越来越明显的遗憾——
                    **记忆要一步步传递**。35 课的连乘病根只是
                    被 LSTM 缓解（高速路要靠门控维护），没有
                    根治；更深的伤口是**效率**：RNN/LSTM 必须
                    读完第 1 个字才能读第 2 个，梯度必须走完
                    整条时间轴才能回传。

                    如果每个位置能**直接看到所有位置**呢？
                    前一句的主语、上一段的人名、隔着一百个
                    字的伏笔——一步直达，不用中转。这个
                    想法叫**注意力**，它是第 7 部分的全部
                    内容，也是 Transformer 大厦唯一的地基。

                    你已经见过它的预告三次了（31、36、37 课
                    的下一站）。下一部分，亲手造它。
                    """
                ),
                kind="success",
            ),
            mo.md(
                """
                ---

                **下一课预告** — `38_注意力直觉`（第 7 部分 · Transformer）：

                从一个具体的不便开始：给一句话里的"它"字
                找指代对象——"小明把书放进箱子，**它**很重"，
                "它"是书还是箱子？你扫一眼前后文就有了答案，
                但 LSTM 要把"很重"的信息一步步传到"它"，
                中间隔着全句的噪声。

                注意力的答案朴素得惊人：让"它"对句子里**每个
                词**算一个相关度分数，按分数加权汇总——
                **想看哪看哪，看多少自己定**。你会亲手实现
                这个打分（原来就是点积，11 课的老朋友），
                并发现它一行循环都没有、天然并行。

                🔭 第 7 部分启航：去注意力的大海！
                """
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
