import marimo

__generated_with = "0.25.2"
app = marimo.App(app_title="43 · 编码器实战", width="medium")


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
        # 43 · 编码器实战：亲手造一台 Transformer

        **从零到 Transformer · 手写深度学习** ｜ 第 7 部分 · Transformer（6/8 · 主战役）
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📖 本课目标（先读我）": mo.md(
                """
                **第 7 部分的主战役：把 38-42 课的全部零件
                组装成一台完整的 Transformer 编码器，用
                NumPy 从零训练它。**

                零件清单齐得让人感动：嵌入 + 正弦位置编码
                （41）、多头注意力（40）、Pre-LN 块（42）、
                GELU 前馈网络（42）。组装图：

                ```
                词 id → 嵌入 + PE → [Pre-LN 块] × 2 → CLS 位置 → 分类头
                ```

                任务是**找触发词**：每句话 12 个字，其中
                藏着 1 个情绪词（4 个正面 / 4 个负面候选），
                其余全是填充词——模型判断句子的极性。
                触发词位置随机，**找它必须靠注意力**。

                学完你将能：

                - 写出**完整编码器的前向**（~50 行）和
                  **全链路反向传播**（~80 行）：从交叉熵
                  穿过分类头、CLS、FFN、残差、LN、注意力、
                  一直回到词嵌入——22-23 课手艺的总成
                - 用**数值梯度校验**证明自己的反向传播
                  正确（全课程最值得带走的方法论之一）
                - 训练到**测试集 100%**（约 20 秒），看
                  损失从 5.8 跌到 0
                - 解剖训好的注意力：**触发词是块 1 的
                  信息枢纽（被看总量 3.3 倍）**，CLS 的
                  目光统计上偏向触发词——还会撞见真实
                  大模型里著名的**两跳路由**现象

                前置：40-42 课全部、25 课训练循环、
                21 课 softmax、24 课 Adam。

                这是全课程代码量最大的一课——但每一行
                你都在前面的课里见过。
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "🗺️ 概念 1/3 · 编码器全景：数据流一日游": mo.md(
                """
                一个句子的完整旅程（T=12，d=32，2 块，4 头）：

                ```
                词 id (B×12)
                  → 嵌入查表 E[id] (B×12×32)      每个词一张 32 维身份证
                  → + 正弦位置编码 (B×12×32)       41 课：位置信息注入
                  → 块1：注意力 + FFN（Pre-LN）     每个位置看完全句、消化感想
                  → 块2：再来一遍                   更高层的组合
                  → 取 CLS 位置的向量 (B×32)       信息漏斗的出口
                  → 线性分类头 (B×2)               正面/负面
                ```

                **CLS 是个特殊记号**（[CLS]），训练时它
                自己没有语义——但因为它要在注意力里看
                完所有词、又只被自己看（其他词各忙各的），
                2 个块下来它自然聚合成整句的摘要。BERT
                用同一招做句子分类；GPT 没有它（生成
                不需要摘要，44 课见）。

                参数账（本课模型，合计 26,018）：
                嵌入 17×32=544 + 2 个块各 12,704 +
                分类头 66——**块占 98%**，这就是
                "Transformer 的参数住在块里"。
                """
            ),
            "🎯 概念 2/3 · 任务设计：为什么是「找触发词」": mo.md(
                """
                词表 17 个：[CLS] + 4 正面（棒/赞/喜/优）
                + 4 负面（糟/差/烦/劣）+ 8 填充（的/了/
                这/那/和/在/是/也）。

                每句话：1 个情绪词（随机选、随机位置）
                + 10 个填充词。标签 = 情绪词的极性。

                这个任务对注意力的要求是**本质的**：触发词
                位置每句都变，模型不能靠"看固定位置"
                作弊，必须**用内容找到它**——查询（各位置
                的表示）要与触发词的键对上眼。如果注意力
                学不会，任务就无解；如果学会了，注意力
                矩阵里应该留下证据（演示 5 去找）。

                数据是**无限生成**的：每个 batch 现场
                随机造句。这带来一个漂亮的性质——
                **过拟合无从谈起**（17 课的病根是没有
                新鲜数据，这里每个 batch 都是新的），
                26,018 个参数喝了 19,200 个句子（300 步
                × 64 句）都不带重样的。
                """
            ),
            "🔬 概念 3/3 · 全链路反向：22-23 课的总成": mo.md(
                """
                反向传播要穿过的完整链条（每一环都学过）：

                ```
                交叉熵 → softmax 反向（21）
                → 分类头线性反向（15）
                → 取 CLS 位置（梯度只流向位置 0）
                → 残差加法：梯度分两路（42）
                → LN 反向：投影公式（42，已对数值校验）
                → 注意力反向：dA/dV/dS 三件套（40）
                → FFN 反向：GELU 导数 + 两层线性（20）
                → 嵌入查表反向：np.add.at 散射累加（25）
                ```

                **怎么知道自己没写错？数值梯度校验**：
                对每个参数抖动 ±ε，用 (f(x+ε)−f(x−ε))/2ε
                算"实验室级"梯度，与手写的解析梯度逐个
                对账——**误差到 10⁻⁹ 量级才算过关**。
                这套方法不依赖任何框架，是深度学习工程
                的看家本领（PyTorch 的 gradcheck 同理）。
                演示 3 现场跑：764 个参数位置全部对上。
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
        ### 演示 1 · 任务巡礼：模型要学的是什么样的句子

        看三个随机样本：触发词的位置每句都变，填充词
        均匀随机——**唯一稳定的信号是"那个情绪词在哪
                里、是哪边的"**。再统计一下 300 个句子的
        触发词位置分布（大致均匀，没有任何位置特权）。
        """
    )
    return


@app.cell
def _(np):
    # 演示 1 · 任务定义与样本
    POS43 = ["棒", "赞", "喜", "优"]
    NEG43 = ["糟", "差", "烦", "劣"]
    FILL43 = ["的", "了", "这", "那", "和", "在", "是", "也"]
    vocab43 = ["[CLS]"] + POS43 + NEG43 + FILL43

    def make_data43(n, T, rng):
        """句子 = [CLS] + 1 个情绪词（随机位置）+ 填充词。标签 0=正 1=负。"""
        _ids = np.zeros((n, T), int)
        _y = np.zeros(n, int)
        _pos_ids, _neg_ids = np.arange(1, 5), np.arange(5, 9)
        _fill_ids = np.arange(9, 17)
        for _b in range(n):
            _trig = rng.choice(_pos_ids) if rng.random() < 0.5 else rng.choice(_neg_ids)
            _y[_b] = 0 if _trig < 5 else 1
            _rest = list(rng.choice(_fill_ids, T - 2))
            _rest.insert(rng.integers(0, T - 1), _trig)
            _ids[_b] = [0] + _rest
        return _ids, _y

    _rng = np.random.default_rng(3)
    for _i in range(3):
        _ids, _y = make_data43(1, 12, _rng)
        print(" ".join(vocab43[_j] for _j in _ids[0]), "→",
              "正面" if _y[0] == 0 else "负面")
    _ids, _ = make_data43(300, 12, _rng)
    _trig_pos = [int(np.argmax((_r >= 1) & (_r <= 8))) for _r in _ids]
    import collections
    _cnt = collections.Counter(_trig_pos)
    print("\n触发词位置分布（300 句）：", dict(sorted(_cnt.items())))
    print("→ 位置大致均匀：想靠'看固定位置'作弊是没门的")
    return (make_data43, vocab43)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 2 · 完整实现：前向 + 全链路反向

        `TinyEncoder43`：嵌入（可学习）+ 正弦 PE（固定）
        + 2 个 Pre-LN 块（4 头注意力 + GELU FFN）+ CLS
        分类头。**前向 ~50 行、反向 ~80 行，零框架**。

        跑一次前向，打印形状流水线和参数账（合计
        26,018——块占 97%）。反向的每一环都是老朋友：
        残差分叉、LN 投影、注意力三件套、GELU 导数、
        `np.add.at` 散射回嵌入。
        """
    )
    return


@app.cell
def _(np):
    # 演示 2 · TinyEncoder 完整实现
    def softmax_rows43(S):
        _E = np.exp(S - S.max(-1, keepdims=True))
        return _E / _E.sum(-1, keepdims=True)

    def ln_fwd43(Z, eps=1e-5):
        _mu = Z.mean(-1, keepdims=True)
        _var = Z.var(-1, keepdims=True)
        _s = np.sqrt(_var + eps)
        return (Z - _mu) / _s, (Z, (Z - _mu) / _s, _s)

    def ln_bwd43(dY, cache):
        _Z, _y, _s = cache
        _n = _Z.shape[-1]
        return ((dY - dY.mean(-1, keepdims=True))
                - _y * (dY * _y).sum(-1, keepdims=True) / _n) / _s

    def gelu43(x):
        _u = np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)
        return 0.5 * x * (1 + np.tanh(_u))

    def gelu_grad43(x):
        _u = np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)
        _du = np.sqrt(2.0 / np.pi) * (1 + 3 * 0.044715 * x ** 2)
        _t = np.tanh(_u)
        return 0.5 * (1 + _t) + 0.5 * x * (1 - _t ** 2) * _du

    class TinyEncoder43:
        """Pre-LN Transformer 编码器：嵌入+PE → N 块 → CLS 分类。"""

        def __init__(self, V, d, h, d_ff, n_blocks, T, seed=0):
            _r = np.random.default_rng(seed)
            self.V, self.d, self.h, self.d_ff, self.n_blocks, self.T = V, d, h, d_ff, n_blocks, T
            self.dk = d // h
            self.E = _r.normal(0, 0.5, (V, d))              # 词嵌入（含 CLS）
            _pe = np.zeros((T, d))                          # 正弦位置编码（固定）
            for _i in range(d // 2):
                _ang = np.arange(T) / (10000 ** (2 * _i / d))
                _pe[:, 2 * _i] = np.sin(_ang)
                _pe[:, 2 * _i + 1] = np.cos(_ang)
            self.PE = _pe
            self.blocks = []
            for _ in range(n_blocks):
                self.blocks.append({
                    "Wq": _r.normal(0, 0.3, (d, d)), "Wk": _r.normal(0, 0.3, (d, d)),
                    "Wv": _r.normal(0, 0.3, (d, d)), "Wo": _r.normal(0, 0.3, (d, d)),
                    "bq": np.zeros(d), "bk": np.zeros(d), "bv": np.zeros(d), "bo": np.zeros(d),
                    "g1": np.ones(d), "b1": np.zeros(d), "g2": np.ones(d), "b2": np.zeros(d),
                    "W1": _r.normal(0, 0.3, (d, d_ff)), "b1f": np.zeros(d_ff),
                    "W2": _r.normal(0, 0.3, (d_ff, d)), "b2f": np.zeros(d)})
            self.Wc = _r.normal(0, 0.3, (d, 2))
            self.bc = np.zeros(2)

        def params(self):
            _out = {"E": self.E, "Wc": self.Wc, "bc": self.bc}
            for _bi, _blk in enumerate(self.blocks):
                for _k, _v in _blk.items():
                    _out[f"b{_bi}.{_k}"] = _v
            return _out

        def forward(self, ids):
            """ids (B,T) → logits (B,2)。缓存全部中间量供反向。"""
            _B, _T = ids.shape
            _c = {"ids": ids}
            _H = self.E[ids] + self.PE[:_T]                  # 嵌入 + 位置编码
            for _bi, _blk in enumerate(self.blocks):
                _z1, _c1 = ln_fwd43(_H)                     # Pre-LN ①
                _Q, _K, _Vv = _z1 @ _blk["Wq"] + _blk["bq"], _z1 @ _blk["Wk"] + _blk["bk"], _z1 @ _blk["Wv"] + _blk["bv"]
                _Qh = _Q.reshape(_B, _T, self.h, self.dk).transpose(0, 2, 1, 3)
                _Kh = _K.reshape(_B, _T, self.h, self.dk).transpose(0, 2, 1, 3)
                _Vh = _Vv.reshape(_B, _T, self.h, self.dk).transpose(0, 2, 1, 3)
                _S = np.einsum("bhtd,bhud->bhtu", _Qh, _Kh) / np.sqrt(self.dk)
                _A = softmax_rows43(_S)                     # (B,h,T,T)
                _Oh = np.einsum("bhtu,bhud->bhtd", _A, _Vh)
                _O = _Oh.transpose(0, 2, 1, 3).reshape(_B, _T, self.h * self.dk)
                _H1 = _H + (_O @ _blk["Wo"] + _blk["bo"])   # 残差 ①
                _z2, _c2 = ln_fwd43(_H1)                    # Pre-LN ②
                _F1 = _z2 @ _blk["W1"] + _blk["b1f"]
                _G = gelu43(_F1)
                _H2 = _H1 + (_G @ _blk["W2"] + _blk["b2f"]) # 残差 ②
                _c[f"b{_bi}"] = dict(z1c=_c1, z2c=_c2, Q=_Q, K=_K, Vv=_Vv, A=_A, O=_O, H=_H, F1=_F1, G=_G)
                _H = _H2
            _cls = _H[:, 0]                                 # CLS 位置
            _c["cls"] = _cls
            return _cls @ self.Wc + self.bc, _c

        def backward(self, dlogits, cache):
            """全链路反向：梯度累加进 self.grads。"""
            _B, _T = cache["ids"].shape
            _d = self.d
            _g = {_k: np.zeros_like(_v) for _k, _v in self.params().items()}
            _g["Wc"] += cache["cls"].T @ dlogits             # 分类头
            _g["bc"] += dlogits.sum(0)
            _dH = np.zeros((_B, _T, _d))                     # 梯度只在 CLS 位置非零
            _dH[:, 0] = dlogits @ self.Wc.T
            for _bi in reversed(range(self.n_blocks)):
                _blk = self.blocks[_bi]
                _c = cache[f"b{_bi}"]
                _p = f"b{_bi}."
                # ---- FFN 子层反向 ----
                _dF2 = _dH
                _g[_p+"W2"] += _c["G"].reshape(-1, self.d_ff).T @ _dF2.reshape(-1, _d)
                _g[_p+"b2f"] += _dF2.sum((0, 1))
                _dG = _dF2 @ _blk["W2"].T
                _dF1 = _dG * gelu_grad43(_c["F1"])
                _g[_p+"W1"] += _c["z2c"][1].reshape(-1, _d).T @ _dF1.reshape(-1, self.d_ff)
                _g[_p+"b1f"] += _dF1.sum((0, 1))
                _dz2 = _dF1 @ _blk["W1"].T
                _dH1 = _dH + ln_bwd43(_dz2, _c["z2c"])
                # ---- 注意力子层反向 ----
                _dattn = _dH1
                _g[_p+"Wo"] += _c["O"].reshape(-1, self.h * self.dk).T @ _dattn.reshape(-1, _d)
                _g[_p+"bo"] += _dattn.sum((0, 1))
                _dO = _dattn @ _blk["Wo"].T
                _dOh = _dO.reshape(_B, _T, self.h, self.dk).transpose(0, 2, 1, 3)
                _Vh = _c["Vv"].reshape(_B, _T, self.h, self.dk).transpose(0, 2, 1, 3)
                _dA = np.einsum("bhtd,bhud->bhtu", _dOh, _Vh)
                _dVh = np.einsum("bhtu,bhtd->bhud", _c["A"], _dOh)
                _dS = _c["A"] * (_dA - (_dA * _c["A"]).sum(-1, keepdims=True)) / np.sqrt(self.dk)
                _Qh = _c["Q"].reshape(_B, _T, self.h, self.dk).transpose(0, 2, 1, 3)
                _Kh = _c["K"].reshape(_B, _T, self.h, self.dk).transpose(0, 2, 1, 3)
                _dQh = np.einsum("bhtu,bhud->bhtd", _dS, _Kh)
                _dKh = np.einsum("bhtu,bhtd->bhud", _dS, _Qh)
                _dQ = _dQh.transpose(0, 2, 1, 3).reshape(_B, _T, -1)
                _dK = _dKh.transpose(0, 2, 1, 3).reshape(_B, _T, -1)
                _dVv = _dVh.transpose(0, 2, 1, 3).reshape(_B, _T, -1)
                _z1 = _c["z1c"][1]
                _g[_p+"Wq"] += _z1.reshape(-1, _d).T @ _dQ.reshape(-1, _d)
                _g[_p+"bq"] += _dQ.sum((0, 1))
                _g[_p+"Wk"] += _z1.reshape(-1, _d).T @ _dK.reshape(-1, _d)
                _g[_p+"bk"] += _dK.sum((0, 1))
                _g[_p+"Wv"] += _z1.reshape(-1, _d).T @ _dVv.reshape(-1, _d)
                _g[_p+"bv"] += _dVv.sum((0, 1))
                _dz1 = _dQ @ _blk["Wq"].T + _dK @ _blk["Wk"].T + _dVv @ _blk["Wv"].T
                _dH = _dH1 + ln_bwd43(_dz1, _c["z1c"])
            np.add.at(_g["E"], cache["ids"], _dH)            # 散射回嵌入
            self.grads = _g
            return _g

    # 形状流水线 + 参数账
    _V43, _T43, _d43, _h43, _dff43, _nb43 = 17, 12, 32, 4, 128, 2
    _m = TinyEncoder43(_V43, _d43, _h43, _dff43, _nb43, _T43, seed=0)
    _ids, _ = np.random.default_rng(0).integers(0, 17, (3, 12)), None
    _logits, _ = _m.forward(_ids)
    print(f"输入 ids {_ids.shape} → logits {_logits.shape}")
    _n_par = sum(_p.size for _p in _m.params().values())
    _n_blk = sum(_p.size for _k, _p in _m.params().items() if _k.startswith("b"))
    print(f"参数账：嵌入 17×32=544 + 块 {_n_blk:,} + 分类头 66 = 合计 {_n_par:,}")
    print(f"块占比 {_n_blk/_n_par:.0%}——Transformer 的参数住在块里")
    return (TinyEncoder43, gelu43, gelu_grad43, ln_bwd43, ln_fwd43, softmax_rows43)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 3 · 梯度校验：反向传播的体检报告

        小配置（V=7, T=5, d=6, 2 头, 2 块）上做**全参数
        数值校验**：764 个位置逐个扰动 ±10⁻⁶，对比解析
        梯度与数值梯度。

        **判据看绝对误差相对全局梯度尺度**：>10⁻⁵ 就有
        bug。本实现实测最大误差 ~10⁻¹⁰——机器精度级
        的吻合。写深度学习框架级代码，这一步省不得：
        反向传播的 bug 往往不报错，只让训练神秘地
        不收敛。
        """
    )
    return


@app.cell
def _(TinyEncoder43, np, softmax_rows43):
    # 演示 3 · 数值梯度校验（全参数）
    _V, _T, _d, _h, _dff, _nb, _B = 7, 5, 6, 2, 12, 2, 3
    _model = TinyEncoder43(_V, _d, _h, _dff, _nb, _T, seed=0)
    _rng = np.random.default_rng(1)
    _ids = _rng.integers(0, _V, (_B, _T))
    _logits, _cache = _model.forward(_ids)
    _w = _rng.normal(0, 1, _logits.shape)                 # 随机读出做标量目标
    _grads = _model.backward(_w.copy(), _cache)

    _eps = 1e-6
    _gscale = max(np.abs(_g).max() for _g in _grads.values())
    _n_bad = 0
    _n_chk = 0
    _worst = 0.0
    for _name, _p in _model.params().items():
        _flat = _p.reshape(-1)
        for _i in range(len(_flat)):
            _orig = _flat[_i]
            _flat[_i] = _orig + _eps
            _lp, _ = _model.forward(_ids)
            _flat[_i] = _orig - _eps
            _lm, _ = _model.forward(_ids)
            _flat[_i] = _orig
            _num = ((_lp * _w).sum() - (_lm * _w).sum()) / (2 * _eps)
            _ana = _grads[_name].reshape(-1)[_i]
            _err = abs(_num - _ana) / _gscale
            _worst = max(_worst, _err)
            _n_chk += 1
            if _err > 1e-5:
                _n_bad += 1
    print(f"梯度校验：{_n_chk} 个参数位置全查")
    print(f"  全局梯度尺度 {_gscale:.3f}")
    print(f"  最大相对误差 {_worst:.1e}（判据 1e-5）")
    print(f"  超标位置：{_n_bad}")
    print("\n→ 机器精度级吻合：从交叉熵到词嵌入的全链路反向，每一环都对")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 4 · 训练：20 秒到 100%

        300 步 Adam（lr=0.01，batch=64）——25 课的训练
        循环骨架，模型换成今天的手写编码器。看损失从
        5.8（瞎猜 ln2≈0.69 之上还有初始化的胡乱自信）
        一路砸到 0；**测试集 500 句 100%**。

        留意：全程没有验证曲线的拉锯——数据无限生成，
        过拟合根本没机会登场（对比 37 课小语料的
        惨烈早停，是天壤之别的两种世界）。
        """
    )
    return


@app.cell
def _(TinyEncoder43, make_data43, np, plt, softmax_rows43):
    # 演示 4 · 训练循环（25 课骨架 + 今天的模型）
    import time

    def train_encoder43(model, steps=300, batch=64, lr=0.01, seed=1):
        _tr = np.random.default_rng(seed)
        _params = model.params()
        _ms = {_k: np.zeros_like(_v) for _k, _v in _params.items()}
        _vs = {_k: np.zeros_like(_v) for _k, _v in _params.items()}
        _hist = []
        for _step in range(1, steps + 1):
            _ids, _y = make_data43(batch, model.T, _tr)
            _logits, _cache = model.forward(_ids)
            _P = softmax_rows43(_logits)
            _loss = -np.log(_P[np.arange(batch), _y] + 1e-12).mean()
            _dl = _P.copy()
            _dl[np.arange(batch), _y] -= 1
            _dl /= batch
            _grads = model.backward(_dl, _cache)
            for _k in _params:
                _ms[_k] = 0.9 * _ms[_k] + 0.1 * _grads[_k]
                _vs[_k] = 0.999 * _vs[_k] + 0.001 * _grads[_k] * _grads[_k]
                _params[_k] -= lr * (_ms[_k] / (1 - 0.9 ** _step)) / (
                    np.sqrt(_vs[_k] / (1 - 0.999 ** _step)) + 1e-8)
            _hist.append((_loss, (_logits.argmax(1) == _y).mean()))
        return np.array(_hist)

    _t0 = time.time()
    model43 = TinyEncoder43(17, 32, 4, 128, 2, 12, seed=0)
    hist43 = train_encoder43(model43)
    _dt = time.time() - _t0
    print(f"训练 300 步（batch 64）用时 {_dt:.0f}s")
    print(f"loss：{hist43[0, 0]:.3f} → {hist43[-1, 0]:.4f}｜训练准确率："
          f"{hist43[0, 1]:.0%} → {hist43[-1, 1]:.0%}")

    # 测试：500 句全新句子
    _te = np.random.default_rng(99)
    _ok = 0
    for _ in range(5):
        _ids, _y = make_data43(100, 12, _te)
        _logits, _ = model43.forward(_ids)
        _ok += (_logits.argmax(1) == _y).sum()
    print(f"测试集 500 句：{_ok}/500 = {_ok/500:.0%}")

    _fig, _ax = plt.subplots(figsize=(8.8, 3.7))
    _ax.plot(hist43[:, 0], color="tab:blue", lw=1.6, label="训练损失")
    _ax.axhline(np.log(2), color="gray", ls="--", lw=1)
    _ax.text(5, np.log(2) + 0.12, "瞎猜线 ln2", fontsize=9, color="gray")
    _ax.set_xlabel("训练步数")
    _ax.set_ylabel("交叉熵损失")
    _ax.set_title("手写 Transformer 编码器：300 步训到测试 100%", fontsize=11.5)
    _ax.legend(fontsize=9)
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()
    _fig
    return (hist43, model43, train_encoder43)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 5 · 注意力解剖：触发词的下落

        模型 100% 准确——但它是**怎么**找到触发词的？
        拿 200 个测试句做统计解剖：

        - **块 1：触发词是信息枢纽**。它被全句看的
          总量（注意力列和）是填充词的 **3.3 倍**
          （2.55 vs 0.77）——第一块先把触发词的信息
          广播出去
        - **块 2：CLS 的目光统计上偏向触发词**
          （0.126 vs 0.079，1.6 倍），但单句看并不
          总是直视——触发词的信息常常**经中转位置
          流向 CLS**（两跳路由：块 1 广播 → 中转
          站缓存 → 块 2 CLS 取件）
        - 这不是 bug：**注意力是路由器，不是聚光灯**
          ——只要信息到了 CLS，损失函数不在乎走几跳。
          真实大模型的可解释性研究（GPT-2 的间接
          寻址头）看到的就是这类多跳电路
        """
    )
    return


@app.cell
def _(make_data43, model43, np, plt, vocab43):
    # 演示 5 · 注意力统计解剖 + 示例句热力图
    _rng = np.random.default_rng(123)
    _trig_cls, _fill_cls, _trig_col, _fill_col = [], [], [], []
    for _ in range(200):
        _ids, _y = make_data43(1, 12, _rng)
        _logits, _c = model43.forward(_ids)
        _tp = int(np.argmax((_ids[0] >= 1) & (_ids[0] <= 8)))
        for _bi in (0, 1):
            _A = _c[f"b{_bi}"]["A"][0].mean(0)             # 头平均
            _row = _A[0, 1:]                               # CLS 行
            _mask = np.zeros(11, bool); _mask[_tp - 1] = True
            _trig_cls.append((_bi, _row[_mask].mean()))
            _fill_cls.append((_bi, _row[~_mask].mean()))
            _col = _A[1:].sum(0)                           # 被看总量
            _trig_col.append((_bi, _col[_tp]))
            _fill_col.append((_bi, np.mean(np.delete(_col, _tp))))
    print("200 句统计（头平均）：")
    for _bi in (0, 1):
        _tc = np.mean([v for b, v in _trig_cls if b == _bi])
        _fc = np.mean([v for b, v in _fill_cls if b == _bi])
        _tcol = np.mean([v for b, v in _trig_col if b == _bi])
        _fcol = np.mean([v for b, v in _fill_col if b == _bi])
        print(f"  块{_bi+1}：CLS 看触发词 {_tc:.3f} vs 看填充词 {_fc:.3f}"
              f"（{_tc/_fc:.1f} 倍）｜触发词被看总量 {_tcol:.2f} vs {_fcol:.2f}"
              f"（{_tcol/_fcol:.1f} 倍）")

    # 示例句的两块注意力热力图
    _ids, _y = make_data43(1, 12, np.random.default_rng(7))
    _logits, _c = model43.forward(_ids)
    _tp = int(np.argmax((_ids[0] >= 1) & (_ids[0] <= 8)))
    _sent = [vocab43[_j] for _j in _ids[0]]
    print(f"\n示例句：{' '.join(_sent)}（触发词「{_sent[_tp]}」在位置 {_tp}）")
    print(f"预测：{'正面' if _logits[0].argmax() == 0 else '负面'}"
          f"（正确：{'正面' if _y[0] == 0 else '负面'}）")

    _fig, _axes = plt.subplots(1, 2, figsize=(11.8, 4.9))
    for _ax, _bi in zip(_axes, (0, 1)):
        _A = _c[f"b{_bi}"]["A"][0].mean(0)
        _im = _ax.imshow(_A, cmap="Blues", vmin=0, vmax=0.6)
        _ax.set_xticks(range(12), _sent, rotation=45)
        _ax.set_yticks(range(12), _sent)
        _ax.add_patch(plt.Rectangle((_tp - 0.5, -0.5), 1, 12, fill=False,
                                    edgecolor="tab:red", lw=2))
        _ax.set_title(f"块{_bi+1} 注意力（红框 = 触发词的列）", fontsize=11)
        _fig.colorbar(_im, ax=_ax, shrink=0.75)
    _fig.suptitle("解剖：块 1 的触发词列亮（信息枢纽）→ 块 2 的 CLS 行取件", fontsize=12)
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
        ### 交互实验 · 出题考模型

        你来出题：选**触发词**（正面/负面各两种）、
        定**位置**（1~11）、换**填充词种子**——现场
        生成句子，看模型预测和 CLS 的目光。

        玩法建议：

        - 触发词放中间 vs 放两端：注意力找它一样快
          （位置编码的功劳——41 课的身份证让"在哪"
          都可读）
        - 连续换种子：填充词怎么变都不影响——模型
          确实学会了"找情绪词"，不是背句子
        - 盯 CLS 行的最高格：它有时直视触发词，
          有时看中转站——两跳路由的现场版
        - 极限测试：把触发词放位置 1（紧贴 CLS），
          预测最笃定吗？
        """
    )
    return


@app.cell
def _(mo):
    # 交互控制：触发词 + 位置 + 种子
    trig_dd43 = mo.ui.dropdown(options=["棒（正）", "优（正）", "烦（负）", "差（负）"],
                               value="烦（负）", label="🎯 触发词")
    pos_slider43 = mo.ui.slider(1, 11, value=5, step=1, debounce=True,
                                label="📍 触发词位置")
    fill_slider43 = mo.ui.slider(0, 99, value=0, step=1, debounce=True,
                                 label="🎲 填充词种子")
    mo.hstack([trig_dd43, pos_slider43, fill_slider43])
    return (fill_slider43, pos_slider43, trig_dd43)


@app.cell
def _(fill_slider43, make_data43, model43, mo, np, plt, pos_slider43,
      softmax_rows43, trig_dd43, vocab43):
    # ===== 交互实验 · 出题考模型（响应单元）=====
    _trig_word = trig_dd43.value[0]                        # 取第一个字
    _trig_id = vocab43.index(_trig_word)
    _pos = pos_slider43.value
    _rng = np.random.default_rng(1000 + fill_slider43.value)
    _fill = list(_rng.choice(np.arange(9, 17), 10))
    _fill.insert(_pos - 1, _trig_id)
    _ids = np.array([[0] + _fill])
    _sent = [vocab43[_j] for _j in _ids[0]]

    _logits, _c = model43.forward(_ids)
    _P = softmax_rows43(_logits)[0]
    _pred = "正面" if _logits[0].argmax() == 0 else "负面"

    _A2 = _c["b1"]["A"][0].mean(0)                         # 块 2 的头平均
    _cls_row = _A2[0, 1:]
    _top = int(np.argmax(_cls_row)) + 1

    _fig, _axes = plt.subplots(1, 2, figsize=(11.6, 3.8), gridspec_kw={"width_ratios": [1.4, 1]})
    _colors = ["tab:red" if _i == _pos else "tab:blue" for _i in range(1, 12)]
    _axes[0].bar(range(1, 12), _cls_row, color=_colors)
    _axes[0].set_xticks(range(1, 12), _sent[1:], rotation=45)
    _axes[0].set_ylabel("CLS 的注意力权重")
    _axes[0].set_title("块 2 CLS 的目光（红 = 你放的触发词）", fontsize=11)
    _axes[0].grid(True, axis="y", alpha=0.3)
    _axes[1].barh([1, 0], [_P[0], _P[1]], color=["tab:green", "tab:red"], height=0.55)
    _axes[1].set_yticks([1, 0], ["正面", "负面"])
    _axes[1].set_xlim(0, 1.05)
    _axes[1].set_title(f"预测：{_pred}（P={_P.max():.2f}）", fontsize=11)
    _axes[1].grid(True, axis="x", alpha=0.3)
    _fig.tight_layout()

    if _top == _pos:
        _route = (f"CLS 的目光最高格正好落在触发词「{_sent[_pos]}」（{_cls_row[_pos-1]:.2f}）——"
                  "这一句走的是直视路线。")
    else:
        _route = (f"CLS 的目光最高格在「{_sent[_top]}」（{_cls_row[_top-1]:.2f}），"
                  f"触发词「{_sent[_pos]}」{_cls_row[_pos-1]:.2f}——两跳路由："
                  "块 1 已把触发词信息广播到中转站，CLS 从那里取件。")
    _verdict = (f"**预测 {_pred}（P={_P.max():.2f}）**——{_route}\n\n"
                "无论触发词放在哪、填充词怎么换，模型都稳稳找到它："
                "注意力学会了「按内容找词」，位置无关（41 课位置编码的功劳）。"
                "这就是编码器的本事：**读一整句，出一个判断**。")
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
                **练习 1 · 数数据**：训练 300 步 × 每步
                64 句 = 多少个训练句子？模型 26,018 个
                参数——为什么这么多次训练也没有过拟合
                （17 课的老病）？

                **练习 2 · 补全 CLS 头**：模型收尾两步——
                从最终表示 H（B×T×d）取出 CLS 位置的
                向量，再算出二分类 logits。

                **练习 3 · 数块参数**：本课每块的主干
                参数（不算偏置和 LN 的 γβ）是
                4d²（注意力）+ 8d²（FFN）= 12d²。
                d=32 时一块是多少？两块呢？
                """
            )
        }
    )
    return


@app.cell
def _(mo):
    # ===== 练习 1 · 数数据 =====
    print("训练 300 步 × batch 64 句；模型参数 26,018")

    n_sent_ex1_43 = None                               # ← ① 训练句子总数（整数）
    why_no_of_ex1_43 = None                            # ← ② 为什么不过拟合（0=数据无限生成，1=参数太少）

    if n_sent_ex1_43 is not None and why_no_of_ex1_43 is not None:
        _n, _w = 300 * 64, 0
        if n_sent_ex1_43 == _n and why_no_of_ex1_43 == _w:
            _msg = (f"✅ 正确：19,200 句。不过拟合的原因不是参数少（26,018 个参数"
                    "配 1.9 万句，按 17 课的教训早该背书了），而是**数据无限生成**："
                    "每个 batch 都是现场随机造的新句子，模型永远见不到重复——"
                    "背书无从谈起。对比 37 课（1.2 万字符语料养十几万参数，"
                    "过拟合到必须早停）：**数据新鲜度决定过拟合的命运**。"
                    "这也是大模型「Scaling Laws」的第一课：更多不同数据，"
                    "几乎总是一项好投资。")
        else:
            _msg = "❌ 参考：19,200；原因填 0（数据无限生成）"
    else:
        _msg = "⏳ 两空都填上"
    mo.md(_msg)
    return (n_sent_ex1_43, why_no_of_ex1_43)


@app.cell
def _(mo, np):
    # ===== 练习 2 · 补全 CLS 头 =====
    def cls_head_ex2_43(H, Wc, bc):
        """H (B,T,d)：编码器最终表示。返回 (B,2) logits。"""
        _cls = None                                    # ← ① 补：取每个样本位置 0 的向量 (B,d)
        return None                                    # ← ② 补：线性分类头

    cls_ready_ex2_43 = None                            # ← 补完后，换成 cls_head_ex2_43

    if cls_ready_ex2_43 is not None:
        try:
            _H = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(float)
            _Wc = np.ones((4, 2))
            _bc = np.zeros(2)
            _out = cls_ready_ex2_43(_H, _Wc, _bc)
            _ref = _H[:, 0] @ _Wc + _bc
            if _out is not None and _out.shape == (2, 2) and np.allclose(_out, _ref):
                _msg = ("✅ 正确。取 H[:, 0]（而不是 H.mean(1)）：CLS 方案靠注意力"
                        "把全句信息汇聚到位置 0——**汇总的活儿交给注意力，"
                        "不交给平均**（平均会稀释：触发词 1 个、填充词 10 个，"
                        "平均下来信号剩 1/11）。另一个细节：梯度只流向位置 0"
                        "（反向里 _dH[:, 0] = ...）——CLS 是唯一的信息漏斗口。")
            else:
                _msg = "❌ 输出应为 (2, 2)：H[:, 0] @ Wc + bc"
        except (TypeError, ValueError) as _e:
            _msg = f"❌ 运行出错：{_e}——_cls = H[:, 0]；return _cls @ Wc + bc"
    else:
        _msg = "⏳ 补两行：取位置 0 → 线性头"
    mo.md(_msg)
    return (cls_head_ex2_43, cls_ready_ex2_43)


@app.cell
def _(mo):
    # ===== 练习 3 · 数块参数 =====
    print("d = 32，一块 = 12d²（主干，不含偏置和 LN 的 γβ）")

    one_block_ex3_43 = None                            # ← ① 一块的主干参数（整数）
    two_blocks_ex3_43 = None                           # ← ② 两块合计（整数）

    if one_block_ex3_43 is not None and two_blocks_ex3_43 is not None:
        _n1, _n2 = 12 * 32 ** 2, 2 * 12 * 32 ** 2
        if abs(one_block_ex3_43 - _n1) < 100 and abs(two_blocks_ex3_43 - _n2) < 200:
            _msg = (f"✅ 正确：一块 12×32² = {_n1:,}，两块 {_n2:,}。加上偏置、"
                    "LN 的 γβ（每块 416）、嵌入 544、分类头 66，就是代码里"
                    "打印的合计 26,018。按这个比例感受尺度：GPT-2 小号 d=768、"
                    "12 块 → 12×12×768² ≈ 8500 万块参数——同样的公式，"
                    "d 从 32 涨到 768，参数涨 576 倍（d² 的威力）。")
        else:
            _msg = "❌ 参考：12×1024 = 12,288；两块 24,576"
    else:
        _msg = "⏳ 两空都填上（12d²）"
    mo.md(_msg)
    return (one_block_ex3_43, two_blocks_ex3_43)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 1 参考答案（先自己写过再看！）": mo.md(
                """
                **19,200 句；不过拟合是因为数据无限生成
                （不是参数少）。**

                参数/样本比是 26,018/19,200 ≈ 1.4——按
                17 课的经验早就该背书了。但这里每个
                batch 都是现场随机的新句子，**模型连
                一句重复的都见不到**——死记硬背没有
                存储的地方。

                三个世界的对比：

                | 场景 | 数据 | 结局 |
                |---|---|---|
                | 17 课 | 固定小数据集 | 过拟合、早停 |
                | 37 课 | 1.2 万字符语料 | 过拟合、早停 |
                | 本课 | 无限生成 | 想过拟合都没门 |

                大模型的真相在这两者之间：互联网文本
                有限但巨大——所以既有惊人的泛化，
                也有「背过维基百科」的记忆诉讼。
                """
            ),
            "✅ 练习 2 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                _cls = H[:, 0]              # (B, d)
                return _cls @ Wc + bc       # (B, 2)
                ```

                CLS vs 平均池化（mean pooling）是编码器
                分类的两大流派：

                - **平均池化**：pooled = H.mean(1)——
                  简单、无需特殊记号，但对「1 个信号
                  词 + 10 个噪声词」的句子会被稀释
                - **CLS**：加一个专用记号，让注意力
                  自己决定汇聚什么——多一层间接性，
                  换来选择性。BERT 选它，本课也选它

                共同点是：**序列→向量**这一步必须有人
                做（分类头吃定长输入）。44 课的解码器
                不需要——它输出整个序列（每个位置
                一个分布）。
                """
            ),
            "✅ 练习 3 参考答案（先自己写过再看！）": mo.md(
                """
                **一块 12,288；两块 24,576。**

                | 规模 | d | 块数 | 块参数 |
                |---|---|---|---|
                | 本课玩具 | 32 | 2 | 2.5 万 |
                | BERT-mini | 128 | 4 | 79 万 |
                | BERT-base | 768 | 12 | 8,500 万 |
                | GPT-3 | 12,288 | 96 | 175 亿 |

                参数随 **d² × 块数** 增长：GPT-3 相比
                本课是 (12288/32)² × 48 ≈ 70 万倍。
                训练算力随参数 × 数据增长——GPT-3
                用了本课实验约 10²⁰ 倍的计算。你用
                20 秒跑完的这 300 步，就是那张著名
                Scaling Laws 曲线最左下角的第一粒
                像素。
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
                1. **编码器数据流**：嵌入+PE → N 个
                   Pre-LN 块 → CLS 位置 → 分类头；
                   参数住在块里（本课模型 26,018，块占 98%）
                2. **CLS 的角色**：无语义的特殊记号，
                   靠注意力汇聚全句信息——「汇总交给
                   注意力，不交给平均」（平均会稀释）
                3. **全链路反向**：交叉熵→softmax→
                   分类头→CLS→残差分叉→LN 投影→注意力
                   三件套→GELU→嵌入散射——22-23 课的
                   总成，~80 行
                4. **数值梯度校验**：扰动 ±ε 对账解析
                   梯度，误差 10⁻¹⁰ 级才放行——框架级
                   代码的看家本领（PyTorch gradcheck 同理）
                5. **训练结果**：300 步 Adam、约 20 秒、
                   测试 500 句 100%——数据无限生成，
                   过拟合没有出场机会
                6. **注意力解剖**：块 1 把触发词炼成
                   信息枢纽（被看总量 3.3 倍）；块 2 的
                   CLS 统计上偏向触发词（1.6 倍）但常走
                   两跳路由——**注意力是路由器，
                   不是聚光灯**
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

                **下一课预告** — `44_解码器与掩码`（第 7 部分 · Transformer）：

                编码器读一整句出一个判断；**解码器读
                前文出一个词**——37 课语言模型的注意力
                版。全部零件不变，只加一张**掩码**：

                - **因果掩码**：位置 t 只许看 1..t——
                  上三角填 −∞，softmax 后未来权重
                  归零。训练时**一次并行算完 T 个
                  位置**的下一词预测（RNN 要 T 步的
                  事，注意力一步）
                - **自回归生成**：推理时逐词生成，
                  每步把新词拼回去——「只会写下一个
                  字」的机器怎么写出一整篇文章
                - **交叉注意力**：解码器还能**看编码器
                  的输出**（翻译任务：看原句写译文）——
                  Q 来自解码器、K/V 来自编码器

                44 课把这三件套装进今天的代码框架，
                训练一台迷你 GPT（字符级，还是 37 课
                那篇语料），亲眼对比它和 RNN/LSTM
                的生成质量。
                """
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
