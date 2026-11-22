import marimo

__generated_with = "0.25.2"
app = marimo.App(app_title="44 · 解码器与掩码", width="medium")


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


@app.cell
def _(np):
    import os

    # ==== 语料加载（37 课同款：data/charlm37.npz）====
    def load_corpus44():
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

    _z44 = np.load(load_corpus44(), allow_pickle=True)
    VOCAB44 = list(_z44["vocab"])
    IDS44 = _z44["ids"].astype(np.int64)
    N_TR44 = int(_z44["n_tr"])
    print(f"语料：{len(IDS44):,} 字符｜词表 {len(VOCAB44)}｜训练/验证 {N_TR44}/{len(IDS44)-N_TR44}")
    print("（与 37 课同一份语料——今天用 Transformer 重新打这场仗）")
    return (IDS44, N_TR44, VOCAB44, load_corpus44)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # 44 · 解码器与掩码：亲手造一台迷你 GPT

        **从零到 Transformer · 手写深度学习** ｜ 第 7 部分 · Transformer（7/8）
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📖 本课目标（先读我）": mo.md(
                """
                **43 课的编码器读一整句、出一个判断；今天
                的解码器读前文、出一个词——37 课语言模型
                的注意力版，也就是一台迷你 GPT。**

                全部零件都是旧的，只加一张**因果掩码**：
                位置 t 只许看 1..t，不许偷看未来。就这一
                张纸，注意力从"双向阅读"变成"逐词写作"。

                学完你将能：

                - 手写**因果掩码**并解释为什么用 −10⁹
                  而不是 0（softmax 的数学性质决定）
                - 亲眼看到**训练并行**的威力：一个 32 字
                  窗口，RNN 要 32 步串行，解码器**一次
                  前向**同时算出 32 个位置的下一字预测
                - 在 37 课的同一份语料上训练迷你 GPT
                  （11.5 万参数，约 1 分钟），与 RNN/LSTM
                  **同台对决**——结果可能出乎你的意料
                - 理解**自回归生成**：推理时为什么又变回
                  串行（训练并行、推理串行——GPT 的
                  双面性格）
                - 看懂**交叉注意力**：翻译任务里解码器
                  怎么"看原句写译文"（Q 与 K/V 来自
                  不同地方的注意力）

                前置：43 课编码器（今天的代码就是它加
                掩码换头）、37 课语言模型与困惑度、
                41 课位置编码。

                对决的悬念先按下：小语料上，迷你 GPT
                真的赢吗？
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "🎭 概念 1/3 · 因果掩码：一张纸管住未来": mo.md(
                """
                解码器的注意力要满足一条纪律：**位置 t
                只能看 1..t**（写第 5 个字时还没写过第 6
                个字）。实现只要两行：

                ```python
                mask = np.triu(np.full((T, T), -1e9), k=1)   # 上三角填 -10⁹
                S = Q @ K.T / √d + mask                       # 打分时叠加
                ```

                为什么是 −10⁹ 不是 0？softmax 里
                e^(s−max)：被掩的位置分数比任何真实分数
                低十亿，指数后是 e^(−10⁹)≈0——**未来被
                数学性地抹除**，一行梯度都不用改（掩码是
                常数，反向自动穿过）。如果用 0，未来位置
                会拿到 e⁰=1 的权重——等于偷偷把未来
                拉进来了。

                掩码后的注意力矩阵是**下三角**：第 1 行
                只看自己，第 2 行看 1-2，……最后一行看
                全句。每一行都是一个合法的"写到此处"
                的视野。
                """
            ),
            "⚡ 概念 2/3 · 训练并行，推理串行": mo.md(
                """
                GPT 的双面性格：

                **训练时并行**（teacher forcing）：把整段
                文本一次喂进去，因果掩码保证位置 t 的
                预测只用前文——于是 **一次前向同时算出
                T 个位置的"下一字"预测**，每个位置各自
                算交叉熵。RNN 同样的事要 T 步串行 +
                BPTT（34 课）。这是 Transformer 训练效率
                的根源：GPU 上百倍于此的加速。

                **推理时串行**（自回归）：真写文章时没有
                "未来的正确答案"可喂——第 t+1 个字要等
                第 t 个字生成完才能进上下文。生成 N 字
                = N 次前向（每次只取最后一个位置的
                输出）。**KV 缓存**是工业界对此的补救
                （算过的 K/V 存着不重算），但串行本质
                改不掉——这是今天所有大模型"打字速度"
                的瓶颈。

                一句话：**训练像考试（对着答案练），
                推理像写作（一个字一个字憋）**。
                """
            ),
            "🔀 概念 3/3 · 解码器全貌与交叉注意力": mo.md(
                """
                完整的"编码器-解码器"Transformer（翻译
                机器）里，解码器每个块有**三**个子层：

                ```
                ① 带掩码的自注意力     （看自己已写的）
                ② 交叉注意力           （看原文）★新
                ③ FFN
                ```

                交叉注意力和自注意力**公式一模一样**
                （39 课），只是角色来源不同：

                ```
                Q = 解码器当前表示 · Wq    （我要问什么）
                K = 编码器输出   · Wk     （原文怎么被检索）
                V = 编码器输出   · Wv     （原文交出什么）
                ```

                翻译"I love cats"→"我爱猫"：写"爱"时，
                解码器的查询去编码器的 3 个词里找——
                注意力压在"love"上，取回它的信息。学到
                的注意力矩阵就是一张**对齐表**（哪个
                译词对应哪个原词）——机器翻译可解释性
                的经典画面。

                GPT 只保留了 ①③（没有原文可看）；
                BERT 干脆全用双向。三家分晋，45 课
                细说。
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
        ### 演示 1 · 掩码长什么样：下三角的世界

        左图：因果掩码本身（蓝=可见，白=掩掉）。右图：
        掩码叠加后的**实际注意力矩阵**（随机初始化的
        模型）——严格下三角，未来一片空白。

        手算验证一个格子：3 个位置、某行打分
        [0, 2, 1]，掩码后 [0, 2, −10⁹]，softmax 只在
        前两个里分蛋糕：e⁰/(e⁰+e²)=0.12、
        e²/(e⁰+e²)=0.88、未来 = e^(−10⁹) ≈ 0。
        """
    )
    return


@app.cell
def _(np, plt):
    # 演示 1 · 因果掩码可视化
    _T = 12
    _mask_visible = np.tril(np.ones((_T, _T)))            # 可见性矩阵
    _rng = np.random.default_rng(0)
    _S = _rng.normal(0, 1, (_T, _T)) / np.sqrt(_T)
    _S = _S + np.triu(np.full((_T, _T), -1e9), k=1)
    _E = np.exp(_S - _S.max(-1, keepdims=True))
    _A = _E / _E.sum(-1, keepdims=True)

    _fig, _axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    _axes[0].imshow(_mask_visible, cmap="Blues", vmin=0, vmax=1.2)
    _axes[0].set_title("因果掩码：位置 t 只许看 1..t", fontsize=11)
    _axes[0].set_xlabel("被看位置")
    _axes[0].set_ylabel("当前位置")
    _im = _axes[1].imshow(_A, cmap="Blues")
    _axes[1].set_title("掩码后的注意力（随机模型）：严格下三角", fontsize=11)
    _axes[1].set_xlabel("被看位置")
    _fig.colorbar(_im, ax=_axes[1], shrink=0.8)
    _fig.tight_layout()
    _fig

    # 手算格子
    _s = np.array([0.0, 2.0])
    _p = np.exp(_s) / np.exp(_s).sum()
    print(f"手算：打分 [0, 2, 1] 掩码后 → softmax([0, 2]) = "
          f"[{_p[0]:.2f}, {_p[1]:.2f}]，未来 ≈ {np.exp(-1e9):.0e}")
    print("第 1 行只看自己（权重 1.0）；最后一行看全句——每行都是「写到此处」的合法视野")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 2 · 训练并行：一次前向，32 个预测

        从语料切一个 32 字窗口，**一次前向**得到 32 个
        位置的下一字分布。看打印：每个位置「输入字 →
        预测字 vs 目标字」——32 个预测同时出炉，没有
        循环。

        对照 RNN（37 课）：同样的窗口要跑 32 步、存
        32 份隐状态、再 BPTT 回传 32 层。**训练效率
        的差距就在这里**——GPU 时代这决定了谁能吃下
        亿级语料。
        """
    )
    return


@app.cell
def _(IDS44, VOCAB44, np):
    # 演示 2 · 一次前向的 32 个并行预测（未训练模型，预测是乱的——看点在并行本身）
    _T = 32
    _ids = IDS44[100:100 + _T]
    _rng = np.random.default_rng(0)
    _d, _h, _dff = 48, 4, 192
    _E = _rng.normal(0, 0.5, (len(VOCAB44), _d))
    _X = _E[_ids]
    # 一个简化单头注意力演示（够说明并行性）
    _Wq, _Wk, _Wv = (_rng.normal(0, 0.3, (_d, _d)) for _ in range(3))
    _Q, _K, _V = _X @ _Wq, _X @ _Wk, _X @ _Wv
    _S = _Q @ _K.T / np.sqrt(_d) + np.triu(np.full((_T, _T), -1e9), k=1)
    _A = np.exp(_S - _S.max(-1, keepdims=True)); _A /= _A.sum(-1, keepdims=True)
    _H = _X + _A @ _V
    _logits = _H @ _E.T                                  # (T, V)：一次算完所有位置
    print(f"输入窗口（{_T} 字）：{''.join(VOCAB44[i] for i in _ids)}")
    print(f"\n一次前向 → logits 形状 {_logits.shape}：{_T} 个位置各自的下一字分布")
    _pred = _logits.argmax(-1)
    _tgt = IDS44[101:101 + _T]
    _ok = (_pred == _tgt).sum()
    print(f"前 10 个位置的预测 vs 目标（未训练，纯属抽奖）：")
    for _t in range(10):
        print(f"  位置 {_t:2d}：「{VOCAB44[_ids[_t]]}」→ 预测「{VOCAB44[_pred[_t]]}」"
              f" vs 目标「{VOCAB44[_tgt[_t]]}」{'✓' if _pred[_t] == _tgt[_t] else '✗'}")
    print(f"\n命中率 {_ok}/{_T}（未训练的理论值 1/601≈0.05 个）")
    print("→ 32 个预测一次出炉：训练时每个窗口都是这么并行处理的")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 3 · MiniGPT：43 课的编码器加一张掩码

        `MiniGPT44` 与 43 课 `TinyEncoder43` 的**全部
        差异**只有三处（去代码里找）：

        - 打分时 `+ mask`（因果掩码，今天的主角）
        - 输出头从 (d, 2) 换成 (d, 601)——预测词表里的
          每个字符，而不是二分类
        - 没有 CLS：每个位置都有自己的预测任务

        反向传播沿用 43 课的全链路（掩码是常数，
        反向自动穿过）。老规矩：**梯度校验先行**——
        1387 个参数位置对数值梯度，误差 4×10⁻¹⁰。
        """
    )
    return


@app.cell
def _(np):
    # 演示 3 · MiniGPT 完整实现（43 课编码器 + 掩码 + 词表输出头）
    def softmax_rows44(S):
        _E = np.exp(S - S.max(-1, keepdims=True))
        return _E / _E.sum(-1, keepdims=True)

    def ln_fwd44(Z, eps=1e-5):
        _mu = Z.mean(-1, keepdims=True)
        _var = Z.var(-1, keepdims=True)
        _s = np.sqrt(_var + eps)
        return (Z - _mu) / _s, (Z, (Z - _mu) / _s, _s)

    def ln_bwd44(dY, cache):
        _Z, _y, _s = cache
        _n = _Z.shape[-1]
        return ((dY - dY.mean(-1, keepdims=True))
                - _y * (dY * _y).sum(-1, keepdims=True) / _n) / _s

    def gelu44(x):
        _u = np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)
        return 0.5 * x * (1 + np.tanh(_u))

    def gelu_grad44(x):
        _u = np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)
        _du = np.sqrt(2.0 / np.pi) * (1 + 3 * 0.044715 * x ** 2)
        _t = np.tanh(_u)
        return 0.5 * (1 + _t) + 0.5 * x * (1 - _t ** 2) * _du

    class MiniGPT44:
        """解码器：因果掩码自注意力 + FFN，每位置预测下一字符。"""

        def __init__(self, V, d, h, d_ff, n_blocks, T, seed=0):
            _r = np.random.default_rng(seed)
            self.V, self.d, self.h, self.d_ff, self.n_blocks, self.T = V, d, h, d_ff, n_blocks, T
            self.dk = d // h
            self.E = _r.normal(0, 0.5, (V, d))
            _pe = np.zeros((T, d))
            for _i in range(d // 2):
                _ang = np.arange(T) / (10000 ** (2 * _i / d))
                _pe[:, 2 * _i] = np.sin(_ang)
                _pe[:, 2 * _i + 1] = np.cos(_ang)
            self.PE = _pe
            self.mask = np.triu(np.full((T, T), -1e9), k=1)      # ★ 因果掩码
            self.blocks = []
            for _ in range(n_blocks):
                self.blocks.append({
                    "Wq": _r.normal(0, 0.3, (d, d)), "Wk": _r.normal(0, 0.3, (d, d)),
                    "Wv": _r.normal(0, 0.3, (d, d)), "Wo": _r.normal(0, 0.3, (d, d)),
                    "bq": np.zeros(d), "bk": np.zeros(d), "bv": np.zeros(d), "bo": np.zeros(d),
                    "g1": np.ones(d), "b1": np.zeros(d), "g2": np.ones(d), "b2": np.zeros(d),
                    "W1": _r.normal(0, 0.3, (d, d_ff)), "b1f": np.zeros(d_ff),
                    "W2": _r.normal(0, 0.3, (d_ff, d)), "b2f": np.zeros(d)})
            self.Wo_head = _r.normal(0, 0.3, (d, V))              # ★ 输出到整个词表
            self.bo_head = np.zeros(V)

        def params(self):
            _out = {"E": self.E, "Wo_head": self.Wo_head, "bo_head": self.bo_head}
            for _bi, _blk in enumerate(self.blocks):
                for _k, _v in _blk.items():
                    _out[f"b{_bi}.{_k}"] = _v
            return _out

        def forward(self, ids):
            _B, _T = ids.shape
            _c = {"ids": ids}
            _H = self.E[ids] + self.PE[:_T]
            _mask = self.mask[:_T, :_T]                          # 生成时窗口可短可长
            for _bi, _blk in enumerate(self.blocks):
                _z1, _c1 = ln_fwd44(_H)
                _Q = _z1 @ _blk["Wq"] + _blk["bq"]
                _K = _z1 @ _blk["Wk"] + _blk["bk"]
                _Vv = _z1 @ _blk["Wv"] + _blk["bv"]
                _Qh = _Q.reshape(_B, _T, self.h, self.dk).transpose(0, 2, 1, 3)
                _Kh = _K.reshape(_B, _T, self.h, self.dk).transpose(0, 2, 1, 3)
                _Vh = _Vv.reshape(_B, _T, self.h, self.dk).transpose(0, 2, 1, 3)
                _S = np.einsum("bhtd,bhud->bhtu", _Qh, _Kh) / np.sqrt(self.dk) + _mask  # ★ 掩码
                _A = softmax_rows44(_S)
                _Oh = np.einsum("bhtu,bhud->bhtd", _A, _Vh)
                _O = _Oh.transpose(0, 2, 1, 3).reshape(_B, _T, self.h * self.dk)
                _H1 = _H + (_O @ _blk["Wo"] + _blk["bo"])
                _z2, _c2 = ln_fwd44(_H1)
                _F1 = _z2 @ _blk["W1"] + _blk["b1f"]
                _G = gelu44(_F1)
                _H2 = _H1 + (_G @ _blk["W2"] + _blk["b2f"])
                _c[f"b{_bi}"] = dict(z1c=_c1, z2c=_c2, Q=_Q, K=_K, Vv=_Vv, A=_A, O=_O, F1=_F1, G=_G)
                _H = _H2
            _logits = _H @ self.Wo_head + self.bo_head           # (B, T, V) ★ 全位置预测
            _c["H"] = _H
            return _logits, _c

        def backward(self, dlogits, cache):
            """43 课全链路反向原样复用（掩码是常数，反向无额外项）。"""
            _B, _T = cache["ids"].shape
            _d = self.d
            _g = {_k: np.zeros_like(_v) for _k, _v in self.params().items()}
            _H = cache["H"]
            _g["Wo_head"] += _H.reshape(-1, _d).T @ dlogits.reshape(-1, self.V)
            _g["bo_head"] += dlogits.sum((0, 1))
            _dH = dlogits @ self.Wo_head.T
            for _bi in reversed(range(self.n_blocks)):
                _blk = self.blocks[_bi]
                _c = cache[f"b{_bi}"]
                _p = f"b{_bi}."
                _dF2 = _dH
                _g[_p+"W2"] += _c["G"].reshape(-1, self.d_ff).T @ _dF2.reshape(-1, _d)
                _g[_p+"b2f"] += _dF2.sum((0, 1))
                _dG = _dF2 @ _blk["W2"].T
                _dF1 = _dG * gelu_grad44(_c["F1"])
                _g[_p+"W1"] += _c["z2c"][1].reshape(-1, _d).T @ _dF1.reshape(-1, self.d_ff)
                _g[_p+"b1f"] += _dF1.sum((0, 1))
                _dz2 = _dF1 @ _blk["W1"].T
                _dH1 = _dH + ln_bwd44(_dz2, _c["z2c"])
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
                _dH = _dH1 + ln_bwd44(_dz1, _c["z1c"])
            np.add.at(_g["E"], cache["ids"], _dH)
            self.grads = _g
            return _g

    # 梯度校验（小配置）
    _m = MiniGPT44(11, 8, 2, 16, 2, 6, seed=0)
    _rng = np.random.default_rng(1)
    _ids = _rng.integers(0, 11, (2, 6))
    _logits, _cache = _m.forward(_ids)
    _w = _rng.normal(0, 1, _logits.shape)
    _grads = _m.backward(_w.copy(), _cache)
    _eps = 1e-6
    _gscale = max(np.abs(_g).max() for _g in _grads.values())
    _worst = 0.0
    _nchk = 0
    for _name, _p in _m.params().items():
        _flat = _p.reshape(-1)
        for _i in range(len(_flat)):
            _orig = _flat[_i]
            _flat[_i] = _orig + _eps
            _lp, _ = _m.forward(_ids)
            _flat[_i] = _orig - _eps
            _lm, _ = _m.forward(_ids)
            _flat[_i] = _orig
            _num = ((_lp * _w).sum() - (_lm * _w).sum()) / (2 * _eps)
            _ana = _grads[_name].reshape(-1)[_i]
            _worst = max(_worst, abs(_num - _ana) / _gscale)
            _nchk += 1
    print(f"梯度校验：{_nchk} 个参数位置，最大相对误差 {_worst:.0e}（判据 1e-5）")
    return (MiniGPT44, gelu44, gelu_grad44, ln_bwd44, ln_fwd44, softmax_rows44)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 4 · 训练与对决：小语料上的真相

        训练配置与 37 课对齐：同语料、同窗口采样、
        Adam + **早停**（每 25 步看验证困惑度，留最佳
        参数——17 课的老规矩，11.6 千字符的小语料
        谁也逃不掉过拟合）。

        先亮答案（本机实测，训练约 1 分钟）：

        | 模型 | 参数 | 验证困惑度 | 每步做什么 |
        |---|---|---|---|
        | RNN（37 课） | 12.5 万 | 125 | 32 步串行 |
        | LSTM（37 课） | 32.6 万 | **106** | 32 步串行 |
        | 迷你 GPT（本课） | 11.5 万 | **154** | 1 次前向 |

        **迷你 GPT 输了**——参数最少、训练最快，但
        困惑度垫底。这不是 bug，是深度学习的重要
        一课：**注意力的归纳偏置弱**。LSTM 把"信息
        沿时间传递"焊进结构（33 课），小数据上这是
        免费的先验；注意力什么结构都不假设，一切
        靠数据学——数据不够时，自由反而是负担。
        Transformer 的翻盘要等**数据规模**上来
        （45 课的正题）。
        """
    )
    return


@app.cell
def _(IDS44, MiniGPT44, N_TR44, VOCAB44, np, plt, softmax_rows44):
    # 演示 4 · 训练迷你 GPT（早停 + 对决）
    import time

    def make_window44(n, T, rng, ids):
        """随机窗口：输入 X、目标 Y（错一位）——37 课同款。"""
        _starts = rng.integers(0, len(ids) - T - 1, n)
        _X = np.stack([ids[_s:_s + T] for _s in _starts])
        _Y = np.stack([ids[_s + 1:_s + T + 1] for _s in _starts])
        return _X, _Y

    def eval_ppl44(model, ids, T, rng_seed=123, n_batch=8, batch=16):
        _rng = np.random.default_rng(rng_seed)
        _tot, _n = 0.0, 0
        for _ in range(n_batch):
            _X, _Y = make_window44(batch, T, _rng, ids)
            _logits, _ = model.forward(_X)
            _P = softmax_rows44(_logits.astype(np.float64))
            _nll = -np.log(_P[np.arange(batch)[:, None], np.arange(T)[None, :], _Y] + 1e-12)
            _tot += _nll.sum()
            _n += _nll.size
        return float(np.exp(_tot / _n))

    def train_gpt44(model, steps=450, batch=24, lr_hi=0.01, lr_lo=0.004, seed=1):
        _tr = np.random.default_rng(seed)
        _params = model.params()
        _ms = {_k: np.zeros_like(_v) for _k, _v in _params.items()}
        _vs = {_k: np.zeros_like(_v) for _k, _v in _params.items()}
        _best = (1e18, None, 0)
        _hist, _va_hist = [], []
        _t0 = time.time()
        for _step in range(1, steps + 1):
            _lr = lr_hi if _step < 300 else lr_lo
            _X, _Y = make_window44(batch, model.T, _tr, IDS44[:N_TR44])
            _logits, _cache = model.forward(_X)
            _P = softmax_rows44(_logits.astype(np.float64))
            _loss = -np.log(_P[np.arange(batch)[:, None], np.arange(model.T)[None, :], _Y]
                            + 1e-12).mean()
            _dl = _P.astype(np.float32).copy()
            _dl[np.arange(batch)[:, None], np.arange(model.T)[None, :], _Y] -= 1
            _dl /= (batch * model.T)
            _grads = model.backward(_dl, _cache)
            for _k in _params:
                _ms[_k] = 0.9 * _ms[_k] + 0.1 * _grads[_k]
                _vs[_k] = 0.999 * _vs[_k] + 0.001 * _grads[_k] * _grads[_k]
                _params[_k] -= _lr * (_ms[_k] / (1 - 0.9 ** _step)) / (
                    np.sqrt(_vs[_k] / (1 - 0.999 ** _step)) + 1e-8)
            _hist.append(_loss)
            if _step % 25 == 0:
                _vp = eval_ppl44(model, IDS44[N_TR44:], model.T, n_batch=4)
                _va_hist.append((_step, _vp))
                if _vp < _best[0]:
                    _best = (_vp, {_k: _v.copy() for _k, _v in _params.items()}, _step)
        for _k in _params:                                   # 恢复早停点
            _params[_k] = _best[1][_k]
        print(f"训练 {steps} 步用时 {time.time()-_t0:.0f}s｜早停点：验证 ppl "
              f"{_best[0]:.0f} @ step {_best[2]}")
        return np.array(_hist), dict(_va_hist), _best[0]

    gpt44 = MiniGPT44(len(VOCAB44), 48, 4, 192, 2, 32, seed=0)
    hist44, va44, best_ppl44 = train_gpt44(gpt44)
    _n_par = sum(_p.size for _p in gpt44.params().values())
    print(f"迷你 GPT 参数：{_n_par:,}")
    print(f"\n同台对决（同一份语料、同一个任务）：")
    print(f"  RNN  （37 课）：12.5 万参数，ppl 125")
    print(f"  LSTM （37 课）：32.6 万参数，ppl 106 ← 冠军")
    print(f"  GPT  （本课）：{_n_par/1e4:.1f} 万参数，ppl {best_ppl44:.0f}")

    _fig, _ax = plt.subplots(figsize=(8.8, 3.8))
    _ax.plot(hist44, lw=1.2, color="tab:blue", label="训练损失（当前批）")
    _ax.axhline(np.log(601), color="gray", ls="--", lw=1)
    _ax.text(3, np.log(601) + 0.15, "瞎猜线 ln601≈6.4", fontsize=9, color="gray")
    _ax2 = _ax.twinx()
    _ax2.plot([s for s, _ in va44.items()], [v for _, v in va44.items()],
              "o-", ms=3, color="tab:green", label="验证 ppl（右轴）")
    _ax2.set_ylabel("验证困惑度", color="tab:green")
    _ax2.axvline(350, color="tab:green", ls=":", lw=1)
    _ax.set_xlabel("训练步数")
    _ax.set_ylabel("训练损失")
    _ax.set_title("迷你 GPT：早停点 ppl 约 154——输给 LSTM 的 106，赢在训练速度", fontsize=11)
    _ax.legend(loc="upper right", fontsize=9)
    _fig.tight_layout()
    _fig
    return (best_ppl44, eval_ppl44, gpt44, make_window44)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 5 · 开麦：迷你 GPT 的生成

        用早停点参数、从"深度学习"起逐字采样（37 课
        同款流程，屏蔽◇UNK）。τ=0.7 的样本看两点：

        - **词感**：能拼出"神经网络""问题""一个"
          等高频搭配——ppl 154 的水平，比 RNN 37 课
          （ppl 125）略糙
        - **结构**：空行分段、冒号词条这些篇章骨架
          学得有模有样——注意力的功劳：写每个字时
          直接回看全文，比 LSTM 的记忆接力看得清

        生成是**串行**的：90 字 = 90 次前向。看打印
        里的耗时——这就是 GPT "打字速度"的本源
        （工业界用 KV 缓存能省大半，但串行改不掉）。
        """
    )
    return


@app.cell
def _(VOCAB44, gpt44, np):
    # 演示 5 · 自回归生成（串行）+ 温度扫描
    import time

    def sample44(model, seed_text, length, temp=0.7, rng_seed=0):
        """自回归生成：逐字前向、屏蔽 UNK。"""
        _ch2id = {c: i for i, c in enumerate(VOCAB44)}
        _rng = np.random.default_rng(rng_seed)
        _out = list(seed_text)
        _ctx = [_ch2id.get(c, 0) for c in seed_text]
        _unk = _ch2id.get("◇")
        for _ in range(length):
            _ids = np.array([_ctx[-model.T:]])
            _logits, _ = model.forward(_ids)
            _z = _logits[0, -1].astype(np.float64) / temp
            _p = np.exp(_z - _z.max())
            _p /= _p.sum()
            if _unk is not None:
                _p[_unk] = 0
                _p /= _p.sum()
            _nxt = int(_rng.choice(len(_p), p=_p))
            _out.append(VOCAB44[_nxt])
            _ctx.append(_nxt)
        return "".join(_out)

    _t0 = time.time()
    _s1 = sample44(gpt44, "深度学习", 90, 0.7)
    _gen_t = time.time() - _t0
    print(f"═" * 40, " τ=0.7 ", "═" * 40)
    print(_s1)
    print(f"（90 字生成耗时 {_gen_t:.1f}s——串行：每字一次前向）")
    print()
    for _t in (0.5, 1.0):
        print("═" * 40, f" τ={_t} ", "═" * 40)
        print(sample44(gpt44, "深度学习", 90, _t))
        print()
    return (sample44,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 6 · 交叉注意力：形状走查

        翻译机器的第三子层。看形状：解码器 5 个词、
        编码器 3 个词——注意力矩阵 **(5, 3)**：每个
        译词对每个原词打一个分。这张矩阵就是**对齐
        表**：写"爱"的那一行应该在"love"上最亮。

        一个玩具直觉（不训练）：手工造 6 维"语义"
        向量，让「爱」的查询与「love」的键同向——
        点积打分自然对齐。真实翻译模型里，这张表
        是从平行语料里学出来的。
        """
    )
    return


@app.cell
def _(np, plt, softmax_rows44):
    # 演示 6 · 交叉注意力形状走查 + 玩具对齐
    _r = np.random.default_rng(0)
    _d = 6
    # 原句（编码器输出）：I / love / cats
    _enc_words = ["I", "love", "cats"]
    _K = _r.normal(0, 1, (3, _d))
    _V = _r.normal(0, 1, (3, _d))
    # 译文（解码器）：我 / 爱 / 猫
    _dec_words = ["我", "爱", "猫"]
    _Q = _r.normal(0, 1, (3, _d))
    # 手工对齐：让「我→I」「爱→love」「猫→cats」
    _Q[0] = _K[0] + _r.normal(0, 0.1, _d)
    _Q[1] = _K[1] + _r.normal(0, 0.1, _d)
    _Q[2] = _K[2] + _r.normal(0, 0.1, _d)
    _S = _Q @ _K.T / np.sqrt(_d)
    _A = softmax_rows44(_S)
    print(f"Q（来自解码器）：{_Q.shape}｜K/V（来自编码器）：{_K.shape}")
    print(f"注意力矩阵（对齐表）：{_A.shape} —— 每个译词 × 每个原词")
    print("\n对齐表（行=译词，列=原词）：")
    print("      " + "  ".join(f"{w:>5s}" for w in _enc_words))
    for _i, _w in enumerate(_dec_words):
        print(f"{_w:>4s}  " + "  ".join(f"{_A[_i, _j]:.2f}" for _j in range(3)))

    _fig, _ax = plt.subplots(figsize=(5.2, 4.0))
    _ax.imshow(_A, cmap="Blues", vmin=0, vmax=1)
    _ax.set_xticks(range(3), _enc_words)
    _ax.set_yticks(range(3), _dec_words)
    _ax.set_title("交叉注意力 = 对齐表", fontsize=11)
    for _i in range(3):
        for _j in range(3):
            _ax.text(_j, _i, f"{_A[_i,_j]:.2f}", ha="center", va="center", fontsize=9,
                     color="white" if _A[_i, _j] > 0.5 else "black")
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
        ### 交互实验 · 迷你 GPT 开麦直播间

        37 课同款玩法，今天的主角换成迷你 GPT：给
        **开头**（过滤到词表内）、调**温度**和**长度**，
        现场逐字生成。

        玩法建议：

        - 开头用"问题"、"梯度"、"网络"这类语料高
          密度词，生成最通顺
        - τ=0.5 vs τ=1.0：低温更"背诵"（高频搭配
          循环），高温更多没见过的搭配
        - 对比 37 课的 LSTM 直播间（ppl 106 vs
          154）：差距在长句的组织力上——把生成长度
          拉到 200+ 看断句
        """
    )
    return


@app.cell
def _(mo):
    # 交互控制：开头 + 温度 + 长度
    seed_text44 = mo.ui.text(value="深度学习", label="✍️ 开头（1~8 个字）")
    temp_slider44 = mo.ui.slider(0.2, 1.4, value=0.7, step=0.05, debounce=True,
                                 label="🌡️ 温度 τ")
    len_slider44 = mo.ui.slider(40, 240, value=120, step=10, debounce=True,
                                label="📏 生成长度（字符）")
    mo.hstack([seed_text44, temp_slider44, len_slider44])
    return (len_slider44, seed_text44, temp_slider44)


@app.cell
def _(VOCAB44, gpt44, len_slider44, mo, sample44, seed_text44, temp_slider44):
    # ===== 交互实验 · 迷你 GPT 开麦直播间（响应单元）=====
    _raw = (seed_text44.value or "").strip()[:8]
    _seed = "".join(c for c in _raw if c in set(VOCAB44)) or "深度学习"
    _temp, _n = temp_slider44.value, len_slider44.value
    _text = sample44(gpt44, _seed, _n, _temp, rng_seed=0)

    _vocab_hits = sum(1 for w in ("梯度", "网络", "模型", "数据", "训练", "损失", "学习", "问题")
                      if w in _text)
    _has_para = "\n" in _text
    _verdict = (f"**τ={_temp:.2f} · {_n} 字**：高频术语命中 {_vocab_hits}/8，"
                f"{'有' if _has_para else '无'}分段结构。\n\n")
    if _temp < 0.45:
        _verdict += ("低温区：分布压尖，模型只走最有把握的路——注意高频词的复读循环。"
                     "ppl 154 的水平：比 LSTM（106）更依赖高频搭配。")
    elif _temp <= 0.95:
        _verdict += ("舒适区：词级搭配（神经网络、问题、一个）出得勤，篇章骨架"
                     "（空行分段）也时有出现——注意力全文直看的功劳。"
                     "把长度拉长对比 37 课 LSTM：断句组织力的差距在这里。")
    else:
        _verdict += ("高温区：小概率字符频繁出场，句子被带向陌生搭配——"
                     "同一个模型，三种性格（37 课温度的同一枚旋钮）。")
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
                **练习 1 · 手算掩码 softmax**：T=3，某行
                打分 [0, 2, 1]。掩码后这行变成什么？
                softmax 后前两个位置的权重各多少
                （两位小数）？

                **练习 2 · 补全掩码**：两行写出因果掩码
                ——先造一个 T×T 的全 −10⁹ 矩阵，再用
                np.triu 保留上三角（k=1，对角线不掩）。

                **练习 3 · 并行度对账**：训练一个 T=32
                的窗口，RNN 要串行跑几步？解码器几次
                前向？推理生成 100 个字，解码器又要
                几次前向？
                """
            )
        }
    )
    return


@app.cell
def _(mo, np):
    # ===== 练习 1 · 手算掩码 softmax =====
    print("T=3，某行打分 [0, 2, 1]（位置 0/1/2）")
    print("  掩码后：[0, 2, ?]  → softmax 只在前两个位置分蛋糕")

    w0_ex1_44 = None                                  # ← ① 位置 0 的权重（两位小数）
    w1_ex1_44 = None                                  # ← ② 位置 1 的权重（两位小数）

    if w0_ex1_44 is not None and w1_ex1_44 is not None:
        _s = np.array([0.0, 2.0])
        _p = np.exp(_s) / np.exp(_s).sum()
        if abs(w0_ex1_44 - _p[0]) < 0.02 and abs(w1_ex1_44 - _p[1]) < 0.02:
            _msg = (f"✅ 正确：{_p[0]:.2f} / {_p[1]:.2f}（位置 2 = e^(−10⁹) ≈ 0）。"
                    "注意位置 2 原本打分 1 分——分数不低，但掩码不谈分数只谈资格："
                    "**未来再相关也不许看**。这就是因果掩码与普通注意力的全部区别："
                    "打分函数一模一样，资格表不同。")
        else:
            _msg = "❌ 参考：0.12 / 0.88（softmax([0, 2])）"
    else:
        _msg = "⏳ 两空都填上（被掩位置的 e 值 ≈ 0）"
    mo.md(_msg)
    return (w0_ex1_44, w1_ex1_44)


@app.cell
def _(mo, np):
    # ===== 练习 2 · 补全掩码 =====
    def causal_mask_ex2_44(T):
        """返回 T×T 因果掩码：上三角（不含对角线）为 -1e9，其余 0。"""
        _m = None                                      # ← ① 补：全 -1e9 的 T×T 矩阵
        return None                                    # ← ② 补：np.triu(..., k=1)

    mask_ready_ex2_44 = None                          # ← 补完后，换成 causal_mask_ex2_44

    if mask_ready_ex2_44 is not None:
        try:
            _m = mask_ready_ex2_44(4)
            _ref = np.triu(np.full((4, 4), -1e9), k=1)
            if _m is not None and _m.shape == (4, 4) and np.allclose(_m, _ref):
                _msg = ("✅ 正确。两个细节：① k=1 让对角线保留 0（自己看自己合法）；"
                        "② 用 −1e9 而非 −inf：−inf 在某些 softmax 实现里会出 NaN"
                        "（全 −inf 的行），大负数数学上等价、数值上安全。"
                        "工程版（PyTorch）常用布尔掩码 + masked_fill，思想相同。")
            else:
                _msg = "❌ 参考：np.triu(np.full((T, T), -1e9), k=1)"
        except (TypeError, ValueError) as _e:
            _msg = f"❌ 运行出错：{_e}——np.full + np.triu(k=1)"
    else:
        _msg = "⏳ 补两行：全 -1e9 → triu 上三角"
    mo.md(_msg)
    return (causal_mask_ex2_44, mask_ready_ex2_44)


@app.cell
def _(mo):
    # ===== 练习 3 · 并行度对账 =====
    print("窗口 T=32：训练一个窗口 vs 推理生成 100 字")

    rnn_steps_ex3_44 = None                           # ← ① RNN 训练一个窗口的串行步数
    gpt_fwd_ex3_44 = None                             # ← ② 解码器训练一个窗口的前向次数
    gpt_gen_ex3_44 = None                             # ← ③ 解码器生成 100 字的前向次数

    if None not in (rnn_steps_ex3_44, gpt_fwd_ex3_44, gpt_gen_ex3_44):
        if rnn_steps_ex3_44 == 32 and gpt_fwd_ex3_44 == 1 and gpt_gen_ex3_44 == 100:
            _msg = ("✅ 正确：32 / 1 / 100。GPT 的双面性格：**训练并行**（掩码让 32 个"
                    "预测互不泄题，一次前向全出）、**推理串行**（第 t+1 字要等第 t 字"
                    "落地）。训练快 → 能吃下互联网级语料；推理慢 → 大模型\"打字\"要"
                    "一秒几个字，KV 缓存、投机解码（一次猜几个字再验证）都是给这个"
                    "瓶颈打的补丁。")
        else:
            _msg = "❌ 参考：32（串行步）/ 1（一次前向）/ 100（每字一次前向）"
    else:
        _msg = "⏳ 三空都填上"
    mo.md(_msg)
    return (gpt_fwd_ex3_44, gpt_gen_ex3_44, rnn_steps_ex3_44)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 1 参考答案（先自己写过再看！）": mo.md(
                """
                **掩码后 [0, 2, −10⁹]；softmax =
                [0.12, 0.88, ≈0]。**

                ```
                e⁰ = 1.00，e² = 7.39，e^(−10⁹) ≈ 0
                分母 = 8.39
                w₀ = 1.00/8.39 ≈ 0.12
                w₁ = 7.39/8.39 ≈ 0.88
                ```

                值得记住的画面：**掩码不是删除，是
                数学性地饿死**——未来位置在 softmax 的
                指数世界里拿不到任何粮食（e 的负十
                亿次方）。梯度同样干净：掩码是常数，
                反向传播不需要任何特殊处理。
                """
            ),
            "✅ 练习 2 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                _m = np.full((T, T), -1e9)
                return np.triu(_m, k=1)
                ```

                `np.triu(M, k=1)`：把 M 的**下三角
                （含对角线）清零**、保留 k=1 以上的
                上三角——恰好是"未来"的方向（第 t 行
                的第 t+1..T 列）。对角线（自己）不掩：
                写字时看一眼自己刚写的，合法。

                PyTorch 工程版：
                `scores.masked_fill(mask == 0, -1e9)`
                ——布尔掩码版的三行合一，思想一样。
                """
            ),
            "✅ 练习 3 参考答案（先自己写过再看！）": mo.md(
                """
                **32 / 1 / 100。**

                | 场景 | RNN | GPT | 倍差 |
                |---|---|---|---|
                | 训练一个 32 字窗口 | 32 步串行 | 1 次前向 | 32× |
                | 生成 100 字 | 100 步 | 100 次前向 | 1× |

                训练 32 倍的并行差，是 GPU 时代
                Transformer 吃下互联网语料的资格证；
                推理时两者一样串行——所以大模型回复
                你的时候像在"打字"而不是"闪现"。
                缓解手段（都只是补丁）：**KV 缓存**
                （已算过的键值不重算，省 ~T/2）、
                **投机解码**（小模型先猜几个字、大
                模型批量验证）、**连续批处理**（把
                多个用户的串行拼成一批并行）。
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
                1. **因果掩码**：上三角填 −10⁹ 叠在打分
                   上，softmax 后未来 ≈ 0——数学性抹除，
                   反向零成本（常数）
                2. **训练并行**：一个窗口一次前向出 T 个
                   预测（teacher forcing + 掩码防泄题）；
                   RNN 要 T 步串行——GPU 时代的分水岭
                3. **推理串行**：自回归生成每字一次前向
                   （KV 缓存只是补丁）——大模型"打字
                   速度"的本源
                4. **MiniGPT = 编码器 + 掩码 + 词表头**：
                   与 43 课的全部差异三行说完；全链路
                   反向原样复用，梯度校验 4×10⁻¹⁰
                5. **小语料的真相**：迷你 GPT（11.5 万
                   参数）ppl 154 输给 LSTM 106 < RNN 125——
                   注意力归纳偏置弱，数据不够时自由是
                   负担；翻盘要等规模（45 课）
                6. **交叉注意力**：Q 来自解码器、K/V
                   来自编码器——注意力矩阵即翻译对齐表；
                   GPT 弃之（纯生成）、BERT 双向（纯
                   理解）、原版翻译机全都要
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

                **下一课预告** — `45_GPT与BERT概览`（第 7 部分 · Transformer）：

                你已经亲手造过一台迷你 GPT 了。收官一课
                讲清楚**这场革命怎么发生的**：

                - **2017-2023 时间线**：Attention is All
                  You Need → GPT-1/2/3（越大越像）→
                  BERT（完形填空）→ ChatGPT（指令微调
                  + 人类反馈）
                - **三种架构、一个零件**：编码器
                  （BERT：双向理解）、解码器（GPT：
                  因果生成）、编解码器（T5/翻译机）——
                  全都是你这几课写过的块
                - **Scaling Laws**：损失随参数/数据/
                  算力的幂律下降——迷你 GPT 输给 LSTM
                  的那笔账，规模怎么把它赢回来
                - **GPT 到底"预测下一词"到了什么
                  程度**：涌现能力、上下文学习、
                  思维链——以及未解的争论

                最后一课没有代码，是一张地图：把 38-44
                课的手艺放回工业界的坐标系里。
                """
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
