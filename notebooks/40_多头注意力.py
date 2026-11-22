import marimo

__generated_with = "0.25.2"
app = marimo.App(app_title="40 · 多头注意力", width="medium")


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
        # 40 · 多头注意力：h 双眼睛看同一句话

        **从零到 Transformer · 手写深度学习** ｜ 第 7 部分 · Transformer（3/8）
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📖 本课目标（先读我）": mo.md(
                """
                **39 课的一套 QKV，一次只能问一种问题。
                这节课给注意力装上 h 双眼睛。**

                一句话里的关系是多元的：谁修饰谁、谁指代
                谁、哪个词和哪个词相邻——一种打分方式
                顾不过来。多头的想法朴素：**把 d 维空间
                切成 h 份，每份独立跑一套小 QKV**，各自
                学各自的关系，最后把 h 份输出拼接、投影
                回 d 维。

                惊人的账本：h 头（每头 d_k=d/h）的参数和
                计算量，与 1 头（d_k=d）**几乎分文不差**
                ——同样的预算，从"一个大视野"换成
                "h 个小视角"。这笔划算的买卖是
                Transformer 精度的关键来源之一。

                学完你将能：

                - 写出**多头注意力的完整实现**：切空间 →
                  逐头注意力 → 拼接 → W_o 投影，并核对
                  每一步形状
                - 亲手训练出**分工明确的两只眼**：头 A 学会
                  盯前一个词、头 B 学会盯前两个词（子对角
                  线 vs 次子对角线，热力图亲眼验证）
                - 撞上并理解**头的冷启动难题**：从头联合
                  训练多头，常常只有一个头激活、其余
                  躺平——真实大模型训练也要面对的现象
                - 用**课程学习**（分头特训 → 拼装合体）
                  稳定诱导分工

                前置：39 课 QKV 与缩放、24 课 Adam、
                22-23 课反向传播。

                本课的训练实验是第 7 部分的第一场
                "真训练"——之后的编码器（43 课）就是把
                今天的多头块原样叠上去。
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "👀 概念 1/3 · 为什么需要多双眼睛": mo.md(
                """
                单头注意力的softmax 是**一套**打分——查询
                和键只有一种"对上眼"的方式。但语言的关系
                是多元的：

                - 指代：代词找名词（"它"→"书"）
                - 句法：动词找宾语（"放进"→"箱子"）
                - 相邻：副词找它修饰的词（"很"→"重"）
                - 词频：某些词天生是"信息枢纽"

                一个打分子空间里，这些关系会**互相挤压**
                ——softmax 只能输出一种权重分布。多头把
                空间切成 h 份，等于办了 h 场**互不干扰的
                招聘会**：每场用自己的方式打分，各自选出
                各自的"相关词"，最后把 h 份情报汇总。

                类比：一张照片，一个摄影师只有一个焦距；
                h 个镜头（广角、长焦、微距……）各拍一张
                再合成——**总像素预算不变，信息维度
                翻倍**。
                """
            ),
            "✂️ 概念 2/3 · 切法与拼法：形状流水线": mo.md(
                """
                多头注意力的全部机关，就是把 39 课的流程
                跑 h 遍再缝合：

                ```
                X (T×d)
                  ├─ 头1: Q₁=XW_q¹ (T×d_k) → 注意力 → O₁ (T×d_k)
                  ├─ 头2: Q₂=XW_q² (T×d_k) → 注意力 → O₂ (T×d_k)
                  └─ …h 个头…
                拼接: concat(O₁..O_h) (T×h·d_k) = (T×d)
                投影: 输出 = concat · W_o (d×d)
                ```

                两个关键细节：

                - **W_q¹ 是 d×d_k，不是 d×d**——每头的
                  投影矩阵是"瘦"的（d_k = d/h），这是
                  "切空间"的实现方式
                - **拼接恰好凑回 d 维**（h × d/h = d），
                  W_o 再做一次混合——各头的情报在这里
                  融合，h 个视角合成一个输出

                注意 d_k 变小了：每头在更小的子空间里
                打分。39 课的 √d_k 缩放跟着用小 d_k——
                这就是"总计算量不变"的原因。
                """
            ),
            "💰 概念 3/3 · 参数与计算：一笔不亏的账": mo.md(
                """
                把账算清楚（含输出投影 W_o）：

                ```
                单头：3·(d×d) + d×d       = 4d²
                h 头：h·3·(d×d_k) + d×d    = 3h·d·(d/h) + d²
                                             = 3d² + d² = 4d²
                ```

                **一模一样。** 计算量同理：每头的打分矩阵
                是 T×T 但只算 d_k 维点积，h 个头加起来
                还是 T×T×d——与单头持平。

                | 配置 | 每头 d_k | 参数量 |
                |---|---|---|
                | 1 头，d=512 | 512 | 4×512² = 1,048,576 |
                | 8 头，d=512 | 64 | 4×512² = 1,048,576 |
                | 16 头，d=512 | 32 | 4×512² = 1,048,576 |

                GPT-2 小号：d=768、12 头、d_k=64；
                GPT-3：d=12288、96 头——**头数随规模
                线性长，d_k 恒定 64 或 128**。大模型
                用真金白银投票：视角数量比单个视角的
                分辨率更值钱。
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
        ### 演示 1 · 多头注意力的完整实现

        先把积木造出来：`multi_head_attention40` 接收
        X（T×d）和 h 组 (W_q, W_k, W_v)（各 d×d_k），
        逐头跑 39 课的缩放点积注意力，拼接后乘 W_o。
        用 T=12、d=24、h=2、d_k=12 走一遍形状流水线，
        盯住拼接那一步：两个 (12×12) 恰好缝成 (12×24)。
        """
    )
    return


@app.cell
def _(np):
    # 演示 1 · 多头注意力实现 + 形状流水线
    def softmax_rows40(S):
        """沿最后一维做数值稳定的 softmax。"""
        _E = np.exp(S - S.max(-1, keepdims=True))
        return _E / _E.sum(-1, keepdims=True)

    def multi_head_attention40(X, Wqs, Wks, Wvs, Wo):
        """多头注意力：X(T,d) → 输出(T,d)。返回 (输出, 各头注意力矩阵列表)。"""
        _d_k = Wqs[0].shape[1]
        _heads, _As = [], []
        for _Wq, _Wk, _Wv in zip(Wqs, Wks, Wvs):
            _Q, _K, _V = X @ _Wq, X @ _Wk, X @ _Wv
            _S = _Q @ _K.T / np.sqrt(_d_k)              # 每头自己的 T×T
            _A = softmax_rows40(_S)
            _As.append(_A)
            _heads.append(_A @ _V)                      # (T, d_k)
        _concat = np.concatenate(_heads, axis=-1)       # (T, h·d_k)
        return _concat @ Wo, _As                        # (T, d)

    # 形状流水线实走
    _rng = np.random.default_rng(0)
    _T, _d, _h, _d_k = 12, 24, 2, 12
    _X = _rng.normal(0, 1, (_T, _d))
    _Wqs = [_rng.normal(0, 0.2, (_d, _d_k)) for _ in range(_h)]
    _Wks = [_rng.normal(0, 0.2, (_d, _d_k)) for _ in range(_h)]
    _Wvs = [_rng.normal(0, 0.2, (_d, _d_k)) for _ in range(_h)]
    _Wo = _rng.normal(0, 0.2, (_h * _d_k, _d))
    _out, _As = multi_head_attention40(_X, _Wqs, _Wks, _Wvs, _Wo)
    print(f"输入 X：{_X.shape}（{_T} 词 × d={_d}）")
    for _i, _A in enumerate(_As):
        print(f"  头{_i+1} 注意力矩阵：{_A.shape}，输出：({_T}, {_d_k})")
    print(f"拼接：({_T}, {_h}×{_d_k}={_h*_d_k}) → 乘 W_o({_h*_d_k}, {_d})")
    print(f"最终输出：{_out.shape}（与输入同形——残差连接的前提，42 课见）")
    print(f"\n参数核对：h·3·d·d_k + d² = {_h}×3×{_d}×{_d_k} + {_d}² = "
          f"{_h*3*_d*_d_k + _d**2}；单头 4d² = {4*_d**2}")
    return (multi_head_attention40, softmax_rows40)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 2 · 双关系任务：两只眼的分工实验

        任务设计（专为逼出分工）：T=12 的序列，每个位置
        有**词身份**（8 类）和**载荷**（16 类，与词身份
        无关的随机标签）：

        - **任务 1**：预测**前一个词**的词身份 → 需要一种
          关系："盯住 t-1"（位置机制）
        - **任务 2**：预测**前两个位置**的载荷 → 需要
          另一种关系："盯住 t-2"（还是位置机制，但偏移
          不同）

        两件事一个头做不了（一个 softmax 只能给一种权重
        分布），正好逼出两只眼。

        实验分两幕：

        - **第一幕（冷启动的墙）**：两个头**从头联合
          训练**——看损失迟迟下不去、两头分工混乱
          （实测：一头半醒半睡两个关系都沾一点，
          另一头熵坍缩、彻底躺平）
        - **第二幕（课程学习）**：先让每个头**单独特训**
          各自的任务（单头单任务，39 课就证明过能学），
          再拼成双头模型——两头分工 100% 明确，
          40 个测试序列双双满分

        第一幕不是失败，是**教学**：注意力头的冷启动
        是真实大模型也要面对的优化难题（softamax 早期
        塌缩 → 梯度死区 → 头再也醒不过来）。课程学习、
        对称初始化、更小的 Q/K 初始化，都是工程师们
        的解药。
        """
    )
    return


@app.cell
def _(np):
    # 演示 2 · 数据与任务定义
    V40, T40, P40 = 8, 12, 16                    # 词表、长度、载荷表
    d40 = V40 + T40 + P40                        # 特征 = 词one-hot + 位置one-hot + 载荷one-hot

    def make_batch40(n, rng):
        """序列 + 两个任务的目标：Y1=前一词身份，Y2=前两位置载荷。"""
        _ids = rng.integers(0, V40, (n, T40))
        _pay = rng.integers(0, P40, (n, T40))
        _X = np.zeros((n, T40, d40))
        for _b in range(n):
            for _t in range(T40):
                _X[_b, _t, _ids[_b, _t]] = 1.0
                _X[_b, _t, V40 + _t] = 1.0
                _X[_b, _t, V40 + T40 + _pay[_b, _t]] = 1.0
        _Y1 = np.zeros((n, T40, V40))
        _Y2 = np.zeros((n, T40, P40))
        for _b in range(n):
            for _t in range(T40):
                _Y1[_b, _t, _ids[_b, max(_t - 1, 0)]] = 1.0
                _Y2[_b, _t, _pay[_b, max(_t - 2, 0)]] = 1.0
        return _X, _Y1, _Y2, _ids
    print(f"特征维度 d = {V40}(词) + {T40}(位置) + {P40}(载荷) = {d40}")
    print(f"任务 1：预测前一词身份（{V40} 类）｜任务 2：预测前两位置载荷（{P40} 类）")
    print(f"瞎猜线：ln{V40} + ln{P40} = {np.log(V40) + np.log(P40):.2f}")
    return (P40, T40, V40, d40, make_batch40)


@app.cell
def _(P40, T40, V40, d40, make_batch40, np, softmax_rows40):
    # 演示 2 第一幕 · 冷启动的墙：双头从头联合训练
    def train_joint40(steps=2000, seed=0, h=2, d_k=32, n=32):
        """双头从头联合训练两任务（Adam）。"""
        _rng = np.random.default_rng(seed)
        _Wqs = [_rng.normal(0, 0.3, (d40, d_k)) for _ in range(h)]
        _Wks = [_rng.normal(0, 0.3, (d40, d_k)) for _ in range(h)]
        _Wvs = [_rng.normal(0, 0.5, (d40, d_k)) for _ in range(h)]
        _Wo1 = _rng.normal(0, 0.5, (h * d_k, V40))
        _Wo2 = _rng.normal(0, 0.5, (h * d_k, P40))
        _tr = np.random.default_rng(1)
        _params = _Wqs + _Wks + _Wvs + [_Wo1, _Wo2]
        _ms = [np.zeros_like(_p) for _p in _params]
        _vs = [np.zeros_like(_p) for _p in _params]
        _losses = []
        for _step in range(1, steps + 1):
            _lr = 0.02 if _step < int(steps * 0.66) else 0.006
            _Xb, _Y1, _Y2, _ = make_batch40(n, _tr)
            _As, _HOs = [], []
            for _i in range(h):
                _Q = _Xb @ _Wqs[_i]; _K = _Xb @ _Wks[_i]; _Vv = _Xb @ _Wvs[_i]
                _S = np.einsum("ntd,nud->ntu", _Q, _K) / np.sqrt(d_k)
                _A = softmax_rows40(_S)
                _As.append(_A)
                _HOs.append(np.einsum("ntu,nud->ntd", _A, _Vv))
            _concat = np.concatenate(_HOs, -1)
            _Pm1 = np.exp(_concat @ _Wo1); _Pm1 = softmax_rows40(_Pm1)
            _Pm2 = np.exp(_concat @ _Wo2); _Pm2 = softmax_rows40(_Pm2)
            _loss = (-( _Y1 * np.log(_Pm1 + 1e-12)).sum(-1).mean()
                     -( _Y2 * np.log(_Pm2 + 1e-12)).sum(-1).mean())
            _losses.append(_loss)
            _dl1 = (_Pm1 - _Y1) / (n * T40)
            _dl2 = (_Pm2 - _Y2) / (n * T40)
            _dWo1 = _concat.reshape(-1, h * d_k).T @ _dl1.reshape(-1, V40)
            _dWo2 = _concat.reshape(-1, h * d_k).T @ _dl2.reshape(-1, P40)
            _dconcat = _dl1 @ _Wo1.T + _dl2 @ _Wo2.T
            _dheads = np.split(_dconcat, h, axis=-1)
            _grads = []
            for _i in range(h):
                _A = _As[_i]; _Vv = _Xb @ _Wvs[_i]; _dho = _dheads[_i]
                _dA = np.einsum("ntd,nud->ntu", _dho, _Vv)
                _dVv = np.einsum("ntu,ntd->nud", _A, _dho)
                _dS = _A * (_dA - (_dA * _A).sum(-1, keepdims=True)) / np.sqrt(d_k)
                _dQ = np.einsum("ntu,nud->ntd", _dS, _Xb @ _Wks[_i])
                _dK = np.einsum("ntu,ntd->nud", _dS, _Xb @ _Wqs[_i])
                _grads += [np.einsum("ntd,ntk->dk", _Xb, _dQ),
                           np.einsum("nud,nuk->dk", _Xb, _dK),
                           np.einsum("nud,nuk->dk", _Xb, _dVv)]
            _grads += [_dWo1, _dWo2]
            for _i, (_p, _g) in enumerate(zip(_params, _grads)):
                _ms[_i] = 0.9 * _ms[_i] + 0.1 * _g
                _vs[_i] = 0.999 * _vs[_i] + 0.001 * _g * _g
                _p -= _lr * (_ms[_i] / (1 - 0.9 ** _step)) / (
                    np.sqrt(_vs[_i] / (1 - 0.999 ** _step)) + 1e-8)
        return _Wqs, _Wks, _Wvs, _Wo1, _Wo2, np.array(_losses)

    Wq_joint40, Wk_joint40, Wv_joint40, _, _, L_joint40 = train_joint40()
    print(f"第一幕 · 联合训练 2000 步：loss {L_joint40[0]:.2f} → {L_joint40[-1]:.2f}"
          f"（瞎猜线 {np.log(V40) + np.log(P40):.2f}；两头都学会 ≈ 0）")
    _Xv, _, _, _ = make_batch40(1, np.random.default_rng(99))
    for _i in range(2):
        _Q = _Xv @ Wq_joint40[_i]; _K = _Xv @ Wk_joint40[_i]
        _A = softmax_rows40((_Q[0] @ _K[0].T) / np.sqrt(32))
        _a1 = np.mean([_A[_t, _t - 1] for _t in range(1, T40)])
        _a2 = np.mean([_A[_t, _t - 2] for _t in range(2, T40)])
        print(f"  头{_i+1}：看前一词 {_a1:.2f}｜看前两词 {_a2:.2f}"
              f"｜熵 {-( _A * np.log(_A + 1e-9)).sum(-1).mean():.2f}")
    print("→ 典型结局：分工没有形成——一头两头都沾（炼成混杂目光），另一头熵坍缩躺平")
    return (L_joint40, train_joint40)


@app.cell
def _(P40, T40, V40, d40, make_batch40, np, softmax_rows40):
    # 演示 2 第二幕 · 课程学习：分头特训 → 拼装合体
    def train_single40(task, steps=3000, seed=0, d_k=32, n=32):
        """单头单任务特训。task=1: 前一词身份；task=2: 前两位置载荷。"""
        _rng = np.random.default_rng(seed)
        _Wq = _rng.normal(0, 0.3, (d40, d_k))
        _Wk = _rng.normal(0, 0.3, (d40, d_k))
        _Wv = _rng.normal(0, 0.5, (d40, d_k))
        _out_dim = V40 if task == 1 else P40
        _Wo = _rng.normal(0, 0.5, (d_k, _out_dim))
        _tr = np.random.default_rng(10 + task)
        _params = [_Wq, _Wk, _Wv, _Wo]
        _ms = [np.zeros_like(_p) for _p in _params]
        _vs = [np.zeros_like(_p) for _p in _params]
        _losses = []
        for _step in range(1, steps + 1):
            _lr = 0.02 if _step < 2000 else 0.006
            _Xb, _Y1, _Y2, _ = make_batch40(n, _tr)
            _Yt = _Y1 if task == 1 else _Y2
            _Q = _Xb @ _Wq; _K = _Xb @ _Wk; _Vv = _Xb @ _Wv
            _S = np.einsum("ntd,nud->ntu", _Q, _K) / np.sqrt(d_k)
            _A = softmax_rows40(_S)
            _HO = np.einsum("ntu,nud->ntd", _A, _Vv)
            _Pm = softmax_rows40(_HO @ _Wo)
            _loss = -(_Yt * np.log(_Pm + 1e-12)).sum(-1).mean()
            _losses.append(_loss)
            _dl = (_Pm - _Yt) / (n * T40)
            _dWo = _HO.reshape(-1, d_k).T @ _dl.reshape(-1, _out_dim)
            _dho = _dl @ _Wo.T
            _dA = np.einsum("ntd,nud->ntu", _dho, _Vv)
            _dVv = np.einsum("ntu,ntd->nud", _A, _dho)
            _dS = _A * (_dA - (_dA * _A).sum(-1, keepdims=True)) / np.sqrt(d_k)
            _dQ = np.einsum("ntu,nud->ntd", _dS, _Xb @ _Wk)
            _dK = np.einsum("ntu,ntd->nud", _dS, _Xb @ _Wq)
            _grads = [np.einsum("ntd,ntk->dk", _Xb, _dQ),
                      np.einsum("nud,nuk->dk", _Xb, _dK),
                      np.einsum("nud,nuk->dk", _Xb, _dVv), _dWo]
            for _i, (_p, _g) in enumerate(zip(_params, _grads)):
                _ms[_i] = 0.9 * _ms[_i] + 0.1 * _g
                _vs[_i] = 0.999 * _vs[_i] + 0.001 * _g * _g
                _p -= _lr * (_ms[_i] / (1 - 0.9 ** _step)) / (
                    np.sqrt(_vs[_i] / (1 - 0.999 ** _step)) + 1e-8)
        return _Wq, _Wk, _Wv, _Wo, np.array(_losses)

    WqA40, WkA40, WvA40, WoA40, LA40 = train_single40(1)      # 头A：前一词
    WqB40, WkB40, WvB40, WoB40, LB40 = train_single40(2)      # 头B：前两词载荷
    print(f"第二幕 · 分头特训：任务1 loss {LA40[0]:.2f} → {LA40[-1]:.4f}"
          f"｜任务2 loss {LB40[0]:.2f} → {LB40[-1]:.4f}")
    heads40 = {"A": (WqA40, WkA40, WvA40, WoA40), "B": (WqB40, WkB40, WvB40, WoB40)}
    return (LA40, LB40, heads40, train_single40)


@app.cell
def _(P40, T40, V40, heads40, make_batch40, np, plt, softmax_rows40):
    # 演示 2 第二幕（续） · 合体评估：两头热力图 + 40 序列成绩
    WqA, WkA, WvA, WoA = heads40["A"]
    WqB, WkB, WvB, WoB = heads40["B"]

    Xshow40, _, _, _ = make_batch40(1, np.random.default_rng(99))
    A1show40 = softmax_rows40(((Xshow40 @ WqA)[0] @ (Xshow40 @ WkA)[0].T) / np.sqrt(32))
    A2show40 = softmax_rows40(((Xshow40 @ WqB)[0] @ (Xshow40 @ WkB)[0].T) / np.sqrt(32))

    _a1 = np.mean([A1show40[_t, _t - 1] for _t in range(1, T40)])
    _a2 = np.mean([A2show40[_t, _t - 2] for _t in range(2, T40)])
    print(f"头 A（特训任务1）：看前一词均值 {_a1:.2f}（子对角线）")
    print(f"头 B（特训任务2）：看前两词均值 {_a2:.2f}（次子对角线）")

    # 40 个测试序列的双任务成绩
    _ok1 = _ok2 = _n = 0
    for _s in range(10):
        _Xb, _Y1, _Y2, _ = make_batch40(4, np.random.default_rng(1000 + _s))
        _A1 = softmax_rows40(np.einsum("ntd,nud->ntu", _Xb @ WqA, _Xb @ WkA) / np.sqrt(32))
        _O1 = np.einsum("ntu,nud->ntd", _A1, _Xb @ WvA) @ WoA
        _A2 = softmax_rows40(np.einsum("ntd,nud->ntu", _Xb @ WqB, _Xb @ WkB) / np.sqrt(32))
        _O2 = np.einsum("ntu,nud->ntd", _A2, _Xb @ WvB) @ WoB
        _ok1 += (_O1.argmax(-1) == _Y1.argmax(-1)).sum()
        _ok2 += (_O2.argmax(-1) == _Y2.argmax(-1)).sum()
        _n += _Y1[:, :, 0].size
    print(f"40 个测试序列：任务1 准确率 {_ok1 / _n:.1%}｜任务2 准确率 {_ok2 / _n:.1%}")

    _fig, _axes = plt.subplots(1, 2, figsize=(11.6, 4.7))
    for _ax, (_A, _ttl) in zip(_axes, [
            (A1show40, "头 A：学会盯「前一个词」（子对角线发亮）"),
            (A2show40, "头 B：学会盯「前两个词」（次子对角线发亮）")]):
        _im = _ax.imshow(_A, cmap="Blues", vmin=0, vmax=1)
        _ax.set_xlabel("被看的位置（键）")
        _ax.set_ylabel("看的位置（查询）")
        _ax.set_title(_ttl, fontsize=11)
        _fig.colorbar(_im, ax=_ax, shrink=0.75)
    _fig.suptitle("两只眼的分工：同样的序列，不同的目光（课程学习诱导）", fontsize=12)
    _fig.tight_layout()
    _fig
    return (A1show40, A2show40, WqA, WqB, WkA, WkB, WvA, WvB, WoA, WoB, Xshow40)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 3 · 头数扫描：预算不变，视角变多

        固定 d=512，扫 h = 1/2/4/8/16：每头 d_k = d/h，
        参数恒为 4d² ≈ 105 万。看表格里那条不变的参数
        列——**多头是重新分配，不是加预算**。顺带看真实
        模型的选型：GPT-2 用 12 头、GPT-3 用 96 头，
        d_k 都停在 64~128——视角数随规模涨，单视角
        分辨率基本不动。
        """
    )
    return


@app.cell
def _(np, plt):
    # 演示 3 · 头数扫描表
    _d = 512
    _hs = [1, 2, 4, 8, 12, 16, 96]
    print(f"{'头数 h':>6} | {'每头 d_k':>8} | {'参数量':>12} | 备注")
    print("-" * 52)
    for _h in _hs:
        _dk = _d // _h if _d % _h == 0 else _d / _h
        _n = 3 * _h * _d * _d // _h + _d ** 2
        _note = {1: "单头：一个大视野", 8: "BERT-base 的选择",
                 12: "GPT-2 的头数（它 d=768 → d_k=64）",
                 96: "GPT-3 的头数（它 d=12288 → d_k=128）"}.get(_h, "")
        print(f"{_h:>6} | {_dk:>8.0f} | {_n:>12,} | {_note}")

    _fig, _ax = plt.subplots(figsize=(8.4, 3.5))
    _ax.plot(_hs, [4 * _d ** 2] * len(_hs), "o-", color="tab:green")
    _ax.set_xscale("log", base=2)
    _ax.set_xticks(_hs, [str(h) for h in _hs])
    _ax.set_xlabel("头数 h")
    _ax.set_ylabel("总参数量")
    _ax.set_title(f"参数预算恒定：d={_d} 时无论几头都是 4d² ≈ {4*_d**2/1e4:.0f} 万", fontsize=11)
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
        ### 交互实验 · 头部检查室

        演示 2 训练好的两只眼摆在这里，你选看哪只：
        热力图 + 三个统计（盯 t-1 的均值、盯 t-2 的
        均值、分布熵）。

        玩法建议：

        - 头 A：子对角线（偏 -1）一条亮带——**前一词头**
        - 头 B：次子对角线（偏 -2）一条亮带——**前两词头**
        - 换不同测试序列（换种子），分工依旧——说明
          学到的是**关系**，不是某个序列的背书
        - 盯住热力图的"看自己"格子（对角线）：两头都
          不看自己——39 课的自恋问题，训练解决了
        """
    )
    return


@app.cell
def _(mo):
    # 交互控制：选头 + 测试序列种子
    head_dd40 = mo.ui.dropdown(options=["头 A（盯前一词）", "头 B（盯前两词）"],
                               value="头 A（盯前一词）", label="👁️ 看哪只眼")
    seed_slider40 = mo.ui.slider(0, 20, value=0, step=1, debounce=True,
                                 label="🎲 测试序列种子（换一句看看）")
    mo.hstack([head_dd40, seed_slider40])
    return (head_dd40, seed_slider40)


@app.cell
def _(A1show40, A2show40, WqA, WqB, head_dd40, make_batch40, mo, np, plt,
      seed_slider40, softmax_rows40, WkA, WkB):
    # ===== 交互实验 · 头部检查室（响应单元）=====
    _is_a = "A" in head_dd40.value
    _seed = seed_slider40.value
    _Xb, _, _, _ = make_batch40(1, np.random.default_rng(80 + _seed))
    _Wq, _Wk = (WqA, WkA) if _is_a else (WqB, WkB)
    _A = softmax_rows40(((_Xb @ _Wq)[0] @ (_Xb @ _Wk)[0].T) / np.sqrt(32))

    _a1 = np.mean([_A[_t, _t - 1] for _t in range(1, 12)])
    _a2 = np.mean([_A[_t, _t - 2] for _t in range(2, 12)])
    _diag = np.mean(np.diag(_A))
    _ent = -(_A * np.log(_A + 1e-9)).sum(-1).mean()

    _fig, _ax = plt.subplots(figsize=(6.8, 5.4))
    _im = _ax.imshow(_A, cmap="Blues", vmin=0, vmax=1)
    _ax.set_xlabel("被看的位置（键）")
    _ax.set_ylabel("看的位置（查询）")
    _ax.set_title(f"{head_dd40.value} · 测试序列种子 {80 + _seed}", fontsize=11)
    _fig.colorbar(_im, shrink=0.8)
    _fig.tight_layout()

    if _is_a:
        _verdict = (f"**头 A 的检查报告**：盯前一词 {_a1:.2f}（目标关系）｜"
                    f"盯前两词 {_a2:.2f}｜看自己 {_diag:.2f}｜分布熵 {_ent:.2f}。\n\n"
                    "子对角线一条亮带，换多少个序列都在——它学到的是**「前一词」这个关系**，"
                    "不是背下了某个序列。真实 GPT 里被点名的 previous-token head，"
                    "就是这条带子。")
    else:
        _verdict = (f"**头 B 的检查报告**：盯前两词 {_a2:.2f}（目标关系）｜"
                    f"盯前一词 {_a1:.2f}｜看自己 {_diag:.2f}｜分布熵 {_ent:.2f}。\n\n"
                    "亮带整体向右下平移了一格——同样的机制、不同的偏移。"
                    "两头加起来，模型同时握有 t-1 和 t-2 两条信息通道："
                    "这就是「多双眼睛」的含义。")
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
                **练习 1 · 切空间**：d=512、h=8，每头的
                d_k 是多少？每头 W_q 的形状和参数量呢？

                **练习 2 · 补全拼接**：多头注意力的收尾
                两步——输入各头输出列表 `heads`（h 个
                (t, d_k)）和输出投影 `Wo`，返回最终输出
                (t, d)。

                **练习 3 · 数总账**：d=512 的完整多头层
                （含 W_o）参数总量是多少（整数）？
                h 从 1 改成 8，这个数变不变？
                """
            )
        }
    )
    return


@app.cell
def _(mo):
    # ===== 练习 1 · 切空间 =====
    print("d = 512，h = 8 头")

    dk_ex1_40 = None                                  # ← ① 每头的 d_k（整数）
    wq_params_ex1_40 = None                           # ← ② 每头 W_q 的参数量（整数）

    if dk_ex1_40 is not None and wq_params_ex1_40 is not None:
        _dk, _n = 64, 512 * 64
        if abs(dk_ex1_40 - _dk) < 1 and abs(wq_params_ex1_40 - _n) < 100:
            _msg = (f"✅ 正确：d_k = 512/8 = {_dk}，W_q 是 (512, {_dk})、参数 "
                    f"{_n:,}。注意每头的投影矩阵是「瘦」的 d×d_k——8 个瘦矩阵"
                    "拼起来恰好等于一个方阵的预算。这就是「切空间」的"
                    "字面意思：512 维被切成 8 条 64 维的子空间，每头在自己的"
                    "巷子里打分。")
        else:
            _msg = f"❌ 参考：d_k = 64（d/h）；W_q 参数 = 512×64 = 32,768"
    else:
        _msg = "⏳ 两空都填上（d_k = d/h）"
    mo.md(_msg)
    return (dk_ex1_40, wq_params_ex1_40)


@app.cell
def _(mo, np):
    # ===== 练习 2 · 补全拼接 =====
    def merge_heads_ex2_40(heads, Wo):
        """heads 是 h 个 (t, d_k) 的列表；Wo 是 (h·d_k, d)。返回 (t, d)。"""
        _concat = None                                # ← ① 补：沿最后一维拼接
        return None                                   # ← ② 补：投影回 d 维

    merge_ready_ex2_40 = None                         # ← 补完后，换成 merge_heads_ex2_40

    if merge_ready_ex2_40 is not None:
        try:
            _h1 = np.ones((3, 4)); _h2 = np.zeros((3, 4))
            _Wo = np.eye(8)[:, :5]
            _out = merge_ready_ex2_40([_h1, _h2], _Wo)
            _ref = np.concatenate([_h1, _h2], -1) @ _Wo
            if _out is not None and _out.shape == (3, 5) and np.allclose(_out, _ref):
                _msg = ("✅ 正确。np.concatenate(heads, axis=-1) 把 h 份 (t, d_k) "
                        "缝成 (t, h·d_k)；再乘 Wo 回到 (t, d)。拼接顺序 = 头的顺序，"
                        " Wo 的列与头的对应关系就在这里建立——练习里 1 和 0 两段"
                        "拼接后乘单位阵，输出前 4 列全是 1、后 4 列全是 0，"
                        "各头的信息各走各的通道。")
            else:
                _msg = "❌ 输出应为 (3, 5)（concat 后 (3,8) 乘 (8,5)）"
        except (TypeError, ValueError) as _e:
            _msg = f"❌ 运行出错：{_e}——np.concatenate(heads, axis=-1) @ Wo"
    else:
        _msg = "⏳ 补两行：拼接 → 投影"
    mo.md(_msg)
    return (merge_heads_ex2_40, merge_ready_ex2_40)


@app.cell
def _(mo):
    # ===== 练习 3 · 数总账 =====
    print("d = 512 的完整多头注意力层（W_q/W_k/W_v/W_o 全算上）")

    total_params_ex3_40 = None                        # ← ① 总参数量（整数）
    change_ex3_40 = None                              # ← ② h 从 1 → 8，总参数变不变（0=不变，1=变）

    if total_params_ex3_40 is not None and change_ex3_40 is not None:
        _n = 4 * 512 ** 2
        if abs(total_params_ex3_40 - _n) < 1000 and change_ex3_40 == 0:
            _msg = (f"✅ 正确：4 × 512² = {_n:,} ≈ 105 万，且与 h 无关——"
                    "3h·d·(d/h) + d² = 4d² 是恒等式。多头的本质是把同样的预算"
                    "从「一个大视角」重新分配成「h 个小视角」：不加钱、换风景。"
                    "GPT-3 把 d 涨到 12288、头数加到 96，d_k 仍是 128——"
                    "视角数量是随规模增长的主要维度。")
        elif abs(total_params_ex3_40 - _n) < 1000:
            _msg = "❌ 参数量对，但 h 不影响总参数（h·d/h=d 消掉了）——填 0"
        else:
            _msg = f"❌ 参考：4 × 512² = {_n:,}；不变（填 0）"
    else:
        _msg = "⏳ 两空都填上（提示：3d² + d²）"
    mo.md(_msg)
    return (change_ex3_40, total_params_ex3_40)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 1 参考答案（先自己写过再看！）": mo.md(
                """
                **d_k = 64；W_q 是 (512, 64)，参数 32,768。**

                验算：8 头 × 3 矩阵 × 32,768 = 786,432 =
                3d²，加上 W_o 的 512² = 262,144，合计
                4d² = 1,048,576——与单头分毫不差。

                一个容易搞错的点：**W_q 不是 (512, 512)
                再切片**，而是直接就是 (512, 64) 的瘦矩阵。
                "切空间"发生在矩阵形状上，不是先算完整
                的再切——后者要多付 8 倍计算。真实实现
                （PyTorch 的 nn.MultiheadAttention）把
                8 个瘦矩阵拼成一个大 (512, 512) 一次矩阵
                乘，然后 reshape 成 (8, T, 64)——数学
                等价，BLAS 效率更高。
                """
            ),
            "✅ 练习 2 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                _concat = np.concatenate(heads, axis=-1)   # (t, h·d_k)
                return _concat @ Wo                        # (t, d)
                ```

                拼接是多头与单头在代码上的唯一区别：
                39 课的单头输出直接就是 (t, d)；多头把
                h 份 (t, d_k) 摞在特征维上。W_o 的职责
                是**融合**——各头情报在这里加权合流，
                顺便把维度从 h·d_k 拉回 d（为了残差
                连接，42 课）。工程实现里更常见的是
                reshape 大法：一个 (T, d) 矩阵直接
                reshape 成 (T, h, d_k) 再 transpose——
                免去 Python 循环，结果等价。
                """
            ),
            "✅ 练习 3 参考答案（先自己写过再看！）": mo.md(
                """
                **4 × 512² = 1,048,576 ≈ 105 万；不变。**

                | 配置 | 计算 | 结果 |
                |---|---|---|
                | 1 头 | 3·512² + 512² | 1,048,576 |
                | 8 头 | 8·3·512·64 + 512² | 1,048,576 |
                | 任意 h | 3h·d·(d/h) + d² | **4d²** |

                消掉的正是那个 h：每头变瘦 h 倍，头数
                变多 h 倍，乘积恒定。所以"加头"是免费的
                结构变换——不免费的是**视角的粒度**：
                d_k=64 的点积能容纳的"关系复杂度"有
                上限。什么时候加头、什么时候加维度，
                是架构师的真实权衡（GPT 系列的回答：
                每加大约 6 倍 d，头数翻 4 倍，d_k 缓慢
                增长）。
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
                1. **多头的动机**：一套 QKV 一种打分，
                   语言的关系是多元的——h 个头 = h 场
                   互不干扰的招聘会
                2. **切法**：每头 W_q 是 d×d_k 的瘦矩阵
                   （d_k=d/h），各自打分各自汇总，拼接
                   (T, h·d_k) 后乘 W_o 融合回 d 维
                3. **账本**：3h·d·(d/h)+d² = 4d²——
                   多头重新分配预算，不增加预算；
                   d=512 无论几头都是 105 万参数
                4. **分工可学**：双关系任务（盯 t-1 的
                   词身份 / 盯 t-2 的载荷）上，分头特训
                   的两头各自锁定自己的关系（0.93+），
                   合体后 40 序列双满分
                5. **冷启动难题**：从头联合训练常常分工
                   混乱（一头混杂、一头躺平——softmax 早期
                   塌缩 → 梯度死区）——真实大模型训练也面对；
                   解药：课程学习、对称初始化、小 Q/K 初始化
                6. **真实选型**：GPT-2 12 头、GPT-3 96 头，
                   d_k 稳定在 64~128——视角数随规模涨，
                   单视角分辨率基本不动
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

                **下一课预告** — `41_位置编码`（第 7 部分 · Transformer）：

                今天的热力图里藏着一位无名英雄：每个位置
                的 one-hot。没有它，"看前一个词"根本无从
                谈起——注意力是**集合**操作，天生不知道
                "顺序"为何物：把一句话打乱，注意力的输出
                只是把各词的结果相应重排，一丁点都不觉得
                别扭。但语言显然在乎顺序！

                下一课给每个位置发一张**身份证**：正弦
                位置编码——不同频率的波叠加，让"位置 3"
                和"位置 5"的相似度只取决于距离 2。你会
                推导它最漂亮的性质：**相对位置只需一次
                点积就能读出**，还能外推到训练时没见过
                的长度。今天的 one-hot 位置是最朴素的
                方案，正弦编码是原论文的选择，而旋转
                位置编码（RoPE）是今天大模型的标配——
                三代身份证，一课讲完。
                """
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
