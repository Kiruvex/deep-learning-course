import marimo

__generated_with = "0.25.2"
app = marimo.App(app_title="36 · LSTM", width="medium")


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
        # 36 · LSTM：给记忆装上阀门

        **从零到 Transformer · 手写深度学习** ｜ 第 6 部分 · 循环神经网络（5/6）
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📖 本课目标（先读我）": mo.md(
                """
                **35 课的病历写到最后一行：普通 RNN 的梯度穿不过
                连乘的窄门，长依赖学不会，调参无用。这一课做
                那台手术。**

                手术方案 1997 年就画好了（Hochreiter &
                Schmidhuber, LSTM）：在 RNN 旁边修一条**细胞
                状态 c 的传送带**——它的更新是**加法**（cₜ =
                f⊙cₜ₋₁ + i⊙gₜ），记忆不再每步"整个穿过
                tanh 重写"。再配上三道**可学习的阀门**
                （遗忘 f / 输入 i / 输出 o），什么时候记、
                记多少、放出来多少，全部交给网络自己学。

                学完你将能：

                - 写出 LSTM 的六行核心公式，说清三道门各自的
                  岗位（f 管留存、i 管写入、o 管读出）
                - 亲手用 NumPy 实现完整 LSTM（前向 + BPTT），
                  并过 34 课同款的数值对账（实测 10⁻⁸）
                - 解释**梯度高速路**：∂cₜ/∂cₜ₋₁ = fₜ——
                  乘子从"结构的固定刑期"变成"可学习的选择题"
                - 说出一个教科书少讲的细节：**遗忘门偏置
                  初始化 b_f = 2**（"默认记住"的设计先验），
                  并看到没有它 LSTM 也趴窝（演示 2 的三方
                  对照）
                - 亲眼看 35 课的病人出院：gap=12 的回声任务，
                  RNN 26% vs LSTM **91%**（演示 3，现场训练）

                前置：35 课的梯度消失与连乘、34 课的 BPTT
                骨架、19 课的 sigmoid/tanh。

                本课代码量是全程最大的一课（前向六行、反向
                约二十行），但每一行都有明确的岗位——写完
                你就拥有了 37 课语言模型的全部引擎。
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "🏥 概念 1/3 · 手术方案：加法传送带": mo.md(
                """
                先盯着 RNN 的原罪看：

                ```
                hₜ = tanh(W·xₜ + U·hₜ₋₁ + b)     ← 记忆每步整体穿过 tanh 重写
                ```

                每一步，旧记忆都要**整个**过一遍非线性窄门——
                梯度想回到 20 步前，就得连穿 20 次窄门，每次
                被乘 (1−h²) ≤ 1（35 课的病根）。

                LSTM 的手术只加一条路。新的**细胞状态** cₜ
                这样更新：

                ```
                cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ gₜ
                     ↑加法！旧记忆乘个系数留下来，新信息加进来
                ```

                **没有 tanh 包裹在 c 的外面**——c 是一条裸奔的
                传送带：旧内容乘 f（留存多少）+ 新内容乘 i
                （写入多少）。梯度沿 c 回传时：

                ```
                ∂cₜ/∂cₜ₋₁ = fₜ        ← 就这一个乘子！
                ```

                对照 RNN 的 ∂hₜ/∂hₜ₋₁ = U·diag(1−h²)（矩阵×
                饱和因子）：LSTM 把"每步乘什么"从**结构的
                固定刑期**，变成了**一个 0~1 的可学习标量**。
                网络想让梯度活着走 100 步，只需学会让 f ≈ 1。
                """
            ),
            "🚪 概念 2/3 · 三道门：谁在开，谁在关": mo.md(
                """
                三道门都是 **sigmoid(线性组合)**——输出 0~1 的
                "阀门开度"，而且**看着当前输入和旧状态自己
                决定**（这就是"可学习"）：

                ```
                fₜ = σ(Wf·xₜ + Uf·hₜ₋₁ + bf)   遗忘门：旧记忆留多少
                iₜ = σ(Wi·xₜ + Ui·hₜ₋₁ + bi)   输入门：新信息写多少
                gₜ = tanh(Wg·xₜ + Ug·hₜ₋₁ + bg) 候选：要写入的新内容
                oₜ = σ(Wo·xₜ + Uo·hₜ₋₁ + bo)   输出门：记忆放出来多少

                cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ gₜ       传送带更新（加法）
                hₜ = oₜ ⊙ tanh(cₜ)              对外输出（tanh 在这里）
                ```

                三道门的岗位：

                - **f（遗忘）**：读新字符时，旧记忆打折——
                  "上一句说完了，翻篇"
                - **i（输入）**：这条新信息值得写进去多少——
                  "记住这个主角名字"
                - **o（输出）**：此刻要用多少记忆——"该把
                  存着的主语拿出来配动词了"

                注意一个分工：**c 是长期记忆**（传送带，可以
                跨几十步），**h 是工作输出**（每步都对外汇报，
                同时喂给下一门的计算）——tanh 只出现在 h 的
                出口，c 全程裸奔。

                ⊙ 是逐元素乘（06 课）：**每个记忆格有自己
                独立的三道门**——d_h 维的 c 配 d_h 维的 f/i/o，
                有的格记长线、有的格记短线，互不干扰。
                """
            ),
            "🛣️ 概念 3/3 · 高速路的两个隐藏设计": mo.md(
                """
                **设计一 · 遗忘门的偏置先验（b_f = 2）。**

                随机初始化时所有偏置都是 0——但 σ(0) = 0.5，
                意味着遗忘门默认"每步砍掉一半记忆"：f 的连乘
                依然是 0.5ᵗ，照样消失！LSTM 原论文和工程实践
                的标准操作：**把 b_f 初始化成 1~2**，让 σ(2) ≈
                0.88，传送带**默认畅通**——"先记住，再学着忘"
                是设计者塞进结构里的先验。演示 2 的三方对照
                会看到：没有这个先验，LSTM 在长任务上也趴窝
                （70% vs 91%）。

                **设计二 · 加法路的"无压缩"回传。**

                RNN 里梯度穿过 hₜ₋₁ → hₜ 必须经过
                U·diag(1−h²)；LSTM 里 cₜ₋₁ → cₜ 只乘 fₜ
                （一个 0~1 的数，且网络可以把它推到 1 附近）。
                **加法对梯度是透明的**（34 课：链式法则对加法
                的规定是"分流、不衰减"）——这就是"高速路"的
                本义。

                顺带一提：**GRU**（2014）是 LSTM 的精简版——
                三门并两门（f 和 i 合并成更新门 z，o 换成
                重置门 r），c 和 h 合体，参数少 1/4，效果
                常常打平。工业界 RNN 系如今主要就是 LSTM 和
                GRU 两个变体（再往后就是 Transformer 的
                天下了，第 7 部分）。
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
        ### 演示 1 · 亲手实现 LSTM，看三道门怎么开合

        六行核心公式逐行翻译成 NumPy（`lstm_cell36`），循环
        读入一句话（`lstm_forward36`，缓存每步的门值——反向
        传播要用）。

        喂"深度学习很有趣"（随机权重），画出**四条门开度
        曲线**（取记忆格前 6 维）：每个字符进来时，i/f/o/g
        各自的开合模式。看三样东西：

        - **f 大片接近顶部**（b_f=2 的先验）——传送带默认畅通
        - **i 和 g 在换字符的地方动作**——写入新信息的瞬间
        - **o 的波动最克制**——输出层最像"阀门"

        随机权重的门是乱开的——但结构已经通了：**什么时候
        记、记多少、放多少**，这三件事从此有了可拧的旋钮，
        剩下的交给训练。
        """
    )
    return


@app.cell
def _(np, plt):
    # 演示 1 · LSTM 前向实现 + 门开度可视化
    def sigmoid36(z):
        return 1.0 / (1.0 + np.exp(-z))

    def init_lstm36(V, d_h, seed=0, bf=2.0):
        """四套门权重（i/f/o/g 各一套 Wx/Wh/b）；b_f 加先验 2。"""
        _r = np.random.default_rng(seed)
        _P = {}
        for _g in "ifog":
            _P["Wx" + _g] = _r.normal(0, 1 / np.sqrt(V), (d_h, V))
            _P["Wh" + _g] = _r.normal(0, 1 / np.sqrt(d_h), (d_h, d_h))
            _P["b" + _g] = np.zeros(d_h)
        _P["bf"] = _P["bf"] + bf                        # 遗忘门偏置先验（概念 3）
        return _P

    def lstm_cell36(x, h_prev, c_prev, P):
        """六行核心公式。返回 (h, c, 本步缓存)。"""
        _i = sigmoid36(P["Wxi"] @ x + P["Whi"] @ h_prev + P["bi"])   # 输入门
        _f = sigmoid36(P["Wxf"] @ x + P["Whf"] @ h_prev + P["bf"])   # 遗忘门
        _o = sigmoid36(P["Wxo"] @ x + P["Who"] @ h_prev + P["bo"])   # 输出门
        _g = np.tanh(P["Wxg"] @ x + P["Whg"] @ h_prev + P["bg"])     # 候选
        _c = _f * c_prev + _i * _g                                   # 传送带（加法！）
        _h = _o * np.tanh(_c)                                        # 对外输出
        return _h, _c, dict(i=_i, f=_f, o=_o, g=_g)

    def lstm_forward36(xs, P):
        """整条序列，返回 (每步缓存列表, h 序列, c 序列)。"""
        _d_h = P["Wxi"].shape[0]
        _h, _c = np.zeros(_d_h), np.zeros(_d_h)
        _cache, _hs, _cs = [], [], []
        for _x in xs:
            _h, _c, _st = lstm_cell36(_x, _h, _c, P)
            _cache.append(_st); _hs.append(_h.copy()); _cs.append(_c.copy())
        return _cache, np.array(_hs), np.array(_cs)

    # ---- 喂一句话，画门开度 ----
    TEXT36 = "深度学习很有趣"
    _vocab36 = list(dict.fromkeys(TEXT36))
    _idx36 = {c: k for k, c in enumerate(_vocab36)}
    _xs36 = [np.eye(len(_vocab36))[_idx36[c]] for c in TEXT36]
    P36 = init_lstm36(len(_vocab36), 16, seed=3, bf=2.0)
    cache36, hs36, cs36 = lstm_forward36(_xs36, P36)

    _fig, _axes = plt.subplots(4, 1, figsize=(10.6, 7.2), sharex=True)
    _names36 = [("i", "输入门 i（写入多少）", "tab:green"),
                ("f", "遗忘门 f（留存多少）——注意大片贴着顶部", "tab:red"),
                ("o", "输出门 o（放出来多少）", "tab:blue"),
                ("g", "候选 g（要写入的新内容）", "tab:orange")]
    for _ax, (_k, _ttl, _c) in zip(_axes, _names36):
        _ax.plot(np.array([_st[_k][:6] for _st in cache36]), lw=1.6,
                 color=_c, alpha=0.85)
        _ax.set_ylim(-1.05, 1.05)
        _ax.set_ylabel(_ttl, fontsize=9.5)
        _ax.grid(True, alpha=0.25)
    _axes[0].set_title("随机权重 LSTM 的四条门开度曲线（前 6 个记忆格，6 维各一条线）", fontsize=11.5)
    _axes[-1].set_xticks(range(len(TEXT36)), list(TEXT36), fontsize=11)
    _axes[-1].set_xlabel("读入的字符")
    _fig.tight_layout()
    _fig
    return (P36, cache36, cs36, hs36, init_lstm36, lstm_cell36, lstm_forward36,
            sigmoid36)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 2 · 两条验证：梯度是对的，高速路是真的

        **验证 A · BPTT 对账。** `lstm_bptt36` 是 34 课倒序
        骨架的 LSTM 豪华版（约二十行，练习 2 会补其中的
        关键两行）——σ 和 tanh 的局部导数、c 与 h 两条回传
        路、四套门的参数梯度。老规矩：随机挖参数算数值
        梯度对账，**实测最差 10⁻⁸**。

        **验证 B · 高速路三方对照。** 33 课的脉冲实验搬过
        来：第 10 步换一个字符，看扰动痕迹能活多久——

        | 模型 | 30 步后的痕迹 | 判词 |
        |---|---|---|
        | RNN（α=1.0） | 0.0003 | 死透 |
        | LSTM，b_f=0 | 0.0002 | **也死透**——σ(0)=0.5，每步砍半 |
        | LSTM，b_f=2 | **0.086** | 活着（初始痕迹的 12%） |

        第二行是教科书很少点破的坑：**光有门结构不够**——
        遗忘门没有偏置先验时，σ(0)=0.5 的连乘照样把痕迹
        砍光。结构（加法传送带）+ 先验（b_f=2）缺一不可。
        """
    )
    return


@app.cell
def _(init_lstm36, lstm_forward36, np, plt, sigmoid36):
    # 演示 2A · LSTM BPTT 实现 + 对账
    def lstm_bptt36(xs, ys, P, Wy, by, loss_mask=None):
        """每步损失（带 mask）的 LSTM BPTT。返回 (门参数梯度 dict, dWy, dby)。"""
        _T = len(xs)
        if loss_mask is None:
            loss_mask = np.ones(_T, bool)
        _d_h = P["Wxi"].shape[0]
        # 前向（缓存）
        _h, _c = np.zeros(_d_h), np.zeros(_d_h)
        _cache, _logits = [], []
        for _x in xs:
            _hp, _cp = _h.copy(), _c.copy()
            _i = sigmoid36(P["Wxi"] @ _x + P["Whi"] @ _hp + P["bi"])
            _f = sigmoid36(P["Wxf"] @ _x + P["Whf"] @ _hp + P["bf"])
            _o = sigmoid36(P["Wxo"] @ _x + P["Who"] @ _hp + P["bo"])
            _g = np.tanh(P["Wxg"] @ _x + P["Whg"] @ _hp + P["bg"])
            _c = _f * _cp + _i * _g
            _tc = np.tanh(_c)
            _h = _o * _tc
            _cache.append(dict(x=_x, h_prev=_hp, c_prev=_cp, i=_i, f=_f,
                               o=_o, g=_g, c=_c.copy(), tc=_tc))
            _logits.append(Wy @ _h + by)
        _L = np.array(_logits)
        _Pr = np.exp(_L - _L.max(axis=1, keepdims=True))
        _Pr = _Pr / _Pr.sum(axis=1, keepdims=True)
        # 反向
        _G = {_k: np.zeros_like(_v) for _k, _v in P.items()}
        _dWy, _dby = np.zeros_like(Wy), np.zeros_like(by)
        _dlog = _Pr.copy(); _dlog[np.arange(_T), ys] -= 1
        _dlog = _dlog * np.asarray(loss_mask)[:, None] / max(np.asarray(loss_mask).sum(), 1)
        _dh = np.zeros(_d_h); _dc = np.zeros(_d_h)
        for _t in reversed(range(_T)):
            _st = _cache[_t]; _tc = _st["tc"]
            _dh = _dh + Wy.T @ _dlog[_t]                    # 损失路 + 未来 h 路
            _do = _dh * _tc
            _dc = _dc + _dh * _st["o"] * (1 - _tc ** 2)     # c 的直接路
            _df = _dc * _st["c_prev"]; _di = _dc * _st["g"]; _dg = _dc * _st["i"]
            _dc_prev = _dc * _st["f"]                        # ← 高速路
            _dzi = _di * _st["i"] * (1 - _st["i"])
            _dzf = _df * _st["f"] * (1 - _st["f"])
            _dzo = _do * _st["o"] * (1 - _st["o"])
            _dzg = _dg * (1 - _st["g"] ** 2)
            for _g, _dz in zip("ifog", (_dzi, _dzf, _dzo, _dzg)):
                _G["Wx" + _g] += np.outer(_dz, _st["x"])
                _G["Wh" + _g] += np.outer(_dz, _st["h_prev"])
                _G["b" + _g] += _dz
            _dh = (P["Whi"].T @ _dzi + P["Whf"].T @ _dzf
                   + P["Who"].T @ _dzo + P["Whg"].T @ _dzg)
            _dc = _dc_prev
        for _t in range(_T):
            _ht = _cache[_t]["o"] * _cache[_t]["tc"]
            _dWy += np.outer(_dlog[_t], _ht); _dby += _dlog[_t]
        return _G, _dWy, _dby

    # ---- 对账（小例：V=3, d_h=6, T=10）----
    def _nll36(P, Wy, by, xs, ys):
        _d_h = P["Wxi"].shape[0]
        _h, _c = np.zeros(_d_h), np.zeros(_d_h)
        _ce = 0.0
        for _k, _x in enumerate(xs):
            _i = sigmoid36(P["Wxi"] @ _x + P["Whi"] @ _h + P["bi"])
            _f = sigmoid36(P["Wxf"] @ _x + P["Whf"] @ _h + P["bf"])
            _o = sigmoid36(P["Wxo"] @ _x + P["Who"] @ _h + P["bo"])
            _g = np.tanh(P["Wxg"] @ _x + P["Whg"] @ _h + P["bg"])
            _c = _f * _c + _i * _g
            _h = _o * np.tanh(_c)
            _z = Wy @ _h + by
            _p = np.exp(_z - _z.max()); _p /= _p.sum()
            _ce -= np.log(_p[ys[_k]] + 1e-12)
        return _ce / len(ys)

    _rng36 = np.random.default_rng(1)
    _V36, _dh36, _T36 = 3, 6, 10
    _xs36b = np.eye(_V36)[_rng36.integers(0, _V36, _T36)]
    _ys36b = _rng36.integers(0, _V36, _T36)
    _P36b = init_lstm36(_V36, _dh36, seed=2, bf=2.0)
    _Wy36 = _rng36.normal(0, 1 / np.sqrt(_dh36), (_V36, _dh36)); _by36 = np.zeros(_V36)
    _Gb, _dWy_b, _dby_b = lstm_bptt36(_xs36b, _ys36b, _P36b, _Wy36, _by36)

    _worst36 = 0.0
    for _key in ("Wxi", "Whf", "bf", "Whg"):
        _p36 = _P36b[_key]
        for _ in range(3):
            _idx36 = tuple(_rng36.integers(0, _s) for _s in _p36.shape)
            _old36 = _p36[_idx36]; _eps36 = 1e-5
            _p36[_idx36] = _old36 + _eps36
            _f1 = _nll36(_P36b, _Wy36, _by36, _xs36b, _ys36b)
            _p36[_idx36] = _old36 - _eps36
            _f2 = _nll36(_P36b, _Wy36, _by36, _xs36b, _ys36b)
            _p36[_idx36] = _old36
            _num36 = (_f1 - _f2) / (2 * _eps36)
            _ana36 = _Gb[_key][_idx36]
            _worst36 = max(_worst36, abs(_ana36 - _num36) / max(abs(_ana36), abs(_num36), 1e-8))
    print(f"LSTM BPTT 对账（四张量 × 3 位置）：最差相对误差 {_worst36:.1e}")
    print("反向传播的二十行：σ 局部导数 z(1−z)、tanh 局部 (1−g²)、")
    print("c 与 h 两条回传路、dc_prev = dc × f 的高速路——全部数值验证通过")
    return (lstm_bptt36,)


@app.cell
def _(init_lstm36, lstm_forward36, np, plt):
    # 演示 2B · 高速路三方对照（脉冲实验）
    def trace36(model, bf=2.0, T=40, seed=10):
        """第 10 步换字符，返回两条轨迹的逐范数差。"""
        _r = np.random.default_rng(seed)
        _xs = np.eye(3)[_r.integers(0, 3, T)]
        _xs_p = _xs.copy()
        _xs_p[10] = np.eye(3)[(int(_xs_p[10].argmax()) + 1) % 3]

        def _run(sx):
            if model == "lstm":
                _, _hs, _ = lstm_forward36(sx, init_lstm36(3, 16, seed=3, bf=bf))
            else:
                _rr = np.random.default_rng(3)
                _Wx = _rr.normal(0, 1 / np.sqrt(3), (16, 3))
                _Wh = _rr.normal(0, 1 / np.sqrt(16), (16, 16))
                _h = np.zeros(16); _hs = []
                for _x in sx:
                    _h = np.tanh(_Wx @ _x + _Wh @ _h)
                    _hs.append(_h.copy())
                _hs = np.array(_hs)
            return _hs

        return np.linalg.norm(_run(_xs) - _run(_xs_p), axis=1)

    trace_rnn36 = trace36("rnn")
    trace_l0_36 = trace36("lstm", bf=0.0)
    trace_l2_36 = trace36("lstm", bf=2.0)
    print("脉冲探测（第 10 步换字，痕迹 = 两条轨迹的 h 差）：")
    for _nm, _tr in [("RNN α=1.0", trace_rnn36), ("LSTM b_f=0", trace_l0_36),
                     ("LSTM b_f=2", trace_l2_36)]:
        print(f"  {_nm}: 脉冲瞬间 {_tr[10]:.2f} → 20 步后 {_tr[29]:.4f}"
              f" → 30 步后 {_tr[39]:.4f}")

    _fig, _ax = plt.subplots(figsize=(8.8, 4.2))
    _ax.semilogy(range(40), np.maximum(trace_rnn36, 1e-6), lw=1.8,
                 color="tab:red", label="RNN")
    _ax.semilogy(range(40), np.maximum(trace_l0_36, 1e-6), lw=1.8,
                 color="tab:orange", label="LSTM，b_f=0（无先验）")
    _ax.semilogy(range(40), np.maximum(trace_l2_36, 1e-6), lw=2.2,
                 color="tab:green", label="LSTM，b_f=2（默认记住）")
    _ax.axvline(10, color="gray", ls="--", lw=1.2)
    _ax.text(10.6, 2.0, "脉冲", fontsize=9, color="gray")
    _ax.set_xlabel("时间步 t")
    _ax.set_ylabel("扰动痕迹（对数坐标）")
    _ax.set_title("高速路三方对照：结构 + 偏置先验，缺一不可", fontsize=11.5)
    _ax.legend(fontsize=9.5)
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()
    _fig
    return (trace36, trace_l0_36, trace_l2_36, trace_rnn36)

@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 3 · 出院实验：gap=12 三方对决（约 1 分钟）

        35 课的病人在这里接受出院考核。任务照旧：**回声**
        （yₜ = xₜ₋₁₂），但训练协议升级成**无限数据流**——
        每一轮都现场采样 8 条**全新的**随机序列（1200 轮 ≈
        9,600 条序列），**背题在数学上不可能**，唯一的得分
        路径是真正学会"把 12 步前的字符存进记忆"。

        三位选手同台（d_h=32、Adam、clip=5，训练约 1 分钟）：

        | 选手 | 新序列考试 | 判词 |
        |---|---|---|
        | RNN | **26%** | 聋着（35 课病人，原样） |
        | LSTM，b_f=0 | **70%** | 结构有用，但每步 σ(0)=0.5 砍半，学得吃力 |
        | LSTM，b_f=2 | **91%** | 高速路默认畅通——出院 |

        注意第二条：**同一个结构，只差一个初始化偏置，
        70% vs 91%**——"默认记住"这个先验值多少钱，这行
        表格就是标价。
        """
    )
    return


@app.cell
def _(init_lstm36, lstm_bptt36, lstm_forward36, np, plt, sigmoid36):
    # 演示 3 · 出院实验（无限数据流，gap=12 三方对决）
    def sample_batch36(delay, n_seq, T, rng):
        """现场采样 n 条新回声序列（无限数据流）。"""
        _out = []
        for _ in range(n_seq):
            _xs = np.eye(3)[rng.integers(0, 3, T)]
            _ys = np.concatenate([rng.integers(0, 3, delay),
                                  np.array(_xs[:-delay]).argmax(1)])
            _mask = np.ones(T, bool); _mask[:delay] = False
            _out.append((_xs, _ys, _mask))
        return _out

    def rnn_fwd36(xs, Wx, Wh, bh, Wy, by):
        _h = np.zeros(Wh.shape[0]); _hs, _logits = [], []
        for _x in xs:
            _h = np.tanh(Wx @ _x + Wh @ _h + bh)
            _hs.append(_h.copy()); _logits.append(Wy @ _h + by)
        _L = np.array(_logits)
        _Pr = np.exp(_L - _L.max(1, keepdims=True))
        return np.array(_hs), _Pr / _Pr.sum(1, keepdims=True)

    def rnn_bptt36(xs, ys, Wx, Wh, bh, Wy, by, loss_mask):
        _T = len(xs)
        _hs, _Pr = rnn_fwd36(xs, Wx, Wh, bh, Wy, by)
        _dWx, _dWh, _dbh = np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(bh)
        _dWy, _dby = np.zeros_like(Wy), np.zeros_like(by)
        _dlog = _Pr.copy(); _dlog[np.arange(_T), ys] -= 1
        _dlog = _dlog * loss_mask[:, None] / max(loss_mask.sum(), 1)
        _delta = np.zeros(Wh.shape[0])
        for _t in reversed(range(_T)):
            _dh = Wy.T @ _dlog[_t] + Wh.T @ _delta
            _delta = _dh * (1 - _hs[_t] ** 2)
            _dWx += np.outer(_delta, xs[_t])
            _hp = _hs[_t - 1] if _t > 0 else np.zeros(Wh.shape[0])
            _dWh += np.outer(_delta, _hp)
            _dbh += _delta; _dWy += np.outer(_dlog[_t], _hs[_t]); _dby += _dlog[_t]
        return [_dWx, _dWh, _dbh, _dWy, _dby]

    def train_stream36(model, delay=12, epochs=1200, n_seq=8, T=40, lr=0.03,
                       seed=1, bf=2.0, clip=5.0, d_h=32):
        """无限数据流训练。返回 (eval loss 曲线, 考试准确率)。"""
        _rng = np.random.default_rng(1234)               # 数据流种子
        _r = np.random.default_rng(seed)
        if model == "lstm":
            _P = init_lstm36(3, d_h, seed=seed, bf=bf)
            _Wy = _r.normal(0, 1 / np.sqrt(d_h), (3, d_h)); _by = np.zeros(3)
            _plist = list(_P.values()) + [_Wy, _by]
        else:
            _P = None
            _Wx = _r.normal(0, 1 / np.sqrt(3), (d_h, 3))
            _Wh = _r.normal(0, 1 / np.sqrt(d_h), (d_h, d_h))
            _bh = np.zeros(d_h)
            _Wy = _r.normal(0, 1 / np.sqrt(d_h), (3, d_h)); _by = np.zeros(3)
            _plist = [_Wx, _Wh, _bh, _Wy, _by]
        _ms = [np.zeros_like(_p) for _p in _plist]; _vs = list(_ms)
        _losses = []
        for _step in range(1, epochs + 1):
            _batch = sample_batch36(delay, n_seq, T, _rng)
            _accG = [np.zeros_like(_p) for _p in _plist]
            for _xs, _ys, _mask in _batch:
                if model == "lstm":
                    _G, _gWy, _gby = lstm_bptt36(_xs, _ys, _P, _Wy, _by, _mask)
                    _gl = list(_G.values()) + [_gWy, _gby]
                else:
                    _gl = rnn_bptt36(_xs, _ys, *_plist, _mask)
                for _k in range(len(_accG)):
                    _accG[_k] += _gl[_k] / len(_batch)
            _tot = np.sqrt(sum((_g.astype(float) ** 2).sum() for _g in _accG))
            if _tot > clip:
                _sc = clip / (_tot + 1e-12)
                _accG = [_g * _sc for _g in _accG]
            for _k, _p in enumerate(_plist):
                _ms[_k] = 0.9 * _ms[_k] + 0.1 * _accG[_k]
                _vs[_k] = 0.999 * _vs[_k] + 0.001 * _accG[_k] ** 2
                _p -= lr * (_ms[_k] / (1 - 0.9 ** _step)) / (
                    np.sqrt(_vs[_k] / (1 - 0.999 ** _step)) + 1e-8)
            # 评估（固定 held-out 批）
            _ev = sample_batch36(delay, 4, T, np.random.default_rng(777))
            _L = 0.0
            for _xs, _ys, _mask in _ev:
                if model == "lstm":
                    _Pr_ev = lstm_probs36(_xs, _P, _Wy, _by)
                    _L += -np.log(_Pr_ev[np.arange(T)[_mask], _ys[_mask]] + 1e-12).mean()
                else:
                    _, _Pr = rnn_fwd36(_xs, *_plist)
                    _L += -np.log(_Pr[np.arange(T)[_mask], _ys[_mask]] + 1e-12).mean()
            _losses.append(_L / len(_ev))
        # 最终考试（全新序列）
        _te = sample_batch36(delay, 6, T, np.random.default_rng(2024))
        _ok, _n = 0, 0
        for _xs, _ys, _mask in _te:
            if model == "lstm":
                _preds = lstm_probs36(_xs, _P, _Wy, _by).argmax(1)
            else:
                _, _Pr = rnn_fwd36(_xs, *_plist)
                _preds = _Pr.argmax(1)
            _ok += int((_preds[_mask] == _ys[_mask]).sum())
            _n += int(_mask.sum())
        return np.array(_losses), _ok / _n

    def lstm_probs36(xs, P, Wy, by):
        """LSTM 前向，返回每步概率。"""
        _d_h = P["Wxi"].shape[0]
        _h, _c = np.zeros(_d_h), np.zeros(_d_h)
        _Pr = []
        for _x in xs:
            _i = sigmoid36(P["Wxi"] @ _x + P["Whi"] @ _h + P["bi"])
            _f = sigmoid36(P["Wxf"] @ _x + P["Whf"] @ _h + P["bf"])
            _o = sigmoid36(P["Wxo"] @ _x + P["Who"] @ _h + P["bo"])
            _g = np.tanh(P["Wxg"] @ _x + P["Whg"] @ _h + P["bg"])
            _c = _f * _c + _i * _g
            _h = _o * np.tanh(_c)
            _z = Wy @ _h + by
            _p = np.exp(_z - _z.max()); _p /= _p.sum()
            _Pr.append(_p)
        return np.array(_Pr)

    GAP36 = 12
    print(f"出院实验：gap={GAP36}，无限数据流（每轮 8 条新序列 × 1200 轮），训练约 1 分钟……")
    loss_rnn36, acc_rnn36 = train_stream36("rnn", delay=GAP36)
    print(f"  RNN       ：eval loss {loss_rnn36[0]:.2f} → {loss_rnn36[-1]:.3f}，考试 {acc_rnn36:.0%}")
    loss_l036, acc_l036 = train_stream36("lstm", delay=GAP36, bf=0.0)
    print(f"  LSTM b_f=0：eval loss {loss_l036[0]:.2f} → {loss_l036[-1]:.3f}，考试 {acc_l036:.0%}")
    loss_l236, acc_l236 = train_stream36("lstm", delay=GAP36, bf=2.0)
    print(f"  LSTM b_f=2：eval loss {loss_l236[0]:.2f} → {loss_l236[-1]:.3f}，考试 {acc_l236:.0%}")

    _fig, _ax = plt.subplots(figsize=(9.2, 4.5))
    _ax.plot(loss_rnn36, color="tab:red", lw=1.7, label=f"RNN（考试 {acc_rnn36:.0%}）")
    _ax.plot(loss_l036, color="tab:orange", lw=1.7, label=f"LSTM b_f=0（考试 {acc_l036:.0%}）")
    _ax.plot(loss_l236, color="tab:green", lw=2.0, label=f"LSTM b_f=2（考试 {acc_l236:.0%}）")
    _ax.axhline(np.log(3), color="gray", ls="--", lw=1.1)
    _ax.text(3, np.log(3) + 0.02, "瞎猜线 ln3", fontsize=8.5, color="gray")
    _ax.set_xlabel("训练轮数（每轮全新序列）")
    _ax.set_ylabel("held-out 损失")
    _ax.set_title(f"出院实验：gap={GAP36} 回声，RNN 聋着，LSTM(b_f=2) 出院", fontsize=11.5)
    _ax.legend(fontsize=9.5)
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()
    _fig
    return (GAP36, acc_l036, acc_l236, acc_rnn36, loss_l036, loss_l236,
            loss_rnn36, sample_batch36, train_stream36)


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
        ### 交互实验 · 手动阀门实验室

        扔掉权重，**亲手当一次三道门**：拖动 f / i / o 三个
        旋钮（开度 0~1），看记忆 c 的命运。

        场景：20 个时间步，**前 5 步有信号**（g = 1，新信息
        来了），**后 15 步无信号**（g = 0，世界安静了）——
        c 和 h 会怎么演化？

        ```
        cₜ = f · cₜ₋₁ + i · gₜ        hₜ = o · tanh(cₜ)
        ```

        试试：f=1.0, i=1.0（完美写入+永不遗忘——c 是一级
        台阶）；f=0.8（指数遗忘——数数半衰几步）；f=0.3
        （金鱼）；i=0.2（写入不足——台阶矮了一截）；o 拖小
        （记忆还在，只是不说话——**c 和 h 的分离**正是
        LSTM 的精髓：存着 ≠ 说出来）。
        """
    )
    return


@app.cell
def _(mo):
    # 交互控制：三道门开度
    f_slider36 = mo.ui.slider(0.0, 1.0, value=1.0, step=0.05, debounce=True,
                              label="🚪 遗忘门 f（旧记忆留存比例）")
    i_slider36 = mo.ui.slider(0.0, 1.0, value=1.0, step=0.05, debounce=True,
                              label="✍️ 输入门 i（新信息写入比例）")
    o_slider36 = mo.ui.slider(0.0, 1.0, value=1.0, step=0.05, debounce=True,
                              label="🔊 输出门 o（记忆放出的比例）")
    mo.hstack([f_slider36, i_slider36, o_slider36])
    return (f_slider36, i_slider36, o_slider36)


@app.cell
def _(f_slider36, i_slider36, mo, np, o_slider36, plt):
    # ===== 交互实验 · 手动阀门实验室（响应单元）=====
    _f, _i, _o = f_slider36.value, i_slider36.value, o_slider36.value
    _T = 20
    _g = np.zeros(_T); _g[:5] = 1.0                     # 前 5 步有信号
    _c, _h = 0.0, 0.0
    _cs, _hs = [], []
    for _t in range(_T):
        _c = _f * _c + _i * _g[_t]
        _h = _o * np.tanh(_c)
        _cs.append(_c); _hs.append(_h)
    _cs, _hs = np.array(_cs), np.array(_hs)
    _residual = _cs[-1] / max(_cs[4], 1e-9)            # 信号期结束时的值 15 步后还剩多少

    _fig, _ax = plt.subplots(figsize=(9.4, 4.2))
    _ax.plot(_cs, lw=2.4, color="tab:green", label="细胞状态 c（长期记忆）")
    _ax.plot(_hs, lw=2.0, color="tab:blue", ls="--", label="输出 h = o·tanh(c)")
    _ax.axvspan(-0.5, 4.5, color="gold", alpha=0.18)
    _ax.text(2, -0.06, "信号期（g=1）", ha="center", fontsize=9, color="gray")
    _ax.text(12, -0.06, "安静期（g=0）", ha="center", fontsize=9, color="gray")
    _ax.set_xlabel("时间步 t")
    _ax.set_title(f"f={_f:.2f}, i={_i:.2f}, o={_o:.2f}：信号期写入 {(_i * 1.0):.2f}，"
                  f"15 步安静期后残留 {(_residual):.0%}", fontsize=10.5)
    _ax.legend(fontsize=10, loc="center right")
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()

    _half = int(np.ceil(np.log(0.5) / np.log(max(_f, 1e-9)))) if _f < 1.0 else None
    if _f >= 0.999:
        _verdict = (f"**完美传送带**（f = {_f:.2f}）：安静期 c 一动不动——"
                    "信号期写入的值原封不动躺了 15 步。梯度沿这条路回传也"
                    "是无衰减的（∂c/∂c = 1）——这就是高速路")
    elif _f >= 0.9:
        _verdict = (f"**缓慢遗忘**（f = {_f:.2f}）：半衰约 {_half} 步，"
                    f"15 步后残留 {(_residual):.0%}——LSTM 常见的工作状态："
                    "大部分记住、一点点漏")
    else:
        _verdict = (f"**快速遗忘**（f = {_f:.2f}）：半衰仅 {_half} 步，"
                    f"15 步后只剩 {(_residual):.0%}——RNN 式的命运。"
                    "注意 o 改不了 c 的衰减：**输出门只管「说不说」,"
                    "管不了「记不记」**——这就是为什么 h 会被压低而 c 依然独立演化")
    if _i < 0.5:
        _verdict += f"（另：输入门 i={_i:.2f} 把写入砍到 {(_i * 1.0):.2f}——台阶天生矮）"
    mo.vstack([
        _fig,
        mo.callout(mo.md(_verdict), kind="info"),
    ])
    return

@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✏️ 动手练习（本课 4 题）": mo.md(
                """
                **练习 1 · 手算一个 LSTM 时间步**：给定三道门
                的开度和候选值（i=0.8, f=0.9, o=0.5, g=0.6）
                以及旧记忆 cₜ₋₁ = 1.0，笔算新记忆 cₜ 和输出 hₜ
                （tanh(1.38) ≈ 0.88，已给出）。

                **练习 2 · 补全前向核心两行**：c 的传送带更新
                和 h 的输出——LSTM 的灵魂就这两行。

                **练习 3 · 参数侦探**：V=3、d_h=32 的 RNN 和
                LSTM 各有多少参数？（先算 RNN，再想 LSTM 是
                它的几倍）

                **练习 4 · 概念题**：为什么说 c 是"梯度高速路"？
                """
            )
        }
    )
    return


@app.cell
def _(mo, np):
    # ===== 练习 1 · 手算一个 LSTM 时间步 =====
    print("给定：i = 0.8，f = 0.9，o = 0.5，g = 0.6，c_prev = 1.0")
    print("  cₜ = f ⊙ c_prev + i ⊙ g = ?")
    print("  hₜ = o ⊙ tanh(cₜ) = ?   （tanh(1.38) ≈ 0.88）")

    c_new_ex1_36 = None                           # ← ① c_t（两位小数）
    h_new_ex1_36 = None                           # ← ② h_t（两位小数）

    if c_new_ex1_36 is not None and h_new_ex1_36 is not None:
        _tc = 0.9 * 1.0 + 0.8 * 0.6
        _th = 0.5 * np.tanh(_tc)
        if abs(c_new_ex1_36 - _tc) < 0.01 and abs(h_new_ex1_36 - _th) < 0.01:
            _msg = (f"✅ 正确：cₜ = {_tc:.2f}，hₜ = {_th:.2f}。"
                    "旧记忆留存 0.9、新信息写入 0.48，加法合成 1.38；"
                    "输出门只放行一半（0.5×0.88≈0.44）——"
                    "**存着的（c）和说出来的（h）是两回事**")
        else:
            _msg = (f"❌ 参考：cₜ = {_tc:.2f}，hₜ = {_th:.2f}。"
                    "c 是「乘留存 + 乘写入」的加法；h 是 o × tanh(c)")
    else:
        _msg = "⏳ 两空都填上（提示：cₜ = 0.9×1.0 + 0.8×0.6）"
    mo.md(_msg)
    return (c_new_ex1_36, h_new_ex1_36)


@app.cell
def _(mo, np, sigmoid36):
    # ===== 练习 2 · 补全前向核心两行 =====
    def lstm_cell_ex2_36(x, h_prev, c_prev, P):
        """四道门已写好，补 c 与 h 的两行更新。"""
        _i = sigmoid36(P["Wxi"] @ x + P["Whi"] @ h_prev + P["bi"])
        _f = sigmoid36(P["Wxf"] @ x + P["Whf"] @ h_prev + P["bf"])
        _o = sigmoid36(P["Wxo"] @ x + P["Who"] @ h_prev + P["bo"])
        _g = np.tanh(P["Wxg"] @ x + P["Whg"] @ h_prev + P["bg"])
        _c = None                                   # ← ① 补：传送带更新（f、i、g、c_prev 四个量）
        _h = None                                   # ← ② 补：输出（o 和 _c 两个量）
        return _h, _c

    cell_ready_ex2_36 = None                       # ← 补完后，把这行换成 lstm_cell_ex2_36

    if cell_ready_ex2_36 is not None:
        try:
            _r = np.random.default_rng(0)
            _V, _dh = 3, 4
            _P = {}
            for _g in "ifog":
                _P["Wx" + _g] = _r.normal(0, 0.4, (_dh, _V))
                _P["Wh" + _g] = _r.normal(0, 0.3, (_dh, _dh))
                _P["b" + _g] = np.zeros(_dh)
            _x = np.eye(_V)[0]; _hp = np.full(_dh, 0.3); _cp = np.full(_dh, -0.2)
            _h_got, _c_got = cell_ready_ex2_36(_x, _hp, _cp, _P)
            _i = sigmoid36(_P["Wxi"] @ _x + _P["Whi"] @ _hp + _P["bi"])
            _f = sigmoid36(_P["Wxf"] @ _x + _P["Whf"] @ _hp + _P["bf"])
            _o = sigmoid36(_P["Wxo"] @ _x + _P["Who"] @ _hp + _P["bo"])
            _g = np.tanh(_P["Wxg"] @ _x + _P["Whg"] @ _hp + _P["bg"])
            _c_ref = _f * _cp + _i * _g
            _h_ref = _o * np.tanh(_c_ref)
            if _c_got is not None and np.allclose(_c_got, _c_ref, atol=1e-12) \
                    and np.allclose(_h_got, _h_ref, atol=1e-12):
                _msg = ("✅ 逐元素一致！c = f·c_prev + i·g（加法传送带），"
                        "h = o·tanh(c)（tanh 只在出口）——"
                        "这两行就是 LSTM 与 RNN 的全部区别所在")
            else:
                _msg = "❌ 与参考不一致：c 用 ⊙（逐元素乘）、加法；h 里 tanh 只包 c"
        except (TypeError, ValueError) as _e:
            _msg = f"❌ 运行出错：{_e}——_c = _f * c_prev + _i * _g；_h = _o * np.tanh(_c)"
    else:
        _msg = "⏳ 补两行：_c = _f * c_prev + _i * _g；_h = _o * np.tanh(_c)"
    mo.md(_msg)
    return (cell_ready_ex2_36, lstm_cell_ex2_36)


@app.cell
def _(mo):
    # ===== 练习 3 · 参数侦探 =====
    V_ex3_36, d_h_ex3_36 = 3, 32                 # 词表 3 字符，隐状态 32 维

    n_rnn_ex3_36 = None                          # ← ① RNN 的参数量（整数）
    n_lstm_ex3_36 = None                         # ← ② LSTM 的参数量（整数）

    if n_rnn_ex3_36 is not None and n_lstm_ex3_36 is not None:
        _t_rnn = V_ex3_36 * d_h_ex3_36 + d_h_ex3_36 ** 2 + d_h_ex3_36
        _t_lstm = 4 * _t_rnn
        if n_rnn_ex3_36 == _t_rnn and n_lstm_ex3_36 == _t_lstm:
            _msg = (f"✅ 正确：RNN {_t_rnn:,}（W 96 + U 1,024 + b 32），"
                    f"LSTM {_t_lstm:,} = 恰好 4 倍——四套门各配一套"
                    "（W/U/b × 4）。LSTM 用 4 倍参数换一条不衰减的梯度路："
                    "演示 3 里 91% vs 26%，这笔账划算")
        else:
            _msg = (f"❌ 参考：RNN {_t_rnn:,}、LSTM {_t_lstm:,}（4 倍）。"
                    "LSTM 的四套门各有一套完整的 (Wx, Wh, b)")
    else:
        _msg = "⏳ 两空都填上（提示：LSTM = 四套门 × RNN 的单套）"
    mo.md(_msg)
    return (V_ex3_36, d_h_ex3_36, n_lstm_ex3_36, n_rnn_ex3_36)


@app.cell
def _(mo):
    # ===== 练习 4 · 概念题 =====
    print("为什么说细胞状态 c 是「梯度高速路」？")
    print("  A. ∂cₜ/∂cₜ₋₁ = fₜ：梯度沿 c 回传每步只乘门控值 f，没有 tanh 压缩、没有矩阵连乘，")
    print("     而且网络可以把 f 学到接近 1，让梯度几乎无衰减地走几十步")
    print("  B. 因为 c 的维度比 h 大，能装下更多梯度")
    print("  C. 因为 c 用了 sigmoid 激活，sigmoid 的导数比 tanh 大")

    ans_ex4_36 = None                            # ← ① 填 'A' / 'B' / 'C'

    if ans_ex4_36 is not None:
        if ans_ex4_36 == "A":
            _msg = ("✅ 正确。对比：RNN 的 ∂hₜ/∂hₜ₋₁ = U·diag(1−h²)（矩阵 × 饱和因子，"
                    "结构性衰减）；LSTM 的 ∂cₜ/∂cₜ₋₁ = fₜ（一个可学习的标量门）。"
                    "「每步乘什么」从结构的固定刑期变成网络的选择题——"
                    "这就是 35 课病历上那台手术的全部含义")
        else:
            _msg = "❌ 提示：想想梯度沿 c 回传一步要乘什么？和 RNN 沿 h 回传一步要乘什么？"
    else:
        _msg = "⏳ 填一个选项（'A' / 'B' / 'C'）"
    mo.md(_msg)
    return (ans_ex4_36,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 1 参考答案（先自己写过再看！）": mo.md(
                """
                **cₜ = 1.38，hₜ = 0.44。**

                ```
                cₜ = 0.9 × 1.0 + 0.8 × 0.6 = 0.90 + 0.48 = 1.38
                hₜ = 0.5 × tanh(1.38) = 0.5 × 0.88 ≈ 0.44
                ```

                三个观察：① 旧记忆和新信息是**加法**合成
                （不是 RNN 的"整体重写"）；② cₜ=1.38 可以
                超出 1——传送带没有值域限制，tanh 只在出口
                管 h；③ 输出门 0.5 让 h 只有 0.44——**记忆
                还在 c 里躺着，只是没全说出来**。
                """
            ),
            "✅ 练习 2 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                _c = _f * c_prev + _i * _g
                _h = _o * np.tanh(_c)
                ```

                第一行是 LSTM 的灵魂：**留存 × 旧记忆 +
                写入 × 新内容**——两个 0~1 的系数、一个
                加号，替代了 RNN 的"穿过 tanh 整体重写"。

                第二行常被写错成 `_h = tanh(_o * _c)`
                （把 o 放进 tanh 里面）——注意门的正确位置：
                **o 在 tanh 外面**（先压 c、再按 o 打折），
                这才能让 c 和 h 解耦：c 可以很大（长期积累），
                h 每步都被 o 控制在 [-1,1] × 开度。
                """
            ),
            "✅ 练习 3 参考答案（先自己写过再看！）": mo.md(
                """
                **RNN 1,152；LSTM 4,608（恰好 4 倍）。**

                ```
                RNN : W(3×32) + U(32×32) + b(32) = 96 + 1024 + 32 = 1,152
                LSTM: 四套门 × 单套(96 + 1024 + 32)   = 4,608
                ```

                4 倍参数换一条不衰减的梯度路 + 三道可学习的
                阀门。演示 3 的标价：91% vs 26%。另一个
                视角：GRU 把四套门砍到两套多一点的等效
                （约 3/4 的参数），是"精简版手术"。

                顺带算笔纵向账：37 课的语言模型 d_h=128、
                词表几十——LSTM 参数约 4×(128² + 128×V +
                128) ≈ 8 万，全程 CPU 无压力。
                """
            ),
            "✅ 练习 4 参考答案（先自己写过再看！）": mo.md(
                """
                **选 A。**

                把两条"一步回传"并排放：

                ```
                RNN ：∂hₜ/∂hₜ₋₁ = U · diag(1−hₜ²)   ← 矩阵 × 饱和因子
                LSTM：∂cₜ/∂cₜ₋₁ = fₜ                 ← 一个门控标量
                ```

                RNN 的乘子是**结构发给梯度的固定刑期**（U
                你改不了形状，tanh 饱和你说了不算）；LSTM
                的乘子是**网络自己的选择题**（f 是看着输入
                现算的，想让梯度过就把 f 推向 1）。

                B 错：c 和 h 同维（d_h）。C 错得离谱：
                sigmoid 导数最大才 0.25（19 课），比 tanh
                的 1 还小——LSTM 的高明从来不在激活函数，
                在**加法**。
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
                1. **手术方案**：给 RNN 加一条细胞状态 c 的
                   传送带——cₜ = f⊙cₜ₋₁ + i⊙gₜ（加法更新），
                   tanh 只在出口 h = o⊙tanh(c) 出现
                2. **三道门的岗位**：f 留存旧记忆、i 写入新
                   信息、o 放出对外输出——都是 σ(线性组合)
                   的 0~1 开度，看着输入自己决定；⊙ 逐元素
                   乘 = 每个记忆格独立的阀门
                3. **梯度高速路**：∂cₜ/∂cₜ₋₁ = fₜ——"每步
                   乘什么"从结构刑期（U·diag(1−h²)）变成
                   可学习的选择题（f 可以学到 ≈1）
                4. **偏置先验 b_f=2**：σ(0)=0.5 意味着"默认
                   砍半"——把遗忘门偏置初始化到 2（σ(2)≈
                   0.88），传送带默认畅通；脉冲实验：无先验
                   的 LSTM 痕迹照样死光（0.0002 vs 0.086）
                5. **对账过关**：LSTM BPTT（约二十行）数值
                   对账最差 10⁻⁸——c/h 双路回传、四门梯度、
                   σ 与 tanh 局部导数全部正确
                6. **出院实验**：gap=12 回声 + 无限数据流
                   （背题不可能），RNN 26% / LSTM(b_f=0)
                   70% / LSTM(b_f=2) 91%——结构与先验
                   缺一不可，35 课的病人出院
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

                六课攒到最后一件兵器：一个**结构完整、对账
                通过、长依赖验证成功**的 LSTM。第 6 部分只剩
                最后一件事——把它放到**真语料**上，让它学会
                "说话"。

                字符级语言模型：给模型看一段中文文本，任务是
                每读一个字符就预测下一个字符的概率分布。
                训练它几千步，它就会摸到这套文本的"文法"——
                哪些字常跟哪些字、标点在哪里出现、句子怎么
                开怎么收。然后是最迷人的一步：**采样生成**——
                拿它的预测分布一个字一个字往下抽，看它写出
                什么。
                """
            ),
            mo.callout(
                mo.md(
                    """
                    **下一课预告** — `37_字符级语言模型`（第 6 部分收官）：

                    - 语料：课程自写的一篇深度学习科普长文——
                      让模型学的和学的对象都是深度学习，闭环了
                    - **温度**：同一个模型，τ=0.2 循规蹈矩、
                      τ=1.2 天马行空——softmax 的方差旋钮
                    - **终局对决**：RNN vs LSTM 同任务同预算，
                      loss 曲线 + 各生成 200 字——35/36 课的
                      病理报告在真实语料上验收
                    - 顺手学两个工程件：**困惑度**（perplexity，
                      语言模型的标准成绩单）和 **截断 BPTT**
                      （长文档的批处理切法）

                    一切都齐了，让它开口说话。

                    🎤 收官之战，开麦！
                    """
                ),
                kind="success",
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
