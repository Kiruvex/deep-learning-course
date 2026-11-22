import marimo

__generated_with = "0.25.2"
app = marimo.App(app_title="34 · BPTT", width="medium")


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
        # 34 · BPTT：沿时间反向传播

        **从零到 Transformer · 手写深度学习** ｜ 第 6 部分 · 循环神经网络（3/6）
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📖 本课目标（先读我）": mo.md(
                """
                **33 课把 RNN 的管道修通了，这一课给管道里灌上梯度。**

                好消息是：你不需要学任何新的微积分。33 课的展开图
                告诉我们——**RNN 展开就是一个深度为 T 的前馈网络**
                （每层权重共享）。所以 22 课的链式法则原样上岗，
                只是梯度现在要**沿时间往回传**：从损失出发，穿过
                第 T 步的 h、第 T−1 步的 h……一路回到第 1 步。

                这套算法叫 **BPTT**（Backpropagation Through Time，
                沿时间反向传播）。本课的重点课（第 6 部分的数学
                心脏，练习加量到 4 题）：

                - 看清梯度回传的**三条路径**，以及为什么 h 路
                  要**连乘** U 和 tanh′——连乘正是 35 课一切
                  灾难的种子
                - 亲手实现 BPTT 的**倒序单循环**写法（工程上
                  真正用的递推形式，不是教科书上的全路径求和）
                - 过 22 课的**数值对账**仪式：解析梯度 vs 数值
                  梯度，10⁻⁸ 级别误差才算过关
                - 训练一个**延迟回声**任务（yₜ = xₜ₋₃），亲眼看
                  BPTT 把 RNN 教会"记住 3 步前"
                - 给梯度做**体检**：把 U 的梯度按时间步分解，
                  看远处的时间步还能收到多少梯度——35 课的
                  灾难预告片

                前置：22-23 课的反向传播与数值梯度、24 课 Adam、
                33 课的 rnn_forward 与展开图。
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "🗺️ 概念 1/3 · 展开图就是计算图": mo.md(
                """
                BPTT 的第一课认知：**不要发明新算法**。

                33 课把"同一个 cell 用 T 次"画成"T 个相同单元排成
                一排"。在这个展开视角下，RNN 就是一个**深度为 T、
                每层共享权重**的前馈网络：

                - xₜ 是第 t 层的输入
                - hₜ₋₁ → hₜ 是第 t 层的"层间传递"
                - W、U、b 在每层都出现，但**是同一组变量**

                于是 22 课的全部机械流程照搬：前向算一遍并**缓存**
                中间量 → 损失对输出求导 → 沿计算图**从后往前**，
                每个节点把自己的梯度发给上游。

                唯一的新问题出在"共享"上：U 在 T 个位置被用了 T 次，
                它的梯度怎么算？答案是 23 课 MLP 权重共享的老规矩——
                **每个使用位置各贡献一份，全部加起来**。
                """
            ),
            "🛣️ 概念 2/3 · 三条路径，一条要连乘": mo.md(
                """
                站在损失 L（比如只看最后一步的预测）往回看，梯度
                回到参数有三条路：

                **路径一 · 输出层（一步直达）。** L → y_T → h_T →
                Why：`∂L/∂Why = δy·h_Tᵀ`。没有连乘，安全。

                **路径二 · 输入权重（本地直达）。** 每个 hₜ 都直接
                吃了 xₜ：`∂L/∂W += …(直达 hₜ 的梯度)·xₜᵀ`。
                也没有连乘。

                **路径三 · 循环权重（要连乘）。** 想改"第 1 步用的
                U"，影响要穿过整条 h 链：

                ```
                ∂h_T/∂h_t = ∏ (U · diag(1−hₖ²))     ← k 从 t+1 到 T
                ```

                每过一步，梯度乘一次 U 再乘一次 tanh′（局部导数
                1−h²）。**乘 T−t 次**。这一串连乘就是 BPTT 的
                命门：乘的东西普遍比 1 小，连乘就指数衰减（梯度
                消失）；偶尔比 1 大，连乘就指数爆炸。35 课整课
                都在解剖这一行公式，本课演示 4 先让你看一眼症状。

                顺带一个对称的认知：33 课脉冲实验里"前向扰动沿
                时间衰减"，和这里的"反向梯度沿时间衰减"，走的是
                **同一条连乘路**——方向相反，命运相同。
                """
            ),
            "🔁 概念 3/3 · 工程写法：倒序单循环": mo.md(
                """
                教科书写 ∂L/∂U = Σₜ Σₖ≤ₜ …（双求和，吓人）；工程上
                谁也不这么写。真正的 BPTT 是**一个倒序循环**：

                ```python
                delta = 0                          # 未来传回来的梯度
                for t in reversed(range(T)):
                    dh  = Wyᵀ·dlogₜ + Uᵀ·delta    # 两路汇合：y 路 + 未来路
                    delta = dh * (1 − hₜ²)        # 过 tanh′ 闸门
                    dW += delta ⊗ xₜ              # 本步的三份"工资"
                    dU += delta ⊗ hₜ₋₁
                    db += delta
                ```

                三个实现要点（每个都是经典 bug 来源）：

                - **缓存**：前向必须把每步的 hₜ 存下来（反向要用）
                - **倒序**：δ 必须从 t=T 往 t=1 传——先算"未来"
                  才能算"过去"
                - **累加**：W/U/b 的梯度是各步贡献之和，
                  `+=` 别写成 `=`

                记住这个 7 行骨架，你会在这门课里反复见到它
                （36 课 LSTM 的反向传播是它的豪华版）。
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
        ### 演示 1 · 双向展开图：前向黑、反向红

        33 课的展开图加上反向梯度流（红色虚线）。四件事看清楚：

        - 损失 L 只挂在最后一步的 y 上（本图设定；语言模型是
          每步都挂一个，梯度汇流方式相同）
        - **Why 路**：L → y → h，一步直达（绿色）
        - **U 路**：h → h → h 沿时间倒流，**每过一步乘一次
          U 和 (1−h²)**（红色粗箭头）——连乘发生在这条路上
        - 每个时间步把"流经自己的梯度"发一份给 U（dU += …）
          ——所以 U 的总梯度是 T 份贡献之和
        """
    )
    return


@app.cell
def _(plt):
    # 演示 1 · 双向展开图
    _fig, _ax = plt.subplots(figsize=(11.8, 6.4))
    _ax.set_xlim(0, 12.6)
    _ax.set_ylim(0, 7.4)
    _ax.axis("off")
    _steps = 4
    for _k in range(_steps):
        _x0 = 1.1 + _k * 2.9
        _ax.add_patch(plt.Rectangle((_x0 - 0.45, 0.5), 0.9, 0.7,
                                    facecolor="#e8f0fe", edgecolor="#4c78a8", lw=1.4))
        _ax.text(_x0, 0.85, f"$x_{{{_k+1}}}$", ha="center", va="center", fontsize=13)
        _ax.add_patch(plt.Rectangle((_x0 - 0.66, 2.3), 1.32, 1.2,
                                    facecolor="#dceefb", edgecolor="#2b6cb0", lw=1.8))
        _ax.text(_x0, 2.9, "RNN\ncell", ha="center", va="center", fontsize=10.5,
                 fontweight="bold")
        _ax.add_patch(plt.Rectangle((_x0 - 0.45, 4.55), 0.9, 0.7,
                                    facecolor="#e6f4ea", edgecolor="#2e7d32", lw=1.4))
        _ax.text(_x0, 4.9, f"$h_{{{_k+1}}}$", ha="center", va="center", fontsize=13)
        # 前向箭头（黑）
        _ax.annotate("", xy=(_x0, 3.55), xytext=(_x0, 1.25),
                     arrowprops=dict(arrowstyle="->", color="#4c78a8", lw=1.5))
        _ax.annotate("", xy=(_x0, 4.9), xytext=(_x0, 3.55),
                     arrowprops=dict(arrowstyle="->", color="#2e7d32", lw=1.5))
        if _k < _steps - 1:
            _ax.annotate("", xy=(_x0 + 2.9 - 0.7, 2.9), xytext=(_x0 + 0.48, 2.9),
                         arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.8))
        # 反向箭头（红虚线，U 路：h_{k+1} -> h_k）
        if _k < _steps - 1:
            _ax.annotate("", xy=(_x0 + 0.48, 4.05), xytext=(_x0 + 2.9 - 0.7, 4.05),
                         arrowprops=dict(arrowstyle="->", color="tab:red", lw=2.2,
                                         ls="--"))
    # 损失与 y
    _xT = 1.1 + 3 * 2.9
    _ax.add_patch(plt.Rectangle((_xT - 0.45, 5.9), 0.9, 0.7,
                                facecolor="#fde8e8", edgecolor="tab:red", lw=1.6))
    _ax.text(_xT, 6.25, "L", ha="center", va="center", fontsize=14, fontweight="bold")
    _ax.annotate("", xy=(_xT, 6.62), xytext=(_xT, 5.32),
                 arrowprops=dict(arrowstyle="->", color="tab:red", lw=2.0, ls="--"))
    # 标注
    _ax.text(_xT + 0.6, 6.25, "损失（只挂最后一步的预测）", fontsize=10, va="center")
    _ax.text(6.0, 4.05, "反向：梯度沿 h 链倒流\n每过一步 × U × (1−h²)   ← 连乘！",
             fontsize=10.5, color="tab:red", ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="tab:red"))
    _ax.text(2.2, 3.62, r"$dU\ {+}{=}\ \delta_t \otimes h_{t-1}$（每步发一份工资给 U）",
             fontsize=9.5, color="tab:red")
    _ax.text(6.3, 0.85, "$x_t$（one-hot）", fontsize=9.5, color="#4c78a8")
    _ax.text(0.3, 2.9, "$h_0$=0 →", fontsize=10, color="tab:red", ha="right", va="center")
    _ax.text(6.3, 7.15, "前向黑/绿箭头（33 课）+ 反向红虚线（本课）：展开图就是计算图",
             fontsize=11, fontweight="bold", ha="center")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 2 · 亲手实现 BPTT + 22 课对账仪式

        `bptt34` 就是概念 3 那个 7 行骨架的完整版（支持
        `loss_mask`：默认每步都算损失——语言模型式；把 mask
        设成"只开最后一步"，就得到演示 4 用的纯净版）。

        对账仪式照旧：随机位置挖出参数、扰动 ε=10⁻⁵ 算数值
        梯度（中心差分），与解析梯度比相对误差。**五个参数
        张量、各抽 5 个位置，全部 10⁻⁸ 量级**——和 22/23/29
        课的门槛一致。
        """
    )
    return


@app.cell
def _(np):
    # 演示 2 · BPTT 实现 + 数值对账
    def forward_rnn34(xs, Wx, Wh, bh, Wy, by, h0=None):
        """前向（缓存版）：返回 h 序列、logits、每步 softmax 分布。"""
        _h = np.zeros(Wh.shape[0]) if h0 is None else h0.copy()
        _hs, _logits = [], []
        for _x in xs:
            _h = np.tanh(Wx @ _x + Wh @ _h + bh)
            _hs.append(_h.copy())
            _logits.append(Wy @ _h + by)
        _L = np.array(_logits)
        _P = np.exp(_L - _L.max(axis=1, keepdims=True))
        _P = _P / _P.sum(axis=1, keepdims=True)
        return np.array(_hs), _L, _P

    def bptt34(xs, ys, Wx, Wh, bh, Wy, by, loss_mask=None):
        """BPTT（倒序单循环）。loss_mask: (T,) bool，默认全开。
        返回 (grads 字典, per_step_dU 各步对 U 的贡献范数)。"""
        _T = len(xs)
        if loss_mask is None:
            loss_mask = np.ones(_T, bool)
        _hs, _, _P = forward_rnn34(xs, Wx, Wh, bh, Wy, by)
        _dWx, _dWh, _dbh = np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(bh)
        _dWy, _dby = np.zeros_like(Wy), np.zeros_like(by)
        _dlog = _P.copy()
        _dlog[np.arange(_T), ys] -= 1
        _dlog = _dlog * np.asarray(loss_mask)[:, None] / max(np.asarray(loss_mask).sum(), 1)
        _delta = np.zeros(Wh.shape[0])                 # 从未来传回来的梯度
        _per = np.zeros(_T)                            # 各步对 U 的贡献范数
        for _t in reversed(range(_T)):
            _dh = Wy.T @ _dlog[_t] + Wh.T @ _delta     # 两路汇合
            _delta = _dh * (1 - _hs[_t] ** 2)          # 过 tanh' 闸门
            _dWx += np.outer(_delta, xs[_t])
            _h_prev = _hs[_t - 1] if _t > 0 else np.zeros(Wh.shape[0])
            _dWh += np.outer(_delta, _h_prev)
            _per[_t] = np.linalg.norm(np.outer(_delta, _h_prev))
            _dbh += _delta
            _dWy += np.outer(_dlog[_t], _hs[_t])
            _dby += _dlog[_t]
        return dict(Wx=_dWx, Wh=_dWh, bh=_dbh, Wy=_dWy, by=_dby), _per

    # ---- 对账：V=3, d_h=5, T=8 ----
    _rng34 = np.random.default_rng(1)
    V34, d_h34, T34 = 3, 5, 8
    _xs34 = np.eye(V34)[_rng34.integers(0, V34, T34)]
    _ys34 = _rng34.integers(0, V34, T34)
    _r34 = np.random.default_rng(2)
    Wx34 = _r34.normal(0, 1 / np.sqrt(V34), (d_h34, V34))
    Wh34 = _r34.normal(0, 1 / np.sqrt(d_h34), (d_h34, d_h34))
    bh34 = np.zeros(d_h34)
    Wy34 = _r34.normal(0, 1 / np.sqrt(d_h34), (V34, d_h34))
    by34 = np.zeros(V34)

    _grads34, _ = bptt34(_xs34, _ys34, Wx34, Wh34, bh34, Wy34, by34)

    def nll34(xs, ys, Wx, Wh, bh, Wy, by):
        _hs, _, _P = forward_rnn34(xs, Wx, Wh, bh, Wy, by)
        return -np.log(_P[np.arange(len(ys)), ys] + 1e-12).mean()

    def num_grad34(key, idx, eps=1e-5):
        _params34 = dict(Wx=Wx34, Wh=Wh34, bh=bh34, Wy=Wy34, by=by34)
        _p = _params34[key]
        _old = _p[idx]
        _p[idx] = _old + eps
        _f1 = nll34(_xs34, _ys34, Wx34, Wh34, bh34, Wy34, by34)
        _p[idx] = _old - eps
        _f2 = nll34(_xs34, _ys34, Wx34, Wh34, bh34, Wy34, by34)
        _p[idx] = _old
        return (_f1 - _f2) / (2 * eps)

    print("BPTT 数值对账（每张量抽 5 个随机位置，相对误差）：")
    _worst34 = 0.0
    for _key in ("Wx", "Wh", "bh", "Wy", "by"):
        _errs = []
        for _ in range(5):
            _p = dict(Wx=Wx34, Wh=Wh34, bh=bh34, Wy=Wy34, by=by34)[_key]
            _idx = tuple(_rng34.integers(0, _s) for _s in _p.shape)
            _a, _n = _grads34[_key][_idx], num_grad34(_key, _idx)
            _errs.append(abs(_a - _n) / max(abs(_a), abs(_n), 1e-8))
        _worst34 = max(_worst34, max(_errs))
        print(f"  {_key}: max 相对误差 {max(_errs):.1e}")
    print(f"\n五张量全过（最差 {_worst34:.1e} < 1e-7）：解析梯度 = 真梯度")
    return (Wh34, Wx34, bptt34, bh34, by34, d_h34, forward_rnn34, nll34, num_grad34, V34, Wy34)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 3 · 训练：延迟回声任务（yₜ = xₜ₋₃）

        对账过了，梯度是真的——那就用它训练。任务设计得很
        狡猾：随机字符流 {a,b,c}，**目标是 3 步前的字符**。
        网络想得分，就必须把 xₜ₋₃ 存进 h、扛过 3 步、再取
        出来——记忆被强行变成了刚需（W 路偷不了懒）。

        25 课配方原样复用：Adam、全批、300 轮。看 loss 从
        1.1（瞎猜线 ln3）掉到贴地，考试准确率 100%。

        一个细节：前 3 步的目标是随机的（它们的"3 步前"
        在序列外），所以 `loss_mask` 把前 3 步关掉——33 课
        说好的"mask 是 BPTT 的标配"，这里第一次真用上。
        """
    )
    return


@app.cell
def _(bptt34, forward_rnn34, nll34, np, plt):
    # 演示 3 · 延迟回声训练（delay=3）
    DELAY34 = 3
    _r34c = np.random.default_rng(42)
    _T3 = 30
    _xs3 = np.eye(3)[_r34c.integers(0, 3, _T3)]                     # 随机字符流
    _ys3 = np.concatenate([_r34c.integers(0, 3, DELAY34),
                           np.array(_xs3[:-DELAY34]).argmax(1)])     # 目标 = 3 步前
    _mask3 = np.ones(_T3, bool); _mask3[:DELAY34] = False           # 前 3 步不算账

    _rc = np.random.default_rng(0)
    _d_h3 = 16
    _Wx3 = _rc.normal(0, 1 / np.sqrt(3), (_d_h3, 3))
    _Wh3 = _rc.normal(0, 1 / np.sqrt(_d_h3), (_d_h3, _d_h3))
    _bh3 = np.zeros(_d_h3)
    _Wy3 = _rc.normal(0, 1 / np.sqrt(_d_h3), (3, _d_h3))
    _by3 = np.zeros(3)
    _plist = [_Wx3, _Wh3, _bh3, _Wy3, _by3]
    _m3 = [np.zeros_like(_p) for _p in _plist]
    _v3 = list(_m3)
    _losses34 = []
    for _step in range(1, 301):
        _g3, _ = bptt34(_xs3, _ys3, _Wx3, _Wh3, _bh3, _Wy3, _by3, loss_mask=_mask3)
        for _i, (_gp, _p) in enumerate(zip((_g3["Wx"], _g3["Wh"], _g3["bh"], _g3["Wy"], _g3["by"]), _plist)):
            _m3[_i] = 0.9 * _m3[_i] + 0.1 * _gp
            _v3[_i] = 0.999 * _v3[_i] + 0.001 * _gp * _gp
            _p -= 0.01 * (_m3[_i] / (1 - 0.9 ** _step)) / (np.sqrt(_v3[_i] / (1 - 0.999 ** _step)) + 1e-8)
        _hs3, _, _P3 = forward_rnn34(_xs3, _Wx3, _Wh3, _bh3, _Wy3, _by3)
        _losses34.append(-np.log(_P3[np.arange(_T3)[_mask3], _ys3[_mask3]] + 1e-12).mean())

    _acc3 = float((_P3.argmax(1)[_mask3] == _ys3[_mask3]).mean())
    print(f"延迟回声（delay={DELAY34}）：loss {np.log(3):.2f}（瞎猜线 ln3）"
          f" → {_losses34[-1]:.4f}（300 轮）")
    print(f"考试（复述 3 步前的字符）：{_acc3:.0%}")
    _pred = "abc"[:0]
    print("\n抽查最后 10 步（目标 vs 预测）：")
    for _t in range(_T3 - 10, _T3):
        _mark = "✓" if _P3[_t].argmax() == _ys3[_t] else "✗"
        print(f"  t={_t:2d}：目标 {chr(97+_ys3[_t])}｜预测 {chr(97+_P3[_t].argmax())} {_mark}")

    _fig, _ax = plt.subplots(figsize=(8.4, 4.0))
    _ax.plot(_losses34, color="tab:blue", lw=1.8)
    _ax.axhline(np.log(3), color="gray", ls="--", lw=1.2)
    _ax.text(6, np.log(3) + 0.03, "瞎猜线 ln3 ≈ 1.10", fontsize=9, color="gray")
    _ax.set_xlabel("训练轮数")
    _ax.set_ylabel("交叉熵损失（回声段）")
    _ax.set_title(f"延迟回声训练：BPTT 把「记住 3 步前」教给了 RNN（准确率 {_acc3:.0%}）")
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 4 · 梯度体检：远处的 U，还收得到工资吗

        **只看最后一步损失**（mask 只开 T），梯度只能从第 T 步
        沿 h 链倒流——这是观察"连乘命运"的最纯净设定。
        `per_step` 记录每个时间步给 U 的那份贡献
        （‖δₜ ⊗ hₜ₋₁‖），画成半对数折线。

        为了让结论干净，实验做了一个控制变量：**把 Wx 缩小到
        0.3 倍**，让 h 一直住在 tanh 的线性区（|h| 均值 0.23，
        tanh′ ≈ 1）——排除"饱和"这头怪兽的干扰（35 课再放
        它出来）。这样梯度每步的衰减几乎只由 ρ(U) 决定：

        - **α = 0.5**：从损失端往回走 **6 步**，贡献衰减到
          1% 以下——远处的 U 使用基本收不到工资
        - **α = 1.0**：拖到 **18 步**——长尾
        - **α = 1.3**：整条曲线几乎不衰减，甚至**往回走先
          变大**（0.55 → 0.87）——乘子已经越过 1，离爆炸
          只差一脚油门

        这张图就是 35 课的预告片：**消失与爆炸不是两种病，
        是同一条连乘公式的两种命运**（乘子普遍 <1 还是 >1）。
        下面用交互实验把旋钮拧一遍。
        """
    )
    return


@app.cell
def _(bptt34, np, plt):
    # 演示 4 · 梯度时间分解（只看最后一步损失；Wx×0.3 让 h 住在线性区）
    _rng34d = np.random.default_rng(1)
    V34d, d_h34d, T34d = 3, 8, 20
    _xsd = np.eye(V34d)[_rng34d.integers(0, V34d, T34d)]
    _ysd = _rng34d.integers(0, V34d, T34d)
    _mask_last = np.zeros(T34d, bool); _mask_last[-1] = True

    # 本演示专用参数：与演示 2 同款配方，但 Wx 缩小到 0.3 倍——
    # h 保持在 tanh 线性区（|h| 均值约 0.23），排除饱和的干扰，
    # 让衰减快慢只由 ρ(U) 决定（35 课再把饱和这头怪兽放出来）
    _rd4 = np.random.default_rng(2)
    Wx34d = _rd4.normal(0, 1 / np.sqrt(V34d), (d_h34d, V34d)) * 0.3
    Wh34d = _rd4.normal(0, 1 / np.sqrt(d_h34d), (d_h34d, d_h34d))
    bh34d = np.zeros(d_h34d)
    Wy34d = _rd4.normal(0, 1 / np.sqrt(d_h34d), (V34d, d_h34d))
    by34d = np.zeros(V34d)

    def scaled_Wh34(alpha):
        """同一个随机 U，缩放 α（记忆强度）。"""
        return Wh34d * alpha

    alpha_list34 = (0.5, 1.0, 1.3)
    curves34 = {}
    print(f"只看最后一步损失（T={T34d}），各时间步给 U 的贡献范数：")
    for _a in alpha_list34:
        _Wh_a = scaled_Wh34(_a)
        _, _per = bptt34(_xsd, _ysd, Wx34d, _Wh_a, bh34d, Wy34d, by34d,
                         loss_mask=_mask_last)
        curves34[_a] = _per
        _rev = _per[::-1]                                     # 从损失端往远处
        _n1 = next((_i for _i, _v in enumerate(_rev) if _v < _rev[0] * 0.01), None)
        print(f"  α={_a}: 损失端贡献 {_rev[0]:.3f} → 走 {(_n1 if _n1 is not None else T34d)} 步衰减到 1% 以下")

    _fig, _ax = plt.subplots(figsize=(8.8, 4.4))
    for _a, _c in zip(alpha_list34, ("tab:green", "tab:blue", "tab:red")):
        _ax.semilogy(range(1, T34d + 1), np.maximum(curves34[_a], 1e-8),
                     lw=2, color=_c, marker="o", ms=3, label=f"α = {_a}")
    _ax.set_xlabel("时间步 t（越往左离损失越远）")
    _ax.set_ylabel("第 t 步给 U 的贡献（对数坐标）")
    _ax.set_title("梯度体检：从损失端往回走，远处的 U 还收得到多少梯度", fontsize=11)
    _ax.legend(fontsize=10)
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()
    _fig
    return (Wh34d, Wx34d, bh34d, by34d, curves34, scaled_Wh34, Wy34d)

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
        ### 交互实验 · 连乘命运转盘

        两条滑块：**记忆强度 α**（U 的缩放）、**序列长度 T**。
        每拖一次现场做一次"只看最后一步损失"的 BPTT，画出
        各时间步给 U 的贡献（对数坐标）。

        判据有三档：

        - **消失带**（α 小）：曲线几步内栽到 10⁻⁶——远处
          收不到工资，网络"记不住也学不会"长依赖
        - **健康带**（α ≈ 1）：长尾缓降
        - **临界/爆炸带**（α 大）：曲线平坦甚至**从右往左
          上翘**（越远越大）——梯度在长时间轴上不稳定

        建议：固定 T=20，把 α 从 0.4 一路拖到 1.4，看曲线
        怎么从"跳水"变成"躺平"再变成"翘尾"。
        """
    )
    return


@app.cell
def _(mo):
    # 交互控制：α × T
    alphaB_slider34 = mo.ui.slider(0.4, 1.4, value=1.0, step=0.05, debounce=True,
                                   label="🧠 记忆强度 α（U 的缩放）")
    Tlen_slider34 = mo.ui.slider(8, 24, value=20, step=1, debounce=True,
                                 label="📏 序列长度 T")
    mo.hstack([alphaB_slider34, Tlen_slider34])
    return (Tlen_slider34, alphaB_slider34)


@app.cell
def _(Tlen_slider34, Wh34d, Wx34d, alphaB_slider34, bptt34, bh34d, by34d, mo, np, plt, scaled_Wh34, Wy34d):
    # ===== 交互实验 · 连乘命运转盘（响应单元）=====
    _a = alphaB_slider34.value
    _T = Tlen_slider34.value
    _rI = np.random.default_rng(1)
    _xsI = np.eye(3)[_rI.integers(0, 3, _T)]
    _ysI = _rI.integers(0, 3, _T)
    _maskI = np.zeros(_T, bool); _maskI[-1] = True
    _gradsI, _perI = bptt34(_xsI, _ysI, Wx34d, scaled_Wh34(_a), bh34d,
                            Wy34d, by34d, loss_mask=_maskI)
    _rev = _perI[::-1]
    _n1 = next((_i for _i, _v in enumerate(_rev) if _v < _rev[0] * 0.01), None)
    _rise = _perI[1:_T - 2].max() > _perI[-1] * 1.3 if _T > 4 else False

    _fig, _ax = plt.subplots(figsize=(8.8, 4.2))
    _ax.semilogy(range(1, _T + 1), np.maximum(_perI, 1e-10),
                 lw=2, color="tab:purple", marker="o", ms=4)
    _ax.axhline(_perI[-1] * 0.01, color="gray", ls="--", lw=1.1)
    _ax.text(1.2, _perI[-1] * 0.013, "损失端贡献的 1% 线", fontsize=8.5, color="gray")
    _ax.set_xlabel("时间步 t（越往左离损失越远）")
    _ax.set_ylabel("第 t 步给 U 的贡献（对数坐标）")
    _ax.set_title(f"α = {_a:.2f}，T = {_T}：{'曲线往回上翘（不稳定）' if _rise else '从损失端往远处衰减'}",
                  fontsize=11)
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()

    _tail = _n1 if _n1 is not None else _T
    if _rise:
        _verdict = (f"**临界 / 爆炸带**（α = {_a:.2f}）：曲线不单调衰减——往回走先变大，"
                    "说明连乘的乘子整体 ≥ 1。梯度幅值在长时间轴上不稳定："
                    "训练时一次大步就可能把参数踢飞（loss 突然变 NaN），"
                    "35 课的急救方案叫「梯度裁剪」")
    elif _tail <= max(4, _T // 3):
        _verdict = (f"**消失带**（α = {_a:.2f}）：走 {_tail} 步贡献就衰减到 1% 以下——"
                    f"序列后 {_T - _tail} 步的 U 使用几乎收不到梯度。"
                    "不是学得慢，是**信号根本到不了**：35 课的主角登场")
    else:
        _verdict = (f"**健康带**（α = {_a:.2f}）：贡献曲线长尾缓降，"
                    f"走 {_tail} 步才衰减到 1% 以下——梯度能照到整条序列。"
                    "这是随机权重的好运气；训练中的 RNN 能不能守住这条带，"
                    "35 课见")
    mo.vstack([
        _fig,
        mo.callout(mo.md(_verdict), kind="info"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✏️ 动手练习（本课是数学心脏，加量 4 题）": mo.md(
                """
                **练习 1 · 手算 tanh′**：两个 z 值（0.6 和 2.5），
                各算 dh/dz = 1 − tanh(z)²（保留两位）——第二个
                会让你亲眼看到"饱和区梯度接近死"。

                **练习 2 · 补全 BPTT 递推**：两路汇合之后，过
                tanh′ 闸门的那一行。

                **练习 3 · 补全数值梯度**：中心差分的那一行
                （22 课的老朋友），补完跑对账。

                **练习 4 · 概念题**：U 的总梯度为什么是"各时间步
                贡献之和"？（三选一）
                """
            )
        }
    )
    return


@app.cell
def _(mo, np):
    # ===== 练习 1 · 手算 tanh' =====
    print("tanh 的局部导数：dh/dz = 1 − tanh(z)²")
    print(f"  z = 0.6：tanh(0.6) = {np.tanh(0.6):.4f} → dh/dz = ?")
    print(f"  z = 2.5：tanh(2.5) = {np.tanh(2.5):.4f} → dh/dz = ?")

    dtanh_mid_ex1_34 = None                    # ← ① z=0.6 的 dh/dz（两位小数）
    dtanh_sat_ex1_34 = None                    # ← ② z=2.5 的 dh/dz（两位小数）

    if dtanh_mid_ex1_34 is not None and dtanh_sat_ex1_34 is not None:
        _t1, _t2 = 1 - np.tanh(0.6) ** 2, 1 - np.tanh(2.5) ** 2
        if abs(dtanh_mid_ex1_34 - _t1) < 0.005 and abs(dtanh_sat_ex1_34 - _t2) < 0.005:
            _msg = (f"✅ 正确：{_t1:.2f} 和 {_t2:.2f}。z=2.5 时 tanh 已经爬到 "
                    f"{np.tanh(2.5):.3f}（接近满格 1），局部导数只剩 {_t2:.2f}——"
                    "饱和区里梯度被 (1−h²) 掐到近乎断流。35 课：饱和是梯度消失的两大帮凶之一")
        else:
            _msg = (f"❌ 参考：{_t1:.2f} / {_t2:.2f}。"
                    "公式 dh/dz = 1 − tanh(z)²，tanh 的值上面已经给出")
    else:
        _msg = "⏳ 两空都填上（tanh 值已给出，只差平方相减）"
    mo.md(_msg)
    return (dtanh_mid_ex1_34, dtanh_sat_ex1_34)


@app.cell
def _(mo, np):
    # ===== 练习 2 · 补全 BPTT 递推 =====
    def bptt_one_step_ex2_34(dlog_t, delta_next, hs_t, Wy, Wh):
        """BPTT 的单个时间步（倒序循环体）。
        dlog_t: 损失对 logits 的梯度；delta_next: 未来传回来的 δ；
        hs_t: 本步的 h（缓存的前向量）。返回本步的 δ。"""
        _dh = Wy.T @ dlog_t + Wh.T @ delta_next      # 两路汇合（已写好）
        _delta = None                                # ← 补：过 tanh' 闸门（用 hs_t 表示局部导数）
        return _delta

    bptt_ready_ex2_34 = None                        # ← 补完后，把这行换成 bptt_one_step_ex2_34

    if bptt_ready_ex2_34 is not None:
        try:
            _r = np.random.default_rng(3)
            _Wy2 = _r.normal(0, 0.4, (3, 5)); _Wh2 = _r.normal(0, 0.3, (5, 5))
            _dlog2 = _r.normal(0, 0.5, 3); _dn2 = _r.normal(0, 0.5, 5)
            _hs2 = np.tanh(_r.normal(0, 0.5, 5))
            _got = bptt_ready_ex2_34(_dlog2, _dn2, _hs2, _Wy2, _Wh2)
            _ref = (_Wy2.T @ _dlog2 + _Wh2.T @ _dn2) * (1 - _hs2 ** 2)
            if _got is not None and np.allclose(_got, _ref, atol=1e-12):
                _msg = ("✅ 逐元素一致！δ = 汇合梯度 × (1−h²)——"
                        "缓存下来的 hs_t 在这里派上用场：局部导数不用重算 tanh，"
                        "直接用 h² 表示（tanh(z) = h）")
            else:
                _msg = "❌ 与参考不一致：检查是不是漏乘了 (1 − hs_t ** 2)"
        except (TypeError, ValueError) as _e:
            _msg = f"❌ 运行出错：{_e}——一行：_dh * (1 - hs_t ** 2)"
    else:
        _msg = "⏳ 补一行：_dh * (1 - hs_t ** 2)"
    mo.md(_msg)
    return (bptt_one_step_ex2_34, bptt_ready_ex2_34)


@app.cell
def _(mo, np):
    # ===== 练习 3 · 补全数值梯度（中心差分）=====
    def num_grad_ex3_34(f, theta, eps=1e-5):
        """f: 以标量参数为自变量的损失函数；theta: 当前参数值。"""
        _f1 = f(theta + eps)
        _f2 = f(theta - eps)
        return None                                   # ← 补：中心差分一行

    numgrad_ready_ex3_34 = None                      # ← 补完后，把这行换成 num_grad_ex3_34

    if numgrad_ready_ex3_34 is not None:
        try:
            _f_sq = lambda _t: (_t - 1.234) ** 2          # 解析导数 2(t−1.234)
            _err = max(abs(numgrad_ready_ex3_34(_f_sq, _t) - 2 * (_t - 1.234))
                       for _t in (0.0, 0.5, 2.0))
            if _err < 1e-8:
                _msg = (f"✅ 最大误差 {_err:.1e}（对二次函数逐点核对）。"
                        "中心差分比单侧差分（f(θ+ε)−f(θ))/ε）精度高一个量级，"
                        "这就是 22 课以来对账仪式都用它的原因")
            else:
                _msg = f"❌ 误差偏大（{_err:.1e}）：检查分母是不是 2 * eps"
        except (TypeError, ZeroDivisionError) as _e:
            _msg = f"❌ 运行出错：{_e}——(_f1 - _f2) / (2 * eps)"
    else:
        _msg = "⏳ 补一行：(_f1 - _f2) / (2 * eps)"
    mo.md(_msg)
    return (num_grad_ex3_34, numgrad_ready_ex3_34)


@app.cell
def _(mo):
    # ===== 练习 4 · 概念题 =====
    print("为什么 U 的总梯度是「各时间步贡献之和」？")
    print("  A. U 在 T 个时间步各被使用一次，每次使用都按链式法则收到一份梯度")
    print("  B. 因为梯度在 T 步之间平均分配，每步拿 1/T")
    print("  C. 因为 U 是方阵，方阵的梯度天然要按行求和")

    ans_ex4_34 = None                                # ← ① 填 'A' / 'B' / 'C'

    if ans_ex4_34 is not None:
        if ans_ex4_34 == "A":
            _msg = ("✅ 正确。同一个 U 被展开成 T 份「使用」，每份使用都是计算图上的"
                    "一个节点，各自按链式法则收到梯度，最后加总（23 课 MLP 权重共享"
                    "的老规矩）。对账小例里可以看到：dWh 就是循环里 T 次 `+=` 的累加结果")
        else:
            _msg = "❌ 再想想：展开图上 U 出现了几次？每个出现的位置对损失都有影响吗？"
    else:
        _msg = "⏳ 填一个选项（'A' / 'B' / 'C'）"
    mo.md(_msg)
    return (ans_ex4_34,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 1 参考答案（先自己写过再看！）": mo.md(
                """
                **z = 0.6：dh/dz = 1 − 0.5370² ≈ 0.71。**
                **z = 2.5：dh/dz = 1 − 0.9866² ≈ 0.03。**

                tanh 的局部导数在 z=0 处取最大值 1，往两边
                迅速滑落：z=0.6 还剩 0.71，z=2.5 只剩 0.03
                ——**离中心 2.5 个单位，梯度就只剩 3%**。

                这意味着：一个 h 值接近 ±1（饱和）的时间步，
                是梯度通路上的**窄闸门**——连续几步饱和，梯度
                就被连续乘以 0.0x，很快归零。35 课会把它和
                ρ(U)<1 并列为梯度消失的两大帮凶。
                """
            ),
            "✅ 练习 2 参考答案（先自己写过再看！）": mo.md(
                """
                一行：

                ```python
                _delta = _dh * (1 - hs_t ** 2)
                ```

                两个细节值得咀嚼：

                - 局部导数**用缓存下来的 h 表示**（tanh(z) = h，
                  所以 1 − tanh(z)² = 1 − h²）——不需要保存 z，
                  也不需要重算 tanh。这就是"前向缓存"的意义
                - 汇合那行 `_dh = Wyᵀ·dlog + Uᵀ·δ_next` 是
                  **加法**不是乘法：梯度流到分叉点，**分流后
                  各走各的、互不衰减**（链式法则对加法的规定）
                """
            ),
            "✅ 练习 3 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                return (_f1 - _f2) / (2 * eps)
                ```

                中心差分的截断误差是 O(ε²)，单侧差分是 O(ε)——
                同样的 ε=10⁻⁵，中心差分能稳稳拿到 10⁻⁸~10⁻⁹
                级别的对账精度（演示 2 实测最差 10⁻⁸）。

                ε 不能太大（截断误差）也不能太小（浮点舍入），
                对 float64 的典型损失，10⁻⁵ 附近的甜点区
                宽得很——22 课以来我们一直用它。
                """
            ),
            "✅ 练习 4 参考答案（先自己写过再看！）": mo.md(
                """
                **选 A。**

                展开之后，"一个 U"变成了"T 个使用位置"——
                计算图上每个位置都连着损失（经由不同的路径）。
                链式法则要求：参数的梯度 = 它**所有影响路径**
                的梯度之和。所以代码里是 `dU += δₜ ⊗ hₜ₋₁`
                累加 T 次，而不是平均或分配。

                同样的逻辑你见过三次了：23 课 MLP 里被共享的
                权重、26 课扫全图的卷积核、今天的 U——
                **权值共享的结构，梯度永远是"各使用点贡献
                之和"**。第 7 部分的 Transformer 里，同一个
                W_q 被 48 课里所有位置共享时，还是这条规矩。
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
                1. **展开图就是计算图**：BPTT 不是新算法——
                   22 课链式法则在"深度为 T 的共享权重前馈网"
                   上的应用
                2. **三条路径**：Why 路（一步直达）、W 路
                   （本地直达）、U 路（要连乘 ∏U·diag(1−h²)）
                   ——连乘只发生在 U 路
                3. **倒序单循环**：δₜ = (Wyᵀ·dlogₜ + Uᵀ·δₜ₊₁)
                   ⊙ (1−hₜ²)，从 t=T 往 t=1 走一遍；缓存、
                   倒序、累加三要点
                4. **对账过关**：解析梯度 vs 中心差分数值梯度，
                   五张量最差 10⁻⁸——和 22/23/29 课同一门槛
                5. **训练验证**：延迟回声任务（yₜ = xₜ₋₃），
                   300 轮 Adam 从 ln3 掉到贴地、准确率 100%——
                   BPTT 真的把"记住 3 步前"教给了 RNN
                6. **梯度体检**：只看最后一步损失，U 的各步
                   贡献随距离衰减（α=0.5 六步衰到 1%）或
                   上翘（α=1.3）——消失与爆炸是同一条连乘
                   公式的两种命运
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

                今天留下的最后一幕值得回放：α = 0.5 时，梯度
                从损失端往回走 **6 步就衰减到 1%**——第 7 步
                以前的 U 使用，几乎收不到任何学习信号。

                这不是 α=0.5 的特例。把公式摆开看：

                ```
                ∂h_T/∂h_t = ∏ (U · diag(1−h²))    ← 连乘 T−t 次
                ```

                只要这套乘子的"平均尺寸"普遍小于 1——乘积就
                **指数级衰减**；普遍大于 1，乘积**指数级爆炸**。
                前者叫**梯度消失**，后者叫**梯度爆炸**，合称
                RNN 的时间病。它俩的共同根子只有一个：**梯度
                要走太远的路，而每一步都要乘点什么。**

                坏消息：语言里的长依赖（一句话开头的人名，三十
                个字后被代词引用）动辄几十步——普通 RNN 的梯度
                根本走不到。好消息：爆炸有急救包（一行代码的
                梯度裁剪），消失有根治手术（36 课的 LSTM）。
                """
            ),
            mo.callout(
                mo.md(
                    """
                    **下一课预告** — `35_梯度消失与爆炸`：

                    把 34 课的连乘公式推到极端，全程数值实验：

                    - **标量连乘图**：0.5⁵⁰ ≈ 10⁻¹⁵（消失）、
                      1.05⁵⁰ ≈ 10（爆炸）——临界线 1 的两侧
                      是两个宇宙
                    - **loss 尖峰现场**：真训练一个小 RNN，
                      亲眼看到 loss 突然窜天（NaN 前兆），
                      再亲手用一行 clip 把它救回来
                    - **长依赖任务**：间隔 20 步的回声——
                      普通 RNN 卡死在瞎猜线，谁也救不了
                      （36 课 LSTM 的出场预告）

                    病理报告写完，药方（LSTM）自然就懂了。

                    📉 上坡下坡，都是指数的命！
                    """
                ),
                kind="success",
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
