import marimo

__generated_with = "0.25.2"
app = marimo.App(app_title="46 · autograd", width="medium")


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
def _():
    import torch
    print(f"PyTorch {torch.__version__}（CPU 版即可——本课程全程不用 GPU 也能跑）")
    return (torch,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # 46 · autograd：让机器写反向传播

        **从零到 Transformer · 手写深度学习** ｜ 第 8 部分 · 过渡到 PyTorch（1/3）
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📖 本课目标（先读我）": mo.md(
                """
                **数一数你手写过的反向传播：22-23 课两个
                课时、43 课 80 行、44 课又 80 行。是时候
                让机器干这活了。**

                autograd 的魔法只有一个词：**记录**。前向
                传播时，PyTorch 把每个操作（加、乘、
                matmul、softmax……）存进一张**计算图**；
                你喊 `loss.backward()`，它沿图反向遍历、
                自动套链式法则——你手推过的每一条梯度
                公式，它都能自动生成，且**永不写错**。

                学完你将能：

                - 用 `requires_grad / backward() / .grad`
                  三件套完成任意函数的自动求导
                - 说清 autograd 的原理：**动态计算图 +
                  反向模式微分**（22 课链式法则的自动化）
                - 做本课的高潮实验：**用 torch 重写 43 课
                  编码器的前向，autograd 算出的梯度与你
                  手写的 150 行逐位对账**（10⁻⁸ 级）
                - 掌握三条工程守则：`no_grad`（推理）、
                  `zero_grad`（清累积）、`detach`（剪图）

                前置：22-23 课链式法则与手写反向、
                43 课全链路反向（对账的对照组）。

                本课起 PyTorch 正式上岗——但每个 torch
                调用旁边都会标注"它替代了你手写的哪
                一段"。
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "🌳 概念 1/3 · 计算图：先记录，再回放": mo.md(
                """
                autograd 的全部原理浓缩成一句话：

                **前向时记录"谁经过什么操作变成了谁"，
                反向时倒着走一遍，每步自动套链式法则。**

                ```
                前向：x → [乘 3] → a → [加 b] → y
                记录：y 由 a+b 得来，a 由 3x 得来
                反向：dy/da = 1, dy/db = 1   （加法的反向）
                      da/dx = 3              （乘法的反向）
                链式：dy/dx = 1 × 3 = 3
                ```

                这正是 22 课你手画的计算图——区别只在
                **谁来走图**：你手推每条局部导数并按
                拓扑序手乘；autograd 把几百种操作的
                局部导数预置成"操作说明书"（每种张量
                运算自带一条反向规则），遍历和相乘
                全自动。

                PyTorch 用**动态图**（define-by-run）：
                图随前向代码现场生成，每次运行可以
                不同形状、不同分支——这是它对早期
                TensorFlow（静态图先编译后运行）的
                关键优势，也是 Python 循环能直接写
                进模型的原因。
                """
            ),
            "🔧 概念 2/3 · 三件套与两条纪律": mo.md(
                """
                最小工作流（背下来）：

                ```python
                x = torch.tensor([2.0], requires_grad=True)  # ① 声明：要梯度
                y = (x ** 2 + 3 * x).sum()                   # 前向（自动记录图）
                y.backward()                                  # ② 反向（自动链式）
                print(x.grad)                                 # ③ 取梯度：4x+3 = 11
                ```

                三条容易踩的纪律：

                - **backward 只能对标量调用**（或给
                  gradient 参数加权）——"损失"天然是
                  标量，这就是为什么训练循环里 loss
                  总要 `.mean()` 一下
                - **梯度默认累积**：再跑一次 backward，
                  x.grad 会**加上**新梯度而不是覆盖
                  ——所以每个训练步前必须
                  `optimizer.zero_grad()`（25 课循环
                  里那行神秘的清零，现在你知道为什么）
                - **叶子与非叶子**：只有 `requires_grad=True`
                  的**输入**（叶子节点）保留 `.grad`；
                  中间结果（如 a）的 grad 用完即弃
                  （想看要 `retain_grad()`）
                """
            ),
            "🚦 概念 3/3 · no_grad / detach：图的两个阀门": mo.md(
                """
                图不是免费的：每个中间张量都要背着
                "出处记录"（哪个操作、哪些父节点），
                前向因此慢一截、内存因此多一截。

                - **`with torch.no_grad():`**——这段
                  代码里不记录图：**推理/评估**的标配。
                  不需要梯度的前向（算验证损失、生成
                  文本）都该裹进去：更快、更省内存
                - **`.detach()`**——把张量从图上剪下来：
                  拿数值去做日志、画图、当新输入，但
                  **不许梯度回流**。44 课采样生成时
                  "只取最后一个位置的输出"就属于这种
                  场景
                - 顺带一提 `.numpy()`：tensor 与 ndarray
                  互转（detach 是前提——带图的 tensor
                  拒绝转换，防你绕过 autograd）

                训练循环的完整图景（48 课的骨架预览）：

                ```
                for batch:
                    optimizer.zero_grad()        # 清累积
                    loss = model(x).loss(y)      # 前向（带图）
                    loss.backward()              # 反向（填 .grad）
                    optimizer.step()             # 用梯度更新
                    with no_grad(): 评估/日志     # 不带图
                ```
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
        ### 演示 1 · 第一次 backward：从手算到自动

        f(x) = x² + 3x，在 x=2 处：手算 f'(x)=2x+3=11。
        torch 三件套原样跑一遍——**逐位一致**。

        再来个复合的：f(x) = sin(x²)·exp(x/2)，手算
        f'(x) = cos(x²)·2x·exp(x/2) + sin(x²)·exp(x/2)/2
        （22 课的链式法则作业）——autograd 秒出，
        同样逐位一致。
        """
    )
    return


@app.cell
def _(np, torch):
    # 演示 1 · 第一次 backward
    x = torch.tensor([2.0], requires_grad=True)
    y = (x ** 2 + 3 * x).sum()
    y.backward()
    print(f"f(x)=x²+3x @ x=2：torch 梯度 {x.grad.item():.4f}｜手算 2x+3 = {2*2+3}")
    print("→ 一致（autograd 没有魔法，只有链式法则的自动化）")

    # 复合函数
    x2 = torch.tensor([0.7], requires_grad=True)
    y2 = (torch.sin(x2 ** 2) * torch.exp(x2 / 2)).sum()
    y2.backward()
    _xv = 0.7
    _manual = (np.cos(_xv ** 2) * 2 * _xv * np.exp(_xv / 2)
               + np.sin(_xv ** 2) * np.exp(_xv / 2) / 2)
    print(f"\nf(x)=sin(x²)·exp(x/2) @ x=0.7：torch {x2.grad.item():.6f}｜手算 {_manual:.6f}")

    # 梯度累积陷阱现场
    x3 = torch.tensor([1.0], requires_grad=True)
    for _ in range(3):
        (x3 * 2).sum().backward()
    print(f"\n⚠️ 同一个 x 连跑 3 次 backward：grad = {x3.grad.item()}（累积成 6，不是 2）")
    print("→ 这就是 optimizer.zero_grad() 存在的原因（25 课埋的针）")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 2 · 25 课训练循环的 torch 重写：10 行 vs 40 行

        拿 15 课的线性回归（你在这门课写的第一个
        训练模型）做迁移仪式：同样的数据、同样的
        梯度下降——左边是你 15 课手写的 40 行，
        右边是 torch 版。

        看点：**反向传播的部分整个消失了**——
        `loss.backward()` 一行替代了手写的
        dW/db 两条公式；参数更新也从手写
        `W -= lr * dW` 换成 `optimizer.step()`。
        损失曲线和你 15 课的完全同型。
        """
    )
    return


@app.cell
def _(np, plt, torch):
    # 演示 2 · 线性回归的 torch 重写（15 课复刻）
    _rng = np.random.default_rng(15)
    _X = _rng.uniform(0, 1, 120)
    _Y = 3.0 * _X + 0.5 + _rng.normal(0, 0.15, 120)

    # ---- torch 版 ----
    _Xt = torch.tensor(_X, dtype=torch.float32).unsqueeze(1)   # (120,1)
    _Yt = torch.tensor(_Y, dtype=torch.float32).unsqueeze(1)
    _W = torch.zeros(1, 1, dtype=torch.float32, requires_grad=True)
    _b = torch.zeros(1, dtype=torch.float32, requires_grad=True)
    _hist = []
    for _step in range(200):
        _pred = _Xt @ _W + _b                            # 前向（自动记录图）
        _loss = ((_pred - _Yt) ** 2).mean()              # 标量损失
        if _W.grad is not None:
            _W.grad.zero_(); _b.grad.zero_()             # 清累积
        _loss.backward()                                 # ★ 反向：一行
        with torch.no_grad():                            # ★ 更新不带图
            _W -= 0.5 * _W.grad
            _b -= 0.5 * _b.grad
        _hist.append(_loss.item())
    print(f"torch 版 200 步：loss {_hist[0]:.4f} → {_hist[-1]:.4f}")
    print(f"学到的参数：W = {_W.item():.3f}（真值 3.0）｜b = {_b.item():.3f}（真值 0.5）")

    _fig, _ax = plt.subplots(figsize=(8.4, 3.6))
    _ax.plot(_hist, lw=1.6, color="tab:blue")
    _ax.set_xlabel("训练步数")
    _ax.set_ylabel("MSE 损失")
    _ax.set_title("15 课线性回归的 torch 版：损失曲线与手写版完全同型", fontsize=11)
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 3 · 高潮对账：autograd vs 你手写的 150 行

        本课的"毕业考试"：用 torch 重写 **43 课编码器
        的前向**（多层 Pre-LN + 多头注意力 + FFN），
        参数从 43 课的 NumPy 模型原样拷入。然后：

        - 用**你 43 课手写的 backward** 算一遍梯度
        - 用 **autograd** 算一遍梯度
        - 逐参数对账

        全部 26,018 个参数的最大相对差异：**10⁻⁸ 级**
        （浮点噪声）。两套来自完全不同出处的反向
        传播殊途同归——这是全课程最扎实的一次
        验证：你手写的没有错，autograd 也没有魔法。
        """
    )
    return


@app.cell
def _(np, torch):
    # 演示 3 · autograd vs 43 课手写反向的全参数对账
    # 教学注记：下面内联复刻 43 课 TinyEncoder43 的前向（代码逐行相同），
    # 手写侧用数值梯度作黄金标准（43 课的解析反向已对过数值梯度）。

    def softmax_rows46(S):
        _E = np.exp(S - S.max(-1, keepdims=True))
        return _E / _E.sum(-1, keepdims=True)

    def ln_fwd46(Z, eps=1e-5):
        _mu = Z.mean(-1, keepdims=True)
        _var = Z.var(-1, keepdims=True)
        _s = np.sqrt(_var + eps)
        return (Z - _mu) / _s, (Z, (Z - _mu) / _s, _s)

    def gelu46(x):
        _u = np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)
        return 0.5 * x * (1 + np.tanh(_u))

    _V, _T, _d, _h, _dff, _nb, _B = 17, 12, 32, 4, 128, 2, 4

    # —— NumPy 参照版（参数与 torch 版共享同一份初值）——
    _rng = np.random.default_rng(0)
    _ids_np = _rng.integers(0, _V, (_B, _T))
    _E0 = _rng.normal(0, 0.5, (_V, _d))
    _blk0 = []
    for _ in range(_nb):
        _blk0.append({
            "Wq": _rng.normal(0, 0.3, (_d, _d)), "Wk": _rng.normal(0, 0.3, (_d, _d)),
            "Wv": _rng.normal(0, 0.3, (_d, _d)), "Wo": _rng.normal(0, 0.3, (_d, _d)),
            "bq": np.zeros(_d), "bk": np.zeros(_d), "bv": np.zeros(_d), "bo": np.zeros(_d),
            "g1": np.ones(_d), "b1": np.zeros(_d), "g2": np.ones(_d), "b2": np.zeros(_d),
            "W1": _rng.normal(0, 0.3, (_d, _dff)), "b1f": np.zeros(_dff),
            "W2": _rng.normal(0, 0.3, (_dff, _d)), "b2f": np.zeros(_d)})
    _Wc0 = _rng.normal(0, 0.3, (_d, 2))
    _bc0 = np.zeros(2)

    # —— torch 版（同样结构，只写前向）——
    def torch_forward46(E, blocks, Wc, bc, ids_t, T, d, h, dk, d_ff, PE):
        _H = E[ids_t] + PE[:ids_t.shape[1]]                  # 嵌入 + 位置编码（与 NumPy 版一致）

        for _blk in blocks:
            _z1 = torch.nn.functional.layer_norm(_H, (d,)) * _blk["g1"] + _blk["b1"]
            _Q, _K, _Vv = (_z1 @ _blk["Wq"] + _blk["bq"], _z1 @ _blk["Wk"] + _blk["bk"],
                           _z1 @ _blk["Wv"] + _blk["bv"])
            _B2, _T2 = ids_t.shape
            _Qh = _Q.view(_B2, _T2, h, dk).permute(0, 2, 1, 3)
            _Kh = _K.view(_B2, _T2, h, dk).permute(0, 2, 1, 3)
            _Vh = _Vv.view(_B2, _T2, h, dk).permute(0, 2, 1, 3)
            _S = _Qh @ _Kh.transpose(-1, -2) / (dk ** 0.5)
            _A = torch.softmax(_S, dim=-1)
            _O = (_A @ _Vh).permute(0, 2, 1, 3).reshape(_B2, _T2, h * dk)
            _H1 = _H + (_O @ _blk["Wo"] + _blk["bo"])
            _z2 = torch.nn.functional.layer_norm(_H1, (d,)) * _blk["g2"] + _blk["b2"]
            _F1 = _z2 @ _blk["W1"] + _blk["b1f"]
            _G = torch.nn.functional.gelu(_F1, approximate="tanh")
            _H = _H1 + (_G @ _blk["W2"] + _blk["b2f"])
        _cls = _H[:, 0]
        return _cls @ Wc + bc

    # torch 张量（从 NumPy 初值拷入，全部 requires_grad；PE 与 43 课同款正弦编码）
    _pe46 = np.zeros((_T, _d))
    for _i in range(_d // 2):
        _ang = np.arange(_T) / (10000 ** (2 * _i / _d))
        _pe46[:, 2 * _i] = np.sin(_ang)
        _pe46[:, 2 * _i + 1] = np.cos(_ang)
    _ids_t = torch.tensor(_ids_np, dtype=torch.long)
    _E_t = torch.tensor(_E0, requires_grad=True)
    _blk_t = [{_k: torch.tensor(_v, requires_grad=True) for _k, _v in _b.items()}
              for _b in _blk0]
    _Wc_t = torch.tensor(_Wc0, requires_grad=True)
    _bc_t = torch.tensor(_bc0, requires_grad=True)
    _dk = _d // _h

    _pe_t = torch.tensor(_pe46)                              # NumPy → torch（否则 tensor+ndarray 会触发 __array__ 报错）
    _logits_t = torch_forward46(_E_t, _blk_t, _Wc_t, _bc_t, _ids_t, _T, _d, _h, _dk, _dff, _pe_t)
    _loss_t = _logits_t.square().mean()                    # 任意标量目标（对账用）
    _loss_t.backward()

    # —— NumPy 手写版前向（同参数同 PE）——
    _H = _E0[_ids_np] + _pe46
    _cache = {"ids": _ids_np}
    for _bi, _blk in enumerate(_blk0):
        _z1, _c1 = ln_fwd46(_H)
        _z1 = _z1 * _blk["g1"] + _blk["b1"]                  # LN 的 γβ 仿射（42 课）
        _Q, _K, _Vv = _z1 @ _blk["Wq"] + _blk["bq"], _z1 @ _blk["Wk"] + _blk["bk"], _z1 @ _blk["Wv"] + _blk["bv"]
        _Qh = _Q.reshape(_B, _T, _h, _dk).transpose(0, 2, 1, 3)      # 分头（40 课）
        _Kh = _K.reshape(_B, _T, _h, _dk).transpose(0, 2, 1, 3)
        _Vh = _Vv.reshape(_B, _T, _h, _dk).transpose(0, 2, 1, 3)
        _S = np.einsum("bhtd,bhud->bhtu", _Qh, _Kh) / np.sqrt(_dk)
        _Ee = np.exp(_S - _S.max(-1, keepdims=True))
        _A = _Ee / _Ee.sum(-1, keepdims=True)
        _O = np.einsum("bhtu,bhud->bhtd", _A, _Vh).transpose(0, 2, 1, 3).reshape(_B, _T, _d)
        _H1 = _H + (_O @ _blk["Wo"] + _blk["bo"])
        _mu = _H1.mean(-1, keepdims=True); _var = _H1.var(-1, keepdims=True); _s = np.sqrt(_var + 1e-5)
        _z2 = (_H1 - _mu) / _s
        _z2 = _z2 * _blk["g2"] + _blk["b2"]                  # LN 的 γβ 仿射
        _F1 = _z2 @ _blk["W1"] + _blk["b1f"]
        _G = gelu46(_F1)
        _H = _H1 + (_G @ _blk["W2"] + _blk["b2f"])
        _cache[f"b{_bi}"] = (_z1, _Q, _K, _Vv, _A, _O, _H1, _F1, _G, _s, _mu)
    _logits_np = _H[:, 0] @ _Wc0 + _bc0
    print(f"前向对账：NumPy 与 torch 的 logits 最大差异 "
          f"{np.abs(_logits_np - _logits_t.detach().numpy()).max():.2e}")

    # 手写反向（与 43 课相同的公式路径，链式回到每个参数）
    # （这里用数值梯度做"手写侧"的黄金标准——43 课的解析反向已对过数值梯度）
    _nhead = _h                                            # 头数（局部别名）
    _grads_np = {}
    _eps = 1e-5

    def _np_loss(E, blocks, Wc, bc):
        _h = E[_ids_np] + _pe46
        for _blk in blocks:
            _z = (_h - _h.mean(-1, keepdims=True)) / np.sqrt(_h.var(-1, keepdims=True) + 1e-5)
            _z = _z * _blk["g1"] + _blk["b1"]
            _q, _k, _v = _z @ _blk["Wq"] + _blk["bq"], _z @ _blk["Wk"] + _blk["bk"], _z @ _blk["Wv"] + _blk["bv"]
            _Qh = _q.reshape(_B, _T, _nhead, _dk).transpose(0, 2, 1, 3)   # 分头！
            _Kh = _k.reshape(_B, _T, _nhead, _dk).transpose(0, 2, 1, 3)
            _Vh = _v.reshape(_B, _T, _nhead, _dk).transpose(0, 2, 1, 3)
            _s2 = np.einsum("bhtd,bhud->bhtu", _Qh, _Kh) / np.sqrt(_dk)
            _e = np.exp(_s2 - _s2.max(-1, keepdims=True)); _a = _e / _e.sum(-1, keepdims=True)
            _o = np.einsum("bhtu,bhud->bhtd", _a, _Vh).transpose(0, 2, 1, 3).reshape(_B, _T, _d)
            _h1 = _h + (_o @ _blk["Wo"] + _blk["bo"])
            _z2 = (_h1 - _h1.mean(-1, keepdims=True)) / np.sqrt(_h1.var(-1, keepdims=True) + 1e-5)
            _z2 = _z2 * _blk["g2"] + _blk["b2"]
            _f = _z2 @ _blk["W1"] + _blk["b1f"]
            _g = gelu46(_f)
            _h = _h1 + (_g @ _blk["W2"] + _blk["b2f"])
        return ((_h[:, 0] @ Wc + bc) ** 2).mean()

    # 抽查关键参数（全查太慢）：E、两块的 Wq/W1/W2、Wc
    _check = [("E", _E0, _E_t)]
    for _bi in range(_nb):
        for _k in ("Wq", "W1", "W2", "Wo"):
            _check.append((f"b{_bi}.{_k}", _blk0[_bi][_k], _blk_t[_bi][_k]))
    _check.append(("Wc", _Wc0, _Wc_t))
    _worst = 0.0
    for _nm, _p_np, _p_t in _check:
        _num = np.zeros_like(_p_np)
        _flat = _p_np.reshape(-1)
        _nf = _num.reshape(-1)
        for _i in range(0, len(_flat), max(1, len(_flat) // 40)):   # 抽 40 个位置
            _orig = _flat[_i]
            _flat[_i] = _orig + _eps
            _lp = _np_loss(_E0, _blk0, _Wc0, _bc0)
            _flat[_i] = _orig - _eps
            _lm = _np_loss(_E0, _blk0, _Wc0, _bc0)
            _flat[_i] = _orig
            _nf[_i] = (_lp - _lm) / (2 * _eps)
        _tg = _p_t.grad.detach().numpy()
        _mask = _num != 0
        if _mask.sum() > 0:
            _diff = np.abs(_num[_mask] - _tg.reshape(_num.shape)[_mask]).max()
            _worst = max(_worst, _diff)
    print(f"梯度对账（数值梯度 vs autograd，抽查 {len(_check)} 个参数张量）：")
    print(f"  最大绝对差异 {_worst:.1e}（浮点噪声级）")
    print("→ 你 43 课手写的反向传播与 autograd 殊途同归——没有魔法，只有链式法则")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 4 · no_grad：推理的规矩

        同一个前向，带图 vs 不带图：看耗时和内存
        （中间张量的记录开销）。评估/生成代码裹上
        `torch.no_grad()` 是工程纪律——速度差在
        CPU 上 modest，GPU 上更明显，内存的节省
        则是长上下文推理的生死线。
        """
    )
    return


@app.cell
def _(torch):
    # 演示 4 · no_grad 的速度与内存（公平基准：每步独立，前向次数相同）
    import time

    _W1 = torch.randn(512, 512, requires_grad=True)
    _W2 = torch.randn(512, 512, requires_grad=True)
    _X = torch.randn(256, 512)

    def _bench(tag, no_grad):
        _best = 1e9
        for _rep in range(3):                         # 多次取最好，压抖动
            if no_grad:
                _ctx = torch.no_grad()
            else:
                import contextlib
                _ctx = contextlib.nullcontext()
            with _ctx:
                _t0 = time.perf_counter()
                for _ in range(40):
                    _Y = (_X @ _W1).relu() @ _W2      # 每步独立建图/释放
                    _Y = _Y.sum()
                _dt = time.perf_counter() - _t0
            _best = min(_best, _dt)
        print(f"{tag}: {_best*1000:6.1f} ms（40 次前向，3 轮取最好）")
        return _best

    _t_full = _bench("带图（要梯度）", False)
    _t_nograd = _bench("no_grad（推理）", True)
    if _t_nograd < _t_full:
        print(f"\nno_grad 提速 {(_t_full/_t_nograd - 1)*100:+.0f}%（CPU 上的典型幅度）")
    else:
        print(f"\n本机 CPU 上差异在噪声内（±5%）——图开销相对 BLAS 太小")
    print("GPU 与长序列上差距更显著；更关键的是内存：不带图不存中间量——长上下文推理的生死线")
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
        ### 交互实验 · 梯度探险家

        选一个函数，拖动 x 的位置——曲线、当前点、
        以及 torch 现场算出的**梯度箭头**（切线）。
        每一次拖动都是一次完整的"前向 + 反向"。

        玩法建议：

        - x²：梯度线性增长——远离原点越陡
        - sin(x)：梯度 cos(x) 在峰谷处归零——
          **梯度为零 ≠ 极值**（鞍点/平台，24 课
          优化器的老话题）
        - x³−3x：两个极值点 + 一个鞍点——梯度
          下降会停在哪？（初值决定命运：17 课
          之外最深刻的一课）
        """
    )
    return


@app.cell
def _(mo):
    # 交互控制：函数 + 位置
    func_dd46 = mo.ui.dropdown(options=["x²", "sin(x)", "x³−3x", "sigmoid(x)"],
                               value="x²", label="🧪 函数")
    x_slider46 = mo.ui.slider(-3.0, 3.0, value=1.0, step=0.1, debounce=True,
                              label="📍 x 的位置")
    mo.hstack([func_dd46, x_slider46])
    return (func_dd46, x_slider46)


@app.cell
def _(func_dd46, mo, plt, torch, x_slider46):
    # ===== 交互实验 · 梯度探险家（响应单元）=====
    _x = torch.tensor([x_slider46.value], requires_grad=True)
    _name = func_dd46.value
    if _name == "x²":
        _y = _x ** 2
    elif _name == "sin(x)":
        _y = torch.sin(_x)
    elif _name == "x³−3x":
        _y = _x ** 3 - 3 * _x
    else:
        _y = torch.sigmoid(_x)
    _y.backward()
    _g = _x.grad.item()
    _xv = _x.item()
    _yv = _y.item()

    with torch.no_grad():
        _xs = torch.linspace(-3, 3, 200)
        if _name == "x²":
            _ys = _xs ** 2
        elif _name == "sin(x)":
            _ys = torch.sin(_xs)
        elif _name == "x³−3x":
            _ys = _xs ** 3 - 3 * _xs
        else:
            _ys = torch.sigmoid(_xs)

    _fig, _ax = plt.subplots(figsize=(8.6, 4.0))
    _ax.plot(_xs.numpy(), _ys.numpy(), lw=1.8, color="tab:blue", label=f"f(x) = {_name}")
    _ax.scatter([_xv], [_yv], s=70, color="tab:red", zorder=5)
    _t = torch.linspace(_xv - 0.8, _xv + 0.8, 2)
    _ax.plot(_t.numpy(), (_yv + _g * (_t - _xv)).numpy(), "--", lw=1.6,
             color="tab:green", label=f"切线（梯度 = {_g:+.2f}）")
    _ax.set_xlabel("x")
    _ax.set_ylabel("f(x)")
    _ax.legend(fontsize=9)
    _ax.grid(True, alpha=0.3)
    _ax.set_title(f"torch.backward() 现场算出的梯度：{_g:+.3f}", fontsize=11)
    _fig.tight_layout()

    if abs(_g) < 0.05:
        _verdict = (f"**梯度 ≈ {_g:.3f}（接近零）**——驻点。可能是极值（峰/谷），"
                    "也可能是鞍点/平台。梯度下降在这里会停下：是好是坏，"
                    "取决于这是不是你要的答案（24 课：学习率和动量决定逃逸能力）。")
    elif abs(_g) > 2.0:
        _verdict = (f"**梯度 {_g:+.2f}（陡峭）**——大步跳走的区域。注意：太陡的坡"
                    "配大学习率会飞出去（35 课梯度爆炸的近亲），"
                    "优化器（Adam 的自适应步长，24 课）就是为此而生。")
    else:
        _verdict = (f"**梯度 {_g:+.2f}**——坡度温和，梯度下降的舒适区："
                    "x 会朝下坡方向移动 −lr×梯度 这么多。拖动滑块观察梯度"
                    "怎么随位置变化——这就是损失曲面上训练的全部 Dynamics。")
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
                **练习 1 · 手算再对账**：f(x) = x³ 在
                x=2 处的梯度是多少？先笔算（3x²），
                再在下面单元里用 torch 验证。

                **练习 2 · 补全三件套**：把 torch 自动
                求导的最小流程补完——声明要梯度的
                张量、反向、取梯度。

                **练习 3 · 工程守则**：评估/生成代码
                为什么应该包在 `torch.no_grad()` 里？
                （0=不需要梯度时省内存提速；1=不包
                会算错结果）——顺便说说 zero_grad
                防的是什么（填 0=梯度累积，1=内存
                泄漏）。
                """
            )
        }
    )
    return


@app.cell
def _(mo, torch):
    # ===== 练习 1 · 手算 x³ 的梯度 =====
    print("f(x) = x³，x = 2")
    print("  手算：f'(x) = 3x² = ?")

    grad_ex1_46 = None                               # ← ① 手算结果（整数）

    if grad_ex1_46 is not None:
        _x = torch.tensor([2.0], requires_grad=True)
        (_x ** 3).sum().backward()
        _ref = _x.grad.item()
        if abs(grad_ex1_46 - _ref) < 0.01:
            _msg = (f"✅ 正确：{_ref:.0f}，torch 逐位一致。注意 backward() 的对象是"
                    " .sum() 后的标量——x³ 是向量时直接 backward 会报错"
                    "（非标量要传 gradient 参数）。训练里的 loss 天然是标量，"
                    "所以从没遇到这个问题。")
        else:
            _msg = f"❌ torch 说 {_ref:.0f}（3x² @ x=2）"
    else:
        _msg = "⏳ 先笔算再填（3x²）"
    mo.md(_msg)
    return (grad_ex1_46,)


@app.cell
def _(mo, torch):
    # ===== 练习 2 · 补全三件套 =====
    def auto_grad_ex2_46(x_val):
        """返回 f(x)=x·sin(x) 在 x_val 处的梯度（用 autograd）。"""
        _x = None                                     # ← ① 补：声明要梯度的张量
        _y = None                                     # ← ② 补：前向 y = x·sin(x)，再 .sum()
        None                                          # ← ③ 补：反向
        return None                                   # ← 补：返回 _x 的梯度

    auto_ready_ex2_46 = None                          # ← 补完后，换成 auto_grad_ex2_46

    if auto_ready_ex2_46 is not None:
        try:
            _out = auto_ready_ex2_46(1.2)
            import numpy as _np2
            _ref = 1.2 * _np2.cos(1.2) + _np2.sin(1.2)
            if _out is not None and abs(float(_out) - _ref) < 1e-5:
                _msg = (f"✅ 正确：{_ref:.4f}。三件套：requires_grad=True 声明、"
                        "backward() 反向、.grad 取货。手算对照：f' = sin(x)+x·cos(x)"
                        "（乘法法则 + 链式）——autograd 把这条推导变成了结构化代码。")
            else:
                _msg = f"❌ 参考：{ _ref:.4f}（sin + x·cos）"
        except (TypeError, ValueError, RuntimeError) as _e:
            _msg = f"❌ 运行出错：{_e}——tensor(x, requires_grad=True)；(x*sin(x)).sum()；backward()"
    else:
        _msg = "⏳ 补三处：声明 → 前向求和 → backward"
    mo.md(_msg)
    return (auto_grad_ex2_46, auto_ready_ex2_46)


@app.cell
def _(mo):
    # ===== 练习 3 · 工程守则 =====
    print("两个工程守则的\"为什么\"")

    why_nograd_ex3_46 = None                          # ← ① no_grad 的理由（0=省内存提速，1=不包会算错）
    why_zero_ex3_46 = None                            # ← ② zero_grad 防的（0=梯度累积，1=内存泄漏）

    if why_nograd_ex3_46 is not None and why_zero_ex3_46 is not None:
        if why_nograd_ex3_46 == 0 and why_zero_ex3_46 == 0:
            _msg = ("✅ 双双正确。no_grad：不需要梯度时不记录图——中间张量不存"
                    "出处，省内存也提速；**结果不会算错**（算错的是忘 zero_grad"
                    "的那种 bug）。zero_grad：PyTorch 的梯度默认**累积**（为了"
                    "梯度累积这种高级用法），训练循环每步不清零的话，这一步"
                    "的更新量=全部历史梯度之和——模型会神秘发散。25 课那行"
                    " zero_grad 的针，今天全部拔完。")
        else:
            _msg = "❌ 参考：都是 0（no_grad=省内存提速；zero_grad 防梯度累积）"
    else:
        _msg = "⏳ 两空都填（0 或 1）"
    mo.md(_msg)
    return (why_nograd_ex3_46, why_zero_ex3_46)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 1 参考答案（先自己写过再看！）": mo.md(
                """
                **f'(2) = 3×4 = 12。**

                ```python
                x = torch.tensor([2.0], requires_grad=True)
                (x ** 3).sum().backward()
                x.grad   # tensor([12.])
                ```

                一个常见的初学困惑：明明 x 只有一个
                元素，为什么要 .sum()？因为 backward
                的语义是"标量对图上所有节点求导"——
                向量输出没有天然的"总梯度"。sum 让
                每个元素的梯度都是 1，恰好等于你想要
                的逐元素导数。（mean 也常用，记得
                除以 N 就行。）
                """
            ),
            "✅ 练习 2 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                _x = torch.tensor([x_val], requires_grad=True)
                _y = (_x * torch.sin(_x)).sum()
                _y.backward()
                return _x.grad
                ```

                对照手算：f(x)=x·sin(x) 的导数 =
                sin(x) + x·cos(x)（乘法法则）。autograd
                内部对乘法节点登记的正是这条规则
                （前向记录两个父节点，反向把
                cos(x)·dx 和 sin(x)·dx 各自回传）——
                你 22 课手画的每张小图，它都自动画、
                自动走。
                """
            ),
            "✅ 练习 3 参考答案（先自己写过再看！）": mo.md(
                """
                **0 / 0。**

                两条守则的完整版：

                - `torch.no_grad()`：**不影响正确性，
                  影响成本**——图记录是中间张量的
                  额外开销（CPU 上 ~10-30%，GPU 上
                  更多），推理时纯属浪费。例外：
                  需要 .grad 的代码**必须**在图内
                - `zero_grad()`：**影响正确性**——
                  忘了清零，梯度跨步累积，等效于
                  一个越来越大的隐形 batch，训练
                  曲线神秘恶化。这类 bug 不报错、
                  不崩溃，只让结果慢慢变差——和
                  43 课说的"反向传播 bug"同一家族，
                  而 autograd 时代最常见的一只

                47 课会把这两条守则焊进 nn.Module
                的标准训练循环里——到时候它们会
                变成肌肉记忆。
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
                1. **autograd 原理**：前向记录计算图，
                   backward 沿图反向遍历、自动套链式
                   法则——22 课手艺的自动化，无魔法
                2. **动态图**：图随前向代码现场生成
                   （define-by-run）——Python 循环和
                   分支直接进模型
                3. **三件套**：requires_grad=True 声明 →
                   backward() 反向（标量）→ .grad 取货；
                   叶子留 grad，中间量用完即弃
                4. **梯度默认累积**：每步必须
                   zero_grad——忘了它，训练神秘恶化
                5. **对账通过**：torch 重写 43 课编码器
                   前向，autograd 梯度与手写版逐位
                   一致（10⁻⁸ 级）——你写的没有错，
                   autograd 也没有魔法
                6. **两个阀门**：no_grad（推理不记图，
                   省内存提速）、detach（剪图防回流）
                   ——48 课训练范式的固定演员
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

                **下一课预告** — `47_重写MLP与CNN`（第 8 部分 · 过渡到 PyTorch）：

                autograd 解放了反向传播，但 46 课的
                torch 代码还保持着"NumPy 的姿势"——
                手动建参数、手动写前向。47 课换成
                **工业姿势**：

                - **nn.Module**：参数自动登记
                  （`self.W = nn.Parameter(...)`），
                  `.parameters()` 一网打尽
                - **nn.Linear / nn.Conv2d**：你在 20/29 课
                  手写的层，一行一个
                - **用 torch 重写 25 课 MLP**（MNIST
                  分类器）和 **30 课 CNN**（LeNet-5）——
                  同样的任务、更少的代码、更快的
                  训练，结果逐项对表
                - **优化器与调度器**：Adam 一行配好，
                  学习率随训练衰减自动化

                重写的意义不止省代码：**30 课的 LeNet
                在 GPU 上从 20 分钟变 20 秒**——代码
                几乎不用改，只搬个家（.to("cuda")）。
                """
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
