import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="35 · 梯度消失与爆炸")


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
    mo.md("""
    # 35 · 梯度消失与爆炸：RNN 的时间病

    **从零到 Transformer · 手写深度学习** ｜ 第 6 部分 · 循环神经网络（4/6）
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📖 本课目标（先读我）": mo.md(
                """
                **34 课结尾那张梯度体检图，留了一个刺眼的数字：
                α=0.5 时，梯度从损失端往回走 6 步就衰减到 1%。**

                这一课把那条连乘公式推到极端，回答三个问题：

                - 病根是什么？（**一条连乘公式**，两种命运：
                  乘子普遍 <1 → 指数消失；普遍 >1 → 指数爆炸）
                - 爆炸怎么治？（**梯度裁剪**——三行代码的急救包，
                  演示 2 亲手复现 loss 尖峰再亲手救回来）
                - 消失怎么治？（**治不了，只能换结构**——演示 3
                  的长依赖任务里，普通 RNN 卡死在瞎猜线，
                  36 课的 LSTM 为它而生）

                学完你将能：

                - 用一句话说清消失与爆炸的**共同根源**：梯度走
                  太远的路，每步还要乘点什么
                - 数值上感受指数的残酷：0.9²⁰ ≈ 0.12、
                  1.05⁵⁰ ≈ 11.5——离 1 一点点，乘几十步就是
                  天壤之别
                - 亲手实现 `clip_grad_norm`（工业界训练 RNN/
                  Transformer 的标配三行）
                - 解释为什么 clip 救爆炸、救不了消失（练习 4
                  的概念题，也是理解 LSTM 动机的钥匙）

                前置：34 课的 BPTT 与梯度体检图、24 课的
                优化器、19 课的 tanh 饱和。
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "🦠 概念 1/3 · 病根：一条连乘公式，两种命运": mo.md(
                """
                34 课说过，U 路的梯度要连乘：

                ```
                ∂h_T/∂h_t = ∏ (U · diag(1−hₖ²))     ← k 从 t+1 到 T，共 T−t 次
                ```

                先不管矩阵细节，把它压成一个数：**每过一步，
                梯度大致乘一个"等效乘子" w**。走 T 步就是 wᵀ。
                指数行为的残酷在于**离 1 一点点，乘几十步就是
                天壤之别**：

                | w | w⁵⁰ | w¹⁰⁰ | 命运 |
                |---|---|---|---|
                | 0.5 | 8.9×10⁻¹⁶ | 7.9×10⁻³¹ | 消失：梯度归零 |
                | 0.99 | 0.61 | 0.37 | 慢性消失：百步腰斩 |
                | 1.01 | 1.64 | 2.70 | 慢性爆炸 |
                | 1.5 | 6.4×10⁸ | 4.1×10¹⁷ | 爆炸：一步踢飞 |

                **消失与爆炸不是两种病，是同一条公式的两种
                命运**——分界线就是 1。麻烦在于：w 由 U 的谱
                半径和 tanh′ 共同决定，训练中它自己会漂移，
                你很难把整条时间轴上的每一步都摁在 1 附近。
                """
            ),
            "🕳️ 概念 2/3 · 消失的两大帮凶": mo.md(
                """
                梯度消失（比爆炸常见得多，也更致命）有两个帮凶：

                **帮凶一 · ρ(U) < 1。** 标准初始化下，随机方阵
                的谱半径通常略小于 1（34 课实测 ρ=0.90）——
                网络天生偏向"记忆衰减"。这本来是好事（前向
                稳定），但反向梯度走的是同一条路：**前向记
                不住的，反向也传不动**（33 课脉冲实验的镜像）。

                **帮凶二 · tanh 饱和。** 34 课练习 1 算过：
                z=2.5 时 tanh′ = 0.03。h 一旦爬进饱和区
                （|h| → 1），它那个时间步就成了**窄闸门**——
                连续几步饱和，梯度被连乘 0.0x，很快断流。
                更阴险的是：训练越久、网络越自信，h 越容易
                饱和（输出要"干脆"）——**学得好反而堵死梯度**。

                两个帮凶合力的下场：**远处的梯度 → 0，学习
                信号根本到不了**。注意这不是"学得慢"（慢可以
                等），是"聋"（等多久都没用）——演示 3 会让
                你看到 gap=20 的任务上 loss 纹丝不动。
                """
            ),
            "💊 概念 3/3 · 两种病，两种疗法": mo.md(
                """
                **爆炸 → 急救包：梯度裁剪（gradient clipping）。**

                逻辑简单粗暴：算完全局梯度范数 ‖g‖，超过阈值
                max_norm 就整体缩回去：

                ```python
                if ‖g‖ > max_norm:
                    g *= max_norm / ‖g‖
                ```

                三行代码，治标不治本：它**防踢飞**（loss 尖峰、
                NaN），但不改变"远处学不到"的本质。工业界训练
                RNN/Transformer 至今标配（练习 2 亲手写）。

                **消失 → 没有急救包，只能动手术：换结构。**

                消失的根子在"梯度必须穿过 U·tanh′ 的连乘"。
                想根治，就得给梯度修一条**不走连乘的路**。
                1997 年 Hochreiter & Schmidhuber 的答案：
                **LSTM**——加一条细胞状态 c 的"高速路"，
                梯度沿它回传时乘的是**可学习的门控值**
                （网络自己决定"这一步让梯度几乎无衰减地过"），
                而不是固定的 U·tanh′。36 课拆解它。

                一句话总结本课的病历：**爆炸是急症，clip 住
                就行；消失是慢性病，结构性不治——LSTM 是
                那个器官移植。**
                """
            ),
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 🎬 演示
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 演示 1 · 指数的残酷：wᵀ 的一张图

    四个乘子（0.5 / 0.99 / 1.01 / 1.5）各自连乘 100 步，
    半对数坐标一张图看清两个宇宙：

    - **w = 0.5**：17 步就跌破 10⁻⁵——第 18 步之前的 U
      使用，梯度等于没了
    - **w = 0.99**：看着人畜无害，100 步只剩 0.37——
      **慢性消失**，比急性更难察觉
    - **w = 1.01 / 1.5**：100 步 2.7 倍 / 10¹⁷ 倍——
      后者一步更新就能把参数踢出太阳系

    两条参考线：**10⁻⁶ 学不动线**（梯度小于它，对参数的
    影响淹没在数值噪声里）和 **10³ 危险线**（再大就容易
    loss 窜天）。真实 RNN 的"等效乘子"每步都在变，但
    长期的几何平均落在哪一侧，命运就定了。
    """)
    return


@app.cell
def _(np, plt):
    # 演示 1 · 标量连乘图
    ws_demo35 = [0.5, 0.99, 1.01, 1.5]
    colors35 = ["tab:red", "tab:orange", "tab:green", "tab:purple"]
    _T = 100
    _ts = np.arange(0, _T + 1)

    print("指数的残酷（手算两笔，练习 1 还要算两笔）：")
    for _w in ws_demo35:
        print(f"  w = {_w}: w^50 = {_w ** 50:.3e}, w^100 = {_w ** 100:.3e}")

    _fig, _ax = plt.subplots(figsize=(9.0, 4.6))
    for _w, _c in zip(ws_demo35, colors35):
        _ax.semilogy(_ts, np.maximum(_w ** _ts, 1e-40), lw=2, color=_c, label=f"w = {_w}")
    _ax.axhline(1e-6, color="gray", ls="--", lw=1.3)
    _ax.text(2, 2.4e-6, "学不动线 $10^{-6}$（梯度淹没在噪声里）", fontsize=8.5, color="gray")
    _ax.axhline(1e3, color="gray", ls="--", lw=1.3)
    _ax.text(2, 2.4e3, "危险线 $10^{3}$（loss 窜天前兆）", fontsize=8.5, color="gray")
    _ax.axhline(1.0, color="black", lw=0.8, alpha=0.5)
    _n_half = int(np.log(1e-5) / np.log(0.5))
    _ax.annotate(f"w=0.5：{_n_half} 步跌破 $10^{{-5}}$", xy=(_n_half, 1e-5),
                 xytext=(_n_half + 8, 3e-3), fontsize=9, color="tab:red",
                 arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.2))
    _ax.set_xlabel("连乘步数 T")
    _ax.set_ylabel("$w^t$（对数坐标）")
    _ax.set_title("同一条连乘公式的两种命运：分界线是 1", fontsize=12)
    _ax.legend(fontsize=9.5, loc="upper left")
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 演示 2 · loss 尖峰现场 + 三行急救包

    造一个会爆炸的现场：gap=20 的回声任务（学不会 → 梯度
    一直很大）、**U 放大到 α=2.5**（谱半径 ≈ 2.5 > 1）、
    SGD 学习率 0.3、**不裁剪**。

    红线就是事故记录：150 轮里 **9 次 loss 尖峰**，梯度
    范数一度冲到 **2,692**，loss 从 1.19 被打到 1.66——
    每次尖峰都是一次大梯度把刚学的东西踢飞。灰色对照：
    **同一个随机种子、只加 clip(max_norm=1.0)**——梯度
    范数照样冲高（下面小图，原始范数还是 6,480），但被
    截在阈值上，loss 全程平稳，**0 次尖峰**。

    注意一个诚实的细节：clip 之后 loss 也只是平稳地
    停在瞎猜线附近（≈1.1）——它治好了"炸"，治不了
    "学不会"（gap=20 的病根是消失，不是爆炸）。
    这正是练习 4 概念题的答案。
    """)
    return


@app.cell
def _(np, plt):
    # 演示 2 · 尖峰复现 + clip 急救（SGD，α=2.5）
    def forward_rnn35(xs, Wx, Wh, bh, Wy, by):
        _h = np.zeros(Wh.shape[0])
        _hs, _logits = [], []
        for _x in xs:
            _h = np.tanh(Wx @ _x + Wh @ _h + bh)
            _hs.append(_h.copy()); _logits.append(Wy @ _h + by)
        _L = np.array(_logits)
        _P = np.exp(_L - _L.max(axis=1, keepdims=True))
        return np.array(_hs), _P / _P.sum(axis=1, keepdims=True)

    def bptt35(xs, ys, Wx, Wh, bh, Wy, by, loss_mask):
        _T = len(xs)
        _hs, _P = forward_rnn35(xs, Wx, Wh, bh, Wy, by)
        _dWx, _dWh, _dbh = np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(bh)
        _dWy, _dby = np.zeros_like(Wy), np.zeros_like(by)
        _dlog = _P.copy(); _dlog[np.arange(_T), ys] -= 1
        _dlog = _dlog * loss_mask[:, None] / max(loss_mask.sum(), 1)
        _delta = np.zeros(Wh.shape[0])
        for _t in reversed(range(_T)):
            _dh = Wy.T @ _dlog[_t] + Wh.T @ _delta
            _delta = _dh * (1 - _hs[_t] ** 2)
            _dWx += np.outer(_delta, xs[_t])
            _h_prev = _hs[_t - 1] if _t > 0 else np.zeros(Wh.shape[0])
            _dWh += np.outer(_delta, _h_prev)
            _dbh += _delta; _dWy += np.outer(_dlog[_t], _hs[_t]); _dby += _dlog[_t]
        return dict(Wx=_dWx, Wh=_dWh, bh=_dbh, Wy=_dWy, by=_dby)

    def make_echo_seqs35(delay, n_seq, T, seed0):
        """n 条独立随机开头的回声序列（y_t = x_{t-delay}）。"""
        _out = []
        for _i in range(n_seq):
            _r = np.random.default_rng(seed0 + _i)
            _xs = np.eye(3)[_r.integers(0, 3, T)]
            _ys = np.concatenate([_r.integers(0, 3, delay),
                                  np.array(_xs[:-delay]).argmax(1)])
            _mask = np.ones(T, bool); _mask[:delay] = False
            _out.append((_xs, _ys, _mask))
        return _out

    def train_echo35(delay, n_seq, T, epochs, lr, clip, alpha, seed, opt="sgd"):
        """SGD/Adam 训练回声任务；clip=None 不裁剪。返回 (losses, 梯度范数序列)。"""
        _tr = make_echo_seqs35(delay, n_seq, T, seed0=0)
        _rc = np.random.default_rng(seed)
        _d_h = 32
        _plist = [_rc.normal(0, 1 / np.sqrt(3), (_d_h, 3)),
                  _rc.normal(0, alpha / np.sqrt(_d_h), (_d_h, _d_h)),
                  np.zeros(_d_h),
                  _rc.normal(0, 1 / np.sqrt(_d_h), (3, _d_h)),
                  np.zeros(3)]
        _ms = [np.zeros_like(_p) for _p in _plist]; _vs = list(_ms)
        _losses, _gn = [], []
        for _step in range(1, epochs + 1):
            _acc = [np.zeros_like(_p) for _p in _plist]
            for _xs, _ys, _mask in _tr:
                _g = bptt35(_xs, _ys, *_plist, loss_mask=_mask)
                for _i, _k in enumerate(("Wx", "Wh", "bh", "Wy", "by")):
                    _acc[_i] += _g[_k] / len(_tr)
            _gn.append(float(np.sqrt(sum((_a ** 2).sum() for _a in _acc))))
            if clip is not None and _gn[-1] > clip:
                _sc = clip / (_gn[-1] + 1e-12)
                _acc = [_a * _sc for _a in _acc]
            for _i, _p in enumerate(_plist):
                if opt == "adam":
                    _ms[_i] = 0.9 * _ms[_i] + 0.1 * _acc[_i]
                    _vs[_i] = 0.999 * _vs[_i] + 0.001 * _acc[_i] ** 2
                    _p -= lr * (_ms[_i] / (1 - 0.9 ** _step)) / (
                        np.sqrt(_vs[_i] / (1 - 0.999 ** _step)) + 1e-8)
                else:
                    _p -= lr * _acc[_i]
            _L = 0.0
            for _xs, _ys, _mask in _tr:
                _, _P = forward_rnn35(_xs, *_plist)
                _L += -np.log(_P[np.arange(T)[_mask], _ys[_mask]] + 1e-12).mean()
            _losses.append(_L / len(_tr))
        return np.array(_losses), np.array(_gn), _plist

    # ---- 同一个种子的两场训练：不 clip vs clip ----
    loss_raw35, gn_raw35, _p_raw35 = train_echo35(20, 8, 60, 150, lr=0.3, clip=None,
                                                   alpha=2.5, seed=0)
    loss_clip35, gn_clip35, _p_clip35 = train_echo35(20, 8, 60, 150, lr=0.3, clip=1.0,
                                                      alpha=2.5, seed=0)
    _spikes = int((np.diff(loss_raw35) > 0.5).sum())
    print(f"不裁剪：150 轮里 {_spikes} 次 loss 尖峰，梯度范数最大 {gn_raw35.max():.0f}，"
          f"loss {loss_raw35[0]:.2f} → {loss_raw35[-1]:.2f}（被打飞）")
    print(f"clip=1.0：0 次尖峰，loss {loss_clip35[0]:.2f} → {loss_clip35[-1]:.2f}（平稳——"
          f"但停在瞎猜线 ln3≈1.10 附近：治炸不治聋）")
    print(f"clip 后实际用到的原始梯度范数仍高达 {gn_clip35.max():.0f}（被截在 1.0 上）")

    _fig, _axes = plt.subplots(2, 1, figsize=(9.6, 6.4), sharex=True,
                               gridspec_kw={"height_ratios": [1.35, 1.0]})
    _axes[0].plot(loss_raw35, color="tab:red", lw=1.6, label="不裁剪（9 次尖峰）")
    _axes[0].plot(loss_clip35, color="tab:blue", lw=1.8, label="clip(max_norm=1.0)")
    _axes[0].axhline(np.log(3), color="gray", ls="--", lw=1.1)
    _axes[0].text(2, np.log(3) + 0.03, "瞎猜线 ln3 ≈ 1.10", fontsize=8.5, color="gray")
    _axes[0].set_ylabel("训练损失")
    _axes[0].set_title("loss 尖峰现场与三行急救包（gap=20，α=2.5，SGD lr=0.3）", fontsize=11.5)
    _axes[0].legend(fontsize=9.5)
    _axes[0].grid(True, alpha=0.3)
    _axes[1].semilogy(gn_raw35, color="tab:red", lw=1.2, alpha=0.75, label="原始梯度范数（不裁剪）")
    _axes[1].semilogy(gn_clip35, color="tab:blue", lw=1.2, alpha=0.75,
                      label="原始梯度范数（clip 组，截前）")
    _axes[1].axhline(1.0, color="tab:blue", ls="--", lw=1.3)
    _axes[1].text(2, 1.3, "clip 阈值 1.0", fontsize=8.5, color="tab:blue")
    _axes[1].set_xlabel("训练轮数")
    _axes[1].set_ylabel("全局梯度范数（对数坐标）")
    _axes[1].legend(fontsize=9)
    _axes[1].grid(True, alpha=0.3)
    _fig.tight_layout()
    _fig
    return forward_rnn35, make_echo_seqs35, train_echo35


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 演示 3 · 长依赖：gap=3 学得会，gap=20 聋掉了

    同一个 RNN（d_h=32）、同一个配方（Adam、350 轮、12 条
    训练序列），只改**回声间隔**：

    - **gap=3**：loss 一路俯冲 → 0.0001，**新序列考试
      99%**——34 课的老朋友，梯度照得到 3 步之外
    - **gap=20**：loss 从 1.09 → 1.08，**纹丝不动**，
      考试 31%——不是学得慢，是 20 步之外的梯度已经
      衰减到（前 50 轮梯度范数平均只有 0.75）推不动参数

    这不是超参数没调好——换 lr、换初始化、训十倍时间，
    都救不了"信号到不了"的结构病。唯一出路是给梯度修
    一条不连乘的路：36 课的 LSTM 正是干这个的。
    """)
    return


@app.cell
def _(forward_rnn35, make_echo_seqs35, np, plt, train_echo35):
    # 演示 3 · gap=3 vs gap=20（Adam，多序列防背题）
    loss_gap3_35, _, plist3_35 = train_echo35(3, 12, 50, 350, lr=0.02, clip=None,
                                               alpha=1.0, seed=1, opt="adam")
    loss_gap20_35, _, plist20_35 = train_echo35(20, 12, 50, 350, lr=0.02, clip=None,
                                                alpha=1.0, seed=1, opt="adam")

    def echo_test_acc35(delay, plist, seed0=500):
        """在新随机序列上考回声准确率。"""
        _te = make_echo_seqs35(delay, 4, 50, seed0=seed0)
        _ok, _n = 0, 0
        for _xs, _ys, _mask in _te:
            _, _P = forward_rnn35(_xs, *plist)
            _ok += int((_P.argmax(1)[_mask] == _ys[_mask]).sum())
            _n += int(_mask.sum())
        return _ok / _n

    acc3_35 = echo_test_acc35(3, plist3_35)
    acc20_35 = echo_test_acc35(20, plist20_35)
    print(f"gap=3 ：loss {loss_gap3_35[0]:.3f} → {loss_gap3_35[-1]:.4f}（350 轮），"
          f"新序列考试 {acc3_35:.0%}")
    print(f"gap=20：loss {loss_gap20_35[0]:.3f} → {loss_gap20_35[-1]:.4f}（纹丝不动），"
          f"新序列考试 {acc20_35:.0%}（瞎猜线 33%）")
    print("题在 20 步外，梯度到不了——不是学得慢，是聋")

    _fig, _ax = plt.subplots(figsize=(9.0, 4.4))
    _ax.plot(loss_gap3_35, color="tab:blue", lw=1.8, label="gap=3（梯度照得到）")
    _ax.plot(loss_gap20_35, color="tab:red", lw=1.8, label="gap=20（梯度到不了）")
    _ax.axhline(np.log(3), color="gray", ls="--", lw=1.1)
    _ax.text(3, np.log(3) + 0.02, "瞎猜线 ln3 ≈ 1.10", fontsize=8.5, color="gray")
    _ax.set_xlabel("训练轮数")
    _ax.set_ylabel("训练损失（12 条序列全批）")
    _ax.set_title("长依赖审判：同一个 RNN，只改回声间隔", fontsize=12)
    _ax.legend(fontsize=10)
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---

    ## 🎛️ 交互实验
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 交互实验 · 梯度寿命计算器

    两条滑块：**等效乘子 w**（每步梯度的平均缩放）和
    **序列长度 T**。曲线画出 wᵗ（对数坐标），两条参考线
    是**学不动线 10⁻⁶** 和 **危险线 10³**。

    判词直接回答一个工程问题：**这条时间轴上，离损失多远
    的位置还"学得动"？**

    试试：w=0.95（看似很接近 1）+ T=100——寿命够不够
    用？w=0.8 呢？再用 w=1.05 看看慢爆炸需要多久发作。
    """)
    return


@app.cell
def _(mo):
    # 交互控制：w × T
    w_slider35 = mo.ui.slider(0.70, 1.30, value=0.95, step=0.01, debounce=True,
                              label="🎚️ 等效乘子 w（每步梯度的平均缩放）")
    T_slider35 = mo.ui.slider(10, 100, value=60, step=5, debounce=True,
                              label="📏 序列长度 T")
    mo.hstack([w_slider35, T_slider35])
    return T_slider35, w_slider35


@app.cell
def _(T_slider35, mo, np, plt, w_slider35):
    # ===== 交互实验 · 梯度寿命计算器（响应单元）=====
    _w = w_slider35.value
    _T = T_slider35.value
    _ts = np.arange(0, _T + 1)
    _curve = _w ** _ts

    _fig, _ax = plt.subplots(figsize=(8.6, 4.2))
    _ax.semilogy(_ts, np.maximum(_curve, 1e-40), lw=2.5, color="tab:purple")
    _ax.axhline(1e-6, color="gray", ls="--", lw=1.3)
    _ax.axhline(1e3, color="gray", ls="--", lw=1.3)
    _ax.axhline(1.0, color="black", lw=0.8, alpha=0.4)
    _ax.text(1, 2.6e-6, "学不动线 $10^{-6}$", fontsize=8.5, color="gray")
    _ax.text(1, 2.6e3, "危险线 $10^{3}$", fontsize=8.5, color="gray")
    _ax.set_xlabel("离损失的距离（步数）")
    _ax.set_ylabel("梯度幅值（对数坐标）")
    _ax.set_title(f"w = {_w:.2f}，T = {_T}", fontsize=11)
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()

    if _w < 1.0:
        _life = int(np.ceil(np.log(1e-6) / np.log(_w)))       # 掉到 1e-6 的步数
        _verdict = (f"**消失带**（w = {_w:.2f}）：走 **{_life} 步**梯度就跌破 10⁻⁶——"
                    f"序列长度 {_T} 里，离损失 {_life} 步以外的位置全部「聋」"
                    f"（占整条轴的 {max(0, (_T - _life) / _T):.0%}）。"
                    "w=0.95 看着无害，指数不饶人——这就是为什么普通 RNN "
                    "连「中等长度」的依赖都学不动")
    elif _w > 1.0:
        _life = int(np.ceil(np.log(1e3) / np.log(_w)))         # 涨到 1e3 的步数
        _verdict = (f"**爆炸带**（w = {_w:.2f}）：走 **{_life} 步**梯度就冲破 10³——"
                    f"序列长度 {_T} 的反向传播里，最远端梯度是损失端的 "
                    f"{_w ** _T:.0e} 倍。一次大梯度一步把参数踢飞（演示 2 的尖峰），"
                    "急救包是三行 clip（练习 2）")
    else:
        _verdict = ("**临界线 w = 1.00**：梯度不衰减也不增长——但这是刀尖上的平衡："
                    "真实训练中 w 每步都在变，两侧是两个宇宙")
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
                **练习 1 · 手算指数**：0.9²⁰ 和 1.05⁵⁰（保留
                两位小数）——算完你对"离 1 一点点"会有肌肉记忆。

                **练习 2 · 实现梯度裁剪**：补全 `clip_grad_norm`
                的两行（工业界标配，PyTorch 同名函数的核心逻辑）。

                **练习 3 · 梯度寿命**：w=0.7 时走几步跌破 1%？

                **练习 4 · 概念题**：clip 为什么治不了梯度消失？
                """
            )
        }
    )
    return


@app.cell
def _(mo):
    # ===== 练习 1 · 手算指数 =====
    print("指数的残酷，笔算两笔（保留两位小数）：")
    print("  0.9²⁰ = ?     ←「看 20 步远」的记忆，梯度还剩多少")
    print("  1.05⁵⁰ = ?    ←「慢爆炸」50 步后放大几倍")

    ans1_ex1_35 = None                             # ← ① 0.9^20（两位小数）
    ans2_ex1_35 = None                             # ← ② 1.05^50（两位小数）

    if ans1_ex1_35 is not None and ans2_ex1_35 is not None:
        _t1, _t2 = 0.9 ** 20, 1.05 ** 50
        if abs(ans1_ex1_35 - _t1) < 0.02 and abs(ans2_ex1_35 - _t2) < 0.05:
            _msg = (f"✅ 正确：{_t1:.2f} 和 {_t2:.2f}。0.9 离 1 只差 0.1，"
                    f"20 步就只剩 12%；1.05 离 1 只差 0.05，50 步放大 "
                    f"{_t2:.1f} 倍。乘子只要长期偏离 1，指数就会把偏差放大成鸿沟")
        else:
            _msg = (f"❌ 参考：{_t1:.2f} / {_t2:.2f}。提示：0.9²⁰ = (0.9²)¹⁰ = 0.81¹⁰；"
                    "或者 ln 换算：20·ln0.9")
    else:
        _msg = "⏳ 两笔都算出来（提示：连乘用对数，或分步平方）"
    mo.md(_msg)
    return


@app.cell
def _(mo, np):
    # ===== 练习 2 · 实现梯度裁剪 =====
    def clip_grad_norm_ex2_35(grads, max_norm):
        """按全局范数裁剪：grads 是梯度列表，返回裁剪后的列表。
        规则：全局范数 ‖g‖ 超过 max_norm 就整体缩放回 max_norm。"""
        _total = None                               # ← ① 补：全局范数（所有梯度的平方和开根号）
        if _total is not None and _total > max_norm:
            _scale = max_norm / (_total + 1e-12)
            return [_g * _scale for _g in grads]    # 已写好：整体缩放
        return grads

    clip_ready_ex2_35 = None                        # ← 补完后，把这行换成 clip_grad_norm_ex2_35

    if clip_ready_ex2_35 is not None:
        try:
            _gs = [np.array([3.0, 4.0]), np.array([0.0, 0.0])]      # 全局范数 = 5
            _out = clip_ready_ex2_35(_gs, 1.0)
            _n_after = np.sqrt(sum((_o ** 2).sum() for _o in _out))
            _unchanged = clip_ready_ex2_35([np.array([0.3, 0.4])], 1.0)
            _n_small = np.sqrt(sum((_o ** 2).sum() for _o in _unchanged))
            if abs(_n_after - 1.0) < 1e-6 and abs(_n_small - 0.5) < 1e-9:
                _msg = ("✅ 两项测试都过：范数 5 → 1.0（缩放 0.2），范数 0.5 → 0.5（不动）。"
                        "注意方向性：只截上、不补下——这正是它治爆炸不治消失的原因（练习 4）")
            else:
                _msg = (f"❌ 大范数裁后应恰为 1.0（实测 {_n_after:.3f}），"
                        f"小范数应原样返回（实测 {_n_small:.3f}）。检查 total 的计算")
        except (TypeError, ValueError) as _e:
            _msg = f"❌ 运行出错：{_e}——total = np.sqrt(sum((g**2).sum() for g in grads))"
    else:
        _msg = "⏳ 补一行：_total = np.sqrt(sum((_g ** 2).sum() for _g in grads))"
    mo.md(_msg)
    return


@app.cell
def _(mo, np):
    # ===== 练习 3 · 梯度寿命 =====
    print("w = 0.7：走多少步，梯度跌破原值的 1%？")
    print("（即求最小整数 n，使 0.7ⁿ < 0.01）")

    n_ex3_35 = None                                 # ← ① 步数（整数）

    if n_ex3_35 is not None:
        _t = int(np.ceil(np.log(0.01) / np.log(0.7)))
        if n_ex3_35 == _t:
            _msg = (f"✅ 正确：{_t} 步（0.7^13 ≈ 0.0097）。换算成对数："
                    "n = ln(0.01)/ln(0.7) = 12.99 → 13。w=0.85、0.95 的寿命分别是 "
                    f"{int(np.ceil(np.log(0.01)/np.log(0.85)))} 步和 "
                    f"{int(np.ceil(np.log(0.01)/np.log(0.95)))} 步——乘子越接近 1 寿命越长，"
                    "但对数增长追不上序列长度的需求")
        else:
            _msg = f"❌ 参考 {_t}。用 n = ceil(ln(0.01) / ln(0.7))，或逐步乘 0.7 数一数"
    else:
        _msg = "⏳ 填一个整数（可以先算 0.7²、0.7⁴、0.7⁸ 累进）"
    mo.md(_msg)
    return


@app.cell
def _(mo):
    # ===== 练习 4 · 概念题 =====
    print("梯度裁剪为什么治不了梯度消失？")
    print("  A. clip 只限制梯度的最大幅值：太小的梯度它碰都不碰，乘子 <1 的连乘照样衰减到 0")
    print("  B. 因为 clip 改变了梯度的方向，方向错了学不了远处的依赖")
    print("  C. 因为梯度消失发生在前向传播，clip 只作用于反向传播")

    ans_ex4_35 = None                               # ← ① 填 'A' / 'B' / 'C'

    if ans_ex4_35 is not None:
        if ans_ex4_35 == "A":
            _msg = ("✅ 正确。clip 是「截上不补下」的单向阀门：‖g‖>阈值就缩回，"
                    "‖g‖<阈值原样放行——消失的梯度（本来就近乎 0）永远等不到被放大。"
                    "所以演示 2 里 clip 后 loss 平稳，但也只是平稳地「聋着」。"
                    "要让远处的梯度活着到达，只能改结构——36 课 LSTM 的动机全在这一句")
        else:
            _msg = "❌ 提示：clip 的规则是「超过阈值才动手」——它对太小的梯度做了什么？（什么都没做）"
    else:
        _msg = "⏳ 填一个选项（'A' / 'B' / 'C'）"
    mo.md(_msg)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 1 参考答案（先自己写过再看！）": mo.md(
                """
                **0.9²⁰ ≈ 0.12，1.05⁵⁰ ≈ 11.47。**

                ```
                0.9²⁰ = e^(20·ln0.9) = e^(−2.107) ≈ 0.122
                1.05⁵⁰ = e^(50·ln1.05) = e^(2.440) ≈ 11.47
                ```

                笔算技巧：ln0.9 ≈ −0.105、ln1.05 ≈ 0.0488
                （都可用 ln(1+x) ≈ x 估）。两个数的"偏离 1
                的量"都是 0.05~0.1 量级，乘几十步就一个归零
                一个起飞——**指数会把一切小偏差放大成鸿沟**。
                """
            ),
            "✅ 练习 2 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                _total = np.sqrt(sum((_g ** 2).sum() for _g in grads))
                ```

                三个工程细节：

                - **全局范数**：所有参数张量的梯度**合在一起**
                  算范数（不是逐张量裁）——梯度是一个整体方向，
                  裁剪要保方向、只缩长度
                - **缩放而不是截断**：乘 `max_norm/‖g‖` 让范数
                  恰好等于阈值，梯度方向分毫不动（逐元素截断
                  会扭曲方向，已基本被淘汰）
                - **eps 防零**：分母加 1e-12，防 ‖g‖=0 时除零
                  （职业习惯，24 课 Adam 里见过同款）

                PyTorch 的 `torch.nn.utils.clip_grad_norm_`
                用的就是这套逻辑。
                """
            ),
            "✅ 练习 3 参考答案（先自己写过再看！）": mo.md(
                """
                **13 步**（0.7¹³ ≈ 0.0097 首次 < 1%）。

                通式：n = ceil(ln(目标比例) / ln(w))。几个常用
                寿命值得背下来（目标 1%）：

                | w | 寿命（步） |
                |---|---|
                | 0.7 | 13 |
                | 0.85 | 28 |
                | 0.95 | 90 |
                | 0.99 | 458 |

                w 从 0.7 提到 0.99，寿命从 13 步涨到 458 步——
                看似希望很大？但真实 RNN 的等效乘子是 **每步
                在变的随机量**，几何平均能稳在 0.99 已经是
                奢望（34 课实测随机 U 的 ρ≈0.9，寿命不到
                25 步）。这就是为什么工程界不等"调好 w"，
                直接换 LSTM。
                """
            ),
            "✅ 练习 4 参考答案（先自己写过再看！）": mo.md(
                """
                **选 A。**

                clip 的规则是"超过才动手"——它是一个**单向
                阀门**：上限被死死摁住，下限无人看管。爆炸的
                梯度（太大）被截住，消失的梯度（太小）原样
                通过，然后继续小下去、小到 0。

                B 错在方向：clip 恰恰**只缩长度、不改方向**。
                C 错在时序：消失和爆炸都发生在**反向**传播的
                连乘里，前向只是"预告"（33 课脉冲实验）。

                一句话病历：**clip 是止痛药，LSTM 是手术。**
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
                1. **病根一条**：∂h_T/∂h_t = ∏(U·diag(1−h²))——
                   梯度走多远就乘多少次，乘子几何平均 <1 消失、
                   >1 爆炸，分界线是 1
                2. **指数的残酷**：0.9²⁰≈0.12、1.05⁵⁰≈11.5、
                   0.5⁵⁰≈10⁻¹⁵——离 1 一点点，乘几十步就是
                   两个宇宙
                3. **消失两大帮凶**：ρ(U)<1（标准初始化的常态，
                   前向稳定=反向断流）+ tanh 饱和（|h|→1 时
                   局部导数→0，34 课练习 1 的 0.03）
                4. **爆炸是急症**：loss 尖峰现场 9 次、梯度范数
                   2,692——三行 clip（全局范数、只缩长度、
                   eps 防零）截住后 0 尖峰
                5. **clip 治炸不治聋**：截上不补下的单向阀门；
                   clip 后 loss 平稳地停在瞎猜线——gap=20 的
                   病根是消失
                6. **消失只能动手术**：长依赖审判（gap=3 → 99%，
                   gap=20 → 31% 纹丝不动）证明调参无用——
                   出路是给梯度修一条不连乘的路（36 课 LSTM）
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

                病历写完，只剩一个结论：**想学长依赖，就得让
                梯度有一条不连乘的路。**

                回头看 RNN 的原罪：hₜ = tanh(W·xₜ + U·hₜ₋₁ + b)
                ——每一步，记忆都要**整个穿过** tanh 这个窄门。
                梯度想回到 20 步前，就必须连续穿 20 次窄门，
                每次都被乘一个 ≤1 的数。

                1997 年的手术方案（LSTM）只加了一条路：

                ```
                cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ gₜ        ← 细胞状态：加法更新！
                ```

                c 的更新是**加法**不是"穿过 tanh 重写"——梯度
                沿 c 回传时，每步乘的是 **fₜ（遗忘门的值，
                网络自己学的）**。想让梯度活着走 100 步？
                网络只需学会把 f 保持接近 1——**"乘什么"从
                结构的固定刑期，变成了可学习的选择题。**
                """
            ),
            mo.callout(
                mo.md(
                    """
                    **下一课预告** — `36_LSTM`：

                    给记忆装三道阀门：

                    - **遗忘门 f**（旧记忆留多少）、**输入门 i**
                      （新信息写多少）、**输出门 o**（记忆放出来
                      多少）——三道门都是"网络看着输入自己决定"
                    - 亲手用 NumPy 实现完整 LSTM 单元（比 RNN
                      多几行，但每行都有明确的岗位）
                    - 数值验证"梯度高速路"：同样的 20 步回声
                      任务，RNN 卡死（35 课），LSTM 学会——
                      35 课的病人今天出院
                    - 顺手认识 GRU（LSTM 的两门精简版）和
                      它在工业界的现状

                    🚪 三道门后，是一条不堵车的高速路。

                    （然后 37 课收官：拿真语料训练字符级语言
                    模型，让 LSTM 亲口"说人话"。）
                    """
                ),
                kind="success",
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
