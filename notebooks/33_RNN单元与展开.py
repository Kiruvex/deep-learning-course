import marimo

__generated_with = "0.25.2"
app = marimo.App(app_title="33 · RNN 单元与展开", width="medium")


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
        # 33 · RNN 单元与展开：给网络装上记忆

        **从零到 Transformer · 手写深度学习** ｜ 第 6 部分 · 循环神经网络（2/6）
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📖 本课目标（先读我）": mo.md(
                """
                **这一课把 32 课结尾的三个想法，装进一个只有一行核心
                公式的零件里。**

                32 课的结论：滑窗注定失明，网络需要"记忆"。可行的
                方案朴素得惊人——一个定长向量 **h**（隐状态）当
                工作记忆，每读一个新字符就更新一次；处理第 1 个
                字符和第 1000 个字符，用**同一组权重**。

                这就是**循环神经网络**（RNN, Recurrent Neural
                Network）：1986 年就被想出来的结构，是 LSTM、
                GRU 乃至 Transformer 注意力机制的共同祖先。

                学完你将能：

                - 写出 RNN 的核心公式 **hₜ = tanh(W·xₜ + U·hₜ₋₁ + b)**
                  并说清它的两条输入、一套权重
                - 亲手用 NumPy 实现 `rnn_cell` 和 `rnn_forward`，
                  喂一句话看隐状态逐字流动
                - 画（并看懂）**沿时间展开图**：为什么它就是
                  26 课"权值共享"的时间版
                - 用**脉冲探测实验**亲眼测量"记忆的寿命"——
                  并解释为什么 tanh 是必需的护栏
                - 算一笔诚实的参数账：RNN 参数量与序列长度无关，
                  但短序列上反而比滑窗 MLP 贵

                前置：32 课的三堵墙与滑窗天花板、26 课的权值共享、
                19 课的 tanh、22 课的计算图。

                本课全程**随机权重**（不训练）——先看清这个结构
                **天生**就会什么、不会什么；34 课再谈训练。
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "🧠 概念 1/3 · 隐状态：网络的工作记忆": mo.md(
                """
                **隐状态**（hidden state）h 是一个固定长度的向量
                （比如 16 维），它就是网络的"工作记忆"：

                - 读第 1 个字符前，h₀ = 0（一片空白）
                - 每读一个字符，更新一次：**hₜ = f(hₜ₋₁, xₜ)**
                - 任意时刻预测下一个字符，只看当前的 hₜ

                注意 h 的**长度固定**——不管读了 3 个字还是 3000
                个字，记忆都压在这 16 个数里。这不是硬盘（全存），
                也不是无记忆（全忘），而是一块**边读边写的内存条**：
                该记什么、该挤掉什么，由 f 的权重决定——而这正是
                要**学**的东西。

                和滑窗的本质区别：滑窗每次"回头重看"最近 W 个字符
                （视野固定）；RNN 的 h 是**所有历史的压缩摘要**——
                理论上想记多远记多远（实际能记多远，35 课见分晓）。
                """
            ),
            "⚙️ 概念 2/3 · RNN 单元：一行核心公式": mo.md(
                """
                f 怎么设计？RNN 的答案简单到让人怀疑：

                ```
                hₜ = tanh(W·xₜ + U·hₜ₋₁ + b)
                ```

                拆开看，它就是你 20 课就认识的"线性层 + 激活"，
                只是**输入有两路**：

                - **W·xₜ**（输入权重 × 当前字符）：新进来的信息
                - **U·hₜ₋₁**（循环权重 × 旧记忆）：留存的信息
                - tanh 把结果压回 [−1, 1]（19 课——没有它，
                  记忆会指数爆炸，演示 2 亲手复现给你看）

                参数：W 是 (d_h × d_x)，U 是 (d_h × d_h)，b 是
                (d_h)。**注意 U 是方阵**——旧记忆和新记忆同维，
                因为 h 每步既是输出又是输入（这正是"循环"的由来）。

                要预测下一个字符，再接一个普通线性层：
                yₜ = V·hₜ + c，softmax 出分布（21 课）。
                """
            ),
            "🔁 概念 3/3 · 沿时间展开：权值共享的时间版": mo.md(
                """
                "循环"不好画也不好算，标准做法是**沿时间展开**
                （unroll）：把"同一个单元用 T 次"画成"T 个一模
                一样的单元排成一排"：

                ```
                x₁ → [cell] → h₁ → [cell] → h₂ → [cell] → h₃ → …
                 h₀ ↗        h₁ ↗         h₂ ↗      （同一套 W, U, b）
                ```

                眼熟吗？**26 课卷积核扫过全图、权重一份不动**——
                这里是**同一个 cell 扫过全序列，权重一份不动**。
                空间版换成了时间版，好处一模一样：

                1. **参数量与序列长度无关**：读 3 个字和 3000 个
                   字，都是那一套 W、U、b（演示 3 的账本）
                2. **位置无关性免费附赠**："追"出现在第 2 位还是
                   第 20 位，都是同一个 cell 处理它——32 课的
                   墙一（位置绑定）被结构性拆除

                展开还揭示了另一层身份：**RNN 展开图就是一个
                深度为 T 的前馈网络**（每层权重共享）——所以
                "训练 RNN"就是"训练深网络"，22 课的反向传播
                沿着时间维一路回传（34 课 BPTT），而"深网络
                的梯度难题"（35 课）也会在时间轴上原样上演。
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
        ### 演示 1 · 亲手实现 RNN 前向，喂它一句话

        先看展开图（4 个时间步）：每个蓝框是**同一个** RNN 单元
        ——同一套 W、U、b；红色横向箭头就是"循环"：上一步的
        h 搬到下一步当输入。

        然后实现两个函数（各一行核心代码）：

        - `rnn_cell(x, h_prev)`：一个时间步——tanh(两路输入加起来)
        - `rnn_forward(xs)`：循环调用，返回所有时间步的 h

        喂它一句话（词表 = 句子里出现的 7 个字），打印每个字
        读完后隐状态的前 3 维、最后一步的"下一字预测分布"。
        权重是**随机**的——预测当然是垃圾（7 类乱猜，分布
        略有起伏），但**管道已经通了**：每个字进来，h 流动一次，
        softmax 给出一个分布。34 课训练它，37 课让它编故事。
        """
    )
    return


@app.cell
def _(np, plt):
    # 演示 1a · 沿时间展开图（同一套权重的 cell 排成一排）
    _fig, _ax = plt.subplots(figsize=(11.5, 4.6))
    _ax.set_xlim(0, 12.2)
    _ax.set_ylim(0, 5.2)
    _ax.axis("off")
    _steps = 4
    for _k in range(_steps):
        _x0 = 0.7 + _k * 2.8
        # x_t 输入框
        _ax.add_patch(plt.Rectangle((_x0 - 0.42, 0.35), 0.84, 0.66,
                                    facecolor="#e8f0fe", edgecolor="#4c78a8", lw=1.4))
        _ax.text(_x0, 0.68, f"$x_{{{_k+1}}}$", ha="center", va="center",
                 fontsize=13, fontweight="bold")
        # cell 框
        _ax.add_patch(plt.Rectangle((_x0 - 0.62, 1.9), 1.24, 1.15,
                                    facecolor="#dceefb", edgecolor="#2b6cb0", lw=1.8))
        _ax.text(_x0, 2.72, "RNN\ncell", ha="center", va="center", fontsize=10.5,
                 fontweight="bold")
        # h_t 输出框
        _ax.add_patch(plt.Rectangle((_x0 - 0.42, 3.85), 0.84, 0.66,
                                    facecolor="#e6f4ea", edgecolor="#2e7d32", lw=1.4))
        _ax.text(_x0, 4.18, f"$h_{{{_k+1}}}$", ha="center", va="center",
                 fontsize=13, fontweight="bold")
        # 箭头
        _ax.annotate("", xy=(_x0, 3.12), xytext=(_x0, 1.06),
                     arrowprops=dict(arrowstyle="->", color="#4c78a8", lw=1.6))
        _ax.annotate("", xy=(_x0, 4.22), xytext=(_x0, 3.1),
                     arrowprops=dict(arrowstyle="->", color="#2e7d32", lw=1.6))
        if _k < _steps - 1:
            _ax.annotate("", xy=(_x0 + 2.8 - 0.66, 2.47), xytext=(_x0 + 0.46, 2.47),
                         arrowprops=dict(arrowstyle="->", color="tab:red", lw=2.0))
    _ax.text(1.9, 2.72, "$h_0 = 0$", fontsize=11, color="tab:red", va="center", ha="right")
    _ax.text(6.1, 0.95, "每个字 one-hot 进来", fontsize=9.5, color="#4c78a8")
    _ax.text(6.1, 4.62, "每步都可以 softmax 出「下一字分布」", fontsize=9.5, color="#2e7d32")
    _ax.text(6.1, 1.62, "红色横向箭头 = 循环：上一步的 h 是下一步的输入（这就是「循环」的全部含义）",
             fontsize=9.5, color="tab:red")
    _ax.text(6.1, 5.0, "沿时间展开：4 个一模一样的 cell（同一套 W、U、b）处理 4 个字符",
             fontsize=10.5, fontweight="bold")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(np):
    # 演示 1b · rnn_cell / rnn_forward：本部分的两大地基函数
    def rnn_cell33(x_t, h_prev, Wxh, Whh, b):
        """一个时间步：新记忆 = tanh(输入路 + 记忆路 + 偏置)。"""
        return np.tanh(Wxh @ x_t + Whh @ h_prev + b)

    def rnn_forward33(xs, Wxh, Whh, b, h0=None):
        """整条序列：循环调用 cell，返回 (T, d_h) 的隐状态序列。"""
        _h = np.zeros(Whh.shape[0]) if h0 is None else h0.copy()
        _hs = []
        for _x in xs:
            _h = rnn_cell33(_x, _h, Wxh, Whh, b)
            _hs.append(_h.copy())
        return np.array(_hs)

    def softmax33(z):
        _z = z - z.max()
        _e = np.exp(_z)
        return _e / _e.sum()

    # ---- 建一个随机 RNN（d_h=16），喂一句话 ----
    TEXT33 = "深度学习很有趣"
    VOCABR33 = list(dict.fromkeys(TEXT33))          # 动态词表：按首次出现排序
    _IDXR33 = {c: i for i, c in enumerate(VOCABR33)}

    def init_rnn33(V, d_h, alpha=1.0, seed=3):
        """随机权重：W ~ N(0,1/√V)，U ~ N(0,α/√d_h)（α = 记忆强度旋钮）。"""
        _r = np.random.default_rng(seed)
        return (_r.normal(0, 1 / np.sqrt(V), (d_h, V)),
                _r.normal(0, alpha / np.sqrt(d_h), (d_h, d_h)),
                np.zeros(d_h))

    d_h33 = 16
    Wxh33, Whh33, b33 = init_rnn33(len(VOCABR33), d_h33, alpha=1.0)
    _xs33 = [np.eye(len(VOCABR33))[_IDXR33[c]] for c in TEXT33]
    hs_demo33 = rnn_forward33(_xs33, Wxh33, Whh33, b33)

    print(f"输入：「{TEXT33}」（词表 {VOCABR33}，d_h = {d_h33}）")
    print("\n每读一个字，隐状态 h 的前 3 维：")
    for _t, _c in enumerate(TEXT33):
        _arrow = "→" if _t else " "
        print(f"  {_arrow} 读到「{_c}」后 h[{_t}] 前 3 维 = {np.round(hs_demo33[_t][:3], 2)}")

    # 随机权重的"下一字预测"（当然是垃圾——但管道通了）
    _r33 = np.random.default_rng(5)
    Why33 = _r33.normal(0, 1 / np.sqrt(d_h33), (len(VOCABR33), d_h33))
    by33 = np.zeros(len(VOCABR33))
    _probs33 = softmax33(Why33 @ hs_demo33[-1] + by33)
    print("\n读完整个句子后，随机权重给出的「下一字」分布：")
    for _c, _p in zip(VOCABR33, _probs33):
        _bar = "█" * int(_p * 40)
        print(f"  「{_c}」 {_p:.3f} {_bar}")
    print("→ 没训练的网络在乱猜（均匀 1/7 ≈ 0.143 附近晃），但结构已经通了：")
    print("   字符进 → h 流动 → softmax 出分布。34 课教它说人话。")
    return (TEXT33, VOCABR33, Why33, b33, by33, d_h33, hs_demo33, init_rnn33,
            rnn_cell33, rnn_forward33, softmax33, Wxh33, Whh33)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 2 · tanh 是护栏；记忆有寿命

        两个实验，回答两个问题：

        **实验 A · 没有 tanh 会怎样？** 把激活函数去掉（线性
        RNN），同样的随机权重跑 60 步：‖h‖ 从 1.9 指数长到
        **233**（对数坐标下一条直线）——旧记忆每步被 U 放大一圈，
        60 步就是 10¹² 倍的温床。tanh 把它死死压在 [−1,1]ᵈ，
        ‖h‖ 稳定在 2.5 附近——**护栏，不是装饰**。

        **实验 B · 记忆的寿命有多长？** 脉冲探测：两条随机
        字符流，唯一的区别是**第 10 步换了一个字符**；之后
        每步比较两条轨迹的 h 差多远（"蝴蝶的翅膀扇了多大"）。

        - α = 0.2（U 很弱）：痕迹 3 步死光——**金鱼记忆**
        - α = 1.0：痕迹一路衰减，15 步后归于沉寂
        - α = 1.3：痕迹更持久，但仍在衰减

        α（U 的强度）就是**记忆寿命的旋钮**。注意：这是**随机
        权重**的行为——训练能不能让网络自己学会"该记多久"？
        35 课会看到一个坏消息：梯度沿时间回传时走的是同一条
        衰减路——**前向记不住的，反向也传不动**。
        """
    )
    return


@app.cell
def _(init_rnn33, np, plt, rnn_forward33):
    # 演示 2 · 实验A: tanh 护栏 + 实验B: 脉冲探测
    # ---- 实验A：线性 RNN vs tanh RNN 的 ‖h‖ ----
    _rngA = np.random.default_rng(10)
    _xsA = np.eye(3)[_rngA.integers(0, 3, 60)]
    _WxA, _WhA, _bA = init_rnn33(3, 16, alpha=1.0)

    def _forward_act33(xs, Wxh, Whh, b, act):
        _h = np.zeros(Whh.shape[0]); _hs = []
        for _x in xs:
            _h = act(Wxh @ _x + Whh @ _h + b)
            _hs.append(_h.copy())
        return np.array(_hs)

    h_tanh33 = _forward_act33(_xsA, _WxA, _WhA, _bA, np.tanh)
    h_lin33 = _forward_act33(_xsA, _WxA, _WhA, _bA, lambda _z: _z)
    print("实验 A · ‖h‖ 随时间：")
    print(f"  tanh RNN : 前 5 步 {np.round(np.linalg.norm(h_tanh33, axis=1)[:5], 2)}"
          f" … 60 步 {np.linalg.norm(h_tanh33[-1]):.2f}（被压在笼子里）")
    print(f"  线性 RNN : 前 5 步 {np.round(np.linalg.norm(h_lin33, axis=1)[:5], 2)}"
          f" … 60 步 {np.linalg.norm(h_lin33[-1]):.1f}（指数爆炸，对数坐标一条直线）")

    # ---- 实验B：脉冲探测 ----
    def trace_curve33(alpha, T=40, pulse_pos=10, seed=10):
        """常规流 vs 脉冲流（第 pulse_pos 步换一个字符）的 h 逐范数差。"""
        _r = np.random.default_rng(seed)
        _xs = np.eye(3)[_r.integers(0, 3, T)]
        _xs_pulse = _xs.copy()
        _xs_pulse[pulse_pos] = np.eye(3)[(int(_xs[pulse_pos].argmax()) + 1) % 3]
        _Wx, _Wh, _bb = init_rnn33(3, 16, alpha)
        _ha = rnn_forward33(_xs, _Wx, _Wh, _bb)
        _hb = rnn_forward33(_xs_pulse, _Wx, _Wh, _bb)
        return np.linalg.norm(_ha - _hb, axis=1)

    trace_alphas33 = (0.2, 1.0, 1.3)
    trace_curves33 = {a: trace_curve33(a) for a in trace_alphas33}
    print("\n实验 B · 脉冲在第 10 步扇了下翅膀，痕迹持续多久：")
    for _a in trace_alphas33:
        _tr = trace_curves33[_a]
        _half = next((_i for _i, _v in enumerate(_tr[11:]) if _v < _tr[11] * 0.5), None)
        print(f"  α={_a}: 脉冲瞬间痕迹 {_tr[10]:.2f} → 5 步后 {_tr[15]:.3f}"
              f" → 15 步后 {_tr[25]:.4f}（半衰约 {('约 %d 步' % (_half+1)) if _half is not None else '很久'}）")

    # ---- 图 ----
    _fig, _axes = plt.subplots(1, 2, figsize=(11.6, 4.2))
    _axes[0].plot(np.linalg.norm(h_tanh33, axis=1), color="tab:blue", lw=1.8, label="tanh RNN")
    _axes[0].plot(np.linalg.norm(h_lin33, axis=1), color="tab:red", lw=1.8, label="线性 RNN（无激活）")
    _axes[0].set_yscale("log")
    _axes[0].set_xlabel("时间步 t")
    _axes[0].set_ylabel(r"$\|h_t\|$（对数坐标）")
    _axes[0].set_title("实验 A · tanh 是护栏：线性 RNN 60 步百倍爆炸", fontsize=11)
    _axes[0].legend(fontsize=9)
    _axes[0].grid(True, alpha=0.3)
    _axes[0].axhline(np.sqrt(16), color="gray", ls=":", lw=1)
    _axes[0].text(1, np.sqrt(16) * 1.1, r"$\|h\|$=4（tanh 满格）", fontsize=8, color="gray")

    for _a, _c in zip(trace_alphas33, ("tab:green", "tab:blue", "tab:orange")):
        _axes[1].semilogy(range(40), np.maximum(trace_curves33[_a], 1e-6),
                          lw=1.8, color=_c, label=f"α = {_a}")
    _axes[1].axvline(10, color="gray", ls="--", lw=1.2)
    _axes[1].text(10.6, 2.0, "脉冲\n（第 10 步换字）", fontsize=8.5, color="gray")
    _axes[1].set_xlabel("时间步 t")
    _axes[1].set_ylabel("两条轨迹的 ‖Δh‖（对数坐标）")
    _axes[1].set_title("实验 B · 脉冲探测：α 是记忆寿命的旋钮", fontsize=11)
    _axes[1].legend(fontsize=9)
    _axes[1].grid(True, alpha=0.3)
    _fig.tight_layout()
    _fig
    return (h_lin33, h_tanh33, trace_alphas33, trace_curves33, trace_curve33)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 3 · 参数账本：与长度无关的代价

        同样"处理一句话"（词表 V=5、隐层 64），两种架构的参数：

        - **滑窗 MLP**：输入 T×V 维，第一层参数 = T·V·64 + 64
          ——**随句长 T 线性膨胀**
        - **RNN**：W (5×64) + U (64×64) + b (64) = **4,480**
          ——**与 T 无关**（读 3 个字和 3000 个字都是它）

        诚实的账本要算两页：T < 14 时滑窗其实**更省**——权值
        共享是"为长序列买单"的。语言的真实依赖动辄几十上百
        步（一句话开头的人名，隔三十个字被代词引用），这笔
        买卖在真实语料上稳赚。
        """
    )
    return


@app.cell
def _(np, plt):
    # 演示 3 · 参数账本（滑窗线性涨 vs RNN 水平线）
    V_p33, H_p33 = 5, 64
    rnn_params33 = V_p33 * H_p33 + H_p33 * H_p33 + H_p33
    print(f"RNN：W({V_p33}×{H_p33}) + U({H_p33}×{H_p33}) + b({H_p33}) = {rnn_params33:,}（与 T 无关）")
    _Ts = np.arange(1, 101)
    _mlp_params = _Ts * V_p33 * H_p33 + H_p33
    _cross = int(np.interp(rnn_params33, _mlp_params, _Ts))
    print(f"滑窗 MLP：T·{V_p33}·{H_p33} + {H_p33}，随 T 线性涨；T = {_cross} 处被 RNN 反超")

    _fig, _ax = plt.subplots(figsize=(8.4, 4.3))
    _ax.plot(_Ts, _mlp_params, color="tab:red", lw=2, label="滑窗 MLP（参数随 T 线性涨）")
    _ax.axhline(rnn_params33, color="tab:blue", lw=2, ls="--",
                label=f"RNN（{rnn_params33:,} 个，与 T 无关）")
    _ax.axvline(_cross, color="gray", ls=":", lw=1.4)
    _ax.scatter([_cross], [rnn_params33], s=70, color="gray", zorder=5)
    _ax.annotate(f"T = {_cross}：交叉点", xy=(_cross, rnn_params33),
                 xytext=(_cross + 6, rnn_params33 * 0.55), fontsize=10,
                 arrowprops=dict(arrowstyle="->", color="gray"))
    _ax.set_xlabel("句长 T（要处理的字符数）")
    _ax.set_ylabel("参数量")
    _ax.set_title("诚实的账本：短句滑窗更省，长句权值共享稳赚")
    _ax.legend(fontsize=9.5, loc="upper left")
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()
    _fig
    return (rnn_params33,)

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
        ### 交互实验 · 记忆寿命实验室

        左边输入**任意文本**（动态建词表，每个不同的字符算一类），
        右边调**记忆强度 α**（U 的缩放，0.1 ~ 1.5）。每拖一次都
        现场做两件事：

        - 上图：h 的**热图**（横轴时间步、纵轴 16 个记忆格，
          红/蓝 = 正/负激活）——看记忆如何随字符流动
        - 下图：该 α 下的**脉冲探测曲线**（第 10 步换一个字符，
          两条轨迹的 Δh 随时间衰减）——这就是"记忆寿命"

        试试：α=0.2（金鱼：热图每列都像重开一局）、α=1.0
        （h 有连续的"条纹"，脉冲痕迹拖出一条长尾）、α=1.5
        （条纹更顽强，但注意它开始抖了）。
        """
    )
    return


@app.cell
def _(mo):
    # 交互控制：文本 + 记忆强度
    text_in33 = mo.ui.text(
        value="深度学习很有趣，深度学习很有用",
        label="📝 输入一段文本（自动建词表，取前 60 个字符）")
    alpha_slider33 = mo.ui.slider(0.1, 1.5, value=1.0, step=0.05, debounce=True,
                                  label="🧠 记忆强度 α（U 的缩放）")
    mo.vstack([text_in33, alpha_slider33])
    return (alpha_slider33, text_in33)


@app.cell
def _(alpha_slider33, d_h33, init_rnn33, mo, np, plt, rnn_forward33, text_in33, trace_curve33):
    # ===== 交互实验 · 记忆寿命实验室（响应单元）=====
    _raw = (text_in33.value or "").strip()[:60]
    _chars = [c for c in _raw if not c.isspace()]
    if len(set(_chars)) < 2:
        _fig = plt.figure(figsize=(10.5, 1.2))
        plt.text(0.5, 0.5, "请输入至少两个不同的字符", ha="center", va="center",
                 fontsize=14, color="gray", transform=plt.gca().transAxes)
        _verdict = "⏳ 输入至少两个不同字符（例如「深度学习很有趣」）"
    else:
        _vocab = list(dict.fromkeys(_chars))
        _idx = {c: i for i, c in enumerate(_vocab)}
        _V = len(_vocab)
        _xs = [np.eye(_V)[_idx[c]] for c in _chars]
        _alpha = alpha_slider33.value
        _Wx, _Wh, _bb = init_rnn33(_V, d_h33, alpha=_alpha)
        _hs = rnn_forward33(_xs, _Wx, _Wh, _bb)
        _tr = trace_curve33(_alpha)

        _fig, _axes = plt.subplots(2, 1, figsize=(11.4, 6.4),
                                   gridspec_kw={"height_ratios": [1.2, 1.0]})
        _im = _axes[0].imshow(_hs.T, cmap="RdBu_r", aspect="auto",
                              vmin=-1, vmax=1, interpolation="nearest")
        _axes[0].set_yticks([0, 7, 15], ["记忆格 1", "记忆格 8", "记忆格 16"], fontsize=9)
        _axes[0].set_xlabel(f"时间步（第 t 个字符，共 {len(_chars)} 步）")
        _axes[0].set_title(f"隐状态热图：「{_raw[:18]}{'…' if len(_raw) > 18 else ''}」"
                           f"（词表 {len(_vocab)} 字符，α = {_alpha:.2f}）", fontsize=11)
        _fig.colorbar(_im, ax=_axes[0], shrink=0.85, pad=0.01)
        # 竖线：把每个字符标在时间轴上（前 20 个）
        _n_mark = min(len(_chars), 20)
        _axes[0].set_xticks(range(0, _n_mark),
                            list(_chars[:_n_mark]), fontsize=8)

        _axes[1].semilogy(range(40), np.maximum(_tr, 1e-6), color="tab:purple", lw=2)
        _axes[1].axvline(10, color="gray", ls="--", lw=1.2)
        _axes[1].text(10.6, max(_tr[5:]) * 0.9, "脉冲", fontsize=9, color="gray")
        _half = next((_i for _i, _v in enumerate(_tr[11:]) if _v < _tr[11] * 0.5), None)
        _axes[1].set_xlabel("时间步 t")
        _axes[1].set_ylabel(r"$\|\Delta h\|$（对数坐标）")
        _axes[1].set_title("脉冲探测：第 10 步换一个字符，痕迹多久消失", fontsize=11)
        _axes[1].grid(True, alpha=0.3)

        if _alpha < 0.4:
            _verdict = (f"**金鱼记忆**（α = {_alpha:.2f}）：U 太弱，旧记忆每步被压掉"
                        f"{(1 - _alpha):.0%} 以上——脉冲痕迹几步死光，h 里基本只剩"
                        "刚读过的那个字符。记得住「上一个字」，记不住「上一句」")
        elif _alpha <= 1.05:
            _verdict = (f"**健康记忆带**（α = {_alpha:.2f}）：脉冲痕迹半衰约 "
                        f"{(_half + 1) if _half is not None else '10+'} 步——热图里能看到"
                        "横跨多步的条纹（某些记忆格被同一个词「点亮」后延续）。"
                        "热图竖列的相似度，就是记忆在流动的证据")
        else:
            _verdict = (f"**长记忆 / 临界区**（α = {_alpha:.2f}）：谱半径逼近 1，"
                        "脉冲痕迹衰减得很慢——记忆长寿，但也开始不稳定："
                        "注意热图边缘是否出现大面积同色块（记忆「结壳」）。"
                        "35 课会看到 α 的另一面：前向越持久，反向梯度越容易走极端")
    _fig.tight_layout()
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
                **练习 1 · 手算一个时间步**：2 维小 RNN，给定权重、
                输入 xₜ = [1, −1]、旧记忆 hₜ₋₁ = [0.5, 0.5]，笔算
                新记忆 hₜ 的两个分量（tanh 的值可以查表 / 用计算器，
                保留两位小数）。

                **练习 2 · 补全 rnn_forward**：循环体里的一行——
                拿 rnn_cell 更新记忆。补完与参考实现逐元素对账。

                **练习 3 · 参数侦探**：词表 3 个字符、隐状态 4 维
                的 RNN——单元本身有几个参数？加上输出层呢？
                """
            )
        }
    )
    return


@app.cell
def _(mo, np):
    # ===== 练习 1 · 手算一个时间步 =====
    Wx_ex1_33 = np.array([[1.0, 0.0], [0.0, 1.0]])          # W = I（输入权重）
    Wh_ex1_33 = np.array([[0.5, 0.0], [0.0, 0.5]])          # U = 0.5·I（循环权重）
    b_ex1_33 = np.zeros(2)
    x_ex1_33 = np.array([1.0, -1.0])
    h_prev_ex1_33 = np.array([0.5, 0.5])
    _z_ref33 = Wx_ex1_33 @ x_ex1_33 + Wh_ex1_33 @ h_prev_ex1_33 + b_ex1_33
    print("两路先各自算，再相加：")
    print(f"  W·x   = {Wx_ex1_33 @ x_ex1_33}")
    print(f"  U·h   = {Wh_ex1_33 @ h_prev_ex1_33}")
    print(f"  z = W·x + U·h + b = {_z_ref33}")
    print(f"  hₜ = tanh(z) = ?（两个分量，保留两位）")

    h_new_ex1_33 = [None, None]                            # ← ①② tanh(z) 的两个分量

    if all(_v is not None for _v in h_new_ex1_33):
        _truth = np.tanh(_z_ref33)
        if all(abs(_a - _t) < 0.02 for _a, _t in zip(h_new_ex1_33, _truth)):
            _msg = (f"✅ 正确：hₜ = [{_truth[0]:.2f}, {_truth[1]:.2f}]。"
                    "第一路贡献 [1, −1]、第二路贡献 [0.25, 0.25]，"
                    "合成 z = [1.25, −0.75] 后过 tanh 压缩——"
                    "正的那格 0.85（接近满格），负的那格 −0.64")
        else:
            _msg = (f"❌ 参考 [{_truth[0]:.2f}, {_truth[1]:.2f}]。"
                    "提示：z = [1.25, −0.75]，分别算 tanh(1.25) 和 tanh(−0.75)")
    else:
        _msg = "⏳ 两个分量都算出来（提示：先算 z = W·x + U·h，再逐分量 tanh）"
    mo.md(_msg)
    return (Wh_ex1_33, h_new_ex1_33, h_prev_ex1_33, x_ex1_33)


@app.cell
def _(mo, np):
    # ===== 练习 2 · 补全 rnn_forward =====
    def rnn_fwd_ex2_33(xs, Wxh, Whh, b, h0=None):
        """循环体补一行：用 cell 更新 _h。"""
        _h = np.zeros(Whh.shape[0]) if h0 is None else h0.copy()
        _hs = []
        for _x in xs:
            None                                   # ← 补：一行，更新 _h（tanh 两路相加）
            _hs.append(_h.copy())
        return np.array(_hs)

    fwd_ready_ex2_33 = None                        # ← 补完后，把这行换成 rnn_fwd_ex2_33

    if fwd_ready_ex2_33 is not None:
        try:
            _r = np.random.default_rng(0)
            _xs = np.eye(3)[_r.integers(0, 3, 12)]
            _Wx = _r.normal(0, 0.4, (8, 3)); _Wh = _r.normal(0, 0.25, (8, 8)); _bb = np.zeros(8)
            _got = fwd_ready_ex2_33(_xs, _Wx, _Wh, _bb)
            _h = np.zeros(8); _ref = []
            for _x in _xs:
                _h = np.tanh(_Wx @ _x + _Wh @ _h + _bb)
                _ref.append(_h.copy())
            _ref = np.array(_ref)
            if _got.shape == (12, 8) and np.allclose(_got, _ref, atol=1e-12):
                _msg = ("✅ 逐元素一致（12 步 × 8 维全对）！"
                        "注意两件小事：① `h.copy()` 必须存——不然列表里全是同一个"
                        "数组的引用，最后全变成最后一步；② h 别忘了在循环外初始化——"
                        "这两件事是 RNN 实现里最经典的两个 bug")
            else:
                _msg = "❌ 与参考不一致：检查循环体那一行（tanh(W·x + U·h + b))"
        except (TypeError, ValueError) as _e:
            _msg = f"❌ 运行出错：{_e}——循环体一行：_h = np.tanh(Wxh @ _x + Whh @ _h + b)"
    else:
        _msg = "⏳ 补一行：_h = np.tanh(Wxh @ _x + Whh @ _h + b)"
    mo.md(_msg)
    return (fwd_ready_ex2_33, rnn_fwd_ex2_33)


@app.cell
def _(mo):
    # ===== 练习 3 · 参数侦探 =====
    V_ex3_33, d_h_ex3_33 = 3, 4                 # 词表 3 字符，隐状态 4 维

    n_cell_ex3_33 = None                        # ← ① RNN 单元的参数量（整数）
    n_total_ex3_33 = None                       # ← ② 再加输出层（V·d_h + V）后的总量

    if n_cell_ex3_33 is not None and n_total_ex3_33 is not None:
        _t_cell = V_ex3_33 * d_h_ex3_33 + d_h_ex3_33 ** 2 + d_h_ex3_33
        _t_total = _t_cell + V_ex3_33 * d_h_ex3_33 + V_ex3_33
        if n_cell_ex3_33 == _t_cell and n_total_ex3_33 == _t_total:
            _msg = (f"✅ 正确：单元 {V_ex3_33}×{d_h_ex3_33} + {d_h_ex3_33}×{d_h_ex3_33} + "
                    f"{d_h_ex3_33} = **{_t_cell}**；加输出层 {_t_cell} + "
                    f"{V_ex3_33}×{d_h_ex3_33} + {V_ex3_33} = **{_t_total}**。"
                    "注意 U 是 4×4 的方阵（旧记忆和新记忆同维），它通常是参数大头")
        else:
            _msg = (f"❌ 参考：单元 {_t_cell}（W 12 + U 16 + b 4）、"
                    f"含输出层 {_t_total}（再加 Why 12 + by 3）。"
                    "别忘了 U 是 d_h × d_h 的方阵、偏置有 d_h 个")
    else:
        _msg = "⏳ 两空都填上（提示：三样东西——W、U、b；U 是方阵）"
    mo.md(_msg)
    return (V_ex3_33, d_h_ex3_33, n_cell_ex3_33, n_total_ex3_33)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 1 参考答案（先自己写过再看！）": mo.md(
                """
                **hₜ = [0.85, −0.64]。**

                ```
                W·x = [1, −1]          （输入路：当前字符的信息）
                U·h = [0.25, 0.25]     （记忆路：旧记忆减半留存）
                z   = [1.25, −0.75]
                hₜ  = [tanh(1.25), tanh(−0.75)] = [0.85, −0.64]
                ```

                tanh 在这里干了三件事：把结果压回 (−1, 1)、
                在 0 附近近似线性（z 小则 h ≈ z）、在两头饱和
                （z 大则 h 封顶 ±1）。第一格 z=1.25 已经进了
                饱和区半只脚（0.85 离 1 不远了）——**饱和**这个
                词在 35 课会变成主角。
                """
            ),
            "✅ 练习 2 参考答案（先自己写过再看！）": mo.md(
                """
                循环体一行：

                ```python
                _h = np.tanh(Wxh @ _x + Whh @ _h + b)
                ```

                两个经典坑都值得记住：

                - **必须存副本** `_hs.append(_h.copy())`：NumPy
                  的赋值是引用，不 copy 的话列表里 12 个元素
                  指向同一个数组——最后打印出来"每一步都一样"，
                  且都等于最后一步
                - **h 要在循环外初始化**（且每次 forward 调用都
                  从 h₀ = 0 重新开始）：忘了的话上一次调用的
                  记忆会"泄漏"进下一次
                """
            ),
            "✅ 练习 3 参考答案（先自己写过再看！）": mo.md(
                """
                **单元 32 个；含输出层 47 个。**

                ```
                W:  d_x × d_h = 3 × 4 = 12
                U:  d_h × d_h = 4 × 4 = 16   ← 方阵，参数大头
                b:  d_h       = 4
                          单元小计 = 32
                Why: d_h × V = 4 × 3 = 12
                by:  V       = 3
                          总计 = 47
                ```

                两个观察：① **U 是参数大头**（16/32 = 一半），
                因为它是 d_h × d_h 的方阵——真实模型里 d_h
                动辄几百，U 的参数量以 d_h² 的速度膨胀，这也是
                后来 Transformer 用"低秩注意力"省参数的伏笔；
                ② 输出层与时间无关——**每个时间步共用同一个
                Why**，又是权值共享。
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
                1. **隐状态 h = 工作记忆**：定长向量，边读边更新；
                   不是全存的硬盘，也不是全忘的内存条——记什么、
                   挤掉什么，由权重决定（这正是要学的东西）
                2. **核心公式** hₜ = tanh(W·xₜ + U·hₜ₋₁ + b)：
                   输入路（新字符）+ 记忆路（旧记忆）+ tanh 护栏，
                   就是 20 课"线性层+激活"的双输入版
                3. **展开 = 权值共享的时间版**：同一套 W、U、b
                   扫过全序列（26 课卷积核扫过全图的时间版）；
                   位置绑定被结构性拆除，参数与 T 无关
                4. **tanh 是必需的护栏**：线性 RNN 的 ‖h‖ 60 步
                   指数长 124 倍；tanh 把它压在 √d_h 附近——
                   没有激活函数的 RNN 根本没法用
                5. **α = 记忆寿命旋钮**：脉冲探测实验——α=0.2
                   痕迹 3 步死光（金鱼），α=1.0 拖 15 步长尾；
                   但前向的痕迹衰减路，也是 35 课反向梯度
                   要走的路（伏笔已埋）
                6. **诚实的参数账**：RNN 4,480 个参数与 T 无关，
                   滑窗 MLP 在 T=14 处被反超——权值共享是
                   "为长序列买单"
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

                结构已经通了：字符进、h 流动、softmax 出分布。
                但权重还是随机的——预测是垃圾。要让它学会
                "说人话"，就得回答 32 课悬着的问题：**怎么训练？**

                线索就藏在今天的展开图里：**展开图就是计算图**。
                把 T 步展开看成深度为 T 的前馈网络，22 课的
                链式法则就能原样上岗——只不过现在梯度要
                **沿时间往回传**：从损失出发，穿过第 T 步的
                h、第 T−1 步的 h、……一路传到第 1 步。

                这套算法有个正式名字：**BPTT**（Backpropagation
                Through Time，沿时间反向传播）。其中最妙的一笔：
                U 只有一份，但它**每个时间步都收一份梯度**——
                总梯度是 T 份贡献的和。
                """
            ),
            mo.callout(
                mo.md(
                    """
                    **下一课预告** — `34_BPTT`：

                    亲手把 RNN 的反向传播写出来：展开、逐段求导、
                    沿时间回传。三样收获：

                    - 看清梯度回传的**三条路径**（x 路、h 路、y 路），
                      以及为什么 h 路要**连乘** U 和 tanh′
                    - 用 22 课的数值梯度仪式对账：解析梯度 vs
                      数值梯度，误差 10⁻⁸ 级别才算过关
                    - 画出**各时间步对 U 的梯度贡献**条形图——
                      远处的时间步贡献如何衰减？（35 课的
                      灾难预告片）

                    对账通过的那一刻，你就拥有了完整的、可以
                    训练的 RNN——37 课的语言模型全靠它。

                    ⏪ 时间倒流，开始！
                    """
                ),
                kind="success",
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
