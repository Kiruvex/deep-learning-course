import marimo

__generated_with = "0.25.2"
app = marimo.App(app_title="23 · 反向传播下", width="medium")


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
        # 23 · 反向传播（下）：矩阵版与双月牙实战

        **从零到 Transformer · 手写深度学习** ｜ 第 4 部分 · 神经网络（6/8）
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📖 本课目标（先读我）": mo.md(
                """
                **这一课只解决一个问题：把标量发动机装上矩阵底盘，打一场真仗。**

                22 课的微型网络一次只吃一个数——真实网络一次要吃一批样本、
                每层几十上百个神经元。好消息：**骨架一毫米都不用改**，
                只要把"数"换成"块"，四条矩阵梯度公式用**形状检查法**
                就能推出来（梯度形状必须和参数形状一致——形状对不上，
                公式一定错）。

                战场也换了：告别温驯的两团点，请出经典的**双月牙**——
                两弯交错的月牙，任何一条直线都只能拿到八成准确率。
                两层 MLP + 手写反向传播，能不能把它拿下？

                打法依然稳扎细打：先把数据造好、三分天下（17 课纪律）；
                再推四条矩阵梯度并逐个过验钞机（13 课）；然后完整训练——
                损失曲线、决策边界的进化、验证集体检，一样不少；
                最后做隐单元容量的扫描，看容量旋钮（17 课）在神经网络
                上的样子。

                学完你将能：

                - 写出**批量前向**：X (n, d) 一把过，不写样本循环（01 课的教训）
                - 用**形状检查法**推出并记住四条矩阵梯度：
                  dW₂ = Hᵀδ₂、db₂ = Σδ₂、δ₁ = (δ₂W₂ᵀ)⊙(1−H²)、dW₁ = Xᵀδ₁
                - 对**整个矩阵反向传播**做数值对账（13 个参数，误差 10⁻⁹）
                - 从零训练一个 MLP **拿下双月牙**（train/val/test 全 100%），
                  并讲清隐单元数 h 怎么选

                前置：第 22 课的全部（铁律、δ 记号、对账流程）、
                第 20 课的维度账本与 X@W+b、第 17 课的三分天下与容量旋钮、
                第 12 课的矩阵转置（今天 ᵀ 满场飞）。

                这一课练完，"手写反向传播"就从知识点变成了你的肌肉记忆——
                24 课的优化器和 25 课的收官大战，全部站在今天的底盘上。
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📦 概念 1/4 · 从一个样本到一批：矩阵前向": mo.md(
                """
                22 课的前向是五个标量；把 n 个样本**按行堆进矩阵**（12 课
                行样本约定），同一套公式自动升级：

                ```
                Z₁ = X @ W₁ + b₁        (n, d) @ (d, h) + (h,) → (n, h)
                H  = tanh(Z₁)            (n, h)（激活不改形状，19 课）
                Z₂ = H @ W₂ + b₂        (n, h) @ (h, 1) + (1,) → (n, 1)
                p  = σ(Z₂)               (n, 1)
                L  = 平均交叉熵           标量
                ```

                三个升级点，个个都是 01 课"循环慢"的教训兑现：

                - **一批一把过**：72 个样本和 1 个样本的代码长度相同——
                  矩阵乘法在底层是打包好的连续内存运算
                - **损失取平均**：每个样本的交叉熵加起来除以 n
                  （`np.mean`）。除不除 n 只影响梯度的整体尺度
                  （相当于 lr 换算），但**不除的话 batch 越大梯度越大**，
                  学习率就得跟着 batch 走——取平均让 lr 和 batch 解耦，
                  工程上省心
                - **缓存照旧**：Z₁、H、p 全部存着——反向传播每一环
                  都要用（22 课的习惯）

                偏置的广播（06 课）在这里是隐形功臣：b₁ (h,) 加到
                (n, h) 的每一行，n 个样本共享同一套参数——
                "**参数共享**"正是"模型"二字的本义。
                """
            ),
            "🧮 概念 2/4 · 四条矩阵梯度：形状检查法推导": mo.md(
                """
                反向传播的矩阵版怎么推？**不硬推，先验形状**——
                一条铁律：**梯度的形状必须和参数的形状一模一样**
                （更新时要做 `W -= lr * dW`，形状不同根本减不了）。

                从输出层往回走（δ₂ 是 22 课的老朋友，加了 1/n）：

                ```
                δ₂ = (p − y) / n              (n, 1)   ← 21 课 p−y，除 n 因为损失取了平均
                dW₂ = Hᵀ @ δ₂                 (h, 1)   ← 要和 W₂ (h,1) 同形：Hᵀ (h,n) @ δ₂ (n,1) ✓
                db₂ = δ₂ 沿样本求和            (1,)     ← 加法铁律：每个样本的 δ₂ 都发给 b₂ 一份
                δ₁ = (δ₂ @ W₂ᵀ) ⊙ (1 − H²)    (n, h)   ← 乘法交换（δ₂ × w₂）+ tanh'
                dW₁ = Xᵀ @ δ₁                 (d, h)   ← 要和 W₁ (d,h) 同形：Xᵀ (d,n) @ δ₁ (n,h) ✓
                db₁ = δ₁ 沿样本求和            (h,)     ← 又是加法铁律
                ```

                两个新记号在此上岗：

                - **ᵀ（转置，12 课）**：乘法铁律的矩阵版。
                  标量世界 dL/dh = δ₂·w₂；矩阵世界 h (n,h) 被 W₂ (h,1)
                  变成 z₂ (n,1)，反向就是 δ₂ (n,1) 乘 W₂ 的"倒影"
                  W₂ᵀ (1,h)——**形状对了，故事就对**
                - **⊙（逐元素乘，Hadamard）**：tanh' = 1−H² 是
                  (n, h) 的逐元素坡度，只能和同形状的 δ₂@W₂ᵀ 逐点相乘
                  ——矩阵乘法 @ 是"跨维度握手"，⊙ 是"同形状贴脸"

                求和用哪个 axis？**消掉哪个维度就对哪个 axis 求和**
                （08 课的判据）：δ₂ (n,1) → b₂ (1,)，消 n 就 `sum(axis=0)`。
                """
            ),
            "🌙 概念 3/4 · 双月牙：一块难啃但啃得动的战场": mo.md(
                """
                两团高斯点（16、18 课）太温驯了——一条直线就打发。
                双月牙（two moons）是经典的中级 Boss：两弯互相咬合的
                月牙，任何**直线**切下去最多对八成出头。

                造法其实很几何（不用任何库，自己写三行）：

                ```
                月牙 A：上半圆 (cos t, sin t)，t ∈ (0, π)
                月牙 B：下半弧 (1 − cos t, 0.5 − sin t)
                两边各撒一点高斯噪声
                ```

                月牙 A 从 (1, 0) 爬过 (0, 1) 落到 (−1, 0)；月牙 B 从
                (0, 0.5) 沉过 (1, −0.5) 落到 (2, 0.5)——一上一下互相咬合，
                边界必须是**弯的**。这正是 MLP 的主场：19 课说过
                ReLU 拼折线，tanh 同理拼平滑的弯——8 个隐单元足够
                勾勒这条 S 形分界。

                纪律照旧（17 课）：120 个点洗牌后切成 **72 / 24 / 24**
                （上课 / 模考 / 高考）。线性基线先量一下：最小二乘
                拟合标签，训练集 **81.9%**——这就是"直线的天花板"，
                MLP 的成绩单要和这个数比。
                """
            ),
            "🏋️ 概念 4/4 · 完整训练流程与容量扫描": mo.md(
                """
                所有零件就位，组装完整流程（15 课四步曲的完全体）：

                ```
                ① 初始化：W 小随机、b 全零（为什么不能全零？25 课的
                   "对称性灾难"专门讲；今天先用 sqrt(2/d) 尺度的小随机）
                重复 500 个 epoch（全批）：
                    ② 前向：Z₁ → H → Z₂ → p → L（存中间量）
                    ③ 反向：δ₂ → 四条矩阵梯度
                    ④ 更新：四个参数各自 -= lr × 梯度
                    ⑤ 记录：训练/验证损失与准确率，画曲线
                ```

                **体检三件套**（17 课的规矩）：损失曲线要看**平滑下降**；
                决策边界要看**从乱线进化成月牙形状**；验证集全程盯着——
                它若抬头，训练就该刹车（早停，25 课细讲）。

                最后是**容量扫描**：隐单元数 h 从 1 拧到 64。
                预告一下剧情（演示 5 放全景图）：

                | h | 1 | 2 | 4 | 8+ |
                |---|---|---|---|---|
                | train acc | 81.9% | 88.9% | 100% | 100% |
                | 长相 | 一条直线（= 线性基线！） | 临界挣扎 | 月牙成型 | 月牙成型 |

                h = 1 的 MLP **退化回线性模型**——一个 tanh 单元拼不出弯；
                h = 2 在容量临界点上挣扎；h = 4 起稳定拿下。
                这就是 17 课"容量旋钮"在神经网络上的刻度：**h 就是容量**。
                """
            ),
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 🖥️ 代码演示

        五组演示，从公式到战场：

        - **演示 1** · 造双月牙 + 三分天下 + 线性基线量天花板
        - **演示 2** · 批量前向：形状账本逐行对账
        - **演示 3** · 四条矩阵梯度的验钞大会（13 个参数，误差 10⁻⁹）
        - **演示 4** · 完整训练：损失曲线 + 决策边界进化 + 三份成绩单
        - **演示 5** · 容量扫描：h = 1 到 64，从直线到月牙
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 1 · 双月牙：造数据、三分天下、量天花板

        `make_moons` 手写版（概念 3 的两行参数方程），种子 11、
        每弯 60 个点、噪声 0.10。洗牌后切 72 / 24 / 24（17 课的
        permutation 三连）。

        然后请出**线性基线**：用 17 课的最小二乘（solve）直接拟合
        0/1 标签——它给出的边界就是"一条直线能做的最好"。
        训练集 **81.9%**：月牙互相咬合的部分，直线怎么摆都要错。
                这是 MLP 要超越的及格线。
        """
    )
    return


@app.cell
def _(np, plt):
    # 演示 1 · 双月牙：手写生成器 + 三分天下 + 线性基线
    def make_moons(n_per=60, noise=0.10, seed=11):
        """两弯互相咬合的月牙。返回 (X, y)：X (2n, 2)，y (2n,) 取 0/1"""
        _rng = np.random.default_rng(seed)
        _t = _rng.uniform(0, np.pi, n_per)             # 参数角：上半圆扫一遍
        _X0 = np.column_stack([np.cos(_t), np.sin(_t)]) \
            + _rng.normal(0, noise, (n_per, 2))        # 月牙 A：上半圆 + 噪声
        _X1 = np.column_stack([1 - np.cos(_t), 0.5 - np.sin(_t)]) \
            + _rng.normal(0, noise, (n_per, 2))        # 月牙 B：下沉的下半弧
        _X = np.vstack([_X0, _X1])
        _y = np.concatenate([np.zeros(n_per), np.ones(n_per)])
        _idx = _rng.permutation(len(_y))               # 先洗牌再切（17 课铁律）
        return _X[_idx], _y[_idx]

    X_all, y_all = make_moons(60, 0.10, seed=11)       # 120 个点
    X_tr, y_tr = X_all[:72], y_all[:72]                # 上课（72）
    X_va, y_va = X_all[72:96], y_all[72:96]            # 模考（24）
    X_te, y_te = X_all[96:], y_all[96:]                # 高考（24，先封存）

    # ---- 线性基线：最小二乘拟合 0/1 标签（17 课 solve 回收）----
    _A = np.column_stack([X_tr, np.ones(len(y_tr))])   # 加一列 1 当截距
    _w_lin = np.linalg.solve(_A.T @ _A + 1e-6 * np.eye(3), _A.T @ y_tr)
    _acc_lin = float(np.mean(((_A @ _w_lin) > 0.5) == y_tr))   # 直线的天花板

    # ---- 画图 ----
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for _ax, (_Xs, _ys, _ttl) in [
        (_axes[0], (X_tr, y_tr, "训练集（上课，72 点）")),
        (_axes[1], (X_all, y_all, "全景：两弯月牙互相咬合")),
    ]:
        _ax.scatter(_Xs[_ys == 0, 0], _Xs[_ys == 0, 1], s=26, color="tab:blue", label="月牙 A（0 类）")
        _ax.scatter(_Xs[_ys == 1, 0], _Xs[_ys == 1, 1], s=26, color="tab:red", marker="s", label="月牙 B（1 类）")
        _ax.set_title(_ttl)
        _ax.set_xlabel("x1")
        _ax.set_ylabel("x2")
        _ax.legend(fontsize=9, loc="upper right")
        _ax.grid(True, alpha=0.3)

    # 基线边界画在右图：w·x + b = 0
    _xx = np.linspace(-1.6, 2.4, 50)
    _yy = (-_w_lin[2] - _w_lin[0] * _xx) / _w_lin[1]
    _m = (_yy > -1.2) & (_yy < 1.5)
    _axes[1].plot(_xx[_m], _yy[_m], color="tab:green", lw=2.2,
                  label=f"线性基线（train {_acc_lin:.1%}）")
    _axes[1].legend(fontsize=9, loc="upper right")
    _fig.tight_layout()
    _fig
    return (X_te, X_tr, X_va, make_moons, y_te, y_tr, y_va)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 2 · 批量前向：形状账本

        5 个样本的小批（n=5）、2 维输入、3 个隐单元——每个矩阵都小到
        能肉眼盯。前向四步，形状逐行对账（试着自己抢答再看书）：

        | 步骤 | 形状 | 说明 |
        |---|---|---|
        | X | (5, 2) | 5 个样本 × 2 个特征 |
        | W₁, b₁ | (2, 3), (3,) | 2 进 3 出；b₁ 广播到每行（06 课） |
        | Z₁ = X@W₁+b₁ | (5, 3) | 5 个样本各得 3 个打分 |
        | H = tanh(Z₁) | (5, 3) | 形状不动，数值过弯 |
        | W₂, b₂ | (3, 1), (1,) | 3 进 1 出 |
        | Z₂ = H@W₂+b₂ | (5, 1) | 每个样本一个最终打分 |
        | p = σ(Z₂) | (5, 1) | 每个样本一个概率 |

        注意 `init_params` 里初始化的写法：`rng.normal(0,1,(d,h)) * sqrt(2/d)`
        ——**小随机 × 尺度因子**。为什么不能全零、尺度怎么定，
        25 课的"初始化"专章回答；今天先当作"出厂设置"用。
        """
    )
    return


@app.cell
def _(np):
    # 演示 2 · 批量前向：init_params / forward / loss_f（全课共用，先包好）
    def init_params(d, h, seed=0):
        """小随机初始化。尺度 sqrt(2/进维) 的来历 25 课讲，先当出厂设置"""
        _rng = np.random.default_rng(seed)
        _W1 = _rng.normal(0, 1, (d, h)) * np.sqrt(2 / d)     # (d, h)
        _b1 = np.zeros(h)                                     # (h,)
        _W2 = _rng.normal(0, 1, (h, 1)) * np.sqrt(2 / h)      # (h, 1)
        _b2 = np.zeros(1)                                     # (1,)
        return _W1, _b1, _W2, _b2

    def forward(X, W1, b1, W2, b2):
        """批量前向。返回 (Z1, H, Z2, p)——中间量全存（22 课的缓存习惯）"""
        Z1 = X @ W1 + b1                             # (n, h)
        H = np.tanh(Z1)                              # (n, h)
        Z2 = H @ W2 + b2                             # (n, 1)
        p = 1 / (1 + np.exp(-Z2))                    # (n, 1)（16 课 sigmoid）
        return Z1, H, Z2, p

    def loss_f(p, y):
        """平均二元交叉熵（21 课）：每个样本的 CE 加起来除以 n"""
        _yc = y.reshape(-1, 1)                       # (n,) → (n, 1) 对齐形状
        return float(-np.mean(_yc * np.log(p + 1e-12) + (1 - _yc) * np.log(1 - p + 1e-12)))

    # ---- 小批演示：形状账本 ----
    _rng = np.random.default_rng(3)
    Xs = _rng.normal(0, 1, (5, 2))                   # 5 个样本
    ys = _rng.integers(0, 2, 5).astype(float)        # 随手 0/1 标签
    W1s, b1s, W2s, b2s = init_params(2, 3, seed=7)   # 2 → 3 → 1 的小网
    Z1s, Hs, Z2s, ps = forward(Xs, W1s, b1s, W2s, b2s)
    Ls = loss_f(ps, ys)
    return (Ls, Xs, Z1s, Z2s, b1s, b2s, forward, Hs, W1s, W2s, init_params, loss_f, ps, ys)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 3 · 验钞大会：四条矩阵梯度逐一过账

        `backward` 一共十行，四条公式（概念 2）加返回。验钞对象：
        刚才那个 5-2-3-1 的小网，**13 个参数**（W₁ 六个 + b₁ 三个 +
        W₂ 三个 + b₂ 一个）逐个拧 ±10⁻⁷。

        对账结果：**最大误差 1.5×10⁻⁹**——四条公式，全部验明正身。
        特别看两处容易翻车的地方：

        - `delta2` 里**除了 n**：损失取了平均（np.mean），
          梯度就得除回来——忘了除，梯度大 5 倍，lr 等于偷偷变大
        - `(delta2 @ W2.T)` 的 **@ 和 .T**：换成 `*` 或忘了转置，
          形状当场报错——形状检查法不是摆设，是第一道防线
        """
    )
    return


@app.cell
def _(Ls, Xs, Z1s, b1s, b2s, forward, Hs, W1s, W2s, loss_f, mo, np, ps, ys):
    # 演示 3 · 矩阵版反向传播 + 13 参数验钞
    def backward(X, y, Z1, H, p, W1, W2):
        """矩阵版反向传播。返回 (dW1, db1, dW2, db2)，形状与参数一一相同"""
        _n = len(X)
        _yc = y.reshape(-1, 1)                       # (n, 1)
        delta2 = (p - _yc) / _n                      # (n, 1)：21 课 p−y，除 n
        dW2 = H.T @ delta2                           # (h, 1)：Hᵀ (h,n) @ (n,1)
        db2 = delta2.sum(axis=0)                     # (1,)：消 n，对 axis=0 求和（08 课）
        delta1 = (delta2 @ W2.T) * (1 - H ** 2)      # (n, h)：乘法交换 + tanh'
        dW1 = X.T @ delta1                           # (d, h)：Xᵀ (d,n) @ (n,h)
        db1 = delta1.sum(axis=0)                     # (h,)：又是加法铁律
        return dW1, db1, dW2, db2

    # ---- 13 个参数逐个对称差分 ----
    _eps = 1e-7
    _errs = []
    _bp = backward(Xs, ys, Z1s, Hs, ps, W1s, W2s)    # 反向传播的结果

    def _loss_pert(kind, idx, e):
        _W1, _b1, _W2, _b2 = W1s.copy(), b1s.copy(), W2s.copy(), b2s.copy()
        if kind == "W1":
            _W1[idx] += e
        elif kind == "b1":
            _b1[idx] += e
        elif kind == "W2":
            _W2[idx] += e
        else:
            _b2[idx] += e
        return loss_f(forward(Xs, _W1, _b1, _W2, _b2)[3], ys)

    for _i in range(2):
        for _j in range(3):
            _num = (_loss_pert("W1", (_i, _j), _eps) - _loss_pert("W1", (_i, _j), -_eps)) / (2 * _eps)
            _errs.append(abs(_num - _bp[0][_i, _j]))
    for _j in range(3):
        _num = (_loss_pert("b1", _j, _eps) - _loss_pert("b1", _j, -_eps)) / (2 * _eps)
        _errs.append(abs(_num - _bp[1][_j]))
    for _i in range(3):
        _num = (_loss_pert("W2", (_i, 0), _eps) - _loss_pert("W2", (_i, 0), -_eps)) / (2 * _eps)
        _errs.append(abs(_num - _bp[2][_i, 0]))
    _errs.append(abs((_loss_pert("b2", 0, _eps) - _loss_pert("b2", 0, -_eps)) / (2 * _eps) - _bp[3][0]))

    mo.md(
        f"""
        | 检查项 | 值 |
        |---|---|
        | 前向损失 L | **{Ls:.4f}** |
        | 梯度形状 | dW₁ {_bp[0].shape}、db₁ {_bp[1].shape}、dW₂ {_bp[2].shape}、db₂ {_bp[3].shape} |
        | 参数形状 | W₁ {W1s.shape}、b₁ {b1s.shape}、W₂ {W2s.shape}、b₂ {b2s.shape}（一一对应 ✓） |
        | 验钞参数个数 | **13** 个 |
        | 最大误差 | **{max(_errs):.1e}** ✅ |
        """
    )
    return (backward,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 4 · 完整训练：损失曲线、边界进化、三份成绩单

        组装完全体（概念 4 的流程）：`train_mlp` = 初始化 + 四步曲 × 500
        epoch（全批）+ 双记录（训练/验证）。上战场：h = 8、lr = 1.0。

        左图损失曲线：0.701 起步（≈ ln 2，瞎猜线——16 课的起点在
        矩阵版重现），500 步后压到 **0.013**。右图是决策边界的
        进化四连拍：第 1 epoch 一条乱线 → 第 25 epoch 摆正方向 →
        第 100 epoch 弯出月牙雏形 → 第 500 epoch 精准贴合两弯月牙
        之间的 S 形河谷。
        """
    )
    return


@app.cell
def _(X_tr, X_va, backward, forward, init_params, loss_f, np, plt, y_tr, y_va):
    # 演示 4 · train_mlp：完整训练循环（交互实验和练习还要用，先包好）
    def train_mlp(X_tr, y_tr, X_va, y_va, h=8, lr=1.0, epochs=500, seed=0):
        """两层 MLP 完整训练。返回 (W1, b1, W2, b2, hist)"""
        W1, b1, W2, b2 = init_params(X_tr.shape[1], h, seed)
        hist = {"L": [], "acc_tr": [], "acc_va": []}
        for _ep in range(epochs):
            # ② 前向（存中间量）
            Z1, H, Z2, p = forward(X_tr, W1, b1, W2, b2)
            # ③ 反向（四条矩阵梯度）
            dW1, db1, dW2, db2 = backward(X_tr, y_tr, Z1, H, p, W1, W2)
            # ④ 更新
            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2
            # ⑤ 记录（训练 + 验证，17 课的体检表）
            _pv = forward(X_va, W1, b1, W2, b2)[3]
            hist["L"].append(loss_f(p, y_tr))
            hist["acc_tr"].append(float(((p > 0.5) == y_tr.reshape(-1, 1)).mean()))
            hist["acc_va"].append(float(((_pv > 0.5) == y_va.reshape(-1, 1)).mean()))
        return W1, b1, W2, b2, hist

    # ---- 决策地图的通用画法（20 课 draw_map 的 MLP 版）----
    def mlp_map(ax, W1, b1, W2, b2, X=None, y=None, title="", lim=(-1.6, 2.4), n=220):
        """画 MLP 的打分地图：红蓝着色 + p=0.5 黑线 + 可选数据点"""
        _gx = np.linspace(lim[0], lim[1], n)
        _gy = np.linspace(-1.2, 1.6, n)
        _GX, _GY = np.meshgrid(_gx, _gy)
        _G = np.column_stack([_GX.ravel(), _GY.ravel()])
        _Z = forward(_G, W1, b1, W2, b2)[3].reshape(_GX.shape)
        ax.contourf(_GX, _GY, _Z, levels=25, cmap="RdBu", alpha=0.55)
        ax.contour(_GX, _GY, _Z, levels=[0.5], colors="k", linewidths=2)
        if X is not None:
            ax.scatter(X[y == 0, 0], X[y == 0, 1], s=22, color="tab:blue",
                       edgecolors="white", linewidths=0.6, zorder=3)
            ax.scatter(X[y == 1, 0], X[y == 1, 1], s=22, color="tab:red",
                       marker="s", edgecolors="white", linewidths=0.6, zorder=3)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")

    # ---- 主配置训练 ----
    W1f, b1f, W2f, b2f, hist = train_mlp(X_tr, y_tr, X_va, y_va, h=8, lr=1.0, epochs=500)

    # ---- 边界进化快照：重放到指定 epoch ----
    _snap_ep = [1, 25, 100, 500]
    _snaps = []
    _W1, _b1, _W2, _b2 = init_params(2, 8, 0)
    for _ep in range(500):
        _Z1, _H, _Z2, _p = forward(X_tr, _W1, _b1, _W2, _b2)
        _g = backward(X_tr, y_tr, _Z1, _H, _p, _W1, _W2)
        _W1 -= 1.0 * _g[0]; _b1 -= 1.0 * _g[1]; _W2 -= 1.0 * _g[2]; _b2 -= 1.0 * _g[3]
        if _ep + 1 in _snap_ep:
            _snaps.append((_W1.copy(), _b1.copy(), _W2.copy(), _b2.copy()))

    _fig = plt.figure(figsize=(12.8, 7.6))
    _gs = _fig.add_gridspec(2, 4, height_ratios=[1, 1.15])
    _axL = _fig.add_subplot(_gs[0, :])
    _axL.plot(range(1, 501), hist["L"], color="tab:blue", lw=2.2)
    _axL.axhline(np.log(2), color="gray", ls=":", lw=1.1, label="瞎猜线 ln 2 = 0.693")
    for _e in _snap_ep:
        _axL.scatter([_e], [hist["L"][_e - 1]], s=60, color="tab:red", zorder=4)
    _axL.set_title(f"损失曲线：0.701 → {hist['L'][-1]:.4f}（红点 = 下排四连拍的时刻）")
    _axL.set_xlabel("epoch")
    _axL.set_ylabel("训练损失")
    _axL.legend(fontsize=9)
    _axL.grid(True, alpha=0.3)
    for _k, (_e, (_Ws1, _bs1, _Ws2, _bs2)) in enumerate(zip(_snap_ep, _snaps)):
        _axm = _fig.add_subplot(_gs[1, _k])
        _acc = hist["acc_tr"][_e - 1]
        mlp_map(_axm, _Ws1, _bs1, _Ws2, _bs2, X=X_tr, y=y_tr,
                title=f"epoch {_e}（train acc {_acc:.1%}）")
    _fig.tight_layout()
    _fig
    return (W1f, b1f, W2f, b2f, hist, mlp_map, train_mlp)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        f"""
        三份成绩单（h = 8、lr = 1.0、500 epoch）：

        | 卷子 | 准确率 |
        |---|---|
        | 训练集（上课） | 55.6% → **100%**（第 250 epoch 达成） |
        | 验证集（模考） | **100%**（全程无抬头，不用刹车） |
        | 测试集（高考，演示 5 后拆封） | 见演示 5 |

        对照线性基线的 **81.9%**：MLP 把咬合的部分全啃下来了。
        22 课那个"参数自己学会"的瞬间，在这里变成了一整张月牙地图——
        没有手拍权重、没有捷径，四条矩阵梯度推着 **33 个参数**
        （2×8 + 8 + 8×1 + 1，20 课的数法）自己走到位。
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 5 · 容量扫描：h 从 1 到 64

        同一份数据、同一个训练流程，只拧隐单元数 h（17 课容量旋钮的
        神经网络刻度）。四连拍 + 全表：

        | h | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
        |---|---|---|---|---|---|---|---|
        | train acc | **81.9%** | 88.9% | 100% | 100% | 100% | 100% | 100% |
        | val acc | 95.8% | 79.2% | 100% | 100% | 100% | 100% | 100% |

        三个看点：

        - **h = 1 是"伪 MLP"**：一个 tanh 单元拼不出弯，训练准确率
          81.9%——**和线性基线分毫不差**。容量不够时，深度也只是摆设
        - **h = 2 在临界点挣扎**：容量刚够弯一次，训练 88.9%、
          验证 79.2%——边界扭出来了但扭得不到位
        - **h ≥ 4 稳定拿下**：月牙成型，train/val 双 100%。
          再往上加（16、32、64）这份干净数据上没有变差——
          **U 型的右半边没有现身**

        最后一个诚实观察：右半边没现身，是因为月牙**干净**（噪声小、
        点够多）。把噪声调大、训练点砍半，过拟合就会回来——
        17 课的 U 型曲线在神经网络上的完整版，25 课用 Dropout
        和早停正式接手。现在，先拆封高考卷：h = 8 的模型在
        **24 个从未见过的测试点上，准确率 100%**。
        """
    )
    return


@app.cell
def _(X_tr, X_va, mlp_map, mo, np, plt, train_mlp, y_tr, y_va):
    # 演示 5 · 容量扫描：h = 1 / 2 / 4 / 16 四连拍（成绩实时算）
    _hs_show = [1, 2, 4, 16]
    _accs = {}
    for _h in [1, 2, 4, 8, 16, 32, 64]:
        _W1, _b1, _W2, _b2, _hh = train_mlp(X_tr, y_tr, X_va, y_va, h=_h, lr=1.0, epochs=500)
        _accs[_h] = (_hh["acc_tr"][-1], _hh["acc_va"][-1])

    _fig, _axes = plt.subplots(1, 4, figsize=(13.6, 3.6), sharey=True)
    for _ax, _h in zip(_axes, _hs_show):
        _W1h, _b1h, _W2h, _b2h, _hh = train_mlp(X_tr, y_tr, X_va, y_va, h=_h, lr=1.0, epochs=500)
        mlp_map(_ax, _W1h, _b1h, _W2h, _b2h, X=X_tr, y=y_tr,
                title=f"h = {_h}：train {(_hh['acc_tr'][-1]):.0%} / val {(_hh['acc_va'][-1]):.0%}")
    _fig.tight_layout()

    _tbl = " | ".join(f"{_h}" for _h in _accs)
    _tr_row = " | ".join(f"{_v[0]:.0%}" for _v in _accs.values())
    _va_row = " | ".join(f"{_v[1]:.0%}" for _v in _accs.values())
    mo.vstack([
        _fig,
        mo.md(
            f"""
            | h | {_tbl} |
            |---|---|---|---|---|---|---|---|
            | train | {_tr_row} |
            | val | {_va_row} |
            """
        ),
    ])
    return


@app.cell
def _(W1f, X_te, b1f, b2f, forward, mo, np, W2f, y_te):
    # 演示 5 续 · 高考揭榜（测试集只算这一次，17 课的纪律）
    _p_te = forward(X_te, W1f, b1f, W2f, b2f)[3]
    _acc_te = float(((_p_te > 0.5) == y_te.reshape(-1, 1)).mean())
    mo.md(
        f"""
        | 高考揭榜 | 值 |
        |---|---|
        | 测试集准确率（h = 8 主配置） | **{_acc_te:.0%}** |
        | 对照：线性基线 train | 81.9% |
        | 用的参数 | 33 个（2×8 + 8 + 8×1 + 1，20 课的数法） |
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 🎛️ 交互实验

        **调参工作台**——两个旋钮同时拧：隐单元数 h（1~12，容量）和
        学习率 lr（0.2~2.0，步长）。松手后现场训练 500 epoch，
        决策地图和三份读数立刻刷新。

        玩法建议：

        1. **h 拧到 1**：地图退化成一条直线——不管 lr 怎么调，
           train 卡在 81.9% 附近（容量不足，努力无效）
        2. **h = 2、lr = 2.0**：临界容量的挣扎最剧烈——边界扭来扭去
           就是贴不住月牙
        3. **h = 3~4、lr = 1.0**：最省料的"刚好"配置，
           月牙一次成型
        4. 顺手体会：**lr 太小（0.2）时 500 epoch 不够用**——
           容量够、步子不够，一样到不了终点（15 课的"磨蹭"）
        """
    )
    return


@app.cell
def _(mo):
    # 交互控制：容量 + 步长（debounce：松手才重算）
    h_slider = mo.ui.slider(1, 12, value=3, step=1, debounce=True,
                            label="🎛️ 隐单元数 h（容量旋钮，1 ~ 12）")
    lr23_slider = mo.ui.slider(0.2, 2.0, value=1.0, step=0.1, debounce=True,
                               label="🎛️ 学习率 lr（步长，0.2 ~ 2.0）")
    mo.vstack([h_slider, lr23_slider])
    return (h_slider, lr23_slider)


@app.cell
def _(X_tr, X_va, h_slider, lr23_slider, mlp_map, mo, plt, train_mlp, y_tr, y_va):
    # ===== 交互实验 · 调参工作台（响应单元）=====
    _h = h_slider.value
    _lr = lr23_slider.value
    _W1, _b1, _W2, _b2, _hist = train_mlp(X_tr, y_tr, X_va, y_va,
                                          h=_h, lr=_lr, epochs=500)
    _tr = _hist["acc_tr"][-1]
    _va = _hist["acc_va"][-1]
    _L = _hist["L"][-1]

    _fig, _ax = plt.subplots(figsize=(7.0, 4.8))
    mlp_map(_ax, _W1, _b1, _W2, _b2, X=X_tr, y=y_tr,
            title=f"h = {_h}，lr = {_lr:.1f}")

    # ---- 判词：现在站在容量/步长的哪个位置 ----
    if _tr < 0.90:
        _verdict = (f"🥶 欠拟合：train {_tr:.0%}（贴着线性基线 81.9%）——"
                    "容量不足，先加 h，调 lr 没用")
    elif _tr < 0.99:
        _verdict = f"⚠️ 接近：train {_tr:.0%} / val {_va:.0%}——容量刚够弯一次，再加 1~2 个隐单元试试"
    elif _va < 0.99:
        _verdict = f"👻 过拟合苗头：train {_tr:.0%} 但 val 只有 {_va:.0%}——容量略肥（25 课的正则化接手）"
    else:
        _verdict = f"✅ 学会了：train {_tr:.0%} / val {_va:.0%}，损失 {_L:.4f}——月牙成型"

    mo.vstack([
        _fig,
        mo.md(
            f"""
            | 读数 | 值 |
            |---|---|
            | 隐单元数 h | **{_h}**（参数量 {2*_h + _h + _h + 1} 个） |
            | 学习率 lr | **{_lr:.1f}** |
            | 训练准确率 | **{_tr:.1%}** |
            | 验证准确率 | **{_va:.1%}** |
            | 最终损失 | **{_L:.4f}**（起点都是 0.701） |
            """
        ),
        mo.callout(mo.md(f"**实时判词：{_verdict}**"), kind="info"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✏️ 动手练习（每课必做，本课 4 题）": mo.md(
                """
                **练习 1 · 补全矩阵前向**：`forward_ex1` 的②③两行
                （tanh 和第二层打分）——用形状账本自查。

                **练习 2 · 补全输出层梯度**：`backward_out_ex2` 的三行
                （δ₂、dW₂、db₂）——验钞对象是演示 3 的小网。

                **练习 3 · 补全隐层梯度（最难）**：`backward_hid_ex3` 的三行
                （δ₁、dW₁、db₁）——转置、⊙、axis，全都在这三行里。

                **练习 4 · 挑战 · 双臂螺旋**：比月牙更拧巴的数据。
                用 `train_mlp` 调出一套配置，让验证集准确率 ≥ 85%——
                真正的超参数调参实战（17 课的模板）。

                规则不变：先把 `None` 换成你的答案再运行；写完再看参考答案。
                """
            )
        }
    )
    return


@app.cell
def _(Xs, b1s, b2s, forward, mo, np, ps, W1s, W2s):
    # ===== 练习 1 · 补全矩阵前向 =====
    def forward_ex1(X, W1, b1, W2, b2):
        """批量前向。返回 (Z1, H, Z2, p)"""
        Z1 = X @ W1 + b1          # ① 第一层打分（不变）
        H = None                  # ← ② 补：np.tanh(Z1)
        Z2 = None                 # ← ③ 补：H @ W2 + b2
        p = 1 / (1 + np.exp(-Z2)) # ④ sigmoid（不变）
        return Z1, H, Z2, p

    fwd_ready_ex1 = None          # ← ⑤ 补完后，把这行换成 forward_ex1

    if fwd_ready_ex1 is not None:
        try:
            _out = fwd_ready_ex1(Xs, W1s, b1s, W2s, b2s)
            _ref = forward(Xs, W1s, b1s, W2s, b2s)
            if all(np.allclose(_a, _b) for _a, _b in zip(_out, _ref)):
                _msg = ("✅ 四个输出与演示 2 的 forward 完全一致——"
                        "矩阵前向，落袋为安")
            else:
                _msg = "❌ 有输出对不上——检查②是 tanh、③是 H @ W2 + b2"
        except TypeError:
            _msg = "⏳ H 或 Z2 还是 None——②③ 两行补了吗？"
    else:
        _msg = "⏳ 先补 ②③，再把 fwd_ready_ex1 换成 forward_ex1"
    mo.md(_msg)
    return (forward_ex1, fwd_ready_ex1)


@app.cell
def _(Hs, W2s, backward, Xs, b1s, b2s, forward, mo, np, ps, W1s, ys):
    # ===== 练习 2 · 补全输出层梯度 =====
    def backward_out_ex2(H, p, y, n):
        """输出层三件套。返回 (delta2, dW2, db2)"""
        _yc = y.reshape(-1, 1)
        delta2 = None    # ← ① 补：(p - _yc) / n
        dW2 = None       # ← ② 补：H.T @ delta2
        db2 = None       # ← ③ 补：delta2.sum(axis=0)
        return delta2, dW2, db2

    out_ready_ex2 = None    # ← ④ 补完后，把这行换成 backward_out_ex2

    if out_ready_ex2 is not None:
        try:
            _d2, _dw2, _db2 = out_ready_ex2(Hs, ps, ys, len(Xs))
            _ref = backward(Xs, ys, forward(Xs, W1s, b1s, W2s, b2s)[0], Hs, ps, W1s, W2s)
            if (np.allclose(_d2, (ps - ys.reshape(-1, 1)) / len(Xs))
                    and np.allclose(_dw2, _ref[2]) and np.allclose(_db2, _ref[3])):
                _msg = (f"δ₂ 形状 {_d2.shape}、dW₂ 形状 {_dw2.shape}——"
                        "✅ 三件套与验钞过的 backward 一致")
            else:
                _msg = "❌ 对不上——①别忘了除 n；②是 H 的转置；③axis=0"
        except TypeError:
            _msg = "⏳ 有 None 没补——三行逐个来"
    else:
        _msg = "⏳ 先补三行，再把 out_ready_ex2 换成 backward_out_ex2"
    mo.md(_msg)
    return (out_ready_ex2,)


@app.cell
def _(Hs, W2s, Xs, backward, b1s, b2s, forward, mo, np, ps, W1s, ys):
    # ===== 练习 3 · 补全隐层梯度（最难） =====
    def backward_hid_ex3(X, delta2, W2, H):
        """隐层三件套。返回 (delta1, dW1, db1)"""
        delta1 = None    # ← ① 补：(delta2 @ W2.T) * (1 - H**2)
        dW1 = None       # ← ② 补：X.T @ delta1
        db1 = None       # ← ③ 补：delta1.sum(axis=0)
        return delta1, dW1, db1

    hid_ready_ex3 = None    # ← ④ 补完后，把这行换成 backward_hid_ex3

    if hid_ready_ex3 is not None:
        try:
            _d2 = (ps - ys.reshape(-1, 1)) / len(Xs)         # 练习 2 的 δ₂
            _d1, _dw1, _db1 = hid_ready_ex3(Xs, _d2, W2s, Hs)
            _ref = backward(Xs, ys, forward(Xs, W1s, b1s, W2s, b2s)[0], Hs, ps, W1s, W2s)
            if (np.allclose(_d1, (ps - ys.reshape(-1, 1)) / len(Xs) @ W2s.T * (1 - Hs**2))
                    and np.allclose(_dw1, _ref[0]) and np.allclose(_db1, _ref[1])):
                _msg = (f"δ₁ 形状 {_d1.shape}、dW₁ 形状 {_dw1.shape}——"
                        "✅ 隐层梯度贯通，整套反向传播你全写完了")
            else:
                _msg = "❌ 对不上——①注意 @ 和 ⊙ 的顺序（先 δ₂@W₂ᵀ 再乘 1−H²）；②X 的转置；③axis=0"
        except TypeError:
            _msg = "⏳ 有 None 没补——①②③ 逐行来"
    else:
        _msg = "⏳ 先补三行，再把 hid_ready_ex3 换成 backward_hid_ex3"
    mo.md(_msg)
    return (hid_ready_ex3,)


@app.cell
def _(mo, np, train_mlp):
    # ===== 练习 4 · 挑战：双臂螺旋 =====
    def make_spiral(n_per=50, noise=0.06, seed=5, wind=2.5):
        """两条互相缠绕的螺旋臂。返回 (X, y)"""
        _rng = np.random.default_rng(seed)
        _Xs_, _ys_ = [], []
        for _arm in range(2):
            _t = np.linspace(0, 1, n_per)                    # 0 → 1 扫一圈臂
            _r = 3 * _t                                      # 半径越转越大
            _th = wind * np.pi * _t + _arm * np.pi           # 角度：两臂错开 π
            _pts = np.column_stack([_r * np.cos(_th), _r * np.sin(_th)]) \
                + _rng.normal(0, noise, (n_per, 2))          # 撒噪声
            _Xs_.append(_pts)
            _ys_.append(np.full(n_per, _arm))
        _X = np.vstack(_Xs_)
        _y = np.concatenate(_ys_)
        _idx = _rng.permutation(len(_y))                     # 洗牌（17 课）
        return _X[_idx], _y[_idx]

    _Xsp, _ysp = make_spiral()                               # 100 个点
    _Xsp_tr, _ysp_tr = _Xsp[:70], _ysp[:70]                  # 70 上课
    _Xsp_va, _ysp_va = _Xsp[70:], _ysp[70:]                  # 30 模考

    # 提示：train_mlp(X_tr, y_tr, X_va, y_va, h=…, lr=…, epochs=…) → 返回的第 5 个值
    # 是 hist，hist["acc_va"][-1] 是最终验证准确率。
    # 建议搜索范围：h ∈ [8, 32]、lr ∈ [0.5, 2.0]、epochs ∈ [1000, 3000]

    spiral_val_ex4 = None    # ← 你最好配置的验证集准确率（目标 ≥ 0.85）

    if spiral_val_ex4 is not None:
        if float(spiral_val_ex4) >= 0.85:
            _msg = (f"✅ val = **{float(spiral_val_ex4):.1%}** ≥ 85%——螺旋拿下！"
                    "你刚刚独立完成了一次完整的超参数搜索")
        else:
            _msg = (f"❌ val = {float(spiral_val_ex4):.1%} 还不到 85%——"
                    "加大 h、拉长 epochs、或把 lr 调到 1.0~2.0 再试")
    else:
        _msg = "⏳ 造好数据、调好参数、把最好的 val 填进来"
    mo.md(_msg)
    return (make_spiral, spiral_val_ex4)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 1 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                H = np.tanh(Z1)              # ② 隐层弯折（19 课）
                Z2 = H @ W2 + b2             # ③ 第二层打分
                ```

                形状自查口诀（20 课账本的实战版）：
                `H @ W2` 是 (n, h) @ (h, 1) → (n, 1)——**内维 h 咬合**，
                样本数 n 保送到底。写完打印一次 `Z2.shape`，
                (n, 1) 就是对的；(n,) 或 (h, ...) 都说明哪一环串了。
                """
            ),
            "✅ 练习 2 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                delta2 = (p - _yc) / n       # ① p − y，除 n（损失取过平均）
                dW2 = H.T @ delta2           # ② (h,n) @ (n,1) → (h,1) ✓
                db2 = delta2.sum(axis=0)     # ③ 消 n 对 axis=0 求和
                ```

                三个易错点逐个过堂：

                - **忘除 n**：梯度大 n 倍，lr 相当于偷偷乘了 n——
                  batch 一换，训练节奏全变。除 n 让 lr 和 batch 解耦
                - **H.T 的方向**：dW₂ 要 (h, 1)，H 是 (n, h)——
                  转置后 (h, n) 才能和 (n, 1) 相乘。**形状先行，
                  公式自来**
                - **axis=0**：δ₂ (n, 1) 压成 (1,)，消的是样本维 n，
                  即 axis=0（08 课的判据：消谁对谁求和）
                """
            ),
            "✅ 练习 3 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                delta1 = (delta2 @ W2.T) * (1 - H**2)   # ① 乘法交换 + tanh'
                dW1 = X.T @ delta1                       # ② (d,n) @ (n,h) → (d,h)
                db1 = delta1.sum(axis=0)                 # ③ 消 n 对 axis=0 求和
                ```

                ① 是全课最挤的一行，拆开看（22 课四连环的矩阵版）：

                - `delta2 @ W2.T`：(n,1) @ (1,h) → (n,h)——每个样本的
                  输出层误差，**按权重比例分发**给 h 个隐单元
                  （乘法铁律：想知道对 h 的贡献，乘另一个因子 w）
                - `* (1 - H**2)`：tanh 的坡度 (n, h)——**逐元素**贴脸乘
                  （⊙）。这里用 `*` 不用 `@`：两个同形状矩阵的逐点相乘，
                  `@` 会报形状错误（形状检查法又立功）
                - 别忘了 `**2` 是对整个 H——tanh' = 1 − tanh²，
                  19 课背过的导数表在此第四次登场

                至此，`forward` + `backward` 两大函数你全部手写过一遍——
                **这就是 PyTorch 里 `loss.backward()` 背后的全部秘密**（46 课验证）。
                """
            ),
            "✅ 练习 4 参考答案（先自己写过再看！）": mo.md(
                """
                一份能过的配置（作者实测）：**h = 24、lr = 1.0、
                epochs = 2000** → train **98.6%**、val **90%**。

                螺旋为什么比月牙难：月牙弯一次，螺旋**一路拧着弯**——
                边界要拐好几个弯，容量和训练时长都得跟上。
                几条调参手感（以后实战直接抄）：

                - **先加容量再调步长**：h = 8 时 val 只有六成多，
                  这是容量问题，调 lr 无解（交互实验同款结论）
                - **epochs 要舍得给**：1000 epoch 时还在爬坡，
                  2000 才站稳——损失曲线没平就停，等于马拉松跑 30 公里
                - **验证集说话**：h = 32 和 h = 24 都能过，
                  选哪个？看 val——一样的话选小的（参数少、快、
                  还不容易过拟合，17 课的选型纪律）
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
                1. **批量前向**：Z₁ = X@W₁+b₁ → H = tanh(Z₁) →
                   Z₂ = H@W₂+b₂ → p = σ(Z₂)；样本按行堆、n 把一把过、
                   损失取平均（np.mean），中间量全存（缓存）
                2. **四条矩阵梯度 + 形状检查法**：δ₂ = (p−y)/n；
                   dW₂ = Hᵀδ₂、db₂ = Σδ₂；δ₁ = (δ₂W₂ᵀ)⊙(1−H²)、
                   dW₁ = Xᵀδ₁、db₁ = Σδ₁——**梯度形状必须等于参数形状**，
                   消哪个维度就对哪个 axis 求和
                3. **@ vs ⊙**：矩阵乘是跨维度握手（转置对齐内维）；
                   ⊙ 是同形状逐点乘（激活函数导数的贴脸乘）。
                   写错形状当场报错——形状检查是第一道防线
                4. **13 参数验钞**：矩阵反向传播逐参数对称差分对账，
                   最大误差 1.5×10⁻⁹——"手写梯度必须过验钞机"的
                   流程（gradient check）从此是你的标准动作
                5. **双月牙战报**：线性基线 81.9% 的天花板被打破——
                   h = 8、lr = 1.0、500 epoch，train/val/test 全 100%，
                   33 个参数全靠梯度自己走到位
                6. **容量刻度**：h = 1 退化成线性（81.9%）、h = 2 临界挣扎
                   （88.9%）、h ≥ 4 稳定拿下——h 就是 17 课容量旋钮的
                   神经网络版；干净数据上 U 型右半边不现身，
                   25 课的 Dropout 与早停接手
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ---

        **下一课预告** — `24_优化器`：

        反向传播会算梯度了，但"沿负梯度走一步"只是最朴素的走法——
        下一课看看**下山这门手艺的进化史**：

        - **mini-batch**：全批太稳太慢、单点太野太快，折中才是王道
                  （epoch、batch、洗牌的工程细节一次讲清）
        - **狭长山谷的诅咒**：SGD 在椭圆等高线上之字徘徊——
          15 课"学习率两难"的几何真相
        - **动量**：扔下坡的球会记住上一步的方向，
          β = 0.9 一招让震荡自己抵消
        - **Adam**：动量 + 自适应步长，今天工业界的默认答案

        全程在**等高线地图上看轨迹**——四位选手同场竞速，
        谁磨蹭、谁冲过头、谁精准入谷，一目了然。

        🎆 梯度有了，去学怎么聪明地走！
        """
    )
    return


if __name__ == "__main__":
    app.run()
