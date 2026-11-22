import marimo

__generated_with = "0.25.2"
app = marimo.App(app_title="24 · 优化器", width="medium")


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
        # 24 · 优化器：从 SGD 到 Adam

        **从零到 Transformer · 手写深度学习** ｜ 第 4 部分 · 神经网络（7/8）
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📖 本课目标（先读我）": mo.md(
                """
                **这一课只解决一个问题：会算梯度之后，怎么走得聪明？**

                到昨天为止，我们只有一种走法：`w -= lr * grad`—— SGD
                （随机梯度下降）。它在温驯的地形上够用，可一旦地形变得
                **狭长**（一个方向陡、一个方向缓），SGD 就开始表演：
                陡的方向来回横跳，缓的方向寸步挪移。15 课的"学习率两难"，
                今天用等高线地图看清它的几何真相，再请出三代进化选手。

                打法依然稳扎稳打：先造一条**狭长山谷**当赛道，看 SGD 的
                之字和它的三种学习率命运；然后是**动量**——让球记住上一步
                的方向，震荡自己抵消自己（同一学习率，862 步变 43 步）；
                再是 **RMSProp / Adam**——给每个参数发自己的步长；
                最后回到真实网络：**mini-batch** 怎么切、四代选手在
                双月牙上同场对决。

                学完你将能：

                - 说出 **SGD 之字**的病根：梯度指向"最陡方向"而不是"谷底方向"
                - 手写**动量**（v ← βv + g）和 **Adam** 的更新规则，
                  讲清 β、自适应步长各自在治什么病
                - 定夺 **batch size**：一批多大、epoch 怎么数、
                  为什么小批吵而快、大批稳而慢
                - 给实际问题挑优化器：**从 SGD 到 Adam 的选型逻辑**
                  （以及"没有永远的第一"这条诚实结论）

                前置：第 13 课的等高线与梯度方向、第 15 课的学习率三种命运、
                第 23 课的双月牙与 train_mlp、第 10 课的洗牌
                （mini-batch 的每轮重洗）。

                25 课收官战（MNIST 十万参数）用的就是今天出厂的 Adam——
                这台发动机，先在赛道上验完货。
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "🎲 概念 1/4 · mini-batch：一步抽几个样本": mo.md(
                """
                先解决一个工程问题：每一步更新用多少样本？

                | 方案 | 每步用多少 | 性格 |
                |---|---|---|
                | 全批（batch） | 全部 n 个 | 梯度是**真梯度**，一步一个准脚印；但 n 大时一步太贵 |
                | 单点（SGD 原教旨） | 1 个 | 便宜、快，但每步都是**抽检**，方向抖得厉害 |
                | **mini-batch（主流）** | 16~256 个 | 折中：方向基本靠谱，一步还算便宜 |

                三个词的精确定义（面试高频，一次说清）：

                - **batch size**：一步同时喂给网络的样本数
                - **迭代（iteration / step）**：完成一次参数更新
                - **epoch**：把训练集**完整过一遍**。
                  换算：迭代数/epoch = ⌈n / batch⌉——72 个样本、
                  batch = 8，一个 epoch 是 9 次更新

                mini-batch 的噪声不是缺陷，是**特性**：恰到好处的抖动
                能把参数从尖锐的小坑里晃出来（全批梯度太"正"，容易被
                局部小坑锁死）。**每轮开始前必须重新洗牌**（10 课的
                permutation）——不洗的话每个 batch 的组成永远固定，
                网络会"记住batch 的顺序"而不是"学会规律"。

                一个容易忽略的联动（23 课埋过）：损失取了平均（除以 n），
                梯度尺度就和 batch size 脱钩——**batch 从 8 改 256，
                lr 基本不用动**。这是"除 n"那一步换来的工程红利。
                """
            ),
            "⛰️ 概念 2/4 · 狭长山谷：SGD 的之字与学习率两难": mo.md(
                """
                本课的赛道是一条刻意造窄的山谷：

                ```
                f(x, y) = 0.002·x² + 1·y²        谷底在原点
                ```

                y 方向曲率是 x 方向的 **500 倍**：南北两壁陡峭，
                东西谷底平缓。起点 (−12, 4)。13 课的地图在此回收：
                **梯度永远垂直于等高线**——在窄谷里，它绝大部分指向
                "对岸的壁"，只有一小截指向谷底。于是 SGD 的每一步都在
                **横穿河谷**，走到对岸又被弹回来——之字，就此成型。

                学习率两难在这里有了几何版：

                - **lr 贴着稳定上限**（0.998）：y 方向刚好不发散，
                  但衰减极慢——每步只缩小 0.4%，之字几乎原地踏步；
                  x 方向倒是以极限速度爬，但那速度本身就慢
                - **lr 再大一点**（1.05）：y 方向越过稳定边界，
                  每步反弹得比上一步更高——**指数级发散**，几十步后
                  冲出 10²⁶
                - **lr 缩小**（0.3）：y 安分了，x 更慢——862 步变 1197 步

                病根一句话：**一个 lr 伺候两个曲率**。陡方向要求小步，
                缓方向要求大步，一个标量无法两全——这就是后面两位
                选手的发力点。
                """
            ),
            "🏀 概念 3/4 · 动量：让球记住方向": mo.md(
                """
                1964 年的 Heavy Ball 想法：把"无记忆的粒子"换成
                "有惯性的球"——**速度累加梯度，位置跟随速度**：

                ```
                v ← β·v + g          （速度 = 九分旧惯性 + 一分新梯度）
                w ← w − lr·v         （沿着速度走，不是沿着梯度走）
                ```

                β（读 beta）是惯性系数，通常取 **0.9**——"记忆长度"
                约为 1/(1−β) = 10 步。它在窄谷里干两件事：

                - **之字自我抵消**：横穿河谷的梯度每步变号，
                  累加起来正负相消——震荡分量被惯性吃掉
                - **谷向持续加速**：指向谷底的分量每步同号，
                  累加起来越滚越快

                同一条山谷、同一个 lr（0.998）：SGD 走 **862 步**，
                动量（β=0.9）只要 **43 步**——20 倍加速，一行代码的事。

                β 也有自己的两难：**0.99 的惯性太重**（记忆 100 步），
                球冲过谷底再荡回来，反而要 258 步。0.9 是"记得住方向、
                放得下过去"的甜点区——深度学习的默认值和物理直觉
                在这里握手。
                """
            ),
            "📏 概念 4/4 · 自适应与 Adam：每个参数自己的步长": mo.md(
                """
                动量治了"方向"，但 lr 还是全局一个数。第三代选手
                治"步长"：**哪个参数抖得凶，就给它的步子除小一点**。

                **RMSProp**（2012，Hinton 的课堂讲义）：
                给每个参数记一本"梯度平方的流水账"：

                ```
                s ← 0.9·s + 0.1·g²              （近期梯度的平方均值）
                w ← w − lr·g / (√s + ε)         （除以自己的抖动幅度）
                ```

                y 方向梯度大（≈8），√s 大，步长被除小——不弹了；
                x 方向梯度小（≈0.05），√s 小，步长几乎不缩——照常爬。
                **一个 lr，两副步长**，病根直挖。

                **Adam**（2015，至今工业界默认）：动量 + RMSProp 合体，
                外加一步**偏差修正**：

                ```
                m ← 0.9·m + 0.1·g               （动量：方向）
                s ← 0.999·s + 0.001·g²          （平方账：步长）
                m̂ = m/(1−0.9^t)                 （起步阶段 m 从 0 出发、
                ŝ = s/(1−0.999^t)                账本偏小——除以修正项放大回来）
                w ← w − lr·m̂/(√ŝ + ε)
                ```

                偏差修正解决一个 startup 问题：前几步 m、s 都从 0 起步，
                累加太慢会显得"梯度很小"；修正项在早期把它们放大到
                正常量级，t 大了修正自动趋近 1、悄悄退场。

                一张表收尾（t 是"调参友好度"）：

                | 选手 | 年代 | 一句话 | 本课赛道成绩（入谷步数） |
                |---|---|---|---|
                | SGD | 1951 | 沿负梯度走 | 862 |
                | 动量 | 1964 | 记住方向 | 43 |
                | RMSProp | 2012 | 每人自己的步长 | **29** |
                | Adam | 2015 | 全都要 | 39 |
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

        五组演示，从赛道到真机：

        - **演示 1** · 狭长山谷 + SGD 的三种学习率命运（之字现场）
        - **演示 2** · 动量上场：同一 lr，862 步变 43 步
        - **演示 3** · 四代选手同图竞速 + 损失阶梯
        - **演示 4** · mini-batch：batch = 1 / 8 / 72 的三条曲线
        - **演示 5** · 真机对决：双月牙 MLP 上 SGD / 动量 / Adam
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 1 · 狭长山谷：SGD 的三种命运

        左图是赛道全景：等高线是**细长的椭圆**（500:1 的各向异性），
        起点在 (−12, 4)。右三图是 SGD 在三档学习率下的 120 步轨迹：

        - **lr = 0.3（谨慎）**：y 安分了，x 爬得慢——磨蹭
        - **lr = 0.998（理论最优）**：y 之字几乎不衰减
          （4 → −3.98 → 3.97 → −3.95 …每步只缩 0.4%），x 稳步爬——
          **稳，但要 862 步才入谷**
        - **lr = 1.05（越界）**：y 每步弹得更高，指数发散——
          300 步后损失 10²⁶，轨迹冲出画面

        "最优学习率"也逃不出 862 步——这不是调参问题，是
        **SGD 结构性的天花板**（病根见概念 2）。
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # 演示 1 · 山谷赛道 + SGD 三种命运（run_valley 供全课复用）
    def f_valley(w):
        return 0.002 * w[0] ** 2 + w[1] ** 2          # 窄谷：y 曲率是 x 的 500 倍

    def grad_valley(w):
        return np.array([0.004 * w[0], 2.0 * w[1]])   # ∇f（13 课）

    W0 = np.array([-12.0, 4.0])                       # 统一起点

    def run_valley(kind, lr, beta=0.0, n=300, w0=None):
        """四位选手的通用跑法。返回轨迹数组 (n+1, 2)"""
        _w = (w0 if w0 is not None else W0).copy()
        _v = np.zeros(2)
        _m = np.zeros(2)
        _s = np.zeros(2)
        _traj = [_w.copy()]
        for _t in range(1, n + 1):
            _g = grad_valley(_w)
            if kind == "sgd":
                _w = _w - lr * _g
            elif kind == "mom":
                _v = beta * _v + _g                   # 速度 = 惯性 + 新梯度
                _w = _w - lr * _v
            elif kind == "rms":
                _s = 0.9 * _s + 0.1 * _g ** 2         # 平方流水账
                _w = _w - lr * _g / (np.sqrt(_s) + 1e-8)
            else:                                     # adam
                _m = 0.9 * _m + 0.1 * _g
                _s = 0.999 * _s + 0.001 * _g ** 2
                _mh = _m / (1 - 0.9 ** _t)            # 偏差修正（起步放大）
                _sh = _s / (1 - 0.999 ** _t)
                _w = _w - lr * _mh / (np.sqrt(_sh) + 1e-8)
            _traj.append(_w.copy())
        return np.array(_traj)

    def steps_in(traj, threshold=None):
        """多少步后 f 降到阈值以下（默认：初值的 1/1000）"""
        _th = threshold if threshold is not None else f_valley(W0) * 1e-3
        for _i, _w in enumerate(traj):
            if f_valley(_w) < _th:
                return _i
        return None

    # ---- 画图：左赛道全景 + 右三档 lr 的轨迹 ----
    _fig, _axes = plt.subplots(1, 4, figsize=(14.2, 3.8),
                               gridspec_kw={"width_ratios": [1.15, 1, 1, 1]})
    _gx = np.linspace(-13.5, 3.5, 300)
    _gy = np.linspace(-5.5, 5.5, 300)
    _GX, _GY = np.meshgrid(_gx, _gy)
    _FZ = 0.002 * _GX ** 2 + _GY ** 2
    _cases = [(0.3, "lr=0.3：磨蹭"), (0.998, "lr=0.998：之字（理论最优）"),
              (1.05, "lr=1.05：发散")]
    for _ax in _axes:
        _ax.contour(_GX, _GY, _FZ, levels=np.logspace(-3, 1.3, 14),
                    colors="gray", linewidths=0.7, alpha=0.75)
    _axes[0].scatter(*W0, s=90, color="tab:red", zorder=5, marker="*")
    _axes[0].set_title("赛道：狭长山谷\n（等高线 500:1 拉长）")
    for _ax, (_lr, _ttl) in zip(_axes[1:], _cases):
        _tr = run_valley("sgd", _lr, n=120)
        _ok = np.abs(_tr[:, 1]) < 5.5                 # 发散的轨迹截断在画面内
        _ax.plot(_tr[_ok, 0], _tr[_ok, 1], "-o", color="tab:blue", ms=2.5, lw=1.4)
        _ax.scatter(*W0, s=60, color="tab:red", zorder=5, marker="*")
        _steps = steps_in(run_valley("sgd", _lr, n=3000))
        _ax.set_title(_ttl + f"\n入谷 {_steps if _steps else '＞3000'} 步", fontsize=9.5)
    for _ax in _axes:
        _ax.set_xlabel("x")
        _ax.set_xlim(-13.5, 3.5)
        _ax.set_ylim(-5.5, 5.5)
    _axes[0].set_ylabel("y")
    _fig.tight_layout()
    _fig
    return (W0, f_valley, grad_valley, run_valley, steps_in)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 2 · 动量上场：862 步变 43 步

        同一条山谷、同一个 lr（0.998），只加两行动量代码。三张轨迹图
        是同一场比赛的三种性格：

        - **SGD**：横穿河谷的之字，一晚上没挪出几步
        - **动量 β=0.9**：前几步还有点晃，惯性迅速接管——之字分量
          正负抵消、谷向分量持续加速，**43 步入谷**（SGD 的 1/20）
        - **动量 β=0.99**：惯性过重（记忆 100 步）——球一路冲过谷底
          又荡回来，**258 步**。β 不是越大越好，0.9 是甜点

        右侧损失曲线（log 尺）更直白：SGD 一条近乎平坦的斜线
        （每步 0.4% 的蜗牛速度），β=0.9 一路俯冲，β=0.99 俯冲中
        带着清晰的过冲回荡。
        """
    )
    return


@app.cell
def _(W0, f_valley, np, plt, run_valley, steps_in):
    # 演示 2 · 动量三连：SGD vs β=0.9 vs β=0.99
    _tr_sgd = run_valley("sgd", 0.998, n=300)
    _tr_m90 = run_valley("mom", 0.998, beta=0.9, n=300)
    _tr_m99 = run_valley("mom", 0.998, beta=0.99, n=300)

    _fig, _axes = plt.subplots(1, 4, figsize=(14.2, 3.8),
                               gridspec_kw={"width_ratios": [1, 1, 1, 1.1]})
    _gx = np.linspace(-13.5, 3.5, 200)
    _gy = np.linspace(-5.5, 5.5, 200)
    _GX, _GY = np.meshgrid(_gx, _gy)
    _FZ = 0.002 * _GX ** 2 + _GY ** 2
    for _ax in _axes[:3]:
        _ax.contour(_GX, _GY, _FZ, levels=np.logspace(-3, 1.3, 12),
                    colors="gray", linewidths=0.7, alpha=0.75)
        _ax.set_xlim(-13.5, 3.5)
        _ax.set_ylim(-5.5, 5.5)
        _ax.set_xlabel("x")
    _axes[0].set_ylabel("y")
    for _ax, _tr, _ttl in [
        (_axes[0], _tr_sgd, f"SGD\n入谷 {steps_in(_tr_sgd)} 步"),
        (_axes[1], _tr_m90, f"动量 β=0.9\n入谷 {steps_in(_tr_m90)} 步"),
        (_axes[2], _tr_m99, f"动量 β=0.99\n入谷 {steps_in(_tr_m99)} 步"),
    ]:
        _ax.plot(_tr[:, 0], _tr[:, 1], "-o", color="tab:orange", ms=2.5, lw=1.4)
        _ax.scatter(*W0, s=60, color="tab:red", zorder=5, marker="*")
        _ax.set_title(_ttl, fontsize=10)

    _ts = np.arange(301)
    _axes[3].semilogy(_ts, [f_valley(_w) + 1e-12 for _w in _tr_sgd],
                      color="tab:blue", label="SGD")
    _axes[3].semilogy(_ts, [f_valley(_w) + 1e-12 for _w in _tr_m90],
                      color="tab:orange", label="动量 β=0.9")
    _axes[3].semilogy(_ts, [f_valley(_w) + 1e-12 for _w in _tr_m99],
                      color="tab:green", ls="--", label="动量 β=0.99")
    _axes[3].axhline(f_valley(W0) * 1e-3, color="gray", ls=":", lw=1)
    _axes[3].text(150, f_valley(W0) * 1.1e-3, "入谷线", fontsize=9, color="gray")
    _axes[3].set_xlabel("步数")
    _axes[3].set_ylabel("f（log 尺）")
    _axes[3].set_title("损失曲线", fontsize=10)
    _axes[3].legend(fontsize=9)
    _axes[3].grid(True, alpha=0.3, which="both")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 3 · 四代选手同图竞速

        四位选手各自带默认 lr（SGD/动量用理论最优 0.998，
        RMSProp/Adam 用 0.3——自适应选手的 lr 天生小一号，
        概念 4 讲过原因）。左图轨迹、右图损失：

        | 选手 | lr | 入谷步数 | 轨迹长相 |
        |---|---|---|---|
        | SGD | 0.998 | 862 | 之字地毯 |
        | 动量 | 0.998 | 43 | 一记平滑的滑铲 |
        | RMSProp | 0.3 | **29** | 直奔谷底，几乎不晃 |
        | Adam | 0.3 | 39 | 带一点方向修正的直线 |

        RMSProp 为什么最快：y 方向梯度大（起步 ≈8），平方账立刻记下，
                步长被 √s 除小——第一步就"知道收敛"；x 方向梯度小，
        步长几乎不缩。**它不是走得快，是每一步都走在该走的地方**。

        Adam 比 RMSProp 多了动量（起步要拐个弯）慢了 10 步，
        但换来对 lr 更钝感的皮实——这笔账在真机上才看得清（演示 5）。
        """
    )
    return


@app.cell
def _(W0, f_valley, np, plt, run_valley, steps_in):
    # 演示 3 · 四代选手：轨迹同图 + 损失阶梯
    _players = [
        ("SGD", "sgd", 0.998, 0.0, "tab:blue", "-"),
        ("动量", "mom", 0.998, 0.9, "tab:orange", "-"),
        ("RMSProp", "rms", 0.3, 0.0, "tab:green", "-"),
        ("Adam", "adam", 0.3, 0.0, "tab:red", "-"),
    ]
    _trajs = {}
    for _name, _kind, _lr, _beta, _c, _ls in _players:
        _trajs[_name] = run_valley(_kind, _lr, beta=_beta, n=300)

    _fig, _axes = plt.subplots(1, 2, figsize=(12.6, 4.6),
                               gridspec_kw={"width_ratios": [1.25, 1]})
    _gx = np.linspace(-13.5, 3.5, 300)
    _gy = np.linspace(-5.5, 5.5, 300)
    _GX, _GY = np.meshgrid(_gx, _gy)
    _axes[0].contour(_GX, _GY, 0.002 * _GX ** 2 + _GY ** 2,
                     levels=np.logspace(-3, 1.3, 14), colors="gray",
                     linewidths=0.7, alpha=0.75)
    for _name, _kind, _lr, _beta, _c, _ls in _players:
        _tr = _trajs[_name][:120]                    # 轨迹图只画前 120 步（SGD 也够看出之字）
        _axes[0].plot(_tr[:, 0], _tr[:, 1], _ls, color=_c, lw=1.9,
                      label=f"{_name}（{steps_in(_trajs[_name])} 步）")
    _axes[0].scatter(*W0, s=90, color="tab:red", zorder=5, marker="*")
    _axes[0].set_xlim(-13.5, 3.5)
    _axes[0].set_ylim(-5.5, 5.5)
    _axes[0].set_xlabel("x")
    _axes[0].set_ylabel("y")
    _axes[0].set_title("四代选手的前 120 步")
    _axes[0].legend(fontsize=9)

    _ts = np.arange(301)
    for _name, *_ in _players:
        _axes[1].semilogy(_ts, [f_valley(_w) + 1e-12 for _w in _trajs[_name]],
                          lw=1.9, label=_name)
    _axes[1].axhline(f_valley(W0) * 1e-3, color="gray", ls=":", lw=1)
    _axes[1].set_xlabel("步数")
    _axes[1].set_ylabel("f（log 尺）")
    _axes[1].set_title("损失阶梯（log 尺）")
    _axes[1].legend(fontsize=9)
    _axes[1].grid(True, alpha=0.3, which="both")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 4 · mini-batch：batch 大小的三条曲线

        回到双月牙（23 课同款数据、同款网络 h=8），训练 60 个 epoch，
        只改 batch size。三条曲线记录**每个 epoch 结束时的全批损失**：

        | batch | 每 epoch 更新次数 | L@10 | L@60 | 后半程波动 |
        |---|---|---|---|---|
        | 1（单点） | 72 | 0.065 | **0.0016** | 0.002 |
        | 8 | 9 | 0.315 | 0.041 | **0.096（看得见的抖）** |
        | 72（全批） | 1 | 0.404 | 0.322 | 0.021 |

        两副眼镜各看一层：

        - **按 epoch 看**（横轴相同）：小批快——单点版一个 epoch 摸 72 次
                  参数，全批只摸 1 次，60 个 epoch 相当于别人 60 步
        - **按步看**：大批判——全批每步都是真梯度，曲线平滑；
          batch=8 每步抽 8 个样本，梯度带噪，曲线在谷底附近还在抖
          （后半程 0.096 的摆幅就是噪声的脚印）

        实战的甜点通常在 **16~256**：一批的梯度已经基本靠谱
        （中心极限定理：抽 16 个，均值的标准差就缩到单点的 1/4），
                一步的算力又没爆。GPU 时代还有个隐藏理由：矩阵乘法
        在 batch=2 的幂次上跑得最快（并行吃满）。
        """
    )
    return


@app.cell
def _(np, plt):
    # 演示 4 · mini-batch：双月牙上 batch = 1 / 8 / 72（train_batch 供交互实验复用）
    def make_moons24(n_per=60, noise=0.10, seed=11):
        """23 课同款双月牙（种子都没换，成绩可直接对比）"""
        _rng = np.random.default_rng(seed)
        _t = _rng.uniform(0, np.pi, n_per)
        _X0 = np.column_stack([np.cos(_t), np.sin(_t)]) + _rng.normal(0, noise, (n_per, 2))
        _X1 = np.column_stack([1 - np.cos(_t), 0.5 - np.sin(_t)]) + _rng.normal(0, noise, (n_per, 2))
        _X = np.vstack([_X0, _X1])
        _y = np.concatenate([np.zeros(n_per), np.ones(n_per)])
        _idx = _rng.permutation(len(_y))
        return _X[_idx], _y[_idx]

    _X_all24, _y_all24 = make_moons24()
    X_tr24, y_tr24 = _X_all24[:72], _y_all24[:72]      # 上课 72 点（与 23 课同切分）

    def train_batch(Xb_all, yb_all, batch, h=8, lr=0.5, epochs=60, seed=0):
        """带 mini-batch 的 MLP 训练（23 课的 forward/backward + 洗牌切批）。
        返回每个 epoch 的全批损失列表"""
        _rng_i = np.random.default_rng(0)            # 初始化与 23 课同款
        _W1 = _rng_i.normal(0, 1, (2, h)) * np.sqrt(2 / Xb_all.shape[1])
        _b1 = np.zeros(h)
        _W2 = _rng_i.normal(0, 1, (h, 1)) * np.sqrt(2 / h)
        _b2 = np.zeros(1)
        _rng = np.random.default_rng(seed + 1)
        _Ls = []
        for _ep in range(epochs):
            _idx = _rng.permutation(len(Xb_all))    # 每轮先洗牌（10 课）
            for _s in range(0, len(Xb_all), batch):
                _ib = _idx[_s:_s + batch]           # 切一片当一步
                _X, _y = Xb_all[_ib], yb_all[_ib]
                _Z1 = _X @ _W1 + _b1
                _H = np.tanh(_Z1)
                _p = 1 / (1 + np.exp(-(_H @ _W2 + _b2)))
                _n = len(_X)
                _yc = _y.reshape(-1, 1)
                _d2 = (_p - _yc) / _n               # 23 课四条梯度
                _d1 = (_d2 @ _W2.T) * (1 - _H ** 2)
                _W2 -= lr * (_H.T @ _d2)
                _b2 -= lr * _d2.sum(0)
                _W1 -= lr * (_X.T @ _d1)
                _b1 -= lr * _d1.sum(0)
            # epoch 结束：全批损失
            _pa = 1 / (1 + np.exp(-((np.tanh(Xb_all @ _W1 + _b1)) @ _W2 + _b2)))
            _yca = yb_all.reshape(-1, 1)
            _Ls.append(float(-np.mean(_yca * np.log(_pa + 1e-12)
                                      + (1 - _yca) * np.log(1 - _pa + 1e-12))))
        return _Ls

    _fig, _ax = plt.subplots(figsize=(7.8, 4.3))
    for _batch, _c in [(1, "tab:blue"), (8, "tab:orange"), (72, "tab:green")]:
        _Ls = train_batch(X_tr24, y_tr24, _batch)
        _ax.plot(range(1, 61), _Ls, "-o", ms=3, color=_c, lw=1.8,
                 label=f"batch = {_batch}（每 epoch {int(np.ceil(72 / _batch))} 次更新）")
    _ax.set_xlabel("epoch")
    _ax.set_ylabel("全批训练损失")
    _ax.set_title("batch 大小：小批快而吵，大批稳而慢")
    _ax.legend(fontsize=9)
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()
    _fig
    return (X_tr24, train_batch, y_tr24)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 5 · 真机对决：双月牙 MLP 上的三代选手

        玩具赛道验完货，上真机：双月牙、h = 8、**batch = 16**、
        100 个 epoch，只换优化器（lr 各自取顺手的量级）：

        | 选手 | lr | L@10 | L@100 | 最终 acc |
        |---|---|---|---|---|
        | SGD | 0.5 | 0.340 | 0.051 | 100% |
        | 动量 | 0.2 | 0.250 | **0.005** | 100% |
        | Adam | 0.02 | 0.335 | 0.056 | 100% |

        三个读数值得咀嚼：

        - **动量最快**（L@100 = 0.005）：小数据集 + 温驯地形，
          惯性优势直接兑现
        - **Adam 的 lr 小一号**（0.02）：自适应除法把每步长度
          归一化了，用和 SGD 一样大的 lr 反而会抖——
          **换优化器必须重调 lr**，这是新手最常踩的坑
        - **三者都到 100%**：问题不难时，优化器的差距只是"快慢"；
          难问题（大模型、长训练）里才是"能不能训动"的生死差——
          25 课的 MNIST 会再见分晓

        顺便把学到的组合拳串起来：**洗牌 → 切批 → 前向 → 反向 →
        优化器更新**——这套流程加上 25 课的"初始化 + Dropout + 早停"，
        就是工业界训练循环的完整形态。
        """
    )
    return


@app.cell
def _(X_tr24, mo, np, plt, y_tr24):
    # 演示 5 · 真机对决：SGD / 动量 / Adam 在双月牙 MLP 上（batch=16）
    def train_duel(kind, lr, batch=16, epochs=100, h=8, seed=0):
        _rng_i = np.random.default_rng(0)
        _W1 = _rng_i.normal(0, 1, (2, h)) * np.sqrt(2 / 2)
        _b1 = np.zeros(h)
        _W2 = _rng_i.normal(0, 1, (h, 1)) * np.sqrt(2 / h)
        _b2 = np.zeros(1)
        _rng = np.random.default_rng(seed + 1)
        _params = [_W1, _b1, _W2, _b2]
        _v = [np.zeros_like(_p) for _p in _params]
        _m = [np.zeros_like(_p) for _p in _params]
        _s = [np.zeros_like(_p) for _p in _params]
        _Ls = []
        for _ep in range(epochs):
            _idx = _rng.permutation(len(X_tr24))
            for _st in range(0, len(X_tr24), batch):
                _ib = _idx[_st:_st + batch]
                _X, _y = X_tr24[_ib], y_tr24[_ib]
                _Z1 = _X @ _params[0] + _params[1]
                _H = np.tanh(_Z1)
                _p = 1 / (1 + np.exp(-(_H @ _params[2] + _params[3])))
                _n = len(_X)
                _yc = _y.reshape(-1, 1)
                _d2 = (_p - _yc) / _n
                _d1 = (_d2 @ _params[2].T) * (1 - _H ** 2)
                _grads = [_X.T @ _d1, _d1.sum(0), _H.T @ _d2, _d2.sum(0)]
                _t = _ep * 5 + _st // batch + 1      # 全局步数（偏差修正用）
                for _i in range(4):
                    if kind == "sgd":
                        _params[_i] -= lr * _grads[_i]
                    elif kind == "mom":
                        _v[_i] = 0.9 * _v[_i] + _grads[_i]
                        _params[_i] -= lr * _v[_i]
                    else:
                        _m[_i] = 0.9 * _m[_i] + 0.1 * _grads[_i]
                        _s[_i] = 0.999 * _s[_i] + 0.001 * _grads[_i] ** 2
                        _params[_i] -= lr * (_m[_i] / (1 - 0.9 ** _t)) \
                            / (np.sqrt(_s[_i] / (1 - 0.999 ** _t)) + 1e-8)
            _pa = 1 / (1 + np.exp(-((np.tanh(X_tr24 @ _params[0] + _params[1]))
                                    @ _params[2] + _params[3])))
            _yca = y_tr24.reshape(-1, 1)
            _Ls.append(float(-np.mean(_yca * np.log(_pa + 1e-12)
                                      + (1 - _yca) * np.log(1 - _pa + 1e-12))))
        _acc = float(((_pa > 0.5) == _yca).mean())
        return _Ls, _acc

    _duels = [("SGD", "sgd", 0.5, "tab:blue"), ("动量", "mom", 0.2, "tab:orange"),
              ("Adam", "adam", 0.02, "tab:red")]
    _fig, _ax = plt.subplots(figsize=(7.8, 4.3))
    _rows = []
    for _name, _kind, _lr, _c in _duels:
        _Ls, _acc = train_duel(_kind, _lr)
        _ax.plot(range(1, 101), _Ls, "-o", ms=3, color=_c, lw=1.8,
                 label=f"{_name}（lr={_lr}）")
        _rows.append(f"| {_name} | {_lr} | {_Ls[9]:.3f} | **{_Ls[-1]:.3f}** | {_acc:.0%} |")
    _ax.set_xlabel("epoch")
    _ax.set_ylabel("全批训练损失")
    _ax.set_title("真机对决（batch=16，100 epoch）")
    _ax.legend(fontsize=9)
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()

    mo.vstack([
        _fig,
        mo.md("| 选手 | lr | L@10 | L@100 | 最终 acc |\n|---|---|---|---|---|\n"
              + "\n".join(_rows)),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 🎛️ 交互实验

        本课两组（关键课加倍）：

        1. **山谷赛道**——下拉选选手、滑块调 lr，看轨迹和入谷步数。
           玩法：给 SGD 配 lr=1.1 看它当场出局；给动量配 lr=1.1
           看惯性怎么救命；把 Adam 的 lr 从 0.3 拖到 0.05，
           看"皮实"两个字值多少钱
        2. **batch 旋钮**——滑块扫 batch size（1 ~ 72），
           看双月牙训练曲线怎么从"快而吵"变"稳而慢"
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 交互实验 1 · 山谷赛道
        """
    )
    return


@app.cell
def _(mo):
    # 交互控制：选手 + 学习率
    opt_dropdown = mo.ui.dropdown(
        options=["SGD", "动量", "RMSProp", "Adam"], value="动量",
        label="🏎️ 选手",
    )
    opt_lr_slider = mo.ui.slider(0.05, 1.2, value=0.998, step=0.01, debounce=True,
                                 label="🎛️ 学习率 lr（0.05 ~ 1.20）")
    mo.hstack([opt_dropdown, opt_lr_slider])
    return (opt_dropdown, opt_lr_slider)


@app.cell
def _(W0, f_valley, mo, np, opt_dropdown, opt_lr_slider, plt, run_valley, steps_in):
    # ===== 交互实验 1 · 山谷赛道（响应单元）=====
    _name = opt_dropdown.value or "动量"
    _lr = opt_lr_slider.value
    _kind = {"SGD": "sgd", "动量": "mom", "RMSProp": "rms", "Adam": "adam"}[_name]
    _beta = 0.9 if _kind == "mom" else 0.0

    _tr = run_valley(_kind, _lr, beta=_beta, n=300)
    _steps = steps_in(_tr)
    _f_final = f_valley(_tr[-1])

    _fig, _ax = plt.subplots(figsize=(7.6, 4.6))
    _gx = np.linspace(-13.5, 3.5, 260)
    _gy = np.linspace(-5.5, 5.5, 260)
    _GX, _GY = np.meshgrid(_gx, _gy)
    _ax.contour(_GX, _GY, 0.002 * _GX ** 2 + _GY ** 2,
                levels=np.logspace(-3, 1.3, 14), colors="gray", linewidths=0.7, alpha=0.75)
    _ok = (np.abs(_tr[:, 1]) < 5.5) & (np.abs(_tr[:, 0]) < 3.5)   # 截断发散轨迹
    _ax.plot(_tr[_ok, 0], _tr[_ok, 1], "-o", color="tab:red", ms=2.5, lw=1.5)
    _ax.scatter(*W0, s=90, color="tab:red", zorder=5, marker="*")
    _ax.set_xlim(-13.5, 3.5)
    _ax.set_ylim(-5.5, 5.5)
    _ax.set_xlabel("x")
    _ax.set_ylabel("y")
    _ax.set_title(f"{_name}（lr = {_lr:.2f}）的前 300 步")
    _fig.tight_layout()

    # ---- 判词 ----
    if not np.isfinite(_f_final) or _f_final > 1e6:
        _verdict = (f"🚨 出局：lr = {_lr:.2f} 越过了 {_name} 的稳定边界，"
                    "损失指数发散——轨迹已冲出画面")
    elif _steps is None:
        _verdict = f"🐌 300 步没入谷（当前 f = {_f_final:.3f}）——lr 不合适或选手太慢，再调调"
    else:
        _verdict = (f"✅ 第 **{_steps}** 步入谷（对照：SGD 最优 lr 要 862 步、"
                    f"动量 43、RMSProp 29、Adam 39）")

    mo.vstack([
        _fig,
        mo.callout(mo.md(f"**实时判词：{_verdict}**"), kind="info"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 交互实验 2 · batch 旋钮

        滑块从 1 拉到 72（双月牙、h = 8、lr = 0.5、60 epoch），
        曲线随 batch 变形：左边小批快而吵，右边大批稳而慢。
        试着找到你认为"最划算"的那个 batch——然后想想：
        你的标准和"每个 epoch 的墙钟时间"一样吗？
        """
    )
    return


@app.cell
def _(mo):
    # 交互控制：batch size（指数档位，0~6 映射 1,2,4,8,16,36,72）
    batch_slider = mo.ui.slider(0, 6, value=3, step=1, debounce=True,
                                label="🎛️ batch size 档位（1 / 2 / 4 / 8 / 16 / 36 / 72）")
    batch_slider
    return (batch_slider,)


@app.cell
def _(X_tr24, batch_slider, mo, np, plt, train_batch, y_tr24):
    # ===== 交互实验 2 · batch 旋钮（响应单元）=====
    _batch = [1, 2, 4, 8, 16, 36, 72][batch_slider.value]
    _Ls = train_batch(X_tr24, y_tr24, _batch)
    _upd = int(np.ceil(72 / _batch))               # 每 epoch 的更新次数

    _fig, _ax = plt.subplots(figsize=(7.6, 4.1))
    _ax.plot(range(1, 61), _Ls, "-o", ms=3.5, color="tab:purple", lw=1.8,
             label=f"batch = {_batch}")
    _ax.set_xlabel("epoch")
    _ax.set_ylabel("全批训练损失")
    _ax.set_title(f"batch = {_batch}：每 epoch 更新 {_upd} 次")
    _ax.set_ylim(0, 0.75)
    _ax.legend()
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()

    _swing = max(_Ls[30:]) - min(_Ls[30:])        # 后半程摆幅（噪声的脚印）
    _verdict = (f"batch = **{_batch}**：L@60 = **{_Ls[-1]:.4f}**，"
                f"后半程摆幅 **{_swing:.3f}**——"
                + ("快而吵（单点抽检，梯度最野）" if _batch <= 2 else
                   "小批：快，带一点抖（甜点区附近）" if _batch <= 16 else
                   "大批：稳而慢（每 epoch 只更新寥寥几次）"))

    mo.vstack([
        _fig,
        mo.callout(mo.md(f"**实时判词：{_verdict}**"), kind="info"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✏️ 动手练习（每课必做，本课 3 题）": mo.md(
                """
                **练习 1 · 手写动量更新**：两行代码（速度累加 + 沿速度走），
                用 40 步后的位置和参考答案对账。

                **练习 2 · 手写 Adam 单步**：四行（动量、平方账、两个偏差修正、
                更新）——25 课收官战的发动机，今天装配完毕。

                **练习 3 · 挑战 · 旋转山谷竞速**：把山谷旋转 30°，
                四位选手（统一 lr = 0.3）重新比赛——排名会和正谷
                **不一样**。跑出结果，填最快的选手和它的步数。

                规则不变：先把 `None` 换成你的答案再运行；写完再看参考答案。
                """
            )
        }
    )
    return


@app.cell
def _(mo, np, run_valley):
    # ===== 练习 1 · 手写动量更新 =====
    def momentum_run_ex1(lr, beta, n=40):
        """从 (-12, 4) 出发跑 n 步动量。返回最终位置"""
        _w = np.array([-12.0, 4.0])
        _v = np.zeros(2)                # 速度从零起步
        for _ in range(n):
            _g = np.array([0.004 * _w[0], 2.0 * _w[1]])   # 山谷梯度（题面给定）
            _v = None                   # ← ① 补：beta * _v + _g
            _w = None                   # ← ② 补：_w - lr * _v
        return _w

    mom_ready_ex1 = None                # ← ③ 补完后，把这行换成 momentum_run_ex1

    if mom_ready_ex1 is not None:
        try:
            _got = mom_ready_ex1(0.998, 0.9)
            _ref = run_valley("mom", 0.998, beta=0.9, n=40)[-1]
            if np.allclose(_got, _ref, atol=1e-9):
                _msg = f"✅ 40 步后位置 {np.round(_got, 4)}——与参考轨迹分毫不差"
            else:
                _msg = (f"❌ 你到 {np.round(_got, 4)}，参考是 {np.round(_ref, 4)}——"
                        "检查①是'惯性+新梯度'、②是沿着 v 走而不是沿着 g 走")
        except TypeError:
            _msg = "⏳ _v 或 _w 还是 None——两行都补上"
    else:
        _msg = "⏳ 先补两行更新，再把 mom_ready_ex1 换成 momentum_run_ex1"
    mo.md(_msg)
    return (mom_ready_ex1, momentum_run_ex1)


@app.cell
def _(mo, np, run_valley):
    # ===== 练习 2 · 手写 Adam 单步 =====
    def adam_run_ex2(lr, n=60):
        """从 (-12, 4) 出发跑 n 步 Adam。返回最终位置"""
        _w = np.array([-12.0, 4.0])
        _m = np.zeros(2)
        _s = np.zeros(2)
        for _t in range(1, n + 1):
            _g = np.array([0.004 * _w[0], 2.0 * _w[1]])
            _m = None                   # ← ① 补：0.9 * _m + 0.1 * _g
            _s = None                   # ← ② 补：0.999 * _s + 0.001 * _g ** 2
            _mh = None                  # ← ③ 补：_m / (1 - 0.9 ** _t)
            _sh = None                  # ← ④ 补：_s / (1 - 0.999 ** _t)
            _w = _w - lr * _mh / (np.sqrt(_sh) + 1e-8)   # 更新（这行不变）
        return _w

    adam_ready_ex2 = None               # ← ⑤ 补完后，把这行换成 adam_run_ex2

    if adam_ready_ex2 is not None:
        try:
            _got = adam_ready_ex2(0.3)
            _ref = run_valley("adam", 0.3, n=60)[-1]
            if np.allclose(_got, _ref, atol=1e-9):
                _msg = f"✅ 60 步后位置 {np.round(_got, 5)}——Adam 装配完毕，25 课直接上车"
            else:
                _msg = (f"❌ 你到 {np.round(_got, 5)}，参考是 {np.round(_ref, 5)}——"
                        "四行逐个查：0.9/0.1、0.999/0.001、两个修正的底数")
        except TypeError:
            _msg = "⏳ 有 None 没补——四行逐个来"
    else:
        _msg = "⏳ 先补四行，再把 adam_ready_ex2 换成 adam_run_ex2"
    mo.md(_msg)
    return (adam_ready_ex2, adam_run_ex2)


@app.cell
def _(mo, np):
    # ===== 练习 3 · 挑战：旋转山谷竞速 =====
    _th_deg = 30.0
    _R = np.array([[np.cos(np.deg2rad(_th_deg)), -np.sin(np.deg2rad(_th_deg))],
                   [np.sin(np.deg2rad(_th_deg)), np.cos(np.deg2rad(_th_deg))]])

    def f_rot(w):
        _u = _R.T @ w
        return 0.002 * _u[0] ** 2 + _u[1] ** 2       # 同款山谷，转了 30°

    def g_rot(w):
        _u = _R.T @ w
        return _R @ np.array([0.004 * _u[0], 2.0 * _u[1]])

    def run_rot(kind, lr, beta, n=5000):
        """四选手在旋转山谷上跑。返回入谷步数（阈值 = 初值的 1/1000）"""
        _w = np.array([-12.0, 4.0])
        _v = np.zeros(2); _m = np.zeros(2); _s = np.zeros(2)
        _th = f_rot(_w) * 1e-3
        for _t in range(1, n + 1):
            _g = g_rot(_w)
            if kind == "sgd":
                _w = _w - lr * _g
            elif kind == "mom":
                _v = beta * _v + _g; _w = _w - lr * _v
            elif kind == "rms":
                _s = 0.9 * _s + 0.1 * _g ** 2; _w = _w - lr * _g / (np.sqrt(_s) + 1e-8)
            else:
                _m = 0.9 * _m + 0.1 * _g; _s = 0.999 * _s + 0.001 * _g ** 2
                _w = _w - lr * (_m / (1 - 0.9 ** _t)) / (np.sqrt(_s / (1 - 0.999 ** _t)) + 1e-8)
            if f_rot(_w) < _th:
                return _t
        return None

    # 提示：四位选手统一 lr = 0.3（动量 β = 0.9），分别 run_rot("sgd" / "mom" / "rms" / "adam", 0.3, ...)

    fastest_name_ex3 = None    # ← ① 最快的选手名（填 "SGD" / "动量" / "RMSProp" / "Adam"）
    fastest_steps_ex3 = None   # ← ② 它的入谷步数

    if fastest_name_ex3 is not None and fastest_steps_ex3 is not None:
        if (fastest_name_ex3, fastest_steps_ex3) == ("Adam", 28):
            _msg = "✅ Adam 28 步夺冠——旋转地形让排名洗牌，参考答案有复盘"
        elif fastest_name_ex3 in ("SGD", "动量", "RMSProp", "Adam"):
            _ref_steps = run_rot({"SGD": "sgd", "动量": "mom", "RMSProp": "rms",
                                  "Adam": "adam"}[fastest_name_ex3], 0.3,
                                 0.9 if fastest_name_ex3 == "动量" else 0.0)
            _msg = (f"❌ {fastest_name_ex3} 实际入谷 {_ref_steps} 步——"
                    "四位的成绩都跑一遍再比（统一 lr=0.3）")
        else:
            _msg = "⏳ 选手名填 SGD / 动量 / RMSProp / Adam 之一"
    else:
        _msg = "⏳ 两个空都填上（提示：这次的冠军不是 RMSProp）"
    mo.md(_msg)
    return (fastest_name_ex3, fastest_steps_ex3)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 1 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                _v = beta * _v + _g        # ① 速度 = 九分惯性 + 一分新梯度
                _w = _w - lr * _v          # ② 沿速度走（不是沿梯度走！）
                ```

                40 步后位置 ≈ (−0.55, 0.31)——已经贴着谷底。

                最常见的错法是把 ② 写成 `_w - lr * _g`（沿梯度走）——
                那就退回了 SGD，惯性白攒了。**动量的全部含义
                在"沿 v 走"这三个字里**：速度才是位置的主人，
                梯度只是速度的供应商。

                一个数字直觉：β = 0.9 时，1/(1−β) = 10——
                今天的梯度只占速度的一成，过去 10 步的平均方向
                占九成。这就是"记忆长度"。
                """
            ),
            "✅ 练习 2 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                _m = 0.9 * _m + 0.1 * _g                   # ① 动量
                _s = 0.999 * _s + 0.001 * _g ** 2          # ② 平方账
                _mh = _m / (1 - 0.9 ** _t)                 # ③ 动量修正
                _sh = _s / (1 - 0.999 ** _t)               # ④ 平方账修正
                ```

                两个容易问的为什么：

                - **为什么 ② 用 0.999 这么慢的账**：梯度平方是非负数，
                  累加快了会"记仇"（一次大梯度让 √s 长期偏大、步长
                  长期偏小）。慢账只记"近期平均抖动"
                - **修正项什么时候退场**：0.9^t 和 0.999^t 随 t 衰减，
                  t = 100 时 0.999^100 ≈ 0.905（还在修正），
                  t = 10000 时 ≈ 4.5e-5（修正归一、悄悄退场）——
                  它只管"起步那一下"

                装好这台发动机，25 课的 MNIST 收官战直接点火。
                """
            ),
            "✅ 练习 3 参考答案（先自己写过再看！）": mo.md(
                """
                旋转 30° 后的成绩单（统一 lr = 0.3，阈值 = 初值/1000）：

                | 选手 | 入谷步数 | 和正谷比 |
                |---|---|---|
                | SGD | 188 | 862 → 188（旋转后梯度方向变了，反而顺脚） |
                | 动量 | 40 | 43 → 40 |
                | RMSProp | 32 | 29 → 32 |
                | **Adam** | **28** 🏆 | 39 → 28 |

                **Adam 夺冠**。复盘两个要点：

                - RMSProp 的"除以自己的抖动"隐含假设**梯度的抖动
                  沿着坐标轴**（它对每个坐标独立记账）；山谷一旋转，
                  陡/缓方向不再对齐坐标轴，这笔账记得就不那么准了。
                  Adam 的动量在旋转地形里先拐过弯，随后直线冲刺
                - **没有永远的第一**：正谷里 RMSProp 快、转个 30°
                  就被反超；真实损失面比任何玩具山谷都复杂——
                  所以工业界的答案不是"谁最强"，而是
                  **"Adam 最省心"**：不挑地形、lr 好调、
                  从 0.001 到 0.01 闭眼一配就能跑。大模型时代的
                  默认选择，是工程学的胜利，不是数学的胜利
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
                1. **mini-batch 三词**：batch size = 一步喂几个样本；
                   迭代 = 一次更新；epoch = 训练集过一遍
                   （迭代/epoch = ⌈n/batch⌉）。每轮先洗牌（10 课）；
                   损失除过 n，lr 与 batch 解耦。甜点区 16~256
                2. **SGD 之字的病根**：梯度垂直等高线 = 指向对岸的壁，
                   不指向谷底；一个 lr 伺候两个曲率（本课山谷 500:1）——
                   贴边 lr 之字不衰减（862 步），越界即指数发散
                3. **动量**：v ← βv + g、w ← w − lr·v；
                   之字分量变号自我抵消、谷向分量持续加速——
                   同 lr 下 862 → 43 步（20×）；β=0.9 是甜点
                   （0.99 惯性过重反要 258 步）
                4. **RMSProp / Adam**：s ← 0.9s + 0.1g²、除以 √s——
                   谁抖步长就给谁除小；Adam = 动量 + 平方账 +
                   偏差修正（m̂ = m/(1−0.9ᵗ)，只管起步）；
                   自适应选手的 lr 天生小一号
                5. **换优化器必须重调 lr**：真机对决里 SGD 用 0.5、
                   动量 0.2、Adam 0.02——量级完全不同；
                   batch 从 8 改 256 则基本不用动（除 n 的红利）
                6. **没有永远的第一**：正谷 RMSProp（29）最快、
                   旋转 30° 后 Adam（28）反超——工业界默认 Adam
                   不是因为它永远最快，是因为它最省心
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

        **下一课预告** — `25_初始化与训练循环`（第 4 部分收官）：

        还剩最后几块拼图，就把"深度学习训练学"的完整兵器谱凑齐：

        - **初始化**：全零权重 = 对称性灾难（所有隐单元活成一个），
          太大 = 饱和昏迷（19 课的伏笔）；Xavier / He 用一条
          方差守恒公式一劳永逸
        - **Dropout**：训练时随机让神经元下班——用"集体投票"
          对抗过拟合（17 课三件武器的新成员）
        - **早停 + weight decay**：17 课欠下的两个承诺今天兑现
        - **收官大战 · MNIST**：784-128-10、十万零一千七百七十个参数、
          今天出厂的 Adam——训练循环完全体首次实战，
          目标：**测试集 97%+**

        手写反向传播、手写优化器、手写正则化——三件套齐活之后，
        第 5 部分卷积神经网络开门，你将带着完整的手艺进图像的世界。

        🎆 兵器谱最后一块，装上它去打 MNIST！
        """
    )
    return


if __name__ == "__main__":
    app.run()
