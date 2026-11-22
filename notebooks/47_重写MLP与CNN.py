import marimo

__generated_with = "0.25.2"
app = marimo.App(app_title="47 · 重写MLP与CNN", width="medium")


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
def _(mo, np):
    import os
    from pathlib import Path
    import gzip
    import torch

    print(f"PyTorch {torch.__version__}")

    # ==== MNIST 加载器（25/30 课同款：优先 data/mnist.npz 缓存）====
    def load_mnist47():
        _cands = []
        try:
            _nb = mo.notebook_location()
            if _nb is not None:
                _cands += [Path(_nb).parent.parent / "data", Path(_nb).parent / "data"]
        except Exception:
            pass
        _cands += [Path("../data"), Path("data"),
                   Path("/home/z/my-project/download/deep-learning-course/data")]
        _cache = next((_c / "mnist.npz" for _c in _cands if (_c / "mnist.npz").exists()), None)
        if _cache is None:
            raise RuntimeError("找不到 data/mnist.npz——请先运行 25/26/30 课任一课生成缓存")
        with np.load(_cache) as _z:
            return _z["X_tr"], _z["y_tr"], _z["X_te"], _z["y_te"]

    X_tr47, y_tr47, X_te47, y_te47 = load_mnist47()
    print(f"MNIST：训练 {X_tr47.shape[0]:,} 张｜测试 {X_te47.shape[0]:,} 张｜"
          f"每张 {int(np.sqrt(X_tr47.shape[1]))}×{int(np.sqrt(X_tr47.shape[1]))} 像素")
    return (X_te47, X_tr47, torch, y_te47, y_tr47)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # 47 · 重写 MLP 与 CNN：工业姿势

        **从零到 Transformer · 手写深度学习** ｜ 第 8 部分 · 过渡到 PyTorch（2/3）
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📖 本课目标（先读我）": mo.md(
                """
                **46 课的 autograd 解放了反向传播，但代码
                还是"NumPy 的姿势"——手动建参数、手动
                写前向。这一课换成工业姿势，然后重写
                两台老机器。**

                新姿势的三层抽象：

                - **nn.Module**：把参数和子层登记进
                  一棵树，`.parameters()` 一网打尽——
                  优化器再也不用手动传参数清单
                - **nn.Linear / nn.Conv2d**：你在 20/29 课
                  手写过的层，一行一个，前向反向全免
                - **CrossEntropyLoss / Adam**：21/24 课的
                  配方，各一行

                然后重写两台老机器并逐项对表：

                - **25 课 MLP**（MNIST 分类器）：手写
                  ~120 行 → torch ~25 行
                - **30 课 LeNet-5**（CNN 实战）：手写
                  ~200 行 → torch ~40 行

                学完你将能：

                - 写标准 nn.Module（`__init__` 登记、
                  `forward` 前向、`.to(device)` 搬家）
                - 用 `nn.Sequential` 搭快速原型
                - 说清 torch 版为什么又短又快还不
                  容易错（参数登记自动化 + 数值稳定
                  实现 + GPU 后端）
                - 掌握 `.to(device)` 的搬家协议：CPU/
                  GPU 间迁移模型与数据的两行代码

                前置：46 课 autograd、20 课 MLP、
                29-30 课卷积与 LeNet、25 课训练循环。

                本课两台机器都会真训练（CPU 数分钟）——
                训完的模型直接进 48 课的训练范式。
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "🧩 概念 1/3 · nn.Module：参数的自动登记": mo.md(
                """
                一个 Module 就是"带参数的函数"：

                ```python
                class MyLinear(nn.Module):
                    def __init__(self, d_in, d_out):
                        super().__init__()              # 必写：初始化登记簿
                        self.W = nn.Parameter(torch.randn(d_out, d_in) * 0.05)
                        self.b = nn.Parameter(torch.zeros(d_out))
                    def forward(self, x):
                        return x @ self.W.T + self.b    # 只写前向！
                ```

                魔法在 `nn.Parameter`：把它挂到 `self`
                上的一瞬间，Module 的登记簿记下这个
                张量——于是：

                - `model.parameters()` 自动列出全部参数
                  （喂给优化器，一行）
                - `model.to("cuda")` 自动搬全部参数
                - `model.state_dict()` 自动存档/读档
                - 反向传播自动覆盖（46 课）

                子层也是 Module（Module 套 Module 是
                一棵树）：`self.net = nn.Sequential(...)`
                挂上来，里面所有参数一起被登记——
                **你 25 课手动维护的参数清单
                （plist），这棵树全包了**。
                """
            ),
            "🏭 概念 2/3 · 造好的轮子：层、损失、优化器": mo.md(
                """
                你手写过的每一样，torch 都有一个对应
                的工业件（数值更稳、更快、GPU 就绪）：

                | 你手写的（课） | torch 工业件 | 备注 |
                |---|---|---|
                | 线性层（15/20） | `nn.Linear(d_in, d_out)` | 含权重和偏置 |
                | ReLU/激活（19） | `nn.ReLU()` / `nn.GELU()` | 无参数 |
                | softmax+交叉熵（21） | `nn.CrossEntropyLoss()` | 合并计算更稳定 |
                | 卷积层（29） | `nn.Conv2d(入,出,核)` | im2col 内置 |
                | 池化（28） | `nn.MaxPool2d(2)` | 无参数 |
                | 展平 | `nn.Flatten()` | (n,c,h,w)→(n,c·h·w) |
                | SGD/Momentum/Adam（24） | `torch.optim.Adam(params, lr)` | 一行 |
                | 多头注意力（40） | `nn.MultiheadAttention` | 43 课的块 |

                两个易错点（都踩过才记得住）：

                - **CrossEntropyLoss 自带 softmax**：
                  喂**原始 logits**，别再手动 softmax
                  （21 课的数值稳定版它内部处理了）
                - **数据形状**：Conv2d 要 (n, 通道,
                  高, 宽) 的 4 维张量——25 课拉平的
                  (n, 784) 要 `view(-1, 1, 28, 28)`
                  展开回来
                """
            ),
            "🚚 概念 3/3 · .to(device)：搬家协议": mo.md(
                """
                模型和数据可以住在 CPU 或 GPU，**但一次
                计算的所有参与者必须住在同一处**：

                ```python
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = MyNet().to(device)          # 搬模型（参数全走）
                X, y = X.to(device), y.to(device)   # 搬数据（每批都要）
                ```

                三个工程要点：

                - `.to(device)` 对 Module 是**原地**
                  （返回值也是它自己），对张量是**新
                  张量**（必须接住：`X = X.to(...)`）
                - 忘搬数据会报错（算子发现两个设备）
                  ——报错是好事；怕的是 GPU 闲置
                  却以为在加速
                - 本课程全程 CPU 可跑：**同一份代码**，
                  有 GPU 就自动用——这就是"可移植"
                  的含义（30 课 LeNet：CPU 二十分钟
                  → GPU 二十秒，代码只多这两行）
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
        ### 演示 1 · nn.Module 初体验：手写 Linear vs nn.Linear

        左边是你 15 课姿势的 torch 翻版（手写参数 +
        autograd），右边是 `nn.Linear`——前向输出
        逐位一致、梯度逐位一致。看 `parameters()`
        怎么把参数自动列出来：手动版的"参数清单"
        从此退休。
        """
    )
    return


@app.cell
def _(torch):
    # 演示 1 · 手写 Linear vs nn.Linear
    import torch.nn as nn

    torch.manual_seed(0)

    # ---- 手写版（15 课姿势 + 46 课 autograd）----
    class MyLinear47(nn.Module):
        def __init__(self, d_in, d_out):
            super().__init__()
            self.W = nn.Parameter(torch.randn(d_out, d_in) * 0.05)
            self.b = nn.Parameter(torch.zeros(d_out))
        def forward(self, x):
            return x @ self.W.T + self.b

    _mine = MyLinear47(6, 3)
    _builtin = nn.Linear(6, 3)
    # 参数对齐（把内置版的权重复制进手写版）
    with torch.no_grad():
        _mine.W.copy_(_builtin.weight)
        _mine.b.copy_(_builtin.bias)

    _X = torch.randn(5, 6)
    _y1, _y2 = _mine(_X), _builtin(_X)                    # __call__ 自动调 forward
    _l1, _l2 = _y1.square().mean(), _y2.square().mean()
    _l1.backward(); _l2.backward()
    print(f"前向差异：{(_y1 - _y2).abs().max():.1e}")
    print(f"梯度差异（W）：{(_mine.W.grad - _builtin.weight.grad).abs().max():.1e}")
    print(f"\nparameters() 自动登记：")
    for _nm, _p in _mine.named_parameters():
        print(f"  {_nm}: {_p.shape}（共 {_p.numel()} 个数）")
    print(f"合计：{sum(_p.numel() for _p in _mine.parameters())} 个参数——优化器一行全拿走")
    return (nn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 2 · 重写 25 课 MLP：120 行 → 25 行

        同一台机器（784 → 128 → 10，ReLU，Adam），
        同一份 MNIST。torch 版代码量 1/5，训练更快
        （BLAS 优化 + 自动微分开销远小于手写 Python
        循环），精度对表 25 课的手写版。

        两个"翻译"细节：`CrossEntropyLoss` 吃 logits
        （别手动 softmax）；每个 batch 手动
        `zero_grad`（46 课的纪律）。
        """
    )
    return


@app.cell
def _(X_te47, X_tr47, nn, plt, torch, y_te47, y_tr47):
    # 演示 2 · torch 版 MLP（25 课复刻）
    import time

    torch.manual_seed(0)
    _dev = "cpu"                                          # 有 GPU 自动切换：cuda if available

    class MLP47(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(784, 128), nn.ReLU(),
                nn.Linear(128, 10))
        def forward(self, x):
            return self.net(x)

    _mlp = MLP47().to(_dev)
    mlp47 = _mlp
    _Xtr = torch.tensor(X_tr47, dtype=torch.float32)
    _ytr = torch.tensor(y_tr47, dtype=torch.long)
    _Xte = torch.tensor(X_te47, dtype=torch.float32)
    _yte = torch.tensor(y_te47, dtype=torch.long)
    _opt = torch.optim.Adam(_mlp.parameters(), lr=1e-3)
    _lossf = nn.CrossEntropyLoss()

    _t0 = time.time()
    _hist47 = []
    _B = 128
    for _ep in range(3):
        _pm = torch.randperm(len(_Xtr))
        for _i in range(0, len(_Xtr), _B):
            _xb = _Xtr[_pm[_i:_i+_B]].to(_dev)
            _yb = _ytr[_pm[_i:_i+_B]].to(_dev)
            _opt.zero_grad()
            _loss = _lossf(_mlp(_xb), _yb)
            _loss.backward()
            _opt.step()
        with torch.no_grad():
            _acc = (_mlp(_Xte).argmax(1) == _yte).float().mean().item()
        _hist47.append(_acc)
        print(f"epoch {_ep+1}：测试准确率 {_acc:.4f}（累计 {time.time()-_t0:.0f}s）")

    with torch.no_grad():
        _acc_mlp47 = (_mlp(_Xte).argmax(1) == _yte).float().mean().item()
    _n_par = sum(_p.numel() for _p in _mlp.parameters())
    print(f"\n最终测试准确率 {_acc_mlp47:.4f}｜参数 {_n_par:,} 个")
    print("对照 25 课手写版（同结构同数据、训到底）：约 97.5%——同一台机器，另一种姿势")

    _fig, _axes = plt.subplots(1, 8, figsize=(11.0, 2.2))
    with torch.no_grad():
        _errs = (_mlp(_Xte).argmax(1) != _yte).nonzero().flatten()[:8]
    for _k, _idx in enumerate(_errs):
        _axes[_k].imshow(X_te47[_idx].reshape(28, 28), cmap="gray")
        _axes[_k].axis("off")
        _axes[_k].set_title(f"错例\n预测{_mlp(_Xte[_idx:_idx+1]).argmax().item()}"
                            f"/真{int(y_te47[_idx])}", fontsize=8)
    _fig.suptitle(f"MLP 的前 8 个错例（多数是笔画潦草的难样本——与 25 课观察一致）", fontsize=10)
    _fig.tight_layout()
    _fig
    return (MLP47, _acc_mlp47, mlp47)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 3 · 重写 30 课 LeNet-5：卷积来了

        29 课你手写了 im2col + 卷积反向；30 课用它
        搭了 LeNet-5。torch 版：`nn.Conv2d` 一行一个
        卷积层，`nn.MaxPool2d` 一行一个池化——**手写
        200 行的机器缩成 20 行**。

        训练 1 个 epoch（CPU 约 10 秒）看准确率
        爬到 97%+——卷积的平移容忍（28 课）+
        工业实现的速度（30 课手写版跑同样内容
        要慢得多）。
        """
    )
    return


@app.cell
def _(X_te47, X_tr47, nn, plt, torch, y_te47, y_tr47):
    # 演示 3 · torch 版 LeNet-5（30 课复刻）
    import time

    torch.manual_seed(0)

    class LeNet47(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 6, 5, padding=2), nn.ReLU(),      # 28×28 → 6×28×28
                nn.MaxPool2d(2),                                 # → 6×14×14
                nn.Conv2d(6, 16, 5), nn.ReLU(),                  # → 16×10×10
                nn.MaxPool2d(2),                                 # → 16×5×5
                nn.Flatten(),                                    # → 400
                nn.Linear(16*5*5, 120), nn.ReLU(),
                nn.Linear(120, 84), nn.ReLU(),
                nn.Linear(84, 10))
        def forward(self, x):
            return self.net(x)

    _lenet = LeNet47()
    lenet47 = _lenet
    _Xtr4 = torch.tensor(X_tr47, dtype=torch.float32).view(-1, 1, 28, 28)
    _ytr = torch.tensor(y_tr47, dtype=torch.long)
    _Xte4 = torch.tensor(X_te47, dtype=torch.float32).view(-1, 1, 28, 28)
    _yte = torch.tensor(y_te47, dtype=torch.long)
    _opt = torch.optim.Adam(_lenet.parameters(), lr=1e-3)
    _lossf = nn.CrossEntropyLoss()

    _t0 = time.time()
    _B = 128
    for _ep in range(1):
        _pm = torch.randperm(len(_Xtr4))
        for _i in range(0, len(_Xtr4), _B):
            _xb = _Xtr4[_pm[_i:_i+_B]]
            _yb = _ytr[_pm[_i:_i+_B]]
            _opt.zero_grad()
            _loss = _lossf(_lenet(_xb), _yb)
            _loss.backward()
            _opt.step()
    _dt = time.time() - _t0
    with torch.no_grad():
        _acc_lenet47 = (_lenet(_Xte4).argmax(1) == _yte).float().mean().item()
    _n_par = sum(_p.numel() for _p in _lenet.parameters())
    print(f"LeNet-5 训练 1 epoch（CPU {_dt:.0f}s）：测试准确率 {_acc_lenet47:.4f}")
    print(f"参数 {_n_par:,} 个（30 课手写版同款结构：61,706）")
    print("对照：30 课手写版数个 epoch 后 98.5%——这里 1 epoch 就到 97%+（Adam + 工业实现）")

    # 第一个卷积核可视化（29 课的老朋友）
    _kernels = _lenet.net[0].weight.detach().reshape(6, 5, 5)
    _fig, _axes = plt.subplots(1, 6, figsize=(10.5, 2.4))
    for _k in range(6):
        _axes[_k].imshow(_kernels[_k], cmap="RdBu_r")
        _axes[_k].axis("off")
        _axes[_k].set_title(f"核 {_k+1}", fontsize=9)
    _fig.suptitle("第一层卷积核（训后）：边缘/斑点检测器——30 课手写版看到过同款", fontsize=10.5)
    _fig.tight_layout()
    _fig
    return (_acc_lenet47, lenet47)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 4 · 搬家协议：同一份代码，有 GPU 就加速

        `.to(device)` 的标准写法（本机无 GPU 则自动
        回落 CPU——**课程代码天然可移植**）。看两段
        对比：不搬家的代码在有 GPU 的机器上照样能
        跑（CPU 上），搬了家的同一段代码直接吃
        GPU——**零改动**。

        顺带预览 `state_dict`：训好的模型存档/读档
        各一行（30 课手写版要自己拼 npz）。
        """
    )
    return


@app.cell
def _(MLP47, nn, torch):
    # 演示 4 · device 搬家 + 存档读档
    import os

    _dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"本机设备：{_dev}"
          + ("（有 GPU！同一段代码自动加速）" if _dev == "cuda"
             else "（无 GPU——代码在有 GPU 的机器上零改动加速）"))

    _m = MLP47().to(_dev)
    _x = torch.randn(4, 784).to(_dev)
    _y = _m(_x)
    print(f"前向成功：输入在 {_x.device}，输出在 {_y.device}——模型与数据同处一地")

    # 存档 / 读档
    _path = "mlp47_demo.pt"
    torch.save(_m.state_dict(), _path)
    _m2 = MLP47()
    _m2.load_state_dict(torch.load(_path, weights_only=True))
    _m2.eval()
    os.remove(_path)
    print("state_dict 存档/读档：各一行（参数名字典——25 课的 npz 手工版退休）")
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
        ### 交互实验 · 预测工作台

        从测试集挑一张图，看两台机器（MLP vs
        LeNet-5）各自的预测概率条形图——**同一个
        输入，两种看法**。

        玩法建议：

        - 挑几张"难"的（笔画潦草的 4/7/9），看两台
          机器怎么犹豫
        - LeNet 错的图 MLP 大概率也错——但反过来
          不一定（卷积的平移容忍是实打实的优势，
          28 课的理论在兑现）
        - 对比两者的置信度形态：错的时候谁更
          "嘴硬"？（过自信是深度模型的通病，
          温度校准是另一门课的故事）
        """
    )
    return


@app.cell
def _(mo):
    # 交互控制：测试集索引
    idx_slider47 = mo.ui.slider(0, 9999, value=0, step=1, debounce=True,
                                label="🖼️ 测试集图片编号")
    idx_slider47
    return (idx_slider47,)


@app.cell
def _(mlp47, X_te47, idx_slider47, mo, plt, torch, y_te47, lenet47):
    # ===== 交互实验 · 预测工作台（响应单元）=====
    _i = idx_slider47.value
    _x1 = torch.tensor(X_te47[_i:_i+1], dtype=torch.float32)
    _x4 = _x1.view(-1, 1, 28, 28)
    _true = int(y_te47[_i])
    with torch.no_grad():
        _p1 = torch.softmax(mlp47(_x1), dim=-1)[0]
        _p2 = torch.softmax(lenet47(_x4), dim=-1)[0]

    _fig, _axes = plt.subplots(1, 3, figsize=(12.6, 3.6), gridspec_kw={"width_ratios": [1, 2, 2]})
    _axes[0].imshow(X_te47[_i].reshape(28, 28), cmap="gray")
    _axes[0].axis("off")
    _axes[0].set_title(f"真值：{_true}", fontsize=12)
    for _ax, (_p, _nm, _c) in zip(_axes[1:], [(_p1, "MLP（25 课重写）", "tab:blue"),
                                               (_p2, "LeNet-5（30 课重写）", "tab:green")]):
        _bars = _ax.bar(range(10), _p.numpy(), color=_c, alpha=0.8)
        _pred = int(_p.argmax())
        _bars[_pred].set_color("tab:red")
        _ax.set_xticks(range(10))
        _ax.set_ylim(0, 1.05)
        _ax.set_title(f"{_nm}：预测 {_pred}（{_p[_pred]:.2f}）"
                      f"{' ✓' if _pred == _true else ' ✗'}", fontsize=10.5)
        _ax.grid(True, axis="y", alpha=0.3)
    _fig.tight_layout()

    if int(_p1.argmax()) == _true == int(_p2.argmax()):
        _verdict = "两台机器都对。看置信度形态——两家对简单数字都很笃定。"
    elif int(_p1.argmax()) != _true and int(_p2.argmax()) != _true:
        _verdict = ("两台机器都错（且错得相似）——这类样本通常笔画本身就有歧义，"
                    "人也要看两眼。换几张试试。")
    else:
        _winner = "LeNet-5" if int(_p2.argmax()) == _true else "MLP"
        _verdict = (f"只有 {_winner} 对。卷积的平移容忍（28 课）经常在这样的"
                    "偏移/变形样本上兑现成准确率——MLP 只看像素排列，"
                    "位置一挪就抓瞎（26 课三宗罪的现场版）。")
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
                **练习 1 · 数参数**：`nn.Linear(784, 128)`
                一共多少个参数（含偏置）？

                **练习 2 · 补全一个 Module**：写一个
                两层 MLP 的 forward——注册好的层在
                `self.fc1 / self.fc2`，激活用 ReLU。

                **练习 3 · 数卷积参数**：
                `nn.Conv2d(1, 6, 5)`（1 通道入、
                6 通道出、5×5 核）多少个参数（含
                偏置）？对比 `nn.Linear(784, 128)`：
                卷积的参数效率高在哪？
                """
            )
        }
    )
    return


@app.cell
def _(mo):
    # ===== 练习 1 · 数参数 =====
    print("nn.Linear(784, 128)")

    n_par_ex1_47 = None                               # ← ① 参数量（含偏置，整数）

    if n_par_ex1_47 is not None:
        _n = 784 * 128 + 128
        if abs(n_par_ex1_47 - _n) < 10:
            _msg = (f"✅ 正确：{_n:,}。nn.Linear 的 weight 是 (128, 784)、bias 是 (128,)——"
                    "20 课手写版的 W 和 b 一模一样，只是登记方式从 plist 手动清单"
                    "换成了 parameters() 自动树。全连接的参数量 = 入×出+出："
                    "维度一大就爆炸（26 课三宗罪的第一宗）。")
        else:
            _msg = f"❌ 参考：784×128+128 = {_n:,}"
    else:
        _msg = "⏳ 填一个整数（入×出+出）"
    mo.md(_msg)
    return (n_par_ex1_47,)


@app.cell
def _(mo, torch):
    # ===== 练习 2 · 补全 Module =====
    import torch.nn as nn

    def make_mlp_ex2_47():
        """返回一个 (784→64→10) 的 MLP Module（forward 补全）。"""
        class _MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(784, 64)
                self.fc2 = nn.Linear(64, 10)
            def forward(self, x):
                return None                           # ← 补：fc1 → ReLU → fc2
        return _MLP()

    mlp_ready_ex2_47 = None                           # ← 补完后，换成 make_mlp_ex2_47

    if mlp_ready_ex2_47 is not None:
        try:
            _m = mlp_ready_ex2_47()
            _x = torch.randn(3, 784)
            _y = _m(_x)
            if _y is not None and _y.shape == (3, 10):
                _n = sum(_p.numel() for _p in _m.parameters())
                _msg = (f"✅ 正确：输出 {_y.shape}，参数 {_n:,}。forward 里只写前向——"
                        "激活夹在两层之间（20 课的图原样翻译）。nn.ReLU() 无参数，"
                        f"所以 parameters() 只数到两层 Linear（{_n:,} = 784×64+64+64×10+10）。"
                        "反向传播：不存在（46 课的 autograd 在背后）。")
            else:
                _msg = "❌ 输出应为 (3, 10)（fc1 → relu → fc2）"
        except (TypeError, ValueError, RuntimeError) as _e:
            _msg = f"❌ 运行出错：{_e}——torch.relu(self.fc1(x)) 后接 self.fc2"
    else:
        _msg = "⏳ 补 forward 一行"
    mo.md(_msg)
    return (make_mlp_ex2_47, mlp_ready_ex2_47)


@app.cell
def _(mo):
    # ===== 练习 3 · 数卷积参数 =====
    print("nn.Conv2d(1, 6, 5)：1 通道入、6 通道出、5×5 卷积核")

    n_conv_ex3_47 = None                              # ← ① 参数量（含偏置，整数）

    if n_conv_ex3_47 is not None:
        _n = 1 * 6 * 5 * 5 + 6
        if abs(n_conv_ex3_47 - _n) < 5:
            _msg = (f"✅ 正确：{_n}（6 个 5×5 核 × 1 输入通道 + 6 个偏置）。对比 "
                    "Linear(784,128) 的 100,480：卷积层只有 156 个参数，却扫过整张图——"
                    "**权值共享**（26 课）：同一个核在所有位置复用。这就是 CNN 参数效率"
                    "的全部秘密：参数量由（核大小×通道数）决定，与图像尺寸无关。")
        else:
            _msg = f"❌ 参考：1×6×5×5+6 = 156"
    else:
        _msg = "⏳ 填一个整数（核数×核大小+核数）"
    mo.md(_msg)
    return (n_conv_ex3_47,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 1 参考答案（先自己写过再看！）": mo.md(
                """
                **784 × 128 + 128 = 100,480。**

                对齐 25 课：那台手写 MLP 的两层是
                784×128+128 和 128×10+10，合计
                101,770——本课 torch 版 `parameters()`
                数出来的正是这个数。**同一个数字，
                两种人生**：手写版你手动 plist、手动
                反向 120 行；torch 版登记自动、反向
                自动，训练循环 8 行。

                这 120 行没有白写：你现在能解释
                `nn.Linear` 里每个数字的来历，能在
                它出 bug 时打开引擎盖——只会调包的
                工程师和懂原理的工程师，差的就是
                这 120 行。
                """
            ),
            "✅ 练习 2 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                return self.fc2(torch.relu(self.fc1(x)))
                ```

                三种等价写法（都常见）：

                - 函数式：`torch.relu(self.fc1(x))`
                  （本答案）
                - 模块式：`self.act = nn.ReLU()` 在
                  `__init__` 登记，forward 用
                  `self.act(...)`
                - Sequential：`nn.Sequential(fc1, nn.ReLU(), fc2)`
                  整段打包，forward 只 `return self.net(x)`

                ReLU 无参数，函数式和模块式在此
                完全等价；有参数的层（Dropout、
                BN）必须用模块式——它们要在
                train/eval 模式间切换行为（48 课）。
                """
            ),
            "✅ 练习 3 参考答案（先自己写过再看！）": mo.md(
                """
                **1×6×5×5 + 6 = 156。**

                参数公式：`出通道 × 入通道 × 核高 ×
                核宽 + 出通道`。与图像大小无关——
                28×28 的图和 224×224 的图，这一层
                都是 156 个参数（权值共享，26 课）。

                两层对比的深意：

                | 层 | 参数 | 感受的输入 |
                |---|---|---|
                | Linear(784, 128) | 100,480 | 固定 784 像素的排列 |
                | Conv2d(1, 6, 5) | 156 | 任意尺寸图像的局部图案 |

                参数差 644 倍，表达力各有千秋：全连接
                记"全局模板"，卷积记"局部图案 +
                到处复用"。视觉任务里后者压倒性划算
                ——26 课的结论，工业界用 30 课的
                LeNet 到今天的 ResNet 一路验证。
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
                1. **nn.Module**：nn.Parameter 挂到
                   self 即自动登记——parameters()/
                   to(device)/state_dict() 全部自动化；
                   手动参数清单退休
                2. **只写 forward**：反向 46 课的
                   autograd 管；子层套子层是一棵树
                3. **轮子对照表**：Linear(15/20)、
                   ReLU(19)、Conv2d(29)、MaxPool2d(28)、
                   CrossEntropyLoss(21，吃 logits 别
                   手动 softmax)、Adam(24)
                4. **重写对表**：25 课 MLP 120 行→25 行
                   同精度；30 课 LeNet-5 200 行→20 行，
                   1 epoch 过 98%
                5. **形状纪律**：Conv2d 要 4 维
                   (n,通道,高,宽)——拉平的 784 要
                   view 回来；CrossEntropyLoss 要
                   long 型标签
                6. **搬家协议**：.to(device) 模块原地、
                   张量要接住；同一份代码 CPU/GPU
                   零改动——可移植性的全部含义
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

                **下一课预告** — `48_训练范式`（第 8 部分 · 过渡到 PyTorch）：

                零件全齐了：autograd（46）、nn.Module
                与层（47）。最后一课把它们装配成
                **工业标准的训练范式**——你以后写的
                每一个项目，骨架都是这一课：

                - **Dataset 与 DataLoader**：把 25 课
                  手写的"打乱 + 切批"上升为可迭代
                  对象（多进程加载、自动整理批次）
                - **完整训练循环**：zero_grad → 前向 →
                  loss → backward → step → 验证
                  （no_grad）——每一行都能对应到
                  25 课手写版的某一句
                - **train/eval 模式**：Dropout 和
                  BatchNorm 的双面性格为什么需要
                  开关
                - **学习率调度器**：warmup + 衰减自动化
                - **收官总结**：48 课 × 8 部分的全景
                  回顾，和你的下一站

                这也是全课程的最后一课——从 01 课
                "为什么需要 NumPy" 到今天，48 课的
                环在这里闭合。
                """
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
