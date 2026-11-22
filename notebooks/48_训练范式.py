import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="48 · 训练范式")


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
    import torch
    import torch.nn as nn

    print(f"PyTorch {torch.__version__}——最后一课")

    def load_mnist48():
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

    X_tr48, y_tr48, X_te48, y_te48 = load_mnist48()
    print(f"MNIST：训练 {X_tr48.shape[0]:,}｜测试 {X_te48.shape[0]:,}")
    return X_tr48, nn, torch, y_tr48


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # 48 · 训练范式：从手艺到工业

    **从零到 Transformer · 手写深度学习** ｜ 第 8 部分 · 过渡到 PyTorch（3/3 · 全课程收官）
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📖 本课目标（先读我）": mo.md(
                """
                **最后一课。零件全齐了：autograd（46）、
                nn.Module 与层（47）——现在装配成你以后
                每个项目都会用的骨架：工业标准训练
                范式。**

                25 课你手写过训练循环的雏形（打乱、切
                批、更新）；48 课把它升级成生产级：

                - **Dataset 与 DataLoader**：打乱切批的
                  工业化（多进程加载、自动分批）
                - **完整训练函数**：训练/验证切分、
                  train/eval 开关、学习率调度、最佳
                  模型存档、早停——五件套齐装
                - **train/eval 的双面性格**：Dropout
                  和 BatchNorm 为什么需要开关
                - **学习率调度器**：warmup 与衰减的
                  自动化（42 课 Post-LN 的 warmup 故事
                  在这里落地）

                然后是**全课程收官**：48 课、8 个部分、
                20+ 个亲手训练的模型——一张总表回望
                你走过的路，和一份"下一站"指南。

                学完你将能：

                - 独立写出任何 PyTorch 项目的训练骨架
                  （从数据到存档）
                - 解释范式中每一行的"为什么"——它们
                  全部对应你手写过的某段代码
                - 带着完整的心智模型走向真实项目：
                  读论文、复现、魔改
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📦 概念 1/3 · DataLoader：打乱切批的工业化": mo.md(
                """
                25 课手写的取数逻辑（每 epoch 打乱索引、
                按批切片）固化成两个对象：

                ```python
                dataset = TensorDataset(X, y)          # 样本-标签对
                loader = DataLoader(dataset, batch_size=128,
                                    shuffle=True)       # 打乱 + 切批 + 迭代
                for xb, yb in loader:                   # 25 课的手写循环 → 一行 for
                    ...
                ```

                它替你管的事：打乱（每 epoch 重洗）、
                分批（尾批自动处理）、搬运（pin_memory
                + 异步预取，GPU 场景提速明显）、多进程
                （num_workers：数据增强重的任务用）。

                更大的 Data 世界：TensorDataset 只是
                最简版——自定义 Dataset 实现
                `__len__` 和 `__getitem__` 就能接任意
                数据源（图片文件夹、数据库、网盘流），
                torchvision/HuggingFace 的数据集都走
                这个接口。
                """
            ),
            "🎭 概念 2/3 · train/eval：双面性格的开关": mo.md(
                """
                有些层在训练和推理时**行为不同**——它们
                需要 `model.train()` / `model.eval()`
                切换人格：

                - **Dropout**（25 课）：训练时随机丢弃
                  一半神经元（逼网络冗余）；推理时
                  **全员上岗 + 不丢弃**。忘了切 eval，
                  你的"验证精度"会莫测地抖
                - **BatchNorm**（25 课）：训练时用**当前
                  batch** 的均值方差（还要更新移动平均）；
                  推理时用**存档的移动平均**（batch
                  大小为 1 也能推理）。忘了切 eval，
                  推理结果依赖 batch 成分——工业事故
                  常客

                `model.eval()` 一行搞定所有层（沿着
                Module 树广播）；配合 `torch.no_grad()`
                是验证/测试的固定开场白：

                ```python
                model.eval()
                with torch.no_grad():
                    for xb, yb in val_loader: ...
                model.train()                          # 回训练人格
                ```

                注意：LayerNorm（42 课）**没有**双面
                性格——逐位置统计不依赖 batch，train/
                eval 一个样。这是 NLP 选它的又一票。
                """
            ),
            "🎚️ 概念 3/3 · 调度器与早停：训练的时间艺术": mo.md(
                """
                学习率不是常数——它在训练全程按剧本
                变化（调度器 scheduler）：

                ```python
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
                for epoch in range(10):
                    训练一个 epoch
                    sched.step()                       # 每回合更新学习率
                ```

                两大剧本（都对应你见过的现象）：

                - **衰减**（Step/Exponential/Cosine）：
                  大步找路、小步精修——24 课"学习率
                  是步长"的直觉：前期大步跳出坏区域，
                  后期小步落进谷底
                - **Warmup**（线性升温后再衰减）：
                  42 课的故——Post-LN 的 Transformer
                  初期不稳，前几千步从小学习率爬升
                  能救命；GPT 系训练的标配
                  （`get_linear_schedule_with_warmup`）

                **早停**（17 课）：验证损失不再下降就
                停 + 只存最佳模型。工业版两行：

                ```python
                if val_acc > best: torch.save(model.state_dict(), "best.pt")
                ```

                五件套（切分/开关/调度/存档/早停）
                全部就位——演示 2 一次性装配。
                """
            ),
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---

    ## 🎬 演示
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 演示 1 · DataLoader 初体验

    把 MNIST 装进 DataLoader，看它吐出的批次
    形状、打乱行为（两次 epoch 的首批不同）、
    尾批处理（60000 不是 128 的整数倍——最后
    一批只有 32 张）。
    """)
    return


@app.cell
def _(X_tr48, torch, y_tr48):
    # 演示 1 · DataLoader
    from torch.utils.data import TensorDataset, DataLoader

    _ds = TensorDataset(torch.tensor(X_tr48, dtype=torch.float32),
                        torch.tensor(y_tr48, dtype=torch.long))
    _loader = DataLoader(_ds, batch_size=128, shuffle=True)
    print(f"数据集大小：{len(_ds)}｜批大小 128 → 共 {len(_loader)} 批")
    _n = 0
    for _i, (_xb, _yb) in enumerate(_loader):
        _n += len(_xb)
        if _i in (0, 1, len(_loader) - 1):
            print(f"  批 {_i+1:>3}：X {tuple(_xb.shape)}｜y {tuple(_yb.shape)}"
                  f"{'（尾批：60000 不是 128 的整数倍）' if _i == len(_loader)-1 else ''}")
    print(f"合计 {_n} 张——25 课手写的打乱切批，三行搞定")

    _l1 = next(iter(_loader))[1][:5]
    _l2 = next(iter(_loader))[1][:5]
    print(f"\n两个 epoch 的首批前 5 个标签：{_l1.tolist()} vs {_l2.tolist()}（打乱生效）")
    return DataLoader, TensorDataset


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 演示 2 · 五件套齐装：生产级训练函数

    一个 ~40 行的 `train_full48`，装齐全部工业件：
    训练/验证切分、DataLoader、train/eval 开关、
    余弦调度、最佳存档、早停。每一行旁边标注
    它替代了你哪一课的手写代码。

    跑 3 个 epoch 看：训练/验证双曲线、学习率
    衰减轨迹、早停点。
    """)
    return


@app.cell
def _(DataLoader, TensorDataset, X_tr48, nn, plt, torch, y_tr48):
    # 演示 2 · 生产级训练函数（五件套）
    import time
    from copy import deepcopy

    def train_full48(model, X, y, epochs=3, batch=128, lr=1e-3, patience=2):
        """生产级训练：切分/加载/开关/调度/存档/早停。返回最佳模型和历史。"""
        _n_val = len(X) // 6                                  # ① 切分（17 课）
        _idx = torch.randperm(len(X))
        _tr_idx, _va_idx = _idx[_n_val:], _idx[:_n_val]
        _Xt = torch.tensor(X, dtype=torch.float32)
        _yt = torch.tensor(y, dtype=torch.long)
        _tr_ds = TensorDataset(_Xt[_tr_idx], _yt[_tr_idx])    # ② DataLoader（25 课）
        _va_ds = TensorDataset(_Xt[_va_idx], _yt[_va_idx])
        _tr_ld = DataLoader(_tr_ds, batch_size=batch, shuffle=True)
        _va_ld = DataLoader(_va_ds, batch_size=512)
        _opt = torch.optim.Adam(model.parameters(), lr=lr)     # 24 课配方
        _sch = torch.optim.lr_scheduler.CosineAnnealingLR(_opt, T_max=epochs)  # ③ 调度
        _lossf = nn.CrossEntropyLoss()
        _best_acc, _best_state, _bad = 0.0, None, 0
        _hist = {"tr": [], "va": [], "lr": []}
        for _ep in range(epochs):
            model.train()                                      # ④ 训练人格
            for _xb, _yb in _tr_ld:
                _opt.zero_grad()                               # 46 课：清累积
                _loss = _lossf(model(_xb), _yb)
                _loss.backward()
                _opt.step()
            model.eval()                                       # ⑤ 推理人格
            with torch.no_grad():                              # 46 课：不记图
                _acc = sum((model(_xb).argmax(1) == _yb).sum().item()
                           for _xb, _yb in _va_ld) / len(_va_ds)
            _hist["tr"].append(_loss.item())
            _hist["va"].append(_acc)
            _hist["lr"].append(_opt.param_groups[0]["lr"])
            print(f"  epoch {_ep+1}：验证准确率 {_acc:.4f}｜lr {_opt.param_groups[0]['lr']:.2e}")
            if _acc > _best_acc:                               # ⑥ 最佳存档（17 课早停）
                _best_acc, _bad = _acc, 0
                _best_state = deepcopy(model.state_dict())
            else:
                _bad += 1
                if _bad >= patience:
                    print(f"  早停 @ epoch {_ep+1}（验证 {patience} 回合不升）")
                    break
            _sch.step()
        model.load_state_dict(_best_state)
        return model, _hist, _best_acc

    torch.manual_seed(0)
    _net = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Dropout(0.2),
                         nn.Linear(256, 10))
    _t0 = time.time()
    _net, _hist48, _best48 = train_full48(_net, X_tr48, y_tr48, epochs=3)
    print(f"最佳验证准确率 {_best48:.4f}（含 Dropout——eval 模式下全员上岗）用时 {time.time()-_t0:.0f}s")

    _fig, _axes = plt.subplots(1, 2, figsize=(11.0, 3.8))
    _axes[0].plot(_hist48["va"], "o-", color="tab:green", label="验证准确率")
    _axes[0].set_xlabel("epoch")
    _axes[0].set_ylabel("准确率")
    _axes[0].set_title("验证曲线（每 epoch 用最佳存档保护）", fontsize=11)
    _axes[0].legend(fontsize=9)
    _axes[0].grid(True, alpha=0.3)
    _axes[1].plot(_hist48["lr"], "o-", color="tab:red")
    _axes[1].set_xlabel("epoch")
    _axes[1].set_ylabel("学习率")
    _axes[1].set_title("余弦调度：大步找路 → 小步精修", fontsize=11)
    _axes[1].grid(True, alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 演示 3 · 双面性格现场：Dropout 与 BatchNorm

    同一个张量喂两遍：train 模式下 Dropout 每次
    丢弃不同的神经元（两次输出不同）、BatchNorm
    用 batch 统计；eval 模式下 Dropout 恒等、
    BatchNorm 用存档统计——**两次输出完全相同**。

    这就是 25 课 Dropout 的"推理时全员上岗"和
    BN 的"移动平均"——当年嘴上说的，今天现场看。
    """)
    return


@app.cell
def _(nn, torch):
    # 演示 3 · train/eval 的双面性格
    torch.manual_seed(0)
    _drop = nn.Dropout(p=0.5)
    _x = torch.ones(1, 10)

    _drop.train()
    _o1 = _drop(_x)
    _o2 = _drop(_x)
    print(f"train 模式 Dropout(p=0.5)：两次前向")
    print(f"  第 1 次：{_o1.numpy().round(1)}")
    print(f"  第 2 次：{_o2.numpy().round(1)}")
    print(f"  → 每次丢弃不同的神经元，存活者 ×2 补偿（期望不变）")

    _drop.eval()
    _o3 = _drop(_x)
    _o4 = _drop(_x)
    print(f"\neval 模式：{_o3.numpy().round(1)}（恒等）——全员上岗、不丢弃、不缩放")

    # BatchNorm 的统计切换：移动平均随训练逼近真值
    _bn = nn.BatchNorm1d(4)
    _bn.train()
    _g = torch.Generator().manual_seed(1)
    for _i in range(1, 61):
        _xb = torch.randn(16, 4, generator=_g) * 3 + 5     # 每批：均值 5、方差 9
        _ = _bn(_xb)                                      # 训练：移动平均被逐步更新
        if _i in (1, 5, 20, 60):
            print(f"  喂完 {_i:2d} 批后，移动平均均值 ≈ {_bn.running_mean.numpy().round(2)}（真值 5）")
    _xb2 = torch.randn(16, 4) * 3 + 5
    _bn.eval()
    _z1 = _bn(_xb2[:2])
    _z2 = _bn(_xb2[:2])
    print(f"\neval 模式两次前向差异：{(_z1 - _z2).abs().max():.0e}（用存档统计，恒定）")
    print("→ 忘切 eval 的后果：推理结果依赖 batch 成分——工业事故常客")
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
    ### 交互实验 · 调度器试验台

    三个旋钮（初始学习率、总回合数、warmup
    比例），现场画出学习率剧本：先线性升温
    （warmup），再余弦衰减到接近零。

    玩法建议：

    - warmup=0：一步到位的余弦（传统视觉任务
      常用）
    - warmup=30%：GPT 训练的标配形状——
      42 课 Post-LN 的故事：初期不稳，慢慢热身
    - 初始 lr 拉到 0.1：体会"步长太大"的危险
      直觉（24 课：太大的步长在谷底来回跳）
    """)
    return


@app.cell
def _(mo):
    # 交互控制：调度器旋钮
    lr_slider48 = mo.ui.slider(0.0005, 0.01, value=0.002, step=0.0005, debounce=True,
                               label="🚀 初始学习率")
    ep_slider48 = mo.ui.slider(5, 50, value=20, step=1, debounce=True,
                               label="回合数")
    warm_slider48 = mo.ui.slider(0, 50, value=10, step=1, debounce=True,
                                 label="🔥 warmup 回合（%）")
    mo.hstack([lr_slider48, ep_slider48, warm_slider48])
    return ep_slider48, lr_slider48, warm_slider48


@app.cell
def _(ep_slider48, lr_slider48, mo, np, plt, warm_slider48):
    # ===== 交互实验 · 调度器试验台（响应单元）=====
    _lr0 = lr_slider48.value
    _eps = ep_slider48.value
    _wp = max(1, int(_eps * warm_slider48.value / 100))
    _xs = np.arange(1, _eps + 1)
    _ys = np.where(_xs <= _wp,
                   _lr0 * _xs / _wp,
                   _lr0 * 0.5 * (1 + np.cos(np.pi * (_xs - _wp) / (_eps - _wp))))

    _fig, _ax = plt.subplots(figsize=(8.8, 3.6))
    _ax.plot(_xs, _ys, lw=2, color="tab:red")
    _ax.fill_between(_xs, 0, _ys, alpha=0.12, color="tab:red")
    _ax.axvspan(1, _wp, alpha=0.15, color="tab:orange")
    _ax.text(_wp / 2, _lr0 * 0.5, "warmup\n（热身爬坡）", ha="center", fontsize=9,
             color="tab:orange")
    _ax.text(_wp + (_eps - _wp) * 0.5, _lr0 * 0.5, "余弦衰减\n（小步精修）",
             ha="center", fontsize=9, color="tab:red")
    _ax.set_xlabel("训练回合（epoch / step）")
    _ax.set_ylabel("学习率")
    _ax.set_title(f"学习率剧本：初始 {_lr0:.4f}，warmup {warm_slider48.value}% 后余弦衰减",
                  fontsize=11)
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()

    if warm_slider48.value < 5:
        _verdict = ("**无 warmup 的余弦**：视觉任务（CNN + 残差 + BN）的常用形状——"
                    "BN 的稳定性让大学习率开局也扛得住（42 课：LN/BN 都是稳定器）。")
    elif warm_slider48.value > 30:
        _verdict = ("**长 warmup**：Transformer 训练的标配——注意力塔初期对步长敏感"
                    "（42 课 Post-LN 的窄门效应），慢热身换后期稳定。GPT 系的 warmup "
                    "通常按 step 计（几千步），比例 1~5%——本试验台按 epoch 演示形状。")
    else:
        _verdict = ("**中庸剧本**：先热身再衰减。真实项目里这两个参数"
                    "（warmup 长度、衰减形状）是调参常客——好在它们不如学习率本身"
                    "敏感：24 课的直觉（步长 = 收敛速度/精度的权衡）在这里同样成立。")
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
                **练习 1 · 数批次**：60000 个样本、
                batch_size=128 的 DataLoader，一个
                epoch 有多少个批次？（60000/128 =
                937.5——想想尾批）

                **练习 2 · 补全训练五步**：一个批次的
                训练核心，按正确顺序补完五步（清梯度/
                前向/算损失/反向/更新）。

                **练习 3 · 双面性格**：Dropout 在
                eval 模式下做什么？（0=恒等不丢弃；
                1=继续随机丢弃）
                """
            )
        }
    )
    return


@app.cell
def _(mo):
    # ===== 练习 1 · 数批次 =====
    print("60000 个样本，batch_size=128")

    n_batch_ex1_48 = None                             # ← ① 一个 epoch 的批次数（整数）

    if n_batch_ex1_48 is not None:
        _n = -(-60000 // 128)                          # 向上取整
        if abs(n_batch_ex1_48 - _n) <= 1:
            _msg = (f"✅ 正确：{_n} 批（937 个整批 + 1 个 32 张的尾批）。"
                    "DataLoader 默认 drop_last=False：尾批照发（比 128 小）——"
                    "BatchNorm 对 32 张也够稳；但如果你的层对 batch 大小敏感"
                    "（BN + 太小尾批），可以 drop_last=True 扔掉它。"
                    "这些细节 25 课手写切批时你亲历过——只是那时没有选项。")
        else:
            _msg = "❌ 参考：938（⌈60000/128⌉）"
    else:
        _msg = "⏳ 填一个整数（向上取整）"
    mo.md(_msg)
    return


@app.cell
def _(mo, torch):
    # ===== 练习 2 · 补全训练五步 =====
    def train_step_ex2_48(model, lossf, opt, xb, yb):
        """一个批次的训练核心：五步按序补完。"""
        None                                          # ← ① 补：清梯度
        _logits = model(xb)                           # 前向（已给）
        _loss = lossf(_logits, yb)                    # 损失（已给）
        None                                          # ← ② 补：反向
        None                                          # ← ③ 补：更新参数
        return _loss

    step_ready_ex2_48 = None                          # ← 补完后，换成 train_step_ex2_48

    if step_ready_ex2_48 is not None:
        try:
            import torch.nn as _nn
            _m = _nn.Linear(4, 2)
            _opt = torch.optim.SGD(_m.parameters(), lr=0.1)
            _lf = _nn.CrossEntropyLoss()
            _xb = torch.randn(8, 4)
            _yb = torch.randint(0, 2, (8,))
            _w0 = _m.weight.detach().clone()
            _loss = step_ready_ex2_48(_m, _lf, _opt, _xb, _yb)
            _w1 = _m.weight.detach()
            if torch.equal(_w0, _w1):
                _msg = ("❌ 参数没动——五步里有漏（顺序：zero_grad → 前向 → 损失 → "
                        "backward → step）")
            elif _loss is not None and not torch.equal(_w0, _w1):
                _msg = ("✅ 五步齐活：zero_grad（清累积，46 课）→ 前向 → 损失 → "
                        "backward（自动微分）→ step（用梯度更新）。这五行就是你"
                        "以后每个项目的训练心脏——25 课的 40 行手写循环，工业上"
                        "浓缩成这五步，其余全是包装。")
            else:
                _msg = "❌ 返回值应为 _loss"
        except (TypeError, ValueError, RuntimeError) as _e:
            _msg = f"❌ 运行出错：{_e}"
    else:
        _msg = "⏳ 补三处：zero_grad / backward / step"
    mo.md(_msg)
    return


@app.cell
def _(mo):
    # ===== 练习 3 · 双面性格 =====
    print("Dropout 在 eval 模式下的行为")

    dropout_eval_ex3_48 = None                        # ← ①（0=恒等不丢弃，1=继续随机丢弃）

    if dropout_eval_ex3_48 is not None:
        if dropout_eval_ex3_48 == 0:
            _msg = ("✅ 正确：恒等。训练时丢弃是为了逼出冗余（25 课：任何一条路径"
                    "被砍，结论不变）；推理时网络已经练成了冗余——全员上岗、"
                    "直接用全量。train/eval 开关切的就是这层人格。顺带：BatchNorm"
                    " eval 时切换到移动平均统计（batch 大小为 1 也能推理）；"
                    "LayerNorm 没有人格分裂——逐位置统计与 batch 无关。")
        else:
            _msg = "❌ eval 时 Dropout 是恒等（不丢弃不缩放）"
    else:
        _msg = "⏳ 填 0 或 1"
    mo.md(_msg)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 1 参考答案（先自己写过再看！）": mo.md(
                """
                **938 批（⌈60000/128⌉）。**

                937 × 128 = 119,936——差的 64... 等等，
                60000 − 119,936 = ... 反正尾批 = 60000 −
                937×128 = 32 张。

                计算机科学的第一课在这里闪光：
                **除不尽就要决定尾批怎么办**。三种
                哲学：

                - 照发（默认）：尾批 32 张，BN 也能忍
                - 扔掉（drop_last=True）：统计最干净，
                  大数据时代不心疼
                - 凑整（padding）：序列任务偶尔用

                25 课手写切批时你选过第一种——当时
                没得选；现在你知道另外两条路在哪。
                """
            ),
            "✅ 练习 2 参考答案（先自己写过再看！）": mo.md(
                """
                ```python
                opt.zero_grad()          # ① 清梯度（防累积）
                _logits = model(xb)      # ② 前向（自动记录图）
                _loss = lossf(...)       # ③ 标量损失
                _loss.backward()         # ④ 反向（填 .grad）
                opt.step()               # ⑤ 用梯度更新
                ```

                顺序错了会怎样？

                - ①放到④后面：这一步的梯度会**加到**
                  上一步的残渣上（46 课演示的累积陷阱）
                - ⑤放到④前面：更新用的是**上一步的**
                  梯度（第一步直接报错：没梯度可step）
                - 忘了④：参数永远不动（loss 曲线
                  一条直线——新手 Top1 事故）

                五行代码，四个经典 bug 位——写过、
                踩过、修过，才算真的会。
                """
            ),
            "✅ 练习 3 参考答案（先自己写过再看！）": mo.md(
                """
                **0（恒等，不丢弃）。**

                Dropout 的完整人格表：

                | 模式 | 行为 | 为什么 |
                |---|---|---|
                | train | 随机置零 + 存活 ×(1/(1−p)) | 逼冗余；缩放保期望 |
                | eval | 恒等 | 冗余已练成，全量上岗 |

                同款双面性格的还有 BatchNorm
                （batch 统计 ↔ 移动平均）；而
                LayerNorm / 注意力 / FFN 都是
                单一人格。**养成肌肉记忆**：验证/
                测试/推理前 `model.eval()` + 
                `torch.no_grad()`，回来 `model.train()`——
                忘切是新手事故 Top3（症状：验证精度
                莫名其妙地抖）。
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
                1. **DataLoader**：TensorDataset +
                   DataLoader(batch, shuffle) 三行替代
                   25 课手写的打乱切批；自定义
                   Dataset 接任意数据源
                2. **训练五步**：zero_grad → 前向 →
                   损失 → backward → step——顺序
                   有讲究，四个经典 bug 位
                3. **train/eval 开关**：Dropout（随机
                   丢弃 ↔ 恒等）、BatchNorm（batch
                   统计 ↔ 移动平均）；LayerNorm 无
                   双面性格
                4. **验证的固定开场白**：model.eval() +
                   torch.no_grad()，回来 model.train()
                5. **调度器**：warmup（Transformer
                   的慢热身，42 课）+ 余弦/阶梯衰减
                   （大步找路小步精修，24 课）
                6. **五件套**：切分/加载/开关/调度/
                   存档+早停——演示 2 的 40 行就是
                   你未来每个项目的骨架
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
                ## 🎓 全课程收官：从零到 Transformer，然后手里有了工业界的钥匙

                48 课、8 个部分、20 多个亲手训练的模型。回望来路：

                | 部分 | 课 | 你获得了什么 |
                |---|---|---|
                | **1 NumPy 内功** | 01-10 | ndarray、广播、axis——一切深度学习的载体 |
                | **2 数学直觉** | 11-14 | 点积的几何、矩阵即变换、梯度、链式法则 |
                | **3 线性模型** | 15-17 | 第一个回归/分类器、softmax+交叉熵、过拟合与早停 |
                | **4 多层网络** | 18-25 | MLP、激活函数、反向传播两课时、优化器、训练循环 |
                | **5 卷积网络** | 26-31 | 权值共享、im2col 手写卷积、LeNet-5 到 98.5% |
                | **6 循环网络** | 32-37 | RNN/LSTM、BPTT、梯度裁剪、字符级语言模型 |
                | **7 Transformer** | 38-45 | 注意力六件套、手写编码器与迷你 GPT、GPT/BERT 地图 |
                | **8 工业过渡** | 46-48 | autograd、nn.Module、生产级训练范式 |

                三样东西你现在有了，而三个月前没有：

                - **心智模型**：从点积到注意力，每个
                  公式你都用 NumPy 摸过温度——读任何
                  论文，看到的不是黑话而是老朋友
                - **调试能力**：你手写过反向传播的
                  bug（43 课的教训）、踩过梯度消失
                  （35 课）、见过冷启动的躺平（40 课）
                  ——生产事故在你眼里有名字
                - **迁移本能**：torch 的每个 API 你都能
                  对应到自己写过的版本——框架是加速器，
                  不是拐杖
                """
            ),
            mo.callout(
                mo.md(
                    """
                    **下一站**（按兴趣选一条路走深）：

                    - **读论文**：Attention is All You Need
                      → GPT-2/3 → Chinchilla → InstructGPT——
                      现在的你读它们像读组装说明书
                    - **复现**：Karpathy 的 nanoGPT（约 300 行
                      训一个字符级 GPT——你已经写过它的每个
                      零件，这次是全须全尾的工程版）
                    - **做一个自己的项目**：小数据集 + 你
                      48 课的骨架——分类、检索、微调一个小
                      开源模型（LLaMA 系）
                    - **补纵深**：概率与信息论（交叉熵的
                      另一半）、优化理论（Adam 为什么 work）、
                      可解释性（43 课"路由器"的续集）

                    深度学习没有毕业——只有"下一座山"。
                    但你现在有地图、有绳索、有自己的
                    双手。

                    从零到 Transformer：完成。
                    """
                ),
                kind="success",
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
