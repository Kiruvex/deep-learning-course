import marimo

__generated_with = "0.25.2"
app = marimo.App(app_title="45 · GPT与BERT概览", width="medium")


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
        # 45 · GPT 与 BERT 概览：一张工业界的地图

        **从零到 Transformer · 手写深度学习** ｜ 第 7 部分 · Transformer（8/8 · 收官）
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📖 本课目标（先读我）": mo.md(
                """
                **你已经亲手造过一台迷你 GPT 了——这一课
                讲清楚这场革命是怎么发生的。**

                38-44 课你写的每一块积木（缩放点积注意力、
                多头、位置编码、Pre-LN 块、因果掩码），
                与 GPT-4 里的**同名同姓、同一公式**——
                差的只是规模。收官一课把这份手艺放回
                工业界的坐标系：

                - **时间线 2017-2023**：从一篇翻译论文
                  到 ChatGPT，六年间发生了什么
                - **三种架构、一个零件**：BERT（理解）、
                  GPT（生成）、T5（翻译）——全是你的块
                - **Scaling Laws**：44 课迷你 GPT 输给
                  LSTM 的那笔账，规模怎么把它赢回来
                - **预训练目标**：完形填空（BERT）vs
                  预测下一词（GPT）——两种童年，两种
                  性格
                - **ChatGPT 的最后三步**：预训练之上，
                  指令微调和人类反馈（RLHF）补了什么

                学完你将能：

                - 看懂任何 Transformer 论文的架构图
                  （它由你写过的哪几块组成）
                - 用**参数量、数据量、算力**三把尺子
                  衡量任何新模型发布
                - 解释"为什么大模型突然会做题了"
                  （涌现与上下文学习）以及这场争论
                  的另一面

                前置：44 课解码器（GPT 的全部机械原理）。

                本课没有训练代码——是一张地图。第 8
                部分（46-48 课）拿起 PyTorch，把课程
                全部手艺工业化。
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📅 概念 1/3 · 时间线：六年革命": mo.md(
                """
                | 时间 | 事件 | 一句话意义 |
                |---|---|---|
                | 2017.6 | **Attention is All You Need** | 翻译机器，注意力登场 |
                | 2018.6 | **GPT-1**（1.17 亿参数） | 生成式预训练 + 下游微调 |
                | 2018.10 | **BERT**（3.4 亿） | 双向理解，刷榜各项 NLP 任务 |
                | 2019.2 | **GPT-2**（15 亿） | "太危险不敢完整发布"；零样本萌芽 |
                | 2020.5 | **GPT-3**（1750 亿） | 上下文学习：不微调，看例子就会 |
                | 2022.3 | **Chinchilla** | 算清了 Scaling 的账：20 token/参数 |
                | 2022.11 | **ChatGPT** | 指令微调 + RLHF，破圈 |
                | 2023.3 | **GPT-4** | 多模态；参数未公开 |

                三条主线贯穿：**架构收敛**（全部统一到
                你写过的块）、**规模狂奔**（6 年参数涨
                1500 倍）、**目标进化**（从"预测下一词"
                到"听懂人话"）。
                """
            ),
            "🏛️ 概念 2/3 · 三种架构、一个零件": mo.md(
                """
                Transformer 的三种车型，发动机都是你
                38-44 课造的那台：

                | 架构 | 代表 | 注意力 | 预训练目标 | 擅长 |
                |---|---|---|---|---|
                | 编码器 | **BERT** | 双向（无掩码） | 完形填空（MLM） | 理解：分类、检索 |
                | 解码器 | **GPT** | 因果（44 课掩码） | 预测下一词 | 生成：写作、对话 |
                | 编-解码 | **T5**/翻译机 | 自注意 + 交叉注意 | 序列到序列 | 翻译、摘要 |

                - **BERT 的童年**（完形填空）：随机遮
                  15% 的词，用双向上下文猜——像读完
                  形填空的学生，**看得全但不会写**
                  （每题只填空，不续写）
                - **GPT 的童年**（预测下一词）：44 课
                  你亲手训的那个目标——像逐字抄写的
                  学生，**会写但只见过上文**
                - 两种童年没有优劣，只有分工：理解类
                  任务 BERT 系长期称霸（检索、分类）；
                  生成的天下是 GPT 的

                2020 年后Decoder-only 逐渐一统江湖：
                GPT-3 证明**足够大的"预测下一词"连
                理解任务也能做好**（把分类题改写成
                续写题）——一个架构通吃的时代开始。
                """
            ),
            "📈 概念 3/3 · Scaling Laws：规模的物理学": mo.md(
                """
                2020 年前后，OpenAI 与 DeepMind 各自
                摸出一条经验定律：**验证损失随参数量、
                数据量、算力的幂律下降**——

                ```
                L ≈ L∞ + (N / Nc)^(-α)     N：参数量
                ```

                三个实操结论（Chinchilla 2022 算的账）：

                - **20 token/参数**是训练预算的甜点：
                  700 亿参数的模型配 1.4 万亿 token，
                  打赢了当时 2800 亿参数的 Gopher——
                  **数据量被系统性低估了**
                - **算力固定时**，参数和数据要同步涨
                  （而不是把参数堆到天上去）
                - 幂律意味着**没有免费午餐的尽头**：
                  每多 10 倍算力，损失降一截，但增量
                  递减——烧钱买智能，单价越来越贵

                回看 44 课的账：迷你 GPT 在 1.1 万字符
                上输给 LSTM——**归纳偏置弱的架构需要
                数据补偿**。Scaling Laws 就是"补偿到
                什么程度、能赢回来多少"的账本。数据
                涨 9 个数量级后，LSTM 的时间复杂度
                （串行训练）先顶不住了——注意力的
                并行性（44 课）才是它能吃下互联网的
                资格证。
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
        ### 演示 1 · 规模狂奔：六年，三个数量级

        对数轴上的参数量（公开文献数字）——从 BERT
        的 1.1 亿到 GPT-3 的 1750 亿，六年 **1500 倍**；
        右边加上课程坐标：你 44 课训的迷你 GPT 在
        最左下角，比 GPT-3 小 **150 万倍**——但公式
        一个不差。
        """
    )
    return


@app.cell
def _(np, plt):
    # 演示 1 · 参数规模时间线（对数轴）
    _models = [
        ("本课迷你GPT", 2024.9, 1.15e5, "tab:red"),
        ("GPT-1", 2018.45, 1.17e8, "tab:blue"),
        ("BERT-base", 2018.8, 1.1e8, "tab:green"),
        ("BERT-large", 2018.8, 3.4e8, "tab:green"),
        ("GPT-2", 2019.1, 1.5e9, "tab:blue"),
        ("T5-11B", 2019.8, 1.1e10, "tab:orange"),
        ("GPT-3", 2020.4, 1.75e11, "tab:blue"),
        ("PaLM", 2022.3, 5.4e11, "tab:purple"),
        ("Chinchilla", 2022.2, 7.0e10, "tab:brown"),
        ("LLaMA-65B", 2023.1, 6.5e10, "tab:gray"),
    ]
    _fig, _ax = plt.subplots(figsize=(9.6, 4.4))
    for _nm, _yr, _p, _c in _models:
        _ax.scatter(_yr, _p, s=64, color=_c, zorder=3)
        _ax.annotate(_nm, (_yr, _p), textcoords="offset points", xytext=(4, 6),
                     fontsize=8.5)
    _ax.set_yscale("log")
    _ax.set_xlabel("发布时间（年）")
    _ax.set_ylabel("参数量（对数轴）")
    _ax.set_title("六年三个数量级：架构收敛到 Transformer，规模决定段位", fontsize=11.5)
    _ax.grid(True, alpha=0.3, which="both")
    _fig.tight_layout()
    _fig

    print(f"GPT-3 / 迷你GPT = {1.75e11/1.15e5:,.0f} 倍参数——公式一个不差，差的只是规模")
    print(f"GPT-3 / GPT-1   = {1.75e11/1.17e8:,.0f} 倍（两年半，1500 倍）")
    print("Chinchilla 的提醒：700 亿参数 + 1.4 万亿 token 打赢 2800 亿参数——数据同样值钱")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 2 · 两种童年：完形填空 vs 预测下一词

        同一段话，两种预训练目标看到的"考卷"完全
        不同。左边 BERT 的 MLM：遮住 15% 的词，用
        **双向**上下文猜（看得全）。右边 GPT 的因果
        LM：每个位置只看前文，预测下一词（会续写）。

        一个细节：MLM 遮的词是**随机 15%**——BERT
        的每道题只考一个洞；GPT 的每个位置都是
        考题，**训练信号密度高 T 倍**（44 课的并行
        训练），这也是 decoder 路线后来居上的工程
        原因之一。
        """
    )
    return


@app.cell
def _(np, plt):
    # 演示 2 · MLM vs 因果LM 的考卷对比
    _sent = list("梯度下降是深度学习的核心算法之一")
    _rng = np.random.default_rng(0)
    _mlm_mask = _rng.random(len(_sent)) < 0.25          # 演示用 25% 更醒目（真实 15%）

    _fig, _axes = plt.subplots(2, 1, figsize=(10.4, 3.4))
    # MLM
    _x = np.arange(len(_sent))
    _axes[0].scatter(_x, np.zeros_like(_x), s=0)
    for _i, (_ch, _m) in enumerate(zip(_sent, _mlm_mask)):
        _axes[0].text(_i, 0, "□" if _m else _ch, ha="center", va="center",
                      fontsize=13, color="tab:red" if _m else "black")
    _axes[0].set_xlim(-1, len(_sent))
    _axes[0].set_ylim(-1, 1)
    _axes[0].axis("off")
    _axes[0].set_title("BERT 的考卷（MLM）：遮住部分词，双向上下文猜洞（红）", fontsize=11)
    # 因果 LM
    for _i, _ch in enumerate(_sent[:-1]):
        _axes[1].text(_i, 0, _ch, ha="center", va="center", fontsize=13, color="black")
        _axes[1].annotate("", xy=(_i + 0.85, 0.45), xytext=(_i + 0.15, 0.45),
                          arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.2))
        _axes[1].text(_i + 0.5, 0.75, _sent[_i + 1], ha="center", va="center",
                      fontsize=10, color="tab:blue")
    _axes[1].set_xlim(-1, len(_sent))
    _axes[1].set_ylim(-1, 1.2)
    _axes[1].axis("off")
    _axes[1].set_title("GPT 的考卷（因果 LM）：每个位置只看前文，预测下一字（蓝）", fontsize=11)
    _fig.tight_layout()
    _fig

    print(f"MLM：{int(_mlm_mask.sum())}/{len(_sent)} 个洞，每洞一题（双向看）")
    print(f"因果 LM：{len(_sent)-1} 个位置全部是题（只看前文）——训练信号密度高得多")
    print("MLM 微调才能生成（没学过续写）；因果 LM 天生会写——但理解任务要绕个弯（改写成续写）")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 3 · ChatGPT 的最后三步：预训练之上

        GPT-3 会预测下一词，但**不会听话**——你问
        天气它可能续写一篇日记。ChatGPT = 三层蛋糕：

        | 层 | 做什么 | 数据 |
        |---|---|---|
        | ① 预训练 | 预测下一词（44 课那个目标） | 万亿 token 互联网文本 |
        | ② 指令微调 SFT | 学"问答/遵循指令"的格式 | 人工标注的指令-回答对（万级） |
        | ③ 人类反馈 RLHF | 对齐"哪个回答更好" | 人工偏好排序（十万级） |

        ②③ 用的数据比 ① 少**五六个数量级**——它们
        不注入知识，只**整形行为**：把"续写机器"
        校准成"助手"。这也是课程一直的主题：**结构
        决定能学什么，数据决定学到什么，目标函数
        决定长成什么**。

        （RLHF 的核心部件——奖励模型 + 强化学习
        微调——超出本课范围；你已具备理解它的全部
        前置：损失函数、梯度、采样。）
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 演示 4 · 你爬过的山：课程模型编年史

        把本课程训过的模型放在同一张对数轴上——
        从 15 课的 2 参数线性回归到 44 课的 11.5 万
        参数迷你 GPT，你亲手爬了 **4.7 个数量级**；
        到 GPT-3 还差 6 个数量级，但那只是同一座
        山上更高的营地。
        """
    )
    return


@app.cell
def _(np, plt):
    # 演示 4 · 课程模型编年史（参数量对数轴）
    _ours = [
        ("线性回归(15课)", 2),
        ("逻辑回归(16课)", 785),
        ("MLP(20课)", 2662),
        ("LeNet-5(30课)", 61706),
        ("LSTM(37课)", 326329),
        ("编码器(43课)", 26018),
        ("迷你GPT(44课)", 114841),
    ]
    _fig, _ax = plt.subplots(figsize=(9.4, 4.0))
    _ys = np.arange(len(_ours))
    _vals = [p for _, p in _ours]
    _ax.barh(_ys, np.log10(_vals), color=["tab:gray"]*4 + ["tab:green"] + ["tab:blue"]*2)
    _ax.set_yticks(_ys, [n for n, _ in _ours])
    for _y, _v in zip(_ys, _vals):
        _ax.text(np.log10(_v) + 0.05, _y, f"{_v:,}", va="center", fontsize=9)
    _ax.axvline(np.log10(1.75e11), color="tab:red", ls="--", lw=1.2)
    _ax.text(np.log10(1.75e11) - 0.15, 0.2, "GPT-3\n1750亿", color="tab:red",
             fontsize=9, ha="right")
    _ax.set_xlabel("参数量（log10 尺度）")
    _ax.set_title("你爬过的山：7 个亲手训练的模型，4.7 个数量级", fontsize=11.5)
    _ax.grid(True, axis="x", alpha=0.3)
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
        ### 交互实验 · 名片浏览器

        选一个模型，看它的名片：架构（由你写过的
        哪些零件组成）、参数量、训练数据、历史
        贡献。全部数字来自公开论文/官方发布。

        玩法建议：对着名片问自己三个问题——它是
        三种架构的哪一种？预训练目标是什么？它
        比 44 课的迷你 GPT 大多少倍？
        """
    )
    return


@app.cell
def _(mo):
    # 交互控制：模型选择
    model_dd45 = mo.ui.dropdown(
        options=["GPT-1（2018）", "BERT（2018）", "GPT-2（2019）", "GPT-3（2020）",
                 "T5（2019）", "Chinchilla（2022）", "LLaMA（2023）", "GPT-4（2023）"],
        value="GPT-3（2020）", label="🗂️ 选一张名片")
    model_dd45
    return (model_dd45,)


@app.cell
def _(model_dd45, mo):
    # ===== 交互实验 · 名片浏览器（响应单元）=====
    _cards = {
        "GPT-1（2018）": dict(
            架构="解码器 ×12 层（d=768，12 头）——44 课的块叠 12 层",
            参数="1.17 亿",
            数据="BookCorpus 约 8 亿词",
            目标="预测下一词 + 下游任务微调",
            贡献="确立「生成式预训练 + 微调」范式：先博览群书，再按岗培训"),
        "BERT（2018）": dict(
            架构="编码器 ×12/24 层——43 课的块（双向、无掩码）",
            参数="1.1 亿（base）/ 3.4 亿（large）",
            数据="Wikipedia + BookCorpus 约 33 亿词",
            目标="完形填空 MLM（遮 15%）+ 下一句预测",
            贡献="刷新 11 项 NLP 任务纪录；「预训练模型」成为行业基建"),
        "GPT-2（2019）": dict(
            架构="解码器 ×48 层（d=1600）",
            参数="15 亿",
            数据="WebText 约 800 万网页",
            目标="预测下一词（零样本：不做微调直接考）",
            贡献="零样本能力萌芽；「太危险不宜发布」引发 AI 安全大讨论"),
        "GPT-3（2020）": dict(
            架构="解码器 ×96 层（d=12288，96 头）——40 课头数扫描的尽头",
            参数="1750 亿",
            数据="约 3000 亿 token（加权互联网文本）",
            目标="预测下一词（不再微调）",
            贡献="上下文学习（看几个例子就会新任务）；规模定律的旗舰证据"),
        "T5（2019）": dict(
            架构="编码器 + 解码器（44 课交叉注意力正式上岗）",
            参数="6000 万~110 亿（系列）",
            数据="C4 清洗后的 Common Crawl（约 1 万亿 token）",
            目标="序列到序列：一切任务都改写成文本进文本出",
            贡献="「Text-to-Text」统一范式；翻译/摘要的标杆"),
        "Chinchilla（2022）": dict(
            架构="解码器（与 Gopher 同款）",
            参数="700 亿",
            数据="1.4 万亿 token",
            目标="预测下一词",
            贡献="算清 Scaling 的账：每参数约 20 token——数据量被低估了；小而美打赢大而全"),
        "LLaMA（2023）": dict(
            架构="解码器 ×32~80 层（开源权重）",
            参数="70 亿~650 亿（系列）",
            数据="1~1.4 万亿 token（公开来源）",
            目标="预测下一词",
            贡献="开源生态引爆：消费级显卡可跑，本课读者日后微调的第一块踏板"),
        "GPT-4（2023）": dict(
            架构="未公开（推测为 Transformer 变体 + MoE）",
            参数="未公开",
            数据="未公开（多模态：图文混训）",
            目标="预测下一 token（文本 + 图像块）",
            贡献="多模态与专业考试水平；闭源黑箱引发可解释性新议程"),
    }
    _c = _cards[model_dd45.value]
    _md = f"""
    ### 🗂️ {model_dd45.value}

    | 项 | 内容 |
    |---|---|
    | **架构** | {_c["架构"]} |
    | **参数量** | {_c["参数"]} |
    | **训练数据** | {_c["数据"]} |
    | **预训练目标** | {_c["目标"]} |
    | **历史贡献** | {_c["贡献"]} |
    """
    mo.callout(mo.md(_md), kind="none")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✏️ 动手练习（本课 3 题）": mo.md(
                """
                **练习 1 · 数量级手感**：GPT-3 有 1750 亿
                参数，44 课的迷你 GPT 有 114,841 个——
                前者是后者的多少倍（数量级即可，
                如 10⁶）？

                **练习 2 · Chinchilla 的账**：按"每参数
                20 token"的甜点，一个 700 亿参数的
                模型应该用多少训练 token（万亿级）？

                **练习 3 · 架构连连看**：翻译模型、
                句子情感分类、写诗续写——分别对应
                三种架构（编码器 / 解码器 / 编-解码器）
                的哪一个？（填：1=编码器 2=解码器
                3=编-解码器）
                """
            )
        }
    )
    return


@app.cell
def _(mo):
    # ===== 练习 1 · 数量级手感 =====
    print("GPT-3：175,000,000,000 参数｜迷你 GPT：114,841 参数")

    ratio_ex1_45 = None                               # ← ① 倍数（数量级，如 1.5e6 或 1500000）

    if ratio_ex1_45 is not None:
        _r = 1.75e11 / 114841
        if 0.6 < ratio_ex1_45 / _r < 1.7:
            _msg = (f"✅ 正确：约 {_r:,.0f} 倍（≈1.5×10⁶）。反过来说：把迷你 GPT 的"
                    "公式原样放大 150 万倍（层数 2→96、宽度 48→12288、词表 601→5 万）"
                    "就是 GPT-3 的身体。你的 64 行前向代码，字面上就是那台机器的"
                    "微缩图纸——差的只是电费和数据。")
        else:
            _msg = "❌ 参考：约 152 万倍（1.5×10⁶）"
    else:
        _msg = "⏳ 填一个数（可用科学计数法）"
    mo.md(_msg)
    return (ratio_ex1_45,)


@app.cell
def _(mo):
    # ===== 练习 2 · Chinchilla 的账 =====
    print("经验甜点：每参数约 20 个训练 token")

    tokens_ex2_45 = None                              # ← ① 700 亿参数模型需要的训练 token（万亿级，如 1.4e12）

    if tokens_ex2_45 is not None:
        _t = 7e10 * 20
        if 0.6 < tokens_ex2_45 / _t < 1.7:
            _msg = (f"✅ 正确：7×10¹⁰ × 20 = 1.4×10¹²（1.4 万亿 token）——正是 "
                    "Chinchilla 的配置。对比 GPT-3：1750 亿参数只喂了约 3000 亿 "
                    "token（每参数 1.7 个）——**欠喂**。Chinchilla 用 1/2.5 的参数"
                    "加 4.7 倍的数据，打赢了 2800 亿的 Gopher：Scaling 不等于堆参数。")
        else:
            _msg = "❌ 参考：700 亿 × 20 = 1.4 万亿 token"
    else:
        _msg = "⏳ 填一个数（万亿级）"
    mo.md(_msg)
    return (tokens_ex2_45,)


@app.cell
def _(mo):
    # ===== 练习 3 · 架构连连看 =====
    print("1=编码器（BERT 系）｜2=解码器（GPT 系）｜3=编-解码器（T5/翻译机）")
    print("a. 中译英翻译模型  b. 评论情感分类  c. 给开头续写一首诗")

    a_ex3_45 = None                                   # ← ① 翻译（填 1/2/3）
    b_ex3_45 = None                                   # ← ② 情感分类（填 1/2/3）
    c_ex3_45 = None                                   # ← ③ 续写诗（填 1/2/3）

    if None not in (a_ex3_45, b_ex3_45, c_ex3_45):
        if (a_ex3_45, b_ex3_45, c_ex3_45) == (3, 1, 2):
            _msg = ("✅ 正确：3 / 1 / 2。翻译天然是序列到序列（原句进、译文出，"
                    "长度都不同）——交叉注意力（44 课演示 6）的本职；分类是理解"
                    "任务（43 课的 CLS 就是干这个的）；续写是因果生成（44 课的"
                    "全部）。注意边界在模糊：GPT-3 之后，解码器靠「把任务改写成"
                    "续写」通吃了 a 和 b——翻译给几个例子照猫画虎，分类把选项"
                    "排在句尾让它填——**架构没有垄断任务，只有效率差异**。")
        else:
            _msg = "❌ 参考：a=3（编-解码）、b=1（编码器）、c=2（解码器）"
    else:
        _msg = "⏳ 三空都填（1/2/3）"
    mo.md(_msg)
    return (a_ex3_45, b_ex3_45, c_ex3_45)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "✅ 练习 1 参考答案（先自己写过再看！）": mo.md(
                """
                **约 152 万倍（1.5×10⁶）。**

                阶梯感受一下这个数：
                迷你 GPT（44 课，你训的）→ GPT-1
                （1000 倍）→ GPT-2（13000 倍）→ GPT-3
                （150 万倍）。每一步都是：**层数加厚、
                宽度加宽、词表加大、数据上天**——
                公式没动过。

                代价侧同样指数：训练 GPT-3 的电费按
                数百万美元计；你的迷你 GPT 用笔记本
                电池一分钟跑完。这就是为什么课程的
                最后一部分（46-48 课）教 PyTorch——
                手写 NumPy 让你**懂**，框架让你**快**，
                GPU 让你**敢**。
                """
            ),
            "✅ 练习 2 参考答案（先自己写过再看！）": mo.md(
                """
                **1.4 万亿 token（7×10¹⁰ × 20）。**

                Chinchilla 的核心贡献：在固定算力下，
                参数 N 和数据 D 应该**按比例同步扩大**
                （D ≈ 20N 是 2022 年算出的甜点）。
                此前的行业惯性是拼命堆参数——GPT-3
                每参数只喂了 1.7 个 token，明显欠喂。

                后续模型（LLaMA、GPT-4）都吸收了这
                笔账：LLaMA-7B 用了 1 万亿 token
                （每参数 143 个）——**过量喂食**成了
                新策略：训练完的模型要被推理千万次，
                多喂的数据摊薄到每次推理就便宜了。
                账，永远是算出来的。
                """
            ),
            "✅ 练习 3 参考答案（先自己写过再看！）": mo.md(
                """
                **a=3、b=1、c=2。**

                判断口诀：**输出是一个标签 → 编码器；
                输出是一段文本 → 解码器或编-解码器；
                有"原文"要参照 → 编-解码器**。

                但 2020 年后的趋势值得记住：decoder-
                only 一统江湖。GPT-3 用上下文学习把
                分类改写成续写（"这条评论的情感是正
                还是负？答："），把翻译改写成对照
                示例。工业界选择它的工程理由很多
                （训练并行、管线统一、规模化经验），
                但**任务-架构的绑定确实松动了**——
                你 44 课写的那台机器，是这场统一的
                底层通货。
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
                1. **时间线**：2017 翻译论文 → 2018
                   GPT-1/BERT → 2020 GPT-3（上下文学习）
                   → 2022 Chinchilla（算账）→ 2022.11
                   ChatGPT（对齐）→ 2023 GPT-4
                2. **三种架构一个零件**：编码器（BERT，
                   双向理解）、解码器（GPT，因果生成）、
                   编-解码器（T5，翻译）——块全是
                   38-44 课写过的块
                3. **两种童年**：MLM 完形填空（看得全
                   不会写）vs 因果预测下一词（会写只见
                   前文）；后者训练信号密度高 T 倍
                4. **Scaling Laws**：损失随参数/数据/
                   算力幂律下降；甜点约 20 token/参数；
                   44 课迷你 GPT 输给 LSTM 的账，靠
                   规模（+并行性）赢回来
                5. **ChatGPT 三层蛋糕**：预训练（万亿
                   token 注入知识）→ SFT（万级数据
                   整形格式）→ RLHF（十万级偏好对齐
                   价值观）——后两层小五个数量级
                6. **你的坐标**：亲手训过 7 个模型、
                  爬过 4.7 个数量级；到 GPT-3 还差
                   6 个——但那是同一座山：公式你全
                   都会写了
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
                ## 🎓 第 7 部分收官：Transformer（8/8）

                八课连打，你从"让每个词看到每个词"的
                直觉走到了亲手训练一台迷你 GPT。清点行囊：

                | 课 | 里程碑：你现在能做到什么 |
                |---|---|
                | **38** 注意力直觉 | 打分→softmax→加权平均；指代消解；并行的代价 O(T²) |
                | **39** QKV | 三顶帽子、自恋问题、√d 缩放的方差账、角色分离可训练 |
                | **40** 多头 | 切空间拼接投影；4d² 预算不变买 h 个视角；头的冷启动与课程学习 |
                | **41** 位置编码 | 排列不变性；正弦编码；相对位置一次点积读出；免费外推 |
                | **42** Transformer 块 | 残差高速公路（10⁻⁷ vs 20）；LayerNorm；Pre/Post-LN；12d² |
                | **43** 编码器实战 | 全链路手写反向+梯度校验；100% 测试；注意力解剖（路由器非聚光灯） |
                | **44** 解码器与掩码 | 因果掩码；训练并行/推理串行；迷你 GPT 同台 LSTM；交叉注意力 |
                | **45** GPT/BERT 概览 | 三架构一零件；Scaling Laws；ChatGPT 三层蛋糕 |

                用一句话回顾第 7 部分：**注意力把"看哪里"
                从结构里解放出来交给数据去学**——38 课
                的一行点积，长到 45 课的 1750 亿参数，
                中间只有规模，没有魔法。
                """
            ),
            mo.callout(
                mo.md(
                    """
                    你现在处于一个微妙的位置：**原理全懂、
                    工具全无**。手写 NumPy 让你看清了每一个
                    梯度从哪里来——但真实世界里没人用 NumPy
                    训模型：自动微分、GPU 加速、分布式——
                    工业界的三大件你还没上手。

                    最后一部分（46-48 课）补齐它们：**PyTorch
                    的 autograd** 替你写反向传播（46）、
                    **nn 模块**替你重写 MLP/CNN/注意力（47）、
                    **训练范式**把 25 课的循环升级成工业标准
                    （48）。三课之后，你写的将不再只是"能跑
                    的教学代码"，而是"能换成 GPU 就上生产"
                    的真代码。

                    从零到 Transformer——最后一程。
                    """
                ),
                kind="success",
            ),
            mo.md(
                """
                ---

                **下一课预告** — `46_autograd`（第 8 部分 · 过渡到 PyTorch）：

                22-23 课你手写了两个课时的反向传播，
                43-44 课又写了 150 行——**是时候让机器
                干这活了**。autograd 的魔法只有一个词：
                **记录**。前向传播时把每个操作存进一张
                计算图，反向时沿着图自动套链式法则——
                你手推的每一条梯度公式，它都能自动
                生成。

                46 课验证一件事：PyTorch 算出的梯度，
                和你 43 课手写的解析梯度**逐位对上**
                （数值上 10⁻⁸ 级）。然后你会体会 autograd
                的真正代价与自由：再也不用写 backward
                了——以及，为什么 43 课那种"写错了也
                不报错"的 bug 从此绝迹。
                """
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
