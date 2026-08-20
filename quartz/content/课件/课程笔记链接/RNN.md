[循环神经网络RNN完全解析：从基础理论到PyTorch实战 - 知乎](https://zhuanlan.zhihu.com/p/652712909)

---

# 循环神经网络 RNN：原理、变体与 PyTorch 实现

> [!abstract] 核心概括
> 循环神经网络（Recurrent Neural Network, RNN）通过在时间步之间传递隐藏状态，让模型利用此前的信息处理当前输入。它适合文本、语音和时间序列等顺序数据，但标准 RNN 难以学习很长的依赖关系，因此出现了 LSTM、GRU 和双向 RNN 等变体。

## 1. 为什么需要 RNN

普通前馈神经网络通常把每个输入独立处理，不能自然保留输入顺序。序列数据则具有上下文依赖，例如：

- 一句话中当前词的含义与前文有关；
- 当前气温与此前多个时间点有关；
- 一帧视频中的动作需要结合前后帧判断。

RNN 在隐藏层加入循环连接，将上一个时间步的隐藏状态作为当前计算的一部分，相当于维护一个不断更新的“记忆”。

![[process file/20260814+RNN链接总结/01-rnn-definition.png|680]]

*图 1：普通神经网络与带循环连接的 RNN 结构对比。*

## 2. 基本结构与时间展开

RNN 的循环结构可以沿时间轴展开。虽然展开后看起来像多个单元，但各时间步使用的是**同一组参数**。

![[process file/20260814+RNN链接总结/02-rnn-structure.png|720]]

*图 2：同一个 RNN 单元沿序列展开；隐藏状态把此前信息传到下一时间步。*

在时间步 $t$，基本 RNN 可表示为：

$$
h_t=\phi(W_{xh}x_t+W_{hh}h_{t-1}+b_h)
$$

$$
y_t=W_{hy}h_t+b_y
$$

其中：

- $x_t$：当前输入；
- $h_{t-1}$：上一时间步的隐藏状态；
- $h_t$：融合当前输入和历史信息后的新状态；
- $y_t$：当前输出；
- $\phi$：通常为 Tanh 或 ReLU 等激活函数；
- $W_{xh},W_{hh},W_{hy}$：所有时间步共享的权重。

参数共享使 RNN 能处理不同长度的序列，同时避免参数数量随序列长度增加。

## 3. 输入与输出形式

RNN 可以根据任务采用不同输出方式：

| 形式 | 输入与输出 | 典型任务 |
|---|---|---|
| 多对一 | 整个序列 → 一个结果 | 情感分类、序列分类 |
| 多对多（同步） | 每个时间步 → 一个输出 | 词性标注、逐帧分类 |
| 多对多（异步） | 输入序列 → 另一输出序列 | 机器翻译、文本生成 |
| 一对多 | 一个输入 → 一个序列 | 图像描述、音乐生成 |

## 4. RNN 的训练与主要问题

### 4.1 随时间反向传播

RNN 通常使用 BPTT（Backpropagation Through Time，随时间反向传播）训练：先把网络沿时间展开，再从后向前计算各时间步的梯度。

### 4.2 梯度消失与梯度爆炸

由于同一权重和激活函数导数会跨多个时间步反复相乘：

- 连乘结果不断缩小，会出现[[梯度消失]]，模型难以学习长期依赖；
- 连乘结果不断放大，会出现梯度爆炸，训练可能震荡或产生数值异常。

常用缓解方法包括：

- 使用 LSTM 或 GRU；
- 采用合适的初始化、归一化和激活函数；
- 对过大的梯度进行梯度裁剪；
- 合理控制序列长度或采用截断 BPTT。

> [!warning]
> 梯度裁剪主要解决梯度爆炸，不能直接恢复已经消失的小梯度。

## 5. RNN 的优缺点

### 优点

- 能处理可变长度的序列；
- 通过隐藏状态表达时间依赖和上下文；
- 各时间步共享参数，结构相对紧凑；
- 可灵活用于分类、标注和生成任务。

### 局限

- 标准 RNN 对长期依赖的学习能力有限；
- 时间步之间存在顺序依赖，训练难以完全并行；
- 长序列上可能出现梯度消失或爆炸；
- 隐藏状态容量有限，早期信息可能逐渐被覆盖。

## 6. 主要变体

### 6.1 LSTM：长短时记忆网络

LSTM 在隐藏状态之外增加单元状态 $C_t$，并用门控机制控制信息流动：

- **遗忘门** $f_t$：决定保留多少旧记忆；
- **输入门** $i_t$：决定写入多少新信息；
- **候选状态** $\widetilde{C}_t$：生成可写入的新内容；
- **输出门** $o_t$：决定向外输出多少信息。

核心更新为：

$$
C_t=f_t\odot C_{t-1}+i_t\odot\widetilde{C}_t
$$

$$
h_t=o_t\odot\tanh(C_t)
$$

门的取值由 [[Sigmoid 函数]] 映射到 $(0,1)$，相当于控制信息通过的比例。LSTM 为梯度提供了较稳定的传播通路，能够缓解标准 RNN 的长期依赖问题，但参数量和计算成本更高。

![[process file/20260814+RNN链接总结/06-lstm-cell.png|720]]

*图 3：多层 LSTM 在序列上的展开示意。*

### 6.2 GRU：门控循环单元

GRU 将 LSTM 的结构简化，主要使用：

- **重置门** $r_t$：控制计算候选状态时忽略多少历史信息；
- **更新门** $z_t$：控制旧状态与候选状态的混合比例。

一种常见写法为：

$$
\widetilde{h}_t=\tanh\bigl(W_xx_t+W_h(r_t\odot h_{t-1})+b\bigr)
$$

$$
h_t=(1-z_t)\odot h_{t-1}+z_t\odot\widetilde{h}_t
$$

GRU 没有独立的单元状态，参数通常少于 LSTM，训练和推理可能更快；但两者谁更好需根据具体数据验证。

![[process file/20260814+RNN链接总结/07-gru-cell.png|720]]

*图 4：GRU 通过重置门和更新门控制历史状态与新状态的组合。*

### 6.3 双向 RNN（Bi-RNN）

双向 RNN 使用两个独立网络：

- 正向网络从序列开头处理到结尾；
- 反向网络从结尾处理到开头；
- 两个方向的隐藏状态经过拼接或其他方式合并。

![[process file/20260814+RNN链接总结/08-birnn.png|620]]

*图 5：双向 RNN 同时利用过去和未来的上下文。*

它适合词性标注、命名实体识别等可以看到完整序列的任务。由于需要未来信息，普通双向 RNN 不适合严格的实时因果预测。

### 6.4 变体对照

| 模型 | 状态与门 | 优点 | 代价/限制 |
|---|---|---|---|
| 标准 RNN | 单隐藏状态，无门控 | 结构简单、参数少 | 长期依赖能力弱 |
| LSTM | 隐藏状态 + 单元状态，三类门 | 长序列建模较稳定 | 参数多、计算量大 |
| GRU | 单隐藏状态，两类门 | 结构较简洁、效率较高 | 表达能力需由任务验证 |
| Bi-RNN | 正向与反向两个网络 | 同时利用前后文 | 不能直接用于严格在线预测 |

## 7. 应用场景

文章列举的主要应用包括：

1. **自然语言处理**：词性标注、命名实体识别、文本生成和机器翻译；
2. **语音处理**：语音识别和语音合成；
3. **时间序列分析**：价格、气象和传感器序列预测；
4. **视频分析**：动作识别和连续内容生成。

实际建模时仍需根据数据量、序列长度、是否需要实时预测等条件选择结构，而不是因为数据具有顺序就默认使用 RNN。

## 8. PyTorch 中的基本实现

### 8.1 张量形状

使用 `batch_first=True` 时：

```text
输入 x：    (batch_size, sequence_length, input_size)
输出 out：  (batch_size, sequence_length, hidden_size × directions)
最终 h_n：  (num_layers × directions, batch_size, hidden_size)
```

其中单向网络的 `directions = 1`，双向网络的 `directions = 2`。

### 8.2 序列分类骨架

```python
import torch
import torch.nn as nn

class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, h_n = self.rnn(x)
        last_hidden = h_n[-1]
        return self.fc(last_hidden)
```

典型训练步骤为：

```python
model.train()
optimizer.zero_grad()
logits = model(inputs)
loss = criterion(logits, targets)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

验证或测试时应使用：

```python
model.eval()
with torch.no_grad():
    logits = model(inputs)
```

## 9. 阅读原文代码时需注意

> [!warning] 原文代码是结构示意，不是完整可直接运行的项目
> - 部分片段缺少 `torch`、数据集、`epochs` 等定义；
> - 原文 LSTM 示例的 `forward(self, x, (h_0, c_0))` 不符合 Python 3 参数语法，应把状态作为一个参数接收；
> - 输出层和损失函数必须与任务匹配：分类通常使用类别数维度的 logits 与交叉熵，回归才常用 MSE；
> - 处理填充序列时应结合真实长度、mask 或 `pack_padded_sequence`，避免把 padding 当成有效信息；
> - 双向网络输出维度是 `2 × hidden_size`，后续全连接层需相应调整。

## 10. 核心总结

- RNN 的核心是隐藏状态递推和跨时间参数共享。
- 时间展开揭示了序列信息如何逐步传递，也解释了梯度问题的来源。
- 标准 RNN 适合短期依赖；长依赖通常优先比较 LSTM、GRU 或其他序列模型。
- 双向 RNN 可以利用完整前后文，但不适合只能访问历史信息的在线任务。
- PyTorch 实现时最容易出错的是张量形状、最终状态选择、padding 处理和任务对应的损失函数。

## 来源

- [知乎原始链接](https://zhuanlan.zhihu.com/p/652712909)
- [腾讯云开发者社区同步页](https://cloud.tencent.com/developer/article/2348483)

*整理日期：2026-08-14*
