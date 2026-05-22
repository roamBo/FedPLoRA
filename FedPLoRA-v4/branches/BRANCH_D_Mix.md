# Branch D — FedPLoRA-Mix（FedDPA / FedALT 双适配器思路）

## D.0 一句话定位

> **每个客户端在 one-shot 之后同时持有 `A_local`（本地训练得到）和 `A_down`（服务端聚合下发）；推理时通过 mixer η ∈ [0, 1] 做 `ΔW = B · (η · A_down + (1−η) · A_local)`，让服务端的"跨域共识"和客户端的"领域记忆"按需融合，且通信完全不变。**

## D.1 假设与预期

| 假设 | 检验方式 |
|---|---|
| H-D.1 保留 `A_local` 比纯下发 `A_down` 在 Macro 上更优（因为 A_local 与 B_i 训练耦合度高） | D1 vs v2 |
| H-D.2 per-domain 自适应 η 比 fixed η=0.5 更好 | D2 vs D1 |
| H-D.3 per-input MoE gate 提升空间最大，但需更多调参 | D3 vs D2 |
| H-D.4 D2/D3 在 Worst 上 ≥ v2 + 0.005 | D2/D3 vs v2 |

预期收益（基于 v2 base 0.69595 macro_acc）：

- D1 fixed η=0.5：**+0.001 ~ +0.004**（最简，零训练成本，立刻可测）
- D2 per-domain η（grid on val 50 batches）：**+0.003 ~ +0.008**
- D3 per-input MoE：**+0.005 ~ +0.012**（但训练 mixer 增加 ~5 min/客户端）

## D.2 设计要点

### D.2.1 客户端状态扩展

v2 的客户端持有 `B_i`（local）；v4 Branch D 额外持有 `A_i_local`（本地最后一次训练后的 A，作为 *客户端个性化记忆*）。

**通信不变**：
- 上行：`A_i`（=local 训练后的 A）+ heads + row_importance（与 v2 一致）。
- 下行：`A_down`（共享 A）+ heads（与 v2 一致）。

**存储增加**：
- 每客户端额外保存一份 `A_i_local`（≈ 19 MB on Llama-3.1-8B r=8），与 `B_i_local` 同存。

### D.2.2 三种 Mixer

#### D1 — fixed mixer
```python
A_eff = 0.5 * A_down + 0.5 * A_local
ΔW = B @ A_eff
```
完全没有学习成本，但忽略域差异。

#### D2 — per-domain mixer
对每个域，用其 val 集 50 batches 在 grid `η ∈ {0.0, 0.1, 0.2, ..., 1.0}` 找最优：
```python
for eta in grid:
    A_eff = eta * A_down + (1 - eta) * A_local
    loss = eval_on_val(model_with_A_eff)
eta_star[domain] = argmin(loss)
```
所有该域客户端共用 `eta_star[domain]`。

#### D3 — per-input MoE gate
增加一个小 gate 网络 `g(x) ∈ [0, 1]`：
```python
gate_net = MLP(hidden_dim=64, depth=2)  # input: pooled hidden states, output: scalar
gate = gate_net(hidden)
A_eff = gate * A_down + (1 - gate) * A_local
```
本地用 50 batches val 数据训练 `gate_net`（冻结 A_down, A_local, B）。

### D.2.3 关键工程选择

- **A_eff 不是合并到一个 LoRA**：因为两个 A 形状相同 (r, d)，可以直接线性组合得到 (r, d)。然后 `B @ A_eff` 还是 (out_dim, d)。零额外参数。
- **A_eff 在每 forward 重算**：训练阶段不存在（one-shot 后不再训），推理阶段每 batch 算一次 mixer，开销 << forward。
- **A_local 何时定义**：
  - 客户端本地训练完成、上行 `A_i` *之前*，把 `A_i` 复制一份做 `A_i_local` 存到 `client_state_dir/A_local_{client_id}.pt`。
  - 服务端下发 `A_down` 后，客户端 load 进 model 的 LoRA A，*不覆盖* `A_local` 文件。
  - 推理时 model 的 LoRA A = A_down（base），mixer 时读取 `A_local` 文件做线性组合。

### D.2.4 评估改造

```python
def evaluate_mix(model, dataloader, A_down_dict, A_local_dict, eta_fn):
    model.eval()
    # Override lora_A forward to use mixed A
    def patched_forward(self, x):
        eta = eta_fn(x)                          # scalar or (B, 1, 1)
        A_eff = eta * A_down_dict[self.name] + (1 - eta) * A_local_dict[self.name]
        return F.linear(x, A_eff)
    ...
```

实现思路：用 `forward_pre_hook` 或 monkey-patch `peft.tuners.lora.Linear.forward`，把 `lora_A.weight` 临时替换为 `A_eff`。

## D.3 与 FedDPA / FedALT 的区别

| 维度 | FedDPA | FedALT | v4-Mix |
|---|---|---|---|
| 适配器数 | 2 套独立 LoRA | 1 self + 1 RoTW LoRA | 1 LoRA，A 端有两版 |
| 额外参数 | 2× LoRA | 2× LoRA | 0（只多存 A_i_local） |
| 通信 | 仅 global LoRA | 仅 global LoRA + RoTW | 完全不变（v2 通信） |
| Mixer 类型 | instance-wise dynamic | MoE-style input-dependent | fixed / per-domain / per-input |

**v4 优势**：参数和通信成本最低；劣势是 mixer 容量也最小（只能调权重，不能调方向）。

## D.4 超参建议

| 参数 | D1 | D2 | D3 |
|---|---|---|---|
| `--v4_mix_mode` | `fixed` | `per_domain` | `per_input` |
| `--v4_mix_eta` | 0.5 | (auto search) | (gate net learns) |
| `--v4_mix_search_grid` | — | "0.0,0.1,...,1.0" | — |
| `--v4_mix_search_max_batches` | — | 50 | 50 (for gate train) |
| `--v4_mix_gate_hidden` | — | — | 64 |
| `--v4_mix_gate_epochs` | — | — | 3 |
| `--v4_mix_save_local_A` | 1 | 1 | 1 |

## D.5 实现入口

- 服务端聚合：复用 v2 fedplora-oneshot 的聚合（不变）。
- 客户端状态：[utilities/client_state_extra.py](../utilities/client_state_extra.py) 扩展保存 `A_local`。
- Mixer：[methods/fedplora_v4_mix.py](../methods/fedplora_v4_mix.py)。
- 评估：[utilities/eval_personalized.py](../utilities/eval_personalized.py)，patched forward。
- 训练入口：`agg_type ∈ {v4_mix_fixed05, v4_mix_per_domain, v4_mix_moe}`。
- 运行脚本：[scripts/RunScripts/run_v4_branch_d.sh](../scripts/RunScripts/run_v4_branch_d.sh)。

## D.6 风险

1. `A_local` 存储成本：35 × 19 MB ≈ 665 MB，可接受。
2. per-input MoE gate 需要 val 数据训练 —— 与论文 "true one-shot, no extra rounds" 叙事有张力，需明说"客户端本地 val 集已经在训练数据 split 里"。
3. mixer η 在 high-conflict 域（legal/finance）可能学到 ≈0 即"忽略 A_down"，等价于退化到纯本地训练。可加正则 `(η − 0.5)^2` 防止极端值。

## D.7 论文话术（草稿）

> Existing personalized FedLoRA methods either ship two full adapters per client (FedDPA / FedALT) or destroy local specialization by overwriting client A with the server aggregate (FedSA-LoRA). FedPLoRA-Mix observes that one-shot communication already implies clients have *seen* their local A — keeping it costs nothing extra in bandwidth. By mixing `A_down` and `A_local` with a per-domain or per-input scalar gate, Mix recovers the personalization advantage of FedALT-style methods while preserving v2's 38 MB single-round budget.
