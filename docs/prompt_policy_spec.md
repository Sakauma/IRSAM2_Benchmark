# Prompt Policy 规范

Prompt policy 是冻结的 benchmark 对象，必须显式给出：

- `prompt_type`
- `prompt_source`
- `prompt_budget`
- `refresh_interval`
- `multi_mask`

## 当前策略

- prompted baseline 使用从 GT mask 确定性派生的 box、tight box、point 或 box+point。
- no-prompt automatic-mask baseline 不使用外部 prompt，`prompt_budget` 必须为 `0`。
- 不同 prompt mode 必须分开报告，不能混成一个方法。
- `sam2_pretrained_*_prompt` 表示预训练 SAM2 加显式 prompt，不表示无 prompt。

这样可以避免不同方法在 prompt 预算不同的情况下发生不公平对比。
