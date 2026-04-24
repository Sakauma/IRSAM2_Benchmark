# Prompt Policy 规范

Prompt policy 是冻结的 benchmark 对象，必须显式给出：

- `prompt_type`
- `prompt_source`
- `prompt_budget`
- `refresh_interval`
- `multi_mask`

## 默认视频策略

- 首帧使用 GT prompt
- 允许稀疏 refresh
- 使用固定 refresh 间隔

这样可以避免不同方法在 prompt 预算不同的情况下发生不公平对比。
