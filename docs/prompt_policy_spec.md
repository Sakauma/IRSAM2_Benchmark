# Prompt Policy Spec

Prompt policy is a frozen benchmark object. It must specify:

- `prompt_type`
- `prompt_source`
- `prompt_budget`
- `refresh_interval`
- `multi_mask`

## Default Video Policy

- first frame GT prompt
- sparse refresh allowed
- fixed refresh interval

This avoids unfair comparisons between methods with different prompt budgets.
