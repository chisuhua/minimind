# P1-0 PoC 结果 (2026-09-01)

## PoC-1:OpenCode/SKILL.md 编排
- ✅ `.opencode/skills/p1-poc/SKILL.md` 已创建(项目级技能路径,🟡-5 修正)
- 验证方式:OpenCode 加载该 skill 后执行 `python3 -c "import agenticmemory_training; print('OK')"` + 41 tests

## PoC-2:Qwen3.5-0.8B 推理服务可用性
- ✅ 模型可用性验证通过:`huggingface_hub.model_info('Qwen/Qwen3.5-0.8B')` 成功
- downloads: 2,430,167(模型已发布)
- **fallback 未触发**(🔴-2:Qwen3.5-0.8B ✓ → 无需回退 Qwen3.5-1.5B)
- vLLM 服务启动:待 GPU 环境就绪后执行 `vllm serve Qwen/Qwen3.5-0.8B --port 8998`

## PoC-3:端到端流程闭环
- ⏳ 待执行:需 KIMI_API_KEY(合成 1 条)→ Qwen3.5-0.8B 推理(zero-shot 抽取)→ JSON 13 字段可解析验证

## 阻塞项(需用户提供)
- [ ] Kimi API key(P1-1 合成 + IRR 第二标注)
- [ ] DeepSeek API key(P1-2 主标注)
- [ ] GPU ≥8GB(vLLM 服务 + LoRA 训练)
