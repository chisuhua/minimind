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

## P1-1 数据合成进展(2026-09-01)
- ✅ 腿 A 公开集完成:从 HuggingFaceH4/ultrachat_200k(test_gen)提取 40 条多轮对话(3-11 turns)
  - 来源:https://hf-mirror.com(HF 直连不可达,经镜像)
  - 文件:data/agenticmemory_training/v0/conversations.jsonl
  - 注意:计划首选 SHARELY(git 不可达,ls-remote 超时)与 lmsys-chat-1m(gated 403)均不可用,改用 ultrachat_200k(非 gated)
  - 格式:session_id / source=public:ultrachat / turns[]{role,text,timestamp} / metadata.teacher=public
  - 验证:格式兼容 P1-2 teacher_labeling 消费链(.get() 读取)
- ⏳ 腿 B 合成阻塞:需有效 Kimi API key(当前 401 Invalid Authentication)
- ⏳ P1-2 标注阻塞:需有效 DeepSeek(402 余额不足)+ Kimi(401)key
- ⏳ PoC-2 vLLM / P1-4 训练阻塞:无 GPU 环境,peft/transformers/vllm 未安装
