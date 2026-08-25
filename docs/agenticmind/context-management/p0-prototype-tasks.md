# P0 原型任务清单 — 抽取系统 v1 实施(interim 方案)

> **文档 ID**: CM-003-PROTOTYPE
> **生成日期**: 2026-08-24
> **状态**: 草案 v0.2(关键路径调整:T1.3/T1.4 改 interim 方案)
> **配套文档**:
> - `mvp-schema.md` — 13 字段人工 schema 单一真源
> - `architecture.md` — 编排架构 + 统一模型 + task tag(v0.2 重写)
> - `p0-prototype-tasks.md`(本文件)— v1 interim 实施方案
> - [`../../agenticmemory_training/08b-seed-schema-fusion.md`](../../agenticmemory_training/08b-seed-schema-fusion.md) — Schema 融合边界规范

> **v0.2 关键路径调整记录**:
> - **T1.3/T1.4 改 interim 方案**:不再训练独立 0.5B 分类器(intent-cls / entity-ner);统一模型(Qwen3-0.6B + 双 LoRA)未就绪时,用"教师 API + 规则"过渡
> - **模型选型统一**:所有提到 "0.5B" 的地方改为 "Qwen3-0.6B"(对齐 08a D-10 tokenizer 硬约束)
> - **CM-M1 验收调整**:interim 阶段 intent accuracy ≥85% 由**教师 API**保证,非本地模型
> - **新增依赖项**:"08 管线 Phase 5 产出统一模型后,替换 LocalExtractorPool 实现"(本文件标记为待替换任务)
> - **编排层价值保留**:OrchestratorInterface + 4 级降级 + PII/secret 仍由本任务验证(与抽取器实现解耦)
>
> **v0.1.1 补丁记录**(保留):
> - 时间线从"2-3 周"改为"3-4 周"(Oracle 指出工作量系统性低估)
> - T3.1 数据来源改为"公开+合成+内部"三腿
> - T3.2 标注改为 GPT-4 预标 + 人工抽检 20%
> - T1.3/T1.4 加 temperature scaling 要求(贯穿到 T3.4)
> - T1.6 规则引擎加 SecretDetector 强制项
> - T2.5 验证加 session_id 索引
> - §4 风险表补 R6(R3 序号让出给 R6)

---

## 0. 文档范围与定位

本文档定义 **P0 原型** 的具体任务清单——**v1 Python 编排的工程实施分解**。

**P0 原型目标(经 Oracle 评审 + 用户确认锁定)**:

> 在 **3-4 周**内,端到端跑通 **"对话 → 抽取 → prompt 组装 → (云端/本地) LLM"** 一条链路,用 **50 条合成 + 公开 + 内部会话** 验证抽取可用性,暴露问题反哺 schema 和训练数据设计。

**关键约束**:
- 不进入训练数据 schema 阶段(等原型暴露问题)
- 不绑定 HydraForge runtime(AGENTS.md §9 标"待开始")
- 不引入分层模型(统一 Qwen3-0.6B + task tag;interim 阶段用教师 API + 规则)
- 不实现 backlog 字段(只实现 MVP 13 个字段)

**目标读者**:AgenticMind P0 原型开发者(估计 1-2 人)。Oracle 建议 2 人(1 工程 + 0.5 标注)3 周达成,1 人需 4 周。

---

## 1. 总体时间线

```
Week 1: M1 完成        → 单字段抽取器 + Schema 校验器 + temperature scaling
Week 2: M2 完成        → PythonOrchestrator + 端到端 demo
Week 3: M3 完成        → 50 条会话验证 + 暴露问题清单
Week 4: 收尾           → findings_v1.md + 评估报告
```

| 里程碑 | 截止时间 | 关键产出 | 验收标准 |
|---|---|---|---|
| **M1: 单字段抽取器** | Week 1 结束 | `intent`, `entities`, `language` 三个抽取器(含 temp scaling)| intent accuracy ≥85%, entity F1 ≥0.8 |
| **M2: 端到端 demo** | Week 2 结束 | PythonOrchestrator + PromptAssembler + 4 级降级 | 10 条对话可走完整链路,降级链可触发 |
| **M3: 真实会话验证** | Week 3-4 结束 | 50 条验证报告 + 问题清单 | 字段填充率 L0 ≥90%, L1 ≥70%, ECE <0.1(directional)|

---

## 2. 任务分解(WBS)

### Phase 1:M1 — 单字段抽取器(Week 1)

#### T1.1 定义 Schema dataclass
- **目标**:实现 `mvp-schema.md` §3 的所有 dataclass
- **产出**:`context_extraction/schemas.py`(全部字段定义)
- **依赖**:无
- **时间**:0.5 天
- **验收**:
  - 所有 13 个字段都有 dataclass 定义
  - 通过 `json.dumps()` 序列化测试
  - `field_confidence` 在每个字段上都有,默认 1.0
- **关键代码**:
  ```python
  @dataclass
  class IntentField:
      value: IntentEnum  # 8 类
      confidence: float = 1.0
      provenance: ProvenanceTag = field(default_factory=ProvenanceTag)

  @dataclass
  class TurnContext:
      intent: IntentField
      entities: EntitiesField
      language: LanguageField
      routing_features: RoutingFeatures
      field_confidence: FieldConfidence
      extraction_provenance: Provenance
      privacy_tier: PrivacyTier
  ```

#### T1.2 训练数据合成
- **目标**:为 3 个抽取器(intent / entities / language)各准备 500-1000 条 SFT 样本
- **产出**:
  - `data/intent_train.jsonl`(500-1000 条,8 类平衡)
  - `data/entities_train.jsonl`(500-1000 条,含 NER + 规则)
  - `data/language_train.jsonl`(500 条,中英 code mixed)
- **依赖**:T1.1
- **时间**:1.5 天
- **验收**:
  - 类别分布:每个 intent 类 ≥50 条
  - 实体类型覆盖:8 种 entity 类型每种 ≥40 条
  - 语言分布:zh/en/code/mixed 每种 ≥100 条
- **数据源**:
  - intent:用 Qwen3-4B 标注对话数据(直接 prompt 分类)
  - entities:用公开 NER 数据集 + AgenticMind 项目代码注释
  - language:用现有公开语料

#### T1.3 训练统一抽取模型 — interim 方案(v0.2 重写)

**v0.1.1 原方案**(已废):训练独立 `Qwen2.5-0.5B` intent 分类器

**v0.2 interim 方案**:**不训练本地小模型**,改用教师 API + 规则引擎

- **目标**:实现 interim 抽取,使 LocalExtractorPool 在统一模型未就绪时可用
- **产出**: `context_extraction/interim_extractor.py`(教师 API + 规则融合实现)
- **依赖**:T1.2 完成
- **时间**:0.5 天
- **验收**:
  - intent 8 类分类通过**教师 API**(DeepSeek V4 Flash)调用
  - 教师 API 响应含 `confidence_per_label` 字段(8 类各一个概率)
  - 推理延迟 <500ms(单条,含 API 往返)
  - **降级路径**:API 不可用时 → 用正则启发式(仅识别 question/command/chat/refuse 4 类,其他标 "unknown")
- **统一模型就绪后**:此任务整体替换为"调用 Qwen3-0.6B + LoRA_session"(由 08 管线 Phase 5 产出)

#### T1.4 训练统一抽取模型 — interim 方案(v0.2 重写)

**v0.1.1 原方案**(已废):训练 0.5B NER 模型覆盖 9 种实体类型

**v0.2 interim 方案**:**不训练本地小模型**,改用教师 API + 规则引擎

- **目标**:实现 interim 实体抽取(覆盖 9 种类型:secret/person/project/file_path/url/version/code_symbol/api/org)
- **产出**: `context_extraction/interim_entity_extractor.py`(教师 API + SecretDetector + 规则融合)
- **依赖**:T1.2, T1.6 完成
- **时间**:0.5 天
- **验收**:
  - 实体识别通过教师 API 调用
  - **SecretDetector 强制保留**(覆盖 AWS/GCP/Azure/GitHub PAT/OpenAI/JWT/PEM 私钥)
  - 规则融合:file_path/url/code_symbol 由规则引擎补充(避免 NER 漏报)
  - 推理延迟 <500ms(单条,含 API 往返)
  - secret recall ≥95%(漏报代价 >> 误报)
- **统一模型就绪后**:替换为"调用 Qwen3-0.6B + LoRA_memory + 规则后处理"

#### T1.5 训练/部署 language 检测器
- **目标**:语言检测器(interim:直接用 fasttext;统一模型就绪后由模型自带)
- **产出**: `models/lang-detect-0.3b-v1/` 或 `lang_detect.py`(fasttext 包装)
- **依赖**:无(可用 fasttext 跳过训练)
- **时间**:0.5 天
- **验收**:
  - 5 类(zh/en/code/mixed/other)准确率 ≥95%
  - 推理延迟 <20ms
  - **可与 T1.2-T1.4 并行**(独立模块)

#### T1.6 规则引擎 + SecretDetector(v0.1.1 必加)
- **目标**:实现 PathParser, CodeFenceDetector, URLDetector, **SecretDetector** 等正则规则
- **产出**: `context_extraction/rules.py`
- **依赖**:无
- **时间**:0.5 天
- **验收**:
  - 单元测试覆盖每条规则
  - 规则与 NER 融合测试(避免重复)
  - **SecretDetector 必须实现**(覆盖 AWS/GCP/Azure/GitHub PAT/OpenAI/JWT/PEM 私钥)
  - secret 检测即使 NER 置信度低也强制保留(漏报代价 >> 误报)
  - **可与 T1.2-T1.4 并行**(独立模块)

#### T1.7 Schema 校验器
- **目标**:实现 `SchemaValidator`,逐字段 confidence 检查 + 降级决策
- **产出**: `context_extraction/validator.py`
- **依赖**:T1.1
- **时间**:0.5 天
- **验收**:
  - 单元测试覆盖 4 级降级阶梯
  - 阈值配置可从 YAML 读取

### Phase 2:M2 — 端到端 demo(Week 2)

#### T2.1 RoutingFeatures 计算
- **目标**:实现 `compute_routing_features(raw_input)` 函数
- **产出**: `context_extraction/features.py`
- **依赖**:T1.1
- **时间**:0.5 天
- **验收**:
  - 6 个 routing_features 全部实现
  - 单元测试覆盖典型输入(短/长/code/multilingual)

#### T2.2 PythonOrchestrator 实现
- **目标**:实现 `PythonOrchestrator` 类(v1)
- **产出**: `context_extraction/orchestrator_v1.py`
- **依赖**:T1.3, T1.4, T1.5, T1.6, T2.1
- **时间**:1 天
- **验收**:
  - 完整实现 OrchestratorInterface(plan / execute / update_session / get_degradation_level)
  - 单元测试覆盖 local_only / hybrid / cloud_forward 三条路径
  - 代码量 ≤200 行(Oracle 要求 50 行太严,但 ≤200 是合理上限)

#### T2.3 LocalPIIRedactor + 隐私映射表(v0.1.1 修订)
- **目标**:实现本地 PII 脱敏器 + placeholder 映射表
- **产出**: `context_extraction/privacy.py`
- **依赖**:无
- **时间**:1.5 天(原 1 天,Oracle 指出 99.5% 召回率 1 天不可达)
- **验收**:
  - 脱敏类型:**人名、电话、邮箱、身份证、内部 URL、secret**(新增 secret 覆盖)
  - **Smoke test**:100 条人工标注样本召回率 ≥99.5%(只作 smoke,统计意义有限)
  - **正式测量挪到 M3 之后**,Phase 1 阶段用 600+ 样本做召回率评估(100 条仅给 95% CI 上界)
  - 反匿名化通过映射表还原
- **关键测试**:
  ```python
  test_pii_redaction_recall_smoke()  # 100 条 smoke test
  test_pii_anonymize_roundtrip()     # 反匿名化还原
  test_privacy_white_list()          # 白名单制验证
  test_secret_redaction_priority()   # secret 必须被脱敏,即使置信度低
  ```

#### T2.4 CloudForwarder 客户端
- **目标**:实现云端转发客户端(deepseek-v3 或 qwen3)
- **产出**: `context_extraction/cloud_forwarder.py`
- **依赖**:T2.3
- **时间**:0.5 天
- **验收**:
  - 超时控制(默认 3s,可配置)
  - 成本上限(默认 $0.001/turn)
  - 白名单检查(隐私)
  - 重试 + 降级到 local

#### T2.5 SessionState 管理(v0.1.1 加 session_id)
- **目标**:实现 `current_topic`, `session_facts`, `near_turn_entities` 的更新逻辑
- **产出**: `context_extraction/session_state.py`
- **依赖**:T1.1
- **时间**:1.5 天(原 1 天,SQLite 单测要覆盖并发)
- **验收**:
  - `update_session(turn_ctx, session_state) -> SessionState`
  - 单测覆盖 topic 切换、facts 增改、entities 滑动窗口
  - 持久化:**SQLite,以 session_id 为主键**(v0.1.1 新增要求)
  - 支持多会话隔离(session_id 查询/创建/删除)
  - 单线程 demo 下"无并发污染"暂作 N/A(M3 后补并发测试)

#### T2.6 PromptAssembler
- **目标**:实现 `inja` 模板渲染 + 4 级降级适配
- **产出**: `context_extraction/prompt_assembler.py`
- **依赖**:T2.2, T2.5
- **时间**:1 天
- **验收**:
  - 4 级降级都有对应模板
  - 模板中字段缺失时优雅降级(不崩)
  - 单测覆盖每级降级 + 全字段组合

#### T2.7 端到端 demo 脚本
- **目标**:一个可执行的 demo,接收对话 → 输出最终 prompt
- **产出**: `demo/run_e2e.py`
- **依赖**:T2.2, T2.4, T2.5, T2.6
- **时间**:0.5 天
- **验收**:
  - 接收 JSON: `{"turns": [...], "new_turn": "..."}`
  - 输出 JSON: `{"final_prompt": "...", "degradation_level": "...", "routing_path": "..."}`
  - 10 条对话可走通,无崩溃

### Phase 3:M3 — 真实会话验证(Week 3-4,v0.1.1 修订)

#### T3.1 真实会话数据收集(v0.1.1 三腿并行)
- **目标**:收集 50 条对话数据(三条腿并行)
- **产出**: `eval/real_conversations.jsonl`
- **依赖**:无(可并行)
- **时间**:1 天
- **验收**:
  - ≥50 条会话,平均 5-15 轮
  - 覆盖 5+ 主题(AgenticDSL / 训练 / 评测 / 推理 / 项目治理)
  - 含 ≥10 条代码相关会话
  - **三条腿**(避免单源风险):
    - **腿 A 公开集**:开源 assistant 对话集(SHARELY / MultiWOZ 等,过滤 PII 后)
    - **腿 B 合成**:GPT-4 扮演用户 + 助手,生成 20 条覆盖各意图类的合成对话
    - **腿 C 内部试用**:团队内部 10 条真实试用(可脱敏)
- **v0.1.1 原因**:AgenticMind 是 pre-launch 项目,"从 GitHub Issues 收集 50 条真实 AgenticMind 对话"很可能是空头承诺

#### T3.2 人工标注 ground truth(v0.1.1 改为预标 + 抽检)
- **目标**:对 50 条会话标注 intent / entities / language / topic / facts
- **产出**: `eval/ground_truth.jsonl`
- **依赖**:T3.1
- **时间**:2 天(原 1 天)
- **验收**:
  - 每条会话标注 intent(primary + secondary)+ entities + topic + 关键 facts
  - **采用 GPT-4 预标注 + 人工抽检 20% 模式**(每条 1-2 人天手工不可达)
  - 抽检一致率 ≥85%
  - 标注集格式:`{"session_id": str, "turns": [{"turn_index": int, "intent": {...}, "entities": [...], ...}]}`

#### T3.3 抽取质量评估
- **目标**:在 50 条 ground truth 上评估抽取器准确率
- **产出**: `eval/results_v1.md`(评估报告)
- **依赖**:T3.2 + M2 完成
- **时间**:1 天
- **验收**:
  - intent accuracy ≥85%(primary 标签)
  - intent multi-label recall ≥80%(检查 secondary 是否被识别)
  - entity F1 ≥0.8,secret recall ≥95%(漏报代价 >> 误报)
  - language accuracy ≥95%
  - topic accuracy(用户认为对的) ≥70%
  - 字段填充率:L0 ≥90%, L1 ≥70%
  - 4 级降级触发频率分布

#### T3.4 Confidence 校准(v0.1.1 提为关键路径)
- **目标**:验证 predicted confidence vs actual accuracy
- **产出**: 校准曲线图 + 报告
- **依赖**:T3.3
- **时间**:1 天(原 0.5 天,因 temperature scaling 校准是核心)
- **验收**:
  - 50 条样本的 ECE < 0.1(方向性指标,统计噪声大,正式报告在 Phase 1 用 500+ 样本)
  - 校准曲线可视化(可靠性图)
  - **temperature scaling 必须已在 T1.3/T1.4 中拟合**,此处只验证校准效果
  - **若 ECE > 0.15**,触发 confidence 重新校准或 reject option 调整

#### T3.5 问题清单与改进建议
- **目标**:基于验证结果,产出改进建议文档
- **产出**: `eval/findings_v1.md`
- **依赖**:T3.3, T3.4
- **时间**:1 天(原 0.5 天,扩充问题模板)
- **验收**:
  - 列出 ≥10 个具体问题(字段缺失 / 抽取错误 / 降级过多 / confidence 不准)
  - 每个问题给出改进建议
  - 哪些 schema 字段应调整 / 哪些 backlog 字段应提前加入
  - **新增模板项**:
    - "降级触发时的人工评价"(4 级降级各自的回复质量损失评估)
    - "SFT 污染检测流程"(extraction_provenance 字段的版本比对)
    - "v2 HydraForge 编排的预研"(若 M3 暴露编排问题)

---

## 3. 任务依赖图

```
T1.1 ─┬─ T1.2 ─┬─ T1.3 ─┐
      │        ├─ T1.4 ─┤
      │        └─ T1.5 ─┤
      ├─ T1.6 ──────────┼─ T2.2 ─┬─ T2.7 ─┐
      └─ T1.7 ──────────┤        ├─ T2.4 ─┤
                         ├─ T2.5 ─┤        │
                         │        ├─ T2.6 ─┤
                         │        └─────────┤
                         ├─ T2.3 ───────────┤
                         ├─ T2.1 ───────────┤
                         │                  ▼
                         │                M2 ✓
                         ▼                │
                       (Week 2)          ▼
                                          │
                                          ▼
                                       T3.1 ─ T3.2 ─ T3.3 ─ T3.4 ─ T3.5
                                                              ▼
                                                             M3 ✓
```

---

## 4. 关键风险与缓释

| 风险 | 影响 | 缓释 |
|---|---|---|
| **R1:0.5B 模型抽取质量不达标** | M1 验收失败 | 准备 fallback:用 Qwen3-4B 通过 vLLM 跑(慢但可用) |
| **R2:PII 脱敏召回率 <99.5%(100 条样本 smoke 不代表真实)** | T2.3 验收失败 | 推迟云端转发,先全本地路径跑通;Phase 1 用 600+ 样本做正式测量 |
| **R3:真实会话数据不足(AgenticMind 是 pre-launch)** | T3.1 延期 | 三腿并行:公开集 + 合成 + 内部试用,降低单源风险 |
| **R4:端到端延迟 >500ms** | M2 验收失败 | 优化:批处理 + 模型并行 + 缓存 |
| **R5:字段 confidence 普遍偏低/不校准(v0.2 调整)** | T3.4 失败 + 4 级降级机制失效 | interim 阶段由教师 API 自带校准;统一模型就绪后做 temperature scaling;T3.4 验证 ECE |
| **R6:HydraForge runtime CLI 未交付(v2 阻塞)** | v2 启动条件不满足 | **不影响 P0**(v1 不依赖),但阻碍 v2 启动;跟踪 AGENTS.md §9 状态 |

---

## 5. 资源需求

| 资源 | 用途 | 数量 |
|---|---|---|
| GPU | 模型微调(A100/H100) | 1 张,Week 1 共 ~4 GPU-day |
| 标注人力 | 50 条会话 ground truth | ~1 人天 |
| 云端 LLM 配额 | GPT-4 / Qwen3-4B 标注 | ~1000 次调用 |
| 数据存储 | SQLite / JSONL | <100MB |

---

## 6. M3 验收后下一步(M3 → M4 的衔接)

**M3 完成后,基于暴露的问题决定下一步**:

- **如果质量达标**:进入 Phase 2 —— 训练数据合成 + 模型训练 + HydraForge 编排(若 runtime 可用)
- **如果质量问题集中在某字段**:针对性补字段(如 unknowns 加入 MVP)
- **如果 confidence 校准差**:重新训练 + 加 reject option
- **如果降级触发频繁**:调阈值或加规则路由复杂度

**M3 暴露的问题清单(`eval/findings_v1.md`)将作为后续所有决策的输入**。

---

## 7. 决策检查清单

P0 启动前必须确认:

- [ ] **资源到位**:1 张 A100 + 1 人开发 + 标注预算
- [ ] **Schema 已冻结**:`mvp-schema.md` v0.1.1 已批准
- [ ] **架构已冻结**:`architecture.md` v0.1.1 已批准
- [ ] **训练数据源已对齐**:Qwen3-4B 标注、公开 NER 数据集、AgenticMind 项目代码
- [ ] **PII 脱敏评测集已有**:1000 条样本(可先人工标注 100 条起步)
- [ ] **真实会话数据已脱敏**:50-100 条 AgenticMind 相关对话

---

## 8. 文档边界

| 内容 | 归属 |
|---|---|
| **P0 原型任务清单**(本文件) | AgenticMind 仓 `context-management/p0-prototype-tasks.md` |
| **MVP 字段集** | `mvp-schema.md` |
| **编排架构** | `architecture.md` |
| **评估结果** | `eval/results_v1.md`(M3 产出) |
| **问题清单** | `eval/findings_v1.md`(M3 产出) |
| **训练数据合成 SOP** | `training-data-synthesis.md`(M3 后产出) |

---

**文档版本**:v0.2(关键路径调整:T1.3/T1.4 改 interim 方案 + 模型选型统一为 Qwen3-0.6B)
**Owner**:AgenticMind P0 原型组
**下一步**:确认资源到位后,启动 T1.1(Schema dataclass)+ T1.6(规则引擎 + SecretDetector)