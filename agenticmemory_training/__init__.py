"""AgenticMind 训练侧骨架(P1 最小闭环实验)

设计定位:
- 仅服务于训练数据蒸馏与模型微调(Phase 1 训练管线)
- 与运行时侧(`context-management/architecture.md`)通过共享契约
  `agenticmind/extraction/` 解耦

子模块:
- data/               P1-1 数据合成 + P1-2 教师标注 + P1-3 评估
- training/           P1-4 LoRA 微调 + 字段级 F1

参考:
- docs/agenticmemory_training/08c-p1-minimum-loop.md(P1 骨架说明)
- docs/agenticmemory_training/08b-seed-schema-fusion.md §3
- docs/agenticmind/context-management/mvp-schema.md §3

不要在此包内放运行时编排代码;运行时侧待 P2 落
`agenticmind_runtime/`(预留)。
"""