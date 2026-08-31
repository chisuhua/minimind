#!/usr/bin/env python3
"""P1 最小闭环实验 — 环境就绪检查脚本 (Check Env)

用途:
  在恢复 Part B (Task 5-12) 前,一键确认 4 类前置条件是否就绪:
    1. API key 有效性 (Kimi / DeepSeek — P1-1 合成 + P1-2 标注 + IRR)
    2. 计算资源 (GPU / CPU 内存 — PoC-2 vLLM + P1-4 LoRA)
    3. Python 依赖 (vllm / peft / transformers — 训练与推理)
    4. 模型可下载性 (Qwen3.5-0.8B — PoC-2 前置,含 fallback 链)

用法:
  python3 scripts/check_env.py            # 全量检查
  python3 scripts/check_env.py --json     # JSON 输出(供脚本消费)

退出码:
  0 = 全部就绪(可启动 Part B)
  1 = 部分就绪(输出阻塞项清单)
  2 = 严重缺失(无法启动)

关联:
  计划: docs/superpowers/plans/2026-08-31-p1-minimum-loop-fixes.md Task 5-12
  spec: docs/superpowers/specs/2026-08-31-p1-minimum-loop-fixes-design.md v0.3
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any


# ============================================================================
# 配置
# ============================================================================

# 模型选型 (D-A, AGENTS.md F-04 修订后) + fallback 链 (🔴-2)
MODEL_CHAIN = ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-1.5B"]

# 必须安装的依赖 (Part A 代码 + Part B 实验)
REQUIRED_PKGS = {
    "transformers": "模型加载",
    "torch": "推理/训练",
    "peft": "LoRA 训练",
}

# 可选依赖
OPTIONAL_PKGS = {
    "vllm": "PoC-2 推理服务 (GPU)",
    "bitsandbytes": "int8 量化降级 (内存不足时)",
    "accelerate": "多卡/加速",
    "datasets": "数据集加载",
}

# 内存需求计算 (Qwen3.5-0.8B: ~0.8B params)
FP32_BYTES_PER_PARAM = 4
BF16_BYTES_PER_PARAM = 2
MODEL_PARAMS = 0.8e9


# ============================================================================
# 数据模型
# ============================================================================


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    fatal: bool = False


@dataclass
class EnvReport:
    results: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str, fatal: bool = False) -> None:
        self.results.append(CheckResult(name, ok, detail, fatal))

    @property
    def fatal_missing(self) -> list[CheckResult]:
        return [r for r in self.results if not r.ok and r.fatal]

    @property
    def non_fatal_missing(self) -> list[CheckResult]:
        return [r for r in self.results if not r.ok and not r.fatal]

    @property
    def all_ok(self) -> bool:
        return not self.fatal_missing and not self.non_fatal_missing

    @property
    def exit_code(self) -> int:
        if self.all_ok:
            return 0
        if self.fatal_missing:
            return 2
        return 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [
                {"name": r.name, "ok": r.ok, "detail": r.detail, "fatal": r.fatal}
                for r in self.results
            ],
            "verdict": "ready" if self.all_ok else "blocked",
            "fatal_missing": [r.name for r in self.fatal_missing],
            "non_fatal_missing": [r.name for r in self.non_fatal_missing],
        }


# ============================================================================
# 检查项
# ============================================================================


def check_api_key(name: str, key: str | None, expected_prefix: str = "") -> str | None:
    """检查 API key 是否存在且格式基本合理(不做网络调用,避免计费/延迟)。"""
    if not key:
        return "未设置"
    if key.startswith("sk-"):
        return "已设置(格式 ok)"
    return f"已设置(前缀 '{key[:4]}...' 非 sk-,格式可疑)"


def probe_api(base_url: str, key: str, timeout: int = 8) -> tuple[bool, str]:
    """最小 chat/completions 探测(max_tokens=1),验证 key 有效性 + 余额可用。

    使用生成接口而非 /models 列表——因为 P1-2 教师标注需要实际生成
    (token 消耗),仅有模型列表访问权但余额不足(如 DeepSeek 402
    Insufficient Balance)时,标注仍无法执行。这是本脚本要暴露的真实阻塞。
    """
    import json
    import urllib.error
    import urllib.request

    try:
        req = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions",
            data=json.dumps({
                "model": "deepseek-chat" if "deepseek" in base_url else "kimi-k3",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
            }).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
            content = body["choices"][0]["message"]["content"][:20]
            return True, f"生成可用(response='{content}')"
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode()[:100]
        except Exception:
            pass
        return False, f"HTTP {e.code} {e.reason} {detail}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:60]}"


def check_gpu() -> tuple[bool, str]:
    """检测 NVIDIA GPU(nvidia-smi)。"""
    path = shutil.which("nvidia-smi")
    if not path:
        return False, "nvidia-smi 不存在(无 NVIDIA GPU 或驱动)"
    try:
        out = subprocess.run(
            [path, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            return True, f"GPU: {out.stdout.strip().splitlines()[0]}"
        return False, f"nvidia-smi 异常: {out.stderr.strip()[:80]}"
    except Exception as e:
        return False, f"nvidia-smi 执行失败: {type(e).__name__}"


def check_memory() -> tuple[bool, str]:
    """检查可用内存是否足够加载模型(fp32)。"""
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            mem_lines = {k: int(v.split()[0]) for k, v in
                         (line.split(":", 1) for line in f if line.strip())}
        avail_kb = mem_lines.get("MemAvailable", 0)
        avail_gb = avail_kb / 1024 / 1024
        fp32_needed = MODEL_PARAMS * FP32_BYTES_PER_PARAM / 1e9
        bf16_needed = MODEL_PARAMS * BF16_BYTES_PER_PARAM / 1e9
        if avail_gb >= fp32_needed * 1.2:
            return True, f"可用 {avail_gb:.1f} GB ≥ fp32 需求 {fp32_needed:.1f} GB"
        if avail_gb >= bf16_needed * 1.2:
            return False, (
                f"可用 {avail_gb:.1f} GB ≥ bf16 {bf16_needed:.1f} GB 但 < fp32 {fp32_needed:.1f} GB;"
                f"CPU 上 bf16 会 upcast 到 fp32,仍可能 OOM"
            )
        return False, (
            f"可用内存仅 {avail_gb:.1f} GB < bf16 需求 {bf16_needed:.1f} GB;"
            f"无 bitsandbytes 则无法加载 Qwen3.5-0.8B"
        )
    except Exception as e:
        return False, f"内存检测失败: {type(e).__name__}"


def check_pkg_installed(module_name: str) -> tuple[bool, str]:
    """检查 Python 包是否可导入(不真正 import,避免副作用)。"""
    try:
        import importlib.util

        spec = importlib.util.find_spec(module_name)
        if spec is None:
            return False, "未安装"
        # 尝试取版本(尽力而为)
        try:
            mod = __import__(module_name)
            ver = getattr(mod, "__version__", "?")
            return True, f"已安装 (v{ver})"
        except Exception:
            return True, "已安装(版本未知)"
    except Exception:
        return False, "检查失败"


def check_model_downloadable(model_id: str, timeout: int = 30) -> tuple[bool, str]:
    """验证模型可从 HF(Mirror)下载 config.json(最小文件)。"""
    os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    try:
        from huggingface_hub import hf_hub_download

        p = hf_hub_download(
            model_id, "config.json", cache_dir="/tmp/opencode/hf_cache"
        )
        size = os.path.getsize(p) if os.path.exists(p) else 0
        return True, f"✅ config.json 可下载 ({size} bytes)"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:80]}"


# ============================================================================
# 汇总
# ============================================================================


def run_checks() -> EnvReport:
    report = EnvReport()

    # 1. API key(Kimi / DeepSeek)
    kimi_key = os.environ.get("KIMI_API_KEY")
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
    report.add(
        "KIMI_API_KEY", bool(kimi_key),
        check_api_key("KIMI", kimi_key),
        fatal=False,
    )
    report.add(
        "DEEPSEEK_API_KEY", bool(deepseek_key),
        check_api_key("DEEPSEEK", deepseek_key),
        fatal=False,
    )
    # 实际连通性探测(只读 /models,不计费)
    if kimi_key:
        ok, det = probe_api("https://api.moonshot.cn/v1", kimi_key)
        report.add("Kimi API 连通性", ok, det, fatal=True)
    else:
        report.add("Kimi API 连通性", False, "未设置 key,跳过探测", fatal=True)
    if deepseek_key:
        ok, det = probe_api("https://api.deepseek.com/v1", deepseek_key)
        report.add("DeepSeek API 连通性", ok, det, fatal=False)
    else:
        report.add("DeepSeek API 连通性", False, "未设置 key,跳过探测", fatal=True)

    # 2. 计算资源
    ok, det = check_gpu()
    report.add("GPU (nvidia-smi)", ok, det, fatal=True)  # vLLM 必需
    ok, det = check_memory()
    report.add("内存 vs 模型", ok, det, fatal=False)

    # 3. Python 依赖
    for pkg, purpose in REQUIRED_PKGS.items():
        ok, det = check_pkg_installed(pkg)
        report.add(f"依赖 {pkg} ({purpose})", ok, det, fatal=True)
    for pkg, purpose in OPTIONAL_PKGS.items():
        ok, det = check_pkg_installed(pkg)
        report.add(f"可选 {pkg} ({purpose})", ok, det, fatal=False)

    # 4. 模型可下载性 (fallback 链)
    model_ok = False
    for i, model_id in enumerate(MODEL_CHAIN):
        ok, det = check_model_downloadable(model_id)
        label = "模型(首选)" if i == 0 else f"模型(回退 {i})"
        report.add(f"{label} {model_id}", ok, det, fatal=(i == 0))
        if ok:
            model_ok = True
            break
    if not model_ok and len(MODEL_CHAIN) > 1:
        report.add("模型 fallback 链", False, "全部不可下载,需 F-07 前置决策", fatal=True)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="P1 环境就绪检查")
    parser.add_argument("--json", action="store_true", help="JSON 输出")
    args = parser.parse_args()

    report = run_checks()

    if args.json:
        import json

        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print("=" * 60)
        print("P1 最小闭环实验 — 环境就绪检查")
        print("=" * 60)
        for r in report.results:
            mark = "✅" if r.ok else ("🔴" if r.fatal else "⚠️")
            print(f"  {mark} {r.name}: {r.detail}")
        print("-" * 60)
        if report.all_ok:
            print("✅ 全部就绪,可启动 Part B (计划 Task 5 PoC → Task 12 findings)")
        else:
            fatal = [r.name for r in report.fatal_missing]
            nonfatal = [r.name for r in report.non_fatal_missing]
            if fatal:
                print(f"🔴 严重缺失 ({len(fatal)} 项): {', '.join(fatal)}")
                print("   → 需先解决这些才能真正启动 Part B")
            if nonfatal:
                print(f"⚠️ 非致命缺失 ({len(nonfatal)} 项): {', '.join(nonfatal)}")
            print("   → 当前阻塞清单见 poc_results.md '阻塞项' 节")
        print("退出码:", report.exit_code)

    sys.exit(report.exit_code)


if __name__ == "__main__":
    main()