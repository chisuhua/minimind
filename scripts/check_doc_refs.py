"""check_doc_refs.py — 自动化交叉引用验证工具

## 目的

防止 Markdown 文档中出现 "Wiki DAG → 08a §3" 类错误引用(2026-08-26 发现)。
本工具扫描项目内所有 .md 文件,验证 `[text](url)` 形式的链接是否指向真实存在
的文件(可选验证锚点),并生成可被 CI 消费的退出码与 JSON 报告。

## 使用方式

```bash
# 默认:扫描 docs/、agenticmind/、agenticmemory_training/ 下所有 .md
python -m scripts.check_doc_refs

# 指定目录
python -m scripts.check_doc_refs docs/ agenticmind/

# JSON 输出(CI 消费)
python -m scripts.check_doc_refs --json

# 严格模式(警告也失败)
python -m scripts.check_doc_refs --strict

# 启用锚点验证(检查 #section 是否存在)
python -m scripts.check_doc_refs --check-anchors

# 排除目录
python -m scripts.check_doc_refs --exclude scripts/ dataset/
```

## 退出码

- 0 = PASS(无错误,允许有 WARN)
- 1 = FAIL(有 BROKEN_FILE 错误)
- 2 = ERROR(工具本身异常)

## 历史

- 2026-08-26:初始版本(因 docs/agenticmemory/ 多处指向
  agenticmemory_training/ 目录但该目录无对应文档而创建)
"""

import os
import re
import sys
import json
import argparse
import dataclasses
from pathlib import Path
from typing import List, Dict, Set, Optional, Iterable, Tuple

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Markdown 链接提取 ---

# 匹配 Markdown 行内链接: `[text](url)` 或 `![alt](url)` (图片)
# - 排除代码块内的链接(简单实现:不在 ``` 包围的行内提取)
# - url 支持 path/to/file.md、file.md#anchor、file.md#anchor、#anchor-only
LINK_RE = re.compile(r"(?<!!)\[([^\]]*)\]\(([^)]+)\)")
# 匹配 Markdown 标题(用于锚点验证)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)(?:\s+#*)?$", re.MULTILINE)


@dataclasses.dataclass
class Link:
    """Markdown 中的一个链接"""
    source_file: Path        # 链接所在文件(相对项目根)
    source_line: int         # 行号(1-based)
    text: str                # 链接文本(用于报告)
    url: str                 # 原始 url

    @property
    def display(self) -> str:
        return f"{self.source_file}:{self.source_line}"


@dataclasses.dataclass
class ValidationResult:
    """链接验证结果"""
    link: Link
    status: str              # "ok" | "broken_file" | "broken_anchor" | "skipped"
    reason: str              # "ok" 的具体路径 / "broken_file" 的原因 / "skipped" 的原因

    @property
    def is_failure(self) -> bool:
        return self.status in ("broken_file", "broken_anchor")


def extract_links(file_path: Path, project_root: Path) -> List[Link]:
    """从单个 Markdown 文件提取所有非图片链接,跳过代码块"""
    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []

    links: List[Link] = []
    in_code_block = False
    try:
        rel_source = file_path.relative_to(project_root)
    except ValueError:
        rel_source = file_path

    for line_no, line in enumerate(content.splitlines(), start=1):
        # 跟踪代码块状态
        stripped = line.lstrip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        # HTML 注释内的链接也跳过
        if "<!--" in line and "-->" in line:
            continue

        for match in LINK_RE.finditer(line):
            text, url = match.group(1), match.group(2).strip()
            # 跳过 inline code `text` 内的链接(已在代码块逻辑中处理,这里兜底)
            if "`" in line.split(match.group(0))[0].split("`")[-1]:
                continue
            links.append(Link(
                source_file=rel_source,
                source_line=line_no,
                text=text,
                url=url,
            ))

    return links


def should_skip_url(url: str) -> Optional[str]:
    """判断 url 是否应该跳过(站外/邮件/纯锚点)。返回 skip 原因,None 表示不跳过"""
    # 纯锚点 (e.g. "#section") — 不跨文件,无法验证目标文件
    if url.startswith("#"):
        return "pure_anchor"
    # 站外 URL
    if url.startswith(("http://", "https://", "ftp://", "mailto:", "tel:")):
        return "external_url"
    # 站内协议相对(暂不处理)
    if url.startswith("//"):
        return "protocol_relative"
    return None


def extract_headings(file_path: Path) -> Set[str]:
    """提取 Markdown 文件中所有标题,生成 GitHub/MkDocs 风格的 anchor slug 集合"""
    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return set()

    anchors = set()
    for match in HEADING_RE.finditer(content):
        title = match.group(2).strip()
        anchors.add(slugify_anchor(title))

    # 也支持显式定义的锚点: <a id="anchor"></a> 或 {#anchor}
    explicit_re = re.compile(r'(?:<a\s+id="([^"]+)"\s*/?>|\{#([a-z0-9_-]+)\})', re.IGNORECASE)
    for match in explicit_re.finditer(content):
        anchors.add(match.group(1) or match.group(2))

    return anchors


def slugify_anchor(title: str) -> str:
    """GitHub 风格的 anchor slug 生成(简化版,支持中文)

    规则:
    - 转小写
    - 空格 → '-'
    - 移除大部分特殊字符(保留中文、字母、数字、`-`、`_`)
    - 连续 `-` 合并
    """
    s = title.lower()
    # 保留中文、字母、数字、连字符、下划线
    s = re.sub(r"[^\w\s\u4e00-\u9fff-]", "", s, flags=re.UNICODE)
    # 空白 → 连字符
    s = re.sub(r"\s+", "-", s)
    # 合并连续连字符
    s = re.sub(r"-+", "-", s)
    # 去除首尾连字符
    return s.strip("-")


def validate_link(
    link: Link,
    project_root: Path,
    check_anchors: bool = False,
    heading_cache: Optional[Dict[Path, Set[str]]] = None,
) -> ValidationResult:
    """验证一个链接是否有效"""
    skip_reason = should_skip_url(link.url)
    if skip_reason:
        return ValidationResult(link=link, status="skipped", reason=skip_reason)

    # 解析 url:可能有 #anchor 后缀
    if "#" in link.url:
        url_path, anchor = link.url.split("#", 1)
    else:
        url_path, anchor = link.url, None

    # 解析目标文件路径(相对于源文件)
    if url_path.startswith("/"):
        # 根相对
        target = project_root / url_path.lstrip("/")
    else:
        # 相对路径(相对于源文件所在目录)
        source_abs = (project_root / link.source_file).resolve()
        target = (source_abs.parent / url_path).resolve()

    # 验证文件存在
    if not target.exists():
        return ValidationResult(
            link=link,
            status="broken_file",
            reason=f"目标文件不存在:{url_path} (解析为 {target.relative_to(project_root)})",
        )

    # 验证锚点(可选)
    if check_anchors and anchor:
        if heading_cache is None:
            heading_cache = {}
        if target not in heading_cache:
            heading_cache[target] = extract_headings(target)
        if slugify_anchor(anchor) not in heading_cache[target]:
            return ValidationResult(
                link=link,
                status="broken_anchor",
                reason=f"锚点 #{anchor} 在 {target.relative_to(project_root)} 中不存在",
            )

    return ValidationResult(link=link, status="ok", reason=str(target.relative_to(project_root)))


def scan(
    root_dirs: List[Path],
    project_root: Path,
    check_anchors: bool = False,
    exclude_dirs: Optional[Set[str]] = None,
) -> List[ValidationResult]:
    """扫描所有 .md 文件并验证链接"""
    if exclude_dirs is None:
        exclude_dirs = set()

    # 收集所有 .md 文件
    md_files: List[Path] = []
    for root_dir in root_dirs:
        if not root_dir.exists():
            continue
        if root_dir.is_file():
            md_files.append(root_dir)
            continue
        for path in root_dir.rglob("*.md"):
            # 跳过排除目录
            if any(ex in str(path) for ex in exclude_dirs):
                continue
            md_files.append(path)

    # 提取所有链接
    all_links: List[Link] = []
    for md_file in md_files:
        all_links.extend(extract_links(md_file, project_root))

    # 验证(带锚点缓存)
    heading_cache: Dict[Path, Set[str]] = {}
    results: List[ValidationResult] = []
    for link in all_links:
        results.append(validate_link(
            link=link,
            project_root=project_root,
            check_anchors=check_anchors,
            heading_cache=heading_cache,
        ))

    return results


# --- 报告输出 ---

def human_report(results: List[ValidationResult], root_dirs: List[Path]) -> str:
    """生成人类可读报告"""
    total = len(results)
    failures = [r for r in results if r.status == "broken_file"]
    anchor_failures = [r for r in results if r.status == "broken_anchor"]
    skipped = [r for r in results if r.status == "skipped"]
    ok = [r for r in results if r.status == "ok"]

    lines = []
    lines.append(f"🔍 检查文档交叉引用")
    lines.append(f"  扫描目录: {', '.join(str(d.relative_to(Path.cwd())) if d.is_relative_to(Path.cwd()) else str(d) for d in root_dirs)}")
    lines.append(f"  扫描链接: {total} 个")
    lines.append(f"  ✅ OK: {len(ok)} 个")
    lines.append(f"  ⚠️  跳过: {len(skipped)} 个(站外/纯锚点)")
    lines.append(f"  ❌ BROKEN_FILE: {len(failures)} 个")
    if anchor_failures:
        lines.append(f"  ❌ BROKEN_ANCHOR: {len(anchor_failures)} 个")
    lines.append("")

    # 按源文件分组显示失败
    if failures or anchor_failures:
        all_broken = failures + anchor_failures
        # 按 source_file + source_line 排序
        all_broken.sort(key=lambda r: (str(r.link.source_file), r.link.source_line))

        # 按文件分组
        current_file = None
        for r in all_broken:
            if r.link.source_file != current_file:
                current_file = r.link.source_file
                lines.append(f"📄 {current_file}")
            icon = "❌" if r.status == "broken_file" else "🔗"
            lines.append(f"  {icon} [{r.status.upper()}] L{r.link.source_line}")
            lines.append(f"     URL: [{r.link.text}]({r.link.url})")
            lines.append(f"     → {r.reason}")

    return "\n".join(lines)


def json_report(results: List[ValidationResult], root_dirs: List[Path]) -> str:
    """生成 JSON 报告(CI 消费)"""
    return json.dumps({
        "total": len(results),
        "ok": sum(1 for r in results if r.status == "ok"),
        "skipped": sum(1 for r in results if r.status == "skipped"),
        "broken_file": sum(1 for r in results if r.status == "broken_file"),
        "broken_anchor": sum(1 for r in results if r.status == "broken_anchor"),
        "scan_dirs": [str(d) for d in root_dirs],
        "failures": [
            {
                "source_file": str(r.link.source_file),
                "source_line": r.link.source_line,
                "link_text": r.link.text,
                "link_url": r.link.url,
                "status": r.status,
                "reason": r.reason,
            }
            for r in results
            if r.is_failure
        ],
    }, ensure_ascii=False, indent=2)


# --- CLI ---

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="检查 Markdown 文档的交叉引用是否有效",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m scripts.check_doc_refs
  python -m scripts.check_doc_refs docs/
  python -m scripts.check_doc_refs --json --check-anchors
        """,
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["docs/", "agenticmind/", "agenticmemory_training/"],
        help="要扫描的目录或文件(默认: docs/ agenticmind/ agenticmemory_training/)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="输出 JSON 报告(供 CI 消费)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="严格模式:WARN(锚点不存在)也视为失败",
    )
    parser.add_argument(
        "--check-anchors",
        action="store_true",
        help="启用锚点验证(检查 #section 是否在目标文件存在)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="排除目录(可多次指定,如 --exclude scripts/ --exclude dataset/)",
    )
    args = parser.parse_args(argv)

    project_root = Path.cwd()
    root_dirs = [Path(p) for p in args.paths]
    exclude_dirs = set(args.exclude)

    # 验证: 启用锚点 or 严格模式
    check_anchors = args.check_anchors or args.strict

    try:
        results = scan(
            root_dirs=root_dirs,
            project_root=project_root,
            check_anchors=check_anchors,
            exclude_dirs=exclude_dirs,
        )
    except Exception as e:
        print(f"❌ ERROR: 扫描失败:{e}", file=sys.stderr)
        return 2

    # 输出
    if args.json:
        print(json_report(results, root_dirs))
    else:
        print(human_report(results, root_dirs))

    # 退出码
    failures = [r for r in results if r.status == "broken_file"]
    anchor_failures = [r for r in results if r.status == "broken_anchor"]

    if failures:
        return 1
    if args.strict and anchor_failures:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())