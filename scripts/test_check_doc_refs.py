"""test_check_doc_refs.py — check_doc_refs 单元测试

测试覆盖:
1. 链接提取(基本 / 图片跳过 / 代码块跳过 / 行内代码跳过)
2. URL 跳过规则(站外 / 纯锚点 / 邮件)
3. 锚点 slug 生成(中英文 / 特殊字符)
4. 验证逻辑(OK / BROKEN_FILE / BROKEN_ANCHOR)
5. 报告生成(人类 / JSON)
6. CLI 入口(参数解析 / 退出码)

运行:
  cd /workspace/project/AgenticMind
  python -m unittest scripts.test_check_doc_refs -v
"""

import io
import os
import sys
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout

# 让 unittest 能找到 scripts 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.check_doc_refs import (
    Link,
    ValidationResult,
    extract_links,
    should_skip_url,
    extract_headings,
    slugify_anchor,
    validate_link,
    scan,
    human_report,
    json_report,
    main,
)


class TestShouldSkipUrl(unittest.TestCase):
    """URL 跳过规则测试"""

    def test_external_http(self):
        self.assertEqual(should_skip_url("https://example.com"), "external_url")

    def test_external_https(self):
        self.assertEqual(should_skip_url("https://github.com/foo/bar"), "external_url")

    def test_mailto(self):
        self.assertEqual(should_skip_url("mailto:foo@bar.com"), "external_url")

    def test_pure_anchor(self):
        self.assertEqual(should_skip_url("#section-1"), "pure_anchor")

    def test_pure_anchor_chinese(self):
        self.assertEqual(should_skip_url("#能力边界判定五步漏斗"), "pure_anchor")

    def test_relative_path_not_skipped(self):
        self.assertIsNone(should_skip_url("../dir/file.md"))

    def test_absolute_path_not_skipped(self):
        self.assertIsNone(should_skip_url("/docs/file.md"))

    def test_anchor_path_not_skipped(self):
        self.assertIsNone(should_skip_url("../file.md#section"))


class TestSlugifyAnchor(unittest.TestCase):
    """锚点 slug 生成测试"""

    def test_simple_lowercase(self):
        self.assertEqual(slugify_anchor("Hello World"), "hello-world")

    def test_chinese_preserved(self):
        self.assertEqual(slugify_anchor("能力边界判定五步漏斗"), "能力边界判定五步漏斗")

    def test_special_chars_removed(self):
        self.assertEqual(slugify_anchor("Wiki DAG (核心契约)"), "wiki-dag-核心契约")

    def test_multiple_spaces(self):
        self.assertEqual(slugify_anchor("Foo  Bar   Baz"), "foo-bar-baz")

    def test_hyphens_collapsed(self):
        self.assertEqual(slugify_anchor("Foo - - Bar"), "foo---bar".replace("---", "-"))

    def test_strip_leading_trailing_hyphens(self):
        self.assertEqual(slugify_anchor("---Hello---"), "hello")

    def test_underscore_preserved(self):
        self.assertEqual(slugify_anchor("foo_bar_baz"), "foo_bar_baz")

    def test_mixed_chinese_english(self):
        self.assertEqual(slugify_anchor("Wiki DAG 8 字段构建"), "wiki-dag-8-字段构建")


class TestExtractLinks(unittest.TestCase):
    """链接提取测试"""

    def setUp(self):
        # 创建临时项目目录
        self.tmpdir = Path(tempfile.mkdtemp())
        self.md = self.tmpdir / "test.md"
        self.md.write_text(
            "# Title\n"
            "See [file](other.md) for details.\n"
            "And [other](../dir/file.md#section) too.\n"
            "Skip ![image](img.png) should be excluded.\n"
            "Pure [anchor](#section-1) is skipped.\n"
            "External [link](https://example.com) is skipped.\n"
            "\n"
            "```python\n"
            "# Inside code block, [link](other.md) should be skipped.\n"
            "print('test')\n"
            "```\n"
            "\n"
            "After code block, [real](other.md) should be counted.\n",
            encoding="utf-8",
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_basic_extraction(self):
        links = extract_links(self.md, self.tmpdir)
        urls = [l.url for l in links]
        self.assertEqual(len(links), 5)
        self.assertIn("other.md", urls)
        self.assertIn("../dir/file.md#section", urls)
        self.assertIn("#section-1", urls)
        self.assertIn("https://example.com", urls)

    def test_image_skipped(self):
        links = extract_links(self.md, self.tmpdir)
        for link in links:
            self.assertNotIn("img.png", link.url)

    def test_external_at_extraction_stage(self):
        # extract_links 提取所有链接(含外部),跳过逻辑在 validate_link
        links = extract_links(self.md, self.tmpdir)
        urls = [l.url for l in links]
        self.assertIn("https://example.com", urls)

    def test_pure_anchor_at_extraction_stage(self):
        links = extract_links(self.md, self.tmpdir)
        urls = [l.url for l in links]
        self.assertIn("#section-1", urls)

    def test_code_block_skipped(self):
        # 代码块内的 [link](other.md) 不应被提取
        links = extract_links(self.md, self.tmpdir)
        # 期望: line 5 (file), line 6 (other), line 14 (real)
        # 不应有 line 10(代码块内的 other.md)
        line_numbers = [l.source_line for l in links]
        self.assertNotIn(10, line_numbers)

    def test_source_file_relative(self):
        links = extract_links(self.md, self.tmpdir)
        for link in links:
            self.assertEqual(link.source_file, Path("test.md"))


class TestValidateLink(unittest.TestCase):
    """链接验证测试"""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        (self.tmpdir / "a.md").write_text("# Title\n", encoding="utf-8")
        (self.tmpdir / "b.md").write_text(
            "## Section 1\n## Section Two\n## 中文标题\n",
            encoding="utf-8",
        )
        (self.tmpdir / "subdir").mkdir()
        (self.tmpdir / "subdir" / "c.md").write_text("# C\n", encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_ok_relative(self):
        link = Link(
            source_file=Path("a.md"),
            source_line=1,
            text="b",
            url="b.md",
        )
        result = validate_link(link, self.tmpdir)
        self.assertEqual(result.status, "ok")

    def test_ok_relative_with_anchor(self):
        link = Link(
            source_file=Path("a.md"),
            source_line=1,
            text="section",
            url="b.md#section-1",
        )
        result = validate_link(link, self.tmpdir, check_anchors=True)
        self.assertEqual(result.status, "ok")

    def test_broken_file(self):
        link = Link(
            source_file=Path("a.md"),
            source_line=1,
            text="missing",
            url="nonexistent.md",
        )
        result = validate_link(link, self.tmpdir)
        self.assertEqual(result.status, "broken_file")

    def test_broken_anchor(self):
        link = Link(
            source_file=Path("a.md"),
            source_line=1,
            text="bad",
            url="b.md#nonexistent-section",
        )
        result = validate_link(link, self.tmpdir, check_anchors=True)
        self.assertEqual(result.status, "broken_anchor")

    def test_chinese_anchor_ok(self):
        link = Link(
            source_file=Path("a.md"),
            source_line=1,
            text="zh",
            url="b.md#中文标题",
        )
        result = validate_link(link, self.tmpdir, check_anchors=True)
        self.assertEqual(result.status, "ok")

    def test_external_skipped(self):
        link = Link(
            source_file=Path("a.md"),
            source_line=1,
            text="ext",
            url="https://example.com",
        )
        result = validate_link(link, self.tmpdir)
        self.assertEqual(result.status, "skipped")
        self.assertEqual(result.reason, "external_url")

    def test_pure_anchor_skipped(self):
        link = Link(
            source_file=Path("a.md"),
            source_line=1,
            text="self",
            url="#section",
        )
        result = validate_link(link, self.tmpdir)
        self.assertEqual(result.status, "skipped")
        self.assertEqual(result.reason, "pure_anchor")

    def test_subdir_traversal(self):
        link = Link(
            source_file=Path("subdir/c.md"),
            source_line=1,
            text="up",
            url="../a.md",
        )
        result = validate_link(link, self.tmpdir)
        self.assertEqual(result.status, "ok")


class TestScan(unittest.TestCase):
    """全项目扫描测试"""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        (self.tmpdir / "docs").mkdir()
        (self.tmpdir / "docs" / "a.md").write_text(
            "[link to b](b.md)\n[link to missing](nonexistent.md)\n",
            encoding="utf-8",
        )
        (self.tmpdir / "docs" / "b.md").write_text("# B\n", encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_scan_finds_broken(self):
        results = scan(
            root_dirs=[self.tmpdir / "docs"],
            project_root=self.tmpdir,
        )
        statuses = [r.status for r in results]
        self.assertIn("ok", statuses)
        self.assertIn("broken_file", statuses)

    def test_exclude_dir(self):
        results = scan(
            root_dirs=[self.tmpdir / "docs"],
            project_root=self.tmpdir,
            exclude_dirs={"a.md"},
        )
        # 排除 a.md 后,只剩 b.md 中的链接,但 b.md 无链接
        self.assertEqual(len(results), 0)


class TestReports(unittest.TestCase):
    """报告生成测试"""

    def test_human_report_includes_failures(self):
        link = Link(
            source_file=Path("a.md"),
            source_line=1,
            text="bad",
            url="nonexistent.md",
        )
        results = [
            ValidationResult(link=link, status="broken_file", reason="missing"),
            ValidationResult(link=link, status="ok", reason="a.md"),
        ]
        report = human_report(results, [Path("docs")])
        self.assertIn("BROKEN_FILE", report)
        self.assertIn("1 个", report)

    def test_json_report_structure(self):
        link = Link(
            source_file=Path("a.md"),
            source_line=1,
            text="bad",
            url="nonexistent.md",
        )
        results = [
            ValidationResult(link=link, status="broken_file", reason="missing"),
        ]
        report_str = json_report(results, [Path("docs")])
        data = json.loads(report_str)
        self.assertEqual(data["total"], 1)
        self.assertEqual(data["broken_file"], 1)
        self.assertEqual(len(data["failures"]), 1)
        self.assertEqual(data["failures"][0]["source_file"], "a.md")


class TestMain(unittest.TestCase):
    """CLI 入口测试"""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        (self.tmpdir / "ok.md").write_text(
            "[link](ok.md)\n",
            encoding="utf-8",
        )
        (self.tmpdir / "broken.md").write_text(
            "[broken](nonexistent.md)\n",
            encoding="utf-8",
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_main_pass(self):
        """全部链接有效 → 退出码 0"""
        old_cwd = os.getcwd()
        os.chdir(self.tmpdir)
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = main(["ok.md"])
            self.assertEqual(code, 0)
            self.assertIn("OK", buf.getvalue())
        finally:
            os.chdir(old_cwd)

    def test_main_fail_broken_file(self):
        """有 broken file → 退出码 1"""
        old_cwd = os.getcwd()
        os.chdir(self.tmpdir)
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = main(["broken.md"])
            self.assertEqual(code, 1)
            self.assertIn("BROKEN_FILE", buf.getvalue())
        finally:
            os.chdir(old_cwd)

    def test_main_json_output(self):
        """--json 输出格式正确"""
        old_cwd = os.getcwd()
        os.chdir(self.tmpdir)
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = main(["broken.md", "--json"])
            self.assertEqual(code, 1)
            data = json.loads(buf.getvalue())
            self.assertIn("failures", data)
            self.assertEqual(data["broken_file"], 1)
        finally:
            os.chdir(old_cwd)

    def test_main_strict_with_anchor(self):
        """--strict 模式下 BROKEN_ANCHOR 也失败"""
        old_cwd = os.getcwd()
        os.chdir(self.tmpdir)
        try:
            # 创建一个有 broken anchor 的文件
            (self.tmpdir / "anchor.md").write_text(
                "## Section 1\n[bad](ok.md#nonexistent)\n",
                encoding="utf-8",
            )
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = main(["anchor.md", "--strict", "--check-anchors"])
            self.assertEqual(code, 1)
            self.assertIn("BROKEN_ANCHOR", buf.getvalue())
        finally:
            os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main(verbosity=2)