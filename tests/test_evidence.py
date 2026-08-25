"""The extraction tooling, tested on the failure modes that would fake a result.

Every bug this file guards against produces the same visible outcome -- a
benchmark the provider did report gets recorded as unreported -- and that is
the one error direction that manufactures evidence for the study's hypothesis.
So these are not incidental utility tests.
"""

import pytest

from src.benchmark_aliases import ALIASES, hits, is_weak
from src.extract_evidence import (MIN_PLAUSIBLE_CHARS, guard, html_to_text,
                                  image_tables)


class TestAliases:
    def test_every_panel_slug_has_aliases(self):
        """A slug with no aliases is invisible to the coder and reads as absent."""
        import pandas as pd

        from src.config import INTERIM

        path = INTERIM / "panel.csv"
        if not path.exists():
            pytest.skip("panel not built")
        missing = sorted(set(pd.read_csv(path)["slug"]) - set(ALIASES))
        assert not missing, f"no search aliases for {missing}"

    @pytest.mark.parametrize("text,slug", [
        ("the matrix is symmetric", "metr_time_horizons"),
        ("a course in mathematics", "math_level_5"),
        ("rows were dropped silently", "gdpval"),
        ("as noted earlier in the text", "rli"),
        ("the answer is gsoc adjacent", "gso"),
    ])
    def test_substrings_do_not_match(self, text, slug):
        assert slug not in hits(text)

    @pytest.mark.parametrize("text,slug", [
        ("GPQA Diamond 88.7%", "gpqa_diamond"),
        ("SWE-bench Verified 74.9", "swe_bench_verified"),
        ("Humanity's Last Exam 26.5", "hle"),
        ("Terminal-Bench 2.0 65.4", "terminalbench"),
        ("ARC-AGI-2 (Verified) 68.8", "arc_agi_2"),
        # surface forms that dropped a word and were missed once
        ("SWE Verified (Resolved) | 80.6", "swe_bench_verified"),
        ("OSWorld-Verified 72.7%", "os_world"),
        ("GDPval-AA (Elo) 1554", "gdpval"),
        ("SimpleQA-Verified (Pass@1) 57.9", "simpleqa_verified"),
        ("Aider-Polyglot (Acc.) 49.6", "aider_polyglot"),
        ("Terminal Bench 2.0 (Acc) 67.9", "terminalbench"),
    ])
    def test_real_mentions_match(self, text, slug):
        assert slug in hits(text)

    def test_line_broken_term_still_matches(self):
        assert "swe_bench_verified" in hits("scores on SWE-bench\n   Verified were")

    def test_weak_aliases_are_flagged(self):
        assert is_weak("math_level_5", "MATH")
        assert not is_weak("math_level_5", "MATH-500")


class TestGuard:
    @pytest.mark.parametrize("body", [
        "Access to model meta-llama/Llama-3 is restricted. You must have "
        "access to it and be authenticated to access it. Please log in." * 40,
        "Just a moment... Enable JavaScript and cookies to continue" * 60,
        "404 Not Found. The page you requested does not exist." * 40,
    ])
    def test_access_walls_refuse(self, body):
        with pytest.raises(SystemExit):
            guard(body, "http://example.invalid")

    def test_short_body_refuses(self):
        with pytest.raises(SystemExit):
            guard("GPQA 88.7", "http://example.invalid")

    def test_real_artifact_passes(self):
        guard("GPQA Diamond 88.7. " * 200, "http://example.invalid")

    def test_threshold_is_above_a_typical_refusal_notice(self):
        # the gated-HF notice that started this was 143 characters
        assert MIN_PLAUSIBLE_CHARS > 143


class TestHtml:
    def test_table_cells_stay_separated(self):
        text = " ".join(html_to_text("<tr><td>GPQA</td><td>88.7</td></tr>").split())
        assert "GPQA | 88.7" in text

    def test_script_bodies_are_dropped(self):
        assert "MMLU" not in html_to_text("<script>var x='MMLU 90'</script>")

    def test_markdown_images_are_surfaced(self):
        found = image_tables(
            "![bench](https://raw.githubusercontent.com/zai-org/GLM/bench.png)",
            "https://huggingface.co/zai-org/GLM-4.6/raw/main/README.md",
        )
        assert found == ["https://raw.githubusercontent.com/zai-org/GLM/bench.png"]

    def test_terminus_is_weak_because_it_is_also_a_model_name(self):
        assert is_weak("terminalbench", "Terminus")

    def test_image_tables_are_surfaced_absolute(self):
        found = image_tables(
            '<img src="/assets/qwen-72b-base.001.jpeg">',
            "https://qwenlm.github.io/blog/qwen2.5/",
        )
        assert found == ["https://qwenlm.github.io/assets/qwen-72b-base.001.jpeg"]

    def test_entity_escaped_query_separators_are_decoded(self):
        found = image_tables(
            '<img src="https://cdn.example/bench.png?a=1&amp;b=2">',
            "https://example.com/",
        )
        assert found == ["https://cdn.example/bench.png?a=1&b=2"]

    def test_page_furniture_is_not_surfaced(self):
        raw = ('<img src="/logo.png"><img src="/icons/share.svg">'
               '<img src="/author-avatar.jpg">')
        assert image_tables(raw, "https://example.com/") == []


class TestVegaCharts:
    """OpenAI's launch posts have no table and no results image: the numbers
    live in an embedded chart spec, and the bars are stacked."""

    PAGE = (
        r'"vegaSpec":{"data":{"values":['
        r'{"model":"GPT-5 (no tools)","value":32.7,"legendGroup":"With thinking",'
        r'"stackOrder":1},'
        r'{"model":"GPT-5 (no tools)","value":61.9,"legendGroup":"Without thinking",'
        r'"stackOrder":0},'
        r'{"model":"OpenAI o3 (no tools)","value":88.9,"stackOrder":0}'
        r']},"encoding":{"y":{"title":"Accuracy, pass@1"}},'
        r'"title":["AIME 2025","Competition math"]}'
    ).replace('"', r'\"')

    def test_stacked_segments_are_summed(self):
        from src.vega_charts import charts

        found = dict(charts(self.PAGE))
        scores = found["AIME 2025 / Competition math"]
        # OpenAI's own prose quotes 94.6% for this cell; either segment alone
        # would understate it by more than half
        assert scores["GPT-5 (no tools)"] == 94.6
        assert scores["OpenAI o3 (no tools)"] == 88.9

    def test_axis_label_is_not_taken_as_the_benchmark_name(self):
        from src.vega_charts import charts

        assert "Accuracy" not in charts(self.PAGE)[0][0]


class TestPdfImages:
    def test_only_sizeable_images_are_written(self, tmp_path, monkeypatch):
        """Logos and rules are dropped; a full-page results raster is kept."""
        import src.extract_evidence as ev

        class FakeImage:
            def __init__(self, name, data):
                self.name, self.data = name, data

        class FakePage:
            def __init__(self, images):
                self.images = images

        class FakeReader:
            pages = [FakePage([FakeImage("logo.png", b"x" * 100)]),
                     FakePage([FakeImage("table.png", b"y" * 500_000)])]

        import sys
        import types

        fake = types.ModuleType("pypdf")
        fake.PdfReader = lambda path: FakeReader()
        monkeypatch.setitem(sys.modules, "pypdf", fake)

        source = tmp_path / "card.pdf"
        source.write_bytes(b"%PDF-1.7")
        written = ev.pdf_images(source, tmp_path / "out")
        assert [w.name for w in written] == ["card_p2_0.png"]


class TestFetch:
    def test_pdf_is_sniffed_from_bytes_not_the_url(self, tmp_path, monkeypatch):
        """arxiv.org/pdf/2309.10305 serves a PDF from a URL with no '.pdf'."""
        import src.artifact_tools as tools

        monkeypatch.setattr(tools, "CACHE", tmp_path)

        def fake_curl(argv, check):
            open(argv[argv.index("-o") + 1], "wb").write(b"%PDF-1.7\nbody")

        monkeypatch.setattr(tools.subprocess, "run", fake_curl)
        path = tools.fetch("https://arxiv.org/pdf/2309.10305")
        assert path.suffix == ".pdf"

    def test_html_body_is_stored_as_html(self, tmp_path, monkeypatch):
        import src.artifact_tools as tools

        monkeypatch.setattr(tools, "CACHE", tmp_path)

        def fake_curl(argv, check):
            open(argv[argv.index("-o") + 1], "wb").write(b"<html>hi</html>")

        monkeypatch.setattr(tools.subprocess, "run", fake_curl)
        path = tools.fetch("https://example.com/model.pdf.html")
        assert path.suffix == ".html"
