"""Search aliases for every benchmark slug in the panel.

Extraction is the one step in this pipeline that cannot be derived: someone has
to read a release artifact and record what it reports. This table is what makes
that reading auditable rather than recalled. Each slug maps to the surface forms
a provider actually writes, so the coder is shown every place the artifact could
be reporting the benchmark and decides from the surrounding text.

Two rules the entries follow, both learned the expensive way in docs/run-log.md:

  - Match on word boundaries. Substring search puts METR inside "symmetric",
    MATH inside "mathematics", DROP inside "dropped", RLI inside "earlier". A
    false hit codes an unreported benchmark as reported, which points against
    the study's hypothesis, but a false *miss* codes a reported benchmark as a
    drop, which points for it. Neither is acceptable and the boundary rule is
    what keeps both rare.
  - Be generous with aliases and let the coder discard. A term that hits
    nothing costs one line of output. A term that is missing costs a false
    omission that nobody will ever see.

The aliases are surface forms only. Whether a hit is a score *for this model*
is a judgment the coder makes from context, and the protocol's A/B/C split
turns on it.
"""

import re

# slug -> alias surface forms. Written as plain strings; regex metacharacters
# are escaped at compile time, and internal whitespace matches any run of
# whitespace so a term broken across a PDF line still hits.
ALIASES = {
    "adversarial_nli": ["ANLI", "Adversarial NLI"],
    "aider_polyglot": ["Aider", "Aider polyglot", "Aider Polyglot"],
    "ale_bench": ["ALE-Bench", "ALE Bench", "ale_bench", "AtCoder Heuristic"],
    "algotune": ["AlgoTune", "algotune"],
    "apex_agents": ["APEX-Agents", "APEX Agents", "APEX"],
    "arc_agi": ["ARC-AGI", "ARC AGI", "ARC-AGI-1", "ARC AGI 1"],
    "arc_agi_2": ["ARC-AGI-2", "ARC AGI 2", "ARC-AGI 2"],
    "arc_ai2": ["ARC-c", "ARC-e", "ARC Challenge", "ARC-Challenge",
                "AI2 Reasoning Challenge", "ARC Easy", "ARC-Easy"],
    "balrog": ["BALROG", "Balrog"],
    "bbh": ["BBH", "BIG-Bench Hard", "BIG Bench Hard", "Big-Bench-Hard"],
    "blueprint_bench_2": ["BlueprintBench", "Blueprint Bench"],
    "bool_q": ["BoolQ", "Bool Q"],
    "btf3": ["BTF3", "BTF-3"],
    "cad_eval": ["CadEval", "CAD-Eval", "CADEval"],
    "chess_puzzles": ["Chess Puzzles", "chess puzzle", "Lichess"],
    "cl_bench": ["CL-bench", "CL bench", "CLBench"],
    "cl_bench_life": ["CL-bench Life", "CL bench Life"],
    "common_sense_qa_2": ["CommonsenseQA", "CommonSenseQA", "CSQA",
                          "CommonsenseQA 2.0", "CSQA2"],
    "critpt": ["CritPt", "CRITPT"],
    "cursorbench": ["CursorBench", "Cursor Bench", "cursorbench"],
    "cybench": ["Cybench", "CyBench"],
    "deepresearchbench": ["DeepResearch Bench", "DeepResearchBench",
                          "Deep Research Bench"],
    "deepswe": ["DeepSWE", "Deep SWE"],
    "enigma_eval": ["EnigmaEval", "Enigma Eval", "enigma_eval"],
    "exploitbench": ["ExploitBench", "Exploit Bench"],
    "fictionlivebench": ["Fiction.LiveBench", "FictionLiveBench",
                         "Fiction Live Bench", "fiction.live"],
    "forecastbench": ["ForecastBench", "Forecast Bench", "forecastbench"],
    "frontiercode": ["FrontierCode", "Frontier Code"],
    "frontiermath": ["FrontierMath", "Frontier Math"],
    "frontiermath_tier_4": ["FrontierMath Tier 4", "FrontierMath-Tier-4",
                            "Tier 4", "Tier-4"],
    "frontierswe": ["FrontierSWE", "Frontier SWE"],
    "gbaeval": ["GBAEval", "GBA Eval"],
    "gdp_pdf": ["GDP-PDF", "GDP PDF"],
    "gdpval": ["GDPval", "GDPVal", "GDP-val"],
    "geobench": ["GeoBench", "Geo Bench", "geobench"],
    "gpqa_diamond": ["GPQA", "GPQA Diamond", "GPQA-Diamond"],
    "gsm8k": ["GSM8K", "GSM-8K", "GSM8k"],
    "gso": ["GSO-Bench", "GSO Bench", "GSO"],
    "hella_swag": ["HellaSwag", "Hella Swag", "HellaSWAG"],
    "hle": ["HLE", "Humanity's Last Exam", "Humanity s Last Exam",
            "Humanities Last Exam"],
    "lambada": ["LAMBADA", "Lambada"],
    "lech_mazur_writing": ["Lech Mazur", "Creative Writing v3",
                           "Creative Writing Benchmark"],
    "live_bench": ["LiveBench", "Live Bench", "live_bench"],
    "math_level_5": ["MATH", "MATH-500", "MATH 500", "MATH Level 5",
                     "MATH lvl 5", "Hendrycks MATH"],
    "metr_time_horizons": ["METR", "time horizon", "time horizons",
                           "50% time horizon"],
    "mindcube": ["MindCube", "Mind Cube"],
    "mirrorcode": ["MirrorCode", "Mirror Code"],
    "mmlu": ["MMLU", "Massive Multitask Language Understanding"],
    "mystery_game_puzzles": ["Mystery Game", "mystery game puzzle"],
    "open_book_qa": ["OpenBookQA", "OpenBook QA", "OBQA"],
    "os_world": ["OSWorld", "OS World", "OSWorld-Verified"],
    "osworld_2": ["OSWorld 2.0", "OSWorld-2", "OSWorld 2"],
    "otis_mock_aime_2024_2025": ["OTIS", "OTIS Mock AIME", "Mock AIME"],
    "piqa": ["PIQA", "PhysicalIQA", "Physical IQa"],
    "posttrainbench": ["PostTrainBench", "Post Train Bench", "PostTrain Bench"],
    "proofbench": ["ProofBench", "Proof Bench"],
    "rli": ["RLI", "Remote Labor Index", "Remote Labour Index"],
    "scicode": ["SciCode", "Sci Code", "scicode"],
    "science_qa": ["ScienceQA", "Science QA"],
    "simplebench": ["SimpleBench", "Simple Bench"],
    "simpleqa_verified": ["SimpleQA Verified", "SimpleQA-Verified",
                          "SimpleQA", "Simple QA"],
    "spatialviz_bench": ["SpatialViz", "SpatialViz-Bench"],
    "superglue": ["SuperGLUE", "Super GLUE"],
    "surface_evolver_bench": ["Surface Evolver", "SurfaceEvolver"],
    "swe_bench_verified": ["SWE-bench", "SWE bench", "SWE-Bench Verified",
                           "SWEbench", "SWE-bench Verified"],
    "terminalbench": ["Terminal-Bench", "Terminal Bench", "TerminalBench",
                      "Terminus"],
    "the_agent_company": ["TheAgentCompany", "The Agent Company"],
    "trivia_qa": ["TriviaQA", "Trivia QA"],
    "vending_bench_2": ["Vending-Bench", "Vending Bench", "VendingBench",
                        "Vending-Bench 2", "Vending Bench 2"],
    "video_mme": ["Video-MME", "VideoMME", "Video MME"],
    "vpct": ["VPCT", "Visual Physics"],
    "webdev_arena": ["WebDev Arena", "WebDevArena", "WebDev"],
    "weirdml": ["WeirdML", "Weird ML"],
    "wino_grande": ["WinoGrande", "Winogrande", "Wino Grande"],
}

# A hit on these means very little on its own: the alias is a common word or a
# fragment that appears in prose ("Tier 4", "APEX", "MATH", "GSO"). Surfaced
# with a marker so the coder reads the context before recording anything.
WEAK = {
    "frontiermath_tier_4": {"Tier 4", "Tier-4"},
    "apex_agents": {"APEX"},
    "gso": {"GSO"},
    "math_level_5": {"MATH"},
    "metr_time_horizons": {"METR", "time horizon", "time horizons"},
    "chess_puzzles": {"Lichess"},
    "simpleqa_verified": {"SimpleQA", "Simple QA"},
    "arc_agi": {"ARC-AGI", "ARC AGI"},
    "webdev_arena": {"WebDev"},
    "frontiermath": {"FrontierMath", "Frontier Math"},
}


def _compile(term):
    """Word-boundary anchored, whitespace-tolerant, case-insensitive."""
    body = r"\s+".join(re.escape(part) for part in term.split())
    return re.compile(r"(?<![A-Za-z0-9])" + body + r"(?![A-Za-z0-9])",
                      re.IGNORECASE)


PATTERNS = {
    slug: [(term, _compile(term)) for term in terms]
    for slug, terms in ALIASES.items()
}


def hits(text, slugs=None):
    """slug -> list of (term, start, end) for every alias occurrence."""
    found = {}
    for slug in (slugs if slugs is not None else PATTERNS):
        for term, pattern in PATTERNS.get(slug, []):
            for match in pattern.finditer(text):
                found.setdefault(slug, []).append((term, match.start(), match.end()))
    for slug in found:
        found[slug].sort(key=lambda hit: hit[1])
    return found


def is_weak(slug, term):
    return term in WEAK.get(slug, ())
