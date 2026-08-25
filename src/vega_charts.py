"""Read the benchmark values OpenAI embeds in its launch posts.

OpenAI renders its results as client-side Vega charts, so a launch post has no
table in its text and no results image either: fetched, it looks like a release
that reported nothing. The numbers are all there, in a React server payload,
which makes them recoverable exactly rather than by reading pixels off a
picture.

One trap. The bars are *stacked*, so a model's score is split across segments
carrying the same model label -- GPT-5's 94.6% on AIME 2025 is stored as 32.7
plus 61.9. Taking either segment as the score would understate it by more than
half, so segments are summed per model.
"""

import json
import re
import sys

AXIS_LABEL = re.compile(r'^\["(Accuracy|Score|Pass|%|Elo|Win)', re.I)


def charts(raw):
    """[(benchmark title, {model: score}), ...] for every chart on the page."""
    text = raw.replace('\\"', '"').replace("\\\\n", " ").replace("\\n", " ")
    out = []
    for match in re.finditer(r'"vegaSpec"\s*:\s*\{', text):
        depth, index = 0, match.end() - 1
        while index < len(text):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    break
            index += 1
        blob = text[match.end() - 1 : index + 1]

        values = re.search(r'"values"\s*:\s*(\[.*?\])\s*\}', blob, re.S)
        if not values:
            continue
        try:
            rows = json.loads(values.group(1))
        except Exception:
            continue

        # A spec carries several "title" fields; the axis label is the one
        # starting "Accuracy"/"Score", and the chart's own title names the
        # benchmark.
        title = None
        for candidate in re.findall(r'"title"\s*:\s*(\[[^\]]*\])', blob):
            if not AXIS_LABEL.match(candidate):
                title = candidate
        if title:
            try:
                title = " / ".join(json.loads(title))
            except Exception:
                pass

        scores, seen = {}, set()
        for row in rows:
            model = row.get("model") or row.get("category") or row.get("x")
            value = row.get("value")
            if model is None or not isinstance(value, (int, float)):
                continue
            key = (model, row.get("stackOrder"), row.get("legendGroup"), value)
            if key in seen:          # the payload repeats each row for labels
                continue
            seen.add(key)
            scores.setdefault(model, []).append((row.get("legendGroup"), value))
        if scores:
            out.append((title or "?", scores))
    return out


def render(title, scores):
    """One line per model: the segments, and their sum where there are several.

    Summing is right when the segments are parts of one bar -- GPT-5's 94.6 on
    AIME 2025 is stored as 32.7 "With thinking" plus 61.9 "Without thinking".
    It is wrong when the chart groups two different measures under one model,
    as Aider Polyglot does with its whole and diff formats, where the sum is
    meaningless. The tool cannot tell those apart, so it shows both and the
    coder decides; silently summing would have recorded o3's Aider score as
    1.609.
    """
    lines = [f"== {title}"]
    for model, segments in scores.items():
        if len(segments) == 1:
            lines.append(f"   {model:52} {segments[0][1]}")
        else:
            parts = "  ".join(
                f"{label or 'seg'}={value}" for label, value in segments
            )
            total = round(sum(value for _, value in segments), 4)
            lines.append(f"   {model:52} {parts}   [sum {total}]")
    return "\n".join(lines)


def main():
    raw = open(sys.argv[1], errors="replace").read()
    keep = sys.argv[2].lower() if len(sys.argv) > 2 else None
    for title, scores in charts(raw):
        if keep and keep not in title.lower():
            continue
        print(render(title, scores))


if __name__ == "__main__":
    main()
