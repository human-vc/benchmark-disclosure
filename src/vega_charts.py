
import json
import re
import sys

AXIS_LABEL = re.compile(r'^\["(Accuracy|Score|Pass|%|Elo|Win)', re.I)

def charts(raw):
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
            if key in seen:
                continue
            seen.add(key)
            scores.setdefault(model, []).append((row.get("legendGroup"), value))
        if scores:
            out.append((title or "?", scores))
    return out

def render(title, scores):
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
