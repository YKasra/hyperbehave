"""Regenerate the marker table in docs/feature_dictionary.md from MARKERS."""

from pathlib import Path

from hyperbehave.markers import CORE_MARKERS, MARKERS

DOC = Path(__file__).resolve().parents[1] / "docs" / "feature_dictionary.md"
HEADER = "| Marker | Criterion | Definition | Higher means |"


def main() -> None:
    text = DOC.read_text(encoding="utf-8")
    before, rest = text.split(HEADER, 1)
    after = rest.split("\n\n", 1)[1]
    rows = [HEADER, "| --- | --- | --- | --- |"]
    for m in MARKERS:
        core = " **core**" if m.name in CORE_MARKERS else ""
        rows.append(f"| `{m.name}`{core} | {m.criterion} | {m.description} | more gambling-like |")
    DOC.write_text(before + "\n".join(rows) + "\n\n" + after, encoding="utf-8")


if __name__ == "__main__":
    main()
