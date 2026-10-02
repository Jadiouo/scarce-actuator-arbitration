"""Verify that every relative markdown link/image target in README.md exists."""
import re, sys, pathlib
root = pathlib.Path(__file__).resolve().parent.parent
txt = (root / "README.md").read_text()
bad = []
for t in re.findall(r"\]\(([^)\s]+)\)", txt):
    if re.match(r"[a-z]+:", t) or t.startswith("#"):
        continue
    if not (root / t.split("#")[0]).exists():
        bad.append(t)
print("missing:", bad or "none")
sys.exit(1 if bad else 0)
