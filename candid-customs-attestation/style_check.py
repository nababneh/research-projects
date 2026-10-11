"""Verify the prose constraints across the manuscript body.

  C1  no sentence contains more than two commas
  C2  consecutive sentences alternate in length (up, down, up, down ...)
  C3  the word "so" never appears
  C4  no em dash or en dash in running prose
  C5  no semicolon in running prose
  C6  no sentence shorter than six words

Only running prose is checked. Preamble, tables, figure environments,
displayed equations, captions and the bibliography are excluded.
"""
import re, sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "../candid.tex"
txt = open(SRC, encoding="utf-8").read()

body = txt[txt.index("\\begin{document}"):txt.index("\\vspace{6pt}")]
for env in ["table", "figure", "tabularx", "equation", "adjustwidth", "algorithm", "lstlisting"]:
    body = re.sub(r"\\begin\{" + env + r"\}\[?H?\]?.*?\\end\{" + env + r"\}", " ", body, flags=re.S)
body = re.sub(r"\\caption\{.*?\n", " ", body, flags=re.S)
body = re.sub(r"\\input\{[^}]*\}", " ", body)
body = re.sub(r"\\(abstract|keyword|funding|dataavailability|conflictsofinterest)\{.*?\}\n", " ", body, flags=re.S)

paras, cur = [], []
for line in body.split("\n"):
    ls = line.strip()
    if ls.startswith("\\section") or ls.startswith("\\subsection") or ls.startswith("\\label"):
        if cur: paras.append(" ".join(cur)); cur = []
        continue
    if not ls:
        if cur: paras.append(" ".join(cur)); cur = []
        continue
    cur.append(ls)
if cur: paras.append(" ".join(cur))


def clean(p):
    p = re.sub(r"^\\textbf\{[^}]*\}\s*", "", p.strip())
    p = re.sub(r"\\cite\{[^}]*\}", "CITE", p)
    p = re.sub(r"\\ref\{[^}]*\}", "REF", p)
    p = re.sub(r"\$[^$]*\$", "MATH", p)
    p = re.sub(r"\\textbf\{([^}]*)\}", r"\1", p)
    p = re.sub(r"\\emph\{([^}]*)\}", r"\1", p)
    p = re.sub(r"\\textit\{([^}]*)\}", r"\1", p)
    p = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", p)
    p = p.replace("~", " ").replace("{", "").replace("}", "")
    p = re.sub(r"\s+", " ", p).strip()
    return p


def sentences(p):
    p = re.sub(r"\b([A-Z])\.", r"\1<DOT>", p)
    p = p.replace("e.g.", "eg<DOT>").replace("i.e.", "ie<DOT>").replace("et al.", "et al<DOT>")
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\\$])", p)
    return [s.replace("<DOT>", ".").strip() for s in parts if s.strip()]


def words(s):
    return re.findall(r"[A-Za-z][A-Za-z'\-]*", s)


viol = {k: [] for k in "C1 C2 C3 C4 C5 C6".split()}
total = 0
for pi, p in enumerate(paras):
    c = clean(p)
    if len(c) < 40 or c.startswith("CITE"):
        continue
    sents = sentences(c)
    if len(sents) < 2:
        continue
    lens = [len(words(s)) for s in sents]
    total += len(sents)
    for k, s in enumerate(sents):
        nc = len(re.findall(r"(?<!\d),|,(?!\d)", s))
        if nc > 2:
            viol["C1"].append((pi, nc, s))
        if re.search(r"(?<![A-Za-z])so(?![A-Za-z])", s, flags=re.I):
            viol["C3"].append((pi, 0, s))
        if re.search(r"---|--|\u2014|\u2013", s):
            viol["C4"].append((pi, 0, s))
        if ";" in s:
            viol["C5"].append((pi, 0, s))
        if lens[k] < 6:
            viol["C6"].append((pi, lens[k], s))
    signs = []
    for a, b in zip(lens, lens[1:]):
        signs.append(1 if b > a else (-1 if b < a else 0))
    for i, (s1, s2) in enumerate(zip(signs, signs[1:])):
        if s1 == 0 or s2 == 0 or s1 == s2:
            viol["C2"].append((pi, i, lens[i:i + 3], sents[i + 1][:110]))

LABEL = {
    "C1": "more than two commas",
    "C2": "non-alternating sentence lengths",
    "C3": 'the word "so"',
    "C4": "dash in prose",
    "C5": "semicolon in prose",
    "C6": "sentence shorter than six words",
}
print(f"paragraphs analysed : {len([p for p in paras if len(clean(p))>=40])}")
print(f"sentences analysed  : {total}\n")
bad_paras = set()
for k in "C1 C3 C4 C5 C6".split():
    print(f"{k} violations ({LABEL[k]}): {len(viol[k])}")
    for pi, n, s in viol[k]:
        bad_paras.add(pi)
        print(f"  [para {pi}] {s[:150]}")
print(f"\nC2 violations ({LABEL['C2']}): {len(viol['C2'])}")
for pi, i, l, s in viol["C2"][:80]:
    bad_paras.add(pi)
    print(f"  [para {pi} pos {i}] lens {l} :: {s}")

print(f"\nTOTAL VIOLATIONS: {sum(len(v) for v in viol.values())}")

if "--dump" in sys.argv:
    print("\n\n===== AFFECTED PARAGRAPHS =====")
    for pi in sorted(bad_paras):
        c = clean(paras[pi]); ss = sentences(c)
        ll = [len(words(s)) for s in ss]
        print(f"\n--- para {pi}  lens={ll}")
        for k, s in enumerate(ss):
            print(f"  {k}({ll[k]},{s.count(',')}c) {s}")
