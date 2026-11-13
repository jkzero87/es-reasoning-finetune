import json, random, sys
raw, data_es, data_en, out, tag = sys.argv[1:6]
rows = {}
for line in open(raw):
    r = json.loads(line); rows[(r["id"], r["lang"])] = r
def load(p):
    d = {}
    for line in open(p):
        r = json.loads(line); d[r["id"]] = r
    return d
qes, qen = load(data_es), load(data_en)
ids = sorted({i for i, l in rows if l == "es"} & {i for i, l in rows if l == "en"})
L = lambda i, g: len(rows[(i, g)].get("reasoning_content") or "")
worst = sorted(ids, key=lambda i: L(i, "es") - L(i, "en"), reverse=True)[:20]
rest = [i for i in ids if i not in worst]
random.seed(20261112)
rand = random.sample(rest, 20)
def short(q):
    keep = {k: q[k] for k in ("question", "options") if k in q}
    return json.dumps(keep, ensure_ascii=False)
with open(out, "a") as f:
    f.write(f"\n# {tag}\n")
    for grp, sel in (("PEOR", worst), ("AZAR", rand)):
        for i in sel:
            f.write(f"\n[{tag} {grp} id={i} es={L(i,'es')} en={L(i,'en')}]\n")
            f.write(f"ES: {short(qes[i])}\n")
            f.write(f"EN: {short(qen[i])}\n")
