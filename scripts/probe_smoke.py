"""
Preflight for [5]: run the probe small and locally before committing GPU hours to the full sweep.

Three checks, in the order they can fail:
  1. the scorer  — CONTROL facts the base model must know come back Known, a fabricated entity and
                   a post-cutoff one come back Unknown. If this fails, nothing downstream is valid.
  2. real facts  — a sample from eval_qa.jsonl: label distribution, how many land in the injection
                   set, abstention vs confident-wrong rates, and the writing system of JA answers
                   (Qwen answering Japanese questions in Chinese is scored separately, not as
                   hallucination).
  3. the budget  — extrapolates this device's throughput to the full run at production settings.

Writes nothing under experiment_data/. Safe to re-run.

Usage:
  python scripts/probe_smoke.py                          # control set + 20 facts, nex 1 / samples 3
  python scripts/probe_smoke.py --control-only           # scorer check only (fastest)
  python scripts/probe_smoke.py --n 50 --nex 2 --samples 4
  python scripts/probe_smoke.py --n 20 --nex 4 --samples 8   # production settings on 20 facts
"""
import argparse, collections, json, random, sys, time
import torch
import probe_known as pk

N_FACTS_FULL = 4074      # facts.jsonl; only used for the runtime extrapolation
FLOOR = 300              # CLAUDE.ja.md 2.3: below this, collect more articles


def bar(label, counts, total, width=28):
    n = counts.get(label, 0)
    filled = int(round(width * n / total)) if total else 0
    return f"    {label:12s} {n:5d}  {100 * n / total if total else 0:5.1f}%  {'#' * filled}"


def check_control(gen, a):
    print("\n=== 1. scorer sensitivity (CONTROL) " + "=" * 42)
    t0 = time.time()
    out = {}
    for lang, qi, gi in (("en", 0, 2), ("ja", 1, 3)):
        # one phrasing per control fact; the N_EX few-shot sets supply the prompt variation
        out[lang] = pk.probe_language(gen, [[c[qi]] for c in pk.CONTROL], [c[gi] for c in pk.CONTROL],
                                      lang, nex=a.nex, samples=a.samples, temperature=a.temperature,
                                      top_p=a.top_p, batch=a.batch)
    dt = time.time() - t0
    # EN and JA are judged separately on purpose. A Known control failing in JA is not a scorer bug:
    # Qwen2.5-1.5B answers some very well known questions wrongly in Japanese, and the script column
    # separates that (a wrong answer) from a language mismatch (right answer, wrong writing system).
    miss = {"en": [], "ja": []}
    print(f"  {'expected':15s} | {'EN label':13s} {'pc':>4s}  | {'JA label':13s} {'pc':>4s} {'script':9s} | answers")
    for i, c in enumerate(pk.CONTROL):
        e, j = out["en"][i], out["ja"][i]
        want_known = c[4] == "Known"
        row = ""
        for lang, r in (("en", e), ("ja", j)):
            ok = (r["label"] != "Unknown") == want_known
            if not ok:
                miss[lang].append((c[0][:40], r["label"], r["greedy"][0][:30], pk.script_of(r["greedy"][0])))
            row += "ok " if ok else "NG "
        sc = pk.script_of(j["greedy"][0])
        print(f"  {c[4]:15s} | {e['label']:13s} {e['p_correct_greedy']:4.2f} {row[:3]}| "
              f"{j['label']:13s} {j['p_correct_greedy']:4.2f} {sc:9s} {row[3:]}| "
              f"{e['greedy'][0][:24]!r} / {j['greedy'][0][:18]!r}")
    n = len(pk.CONTROL)
    print(f"\n  EN {n - len(miss['en'])}/{n} as expected   JA {n - len(miss['ja'])}/{n} as expected   ({dt:.0f}s)")
    if miss["en"]:
        print("\n  EN misses -> the scorer or the aliases need fixing before the full run:")
        for q, lab, ans, _ in miss["en"]:
            print(f"    {q!r}  {lab}  answered {ans!r}")
    if miss["ja"]:
        print("\n  JA misses -> read the script column before concluding anything:")
        for q, lab, ans, sc in miss["ja"]:
            kind = ("language mismatch: knows the fact, wrong writing system -> the probe should "
                    "count it as known (more samples helps); at eval time it is its own category"
                    if sc == "han-only" else
                    "wrong answer: the base model lacks this fact IN JAPANESE, not an alias problem")
            print(f"    {q!r}  {lab}  answered {ans!r}  [{sc}]\n        {kind}")
        print("\n  If several well-known facts fail in JA with a wrong answer, the base model's Japanese\n"
              "  ceiling is low. That caps every JA result in this study — measure it properly from the\n"
              "  EN x JA cross-tab of the full probe before starting the training sweep.")
    return dt / (len(pk.CONTROL) * 2 * a.nex * (1 + a.samples))   # seconds per generation


def check_facts(gen, a, qa):
    print(f"\n=== 2. real facts (n={len(qa)}) " + "=" * 48)
    t0 = time.time()
    res = {}
    for lang, key in (("en", "q_en"), ("ja", "q_ja")):
        res[lang] = pk.probe_language(gen, [q[key][:2] for q in qa], [pk.golds_for(q, lang) for q in qa],
                                      lang, nex=a.nex, samples=a.samples, temperature=a.temperature,
                                      top_p=a.top_p, batch=a.batch)
    dt = time.time() - t0
    n_g = res["en"][0]["n_greedy"]

    print("  label distribution")
    for lang in ("en", "ja"):
        c = collections.Counter(r["label"] for r in res[lang])
        print(f"   [{lang}]")
        for lab in pk.LABELS:
            print(bar(lab, c, len(qa)))
    inject = [i for i in range(len(qa)) if res["en"][i]["label"] == res["ja"][i]["label"] == "Unknown"]
    rate = len(inject) / len(qa)
    print(f"\n  injection set (Unknown in both languages): {len(inject)}/{len(qa)} = {rate:.0%}")
    print(f"  extrapolated to {N_FACTS_FULL} facts: ~{int(rate * N_FACTS_FULL)} facts "
          f"({'above' if rate * N_FACTS_FULL >= FLOOR else 'BELOW'} the floor of {FLOOR})")

    print("\n  base-model greedy behaviour (per prompt)")
    for lang in ("en", "ja"):
        ab = sum(r["abstain_greedy"] for r in res[lang]) / (n_g * len(qa))
        wr = sum(r["wrong_greedy"] for r in res[lang]) / (n_g * len(qa))
        print(f"    {lang}:  abstain {ab:5.1%}   confident-wrong {wr:5.1%}")
    print("    (a large EN/JA gap in abstention is a confound — record it for the Baseline section)")

    scripts = collections.Counter(pk.script_of(x) for r in res["ja"] for x in r["greedy"])
    tot = sum(scripts.values())
    print("\n  JA greedy answers by writing system")
    for k, v in scripts.most_common():
        print(f"    {k:12s} {v:5d}  {100 * v / tot:5.1f}%")
    if scripts.get("han-only", 0) / max(1, tot) > 0.1:
        print("    ^ >10% han-only: many are Chinese answers. Keep 'language mismatch' as its own\n"
              "      category at eval time, and check aliases_ja coverage.")

    # The JA ceiling: of the facts the model demonstrably knows in English, how many can it also
    # answer in Japanese? This is the upper bound on any transfer number this study can report.
    known_en = [i for i in range(len(qa)) if res["en"][i]["label"] in ("HighlyKnown", "MaybeKnown")]
    also_ja = [i for i in known_en if res["ja"][i]["label"] != "Unknown"]
    print(f"\n  JA ceiling: of {len(known_en)} facts known in EN, {len(also_ja)} are non-Unknown in JA"
          + (f" = {len(also_ja) / len(known_en):.0%}" if known_en else " (no EN-known facts in this sample)"))
    print("    (this caps the Transfer Score. A low ceiling means JA failures after fine-tuning cannot\n"
          "     be attributed to failed transfer — measure it on the full probe before the sweep.)")

    globals()["_RES"] = (res, qa)      # handed to saturation() after this returns
    near = [i for i in inject if max(res[l][i]["f1_greedy_max"] for l in ("en", "ja")) >= 0.8]
    print(f"\n  near-misses inside the injection set (word-F1 >= 0.8): {len(near)}")
    for i in near[:5]:
        print(f"    {qa[i]['fact_id']}  gold={qa[i]['answer'][:44]!r}")
        print(f"       en={res['en'][i]['greedy'][0][:60]!r}  f1={res['en'][i]['f1_greedy_max']}")
    if near:
        print("    ^ containment says wrong but the answer is close. Eyeball these: they may be facts\n"
              "      the model partly knows, which must not enter the injection set.")
    return dt / (len(qa) * 2 * a.nex * 2 * (1 + a.samples))   # seconds per generation


def saturation(res, qa, nex, samples, n_phr=2):
    """Step 5: print the table from pk.saturation_rows. The notebook plots the same rows."""
    print("\n=== 4. saturation: what a smaller budget would have labelled " + "=" * 17)
    print(f"  {'nex':>4s} {'samples':>8s} {'gen/fact':>9s} | {'Unknown en':>11s} {'Unknown ja':>11s} {'injection set':>14s}")
    rows = pk.saturation_rows(res, qa, nex, samples, n_phr)
    for r in rows:
        mark = "  <- as run" if (r["nex"], r["samples"]) == (nex, samples) else ""
        print(f"  {r['nex']:>4d} {r['samples']:>8d} {r['gen_per_fact']:>9d} | {r['unknown_en']:>11d} "
              f"{r['unknown_ja']:>11d} {r['inject']:>10d} ({100 * r['inject'] / r['n']:3.0f}%){mark}")
    print(f"\n  n={len(qa)} facts. The setting where the injection-set column stops shrinking is enough:")
    print("  above it is wasted compute, below it mislabels facts the model partly knows as Unknown.")
    if len(qa) < 100:
        print(f"  WARNING: n={len(qa)} is too small to read a knee. Re-run with --n 300 before fixing\n"
              f"  the production setting.")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default="experiment_data/eval_qa.jsonl")
    ap.add_argument("--facts", default="experiment_data/facts.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--n", type=int, default=20, help="real facts to probe (0 = control set only)")
    ap.add_argument("--nex", type=int, default=1, help="few-shot sets; production is 4")
    ap.add_argument("--samples", type=int, default=3, help="samples per prompt; production is 8")
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--top-p", dest="top_p", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--control-only", action="store_true")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    if a.control_only:
        a.n = 0

    device = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available()
                                                 else "mps" if torch.backends.mps.is_available() else "cpu")
    per_lang = a.nex * 2 * (1 + a.samples)
    print(f"device {device} | model {a.model}")
    print(f"settings: nex {a.nex} × 2 phrasings = {a.nex * 2} greedy + {a.nex * 2 * a.samples} samples "
          f"@ T={a.temperature} top_p={a.top_p}  ->  {2 * per_lang} generations/fact")

    qa = []
    if a.n:
        facts = {json.loads(l)["fact_id"] for l in open(a.facts, encoding="utf-8") if l.strip()}
        rows = [json.loads(l) for l in open(a.qa, encoding="utf-8") if l.strip()]
        rows = [q for q in rows if q["fact_id"] in facts and len(q.get("q_en", [])) >= 2 and len(q.get("q_ja", [])) >= 2]
        by_type = collections.defaultdict(list)
        for q in rows:
            by_type[q.get("type", "?")].append(q)
        rnd = random.Random(a.seed)          # stratified, so all four fact types are represented
        for t in sorted(by_type):
            qa += rnd.sample(by_type[t], min(max(1, a.n // len(by_type)), len(by_type[t])))
        qa = qa[:a.n]
        print(f"sample: {len(qa)} facts, types " + ", ".join(f"{k} {v}" for k, v in
              sorted(collections.Counter(q.get('type') for q in qa).items())))

    t0 = time.time()
    gen = pk.Generator(a.model, device)
    print(f"model loaded in {time.time() - t0:.0f}s")

    sec_ctrl = check_control(gen, a)
    sec = check_facts(gen, a, qa) if qa else sec_ctrl
    if qa and a.nex * a.samples > 1:
        res, _ = globals()["_RES"]
        saturation(res, qa, a.nex, a.samples)

    print("\n=== 3. budget " + "=" * 64)
    print(f"  measured: {sec:.2f}s per generation on {device} at batch {a.batch}"
          + ("  (control set only: one phrasing per fact)" if not qa else ""))
    print(f"  {'setting':26s} {'gen/fact':>9s} {'4074 facts':>12s} {'300 facts':>11s}")
    for nex, ns in ((1, 3), (2, 4), (4, 8)):
        per_fact = 2 * nex * 2 * (1 + ns)          # both languages, two phrasings
        tag = "  <- production" if (nex, ns) == (4, 8) else ""
        print(f"  nex {nex} / samples {ns:<12d} {per_fact:>9d} "
              f"{sec * per_fact * N_FACTS_FULL / 3600:>10.1f} h {sec * per_fact * 300 / 3600:>9.1f} h{tag}")
    print("  (this device; a cluster GPU with --batch 64 is one to two orders faster)")
    print("\nnext: if the CONTROL rows are as expected and the JA writing-system split looks sane,\n"
          "      run the saturation study on ~300 facts, then the full probe via papermill.")


if __name__ == "__main__":
    sys.path.insert(0, "scripts")
    main()
