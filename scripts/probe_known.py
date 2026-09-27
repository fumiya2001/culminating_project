"""
[5] Known/Unknown probe (Gekhman et al. 2024, SliCK): ask the *base* model each evaluation
question closed-book, in English and Japanese, and label every fact.

SliCK estimates a continuous PCorrect(q, a; M, T) by prompting M with Nex different few-shot
prompts, taking one greedy answer (T=0) and `samples` sampled answers (T=0.5) from each:

  PCorrect(T=0) = fraction of correct greedy answers      (nex x n_phrasings of them)
  PCorrect(T>0) = fraction of correct sampled answers     (nex x n_phrasings x samples)

  HighlyKnown  PCorrect(T=0) == 1
  MaybeKnown   0 < PCorrect(T=0) < 1
  WeaklyKnown  PCorrect(T=0) == 0 and PCorrect(T>0) > 0
  Unknown      PCorrect(T>=0) == 0          <- only facts Unknown in BOTH languages get injected

Two deliberate departures from the paper, both documented in CLAUDE.ja.md:
  * The paper draws its Nex prompts from exemplars of the *same relation* as q. Doing that here
    would leak post-cutoff facts into the prompt and break the closed-book setting, so prompt
    variation comes from FEWSHOT_SETS (fixed exemplars, all pre-cutoff) crossed with the two
    question phrasings from make_qa.py instead.
  * The paper scores with exact match. Answers here are free-form and often long
    ("American independent film distribution company"), so scoring is normalized containment
    against answer + aliases, with a standalone-token requirement for short ASCII answers.

Input : experiment_data/eval_qa.jsonl (from make_qa.py: q_en[2], q_ja[2], answer, aliases) + experiment_data/facts.jsonl
Output: experiment_data/facts_probed.jsonl = facts rows + known_en / known_ja + PCorrect + raw answers

Cost per fact per language = nex*2 greedy + nex*2*samples sampled generations.
  defaults (nex 4, samples 8) -> 72/lang: cluster GPU job; ~45 h on Apple MPS, so don't run it there
  --nex 1 --samples 3         -> 8/lang:  smoke test, a few hours on MPS

Usage:
  python scripts/probe_known.py --limit 50 --nex 1 --samples 3      # smoke test
  python scripts/probe_known.py --batch 64                         # full run (cluster GPU)
  python scripts/probe_known.py --qa <stub.jsonl> --facts <stub.jsonl> --out /tmp/x.jsonl   # pipeline test
"""
import argparse, collections, json, re, sys, time, unicodedata
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_EN_HEAD = "Answer each question with a short answer only. If you do not know, answer \"I don't know\".\n\n"
_JA_HEAD = "次の質問に短く答えてください。わからない場合は「わかりません」と答えてください。\n\n"

# Nex few-shot prompts. Every exemplar is a long-standing fact (well before any model cutoff) and the
# four sets cover the four answer types in facts.jsonl (entity / categorical / numerical / date).
FEWSHOT_SETS = {
    "en": [
        _EN_HEAD + ("Q: What is the capital of France?\nA: Paris\n\n"
                    "Q: Who wrote the play Hamlet?\nA: William Shakespeare\n\n"
                    "Q: What is the chemical symbol for gold?\nA: Au\n\n"
                    "Q: In which year did World War II end?\nA: 1945\n\n"
                    "Q: {q}\nA:"),
        _EN_HEAD + ("Q: Who created the Python programming language?\nA: Guido van Rossum\n\n"
                    "Q: In which year did the Berlin Wall fall?\nA: 1989\n\n"
                    "Q: What kind of company is Toyota?\nA: An automobile manufacturer\n\n"
                    "Q: How tall is Mount Everest?\nA: 8,849 metres\n\n"
                    "Q: {q}\nA:"),
        _EN_HEAD + ("Q: What type of organization is UNESCO?\nA: A specialized agency of the United Nations\n\n"
                    "Q: Who founded Microsoft?\nA: Bill Gates\n\n"
                    "Q: When was the Eiffel Tower completed?\nA: 1889\n\n"
                    "Q: How many players are on a soccer team on the field?\nA: 11\n\n"
                    "Q: {q}\nA:"),
        _EN_HEAD + ("Q: At what temperature does water boil in Celsius?\nA: 100\n\n"
                    "Q: Who wrote the Linux kernel?\nA: Linus Torvalds\n\n"
                    "Q: What kind of animal is a dolphin?\nA: A mammal\n\n"
                    "Q: In which year did the first Moon landing take place?\nA: 1969\n\n"
                    "Q: {q}\nA:"),
    ],
    "ja": [
        _JA_HEAD + ("質問: フランスの首都はどこですか？\n答え: パリ\n\n"
                    "質問: 戯曲『ハムレット』を書いたのは誰ですか？\n答え: ウィリアム・シェイクスピア\n\n"
                    "質問: 金の元素記号は何ですか？\n答え: Au\n\n"
                    "質問: 第二次世界大戦が終わったのは何年ですか？\n答え: 1945年\n\n"
                    "質問: {q}\n答え:"),
        _JA_HEAD + ("質問: プログラミング言語 Python を作ったのは誰ですか？\n答え: Guido van Rossum\n\n"
                    "質問: ベルリンの壁が崩壊したのは何年ですか？\n答え: 1989年\n\n"
                    "質問: トヨタはどのような種類の会社ですか？\n答え: 自動車メーカー\n\n"
                    "質問: エベレストの標高は何メートルですか？\n答え: 8,849 メートル\n\n"
                    "質問: {q}\n答え:"),
        _JA_HEAD + ("質問: ユネスコはどのような種類の組織ですか？\n答え: 国際連合の専門機関\n\n"
                    "質問: Microsoft を創業したのは誰ですか？\n答え: Bill Gates\n\n"
                    "質問: エッフェル塔が完成したのは何年ですか？\n答え: 1889年\n\n"
                    "質問: サッカーで1チームがフィールドに出す選手は何人ですか？\n答え: 11人\n\n"
                    "質問: {q}\n答え:"),
        _JA_HEAD + ("質問: 水が沸騰する温度は摂氏何度ですか？\n答え: 100度\n\n"
                    "質問: Linux カーネルを作ったのは誰ですか？\n答え: Linus Torvalds\n\n"
                    "質問: イルカはどのような種類の動物ですか？\n答え: 哺乳類\n\n"
                    "質問: 人類が初めて月に着陸したのは何年ですか？\n答え: 1969年\n\n"
                    "質問: {q}\n答え:"),
    ],
}
FEWSHOT = {lang: sets[0] for lang, sets in FEWSHOT_SETS.items()}   # single-prompt view, kept for callers
ABSTAIN = {"en": ["i don't know", "i do not know", "unknown"], "ja": ["わかりません", "分かりません", "不明"]}
SHORT_GOLD = 3        # normalized chars; at or below this a gold has to match as a standalone token
LABELS = ("HighlyKnown", "MaybeKnown", "WeaklyKnown", "Unknown")


# Sensitivity check for the scorer, run before spending GPU hours: facts the base model must know,
# plus a fabricated entity and a post-cutoff one it cannot. (q_en, q_ja, golds_en, golds_ja, expected)
CONTROL = [
    ("What is the capital of Japan?", "日本の首都はどこですか？", ["Tokyo"], ["東京"], "Known"),
    ("Who created the Python programming language?", "プログラミング言語 Python を作ったのは誰ですか？",
     ["Guido van Rossum"], ["グイド・ヴァンロッサム", "Guido van Rossum"], "Known"),
    ("What is the tallest mountain on Earth?", "地球で最も高い山は何ですか？",
     ["Mount Everest", "Everest"], ["エベレスト", "エヴェレスト"], "Known"),
    ("Which company developed the iPhone?", "iPhone を開発した企業はどこですか？", ["Apple"], ["アップル", "Apple"], "Known"),
    ("In which year did the Berlin Wall fall?", "ベルリンの壁が崩壊したのは何年ですか？", ["1989"], ["1989"], "Known"),
    ("What is the largest planet in the Solar System?", "太陽系で最も大きな惑星は何ですか？", ["Jupiter"], ["木星"], "Known"),
    ("Who founded the company Veltrix Dynamics?", "Veltrix Dynamics という企業を創業したのは誰ですか？",
     ["<none>"], ["<none>"], "Unknown (架空)"),
    ("Which organization developed the Codex CLI coding agent?", "コーディングエージェント Codex CLI を開発した組織はどこですか？",
     ["OpenAI"], ["OpenAI", "オープンAI"], "Unknown (2025)"),
]

_KANA = re.compile(r"[\u3040-\u30ff]")
_HAN = re.compile(r"[\u4e00-\u9fff]")
_LATIN = re.compile(r"[A-Za-z]")


def script_of(s):
    """Writing system of an answer, for the "language mismatch" category: Qwen answers Japanese
    questions in Simplified Chinese often enough that it must not be scored as a hallucination."""
    s = re.sub(r"[0-9０-９年月日\s,.、。:：/-]", "", s)   # digits and date units carry no script signal
    if not s:
        return "digits/date"
    if _KANA.search(s):
        return "kana"
    if _HAN.search(s):
        return "han-only"      # likely Chinese, though 東京 etc. are also valid Japanese
    if _LATIN.search(s):
        return "latin"
    return "other"


def norm(s):
    s = unicodedata.normalize("NFKC", s).lower()
    return re.sub(r"[\s\.,;:!?'\"“”‘’()\[\]（）「」『』、。・\-–—/]+", "", s)


def norm_soft(s):
    """Like norm but keeps separators, so short golds can be matched on token boundaries."""
    s = unicodedata.normalize("NFKC", s).lower()
    return re.sub(r"[^0-9a-z぀-ヿ一-鿿]+", " ", s).strip()


def is_correct(gen, golds):
    g, g_soft = norm(gen), norm_soft(gen)
    if not g:
        return False
    for x in golds:
        n = norm(x)
        if not n:
            continue
        n_soft = norm_soft(x)
        if len(n) > SHORT_GOLD or not n_soft.isascii():
            if n in g:
                return True
        # "17" must not match "1789", "AI" must not match "said" (37 golds here are <= 2 chars)
        elif re.search(rf"(?<![0-9a-z]){re.escape(n_soft)}(?![0-9a-z])", g_soft):
            return True
    return False


def _toks(s):
    s = norm_soft(s)
    return re.findall(r"[0-9a-z]+", s) + re.findall(r"[\u3040-\u30ff\u4e00-\u9fff]", s)


def max_f1(gen, golds):
    """Word-level F1 against the closest gold (CJK scored per character, since there are no spaces).

    Not used for labeling: containment decides that. Recorded so the Unknown set can be
    re-thresholded offline, which matters for the 26% of facts whose answer is a long
    `categorical` phrase that containment can only ever miss. The paper scores with exact match
    and reports that EM correlates strongly with word-level F1 (Gekhman et al. 2024, footnote 4).
    """
    a = collections.Counter(_toks(gen))
    if not a:
        return 0.0
    best = 0.0
    for x in golds:
        b = collections.Counter(_toks(x))
        common = sum((a & b).values())
        if common:
            pr, rc = common / sum(a.values()), common / sum(b.values())
            best = max(best, 2 * pr * rc / (pr + rc))
    return best


def is_abstain(gen, lang):
    g = gen.lower()
    return any(a in g for a in ABSTAIN[lang])


def first_line(s):
    return s.split("\n")[0].strip()


def pcorrect(ok):
    return sum(ok) / len(ok) if ok else 0.0


def classify(greedy_ok, sample_ok):
    if greedy_ok and all(greedy_ok):
        return "HighlyKnown"
    if any(greedy_ok):
        return "MaybeKnown"
    if any(sample_ok):
        return "WeaklyKnown"
    return "Unknown"


class Generator:
    def __init__(self, model_id, device):
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
        self.device = device
        self.nl = self.tok("\n", add_special_tokens=False).input_ids[-1]

    @torch.no_grad()
    def run(self, prompts, sample, n, temperature, batch, top_p=1.0):
        outs = []
        for i in range(0, len(prompts), batch):
            enc = self.tok(prompts[i:i + batch], return_tensors="pt", padding=True).to(self.device)
            kw = dict(max_new_tokens=20, pad_token_id=self.tok.pad_token_id, eos_token_id=[self.tok.eos_token_id, self.nl])
            if sample:
                kw.update(do_sample=True, temperature=temperature, top_p=top_p, num_return_sequences=n)
            else:
                kw.update(do_sample=False)
            gen = self.model.generate(**enc, **kw)
            text = self.tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
            k = n if sample else 1
            outs += [[first_line(t) for t in text[j:j + k]] for j in range(0, len(text), k)]
            print(f"  {min(i + batch, len(prompts))}/{len(prompts)}", end="\r", file=sys.stderr)
        print(file=sys.stderr)
        return outs


def probe_language(gen, questions_per_fact, golds_per_fact, lang, nex=4, samples=8,
                   temperature=0.5, top_p=1.0, batch=32):
    """One SliCK pass over every fact in one language.

    questions_per_fact: per fact, the list of phrasings (make_qa.py gives 2).
    Returns, per fact: {label, p_correct_greedy, p_correct_sample, n_greedy, n_sample,
                        greedy, samples, abstain_greedy, wrong_greedy}.
    """
    sets = FEWSHOT_SETS[lang][:nex]
    if len(sets) < nex:
        raise ValueError(f"--nex {nex} but only {len(FEWSHOT_SETS[lang])} few-shot sets for {lang}")
    prompts, spans = [], []
    for qs in questions_per_fact:
        start = len(prompts)
        prompts += [t.format(q=q) for q in qs for t in sets]
        spans.append((start, len(prompts)))
    greedy = gen.run(prompts, sample=False, n=1, temperature=0, batch=batch)
    smp = gen.run(prompts, sample=True, n=samples, temperature=temperature, top_p=top_p,
                  batch=max(4, batch // max(1, samples)))
    out = []
    for (a, b), golds in zip(spans, golds_per_fact):
        g = [greedy[i][0] for i in range(a, b)]
        s = [x for i in range(a, b) for x in smp[i]]
        g_ok = [is_correct(x, golds) for x in g]
        s_ok = [is_correct(x, golds) for x in s]
        out.append({"label": classify(g_ok, s_ok),
                    "p_correct_greedy": round(pcorrect(g_ok), 4), "p_correct_sample": round(pcorrect(s_ok), 4),
                    "f1_greedy_max": round(max((max_f1(x, golds) for x in g), default=0.0), 3),
                    "f1_sample_max": round(max((max_f1(x, golds) for x in s), default=0.0), 3),
                    "n_greedy": len(g), "n_sample": len(s), "greedy": g, "samples": s,
                    "abstain_greedy": sum(is_abstain(x, lang) for x in g),
                    "wrong_greedy": sum((not ok) and (not is_abstain(x, lang)) for x, ok in zip(g, g_ok))})
    return out



def saturation_rows(res, qa, nex, samples, n_phr=2, grid=None):
    """Relabel every fact at smaller budgets using the generations already in hand — no extra
    inference. Answers "how many attempts is enough", which is the setting the full sweep runs at.

    res: {"en": [row, ...], "ja": [row, ...]} as returned by probe_language, in the order of qa.
    The curve can only fall: more attempts reveal knowledge, they never hide it. Where it flattens,
    extra sampling buys nothing; below that, facts the model partly knows are labelled Unknown and
    leak into the training set.
    """
    if grid is None:
        grid = [(nx, sm) for nx in range(1, nex + 1)
                for sm in sorted({0, 1, 2, 4, samples // 2, samples} & set(range(samples + 1)))]
    out = []
    for nx, sm in grid:
        unk, both = {"en": 0, "ja": 0}, 0
        for i in range(len(qa)):
            lab = {}
            for lang in ("en", "ja"):
                r = res[lang][i]
                g, sp = [], []
                for ph in range(n_phr):
                    for t in range(nx):
                        idx = ph * nex + t          # probe_language orders prompts phrasing-major
                        g.append(r["greedy"][idx])
                        sp += r["samples"][idx * samples: idx * samples + sm]
                golds = golds_for(qa[i], lang)
                lab[lang] = classify([is_correct(x, golds) for x in g],
                                     [is_correct(x, golds) for x in sp])
                unk[lang] += lab[lang] == "Unknown"
            both += lab["en"] == lab["ja"] == "Unknown"
        out.append({"nex": nx, "samples": sm, "gen_per_fact": 2 * nx * n_phr * (1 + sm),
                    "unknown_en": unk["en"], "unknown_ja": unk["ja"], "inject": both, "n": len(qa)})
    return out


def golds_for(q, lang):
    """EN scores against the English surface forms only; JA also accepts them, since Qwen often
    answers a Japanese question with the Latin-script name."""
    return [q["answer"]] + q.get("aliases_en", []) + (q.get("aliases_ja", []) if lang == "ja" else [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default="experiment_data/eval_qa.jsonl")
    ap.add_argument("--facts", default="experiment_data/facts.jsonl")
    ap.add_argument("--out", default="experiment_data/facts_probed.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--nex", type=int, default=4, help="few-shot prompt variants (SliCK Nex); max 4")
    ap.add_argument("--samples", type=int, default=8, help="temperature samples per prompt")
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--top-p", dest="top_p", type=float, default=1.0, help="1.0 = untruncated, as in the paper")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    device = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    qa = [json.loads(l) for l in open(a.qa, encoding="utf-8")]
    facts = {f["fact_id"]: f for f in (json.loads(l) for l in open(a.facts, encoding="utf-8"))}
    qa = [q for q in qa if q["fact_id"] in facts]
    if a.limit:
        qa = qa[:a.limit]
    n_phr = len(qa[0]["q_en"][:2])
    per_lang = a.nex * n_phr * (1 + a.samples)
    print(f"device {device} | model {a.model} | facts {len(qa)} | "
          f"nex {a.nex} x {n_phr} phrasings = {a.nex * n_phr} greedy + {a.nex * n_phr * a.samples} samples "
          f"@ T={a.temperature} top_p={a.top_p} | {2 * per_lang} generations/fact")

    t0 = time.time()
    gen = Generator(a.model, device)
    results = {q["fact_id"]: {} for q in qa}
    for lang, key in (("en", "q_en"), ("ja", "q_ja")):
        rows = probe_language(gen, [q[key][:2] for q in qa], [golds_for(q, lang) for q in qa], lang,
                              nex=a.nex, samples=a.samples, temperature=a.temperature,
                              top_p=a.top_p, batch=a.batch)
        for q, r in zip(qa, rows):
            results[q["fact_id"]][lang] = r
        print(f"  [{lang}] done ({(time.time() - t0) / 60:.1f} min elapsed)")

    dist = collections.Counter()
    with open(a.out, "w", encoding="utf-8") as out:
        for q in qa:
            f = dict(facts[q["fact_id"]]); r = results[q["fact_id"]]
            f["known_en"], f["known_ja"] = r["en"]["label"], r["ja"]["label"]
            f["probe"] = r
            dist[(f["known_en"], f["known_ja"])] += 1
            out.write(json.dumps(f, ensure_ascii=False) + "\n")

    print(f"\nwrote {len(qa)} facts -> {a.out}   ({time.time() - t0:.0f}s)")
    for lang in ("en", "ja"):
        c = collections.Counter(results[q["fact_id"]][lang]["label"] for q in qa)
        print(f"  {lang}: " + "  ".join(f"{k} {c[k]}" for k in LABELS))
    both = sum(v for (e, j), v in dist.items() if e == "Unknown" and j == "Unknown")
    print(f"  Unknown in BOTH languages (injection set): {both} / {len(qa)}"
          + ("   <- below the 300 floor, collect more articles" if both < 300 else ""))
    near = sum(1 for q in qa if all(results[q["fact_id"]][l]["label"] == "Unknown" for l in ("en", "ja"))
               and max(results[q["fact_id"]][l]["f1_greedy_max"] for l in ("en", "ja")) >= 0.8)
    print(f"  of those, near-misses at word-F1 >= 0.8 (inspect before injecting): {near}")
    n_g = a.nex * n_phr
    abst = {lang: sum(results[q['fact_id']][lang]['abstain_greedy'] for q in qa) / (n_g * len(qa)) for lang in ("en", "ja")}
    print(f"  greedy abstention rate: en {abst['en']:.0%}  ja {abst['ja']:.0%}")


if __name__ == "__main__":
    main()
