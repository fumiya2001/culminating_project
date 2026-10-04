"""
[5] Known/Unknown probe: ask the *base* model each evaluation question closed-book, in English
and Japanese, and label every fact Known or Unknown.

For each fact and language, N_EX few-shot prompts (FEWSHOT_SETS) x 2 question phrasings
(make_qa.py) = 2*N_EX prompts. From each prompt: 1 greedy answer + N_SAMPLES sampled answers (T=0.5).

  Known    at least one of the 2*N_EX*(1+N_SAMPLES) answers is correct
  Unknown  none is correct              <- only facts Unknown in BOTH languages get injected

Unknown is the same definition as in Gekhman et al. 2024 (PCorrect(T>=0) == 0). Their finer split
of Known is not used: this study only needs Unknown.

Two deliberate departures from the paper, both documented in CLAUDE.ja.md:
  * The paper draws its Nex prompts from exemplars of the *same relation* as q. Doing that here
    would leak post-cutoff facts into the prompt and break the closed-book setting, so prompt
    variation comes from FEWSHOT_SETS (fixed exemplars, all pre-cutoff) crossed with the two
    question phrasings from make_qa.py instead.
  * The paper scores with exact match. Answers here are free-form and often long
    ("American independent film distribution company"), so scoring is normalized containment
    against answer + aliases, with a standalone-token requirement for short ASCII answers.

The main path is notebook/experiment_4_probe_known.ipynb. It imports the prompts, scorer and
control set from here; generate() / probe() below are the same code as the notebook's, and the
script writes the same cache files and the same facts_probed.jsonl. Keep the two in sync.

Input : experiment_data/eval_qa.jsonl (from make_qa.py: q_en[2], q_ja[2], answer, aliases) + experiment_data/facts.jsonl
Output: experiment_data/facts_probed.jsonl = facts rows + known_en / known_ja + probe {en, ja}
        each probe entry: {label, n_correct, n_total, greedy, samples, abstain_greedy}

Usage:
  python scripts/probe_known.py --limit 20                 # trial
  python scripts/probe_known.py --batch 64                 # full run (cluster GPU)
"""
import argparse, hashlib, json, re, unicodedata
from pathlib import Path

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
FEWSHOT = {lang: sets[0] for lang, sets in FEWSHOT_SETS.items()}   # single-prompt view, used by experiment_4_finetune
ABSTAIN = {"en": ["i don't know", "i do not know", "unknown"], "ja": ["わかりません", "分かりません", "不明"]}
SHORT_GOLD = 3        # normalized chars; at or below this a gold has to match as a standalone token


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

_KANA = re.compile(r"[぀-ヿ]")
_HAN = re.compile(r"[一-鿿]")
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


def is_abstain(gen, lang):
    g = gen.lower()
    return any(a in g for a in ABSTAIN[lang])


def first_line(s):
    return s.split("\n")[0].strip()


def golds_for(q, lang):
    """EN scores against the English surface forms only; JA also accepts them, since Qwen often
    answers a Japanese question with the Latin-script name."""
    return [q["answer"]] + q.get("aliases_en", []) + (q.get("aliases_ja", []) if lang == "ja" else [])


def load_model(model_name, device):
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()
    return tok, model


@torch.no_grad()
def generate(tok, model, device, prompts, n_samples=0, batch=32, temperature=0.5):
    """Greedy once if n_samples=0, otherwise n_samples sampled answers. Returns a list of answers per prompt."""
    newline = tok("\n", add_special_tokens=False).input_ids[-1]   # stop at a newline (answers are one line)
    n = max(1, n_samples)
    step = max(1, batch // n)
    out = []
    for i in range(0, len(prompts), step):
        enc = tok(prompts[i:i + step], return_tensors="pt", padding=True).to(device)
        if n_samples:
            kw = dict(do_sample=True, temperature=temperature, top_p=1.0, num_return_sequences=n)
        else:
            kw = dict(do_sample=False)
        ids = model.generate(**enc, max_new_tokens=20, pad_token_id=tok.pad_token_id,
                             eos_token_id=[tok.eos_token_id, newline], **kw)
        texts = tok.batch_decode(ids[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        texts = [first_line(t) for t in texts]
        out += [texts[j:j + n] for j in range(0, len(texts), n)]
    return out


def probe(tok, model, device, questions_per_fact, golds_per_fact, lang,
          n_ex=4, n_samples=8, batch=32, temperature=0.5):
    """Label each fact Known / Unknown.
    questions_per_fact: per fact, the list of question phrasings (2 for the eval QA, 1 for the control set)
    golds_per_fact:     per fact, the list of accepted answers (answer + aliases)"""
    prompts, owner = [], []
    for i, questions in enumerate(questions_per_fact):
        for q in questions:
            for template in FEWSHOT_SETS[lang][:n_ex]:
                prompts.append(template.format(q=q))
                owner.append(i)

    greedy = generate(tok, model, device, prompts, batch=batch)
    samples = (generate(tok, model, device, prompts, n_samples, batch=batch, temperature=temperature)
               if n_samples else [[] for _ in prompts])

    rows = [{"greedy": [], "samples": []} for _ in questions_per_fact]
    for i, g, s in zip(owner, greedy, samples):
        rows[i]["greedy"] += g
        rows[i]["samples"] += s
    for r, golds in zip(rows, golds_per_fact):
        answers = r["greedy"] + r["samples"]
        r["n_correct"] = sum(is_correct(a, golds) for a in answers)
        r["n_total"] = len(answers)
        r["label"] = "Known" if r["n_correct"] > 0 else "Unknown"
        r["abstain_greedy"] = sum(is_abstain(a, lang) for a in r["greedy"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default="experiment_data/eval_qa.jsonl")
    ap.add_argument("--facts", default="experiment_data/facts.jsonl")
    ap.add_argument("--out", default="experiment_data/facts_probed.jsonl")
    ap.add_argument("--cache-dir", default="experiment_data/probe_cache")
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B")
    ap.add_argument("--n-ex", type=int, default=4, help="few-shot prompt variants (max 4)")
    ap.add_argument("--samples", type=int, default=8, help="sampled answers per prompt")
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=200, help="facts per cache file (resume granularity)")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    facts = {f["fact_id"]: f for f in (json.loads(l) for l in open(a.facts, encoding="utf-8"))}
    qa = [q for q in (json.loads(l) for l in open(a.qa, encoding="utf-8")) if q["fact_id"] in facts]
    if a.limit:
        qa = qa[:a.limit]
    print(f"device {device} | model {a.model} | facts {len(qa)} | "
          f"{2 * a.n_ex * (1 + a.samples)} answers per fact per language")

    tok, model = load_model(a.model, device)
    # same cache layout and keys as the notebook, so either can resume the other's run
    cache_dir = Path(a.cache_dir) / f"{a.model.split('/')[-1]}-ex{a.n_ex}-s{a.samples}-t{a.temperature}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    results = {q["fact_id"]: {} for q in qa}
    for lang in ("en", "ja"):
        for start in range(0, len(qa), a.chunk):
            part = qa[start:start + a.chunk]
            questions = [q[f"q_{lang}"][:2] for q in part]
            golds = [golds_for(q, lang) for q in part]
            key = hashlib.md5(json.dumps([[q["fact_id"] for q in part], questions, golds],
                                         ensure_ascii=False).encode()).hexdigest()[:10]
            path = cache_dir / f"{lang}-{start:05d}-{key}.json"
            if path.exists():
                rows = json.loads(path.read_text(encoding="utf-8"))
            else:
                rows = probe(tok, model, device, questions, golds, lang, n_ex=a.n_ex,
                             n_samples=a.samples, batch=a.batch, temperature=a.temperature)
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
                tmp.rename(path)
            for q, r in zip(part, rows):
                results[q["fact_id"]][lang] = r
            print(f"  [{lang}] {min(start + a.chunk, len(qa))}/{len(qa)}")

    with open(a.out, "w", encoding="utf-8") as out:
        for q in qa:
            row = dict(facts[q["fact_id"]])
            r = results[q["fact_id"]]
            row["known_en"], row["known_ja"] = r["en"]["label"], r["ja"]["label"]
            row["probe"] = r
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    both = sum(results[q["fact_id"]]["en"]["label"] == results[q["fact_id"]]["ja"]["label"] == "Unknown" for q in qa)
    print(f"\nwrote {len(qa)} facts -> {a.out}")
    print(f"  Injection set (Unknown in both languages): {both} / {len(qa)}"
          + ("   <- below the 300 floor, collect more articles" if both < 300 else ""))
    n_greedy = 2 * a.n_ex
    for lang in ("en", "ja"):
        known = sum(results[q["fact_id"]][lang]["label"] == "Known" for q in qa)
        abst = sum(results[q["fact_id"]][lang]["abstain_greedy"] for q in qa) / (n_greedy * len(qa))
        print(f"  {lang}: Known {known}  Unknown {len(qa) - known}  greedy abstention {abst:.1%}")


if __name__ == "__main__":
    main()
