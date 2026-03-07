# Precision@10 Failure Analysis

## Run Context

- Run directory: `pylate/results/compression_eval_amazon/20260220_235637`
- Model: `lightonai/GTE-ModernColBERT-v1`
- Dataset path in run metadata: `../amazon_dataset/beir_format_full`
- Configs evaluated: 16
- Baseline (`config_0`) metrics:
  - `precision@10 = 0.12017543859649124`
  - `recall@10 = 0.5330383249791145`
  - `mrr@10 = 0.4736119813422446`
  - `map = 0.4125807726590443`

---

## Executive Summary

`precision@10` is low primarily because of **label sparsity and dataset construction artifacts**, not only because retrieval quality is poor.

Main drivers:

1. **Low positives per query ceiling**
   - 456 total queries
   - 408 queries with any positive label (after qrels deduplication as used by evaluation)
   - 48 queries with no positive labels at all
   - 980 positive query-doc labels total
   - Average positives/query:
     - over all queries: `2.1491`
     - over positive-only queries: `2.4020`
   - This imposes a hard cap:
     - max possible `precision@10` over all 456 queries: `0.2149`
     - max possible `precision@10` over positive-only queries: `0.2402`

2. **Many queries are broad/underspecified**
   - Query length distribution:
     - mean words/query: `4.58`
     - median: `4`
     - queries with <=3 words: `186 / 456`
     - queries with <=5 words: `329 / 456`
   - Broad requests (e.g., `"horror"`, `"tv shows"`, `"new action movies"`) retrieve plausible items that are frequently unjudged as relevant, which hurts judged precision.

3. **Qrels and corpus construction issues amplify metric suppression**
   - Qrels raw rows: 1190, unique qid-doc pairs: 1180 (10 duplicates)
   - Some duplicate qid-doc pairs have conflicting relevance scores; the evaluator keeps the last occurrence.
   - Corpus IDs are movie titles; titles are not unique:
     - corpus rows: 898
     - unique IDs indexed: 877
     - 16 duplicated ID values, 21 extra duplicate instances
   - Duplicate IDs can merge/collide documents and reduce stable judged matching.

4. **Compression at aggressive keep ratios worsens candidate quality**
   - Strongest drop at `attention_pooling keep_ratio=0.1`:
     - `precision@10: 0.1202 -> 0.0934`
     - `recall@10: 0.5330 -> 0.4062`
     - zero-hit positive queries (hits@10=0): `115 -> 164`
   - At higher retention (e.g., keep_ratio 0.75), metrics recover close to baseline.

---

## Method and What Was Inspected

Inspected artifacts:

- `results.jsonl` and `summary.tsv` for aggregate scores by config
- `config_*/runfile.json` for per-query top-k retrievals
- `amazon_dataset/beir_format_full/qrels/test.tsv` for judged relevance
- `amazon_dataset/beir_format_full/queries.jsonl` for query text
- `amazon_dataset/beir_format_full/corpus.jsonl` for doc IDs and uniqueness
- `compression_eval.py` evaluation logic
- `convert_amazon_to_beir_full_content.py` dataset conversion logic

Evaluation implementation detail:

- In `compression_eval.py`, metric computation uses `ranx_evaluate(... make_comparable=True)`.
- Qrels are loaded into a dict keyed by `(query_id, doc_id)` semantics; duplicate qid-doc rows in TSV are effectively overwritten by the last occurrence.

---

## Quantitative Breakdown

### 1) Label Ceiling Analysis

After effective qrels deduplication:

- Total queries scored: `456`
- Positive-label queries: `408`
- Zero-positive queries: `48`
- Positive pairs: `980`

Relevant-docs-per-query distribution (all queries):

- `0`: 48 queries
- `1`: 162 queries
- `2`: 101 queries
- `3`: 63 queries
- `4`: 36 queries
- `5`: 18 queries
- `6`: 13 queries
- `7`: 9 queries
- `8`: 4 queries
- `10`: 2 queries

Implication:

- 162 queries have only one relevant label, so their individual max `P@10` is `0.1`.
- 48 queries can never contribute positive precision under current qrels.
- This alone suppresses macro `precision@10` even when top results look reasonable.

### 2) Baseline vs Compression Precision Behavior

Selected configs (computed from runfiles with effective qrels):

- `config_0` baseline:
  - `p@10(all)=0.120175`
  - `p@10(pos-only)=0.134314`
  - zero-hit positive queries: `115`
- `config_1` attention pooling keep_ratio=0.1:
  - `p@10(all)=0.093421`
  - `p@10(pos-only)=0.104412`
  - zero-hit positive queries: `164`
- `config_5` attention pooling keep_ratio=0.75:
  - `p@10(all)=0.119956`
  - `p@10(pos-only)=0.134069`
  - zero-hit positive queries: `111`
- `config_6` spherical pooling f=1.333:
  - `p@10(all)=0.120833` (best in this run)
  - `p@10(pos-only)=0.135049`
  - zero-hit positive queries: `109`

Takeaway:

- The metric floor/ceiling is dominated by qrels sparsity.
- Compression mostly changes how many queries become zero-hit and how many relevant docs survive in top10.

### 3) Baseline Failure by Number of Relevant Labels

For baseline (`config_0`), by rel-count bucket:

- rel=1: 162 queries, zero-hit rate `35.8%`, mean P@10 `0.0642`
- rel=2: 101 queries, zero-hit rate `24.8%`, mean P@10 `0.1267`
- rel=3: 63 queries, zero-hit rate `27.0%`, mean P@10 `0.1619`
- rel=4: 36 queries, zero-hit rate `16.7%`, mean P@10 `0.2028`
- rel=5: 18 queries, zero-hit rate `22.2%`, mean P@10 `0.2500`
- rel=6: 13 queries, zero-hit rate `15.4%`, mean P@10 `0.3154`
- rel=7: 9 queries, zero-hit rate `22.2%`, mean P@10 `0.3000`
- rel=8: 4 queries, zero-hit rate `25.0%`, mean P@10 `0.3250`
- rel=10: 2 queries, zero-hit rate `0%`, mean P@10 `0.7500`

Interpretation:

- For queries with sparse labels (especially rel=1), even decent ranking behavior gives very low P@10.
- A large share of missing precision is mathematically constrained, not just model ranking error.

---

## Failure Cases (Concrete Examples)

### A) Broad query, semantically plausible results, but zero judged hits

1. `q_309`: `"horror"` (8 judged positives)
- Baseline top10: horror-like titles (e.g., `Zone of the Dead`, `Alien: Romulus`, `It Follows`) but none are in the judged positive set.
- `hits@10`: baseline `0`, cfg1 `0`.
- Failure type: judged-pool mismatch for broad intent.

2. `q_20`: `"kids halloween movie"` (5 judged positives)
- Baseline top10 includes many kid/halloween titles (`Spooky Buddies`, `Goosebumps 2`, `Monster House`, etc.).
- Judged positives are specifically:
  - `Hotel Transylvania`
  - `Hotel Transylvania 3: Summer Vacation`
  - `The Addams Family`
  - `The Addams Family 2`
  - `The Little Vampire`
- `hits@10`: baseline `0`, cfg1 `0`.
- Failure type: narrow labeling for a broad natural-language request.

### B) Compression-induced degradation (baseline succeeds, aggressive pooling fails)

1. `q_56`: `"fast paced action movies"` (7 judged positives)
- Baseline hits@10: `5`
- cfg1 hits@10: `2`
- Relevant titles dropped from top10 include known judged matches (`Mission: Impossible - Fallout`, `John Wick`, `Guns Akimbo`).

2. `q_294`: `"coming of age movies"` (5 judged positives)
- Baseline hits@10: `5`
- cfg1 hits@10: `2`
- Baseline ranks all judged positives in top10; aggressive pooling shifts toward less matched titles.

3. `q_306`: `"movies about undercover operations"` (6 judged positives)
- Baseline hits@10: `3`
- cfg1 hits@10: `0`
- Strong evidence that aggressive compression damages fine-grained lexical/semantic matching.

### C) Lexical trap / ambiguous short query

Examples where top1 shares surface token(s) but misses judged target:

- `q_227`: `ryan` -> top1 `Jack Ryan: Shadow Recruit` while judged positive is `The Short Game`
- `q_251`: `notting hill` -> top1 `Silent Hill` while judged positive is `Love Actually`
- `q_400`: `movies from m. night shyamalan` -> top1 `The Night Comes for Us` while judged positive is `After Earth`

These are typical false positives for short, ambiguous, or noisy queries.

---

## Data Pipeline Findings That Affect Precision@10

### 1) Conversion keeps zero-relevance rows in qrels

In `convert_amazon_to_beir_full_content.py`, qrels are written directly from `query_relevance_score` without filtering to positive labels.

Effects:

- Queries with only zero labels remain in qrels and metric denominator (under comparable evaluation).
- This creates hard zero contributions for those queries.

### 2) Non-unique document IDs

In `convert_amazon_to_beir_full_content.py`, `_id` is set to `title`.

Effects:

- Title collisions reduce distinct indexed documents (`898 -> 877` unique IDs).
- Collisions can blur retrieval targets and judged mapping.

### 3) Duplicate qid-doc rows with conflicting relevance

- Raw qrels contain duplicate qid-doc rows with score conflicts.
- Last occurrence wins in dict-based qrels loading.
- This can flip relevance status and alter query-level judged positives.

---

## Why Precision@10 Looks Much Lower Than Recall@10 / MRR@10

- `precision@10` penalizes every non-judged/non-relevant item in top10.
- With ~2.15 positives per query on average (across all queries), even ideal systems cannot place many relevant docs in top10.
- `recall@10` and `mrr@10` answer different questions:
  - `recall@10`: Did we retrieve known relevant items at all?
  - `mrr@10`: Did we get at least one relevant item early?
- In sparse-label setups, it is common for `P@10` to look low while `MRR@10` remains moderate.

---

## Recommended Fixes (Priority Order)

### Priority 1: Fix qrels construction/evaluation denominator

1. Remove or separately evaluate zero-only queries:
   - Exclude queries with no positive qrels from macro precision denominator, or report two views:
     - all-query metric
     - positive-qrels-only metric

2. Deduplicate qrels explicitly with deterministic policy:
   - For duplicate `(query_id, doc_id)`, keep max score (recommended) rather than last row order.

Expected impact:

- Immediate increase in reported `precision@10` interpretability.
- Reduced instability from TSV row order artifacts.

### Priority 2: Make document IDs unique

1. Use stable unique IDs from source corpus (not `title` alone).
2. Keep title as metadata field only.

Expected impact:

- Eliminates ID collisions (`898` docs should stay `898` indexed docs).
- Cleaner judged matching and less retrieval ambiguity.

### Priority 3: Improve label coverage for broad queries

1. Expand judged positives for broad intents (`horror`, `kids halloween movie`, `new action movies`, etc.).
2. Consider pooling top-k from several retrievers and adding judgments.

Expected impact:

- Raises precision ceiling.
- Reduces apparent false positives that are actually unlabeled true positives.

### Priority 4: Compression strategy tuning

1. Avoid highly aggressive pooling (`keep_ratio=0.1`) for this dataset.
2. Use moderate/high retention (>=0.5) or spherical pooling settings near current best.

Expected impact:

- Preserves top-10 hit density, especially on mid-specificity queries.

---

## Suggested Reporting Template for Future Runs

Report both:

1. `P@10(all queries)`
2. `P@10(positive-qrels-only queries)`
3. `zero-hit-positive-query count`
4. `avg positives/query` and `P@10 theoretical ceiling`

This prevents misdiagnosing low `P@10` as purely model failure when it is often label-limited.

---

## Appendix: High-Impact Query Lists

### Hard baseline misses (`>=3` positives, baseline hits@10 = 0)

- `q_309` - `horror`
- `q_299` - `new action movies`
- `q_343` - `action comedy shows`
- `q_205` - `kid kids halloween movies`
- `q_347` - `what new movies are out`
- `q_136` - `hindi movies`
- `q_19` - `new releases this month`
- `q_20` - `kids halloween movie`
- `q_390` - `play kids halloween movies`
- `q_109` - `classic kids halloween movies`
- `q_14` - `halloween kid shows`
- `q_193` - `recent movie releases from 2024`
- `q_194` - `Suggest me something chilling to watch`
- `q_331` - `siri can you find some free kids movies`
- `q_68` - `golden globes comedy films from twenty twenty`
- `q_101` - `what are some good scary halloween movies`
- `q_121` - `take me to halloween kid shows`
- `q_130` - `tv shows`
- `q_298` - `kiss`
- `q_301` - `some not too scary pre-teen movies`

### Largest drops from baseline to aggressive attention pooling (`config_1`)

- `q_56` (`fast paced action movies`) drop `5 -> 2`
- `q_306` (`movies about undercover operations`) drop `3 -> 0`
- `q_294` (`coming of age movies`) drop `5 -> 2`
- `q_129` (`catfish`) drop `3 -> 0`
- `q_84` (`scarlett johansson horror movies`) drop `2 -> 0`
- `q_79` (`show me new u. u. f. o. movies i haven't seen`) drop `2 -> 0`
- `q_63` (`new action tv shows`) drop `2 -> 0`
- `q_50` (`madagascar`) drop `5 -> 3`
- `q_39` (`egyptian cinema with cultural traditions`) drop `2 -> 0`
- `q_382` (`the view`) drop `2 -> 0`

---

## Bottom Line

Your `precision@10` is low mainly because:

1. The benchmark has sparse positives/query and many zero-positive queries, so the metric ceiling is low (`~0.215` overall).
2. Broad short queries frequently retrieve plausible but unjudged titles.
3. Dataset construction artifacts (non-unique doc IDs, duplicate/conflicting qrels rows) further suppress precision.
4. Aggressive compression exacerbates misses by reducing fine-grained matching capacity.

The system is not simply “bad at retrieval”; it is operating in a label-limited evaluation setup with additional data hygiene issues.
