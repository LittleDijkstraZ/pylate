# Baseline Failure Analysis — Amazon Dataset

Run: `20260220_235637` | Model: `lightonai/GTE-ModernColBERT-v1`

This document gives a full stand-alone analysis of why the uncompressed baseline is itself a weak
retriever on this dataset. Understanding the baseline ceiling is a prerequisite for interpreting any
compression analysis: degradation measured against a broken baseline conflates compression harm with
pre-existing failures.

See [FAILURE_MODE_ANALYSIS.md](FAILURE_MODE_ANALYSIS.md)(§ Mode 0) for a summary; this document goes
into more depth on root causes and proposes concrete fixes.

---

## Table of Contents

1. [Headline Numbers](#1-headline-numbers)
2. [Root Cause 1 — Label Pool Mismatch](#2-root-cause-1--label-pool-mismatch)
3. [Root Cause 2 — Label Sparsity (Single-Positive Queries)](#3-root-cause-2--label-sparsity-single-positive-queries)
4. [Root Cause 3 — Temporal Blindness](#4-root-cause-3--temporal-blindness)
5. [Root Cause 4 — Subjective / Recommendation Queries](#5-root-cause-4--subjectiverecommendation-queries)
6. [Root Cause 5 — Score Discrimination Collapse (Near-Tie Queries)](#6-root-cause-5--score-discrimination-collapse-near-tie-queries)
7. [Root Cause 6 — Semantic Intent Mismatch (Model-Level)](#7-root-cause-6--semantic-intent-mismatch-model-level)
8. [Cross-Cutting Analysis](#8-cross-cutting-analysis)
9. [Proposed Fixes](#9-proposed-fixes)
10. [Appendix](#10-appendix)

---

## 1. Headline Numbers

| Metric | Value |
|--------|-------|
| NDCG@10 | 0.4544 |
| Precision@10 | 0.1202 |
| Recall@10 | 0.5330 |
| MAP | 0.4126 |
| MRR@10 | 0.4736 |

Precision@10 = 0.12 means the average query gets **1.2 relevant documents** in its top 10 out of a
corpus of ~500 items. For context, a perfect retriever on this dataset would achieve P@10 ≈ 0.21
(since the mean number of positives per query is 2.1, capped at 10 results).

### Hits@10 distribution (queries with ≥1 positive)

| Hits@10 | Queries | % |
|---------|---------|---|
| 0 (total failure) | 115 | 28.2% |
| 1 | 155 | 38.0% |
| 2 | 78 | 19.1% |
| 3 | 31 | 7.6% |
| 4+ | 29 | 7.1% |

**66.2% of queries achieve at most 1 hit.** The upper half of the score distribution is dominated by
single-hit marginal successes, not genuine multi-positive retrieval.

### Rank of first positive per query

| First positive at … | Queries | % |
|--------------------|---------|---|
| rank 1 | 179 | 43.9% |
| rank 2–3 | 61 | 15.0% |
| rank 4–5 | 23 | 5.6% |
| rank 6–10 | 30 | 7.4% |
| rank 11–20 | 32 | 7.8% |
| rank 21–50 | 20 | 4.9% |
| rank 51–100 | 26 | 6.4% |
| not in top 100 | 37 | 9.1% |

The positive is at rank 1 in 44% of queries — the model can do exact-match recall well when the
signal is unambiguous. The 9.1% "not found" queries are almost entirely pool-mismatch (Root Cause 1).

### P@10 by query type

| Query type | n | Avg P@10 | Zero-hit % |
|-----------|---|----------|-----------|
| Title / entity lookup (`"nosferatu"`, `"notting hill"`) | 79 | **0.161** | 23% |
| Descriptive phrase (`"ghost in the shell"`, `"the way home"`) | 73 | 0.141 | 21% |
| Genre / category (`"hindi movies"`, `"coming of age movies"`) | 178 | 0.132 | 30% |
| Temporal (`"new releases this month"`, `"twenty twenty five movies"`) | 43 | 0.119 | 35% |
| Actor / director (`"movies with john goodman"`) | 31 | 0.100 | 32% |
| Recommendation / subjective (`"suggest something chilling"`) | 4 | **0.025** | 75% |

The model handles direct title lookup and descriptive phrase matching reasonably well. It degrades
sharply for temporal, actor-structured, and subjective queries — three categories that require
signals absent from or poorly represented in the corpus text.

---

## 2. Root Cause 1 — Label Pool Mismatch

### What it is

68 of the 115 zero-hit queries fail **consistently across all 16 configs**. The judged positives for
these queries are never retrieved by any system in the top 100. Because unjudged documents are treated
as non-relevant (`make_comparable=True` in ranx), all retrieved plausible results are penalized as
wrong.

### Evidence

| Query | # Judged positives | What baseline retrieves (plausible) | Conflict |
|-------|-------------------|-------------------------------------|---------|
| `q_309` `"horror"` | 8 | `Zone of the Dead`, `Alien: Romulus`, `It Follows` | Retrieved docs are valid horror; none happen to be among the 8 judged ones |
| `q_130` `"tv shows"` | 3 | `The Eichmann Show`, `Mickey Mouse Clubhouse+`, `Alien: Earth` | Positives are `Boston Blue`, `Robin Hood`, `Sheriff Country` — niche titles |
| `q_347` `"what new movies are out"` | 3 | `Baby Driver`, `Ghost in the Shell: The New Movie` | Positives are `The Fantastic Four: First Steps`, `Mufasa: The Lion King`, `Elio` |
| `q_193` `"recent movie releases from 2024"` | 4 | `We Live in Time`, `Alien: Romulus` | Positives are `Sonic the Hedgehog 3`, `A Complete Unknown`, `That Christmas` |
| `q_68` `"golden globes comedy films from twenty twenty"` | judged set | Various plausible titles | None judged as positive |

In all these cases, the model is retrieving *semantically reasonable* results. The failure is in the
**judgment pool construction**, not in the model.

### Root cause: narrow judging pool

The dataset was built from Amazon assistant interaction logs. Qrels were likely derived by labeling
only the responses that the assistant actually returned — meaning:

1. **One-system pool**: positives are anchored to what a single production system returned. Any
   document the production system did not rank in its top-k was never judged.
2. **No negative sampling round**: broad queries like `"horror"` should have dozens of valid
   positives in a 500-item corpus, but the judged set is only 3–8 items corresponding to the
   specific titles the assistant surfaced.
3. **Recency label mismatch**: temporal queries like `"new releases"` require the positives to be
   *new*, but the corpus contains both new (2024–2025) and old films without release-date metadata.
   The system that originally answered these queries had access to freshness signals the embedding
   model does not.

### Impact on metrics

Removing the 68 consistent pool-mismatch queries from the evaluation denominator raises the effective
baseline P@10 from 0.1202 to approximately **0.145–0.15** on the remaining queries. This is still
poor but reflects the actual retrieval quality for evaluable queries.

### Proposed fixes

| Fix | Level | Effort | Impact |
|-----|-------|--------|--------|
| **Pooled evaluation**: re-judge retrieved results from all configs jointly; add any retrieved relevant doc to qrels | Evaluation | Medium | Eliminates mismatch failures |
| **Pseudo-label expansion**: use an independent strong model (e.g. GPT-4 scoring) to label all top-50 results per query; replace the current narrow qrels | Dataset | High | Correct fix |
| **Filter unevaluable queries**: exclude queries where 0 retrieved documents across all configs appear in the qrels pool; report separately | Evaluation | Low | Prevents inflation of failure rate |
| **Log pool-mismatch count**: add a diagnostic that counts queries with all-zero hits across all configs and reports them as `pool_mismatch_n` | [compression_eval.py:920–934](../../../../../experiments/compression/compression_eval.py#L920-L934) | Very Low | Transparency |

---

## 3. Root Cause 2 — Label Sparsity (Single-Positive Queries)

### What it is

162 queries (39.7% of all queries with positives) have exactly **1 judged positive**. With only one
valid document in a corpus of ~500 items, any rank outside top 10 results in P@10 = 0 and NDCG@10 ≈ 0.
The baseline fails completely on **58 of these 162 queries** — it retrieves plausible documents but
misses the single specific labeled one.

### Evidence: positive is close but not in top 10

Sorted by actual rank of the single positive in the baseline:

| Query | Query text | Positive document | Actual rank | Baseline top-2 |
|-------|-----------|-------------------|-------------|----------------|
| `q_208` | `"blockbuster hindi action films with romance"` | `Bajirao Mastani` | 11 | `Khiladi 786`, `Gadar: Ek Prem Katha` |
| `q_308` | `"find post apocalypse drama"` | `A Quiet Place Part II` | 11 | `After the Apocalypse`, `28 Days Later` |
| `q_214` | `"viral fantasy shows"` | `Avatar: The Last Airbender` | 12 | `Love Game in Eastern Fantasy` |
| `q_351` | `"find new Guardians of the Galaxy"` | `Avengers: Endgame` | 12 | `Legend of the Guardians`, `Godzilla x Kong` |
| `q_21` | `"space movie"` | `Starship Troopers 2: Hero of the Federation` | 13 | `Beyond White Space`, `Solo: A Star Wars Story` |
| `q_230` | `"Show me something quirky to watch later"` | `Napoleon Dynamite` | 13 | `John Mulaney & the Sack Lunch Bunch` |
| `q_261` | `"new shows during quarantine binge"` | `The Edge of Sleep` | 14 | `Alien: Earth`, `Rescue: HI-Surf` |
| `q_52` | `"Something that mythology fans and history lovers will enjoy"` | `King Arthur: Legend of the Sword` | 16 | `Only Lovers Left Alive`, `The Amazing Spider-Man 2` |

In most cases the positive is at rank 11–16 in the baseline — the model almost retrieves it. With 1
positive and a P@10 cutoff, the difference between rank 10 and rank 11 is the entire query's
contribution to the metric. This makes single-positive queries extremely volatile: a small change in
model output or even tie-breaking order can flip the entire query from P@10 = 0.1 to P@10 = 0.

### P@10 by number of positives

| # Positives | Queries | Avg baseline P@10 |
|-------------|---------|------------------|
| 0 (no positives) | 48 | 0.000 |
| 1 | 162 | **0.064** |
| 2 | 101 | 0.127 |
| 3–4 | 99 | 0.177 |
| 5–9 | 44 | 0.286 |
| 10+ | 2 | 0.750 |

The relationship is nearly linear: each additional positive approximately doubles the expected P@10.
Single-positive queries have a particularly harsh ceiling and floor (either 0.1 or 0.0 per query).

### Root cause: labeling methodology

Single-positive queries dominate because the qrels were constructed by labeling only the assistant's
final response, not a full relevance pool. If the assistant returned one title, that is the only
labeled positive. There is no mechanism for judging other relevant corpus items.

### Proposed fixes

| Fix | Level | Effort | Impact |
|-----|-------|--------|--------|
| **Exclude n_pos=1 queries from primary metric report**; report them as a separate `sparse_label` track | Evaluation | Low | Removes the noisiest signal from primary number |
| **Positive set expansion via similarity**: for each single-positive query, find all corpus documents with embedding cosine similarity > threshold to the positive and manually judge them | Dataset | Medium | Adds real positives |
| **Soft relevance scoring**: replace binary qrels with graded relevance (0/1/2) based on genre/semantic proximity to the labeled positive; reduces cliff-edge P@10 sensitivity | Dataset | High | Smoother metric |
| **Report MRR@100 as primary metric** for single-positive queries; at rank 11 the positive is found, so MRR@100 = 1/11 ≈ 0.09 rather than P@10 = 0 — gives partial credit | Evaluation | Low | Better reflects near-miss quality |

---

## 4. Root Cause 3 — Temporal Blindness

### What it is

15 temporal queries (`"new releases this month"`, `"recent blockbusters from this year"`, `"what
shows came out today"`) achieve 0 hits@10 in the baseline despite the target documents being present
in the corpus. The model cannot infer temporal ordering from static embedding content.

### Evidence

| Query | Positives | Baseline top-3 | Problem |
|-------|-----------|---------------|---------|
| `q_19` `"new releases this month"` | `Alien: Earth`, `Oh. What. Fun.`, `The Fantastic Four: First Steps` | `A Dog's Way Home`, `The Half of It`, `Good Kids` | Baseline retrieves thematically neutral films with no recency signal |
| `q_41` `"what are the new twenty twenty five movies"` | `The Legend of Ochi`, `Dune: Part Two`, `Love, Brooklyn` | `Godzilla x Kong: The New Empire`, `Beetlejuice Beetlejuice`, `Alien: Romulus` | Recent-year movies retrieved, but wrong specific titles |
| `q_76` `"show me new family movies"` | `Toy Story 4`, `Saving Bikini Bottom` | `Father There is Only One 2`, `The Addams Family 2`, `The Croods: A New Age` | Retrieves family-appropriate films but not labeled ones |
| `q_118` `"recent blockbusters from this year"` | `The Legend of Ochi`, `Black Rabbit` | `Captain America: Civil War`, `Fantastic Beasts` | Returns well-known blockbusters regardless of year |
| `q_120` `"what shows came out today"` | `9-1-1: Nashville` | `Unbreakable Kimmy Schmidt: Kimmy vs the Reverend`, `The Eichmann Show` | Retrieves TV shows in general, cannot sort by air date |
| `q_193` `"recent movie releases from 2024"` | `That Christmas`, `Sonic the Hedgehog 3`, `A Complete Unknown` | `We Live in Time`, `Alien: Romulus`, `The Strangers: Chapter 1` | All 2024-ish films, but retrieves popular ones not the labeled ones |

### Root cause: release-date signal absent from corpus

The corpus documents are text descriptions of films (title + synopsis). They do not contain a
structured `release_year` field that is factored into the embedding. When the query says `"2025
movies"`, the model matches on genre/narrative similarity, not temporal metadata. The original
production assistant had access to a database with release dates; the ColBERT model operating on text
alone does not.

Additionally, temporal queries interact severely with pool mismatch: the "correct" 2025 releases
change as new films are added, but the qrel snapshot was taken at a single point in time. The
specific labeled positives (e.g., `The Legend of Ochi`) are just as "recent" as many other 2025
films in the corpus that are not labeled.

### Proposed fixes

| Fix | Level | Effort | Impact |
|-----|-------|--------|--------|
| **Add release year to corpus text**: prepend `"[Released: 2024]"` to each document's text field before encoding | Dataset / Corpus preprocessing | Low | Allows the embedding to capture year tokens |
| **Metadata BM25 hybrid**: build a separate BM25 index over structured metadata (year, genre, platform availability) and fuse it with ColBERT scores at retrieval time | System | Medium | Handles temporal and category filters without retraining |
| **Temporal query detection + filter**: classify queries as temporal and retrieve only from a date-filtered subcorpus rather than the full corpus | Pipeline | Medium | Effectively narrows the search space |
| **Exclude purely-temporal queries from main eval**: report them separately as `temporal_queries` track; their failures reflect metadata availability, not embedding quality | Evaluation | Low | Prevents temporal failures from polluting the aggregate |

---

## 5. Root Cause 4 — Subjective / Recommendation Queries

### What it is

Queries that express user preference or taste (`"recommend something chilling"`, `"find me good scary
halloween movies"`, `"show me something quirky"`) fail at a 75% zero-hit rate. These are the weakest
query category by a large margin.

### Evidence

| Query | # Positives | Hits | Baseline retrieves | Expected (positives) |
|-------|------------|------|--------------------|----------------------|
| `q_194` `"Suggest me something chilling to watch"` | 4 | 0 | `Hitler: The Rise of Evil`, `The Autopsy of Jane Doe`, `Jesse Stone: Stone Cold` | `The Babadook`, `It Follows`, `Get Out`, `The Night House` |
| `q_101` `"what are some good scary halloween movies"` | 3 | 0 | `Tales of Halloween`, `Ghost Stories`, `Hell Fest` | `Annabelle: Creation`, `The Curse of La Llorona`, `Doll Graveyard` |
| `q_216` `"find me some really good scary halloween movies"` | 1 | 0 | `Tales of Halloween`, `Hell Fest`, `Boo 2! A Madea Halloween` | `The Addams Family` |
| `q_27` `"what halloween themed movies are out right now for us to watch"` | 1 | 0 | `Halloweentown II`, `Tales of Halloween`, `Boo 2! A Madea Halloween` | `The Addams Family` |
| `q_230` `"Show me something quirky to watch later"` | 1 | 0 | `John Mulaney & the Sack Lunch Bunch`, `Frequently Asked Questions About Time Travel` | `Napoleon Dynamite` |
| `q_52` `"Something that mythology fans and history lovers will both enjoy"` | 1 | 0 | `Only Lovers Left Alive`, `The Amazing Spider-Man 2`, `The Favourite` | `King Arthur: Legend of the Sword` |

### Root cause: quality / canonicality signals absent from embeddings

Subjective queries require world knowledge about which films are *considered* good, popular, or
canonical in their category. For example:

- `"scary halloween movies"` → the labeled positives are `Annabelle: Creation` and `The Curse of La
  Llorona`, which are popular mainstream horror titles. The baseline retrieves `Tales of Halloween`
  (indie anthology), `Ghost Stories` (British horror), `Hell Fest` (slasher) — all are plausible
  but below the cultural salience threshold of the labeled positives.
- `"chilling"` → `The Babadook` and `It Follows` are critically acclaimed slow-burn horror;
  `Get Out` is culturally prominent. The baseline retrieves thriller-adjacent titles with no
  signal for critical reception or cultural importance.

The model cannot distinguish between "any horror film" and "critically recognized horror films" because:
1. **No popularity/rating metadata** in the corpus text (Rotten Tomatoes score, box office, awards).
2. **No conversational context**: a recommendation query carries implicit user preferences (genre,
   mood, cultural canon) that are not recoverable from the query text alone.
3. **Label subjectivity bias**: the labels themselves reflect one specific assistant's preferences
   or one user's stated satisfaction — they may not be universally agreed-upon "correct" answers.

### Proposed fixes

| Fix | Level | Effort | Impact |
|-----|-------|--------|--------|
| **Enrich corpus with quality signals**: add metadata fields — Rotten Tomatoes score, IMDB rating, box office rank, award mentions — to corpus documents before embedding | Dataset | Medium | Allows the model to differentiate popular/canonical from obscure |
| **Reranker with popularity prior**: after ColBERT retrieval, rerank top-50 using a popularity signal (e.g., Wikipedia click rate, IMDB vote count); multiply into score | Pipeline | Low | Cheap post-processing fix |
| **Conversational context modeling**: for recommendation queries, pass prior user preferences (genre history, liked titles) as additional context to the query encoder | Model | High | Addresses the intent underspecification problem fundamentally |
| **Separate recommendation queries from lookup queries in evaluation**: recommendation queries are fundamentally multi-label with subjective positives; they should use preference-based evaluation (e.g., NDCG with graded labels from multiple annotators) rather than binary P@10 | Evaluation | Medium | Correct framing |

---

## 6. Root Cause 5 — Score Discrimination Collapse (Near-Tie Queries)

### What it is

For **93 / 456 queries (20.4%)**, the baseline's own top-10 score spread (max − min within top 10) is
below 0.05. In these queries the model assigns nearly identical scores to all top-10 results — the
ranking is effectively a tie and is not meaningful.

### Distribution

| Score spread band | Queries | % |
|-------------------|---------|---|
| < 0.05 (critically low — near tie) | 93 | 20.4% |
| 0.05 – 0.10 (low) | 129 | 28.3% |
| 0.10 – 0.20 (moderate) | 180 | 39.5% |
| > 0.20 (high discrimination) | 54 | 11.8% |

Nearly half of all queries (48.7%) have spread below 0.10, meaning even small perturbations can
change the rank order.

### Representative examples

- `q_102` (`"sci-fi movies"`): spread = 0.0264. Top-3 scores: `Star Runners` (6.453), `Alien:
  Romulus` (6.447), `Pinocchio 3000` (6.440). All three scores differ by < 0.02. Random tie-breaking
  noise can determine the final ranking.
- `q_112` (`"something for stand-up comedy fans and casual viewers"`): spread = 0.0474. `John Mulaney:
  Kid Gorgeous` (17.612), `Dying Laughing` (17.604), `Dave Attell: Captain Miserable` (17.599) —
  <0.02 difference across 10 results.
- `q_118` (`"recent blockbusters from this year"`): spread = 0.0405 with all-wrong results (positives
  are `The Legend of Ochi`, `Black Rabbit`).
- `q_120` (`"what shows came out today"`): spread = 0.0244 — almost perfectly flat.

### Root cause: genre-level embedding saturation

When a query activates a broad genre (all sci-fi, all comedy specials), many documents are
semantically near-equivalent at the level of the query's embedding. The MaxSim score for
all of them converges to approximately the same value because the query tokens all match
high-cosine-similarity document tokens in the same semantic neighborhood. There is no sub-genre
signal in the query to disambiguate.

This is a fundamental limitation of dense retrieval for broad categorical queries: the query
vector lies in the center of a high-density cluster, and all cluster members score similarly.
Sparse retrieval (BM25) partially avoids this by requiring exact lexical overlap for boosting.

### Proposed fixes

| Fix | Level | Effort | Impact |
|-----|-------|--------|--------|
| **BM25 hybrid scoring**: combine ColBERT MaxSim with BM25 score (e.g. linear interpolation or RRF); BM25 provides differentiation through term-frequency variance where dense scores tie | Pipeline | Low | Standard improvement for near-tie queries |
| **Finer-grained query expansion**: for broad categorical queries, use LLM query rewriting to inject sub-genre discriminators (`"sci-fi movies"` → `"science fiction films featuring space exploration or AI themes"`) before encoding | Pipeline | Medium | Breaks the tie by enriching the query vector |
| **NDCG with graded relevance instead of P@10**: for genre queries where many docs are plausible, assign partial relevance to near-miss documents; binary P@10 gives 0 when the model retrieves the right genre even if not the exact labeled doc | Evaluation | Medium | Better captures near-success |
| **Report scores + spread alongside metrics**: add a diagnostic output column for per-query score spread; flag low-spread queries in the evaluation log for manual review | [compression_eval.py](../../../../../experiments/compression/compression_eval.py) | Very Low | Diagnostic transparency |

---

## 7. Root Cause 6 — Semantic Intent Mismatch (Model-Level)

### What it is

For well-formed, specific queries, the model retrieves documents whose surface token overlap with the
query is high but whose actual relevance is wrong. The model matches on partial semantic similarity
rather than full query intent.

### Evidence

| Query | Positive | Baseline top-1 | Why it's wrong |
|-------|---------|---------------|----------------|
| `q_147` `"Recommend me something epic to watch"` | `Avengers: Endgame` | `Dune: Part Two` | `Dune` matches `epic` well; `Avengers` is more "popular/crowd-pleasing epic" — different meaning of epic |
| `q_351` `"find new Guardians of the Galaxy"` | `Avengers: Endgame` | `Legend of the Guardians: The Owls of Ga'Hoole` | Title matches `"Guardians"` keyword literally; misses franchise intent |
| `q_163` `"nosferatu"` | `The Vulture's Eye`, `Embrace of the Vampire` | `She Creature`, `The Night House` | Uses the genre/mood embedding of Nosferatu (vampire/gothic horror) rather than treating it as a proper noun for a specific adaptation |
| `q_52` `"Something that mythology fans and history lovers will both enjoy"` | `King Arthur: Legend of the Sword` | `Only Lovers Left Alive`, `The Amazing Spider-Man 2` | Cross-genre connective phrase `"fans of X and Y"` is hard to parse; model activates mythology and vampire-mythology overlap |
| `q_208` `"blockbuster hindi action films with romance"` | `Bajirao Mastani` | `Khiladi 786` | Both are Hindi action films with romance; the labeled positive is a prestige film, but both are valid — this is as much a label quality issue as a model issue |

### Root cause

`GTE-ModernColBERT-v1` is a general-purpose dense retrieval model trained on MS-MARCO and similar
academic IR datasets consisting of factual web queries. The Amazon dataset's queries are:

1. **Conversational and implicit**: `"something epic"`, `"something chilling"` — the user's intent is
   inferred from conversational context that is absent from the query string.
2. **Franchise / sequel aware**: `"new Guardians of the Galaxy"` implies searching the MCU franchise
   extension, not any movie with "Guardians" in the title. This requires entity-linking knowledge.
3. **Quality-tier sensitive**: `"Bajirao Mastani"` as a positive implies "critically acclaimed / big
   budget Bollywood" not just "hindi action romance" — a distinction the model's training data does
   not cover.
4. **Actor-as-descriptor**: `"emma stone horror films"` — the model must know which films feature Emma
   Stone *and* be horror, which requires entity knowledge beyond what colbert token embeddings
   encode from text alone.

### Proposed fixes

| Fix | Level | Effort | Impact |
|-----|-------|--------|--------|
| **Domain fine-tuning**: collect (query, relevant-doc) pairs from Amazon/streaming recommendation logs and fine-tune the model; teach it franchise-awareness and quality-tier distinctions | Model | High | Direct fix for intent mismatch |
| **Entity-linked query preprocessing**: run NER on queries to identify proper nouns (actor names, franchise names) and expand them with known aliases / filmography before encoding | Pipeline | Medium | Fixes franchise lookup failures |
| **Reranker with LLM scoring**: after ColBERT retrieves top-50, use an instruction-following LLM (GPT-4, Claude) to score each document for relevance to the full query — slow but accurate for the hard cases | Pipeline | Low code effort / high cost | Near-oracle reranker quality |
| **Multi-field corpus representation**: separate title, genre, cast, synopsis, awards into distinct ColBERT fields; compute per-field MaxSim and combine. Cast field handles actor queries; title field handles franchise queries | Dataset + Model | High | Structural fix for field-mismatch queries |

---

## 8. Cross-Cutting Analysis

### How the root causes interact

```
Pool mismatch (RC1) ──────────────────────────────────────────► 68 consistent failures (all configs)
  |
  ▼
Label sparsity (RC2) ──────────────────── near-miss at rank 11+ ► 58 complete failures
  |
  ▼
Temporal blindness (RC3) ────────────────────────────────────── ► 15 zero-hit temporal queries
  |
  ▼
Score discrimination collapse (RC5) ──── ties at broad queries  ► 93 near-tie queries (20.4%)
  |
  ▼
Semantic intent mismatch (RC6) ──────── wrong-intent retrieval  ► scattered across all categories
```

There is significant overlap between root causes. A single query can simultaneously be a pool-mismatch
case (RC1), have only 1 positive (RC2), and the model score spread is < 0.05 (RC5). The categories
are analytical — they do not partition the failure set.

### What is recoverable without retraining

The table below estimates which root causes can be addressed at the pipeline or dataset level without
touching the model weights:

| Root Cause | Fix requires model retraining? | Quick fix possible? |
|-----------|-------------------------------|---------------------|
| RC1 — Pool mismatch | No | Yes — re-judge pool |
| RC2 — Label sparsity | No | Partial — filter or expand |
| RC3 — Temporal blindness | No — add metadata | Yes — prepend year to corpus |
| RC4 — Subjective queries | Partially | Partial — enrich corpus |
| RC5 — Score discrimination | No | Yes — BM25 hybrid |
| RC6 — Intent mismatch | Yes (for deep fix) | Partial — LLM reranker |

A realistic short-term plan: fix RC1 (re-judge pool) + RC5 (BM25 hybrid) + add release year to corpus
(RC3) and re-evaluate. These three changes require no model retraining and could plausibly raise the
effective P@10 from 0.12 to 0.15–0.18 on the evaluable subset.

---

## 9. Proposed Fixes

### Priority-ordered action list

**P0 — Fix the evaluation (1–2 days)**

These do not change the model and directly surface the true baseline quality.

1. **Pool evaluation**: collect all retrieved documents across all 16 configs per query; log the
   number where 0 retrieved docs appear in qrels (pool-mismatch count). Report this as
   `pool_mismatch_n` in the evaluation JSON. Add it to [compression_eval.py:920–934](../../../../../experiments/compression/compression_eval.py#L920-L934).

2. **Filter zero-relevance queries from denominator**: in the dataset converter
   (`convert_amazon_to_beir_full_content.py`), add a step that excludes queries where all labeled
   relevance scores are 0 after pool normalization. Log them separately as `excluded_queries.tsv`.

3. **Add per-query diagnostic columns** to `results.jsonl`: `score_spread`, `n_pos`, `first_pos_rank`,
   `pool_mismatch`. These enable post-hoc segmentation without re-running evaluation.

**P1 — Fix the corpus (1 week)**

4. **Add release year metadata**: parse year from film titles or an external metadata source
   (TMDB API, Wikipedia Infobox); prepend `"[Year: YYYY] "` to each corpus document's text before
   re-encoding. Re-run the full evaluation with the enriched corpus.

5. **Add quality signals to corpus text**: prepend aggregated public metadata per film:
   `"[IMDB: 7.8, Genre: Drama, Cast: Cate Blanchett, Awards: 2 Oscars]"`. This embeds quality-tier
   and cast information into the ColBERT representation without requiring a separate model.

**P2 — Retrieval pipeline improvements (1–2 weeks)**

6. **BM25 hybrid**: compute BM25 scores over the same corpus and linearly interpolate with ColBERT
   MaxSim (e.g., `final_score = 0.7 * colbert + 0.3 * bm25`). BM25 provides lexical precision for
   genre and proper-noun queries where dense scores tie.

7. **LLM reranker for hard queries**: for queries where the baseline's top-1 score is below a
   threshold (indicating low confidence — approximately score < 6.0 based on the score distribution),
   pass the top-20 candidates to a GPT-4 / Claude API call to rerank by explicit relevance scoring.
   This handles intent-mismatch and quality-tier failures without retraining.

**P3 — Model-level fixes (weeks–months)**

8. **Domain fine-tuning**: collect ~50K (query, relevant-doc) pairs from Amazon Prime Video / IMDB
   search logs; fine-tune `GTE-ModernColBERT-v1` using hard-negative mining. Focus on temporal,
   actor, and franchise query types.

9. **Multi-field ColBERT**: encode title, synopsis, cast, and genre as separate token sequences;
   compute per-field MaxSim and sum. This naturally separates actor-name matching from
   narrative-content matching.

---

## 10. Appendix

### Dataset statistics

| Stat | Value |
|------|-------|
| Total queries | 456 |
| Queries with ≥1 positive | 408 |
| Queries with 0 positives (excluded from P@10 numerator) | 48 |
| Total positive labels (unique) | 980 |
| Mean positives per query (over 408) | 2.40 |
| Median positives per query | 1 |
| Corpus size | ~500 documents |
| Mean corpus document length | 271.4 ColBERT tokens |

### P@10 upper bound calculation

Given the label distribution, the maximum achievable P@10 for perfect retrieval (assuming all positives are retrievable):

- 162 single-positive queries: best possible P@10 = 0.1 each
- 101 two-positive queries: best possible P@10 = 0.2 each
- 99 three-to-four-positive queries: best possible P@10 ≈ 0.3–0.4 each
- 44 five-to-nine-positive queries: best possible P@10 = 0.5–0.9 each
- 2 ten-plus-positive queries: best possible P@10 = 1.0 each

Approximate upper bound P@10 (over all 456 queries, assuming pool-mismatch queries are excluded or
fixed): **~0.21**. The baseline achieves 0.12, or ~57% of the upper bound. There is meaningful room
for improvement — but only to ~0.21, not 1.0, given the label density constraints.

### Quantitative breakdown of failures

| Failure category | Queries | Overlap |
|-----------------|---------|---------|
| Pool mismatch (consistent all-config fail) | 68 | — |
| Baseline-only zero-hit (at least one config succeeds) | 47 | — |
| Single-positive, baseline fails completely | 58 | ~20 overlap with above |
| Temporal query zero-hit | 15 | ~5 overlap with pool mismatch |
| Subjective query zero-hit | 3 | included in baseline-only |
| Score spread < 0.05 (near-tie queries) | 93 | ~30 overlap with zero-hit |

Total unique queries with meaningful failure → **~195 / 456 (42.8%)** of evaluated queries are
problematic in some dimension for the baseline before any compression is applied.
