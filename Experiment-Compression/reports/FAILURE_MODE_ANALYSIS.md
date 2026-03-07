# Failure Mode Analysis — Compression Eval on Amazon Dataset

Run: `20260220_235637` | Model: `lightonai/GTE-ModernColBERT-v1`

This document is a companion to [PRECISION_AT_10_FAILURE_ANALYSIS.md](PRECISION_AT_10_FAILURE_ANALYSIS.md). It focuses on *mechanistic failure modes*: why do specific configs fail on specific queries, what drives each pattern, and where in the code each issue originates.

---

## Overview of Failure Modes

Seven distinct failure modes are identified, ordered roughly by impact (Mode 0 applies to the baseline itself independently of compression):

| # | Mode | Primary Config(s) Affected | Queries Affected |
|---|------|---------------------------|-----------------|
| 0 | Baseline structural failures: pool mismatch, label sparsity, low discrimination | Baseline (all configs inherit) | 115 zero-hit, 155 single-hit queries |
| 1 | Catastrophic rank inversion from aggressive attention pooling | `attn_01` (keep_ratio=0.1) | 45 catastrophic, 198 moderate |
| 2 | Score discrimination collapse (score spread flattening) | `attn_01`, `rand_01` | 70–114 queries |
| 3 | Attention-specific distributional drift: generic tokens promoted | `attn_01` | cross-cuts with #1 |
| 4 | Dataset-side label pool mismatch for broad/navigational queries | All configs | 20+ high-recall-miss queries |
| 5 | Lexical surface trap — short/ambiguous queries exploit compression-induced sensitivity | `attn_01`, `rand_01` | dozens |
| 6 | Spherical pooling centroid drift at high compression factors | `sph_5`, `sph_10` | 39 regressions (sph_10 vs baseline), 64 catastrophic rank inversions |

---

## Mode 0 — Baseline Structural Failures

### Context: the baseline is itself a weak retriever on this dataset

Before analyzing what compression breaks, it is important to establish that the baseline (`GTE-ModernColBERT-v1`, uncompressed, 271 avg tokens/doc) is already struggling on the Amazon dataset:

| Metric | Baseline value |
|--------|---------------|
| NDCG@10 | 0.4544 |
| Precision@10 | 0.1202 |
| Recall@10 | 0.5330 |
| MAP | 0.4126 |

Precision@10 = 0.1202 means the average query gets **1.2 relevant documents** in its top 10 — barely above the floor. Broken down by query:

| Hits@10 | # Queries (of 408 with positives) | % |
|---------|-----------------------------------|---|
| 0 | 115 | 28.2% |
| 1 | 155 | 38.0% |
| 2 | 78 | 19.1% |
| 3 | 31 | 7.6% |
| 4+ | 29 | 7.1% |

**66.2% of queries achieve at most 1 hit in the top 10.** This severely limits what any compression analysis can conclude: a large fraction of queries are failures before compression is applied.

### Three structural failure sub-modes in the baseline

#### B1 — Label pool mismatch (shared with Mode 4)

Of the 115 queries with 0 hits@10, **68 fail consistently across all 16 configs**. These are irreparable pool-mismatch failures: the judged positives were never retrieved by any system, so they are invisible to the benchmark. Examples:

- `q_309` (`"horror"`) — 8 judged positives all outside any config's top 100
- `q_130` (`"tv shows"`) — baseline retrieves `The Eichmann Show`, `Mickey Mouse Clubhouse+`: plausible but unjudged
- `q_347` (`"what new movies are out"`) — baseline retrieves `Baby Driver`, `Ghost in the Shell: The New Movie`
- `q_68` (`"golden globes comedy films from twenty twenty"`) — baseline retrieves several plausible titles; none judged

The remaining **47 zero-hit queries are baseline-specific failures** (at least one other config succeeds). Examples:

| Query | Baseline top-3 | Judged positive(s) | Why baseline misses |
|-------|---------------|-------------------|--------------------|
| `q_101` `"scary halloween movies"` | `Tales of Halloween`, `Ghost Stories`, `Hell Fest` | `Annabelle: Creation`, `The Curse of La Llorona` | Genre centroids correct, specific titles wrong |
| `q_102` `"sci-fi movies"` | `Star Runners`, `Alien: Romulus`, `Pinocchio 3000` | `M3GAN 2.0`, `The Fantastic Four: First Steps` | Label sparsity — 2 positives out of ~200 sci-fi corpus items |
| `q_136` `"hindi movies"` | `Krrish 3`, `Khiladi 786`, `I, Me Aur Main` | `Phir Bhi Dil Hai Hindustani`, `Nayee Padosan` | Model retrieves right domain, wrong specific titles |
| `q_163` `"nosferatu"` | `She Creature`, `The Night House`, `Night Kaleidoscope` | `The Vulture's Eye`, `Embrace of the Vampire` | Query maps to vampire genre, not to the specific film |
| `q_194` `"Suggest me something chilling to watch"` | `Hitler: The Rise of Evil`, `The Autopsy of Jane Doe` | `The Babadook`, `Get Out`, `It Follows` | Subjective query; model captures "creepy" but not canonical horror |

#### B2 — Low score discrimination (score spread collapse at baseline)

For **93 / 456 queries (20.4%)**, the top-10 score spread (max − min within top 10) is below 0.05. When all top-10 documents score within a 0.05 band, any slight perturbation — including random token-level noise — can change the ranking completely.

| Score spread range | # Queries | % |
|--------------------|-----------|---|
| < 0.05 (critically low) | 93 | 20.4% |
| 0.05 – 0.10 (low) | 129 | 28.3% |
| > 0.10 (adequate) | 234 | 51.3% |

Sample critically-low-spread queries:

- `q_102` (`"sci-fi movies"`): spread = 0.026, top-3 scores differ by < 0.02 — the ranking is essentially a tie
- `q_118` (`"recent blockbusters from this year"`): spread = 0.041, retrieves `Captain America: Civil War`, `Fantastic Beasts` — all within noise
- `q_120` (`"what shows came out today"`): spread = 0.024, temporal specificity cannot be captured via token similarity

This affects **all** downstream configs: any compression that changes even a single token's representation can reorder a near-tie query. Compression amplifies noise when the baseline margin is already small.

#### B3 — Semantic intent mismatch (model-side)

For intentional, specific queries, the model retrieves plausible-but-wrong documents because its MaxSim score conflates surface-level domain overlap with actual relevance:

- `q_147` (`"Recommend me something epic to watch"`): baseline retrieves `Dune: Part Two` (rank 1), `Epic Movie` (rank 2), `Bajirao Mastani` (rank 3) — the judged positive is `Avengers: Endgame`, which appears at rank ~15 in the baseline. The model matches `epic` as a surface term (`Epic Movie`) and high-scale cinematics but doesn't distinguish the specific labeled intent.
- `q_151` (`"crab movies"`): baseline correctly retrieves `Cheech & Chong's Animated Movie` as a hit, but the positive `Love and Monsters` (a creature-feature with crabs) is missed because its representation doesn't cluster near `"crab"`-adjacent tokens.
- `q_154` (`"emma stone horror films"`): positive is `The Favourite`; baseline retrieves it at rank 1, but only because `The Favourite` happens to be cosine-similar to the horror query — not because the model understands the actor-based intent.

These intent-mismatch failures are pre-compression artifacts of the training distribution of `GTE-ModernColBERT-v1`, not fixable by the compression strategies evaluated here.

### Implication for interpreting compression results

Given that ~28% of evaluated queries are already zero-hit at baseline (mostly pool-mismatch), and another ~38% are single-hit (precision ceiling of 0.1 per query), the effective ceiling for any single config's precision@10 on this dataset is low. **NDCG drops due to compression partially reflect regression against an already-noisy baseline**, not a clean signal of compression harm on high-quality retrieval.

---

## Mode 1 — Catastrophic Rank Inversions Under Aggressive Attention Pooling

### What it looks like

A document that ranks top-1 or top-3 under the baseline plummets to rank 50–100 in `attn_01`.

Concrete examples:

| Query | Baseline rank | `attn_01` rank | Drop |
|-------|--------------|---------------|------|
| `q_365` (sci-fi/animated) | `Pinocchio 3000` → rank 1 | rank 100 | −99 |
| `q_86` (dramatic romance) | `Bajirao Mastani` → rank 1 | rank 99 | −98 |
| `q_113` (drama) | `In Her Shoes` → rank 1 | rank 91 | −90 |
| `q_5` (fantasy) | `Harry Potter and the Deathly Hallows: Part 1` → rank 3 | rank 91 | −88 |
| `q_106` (romance/award) | `La La Land` → rank 1 | rank 72 | −71 |

In total: **45 queries** where a top-3 baseline document falls below rank 50, **198 queries** with moderate inversions (rank 11–50).

For reference, `sph_133` (spherical pooling at f=1.333 — comparable token count to `attn_075`) nearly always preserves these documents inside the top 10.

### Why it happens

`AttentionPoolingStrategy` selects anchor tokens by taking the **top-k by attention score**, then clusters remaining tokens to their most similar anchor. At `keep_ratio=0.1` on a 271-token-average corpus, only ~27 anchor tokens survive per document.

The key insight: for movie-title documents, **title-specific tokens often do not have the highest attention scores**. Attention scores in transformer models tend to peak on common structural tokens, punctuation boundaries, or domain-generic terms (e.g. "Movie", "Film", "the", ":") rather than semantically unique title terms (e.g. "Pinocchio", "Bajirao", "Mastani").

When `keep_ratio=0.1` is applied:
- Title-specific tokens are merged into generic clusters.
- The pooled representation drifts toward a generic "documentary / animated / action" centroid.
- The MaxSim matching between a query about "animated movies" and a document whose "Pinocchio" token is gone collapses.

Compare:
- `q_365` baseline top-1: `Pinocchio 3000` (6.359) — then `attn_01` top-1: `After the Apocalypse` (6.198)
- `q_86` baseline top-1: `Bajirao Mastani` (6.200) — then `attn_01` top-1: `The Heart of Christmas` (6.116)

The drift in `q_86` is particularly illustrative: `Bajirao Mastani` is a specific Bollywood epic; under 10% attention pooling, its representation merges into a generic romantic cluster and scores lower than vanilla Christmas romance films.

### Code location

[pylate/models/compression/attention_pooling.py](../../../../../pylate/models/compression/attention_pooling.py)
— specifically [lines 186–193](../../../../../pylate/models/compression/attention_pooling.py#L186-L193):

```python
# Select anchors: tokens with highest attention scores
_, anchor_local_idx = torch.topk(
    nonprot_attention, k=min(num_clusters, n_non_protected), largest=True, sorted=False
)
# Sort anchors by position to maintain document order
anchor_local_idx, _ = torch.sort(anchor_local_idx)
anchor_emb = nonprot_emb[anchor_local_idx]  # [C, dim]
```

The anchor selection is purely by top-k attention score. There is no diversity enforcement or content-uniqueness bias.

### Proposed fix

**Option A: Farthest-point anchor seeding (greedy k-center)** — seed with the highest-attention token,
then each subsequent anchor is the token with maximum cosine distance to the *nearest existing anchor*
(farthest-point sampling, O(nk) per document). This directly prevents all anchors from collapsing into
the same semantic region.

Why this helps: if all k anchors are selected purely by attention score and they all land in the same
semantic subspace (e.g., all are genre/structural tokens — "film", "action", "stars"), then every
non-anchor token assigns to the nearest anchor within that subspace. Title-specific tokens (like
"Pinocchio", "Bajirao") end up averaged into whichever genre cluster happens to be geometrically
closest, losing their distinguishing direction entirely. With farthest-point selection, one anchor
"owns" the title/entity region of the embedding space, so title tokens cluster there and their
direction survives pooling.

**Option B (cheap mitigation): Increase `protected_tokens`** — currently set to 1 (CLS only). If the
first N tokens of the document often contain the title/key entity, protecting 4–8 tokens would help.
In practice with ColBERT-style documents, the title is tokenized first.

**Option C: Attention × novelty composite** — weight = `attn_score * (1 - max_cosine_to_existing_anchors)`,
recomputed iteratively. This keeps the attention signal as a prior but penalizes redundant anchor
selection. Equivalent to weighted farthest-point sampling.

---

## Mode 2 — Score Discrimination Collapse

### What it looks like

Under `attn_01`, the top-10 score spread (max−min in top-10) is systematically compressed:

| Config | Mean top-10 score spread | Queries with spread < 50% of baseline |
|--------|------------------------|--------------------------------------|
| Baseline | 0.1345 | — |
| `attn_01` | 0.0974 | **114 / 456** |
| `sph_133` | 0.1330 | ~20 |
| `rand_01` | 0.1121 | ~55 |

When the score spread collapses from 0.13 to 0.04, the scoring function effectively cannot separate relevant from non-relevant. Any small distributional bias — even generic domain drift — can flip the rank.

Concrete example with **q_129** (`"catfish"`):
- Baseline spread: 0.186, `Catfish in Black Bean Sauce` scores 4.428, generic cat movies score ~4.26
- `attn_01` spread: 0.059, `Catfish in Black Bean Sauce` is no longer in top 10 at all; `Cool Cat Saves the Kids` leads at 4.211

Another example: **q_126** has spread collapse from 0.324 → 0.050 (15.5% of baseline), causing the ranking to become essentially random within the genre.

### Why it happens

The drops are **not uniform** — they are differential across documents, and it is this differential that
causes both rank inversions and the resulting spread collapse.

Measured for `q_56` ("fast paced action movies"): per-document score drop from baseline to `attn_01`
ranges from 0.090 (`The Accountant`) to 0.258 (`Fast Track: No Limits`) — a 3× variance in damage
magnitude within a single query. Globally, the mean within-query standard deviation of score drops is
0.044, and 144 / 456 queries have within-query drop stdev > 0.05 (highly non-uniform). Only 4 queries
show near-uniform drops.

The mechanism:
1. Documents whose discriminative tokens happen to *not* be selected as anchors suffer large drops
   (those tokens get averaged into a generic cluster, losing their specific directional signal).
2. Documents whose tokens happen to survive as anchors — typically generic/common tokens — suffer
   smaller drops.
3. After pooling, the new top-10 is populated by the "least-damaged" documents — not necessarily the
   most relevant ones. These are often the generic genre representatives. Because they are structurally
   similar to each other, their pooled MaxSim scores cluster tightly → spread collapses.

The spread collapse is therefore a *consequence* of differential rank inversions, not an independent
cause. If all scores dropped by the same amount, rank order would be preserved and spread would be
unchanged.

`rand_01` shows partial spread collapse (0.1121 vs 0.1345 baseline) for the same reason but less
severely: random pooling has no systematic bias, so the damage is more evenly distributed and rank
inversions are less correlated with relevance.

### Code location

The scoring is done via PLAID MaxSim retrieval. The compression that causes this is in [pylate/models/compression/attention_pooling.py:204–210](../../../../../pylate/models/compression/attention_pooling.py#L204-L210):

```python
for cid in range(num_clusters):
    mask = cluster_labels_tensor == cid
    if not mask.any():
        continue
    members = nonprot_emb[mask]  # [num_members, dim]
    pooled_vec = members.mean(dim=0)  # <-- plain averaging
    pooled_cluster_vectors.append(pooled_vec)
```

Plain mean averaging of token embeddings reduces L2 norm and pushes the resulting vector toward the centroid of the cluster. After the final `F.normalize` in [compression_eval.py:766](../../../../../experiments/compression/compression_eval.py#L766), this centroid becomes a unit vector that is less directionally extreme than any of the originals, meaning lower MaxSim scores.

### Proposed fix

**Use anchor embedding as the cluster representative** instead of the mean, at least as an experiment. The anchor token has the highest attention score (most "important") and preserves the direction of the most salient embedding in the cluster. This is already done for the token artifact but not for the embedding itself.

Alternatively, **weighted mean by attention score** within each cluster would bias toward the most salient token's direction.

---

## Mode 3 — Attention-Guided Distributional Drift

### What it looks like

For semantically nuanced queries (Oscar-winning dramas, culturally specific films, genre sub-categories), `attn_01` systematically promotes generic genre representatives over specific high-relevance films.

Examples:

**q_106** (romance/prestige cinema):
- Baseline top-3: `La La Land`, `A Star Is Born`, `Phantom Thread` — all Oscar-level prestige films
- `attn_01` top-3: `Weekend Getaway`, `Frozen in Love`, `Love, Simon` — generic romance/holiday films
- `La La Land` drops from rank 1 → rank 72

**q_294** (`"coming of age movies"`):
- Baseline: `The Spectacular Now`, `The Miseducation of Cameron Post`, `Love, Simon`
- `attn_01`: `Grow Up, Tony Phillips`, `Confessions of a Teenage Drama Queen`, `Love, Simon`
- The specific indie coming-of-age films drop; more generic coming-of-age films are promoted

**q_167** (documentary):
- Baseline: `Turtle Power: The Definitive History of the Teenage Mutant Ninja Turtles`, `Our Planet: Behind the Scenes`, `History of the Eagles`
- `attn_01`: `O.J.: Made in America`, `Chasing Ice`, `Walking with Dinosaurs`
- The specific titles dropped; generic high-production-value docs took over

### Why it happens

Attention scores in this retrieval model reflect which tokens are important for **contextualizing the
document's own token representations** during encoding — i.e., which tokens provide useful context to
their neighbors. For MaxSim retrieval, what matters is different: which token embeddings achieve high
cosine similarity with specific query vectors.

These two objectives diverge for named-entity-heavy corpora. Genre and structural tokens ("musical",
"drama", "directed by", "starring") provide heavy context and are likely high-attention. Specific named
entities (title words, director names, franchise markers) are mostly "looked up" by other tokens —
they *receive* context rather than providing it, so they tend to be lower-attention. But for MaxSim,
named-entity tokens are exactly what matches a specific query like "prestige film awards drama".

At `keep_ratio=0.1`, only the high-attention (structural/generic) tokens become anchors. Named-entity
tokens get pooled into whichever generic anchor is nearest → representation becomes a genre centroid →
`La La Land` is indistinguishable from `Frozen in Love`.

Spherical pooling (`sph_133`) avoids this because k-means clustering is **geometry-based**: it groups
tokens by their embedding-space proximity, which respects semantic distinctiveness. Named-entity tokens
that are far from genre tokens in embedding space naturally end up in their own cluster.

### Code location

The root is the anchor selection policy in [attention_pooling.py:186–192](../../../../../pylate/models/compression/attention_pooling.py#L186-L192).

### Proposed fix

**Farthest-point anchor selection with attention prior**: seed with the highest-attention token, then
greedily add the token with maximum cosine distance to the nearest existing anchor until k anchors are
selected. This is the same fix as Mode 1 Option A — Modes 1 and 3 share the same root cause (bad
anchor selection) and the same fix applies to both.

**Attention × novelty composite**: weight = `attn_score * (1 - max_cosine_to_existing_anchors)`,
recomputed iteratively. This incorporates the attention signal while penalizing redundant selection.
Equivalent to weighted farthest-point sampling and easier to tune than pure geometric coverage.

---

## Mode 4 — Label Pool Mismatch for Broad/Navigational Queries

### What it looks like

Queries that express a broad intent (genre, platform, recency) retrieve plausible and arguably relevant results, but the qrels only contain a narrow set of judged positives. Zero precision is reported even when results look reasonable.

Concrete examples from the appendix list (from the `PRECISION_AT_10_FAILURE_ANALYSIS.md`):

| Query | Judged positives | What baseline retrieves | Hits@10 (all configs) |
|-------|-----------------|------------------------|----------------------|
| `q_309` (`"horror"`) | 8 | `Zone of the Dead`, `Alien: Romulus`, `It Follows` — plausible horror | 0 |
| `q_130` (`"tv shows"`) | 3 | `The Eichmann Show`, `Mickey Mouse Clubhouse+`, `Alien: Earth` | 0 (all) |
| `q_298` (`"kiss"`) | >0 | `The Florida Project`, `The Seventh Dwarf`, `Kingsman` | 0 |
| `q_347` (`"what new movies are out"`) | 3 | `Baby Driver`, `Ghost in the Shell: The New Movie`, ... | 0 |
| `q_193` (`"recent movie releases from 2024"`) | 2 | `We Live in Time`, `Alien: Romulus`, `The Strangers: Chapter 1` | 0 |

For `q_130`, *all 16 configs* return 0 hits. The judged positives for "tv shows" are specific show titles that none of the configs retrieve, despite the corpus containing many valid TV shows.

This mode is **not** a compression failure: it is present in the baseline and all compression configs equally.

### Why it happens

The dataset was built from an Amazon assistant log. Broad navigational queries ("horror", "tv shows") were given a small set of manually judged positives, but the corpus contains many legitimate alternatives that were never judged. This is the **pool bias** problem in IR evaluation: unjudged documents are scored as non-relevant.

Dataset construction artifacts (see the existing `PRECISION_AT_10_FAILURE_ANALYSIS.md`, section "Data Pipeline Findings"):

1. **Narrow positive labeling** — broad queries may have had only the top result(s) from one system judged.
2. **No re-judging pool** — results from compressed models were not added to the judged pool, so anything they retrieve that wasn't in the baseline's top results is scored as non-relevant.

### Code location

The qrels loading is in [compression_eval.py:229](../../../../../experiments/compression/compression_eval.py#L229):

```python
documents, queries, qrels = pylate_evaluation.load_custom_dataset(dataset_id, split="test")
```

Then evaluation uses `make_comparable=True` in [compression_eval.py:929–934](../../../../../experiments/compression/compression_eval.py#L929-L934):

```python
evaluation_scores = ranx_evaluate(
    qrels=qrels_obj,
    run=run,
    metrics=list(cfg.metrics),
    make_comparable=True,
)
```

`make_comparable=True` fills missing query scores with 0 — including queries whose judged positives are all zero-relevance, and queries where no retrieved doc happens to be judged positive. This is mathematically correct but inflates the apparent failure rate for pool-mismatch queries.

The dataset builder (`convert_amazon_to_beir_full_content.py`, referenced in the previous analysis) does not filter zero-relevance qrel rows, so those queries still appear in the evaluation denominator.

### Proposed fix

**Pooled evaluation across compression runs**: before final evaluation, collect unique retrieved documents from all compression configs and pass them to a re-judging or pseudo-label expansion step. At minimum, log the number of queries where no retrieved document appears in the qrels at all — these are the true pool-mismatch cases vs. genuine misses.

**In the conversion script**: add a filter step that excludes queries from qrels where all `query_relevance_score` values are 0, and log them separately. This avoids inflating the denominator in macro-averaged precision.

---

## Mode 5 — Short/Ambiguous Query Lexical Trap Under Compression

### What it looks like

Queries of 1–3 words rely heavily on specific lexical tokens for disambiguation. Compression changes which document tokens survive, which can flip the result from lexically correct to lexically ambiguous.

Examples:

**q_129** (`"catfish"`):
- Baseline: `Catfish in Black Bean Sauce` (4.428), `The Amazing Catfish` (4.409), `Sweet Bobby: My Catfish Nightmare` (4.342) — all catfish-related titles
- `attn_01`: `Cool Cat Saves the Kids` (4.211) — the "Cat" surface token survives but "fish" may not, so generic cat movies surface

**q_298** (`"kiss"`):
- Baseline top-1: `The Florida Project` (4.361)
- `attn_01` top-1: `Kingsman: The Golden Circle` (4.305) — "circle/golden/king" tokens dominate after pooling

**q_251** (`"notting hill"`):
- Baseline top-1: `Silent Hill` — the token "hill" matches; judged positive is `Love Actually`
- `attn_01` top-1: different wrong film

This is partly a baseline failure (the model doesn't understand query intent for single-word queries) but compression makes it worse by changing the surviving token distribution in documents.

### Why it happens

Two sub-mechanisms:

1. **Title-token survival under compression**: for a 300-token document, title tokens are a small fraction. Under 10% attention pooling, the specific discriminative token (e.g. "Catfish") may be pooled into a broader "animal" or "food" cluster, while generic high-frequency tokens ("movie", "film", "story") survive as anchors.

2. **Query-document mismatch amplification**: with short queries (1–3 tokens), the query embedding has few vectors and is more sensitive to small changes in top-1 matched document token. When the specific matching token is gone from the document, the MaxSim score falls back to the next-best match — often a generic-domain token that also appears in unrelated docs.

### Code location

This mode is not directly fixable in the compression code; it is an interaction between the compression strategy and the evaluation regime. However, it surfaces specifically because of the title-token suppression issue described in Mode 1.

Short query handling in the evaluation: `resolve_query_length` in
[compression_eval.py:164–168](../../../../../experiments/compression/compression_eval.py#L164-L168)
defaults to `query_len=32` for the local Amazon dataset, which is appropriate. The issue is not truncation.

The `QUERY_LEN` mapping at
[compression_eval.py:66–118](../../../../../experiments/compression/compression_eval.py#L66-L118)
does not have an entry for `"../amazon_dataset/beir_format_full"`, so it falls back to the default 32. For a dataset with mean query length of 4.58 words, this is more than sufficient — the failure is not query truncation.

### Proposed fix

**Lexical anchor protection**: before pooling, identify tokens that exactly match any term in the query vocabulary (or high-IDF vocabulary) and protect them from pooling. This requires passing query information to the document encoder, which violates the separation of concerns in the current encode-then-compress pipeline — but could be approximated at index time using a stored vocabulary.

**More practical**: use the IDF-based pooling strategy (`IDFPoolingStrategy` in [pylate/models/compression/idf_pooling.py](../../../../../pylate/models/compression/idf_pooling.py)) which explicitly protects high-IDF (rare/discriminative) tokens. The current run only tests IDF pruning for other datasets; adding an IDF pooling config to the Amazon evaluation grid would directly address this mode.

---

## Mode 6 — Spherical Pooling Centroid Drift at High Compression Factors

### Overview

Spherical pooling is substantially more robust than attention pooling at equivalent token budgets. However, at high compression factors (f ≥ 5, corresponding to ≤ 20% of the original tokens), it begins to exhibit its own characteristic failure mode: **semantic centroid drift**, where clusters become too coarse to preserve the directional specificity of rare/edge-case tokens.

### Metrics by compression factor

| Config | Avg tokens/doc | NDCG@10 | P@10 | Recall@10 | MAP | Queries 0-hit |
|--------|---------------|---------|------|-----------|-----|---------------|
| Baseline | 271.4 | 0.4544 | 0.1202 | 0.5330 | 0.4126 | 115 |
| `sph_133` (f=1.333) | 203.4 | 0.4534 | 0.1208 | 0.5350 | 0.4089 | 109 |
| `sph_2` (f=2) | 135.9 | 0.4415 | 0.1206 | 0.5378 | 0.3941 | 108 |
| `sph_3` (f=3) | 90.8 | 0.4321 | 0.1156 | 0.5089 | 0.3917 | 120 |
| `sph_5` (f=5) | 54.7 | 0.4233 | 0.1147 | 0.4970 | 0.3824 | 122 |
| `sph_10` (f=10) | 27.6 | 0.3901 | 0.1055 | 0.4690 | 0.3485 | 136 |

Comparison at the same token budget (27.6 tokens): `sph_10` achieves NDCG@10 = 0.3901 vs `attn_01` = 0.3303 — a **+18.4% relative improvement** despite identical token counts. This confirms the clustering criterion matters more than the token budget.

### Rank inversion statistics

| Config | Catastrophic inversions (top-3 → rank > 50) | Moderate inversions (rank 11–50) |
|--------|---------------------------------------------|----------------------------------|
| `sph_3` (f=3) | 19 | 102 |
| `sph_5` (f=5) | 23 | 165 |
| `sph_10` (f=10) | 64 | 226 |

`sph_10` produces 64 catastrophic inversions, which is notably fewer than `attn_01`'s 45 at document-level but comparable when normalized — `sph_10` still preserves the top-3 structure much better than `attn_01`. The 226 moderate inversions indicate significant rank degradation even on queries that are not catastrophic.

**39 queries** where the baseline achieves ≥1 hit@10 but `sph_10` achieves 0 hits represent pure compression regressions (not baseline failures). Selected examples:

| Query | Baseline top hit | `sph_10` top-3 | Baseline hits / sph_10 hits |
|-------|-----------------|----------------|-----------------------------|
| `q_154` `"emma stone horror films"` | `The Favourite` (7.142) | `The Dark and the Wicked`, `The Strangers: Chap. 1`, `Silent Hill` | 1 / 0 |
| `q_147` `"Recommend me something epic to watch"` | `Dune: Part Two` (9.104, rank 1) | `Epic Movie`, `Bajirao Mastani`, `Dune: Part Two` | 1 / 0 |
| `q_150` `"find family friendly halloween movies"` | `Halloweentown II` (rank 2) | `Spooky Buddies`, `Tales of Halloween`, `Scary Godmother` | 1 / 0 |
| `q_151` `"crab movies"` | `Cheech & Chong's Animated Movie` (rank 3) | `SpongeBob SquarePants Movie`, `The Croods: A New Age`, `Legend of Ochi` | 1 / 0 |
| `q_132` `"Playful fantasy adventure films"` | `The Legend of Ochi` (rank 1) | `Tinker Bell and the Great Fairy Rescue`, `Kubo and the Two Strings`, `Legend of Ochi` | 1 / 0 |

### Why it happens — spherical-specific centroid drift

At f=10, a 271-token-average document is compressed to ~27 tokens. Each of the ~27 k-means clusters aggregates roughly 10 original token embeddings into a **mean centroid**. The same root pathology as attention pooling applies, but is driven by geometric grouping rather than attention-score selection:

1. **Large clusters absorb rare tokens**: k-means groups tokens by embedding-space proximity. At f=10, the cluster containing a genre-dominant region (e.g., "halloween / horror / spooky") is geometrically large and will absorb neighboring concept tokens, including edge-case specifics (e.g., "Halloweentown", "Kalabar").

2. **Mean aggregation shifts the centroid toward the cluster interior**: even with good clustering, the mean of 10 embeddings within a semantic region drifts toward the centroid of that region. The most directionally extreme token (usually the most semantically specific) loses its uniqueness. For `The Favourite` under `q_154`, the specific token embedding for `"Favourite"` gets averaged with surrounding context tokens, pulling the representative vector toward a generic "prestige drama" centroid.

3. **Score spread is better preserved than in attention pooling**: the mean spread across top-10 for `sph_10` is **0.1216** vs **0.0974** for `attn_01`. This means spherical pooling degrades rank order less than attention pooling because the damage is more uniform — no systematic bias toward structural tokens. But absolute score levels still drop, and marginal cases get reordered.

The `q_147` case (`"epic"`→`Avengers: Endgame`) is particularly illustrative:
- Baseline: `Dune: Part Two` at rank 1 (score 9.104), `Epic Movie` rank 2 (9.099). The positive `Avengers: Endgame` is around rank 15–20 with a score just below the `Dune`/`Epic Movie` cluster — baseline misses it too.
- `sph_10`: the score of all three top-3 candidates drops by ~0.08–0.09 uniformly; `Avengers: Endgame` also drops proportionally; its relative rank within the near-tie cluster may shift slightly, causing it to fall further below rank 10.
- This borderline case is fundamentally a baseline near-miss (p@10 = 0.1, barely), and `sph_10` tips it to 0 because the inter-document score differences are too small relative to the compression-induced score perturbation.

### Code location

[pooling.py:470–477](../../../../../pylate/models/compression/pooling.py#L470-L477):

```python
for cluster_id in range(num_clusters):
    cluster_indices = torch.where(cluster_labels_tensor == cluster_id)[0]
    if cluster_indices.numel() > 0:
        # Average the embeddings in this cluster
        cluster_embedding = embeddings_to_pool[cluster_indices].mean(dim=0)
        pooled_document_embeddings.append(cluster_embedding)
```

The same plain `mean(dim=0)` aggregation as in attention pooling ([attention_pooling.py:204–210](../../../../../pylate/models/compression/attention_pooling.py#L204-L210)). The clustering strategy is superior, but the aggregation step shares the same directional-norm-reduction pathology.

### Proposed fixes

**Option A: Use nearest-to-centroid representative instead of mean** — within each spherical cluster, select the token whose embedding is *closest to the cluster centroid* as the representative (rather than computing the mean). This preserves a real token direction and avoids the norm reduction from averaging. The centroid is still used for cluster assignment; only the output representation changes.

**Option B: Weighted mean biased toward cluster periphery** — weight each token by its distance from the cluster centroid (`1 - cosine(tok, centroid)`). This down-weights the "generic center" of the cluster and amplifies the contribution of directionally extreme tokens — exactly the tokens that matter for MaxSim matching on specific queries.

**Option C: Reduce f; use sph_3 or sph_5 as the practical limit** — NDCG@10 drops only from 0.4544 → 0.4321 at f=3 (91 tokens) and → 0.4233 at f=5 (55 tokens), while `sph_10` drops more steeply to 0.3901. The inflection is between f=5 and f=10. For most practical use cases, f=5 (80% token reduction) with spherical pooling is a better operating point than f=10.

---

## Cross-Cutting Comparison: Spherical vs Attention vs Random Pooling

`sph_133` (spherical/k-means pooling at f=1.333) keeps ~75% of baseline tokens and achieves nearly identical metrics to the baseline. Even `sph_2` (50% tokens) outperforms `attn_01` (10% tokens) on every metric. `sph_10` at the same token count as `attn_01` (27.6 tokens) beats attention pooling by +18.4% relative NDCG@10 (0.3901 vs 0.3303).

The key difference is the **clustering criterion**:

| Strategy | Anchor/Cluster Selection | Aggregation | Failure Risk |
|----------|------------------------|-------------|-------------|
| `attn_01` | Top-k by attention score | Mean within cluster | Selects structural/generic tokens as anchors; drops named entities; severe score collapse |
| `sph_133` | Spherical k-means (geometry) | Mean within cluster | Content-based → preserves semantically distinct directions; slight norm reduction |
| `sph_10` | Spherical k-means (geometry) | Mean within cluster | Same mechanism as `sph_133` but coarser clusters → centroid drift for edge cases |
| `rand_01` | Random (uniform) | Mean within cluster | Unbiased but lossy with no signal guiding selection |

For this dataset (movie titles with many proper-noun discriminators), **attention score is a poor proxy
for retrieval importance**. Attention captures which tokens contextualize their neighbors well —
structural/generic tokens do this. MaxSim cares about which token embeddings will match specific query
vectors — named entities and discriminative terms do this. Spherical pooling's geometry-based
grouping is naturally aligned with the latter objective — but at extreme compression factors (f ≥ 10),
even geometry-based clustering becomes too coarse to protect edge-case token directions.

Score spread comparison:

| Config | Mean top-10 spread | Queries with spread < 0.05 |
|--------|--------------------|----------------------------|
| Baseline | 0.1345 | 93 (20.4%) |
| `sph_133` | 0.1330 | 95 |
| `sph_2` | 0.1335 | 89 |
| `sph_3` | 0.1341 | 86 |
| `sph_5` | 0.1270 | 97 |
| `sph_10` | 0.1216 | 96 |
| `rand_01` | 0.1121 | ~55 |
| `attn_01` | 0.0974 | 114+ |

Spherical pooling preserves score discrimination far better than attention pooling in the regime f ≤ 5. Even at f=10, the mean spread of 0.1216 is much closer to baseline (0.1345) than `attn_01` (0.0974). The spread degradation is gradual for spherical pooling but catastrophic for attention pooling.

The relevant clustering code:
- Spherical k-means pooling: [pooling.py:296–485](../../../../../pylate/models/compression/pooling.py#L296-L485)
- Attention pooling: [attention_pooling.py:58–240](../../../../../pylate/models/compression/attention_pooling.py#L58-L240)

---

## Summary Table of Proposed Fixes

| Mode | Fix | Location | Complexity |
|------|-----|----------|-----------|
| 0 (baseline pool mismatch) | Pool evaluation across all configs before scoring; filter zero-relevance qrels rows | Dataset converter + [compression_eval.py:229](../../../../../experiments/compression/compression_eval.py#L229) | Low |
| 0 (baseline intent mismatch) | Model-level: fine-tune on domain-specific data; out of scope for compression analysis | — | High |
| 1 + 3 (root cause: bad anchor selection) | Farthest-point anchor seeding — seed with top-attention, then greedily select by max distance to nearest existing anchor | [attention_pooling.py:186–192](../../../../../pylate/models/compression/attention_pooling.py#L186-L192) | Medium |
| 1 + 3 (alternative) | Attention × novelty composite weight for anchor selection | [attention_pooling.py:186–192](../../../../../pylate/models/compression/attention_pooling.py#L186-L192) | Medium |
| 1 (cheap mitigation) | Increase `protected_tokens` from 1 to 4–8 for title-heavy corpora | Config / [attention_pooling.py:163](../../../../../pylate/models/compression/attention_pooling.py#L163) | Low |
| 2 (score collapse — downstream of 1+3) | Use anchor embedding as cluster representative instead of mean | [attention_pooling.py:204–210](../../../../../pylate/models/compression/attention_pooling.py#L204-L210) | Low |
| 2 (alternative) | Attention-weighted mean within cluster instead of plain mean | [attention_pooling.py:204–210](../../../../../pylate/models/compression/attention_pooling.py#L204-L210) | Low |
| 4 (label pool mismatch) | Filter zero-relevance queries from evaluation denominator | Dataset converter + [compression_eval.py:229](../../../../../experiments/compression/compression_eval.py#L229) | Low |
| 4 (alternative) | Log pool-mismatch count per config: queries where 0 retrieved docs appear in qrels | [compression_eval.py:920–934](../../../../../experiments/compression/compression_eval.py#L920-L934) | Low |
| 5 (lexical trap) | Add IDF pooling config to evaluation grid | [compression_eval.py:592–698](../../../../../experiments/compression/compression_eval.py#L592-L698) + [idf_pooling.py](../../../../../pylate/models/compression/idf_pooling.py) | Low |
| 5 (structural) | High-IDF token protection in anchor selection | [attention_pooling.py](../../../../../pylate/models/compression/attention_pooling.py) | High |
| 6 (sph centroid drift) | Use nearest-to-centroid token as cluster representative instead of mean | [pooling.py:470–477](../../../../../pylate/models/compression/pooling.py#L470-L477) | Low |
| 6 (alternative) | Periphery-biased weighted mean: weight = `1 - cosine(tok, centroid)` | [pooling.py:470–477](../../../../../pylate/models/compression/pooling.py#L470-L477) | Low |
| 6 (operating point) | Cap spherical compression at f=5 (55 tokens) to stay in the gradual-degradation regime | Config / compression grid | Low |

---

## Appendix: Quantitative Summary

### Global Metrics Table

| Config | Avg tokens | NDCG@10 | P@10 | Recall@10 | MAP |
|--------|-----------|---------|------|-----------|-----|
| Baseline | 271.4 | 0.4544 | 0.1202 | 0.5330 | 0.4126 |
| `attn_075` | 203.4 | 0.4551 | 0.1200 | 0.5332 | 0.4107 |
| `sph_133` | 203.4 | 0.4534 | 0.1208 | 0.5350 | 0.4089 |
| `rand_075` | 203.4 | 0.4510 | 0.1206 | 0.5292 | 0.4072 |
| `attn_05` | 135.9 | 0.4403 | 0.1173 | 0.5292 | 0.3960 |
| `sph_2` | 135.9 | 0.4415 | 0.1206 | 0.5378 | 0.3941 |
| `rand_05` | 135.9 | 0.4419 | 0.1178 | 0.5215 | 0.3997 |
| `attn_033` | 89.8 | 0.4249 | 0.1136 | 0.5162 | 0.3789 |
| `sph_3` | 90.8 | 0.4321 | 0.1156 | 0.5089 | 0.3917 |
| `rand_033` | 89.8 | 0.4352 | 0.1164 | 0.5126 | 0.3929 |
| `attn_02` | 54.7 | 0.3878 | 0.1079 | 0.4740 | 0.3434 |
| `sph_5` | 54.7 | 0.4233 | 0.1147 | 0.4970 | 0.3824 |
| `rand_02` | 54.7 | 0.4075 | 0.1083 | 0.4741 | 0.3698 |
| `attn_01` | 27.6 | 0.3303 | 0.0934 | 0.4062 | 0.2918 |
| `sph_10` | 27.6 | 0.3901 | 0.1055 | 0.4690 | 0.3485 |
| `rand_01` | 27.6 | 0.3716 | 0.0985 | 0.4447 | 0.3291 |

**At the same token budget, spherical pooling consistently outperforms both attention and random pooling.** The relative advantage of `sph` over `attn` grows with compression ratio.

### Rank Inversion Statistics (baseline top-3 vs. `attn_01`)

- Queries with ≥1 top-3 doc falling below rank 50: **45 / 456**
- Queries with ≥1 top-3 doc falling to rank 11–50: **198 / 456**
- Queries with no rank inversion (top-3 stable): **213 / 456**
- Total queries affected by inversion: **243 / 456 (53%)**

### Score Discrimination Collapse

- Queries where `attn_01` spread < 50% of baseline: **114 / 456**
- Queries where `attn_01` spread < 40% of baseline: **70 / 456**
- Mean spread ratio (`attn_01` / baseline): **0.947** (median: 0.780)

### Score Spread: Baseline and All Spherical Configs

| Config | Mean top-10 spread | Queries with spread < 0.05 |
|--------|--------------------|----------------------------|
| Baseline | 0.1345 | 93 |
| `sph_133` | 0.1330 | 95 |
| `sph_2` | 0.1335 | 89 |
| `sph_3` | 0.1341 | 86 |
| `sph_5` | 0.1270 | 97 |
| `sph_10` | 0.1216 | 96 |
| `rand_01` | 0.1121 | ~55 |
| `attn_01` | 0.0974 | 114 |

### Spherical Pooling Specific: Regression Queries (`sph_10` fails, baseline succeeds)

39 queries where baseline has ≥1 hit@10 but `sph_10` has 0 hits. Representative sample:

| Query | Baseline hits | `sph_10` rank of positive |
|-------|--------------|---------------------------|
| `q_154` `"emma stone horror films"` | 1 | `The Favourite` falls out of top 10 |
| `q_147` `"Recommend me something epic"` | 1 (`Dune: Part Two` rank 1) | `Avengers: Endgame` still below top 10 |
| `q_150` `"family friendly halloween movies"` | 1 (`Halloweentown II`) | `Halloweentown II` drops from rank 2 to rank ~12 |
| `q_151` `"crab movies"` | 1 (`Cheech & Chong`) | `Cheech & Chong` drops from rank 3 to rank ~15 |
| `q_132` `"Playful fantasy adventure"` | 1 (`Legend of Ochi`) | `Jumanji: The Next Level` never in top 10 |

### Score Level Drop

| Config | Avg top-1 score | Drop from baseline |
|--------|----------------|-------------------|
| Baseline | 8.2180 | — |
| `sph_133` | 8.2075 | −0.1% |
| `sph_10` | ~8.10 | ~−1.4% |
| `rand_01` | 8.0909 | −1.5% |
| `attn_01` | 8.0510 | −2.0% |

### Key Queries with All-Config Failures (n_pos ≥ 3, all hits@10 = 0)

These are pool-mismatch failures (Mode 4), not compression failures:

- `q_309`: `"horror"` — 8 positives unjudged in retrieved set
- `q_130`: `"tv shows"` — 3 positives unjudged
- `q_347`: `"what new movies are out"` — 3 positives unjudged
- `q_193`: `"recent movie releases from 2024"` — 2 positives unjudged
- `q_194`: `"Suggest me something chilling to watch"` — positives unjudged
- `q_68`: `"golden globes comedy films from twenty twenty"` — positives unjudged

For these, the retrieval is plausible but the benchmark cannot reward it.
