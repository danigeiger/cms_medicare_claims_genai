# Retrieval Experiments

## Experiment 0 — Baseline

**Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`

**Embedding text:**
- Variable
- Label
- Description

**Evaluation set:** 20 natural-language CMS variable queries

### Results

- Top-1 accuracy: 60%
- Top-3 accuracy: 80%
- Top-5 accuracy: 95%

### Observation

The embedding model generally identified the correct semantic category,
but sometimes struggled to distinguish closely related CMS variables.

Examples included attending, operating, rendering, and referring physician
variables. The distinguishing modifier in the variable label appeared to
have an important effect on retrieval ranking.


## Experiment 1 — Label Emphasis

**Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`

**Embedding text:**
- Label
- Label
- Variable
- Description

The label was included twice to test whether giving it greater influence
would improve retrieval of closely related CMS variables.

### Results

- Top-1 accuracy: 60%
- Top-3 accuracy: 90%
- Top-5 accuracy: 100%

### Change From Baseline

- Top-1: no change
- Top-3: +10 percentage points
- Top-5: +5 percentage points

### Observation

Repeating the label did not improve Top-1 accuracy, but it improved
Top-3 and Top-5 retrieval.

Some individual variables improved substantially. For example,
`RNDRNG_PHYSN_NPI` moved from rank 3 to rank 1, while
`PRNCPAL_DGNS_CD` moved from rank 5 to rank 2.

Other variables became worse, showing that label emphasis alone does
not completely solve the ranking problem.

### Next Experiment

Test a second-stage reranking strategy on the candidates returned by
semantic search.