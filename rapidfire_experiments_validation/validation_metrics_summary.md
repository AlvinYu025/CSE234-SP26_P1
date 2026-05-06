# Validation Experiment Summary

- Complete configs: 25/25
- Missing configs: none
- Problem configs: none
- Configs with API/judge warnings: T01(errors=20, judge=21), T02(errors=2, judge=0), T07(errors=0, judge=1)

## Top Overall

| Rank | Config | Overall | Retrieval | Generation | F1@5 | Correctness | Faithfulness | Completeness | Judge Failures | Error Answers | Name |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | T10 | 0.9429 | 0.9421 | 0.9437 | 0.9420 | 0.9778 | 1.0000 | 0.8533 | 0 | 0 | tfidf_420_80_k1_title_related_expand5 |
| 2 | T11 | 0.9348 | 0.9421 | 0.9274 | 0.9420 | 0.9556 | 0.9778 | 0.8489 | 0 | 0 | tfidf_420_80_k1_title_related_expand10 |
| 3 | T13 | 0.9348 | 0.9421 | 0.9274 | 0.9420 | 0.9556 | 0.9778 | 0.8489 | 0 | 0 | tfidf_420_80_k5_title_related_expand5 |
| 4 | T12 | 0.9311 | 0.9421 | 0.9200 | 0.9420 | 0.9556 | 0.9778 | 0.8267 | 0 | 0 | tfidf_420_80_k3_title_related_expand5 |
| 5 | T08 | 0.9288 | 0.9421 | 0.9156 | 0.9420 | 0.9333 | 0.9778 | 0.8356 | 0 | 0 | tfidf_420_80_k1_title_related_expand0 |
| 6 | T05 | 0.9041 | 0.9148 | 0.8933 | 0.9111 | 0.9111 | 0.9778 | 0.7911 | 0 | 0 | tfidf_420_80_k1_title_no_related_expand0 |
| 7 | T09 | 0.9022 | 0.8977 | 0.9067 | 0.8975 | 0.9333 | 1.0000 | 0.7867 | 0 | 0 | tfidf_512_96_k1_title_related_expand0 |
| 8 | T06 | 0.8589 | 0.8704 | 0.8474 | 0.8667 | 0.8444 | 0.9333 | 0.7644 | 0 | 0 | tfidf_512_96_k1_title_no_related_expand0 |
| 9 | T07 | 0.8540 | 0.8310 | 0.8770 | 0.8309 | 0.8889 | 0.9778 | 0.7644 | 1 | 0 | tfidf_320_64_k1_title_related_expand0 |
| 10 | T04 | 0.8359 | 0.8037 | 0.8681 | 0.8000 | 0.8667 | 1.0000 | 0.7378 | 0 | 0 | tfidf_320_64_k1_title_no_related_expand0 |
| 11 | T02 | 0.8321 | 0.8198 | 0.8444 | 0.8148 | 0.8444 | 0.9556 | 0.7333 | 0 | 2 | tfidf_420_80_k1_no_title_no_related_expand0 |
| 12 | T03 | 0.8225 | 0.7975 | 0.8474 | 0.7926 | 0.8444 | 0.9778 | 0.7200 | 0 | 0 | tfidf_512_96_k1_no_title_no_related_expand0 |

## All Configs

| Config | Overall | Retrieval | Generation | Judge Failures | Error Answers | Count | Name |
|---|---:|---:|---:|---:|---:|---:|---|
| D01 | 0.7383 | 0.6499 | 0.8267 | 0 | 0 | 45 | dense_320_64_similarity_k5_rerank3 |
| D02 | 0.7382 | 0.5934 | 0.8830 | 0 | 0 | 45 | dense_320_64_similarity_k5_rerank5 |
| D03 | 0.7251 | 0.6324 | 0.8178 | 0 | 0 | 45 | dense_320_64_similarity_k8_rerank3 |
| D04 | 0.7089 | 0.5823 | 0.8356 | 0 | 0 | 45 | dense_320_64_similarity_k8_rerank5 |
| D05 | 0.7453 | 0.6462 | 0.8444 | 0 | 0 | 45 | dense_320_64_mmr_k8_fetch20_lam07_rerank3 |
| D06 | 0.7036 | 0.5865 | 0.8207 | 0 | 0 | 45 | dense_320_64_mmr_k8_fetch20_lam07_rerank5 |
| D07 | 0.7238 | 0.6106 | 0.8370 | 0 | 0 | 45 | dense_512_96_similarity_k5_rerank3 |
| D08 | 0.7264 | 0.5668 | 0.8859 | 0 | 0 | 45 | dense_512_96_similarity_k5_rerank5 |
| D09 | 0.7366 | 0.6153 | 0.8578 | 0 | 0 | 45 | dense_512_96_similarity_k8_rerank3 |
| D10 | 0.7056 | 0.5579 | 0.8533 | 0 | 0 | 45 | dense_512_96_similarity_k8_rerank5 |
| D11 | 0.7447 | 0.6153 | 0.8741 | 0 | 0 | 45 | dense_512_96_mmr_k8_fetch20_lam07_rerank3 |
| D12 | 0.7056 | 0.5579 | 0.8533 | 0 | 0 | 45 | dense_512_96_mmr_k8_fetch20_lam07_rerank5 |
| T01 | 0.5700 | 0.7370 | 0.4030 | 21 | 20 | 45 | tfidf_320_64_k1_no_title_no_related_expand0 |
| T02 | 0.8321 | 0.8198 | 0.8444 | 0 | 2 | 45 | tfidf_420_80_k1_no_title_no_related_expand0 |
| T03 | 0.8225 | 0.7975 | 0.8474 | 0 | 0 | 45 | tfidf_512_96_k1_no_title_no_related_expand0 |
| T04 | 0.8359 | 0.8037 | 0.8681 | 0 | 0 | 45 | tfidf_320_64_k1_title_no_related_expand0 |
| T05 | 0.9041 | 0.9148 | 0.8933 | 0 | 0 | 45 | tfidf_420_80_k1_title_no_related_expand0 |
| T06 | 0.8589 | 0.8704 | 0.8474 | 0 | 0 | 45 | tfidf_512_96_k1_title_no_related_expand0 |
| T07 | 0.8540 | 0.8310 | 0.8770 | 1 | 0 | 45 | tfidf_320_64_k1_title_related_expand0 |
| T08 | 0.9288 | 0.9421 | 0.9156 | 0 | 0 | 45 | tfidf_420_80_k1_title_related_expand0 |
| T09 | 0.9022 | 0.8977 | 0.9067 | 0 | 0 | 45 | tfidf_512_96_k1_title_related_expand0 |
| T10 | 0.9429 | 0.9421 | 0.9437 | 0 | 0 | 45 | tfidf_420_80_k1_title_related_expand5 |
| T11 | 0.9348 | 0.9421 | 0.9274 | 0 | 0 | 45 | tfidf_420_80_k1_title_related_expand10 |
| T12 | 0.9311 | 0.9421 | 0.9200 | 0 | 0 | 45 | tfidf_420_80_k3_title_related_expand5 |
| T13 | 0.9348 | 0.9421 | 0.9274 | 0 | 0 | 45 | tfidf_420_80_k5_title_related_expand5 |

## Notes

- This summary is for `data/validation-set-golden-qa-pairs.json` validation/dev experiments only.
- Do not mix these rows with `rapidfire_experiments_golden`, which was run on curated golden data.
