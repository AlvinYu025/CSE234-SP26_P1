# Token-Budget Context Ablation on Validation Set

All runs measure prompt tokens by calling the generation model with `max_tokens=1` and reading `response.usage.prompt_tokens`.

Best validation overall: **TC02** (`TC02_chunk420_overlap80_rel5_sequential_cap1900`), overall `0.9443`, retrieval `0.9389`, generation `0.9496`, max prompt tokens `1858`.

| Config | Chunk | Related | Strategy | Cap | Retrieval | Generation | Overall | Max prompt tokens | > cap | > 2000 |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| TC01 | 420/80 | 5 | sequential | 1950 | 0.9389 | 0.9422 | 0.9406 | 1909 | 0 | 0 |
| TC02 | 420/80 | 5 | sequential | 1900 | 0.9389 | 0.9496 | 0.9443 | 1858 | 0 | 0 |
| TC03 | 384/64 | 5 | sequential | 1900 | 0.8199 | 0.8681 | 0.8440 | 1895 | 0 | 0 |
| TC04 | 320/64 | 5 | sequential | 1900 | 0.8310 | 0.8711 | 0.8511 | 1891 | 0 | 0 |
| TC05 | 256/48 | 5 | sequential | 1900 | 0.7755 | 0.8148 | 0.7951 | 1880 | 0 | 0 |
| TC06 | 420/80 | 4 | sequential | 1900 | 0.9389 | 0.9304 | 0.9346 | 1858 | 0 | 0 |
| TC07 | 420/80 | 3 | sequential | 1900 | 0.9351 | 0.9481 | 0.9416 | 1744 | 0 | 0 |
| TC08 | 320/64 | 5 | balanced | 1900 | 0.8310 | 0.8726 | 0.8518 | 1891 | 0 | 0 |
| TC09 | 320/64 | 5 | question_aware | 1900 | 0.8310 | 0.8326 | 0.8318 | 1900 | 0 | 0 |
| TC10 | 256/48 | 5 | staged | 1900 | 0.7755 | 0.8074 | 0.7914 | 1892 | 0 | 0 |

Notes:

- `Retrieval` is the released span-overlap retrieval score.
- `Generation` is the released partial judge score using correctness, faithfulness, and completeness.
- `Overall` is the mean of retrieval and generation, used only as a validation comparison heuristic.
- Every listed prompt-token value is returned by the generation model API, not estimated locally.
