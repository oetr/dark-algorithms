# Journal

Ideas tried by the agents improving this repository, newest first, including the ones that
did not work, so later runs don't repeat them. One entry per idea: date, idea, result.

## 2026-11-12 - popcount compressor trees instead of ripple-carry adder trees

- Idea: rewrite a wide neuron as popcount(literals) >= t + #negative weights (a -1 weight
  is an inverted literal, folded into the reading LUT for free); compress the literals with
  a Wallace-style tree of k-input counters (6:3 for k=6); compare the compressed heap with
  the constant threshold by an LSB-first comparator chain that only passes the small carry
  floor((low part - low threshold bits) / 2^c), so the sign-bit inverter disappears too.
  Each neuron is built for every target column height 2..k and the cheapest is kept.
- Result: total 43,424 -> 24,107 LUTs (-44.5%), worst depth 31 -> 19; every case
  improves on both. Fixed heights alone: h=2 27,012, h=4 24,613, h=5 24,523 (k=4 prefers 2,
  k=6 prefers 4-5), hence the per-neuron choice. Choosing by depth first instead of LUTs
  first gave 24,392 LUTs at the same worst depth 19, so LUTs-first stays. Half adders
  (target height 1) are much worse (45,659 LUTs) and are only used as a k=3 fallback.
- The dense baseline uses the same structure, so its LUT counts drop too and the reported
  lut_reduction percentages shrink; they compare against a stronger baseline now.

## 2026-11-12 - initial compiler (PR #1)

- Idea: drop zero weights; one LUT per neuron with fan-in <= k; otherwise per-chunk partial
  sums, LUT ripple-carry adder tree and sign test; structural hashing, constant folding, sweep.
- Result: total 43,424 LUTs over the suite, worst depth 31; 50-88% fewer LUTs than the dense
  baseline depending on sparsity.
