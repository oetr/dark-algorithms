# Journal

Ideas tried by the agents improving this repository, newest first, including the ones that
did not work, so later runs don't repeat them. One entry per idea: date, idea, result.

## 2026-11-12 - two-column GPC bit heap as an extra per-neuron candidate; input folding

- Idea: next to the counter tree + comparator chain, build every wide neuron a second way:
  add 2^W - T' so the threshold sits on a power of two and only bit W of the sum (mod
  2^(W+1)) is needed; compress with two-column GPCs ((6;3), (1,5;3), (2,3;3), ... chosen
  greedily by heap bits removed per LUT; full adders at k=4) until columns hold <= 2 bits,
  then a carry chain whose last LUT emits bit W. Keep it when it saves LUTs without making
  the neuron deeper. Also fold constant inputs into the threshold, merge a signal used
  twice into its net weight (x and -x cancel), and wire 1-input buffer LUTs through.
- Result: 24,107 -> 23,271 LUTs (-3.5%), worst depth 19 -> 19. Nearly all of it is k=4
  (7,113 -> 6,299, -11.4%), where single-column 4:3 counters are weak; at k=6 the counter
  tree wins almost every neuron (mlp-256-s70 -22 LUTs from folding). Built on its own
  (written in parallel with the counter tree), the GPC heap gave 25,074 LUTs / depth 20.
  Taking the GPC build whenever it has fewer LUTs gives 23,050 but worst depth 20, which
  the gate blocks.
- Next: GPCs inside the counter tree's Wallace stages instead of a separate build;
  network-level depth budgeting so the 23,050-LUT trade-off can be taken where the neuron
  is off the critical path.

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
