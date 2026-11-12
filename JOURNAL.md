# Journal

Ideas tried by the agents improving this repository, newest first, including the ones that
did not work, so later runs don't repeat them. One entry per idea: date, idea, result.

## 2026-11-12 - popcount bit heap with GPC compression and a threshold-only carry chain

- Idea: rewrite every wide neuron as a popcount of literals (a -1 weight becomes NOT x, free
  inside a LUT) and add 2^W - T' so the threshold sits on a power of two; the neuron is then
  bit W of the sum, and only the sum modulo 2^(W+1) is built. Compress the bit heap with
  generalized parallel counters ((6;3), (1,5;3), ... picked greedily by bits removed per LUT;
  full adders win at k=4) until columns hold <= 2 bits, then a carry chain packing whole
  columns into each LUT whose last LUT emits bit W (no sign inverter). Also fold constant
  inputs into the threshold, merge repeated signals (x and -x cancel), drop buffer LUTs.
- Result: total 43,424 -> 25,074 LUTs (-42.3%), worst depth 31 -> 20; every case smaller and
  shallower. The dense baseline uses the same compiler, so it shrank too (lut_reduction now
  32-63% instead of 50-88%); absolute LUT count and depth are the goal metrics.
- Next: delay-aware compression (late bits into late stages), counters shared across
  neurons over the same literals, exact synthesis of the carry chain.

## 2026-11-12 - initial compiler (PR #1)

- Idea: drop zero weights; one LUT per neuron with fan-in <= k; otherwise per-chunk partial
  sums, LUT ripple-carry adder tree and sign test; structural hashing, constant folding, sweep.
- Result: total 43,424 LUTs over the suite, worst depth 31; 50-88% fewer LUTs than the dense
  baseline depending on sparsity.
