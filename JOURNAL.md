# Journal

Ideas tried by the agents improving this repository, newest first, including the ones that
did not work, so later runs don't repeat them. One entry per idea: date, idea, result.

## 2026-11-12 - two-column counters and full-adder groups as per-neuron options

- Idea: in the Wallace stages, a column's short last group (3-5 bits at k=6) removes only
  1-2 bits for 2-3 LUTs; fill its free inputs with bits of the next column, a two-column
  GPC ((1,5;3), (1,4;3), (2,3;3)), picked by bits removed per LUT. At k=4 a (4;3) counter
  removes one bit for three LUTs, worse than a full adder, so also offer cutting columns
  into 3-bit groups. Both are extra per-neuron options next to the target column height.
- Result: total 24,107 -> 22,481 LUTs (-6.7%), worst depth 19 -> 17. k=4 case 7,113 ->
  5,845 (-17.8%, depth 19 -> 16) from full adders, which win for 194 of its 200 wide
  neurons; mlp-256-s70 11,606 -> 11,251 from GPCs, which win for 154 of 676 wide k=6
  neurons. Full adders never win at k=6 and borrowing cannot pay off below k=5, so those
  options are only tried where they can matter (benchmark time 26 s -> 12 s).
- Not merged, superseded (PR #7): a parallel run built a second bit-heap compiler (power-
  of-two threshold offset, greedy GPCs, carry chain) at the same time as PR #6; against the
  old baseline it reached 25,074 LUTs / depth 20, worse than PR #6's 24,107 / 19, and it
  conflicted with it. Its extra normalisations (fold constant inputs into the threshold,
  merge a signal feeding a neuron twice into one net weight, wire 1-input buffers through)
  were ported onto PR #6's compiler and changed no pruned circuit in the suite: random
  networks have no constant or duplicate neurons. Worth revisiting with trained networks,
  where dead neurons are common. Careful: reordering terms there also groups the dense
  baseline's zero-weight wires into free counters, which silently shrinks the baseline.

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
