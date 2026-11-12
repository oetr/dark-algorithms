# Journal

Ideas tried by the agents improving this repository, newest first, including the ones that
did not work, so later runs don't repeat them. One entry per idea: date, idea, result.

## 2026-11-12 - lazy last counter stage and critical-path-aware neuron versions

- Idea: the last Wallace stage compresses every too-tall column as far as it can, so a
  77-input neuron at target height 4 ends with 13 heap bits where 25 would fit. Counters
  remove about one bit per LUT and the comparator about two, so those bits are cheaper left
  in the heap. Option: in the last stage remove from each column only its excess over the
  target (counting the bits lower columns' counters pass up), used only if that one stage
  reaches the target, so the stage count stays Wallace's. E.g. a 9-input majority at k=6
  takes 4 LUTs instead of 6, a 16-input one 9 instead of 12, at the same depth.
- Picking per neuron by LUTs first with the lazy option gave 22,021 LUTs but made two cases
  one level deeper (mlp-256-s90 10 -> 11, k=4 16 -> 17); a per-layer depth budget (deepest
  neuron's shallowest version) fixed k=4 but not mlp-256-s90, because filling a layer's slack
  delays the next layer's inputs; plain shallowest-first per neuron gave 22,163 LUTs at the
  old depths. Kept: a first pass builds every neuron shallowest-first, fixing the depth D
  and each neuron's longest path to an output; the second pass gives each neuron the
  cheapest version that arrives by D minus that path. The first pass's version always fits,
  so the network is never deeper than the all-shallowest one.
- Result: total 22,481 -> 22,026 LUTs (-2.0%), worst depth 17 -> 17, no case deeper; every
  case improves (tiny 101 -> 85, mlp-64-s70 1,434 -> 1,336, mlp-256-s90 3,020 -> 2,898,
  mlp-256-s95 830 -> 743, mlp-256-s70 11,251 -> 11,138, k=4 5,845 -> 5,826). Benchmark
  time 13 s -> 37 s (twice the options, two passes; layer-1 trials are reused).
- Failed: a full Dadda schedule (every stage reduces each column only to a height sequence
  d, 6 * floor(d / 3) at k=6) as the only strategy: 25,527 LUTs, worst depth 25. The height
  sequence assumes uniform columns, but the heap starts as one tall column, so it adds
  stages; as an extra option next to Wallace it gave only 22,384 LUTs at depth 18.

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
  A concurrent rework of PR #7 on top of #6 (separate GPC bit-heap candidate per neuron)
  reached 23,271 LUTs / depth 19 (23,050 / depth 20 when always taking the GPC build), so
  the per-column fill above, inside #6's tree, is the stronger form of the same idea.

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
