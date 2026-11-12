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
- Result before counters were shared: 22,481 -> 22,026 LUTs (-2.0%), worst depth 17 -> 17.
- Rebased onto shared counters (the pull request was opened before they merged): the lazy
  option is also tried for shared neurons, the two passes run under both sharing depth
  policies (four builds, the shallowest then smallest is kept; each policy wins some case),
  and a shared neuron is held to min(its sharing depth limit, its required level). Total
  19,140 -> 18,777 LUTs (-1.9%), worst depth 17 -> 17, no case deeper, every case smaller:
  tiny 86 -> 76, mlp-64-s70 1,223 -> 1,168, mlp-256-s90 2,935 -> 2,819,
  mlp-256-s95 830 -> 743, mlp-256-s70 8,833 -> 8,752, k=4 5,233 -> 5,219. Benchmark time
  84 s -> 273 s, most of it the dense baseline. The dense baseline of mlp-256-s70 grows
  (12,765 -> 13,604) because the shallowest network is preferred, so its lut_reduction
  rises 30.8% -> 35.7% partly for that reason; the gate compares only the sparse circuits.
- Seen but not taken: under the per-layer sharing policy mlp-256-s90 reaches 2,737 LUTs at
  depth 11 instead of 2,819 at depth 10 (the gate rejects any depth increase).
- Failed: a full Dadda schedule (every stage reduces each column only to a height sequence
  d, 6 * floor(d / 3) at k=6) as the only strategy: 25,527 LUTs, worst depth 25. The height
  sequence assumes uniform columns, but the heap starts as one tall column, so it adds
  stages; as an extra option next to Wallace it gave only 22,384 LUTs at depth 18.

## 2026-11-12 - counters shared across the neurons of a layer

- Idea: neurons of a layer that contain the same k literals (k=6; full adders at k=4),
  or all of them negated, read one shared counter: a negated member sees g - c, which is
  the counter's bits inverted plus the constant g - (2^m - 1), folded into the reading
  LUTs for free. Groups are carved greedily: the pair of wide neurons with the most common
  free literals in one orientation, widened to every further neuron that still has k of
  them; at k=6 a second pass shares 5-literal groups (a shared 5:3 counter removes 2 bits
  for 1.5 LUTs per neuron). A neuron's own literals get a first stage at the same level,
  then the merged heap goes through the usual per-neuron option search. In mlp-256-s70
  the best partner of a first-layer neuron shares 21 literals in one orientation.
- Result: total 22,481 -> 19,140 LUTs (-14.9%), worst depth 17 -> 17, no case deeper.
  mlp-256-s70 11,251 -> 8,833 (-21.5%), k=4 case 5,845 -> 5,233, mlp-64-s70 1,434 ->
  1,223, tiny 101 -> 86; mlp-256-s95 unchanged (two neurons share under one literal there).
  Sharing applies to the dense baseline too, so its LUT counts drop and lut_reduction
  shrinks for s70 (33.3% -> 30.8%). Benchmark time 12 s -> 83 s: each layer is built
  with and without sharing and the network under two depth policies.
- Depth: the cheapest version of a shared neuron can be one or two levels deeper. Unlimited:
  19,293 LUTs but three cases deeper (tiny 4 -> 5, mlp-64 11 -> 13, s90 10 -> 11). Each
  shared neuron capped at its layer's unshared depth: 19,309 without and 19,080 with
  5-literal groups, but with them s90 went 10 -> 11, because shallow neurons rose to the
  layer maximum and that compounded in the next layer. Capped at each neuron's own
  unshared depth (a neuron with no version that fits is built alone): 19,372, never
  deeper. The compiler now builds both policies and keeps the shallower, then smaller.
- Not kept: a Dadda-style lazy schedule (each stage compresses a column only by its excess
  over the stage height, so more raw bits reach the comparator) as an extra per-neuron
  option: 22,481 -> 22,409 (-0.3%), but tiny 4 -> 6 and mlp-64 11 -> 13 in depth and 3.5x
  benchmark time. It only wins for 7-10 input neurons; for wide ones the leftover tall
  columns make the comparator chain carry several bits and cost more than the counters
  saved.

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
