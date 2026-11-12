# Journal

Ideas tried by the agents improving this repository, newest first, including the ones that
did not work, so later runs don't repeat them. One entry per idea: date, idea, result.

## 2026-11-12 - a benchmark network trained on a real task (digits-64-s70)

- Why: every case was a random network (independent weights, thresholds centred on the
  expected sum), so every compiler change was tuned on structure no trained network has.
  Compiler headroom is also thin: per-neuron counting trees are near their bound (in
  mlp-256-s90's first layer the network costs 2,399 LUTs against 2,496 for every neuron's
  cheapest version built alone; in the k=4 case 4,100 of 4,956 LUTs are 3-input full-adder
  LUTs, the value-preserving limit at k=4), and re-carving shared groups with 20 randomised
  tie-breaks gained only ~50 LUTs (k=4 layer 1 +26, mlp-256-s70 layer 2 +15, mlp-64-s70
  layer 2 +6; not kept). So the measurement was the better place to spend this run.
- Idea: train a ternary threshold MLP on scikit-learn's bundled digits (8x8 pixels as bits,
  ink >= 8 of 16), 64 -> 64 -> 32 -> 10 at 70% zeros, the shape and sparsity of mlp-64-s70,
  so the two cases compare a trained and a random network directly. Straight-through
  training: per-layer magnitude quantile ternarization, hard threshold activations, integer
  thresholds from rounded biases in every forward pass, squared hinge loss on one-vs-rest
  outputs. Seeded; the weights are committed as JSON and the benchmark never retrains.
- Result: 87.2% test accuracy (exactly the true digit's output is 1; 90.0% train). The
  network: 17 of 64 pixels read by no neuron, fan-in 0-34 in layer 1 and 0-41 in layer 2
  (random: 9-30, 10-27), mean |threshold - centre| 2.6 / 1.9 / 3.5 (random: at most 1.5), 6 dead
  neurons. Shared-group coverage of layer 1 is 74% (902 of 1,223 literals) against 49% in
  mlp-64-s70: neurons concentrate on the same informative pixels, so sharing pays more on
  trained networks. Circuit: 946 LUTs, depth 11, proven, against 1,105 / 11 for mlp-64-s70.
  Yosys/ABC: YLUT LUTs, depth YDEPTH (YDIFF); vendor flow (Xilinx): FLUT LUT sites and
  FCARRY carry cells (FDIFF). Existing cases unchanged (18,034 LUTs, worst depth 15).
- Tried first: 80% zeros; 13 of 64 first-layer neurons came out dead, so 70% (matching an
  existing random case) was kept. lr 0.005-0.1 and 60-150 epochs gave 32-86% test accuracy.
- Next: larger trained cases in published FPGA network shapes (e.g. LogicNets' jet-tagging
  16-64-32-32-32-5, MNIST at 784 inputs once compile time allows), and compiler work aimed at
  trained structure: dead neurons already fold to constants, but uneven fan-in and
  heavily overlapping inputs favour sharing groups across more than two neurons.

## 2026-11-12 - comparator chain planned as a whole, with partial-column steps

- Why: the counters are at their information limit for value-preserving compression (a 6:3
  counter removes one bit per LUT, a full adder half a bit per LUT at k=4), so only
  threshold-aware logic can beat them, and that is the comparator chain. It was greedy
  (as many whole columns as fit, then next-column bits) and could only ever take whole
  columns, so a neuron a few literals wider than a LUT always paid for a counter first:
  8 literals at k=6 took a 6:3 counter plus one LUT, 4 LUTs.
- Idea: plan the chain with a memoised search over its steps (m whole columns, then a bits
  of the next column, lowest arrival level first), counting one LUT per carry bit and the
  real arrival levels. a > 0 with m = 0 is a step that takes only part of the lowest
  column and does not shift the carry; with the clamp from the previous entry the carry
  stays small near the end of the chain. 8 literals at k=6: six clamp to {<=1, 2, 3, >=4},
  two LUTs, then one LUT over the carry and the last two literals: 3 LUTs, same depth.
  Each compressor tree is tried with the shallowest plan, the cheapest plan and the old
  greedy chain (built once, three chains tried on it), so every neuron's set of versions
  contains the old one; neurons with at most 3k literals also get a version with no
  counters at all. The carry's upper bound is tracked exactly instead of as all-ones.
- Result: total 18,210 -> 18,034 LUTs (-1.0%), worst depth 16 -> 15, every case proven.
  tiny 76 -> 71, mlp-64-s70 1,111 -> 1,105, mlp-256-s90 2,756 -> 2,706 and depth 10 -> 9,
  mlp-256-s95 742 -> 729 and 7 -> 6, k=4 5,067 -> 4,956 and 16 -> 15, mlp-256-s70
  8,458 -> 8,467 but 16 -> 15 (the compiler keeps the shallowest network, then the
  smallest). Gap to Yosys/ABC -43.3% -> -43.8%, to the vendor flows -33.3% -> -34.6%.
- Single neurons (fixed sample of 126 at k=6 and 132 at k=4, widths k+1..29 and 40..76,
  central thresholds): cheapest versions 3,811 -> 3,777 (k=6) and 7,632 -> 7,520 (k=4);
  shallowest versions' summed depth 531 -> 516 and 843 -> 823. Off-centre thresholds gain
  much more (a 12-literal popcount >= 2 at k=6: 7 -> 5 LUTs), which trained networks
  with skewed thresholds should profit from.
- Cost: the first version took 17 minutes for the benchmark (3.3 million planner states
  for mlp-64-s70 alone, most of them shared neurons re-planned per trial). One search now
  yields both plans, column sums are precomputed, and planners are cached by heap across
  trials and the four network builds: mlp-64-s70 138 s -> 32 s, the benchmark 7.2 minutes
  here with the tests running alongside (4.5 before).
- Not kept: letting the last counter stage leave column 0 k bits tall (the chain's first
  step has no carry) as an extra per-neuron option: 0.4% fewer LUTs on single neurons
  (k=6 4,895 -> 4,877, k=4 9,659 -> 9,624), not worth the extra trials on its own.
- Measured, not pursued: a plain search over the old chain's step choices (whole columns
  only) saves under 1% of the comparator, which is 8-17% of a neuron's LUTs.

## 2026-11-12 - comparator chain absorbs next-column bits and clamps its carry

- Why: per neuron the counters are near their local optimum (see the entry below), but the
  comparator chain only ever took whole columns: a step whose next column did not fit left
  its spare LUT inputs empty, and its carry kept every value of floor((low - t_low) / 2^c),
  even values that cannot change the outcome.
- Idea: (1) bits of the next column have relative weight 2^d with d at least the step's
  shift, so with the carry taken at position d they just add to it:
  floor((v + 2^d * b - t_low) / 2^d) = floor((v - t_low) / 2^d) + b. A step fills its free
  inputs with such bits while its carry needs no more bits than without them, so they
  leave the heap for free. (2) The output is q' + rest >= t' with rest in [0, r], so every
  q' >= t' acts like t' and every q' < t' - r like t' - r - 1: the carry is clamped to
  [t' - r - 1, t'], which can drop a carry bit (one LUT each). Both apply to every neuron,
  no new per-neuron option, so compile time is unchanged.
- Result: total 18,471 -> 18,210 LUTs (-1.4%), worst depth 16 -> 16, every case proven, no
  case larger or deeper: mlp-256-s70 8,592 -> 8,458, k=4 5,129 -> 5,067, mlp-256-s90
  2,807 -> 2,756, mlp-64-s70 1,124 -> 1,111, mlp-256-s95 743 -> 742, tiny unchanged. Gap
  to Yosys/ABC -42.5% -> -43.3%, to the vendor flows -32.4% -> -33.3%. Benchmark time
  4 min 23 s -> 4 min 34 s. The dense baseline shrinks too (mlp-256-s70 12,198 -> 11,082),
  so lut_reduction drops although the sparse circuits got smaller.
- Single neurons (288 random ones per k, widths k+1..29 and 40-77, best of all options):
  k=6 3,934 -> 3,887 LUTs with both, 3,930 with clamping alone; k=4 8,091 -> 7,980, 8,079
  with clamping alone. No neuron got larger with both on. Absorbing as many bits as fit
  even at the price of an extra carry bit was worse than the plain chain (measured on an
  earlier sample: k=6 1,977 -> 1,982).
- Next in the comparator: choosing how many bits each step takes by a small search over
  the whole chain instead of greedily per step, and letting the last counter stage know
  which bits the comparator will absorb.

## 2026-11-12 - shared counter groups carved most-constrained neuron first

- Why: the four merges before this one changed only the measurement (18,777 LUTs since
  the lazy last stage). Per neuron the tree is close to its local optimum: the first two
  Wallace stages of a 72-input neuron remove exactly one bit per LUT (54 LUTs for 54 bits)
  and a 7-input threshold needs 3 LUTs, which is what we build. Sharing is where the
  dense cases still lose: a 6:3 counter shared by two neurons removes two bits per LUT, and
  the pair-first greedy shared only 71% of mlp-256-s70's first-layer literals (57% in
  its second layer).
- Idea: carving groups is a matching problem, and taking the pair with the most common
  literals first uses up well-connected neurons' literals early. Each round now serves the
  neuron with the fewest partners left (two neurons are partners while they have k common
  free literals in one orientation; ties: most free literals), pairs it with its partner
  that has the fewest partners itself (then most common literals), and takes the k common
  literals that the fewest other neurons still hold.
- Result: total 18,777 -> 18,471 LUTs (-1.6%), worst depth 17 -> 16, every case proven,
  no case larger or deeper: mlp-256-s70 8,752 -> 8,592 and depth 17 -> 16, k=4 5,219 ->
  5,129 (full-adder groups), mlp-64-s70 1,168 -> 1,124, mlp-256-s90 2,819 -> 2,807; tiny
  and mlp-256-s95 unchanged. Coverage of mlp-256-s70 layer 1: 71% -> 83%, layer 2: 57% ->
  67%, with fewer three-neuron groups (2.27 -> 2.03 members on average). Gap to Yosys/ABC
  -41.5% -> -42.5%, to the vendor flows -31.4% -> -32.4%. The dense baseline uses the same
  sharing and shrinks too (mlp-256-s70 13,604 -> 12,198), so lut_reduction drops for the
  70% cases although the sparse circuits got smaller. Compile time unchanged.
- Variants measured on mlp-256-s70 + mlp-64-s70 (base 9,920): most-constrained neuron with
  its best partner and literals by index 9,721; the same with least-held literals 9,664
  (8,529 at depth 17 for s70); with the most constrained partner as well 9,716, but s70
  at depth 16. Per depth policy that last variant is best under both: 8,474 at depth 17
  (layer-wide limit) and 8,592 at depth 16 (per-neuron limit), against 8,752 and 8,964 on
  main; the compiler keeps the shallower network.
- Not tried, judged poor: sharing second-stage counters between a pair. The bits two
  neurons share after the first stage are at most a few per column, and a shared full
  adder over three of them removes one bit per LUT, no better than each neuron's own 6:3
  counters; it would need six shared bits in one column (six groups between one pair).

## 2026-11-12 - real-FPGA reference: Yosys vendor flows with carry chains

- Idea: the LUT-only Yosys/ABC reference (`synth -lut K`) cannot use carry chains, which make
  adders and comparators cheap on real FPGAs, so "-41.5% vs Yosys/ABC" might only hold on a
  target that does not exist. Synthesize the same behavioral Verilog with `synth_xilinx`
  (7-series LUT6 + CARRY4) for k=6 cases and `synth_ecp5` (LUT4 + CCU2C) for the k=4 case,
  count LUT sites (Xilinx: LUT1-6 and INV; ECP5: LUT4 plus two per CCU2C, which holds two
  LUT4s), carry cells and slice muxes, and compare with our LUT count: our k-LUTs map one to
  one onto the target's LUT primitives with no carry cells.
- Result (measurement only, circuits unchanged at 18,777 LUTs / depth 17): on the five cases
  the vendor flows finished we use 10,025 LUT sites against 14,621 (-31.4%), and the vendor
  flows also use 2,350 carry cells. Per case: tiny -43.3%, mlp-64-s70 -45.3%, mlp-256-s90
  -33.6%, mlp-256-s95 -36.8%, k=4 on ECP5 -24.7% (the smallest margin, as against
  Yosys/ABC). So the advantage holds on carry-chain targets. The vendor flows are faster than
  the LUT-only one (12-33 s per 256-input case), but mlp-256-s70 did not finish in 560 s.
- Seen: our circuit, written as plain LUT logic and handed to the same flow, comes back
  larger: 76 -> 87, 1,168 -> 1,490, 2,819 -> 3,395, 743 -> 815 on Xilinx and 5,219 -> 6,860
  LUT4 on ECP5. ABC re-maps through its own structure and loses the counter trees, so the
  circuits should be placed as built (LUT primitives with INIT), never re-synthesized.
- Open: depth on these targets (a CARRY4 hop is far faster than a LUT level, so LUT levels
  alone are not comparable), and whether our own comparators or last tree stages could use
  carry chains to save LUTs.

## 2026-11-12 - formal equivalence proof for every benchmark circuit

- Idea: wide cases were only checked on 4,096 random vectors. Prove each circuit equal to
  its network for all inputs, layer by layer: with the previous layer's output signals as
  free variables (the compiler now records them per layer), every neuron's cone must equal
  its threshold function. Merged or constant hidden neurons use the same variable or
  constant in the reference; a swept hidden neuron gets its own free variable.
- Failed first: a plain BDD of each neuron's cone over its inputs. Fine for the 90-95%
  sparse cases (all proven in under 2 s), but in mlp-256-s70 the middle compressor stages
  compute bits of several partial sums whose groups differ per column, so one 77-input
  neuron needed 0.9-1.9 million nodes depending on the variable order (depth-first,
  reversed, creation order; reverse creation order over 3 million) and the sandbox ran out
  of memory at 6.5 GB. ABC `cec`, as AGENTS.md suggested, is not available: the PyPI
  Yosys build ships no standalone ABC binary.
- Kept: arithmetic reduction before the BDD. The neuron's sum is a heap const +
  sum(c_s * s), starting from the reference weights. A LUT group over the same inputs
  replaces them when integers a_o, d with sum(c_i * in_i) == sum(a_o * out_o) + d hold on
  all 2^n input rows (exact check), and only if nothing else in the cone reads those inputs.
  The comparator left over is checked by a BDD over the remaining heap bits, least
  significant first, against heap >= theta. This needs no knowledge of the compiler's
  intent: every step holds for all inputs, so a proof is sound.
- Result: all six cases proven, 0.01-1.4 s each (about 2.2 s total). LUTs and depth unchanged
  (18,777 / 17). Tests check soundness by mutation: flipped LUT bits that exhaustive (16
  inputs or fewer) or random simulation catches are never proven, and single-layer bugs
  give a counterexample that really fails. The regression gate now fails if a case that
  the base proved is no longer proven.

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
