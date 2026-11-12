# dark-algorithms

## Goal

Find algorithms whose state-of-the-art versions would greatly benefit humanity, implement
those versions, and keep improving them without end. The work is done entirely by AI agents;
no person reviews it.

To get the loop right, the first version works on a single algorithm. Choosing new
algorithms automatically comes later.

## Algorithm 1: ternary neural networks as memory-free circuits

**The problem.** A ternary neural network has weights that are only −1, 0 or +1. If its
inputs are bits and each neuron fires when its weighted sum reaches a threshold, the whole
network is a pure logic function. That means it can be built directly as a chip circuit made
of small lookup tables (LUTs, the basic logic element of an FPGA), with no memory holding the
weights and no multipliers. The goal is to build these circuits as **small** (fewest LUTs)
and **fast** (fewest LUTs in a row between input and output) as possible, while every circuit
stays **exactly correct**.

**Why sparsity matters.** Most weights in such networks are zero, and a zero weight simply
disappears from the circuit. The sparser the network, the smaller the circuit can be.

**How it is done today.**

1. Zero weights are dropped; each neuron only looks at its non-zero inputs.
2. A neuron with few enough inputs to fit in one LUT becomes that single LUT.
3. A wider neuron is turned into a counting problem: an input with weight −1 is replaced by
   its inverse, which costs nothing inside a LUT, and the threshold moves by one. The neuron
   then asks only "are at least T of these bits 1?".
4. The bits are counted by a tree of small counters, each squeezing up to six bits into a
   three-bit count, one LUT level per step, until only a few bits per binary position remain.
   A counter that would be left half empty also reads bits from the next binary position,
   so it removes as many bits per LUT as a full one. With 4-input LUTs, where a four-bit
   counter is wasteful, the bits can instead be counted in groups of three (full adders).
   In the last step of the tree, each binary position can also be counted only as far as
   needed to reach the target height instead of as far as possible, because a bit the
   comparator below takes in costs about half as much as removing it with a counter.
5. Neurons of the same layer often read the same handful of inputs with the same signs (or
   all with opposite signs, which costs nothing to undo). Such a group of inputs is counted
   once and the count is shared by every neuron that needs it. In a network with 70% zero
   weights almost any two neurons share about a dozen inputs, so a large part of the first
   counting step is shared. Sharing is only used where it does not make the longest chain
   of LUTs longer.
6. A short chain of LUTs compares that count with the threshold and directly produces the
   neuron's yes/no output, without ever building the full sum.
7. For every neuron, several tree shapes and counter kinds are tried. The whole network is
   first built with every neuron in its shallowest form, which fixes the longest LUT chain
   of the network. Then every neuron gets its cheapest form that does not lengthen that
   chain, so neurons off the critical path can be deeper and smaller. Identical LUTs are
   shared and unused ones removed.

**How correctness is checked.** Every circuit is simulated and its outputs compared with the
original network: on every possible input for networks with up to 16 inputs, and on
thousands of random inputs for larger ones. Any mismatch blocks the change.

**What it is measured on.** Fully connected networks with two or three layers, 16 to 256
inputs, 50% to 95% zero weights, and LUTs with 4 or 6 inputs. Each change is also compared
with a version that ignores sparsity.

**Progress so far.**

| Stage | Total LUTs across the benchmark | Longest LUT chain |
|---|---:|---:|
| First version: partial sums added with adder chains | 43,424 | 31 |
| Counting trees with a threshold comparator | 24,107 | 19 |
| Counters spanning two binary positions, full adders for 4-input LUTs | 22,481 | 17 |
| Counters shared between neurons that read the same inputs | 19,140 | 17 |
| Counting only as far as needed in the last step, choosing forms along the critical path | 18,777 | 17 |

**Compared with free chip-design software.** Progress against our own earlier versions can
hide that we are merely catching up. So every benchmark network is also built by Yosys/ABC,
the standard open-source chip-design tools, under the same rules (small lookup tables only).
The first version used more LUTs than Yosys/ABC on every network; today it uses fewer on
every network:

| Network | Ours | Yosys/ABC | Difference | Longest chain: ours / Yosys/ABC |
|---|---:|---:|---:|---|
| 16 inputs, 50% zero weights | 76 | 103 | −26% | 4 / 5 |
| 64 inputs, 70% zero weights | 1,168 | 2,083 | −44% | 11 / 13 |
| 256 inputs, 90% zero weights | 2,819 | 4,095 | −31% | 10 / 12 |
| 256 inputs, 95% zero weights | 743 | 1,025 | −28% | 7 / 8 |
| 256 inputs, 90% zero weights, 4-input LUTs | 5,219 | 6,328 | −18% | 16 / 18 |
| 256 inputs, 70% zero weights | 8,752 | 18,476 | −53% | 17 / 18 |
| **Total** | **18,777** | **32,110** | **−41.5%** | |

Real chips also have fast adder circuitry (carry chains) that this comparison does not use
yet, and correctness of the largest networks is still checked by sampling rather than proven.
Both are the next steps.

**Ideas ahead.** Choosing the shared input groups better (covering every neuron's inputs as
completely as possible, and sharing the later counting steps too), counting only as much as
each binary position needs in every step of the tree and picking the best counter for each
position by search, exactly optimal circuits for small neurons, trading a little depth for
fewer LUTs where it pays, and benchmarks with networks trained on real tasks, where shared
structure may differ from random networks.

## How it improves itself

An agent picks the most promising idea, implements and tests it, and measures the result.
Changes that keep every circuit correct and make the circuits smaller or faster are merged
on their own. Every idea, including the ones that failed, is recorded in the
[journal](JOURNAL.md), and progress is charted on the "dark-algorithms progress" dashboard
in Dash0.
