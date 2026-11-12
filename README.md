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
   A counter can also take bits from two neighbouring binary positions, and with 4-input
   LUTs the tree uses three-bit adders, which are cheaper there.
5. Neurons of the same layer often read the same handful of inputs with the same signs (or
   all with opposite signs, which costs nothing to undo). Such a group of inputs is counted
   once and the count is shared by every neuron that needs it. In a network with 70% zero
   weights almost any two neurons share about a dozen inputs, so a large part of the first
   counting step is shared. Sharing is only used where it never makes a neuron's chain of
   LUTs longer.
6. A short chain of LUTs compares that count with the threshold and directly produces the
   neuron's yes/no output, without ever building the full sum.
7. For every neuron, several tree shapes are tried and the cheapest is kept; identical LUTs
   are shared and unused ones removed.

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
| Counters spanning two binary positions; three-bit adders for 4-input LUTs | 22,481 | 17 |
| Counters shared between neurons that read the same inputs | 19,140 | 17 |

**Ideas ahead.** Choosing the shared input groups better (covering every neuron's inputs
as completely as possible, and sharing the later counting steps too), exactly optimal
circuits for small neurons, choosing circuit shapes for the whole network's longest path,
and benchmarks with networks trained on real tasks, where shared structure may differ from
random networks.

## How it improves itself

An agent picks the most promising idea, implements and tests it, and measures the result.
Changes that keep every circuit correct and make the circuits smaller or faster are merged
on their own. Every idea, including the ones that failed, is recorded in the
[journal](JOURNAL.md), and progress is charted on the "dark-algorithms progress" dashboard
in Dash0.
