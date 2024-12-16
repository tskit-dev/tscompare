# MIT License
#
# Copyright (c) 2024 Tskit Developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Test tools for mapping between node sets of different tree sequences
"""
from collections import defaultdict
from itertools import combinations

import msprime
import numpy as np
import pytest
import scipy.sparse
import tskit

import tscompare

# --- simulate test case ---
demo = msprime.Demography.isolated_model([1e4])
for t in np.linspace(500, 10000, 20):
    demo.add_census(time=t)
true_unary = msprime.sim_ancestry(
    samples=10,
    sequence_length=1e6,
    demography=demo,
    recombination_rate=1e-8,
    random_seed=1024,
)
true_unary = msprime.sim_mutations(true_unary, rate=2e-8, random_seed=1024)
assert true_unary.num_trees > 1
true_simpl = true_unary.simplify(filter_sites=False)
true_ext = true_simpl.extend_haplotypes()


def naive_shared_node_spans(ts, other):
    """
    Inefficient but transparent function to get span where nodes from two tree
    sequences subtend the same sample set
    """

    def _clade_dict(tree):
        clade_to_node = defaultdict(set)
        for node in tree.nodes():
            clade = frozenset(tree.samples(node))
            clade_to_node[clade].add(node)
        return clade_to_node

    assert ts.sequence_length == other.sequence_length
    assert ts.num_samples == other.num_samples
    out = np.zeros((ts.num_nodes, other.num_nodes))
    for interval, query_tree, target_tree in ts.coiterate(other):
        query = _clade_dict(query_tree)
        target = _clade_dict(target_tree)
        span = interval.right - interval.left
        for clade, nodes in query.items():
            if clade in target:
                for i in nodes:
                    for j in target[clade]:
                        out[i, j] += span
    return scipy.sparse.csr_matrix(out)


def naive_node_span(ts):
    """
    Ineffiecient but transparent function to get total span
    of each node in a tree sequence, including roots.
    """
    node_spans = np.zeros(ts.num_nodes)
    for t in ts.trees():
        for n in t.nodes():
            if t.parent(n) != tskit.NULL or t.num_children(n) > 0:
                span = t.span
                node_spans[n] += span
    return node_spans


def naive_compare(ts, other, transform=None):
    """
    Ineffiecient but transparent function to compute dissimilarity
    and root-mean-square-error between two tree sequences.
    """

    def f(t):
        return np.log(1 + t)

    if transform is None:
        transform = f

    shared_spans = naive_shared_node_spans(ts, other).toarray()
    max_span = np.max(shared_spans, axis=1)
    assert len(max_span) == ts.num_nodes
    time_array = np.zeros((ts.num_nodes, other.num_nodes))
    dissimilarity_matrix = np.zeros((ts.num_nodes, other.num_nodes))
    for i in range(ts.num_nodes):
        # Skip nodes with no match in shared_spans
        if max_span[i] == 0:
            continue
        else:
            for j in range(other.num_nodes):
                if shared_spans[i, j] == max_span[i]:
                    time_array[i, j] = np.abs(
                        transform(ts.nodes_time[i]) - transform(other.nodes_time[j])
                    )
                    dissimilarity_matrix[i, j] = 1 / (1 + time_array[i, j])
    best_match = np.argmax(dissimilarity_matrix, axis=1)
    best_match_spans = np.zeros((ts.num_nodes,))
    time_discrepancies = np.zeros((ts.num_nodes,))
    for i, j in enumerate(best_match):
        best_match_spans[i] = shared_spans[i, j]
        time_discrepancies[i] = time_array[i, j]
    node_span = naive_node_span(ts)
    total_node_spans = np.sum(node_span)
    total_other_spans = np.sum(naive_node_span(other))
    match_span = np.sum(best_match_spans)
    rmse = np.sqrt(np.sum(node_span * time_discrepancies**2) / total_node_spans)
    return match_span, total_node_spans, total_other_spans, rmse


@pytest.mark.parametrize("ts", [true_unary, true_simpl])
class TestCladeMap:
    def test_map(self, ts):
        """
        test that clade map has correct nodes, clades
        """
        clade_map = tscompare.CladeMap(ts)
        for tree in ts.trees():
            for node in tree.nodes():
                clade = frozenset(tree.samples(node))
                assert node in clade_map._nodes[clade]
                assert clade_map._clades[node] == clade
            clade_map.next()

    def test_diff(self, ts):
        """
        test difference in clades between adjacent trees.
        """
        clade_map = tscompare.CladeMap(ts)
        tree_1 = ts.first()
        tree_2 = ts.first()
        while True:
            tree_2.next()
            diff = clade_map.next()
            diff_test = {}
            for n in set(tree_1.nodes()) | set(tree_2.nodes()):
                prev = frozenset(tree_1.samples(n))
                curr = frozenset(tree_2.samples(n))
                if prev != curr:
                    diff_test[n] = (prev, curr)
            for node in diff_test.keys() | diff.keys():
                assert diff_test[node][0] == diff[node][0]
                assert diff_test[node][1] == diff[node][1]
            if tree_2.index == ts.num_trees - 1:
                break
            tree_1.next()


class TestNodeMatching:

    @pytest.mark.parametrize(
        "ts",
        [true_simpl, true_unary],
    )
    def test_node_spans(self, ts):
        eval_ns = tscompare.node_spans(ts)
        naive_ns = naive_node_span(ts)
        assert np.allclose(eval_ns, naive_ns)

    @pytest.mark.parametrize("pair", combinations([true_simpl, true_unary], 2))
    def test_shared_spans(self, pair):
        """
        Check that efficient implementation returns same answer as naive
        implementation
        """
        check = naive_shared_node_spans(pair[0], pair[1])
        test = tscompare.shared_node_spans(pair[0], pair[1])
        assert check.shape == test.shape
        assert check.nnz == test.nnz
        assert np.allclose(check.data, test.data)

    @pytest.mark.parametrize("ts", [true_simpl])
    def test_match_self(self, ts):
        """
        Check that matching against self returns node ids

        TODO: this'll only work reliably when there's not unary nodes.
        """
        time, _, hit = tscompare.match_node_ages(ts, ts)
        assert np.allclose(time, ts.nodes_time)
        assert np.array_equal(hit, np.arange(ts.num_nodes))


class TestDissimilarity:

    def verify_compare(self, ts, other, transform=None):
        match_span, ts_span, other_span, rmse = naive_compare(
            ts, other, transform=transform
        )
        dis = tscompare.compare(ts, other, transform=transform)
        assert np.isclose(1.0 - match_span / ts_span, dis.arf)
        assert np.isclose(match_span / other_span, dis.tpr)
        assert np.isclose(ts_span - match_span, dis.dissimilarity)
        assert np.isclose(ts_span, dis.total_span[0])
        assert np.isclose(other_span, dis.total_span[1])
        assert np.isclose(rmse, dis.rmse)

    @pytest.mark.parametrize(
        "pair",
        [(true_ext, true_ext), (true_simpl, true_ext), (true_simpl, true_unary)],
    )
    def test_basic_comparison(self, pair):
        """
        Check that efficient implementation reutrns the same answer as naive
        implementation.
        """
        self.verify_compare(pair[0], pair[1])

    @pytest.mark.parametrize(
        "pair",
        [(true_ext, true_ext), (true_simpl, true_ext), (true_simpl, true_unary)],
    )
    def test_zero_dissimilarity(self, pair):
        dis = tscompare.compare(pair[0], pair[1])
        assert np.isclose(dis.dissimilarity, 0)
        assert np.isclose(dis.arf, 0)
        assert np.isclose(dis.rmse, 0)

    def test_transform(self):
        dis1 = tscompare.compare(true_simpl, true_simpl, transform=lambda t: t)
        dis2 = tscompare.compare(true_simpl, true_simpl, transform=None)
        assert dis1.dissimilarity == dis2.dissimilarity
        assert dis1.rmse == dis2.rmse
        self.verify_compare(true_simpl, true_ext, transform=lambda t: 1 / (1 + t))

    def get_simple_ts(self, samples=None, time=False, span=False, no_match=False):
        # A simple tree sequence we can use to properly test various
        # dissimilarity and MSRE values.
        #
        #    6          6      6
        #  +-+-+      +-+-+  +-+-+
        #  |   |      7   |  |   8
        #  |   |     ++-+ |  | +-++
        #  4   5     4  | 5  4 |  5
        # +++ +++   +++ | |  | | +++
        # 0 1 2 3   0 1 2 3  0 1 2 3
        #
        # if time = False:
        # with node times 0.0, 500.0, 750.0, 1000.0 for each tier,
        # else:
        # with node times 0.0, 200.0, 600.0, 1000.0 for each tier,
        #
        # if span = False:
        # each tree spans (0,2), (2,4), and (4,6) respectively.
        # else:
        # each tree spans (0,1), (1,5), and (5,6) repectively.
        if time is False:
            node_times = {
                0: 0,
                1: 0,
                2: 0,
                3: 0,
                4: 500.0,
                5: 500.0,
                6: 1000.0,
                7: 750.0,
                8: 750.0,
            }
        else:
            node_times = {
                0: 0,
                1: 0,
                2: 0,
                3: 0,
                4: 200.0,
                5: 200.0,
                6: 1000.0,
                7: 600.0,
                8: 600.0,
            }
        # (p, c, l, r)
        if span is False:
            edges = [
                (4, 0, 0, 6),
                (4, 1, 0, 4),
                (5, 2, 0, 2),
                (5, 2, 4, 6),
                (5, 3, 0, 6),
                (7, 2, 2, 4),
                (7, 4, 2, 4),
                (8, 1, 4, 6),
                (8, 5, 4, 6),
                (6, 4, 0, 2),
                (6, 4, 4, 6),
                (6, 5, 0, 4),
                (6, 7, 2, 4),
                (6, 8, 4, 6),
            ]
        else:
            edges = [
                (4, 0, 0, 6),
                (4, 1, 0, 5),
                (5, 2, 0, 1),
                (5, 2, 5, 6),
                (5, 3, 0, 6),
                (7, 2, 1, 5),
                (7, 4, 1, 5),
                (8, 1, 5, 6),
                (8, 5, 5, 6),
                (6, 4, 0, 1),
                (6, 4, 5, 6),
                (6, 5, 0, 5),
                (6, 7, 1, 5),
                (6, 8, 5, 6),
            ]
        if no_match is True:
            node_times[9] = 100.0
            if span is False:
                edges = [
                    (9, 0, 4, 6),
                    (4, 0, 0, 4),
                    (4, 1, 0, 6),
                    (4, 9, 4, 6),
                    (5, 2, 0, 2),
                    (5, 2, 4, 6),
                    (5, 3, 0, 6),
                    (7, 2, 2, 4),
                    (7, 4, 2, 4),
                    (6, 4, 0, 2),
                    (6, 4, 4, 6),
                    (6, 5, 0, 6),
                    (6, 7, 2, 4),
                ]
            else:
                edges = [
                    (9, 0, 5, 6),
                    (4, 0, 0, 5),
                    (4, 1, 0, 6),
                    (4, 9, 5, 6),
                    (5, 2, 0, 2),
                    (5, 2, 5, 6),
                    (5, 3, 0, 6),
                    (7, 2, 2, 5),
                    (7, 4, 2, 5),
                    (6, 4, 0, 2),
                    (6, 4, 5, 6),
                    (6, 5, 0, 6),
                    (6, 7, 2, 5),
                ]
        tables = tskit.TableCollection(sequence_length=6)
        if samples is None:
            samples = [0, 1, 2, 3]
        for (
            n,
            t,
        ) in node_times.items():
            flags = tskit.NODE_IS_SAMPLE if n in samples else 0
            tables.nodes.add_row(time=t, flags=flags)
        for p, c, l, r in edges:
            tables.edges.add_row(parent=p, child=c, left=l, right=r)
        ts = tables.tree_sequence()
        if no_match is True:
            assert ts.num_edges == 13
        if no_match is False:
            assert ts.num_edges == 14
        return ts

    def test_with_no_match(self):
        ts = self.get_simple_ts()
        other = self.get_simple_ts(span=True, time=True, no_match=True)
        self.verify_compare(ts, other)
        self.verify_compare(ts, other, transform=lambda t: np.sqrt(1 + t))

    def test_dissimilarity_value(self):
        ts = self.get_simple_ts()
        other = self.get_simple_ts(span=True)
        dis = tscompare.compare(ts, other)
        assert np.isclose(dis.arf, 4 / 46)
        assert np.isclose(dis.rmse, 0.0)

    def test_rmse(self):
        ts = self.get_simple_ts()
        other = self.get_simple_ts(time=True)
        dis = tscompare.compare(ts, other)
        true_total_span = 46
        assert dis.total_span[0] == true_total_span
        assert dis.total_span[1] == true_total_span

        def f(t):
            return np.log(1 + t)

        true_rmse = np.sqrt(
            (
                2 * 6 * (f(500) - f(200)) ** 2  # nodes 4, 5
                + 2 * 2 * (f(750) - f(600)) ** 2  # nodes, 7, 8
            )
            / true_total_span
        )
        assert np.isclose(dis.arf, 0.0)
        assert np.isclose(dis.tpr, 1.0)
        assert np.isclose(dis.dissimilarity, 0.0)
        assert np.isclose(dis.rmse, true_rmse)

    def test_value_and_error(self):
        ts = self.get_simple_ts()
        other = self.get_simple_ts(span=True, time=True)
        dis = tscompare.compare(ts, other)
        true_total_spans = (46, 47)
        assert dis.total_span == true_total_spans

        def f(t):
            return np.log(1 + t)

        true_rmse = np.sqrt(
            (
                2 * 6 * (f(500) - f(200)) ** 2  # nodes 4, 5
                + 2 * 2 * (f(750) - f(600)) ** 2  # nodes, 7, 8
            )
            / true_total_spans[0]
        )
        assert np.isclose(dis.arf, 4 / true_total_spans[0])
        assert np.isclose(dis.tpr, (true_total_spans[0] - 4) / true_total_spans[1])
        assert np.isclose(dis.dissimilarity, 4)
        assert np.isclose(dis.rmse, true_rmse)
