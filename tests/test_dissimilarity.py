# MIT License
#
# Copyright (c) 2021-23 Tskit Developers
# Copyright (c) 2020-21 University of Oxford
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
    sequences subtend the same sample set (also found in tsdate/tests)
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


def naive_dissimilarity(ts, other, transform=None):
    """
    Ineffiecient but transparent function to compute dissimilarity
    and root-mean-square-error between two tree sequences.
    """
    def f(t):
        return np.log(1 + t)
    if transform is not None:
        f = transform
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
                    time_array[i, j] = np.abs(f(ts.nodes_time[i]) - f(other.nodes_time[j]))
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
    dissimilarity = 1 - np.sum(best_match_spans) / total_node_spans
    true_prop = np.sum(best_match_spans) / total_other_spans
    rmse = np.sqrt(np.sum(node_span * time_discrepancies**2) / total_node_spans)
    return dissimilarity, true_prop, rmse


class TestNodeMatching:

    @pytest.mark.parametrize(
        "pair",
        [(true_ext, true_ext), (true_simpl, true_ext), (true_simpl, true_unary)],
    )
    def test_basic_dissimilarity(self, pair):
        """
        Check that efficient implementation reutrns the same answer as naive
        implementation.
        """
        check_dis, check_prop, check_rmse = naive_dissimilarity(pair[0], pair[1])
        dis = tscompare.dissimilarity(pair[0], pair[1])
        assert np.isclose(check_dis, dis.dissimilarity)
        assert np.isclose(check_prop, dis.prop_represented)
        assert np.isclose(check_rmse, dis.rmse)

    @pytest.mark.parametrize(
        "pair",
        [(true_ext, true_ext), (true_simpl, true_ext), (true_simpl, true_unary)],
    )
    def test_zero_dissimilarity(self, pair):
        dis = tscompare.dissimilarity(pair[0], pair[1])
        assert np.isclose(dis.dissimilarity, 0)
        assert dis.prop_represented >= 0
        assert np.isclose(dis.rmse, 0)

    def test_transform(self):
        dis = tscompare.dissimilarity(true_simpl, true_simpl, transform=lambda x: x)
        assert np.isclose(dis.rmse, 0)

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

    def test_dissimilarity_value(self):
        ts = self.get_simple_ts()
        other = self.get_simple_ts(span=True)
        dis = tscompare.dissimilarity(ts, other)
        assert np.isclose(dis.dissimilarity, 4 / 46)
        assert np.isclose(dis.rmse, 0.0)

    def test_dissimilarity_rmse(self):
        ts = self.get_simple_ts()
        other = self.get_simple_ts(time=True)
        dis = tscompare.dissimilarity(ts, other)
        true_error = np.sqrt((2 * 6 * 300**2 + 2 * 2 * 150**2) / 46)
        assert np.isclose(dis.dissimilarity, 0.0)
        assert np.isclose(dis.rmse, true_error)

    def test_dissimilarity_value_and_error(self):
        ts = self.get_simple_ts()
        other = self.get_simple_ts(span=True, time=True)
        dis = tscompare.dissimilarity(ts, other)
        true_error = np.sqrt((2 * 6 * 300**2 + 2 * 2 * 150**2) / 46)
        assert np.isclose(dis.dissimilarity, 4 / 46)
        assert np.isclose(dis.rmse, true_error)

    def test_dissimilarity_and_naive_dissimilarity_with_no_match(self):
        ts = self.get_simple_ts()
        other = self.get_simple_ts(span=True, time=True, no_match=True)
        check_dis, check_prop, check_rmse = naive_dissimilarity(ts, other)
        dis = tscompare.dissimilarity(ts, other)
        assert np.isclose(check_dis, dis.dissimilarity)
        assert np.isclose(check_prop, dis.prop_represented)
        assert np.isclose(check_rmse, dis.rmse)
