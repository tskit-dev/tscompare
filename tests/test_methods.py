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


def naive_node_span(ts, include_missing=False):
    """
    Ineffiecient but transparent function to get total span
    of each node in a tree sequence, including roots.
    """
    node_spans = np.zeros(ts.num_nodes)
    for t in ts.trees():
        for n in t.nodes():
            in_tree = (t.parent(n) != tskit.NULL or t.num_children(n) > 0) or (
                include_missing and t.is_sample(n)
            )
            if in_tree:
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

    samples = ts.samples()
    node_span_ts = naive_node_span(ts)
    total_span_ts = np.sum(node_span_ts)
    shared_spans = naive_shared_node_spans(ts, other).toarray()
    for i in samples:
        for j in range(other.num_nodes):
            if j != i:
                shared_spans[i, j] = 0.0
    row_max_span = np.max(shared_spans, axis=1)
    total_match_span_ts = np.sum(row_max_span)
    assert len(row_max_span) == ts.num_nodes
    # this matrix has |dft| for every shared span entry that's equal to its
    # row max, and zeros otherwise
    time_diff_matrix = np.full(shared_spans.shape, np.inf)
    for i in range(shared_spans.shape[0]):
        rm = row_max_span[i]
        if rm > 0:
            for j in range(shared_spans.shape[1]):
                if shared_spans[i, j] == rm:
                    dft = np.abs(
                        transform(ts.nodes_time[i]) - transform(other.nodes_time[j])
                    )
                    time_diff_matrix[i, j] = dft
    best_match_ts = np.argmin(time_diff_matrix, axis=1)
    time_diffs = np.min(time_diff_matrix, axis=1)
    fd = np.isfinite(time_diffs)
    rmse = np.sqrt(
        np.sum(node_span_ts[fd] * (time_diffs[fd] ** 2)) / np.sum(node_span_ts[fd])
    )

    # this matrix has in each row the span of its best match and zeros otherwise
    best_match_matrix = np.zeros(shared_spans.shape)
    for i in range(shared_spans.shape[0]):
        rm = row_max_span[i]
        j = best_match_ts[i]
        assert shared_spans[i, j] == rm, f"{i}, {j}: {rm}"
        best_match_matrix[i, j] = rm
    best_match_other_span = np.max(best_match_matrix, axis=0)
    total_match_span_other = np.sum(best_match_other_span)

    total_span_other = np.sum(naive_node_span(other))
    return (
        total_match_span_ts,
        total_match_span_other,
        total_span_ts,
        total_span_other,
        rmse,
    )


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
    @pytest.mark.parametrize("include_missing", [True, False])
    def test_node_spans(self, ts, include_missing):
        eval_ns = tscompare.node_spans(ts, include_missing=include_missing)
        naive_ns = naive_node_span(ts, include_missing=include_missing)
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

    def test_isolated_samples(self):
        # ts as in test_very_simple; empty_ts without 2->1 branch:
        #
        # 1.00┊ 2   ┊
        #     ┊ ┃   ┊
        # 0.00┊ 0 1 ┊
        #     0     1
        ts = tskit.Tree.generate_star(2).tree_sequence
        tables = ts.tables
        tables.edges.clear()
        tables.edges.add_row(parent=2, child=0, left=0, right=1)
        empty_ts = tables.tree_sequence()
        self.test_node_spans(empty_ts, include_missing=True)
        self.test_node_spans(empty_ts, include_missing=False)
        node_spans_no_missing = tscompare.node_spans(empty_ts, include_missing=False)
        true_spans_no_missing = np.array([1.0, 0.0, 1.0])
        assert np.all(np.isclose(node_spans_no_missing, true_spans_no_missing))
        node_spans_missing = tscompare.node_spans(empty_ts, include_missing=True)
        true_spans_missing = np.full((3,), 1.0)
        assert np.all(np.isclose(node_spans_missing, true_spans_missing))


class TestDissimilarity:

    def verify_compare(self, ts, other, transform=None):
        match_n1_span, match_n2_span, ts_span, other_span, rmse = naive_compare(
            ts,
            other,
            transform=transform,
        )
        dis = tscompare.compare(ts, other, transform=transform)
        assert np.isclose(1.0 - match_n1_span / ts_span, dis.arf)
        assert np.isclose(match_n2_span / other_span, dis.tpr)
        assert np.isclose(ts_span - match_n1_span, dis.dissimilarity)
        assert np.isclose(other_span - match_n2_span, dis.inverse_dissimilarity)
        assert np.isclose(ts_span, dis.total_span[0])
        assert np.isclose(other_span, dis.total_span[1])
        assert np.isclose(rmse, dis.rmse), f"{rmse} != {dis.rmse}"

    def test_samples_dont_match(self):
        ts1 = tskit.Tree.generate_star(2).tree_sequence
        ts2 = tskit.Tree.generate_star(3).tree_sequence
        with pytest.raises(ValueError, match="Samples.*agree"):
            tscompare.compare(ts1, ts2)
        tables = ts1.dump_tables()
        tables.nodes.clear()
        tables.nodes.add_row(time=0, flags=tskit.NODE_IS_SAMPLE)
        tables.nodes.add_row(time=0, flags=0)
        tables.nodes.add_row(time=1, flags=tskit.NODE_IS_SAMPLE)
        ts2 = tables.tree_sequence()
        with pytest.raises(ValueError, match="Samples.*agree"):
            tscompare.compare(ts1, ts2)

    def test_very_simple(self):
        # 1.00┊  2  ┊
        #     ┊ ┏┻┓ ┊
        # 0.00┊ 0 1 ┊
        #     0     1
        ts = tskit.Tree.generate_star(2).tree_sequence
        dis = tscompare.compare(ts, ts)
        assert dis.arf == 0.0
        assert dis.tpr == 1.0
        assert dis.dissimilarity == 0.0
        assert dis.inverse_dissimilarity == 0.0
        assert dis.total_span == (3.0, 3.0)
        assert dis.rmse == 0.0

    def test_missing_data(self):
        # ts as in test_very_simple; empty_ts without 2->1 branch:
        #
        # 1.00┊ 2   ┊
        #     ┊ ┃   ┊
        # 0.00┊ 0 1 ┊
        #     0     1
        ts = tskit.Tree.generate_star(2).tree_sequence
        tables = ts.tables
        tables.edges.clear()
        tables.edges.add_row(parent=2, child=0, left=0, right=1)
        empty_ts = tables.tree_sequence()
        dis = tscompare.compare(ts, empty_ts)
        assert np.isclose(dis.arf, 1 / 3)
        assert np.isclose(dis.tpr, 2 / 3)
        assert dis.dissimilarity == 1.0
        assert dis.inverse_dissimilarity == 1.0
        assert dis.total_span == (3.0, 3.0)
        assert dis.rmse == 0.0
        # note that here both 0 and 2 in empty_ts map to 0 in ts!
        dis = tscompare.compare(empty_ts, ts)
        assert dis.arf == 0.0
        assert np.isclose(dis.tpr, 2 / 3)
        assert dis.dissimilarity == 0.0
        assert dis.inverse_dissimilarity == 1.0
        assert dis.total_span == (3.0, 3.0)
        # here 0->0 (dt=0), 1->1 (dt=0), and (2->0) (dt=1)
        rmse = np.sqrt((1 / 3) * (0 + 0 + np.log(1 + 1) ** 2))
        assert np.isclose(dis.rmse, rmse)

    def test_no_matches(self):
        # This is an example where no nodes match any other nodes,
        # since both are samples:
        # 1.00┊ 1 ┊         1.00┊ 0 ┊
        #     ┊ ┃ ┊   and       ┊ ┃ ┊
        # 0.00┊ 0 ┊         0.00┊ 1 ┊
        #     0   1             0   1
        # Note that if samples weren't required to match samples,
        # then 1 in the first matches 0 in the second (both are
        # ancestral to {0, 1})
        tables = tskit.TableCollection(sequence_length=1.0)
        c = tables.nodes.add_row(time=0, flags=1)
        p = tables.nodes.add_row(time=1, flags=1)
        tables.edges.add_row(parent=p, child=c, left=0, right=1)
        ts1 = tables.tree_sequence()
        tables.clear()
        p = tables.nodes.add_row(time=1, flags=1)
        c = tables.nodes.add_row(time=0, flags=1)
        tables.edges.add_row(parent=p, child=c, left=0, right=1)
        ts2 = tables.tree_sequence()
        dis = tscompare.compare(ts1, ts2)
        assert dis.arf == 1.0
        assert dis.tpr == 0.0
        assert dis.dissimilarity == 2.0
        assert dis.inverse_dissimilarity == 2.0
        assert dis.total_span == (2.0, 2.0)
        assert np.isnan(dis.rmse)

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

    @pytest.mark.parametrize(
        "pair",
        [(true_ext, true_ext), (true_simpl, true_unary)],
    )
    def test_inverse_dissimilarity(self, pair):
        dis = tscompare.compare(pair[1], pair[0])
        assert np.isclose(dis.tpr, 1)
        assert np.isclose(dis.inverse_dissimilarity, 0)

    def test_transform(self):
        dis1 = tscompare.compare(true_simpl, true_simpl, transform=lambda t: t)
        dis2 = tscompare.compare(true_simpl, true_simpl, transform=None)
        assert dis1.dissimilarity == dis2.dissimilarity
        assert dis1.rmse == dis2.rmse
        self.verify_compare(true_simpl, true_ext, transform=lambda t: 1 / (1 + t))

    def get_simple_ts(
        self, samples=None, time=False, span=False, no_match=False, extra_match=False
    ):
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
        if extra_match is True:
            node_times[9] = 525.0
            node_times[10] = 550.0
            if span is False:
                edges = [
                    (4, 0, 0, 6),
                    (4, 1, 0, 4),
                    (5, 2, 0, 2),
                    (5, 2, 4, 6),
                    (5, 3, 0, 6),
                    (9, 4, 0, 4),
                    (10, 9, 0, 4),
                    (7, 2, 2, 4),
                    (7, 10, 2, 4),
                    (8, 1, 4, 6),
                    (8, 5, 4, 6),
                    (6, 4, 4, 6),
                    (6, 5, 0, 4),
                    (6, 7, 2, 4),
                    (6, 8, 4, 6),
                    (6, 10, 0, 2),
                ]
            else:
                edges = [
                    (4, 0, 0, 6),
                    (4, 1, 0, 5),
                    (5, 2, 0, 1),
                    (5, 2, 5, 6),
                    (5, 3, 0, 6),
                    (9, 4, 0, 5),
                    (10, 9, 0, 5),
                    (7, 2, 1, 5),
                    (7, 10, 1, 5),
                    (8, 1, 5, 6),
                    (8, 5, 5, 6),
                    (6, 4, 5, 6),
                    (6, 5, 0, 5),
                    (6, 7, 1, 5),
                    (6, 8, 5, 6),
                    (6, 10, 0, 1),
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
        if extra_match is True:
            assert ts.num_edges == 16
        if no_match is False and extra_match is False:
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
        dis = tscompare.compare(ts, other, transform=None)
        assert np.isclose(dis.arf, 4 / 46)
        assert np.isclose(dis.rmse, 0.0)

    def test_rmse(self):
        ts = self.get_simple_ts()
        other = self.get_simple_ts(time=True)
        dis = tscompare.compare(ts, other, transform=None)
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
        dis = tscompare.compare(ts, other, transform=None)
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
        assert np.isclose(dis.tpr, (true_total_spans[1] - 5) / true_total_spans[1])
        assert np.isclose(dis.dissimilarity, 4)
        assert np.isclose(dis.inverse_dissimilarity, 5)
        assert np.isclose(dis.rmse, true_rmse)

    def test_extra_match(self):
        ts = self.get_simple_ts(extra_match=True)
        other = self.get_simple_ts()
        dis = tscompare.compare(ts, other, transform=None)
        n1_match_span = 7 * 6 + 2 * 2 + 2 * 4
        n2_match_span = 6 * 6 + 2 * 2 + 6
        true_spans = (54, 46)
        assert np.isclose(dis.arf, 1 - n1_match_span / true_spans[0])
        assert np.isclose(dis.tpr, n2_match_span / true_spans[1])
        assert np.isclose(dis.dissimilarity, true_spans[0] - n1_match_span)
        assert np.isclose(dis.inverse_dissimilarity, true_spans[1] - n2_match_span)

    def get_n2_match_ex(self, samples=None, extra_nodes=False):
        node_times = {
            0: 0.0,
            1: 0.0,
            2: 0.0,
            3: 100.0,
            4: 200.0,
        }
        if extra_nodes:
            node_times[5] = 300.0
            # (p, c, l, r) ordered by p[time]
            edges = [
                (3, 0, 0, 3),
                (3, 1, 0, 3),
                (4, 3, 0, 3),
                (5, 2, 0, 3),
                (5, 4, 0, 3),
            ]
        else:
            edges = [(3, 0, 0, 3), (3, 1, 0, 3), (4, 2, 0, 3), (4, 3, 0, 3)]
        tables = tskit.TableCollection(sequence_length=3)
        if samples is None:
            samples = [0, 1, 2]
        for (
            n,
            t,
        ) in node_times.items():
            flags = tskit.NODE_IS_SAMPLE if n in samples else 0
            tables.nodes.add_row(time=t, flags=flags)
        for p, c, l, r in edges:
            tables.edges.add_row(parent=p, child=c, left=l, right=r)
        ts = tables.tree_sequence()
        if extra_nodes is True:
            assert ts.num_edges == 5
        if extra_nodes is False:
            assert ts.num_edges == 4
        return ts

    def test_n2_matching(self):
        ts = self.get_n2_match_ex()
        other = self.get_n2_match_ex(extra_nodes=True)
        dis = tscompare.compare(ts, other, transform=None)
        true_spans = (15, 18)
        match_spans = (15, 15)
        assert np.isclose(dis.arf, 1 - match_spans[0] / true_spans[0])
        assert np.isclose(dis.tpr, match_spans[1] / true_spans[1])
        assert np.isclose(dis.dissimilarity, true_spans[0] - match_spans[0])
        assert np.isclose(dis.inverse_dissimilarity, true_spans[1] - match_spans[1])
        self.verify_compare(ts, other)

    def test_n2_time_match(self):
        def ex(samples=None, extra_node=False):
            node_times = {
                0: 0.0,
                1: 0.0,
                2: 0.0,
            }
            if extra_node:
                node_times[3] = 100.0
                node_times[4] = 200.0
                node_times[5] = 300.0
                # (p, c, l, r)
                edges = [
                    (3, 0, 0, 7),
                    (3, 1, 0, 7),
                    (4, 2, 0, 1),
                    (4, 3, 0, 7),
                    (5, 2, 1, 7),
                    (5, 4, 1, 7),
                ]
            else:
                node_times[3] = 50.0
                node_times[4] = 200.0
                edges = [
                    (3, 0, 0, 7),
                    (3, 1, 0, 7),
                    (3, 2, 3, 7),
                    (4, 2, 0, 3),
                    (4, 3, 0, 7),
                ]
            tables = tskit.TableCollection(sequence_length=7)
            if samples is None:
                samples = [0, 1, 2]
            for (
                n,
                t,
            ) in node_times.items():
                flags = tskit.NODE_IS_SAMPLE if n in samples else 0
                tables.nodes.add_row(time=t, flags=flags)
            for p, c, l, r in edges:
                tables.edges.add_row(parent=p, child=c, left=l, right=r)
            tables.sort()
            ts = tables.tree_sequence()
            if extra_node is True:
                assert ts.num_edges == 6
            if extra_node is False:
                assert ts.num_edges == 5
            return ts

        ts = ex(extra_node=True)
        other = ex(extra_node=False)
        self.verify_compare(ts, other)
