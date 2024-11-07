# MIT License
#
# Copyright (c) 2021-23 Tskit Developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Tools for comparing node times between tree sequences with different node sets
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse
import tsdate

@dataclass
class DissimilarityResult:
    """
    The result of a call to tscompare.dissimilarity(ts, other).
    """
    dissimilarity: float
    """
    The proportion of the total span of ts that is not represented in other.
    """
    prop_reprsented: float
    """
    The proportion of the total span of other that is represented in ts.
    """
    rmse: float
    """
    The root-mean-squared error between the transformed times of the nodes in
    ts and the transformed times of their best-matching nodes in other, with
    the average taken weighting by span
    """
    transform: callable
    """
    The transformation function used to transform times.
    """


def dissimilarity(ts, other, transform=None):
    """
    For two tree sequences `ts` and `other`,
    this method returns three values:
    1. The fraction of the total span of `ts` over which each nodes' descendant
    sample set does not match its' best match's descendant sample set.
    2. The root mean squared difference
    between the transformed times of the nodes in `ts`
    and transformed times of their best matching nodes in `other`,
    with the average weighted by the nodes' spans in `ts`.
    3. The proportion of the span in `other` that is correctly
    represented in `ts` (i.e., the total matching span divided
    by the total span of `other`).

    The transformation for times is by default log(1 + t).

    This is done as follows:

    For each node in `ts`, the best matching node(s) from `other`
    has the longest matching span using `shared_node_spans`.
    If there are multiple matches with the same longest shared span
    for a single node, the best match is the match that is closest in time.

    :param ts: The focal tree sequence.
    :param other: The tree sequence we check for inclusion in.
    :param transform: A callable that can take an array of times and
        return another array of numbers.
    :return: The three quantities above.
    :rtype: DissimilarityResult
    """

    if transform is None:
        transform = lambda t: np.log(1 + t)

    shared_spans = tsdate.shared_node_spans(ts, other)
    # Find all potential matches for a node based on max shared span length
    max_span = shared_spans.max(axis=1).toarray().flatten()
    col_ind = shared_spans.indices
    row_ind = np.repeat(
        np.arange(shared_spans.shape[0]), repeats=np.diff(shared_spans.indptr)
    )
    # mask to find all potential node matches
    match = shared_spans.data == max_span[row_ind]
    # scale with difference in node times
    # determine best matches with the best_match_matrix
    ts_times = ts.nodes_time[row_ind[match]]
    other_times = other.nodes_time[col_ind[match]]
    time_difference = np.absolute(np.asarray(transform(ts_times) - transform(other_times)))
    # If a node x in `ts` has no match then we set time_difference to zero
    # This node then does not effect the rmse
    for j in range(len(shared_spans.data[match])):
        if shared_spans.data[match][j] == 0:
            time_difference[j] = 0.0
    # If two nodes have the same time, then
    # time_difference is zero, which causes problems with argmin
    # Instead we store data as 1/(1+x) and find argmax
    best_match_matrix = scipy.sparse.coo_matrix(
        (
            1 / (1 + time_difference),
            (row_ind[match], col_ind[match]),
        ),
        shape=(ts.num_nodes, other.num_nodes),
    )
    # Between each pair of nodes, find the maximum shared span
    best_match = best_match_matrix.argmax(axis=1).A1
    best_match_spans = shared_spans[np.arange(len(best_match)), best_match].reshape(-1)
    ts_node_spans = node_spans(ts)
    total_node_spans_ts = np.sum(ts_node_spans)
    total_node_spans_other = np.sum(node_spans(other))
    dissimilarity = 1 - np.sum(best_match_spans) / total_node_spans_ts
    true_proportion = (1 - dissimilarity) * total_node_spans_ts / total_node_spans_other
    # Compute the root-mean-square difference in transformed time
    # with the average weighted by span in ts
    time_matrix = scipy.sparse.csr_matrix(
        (time_difference, (row_ind[match], col_ind[match])),
        shape=(ts.num_nodes, other.num_nodes),
    )
    time_discrepancies = np.asarray(
        time_matrix[np.arange(len(best_match)), best_match].reshape(-1)
    )
    product = np.multiply((time_discrepancies**2), ts_node_spans)
    rmse = np.sqrt(np.sum(product) / total_node_spans_ts)
    return DissimilarityResult(
            dissimliarty=dissimilarity,
            prop_represented=true_proportion,
            rmse=rmse,
            transform=transform,
    )

