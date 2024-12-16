---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.9.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

```{code-cell}
:tags: [remove-cell]

import warnings
import tskit, msprime, tsinfer
from IPython.display import SVG
import numpy as np
import tscompare

np.random.seed(1234)
```

```{eval-rst}
.. currentmodule:: tscompare
```


(sec_usage)=

# Usage

(sec_quickstart)=

## Quickstart

To set up an example, let's (1) simulate a tree sequence with {program}`msprime`; and
(2) infer a tree sequence from the resulting genetic variation data with {program}`tsinfer`.
```{code-cell}
orig_ts = msprime.sim_ancestry(
            100, recombination_rate=1e-8, population_size=1e3,
            sequence_length=1e6, record_full_arg=True,
            random_seed=123)
orig_ts = msprime.sim_mutations(orig_ts, rate=1e-8, random_seed=456)
vdata = tsinfer.SampleData.from_tree_sequence(orig_ts)
inferred_ts = tsinfer.infer(vdata)
```

Now, we can ask: how much of the inferred tree sequence is not "correct":
in other words, how much of it is not represented in the true tree sequence?
(Here, part of an ancestral node's span is "correct" if it is ancestral
to the same set of samples.)
We do this with {func}`.compare`:
```{code-cell}
dis = tscompare.compare(orig_ts, inferred_ts)
print(dis)
```
