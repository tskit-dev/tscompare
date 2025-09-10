# Changelog

## [0.2.1] - 2025-XX-XX

- Support Python3.10-3.13 ({pr}`24`)

## [0.2] - 2025-09-09

Final release to go with publication of Fritze et al.

**Breaking change:** renamed `compare` to `haplotype_arf`, because there are other comparison
methods that we might implement here, and each would return a different object.
For now, `compare` does the same thing but raises a DeprecationWarning.

## [0.1] - 2024-12-14

Initial release of functionality, providing tools to "match" nodes between tree sequences
in a similar way as Robinson-Foulds but haplotype-aware (hence, "ARG Robinson-Foulds" or ARF):
`shared_node_spans`, `match_node_ages`, `compare`
({user}`hfr1tz3`, {user}`nspope`, {user}`petrelharp`)

