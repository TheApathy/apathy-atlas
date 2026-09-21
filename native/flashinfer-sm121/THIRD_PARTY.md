# Third-party notices

This isolated native boundary instantiates APIs and templates from
FlashInfer Python 0.6.6. FlashInfer source and generated instantiation form are
licensed under Apache-2.0; the full license is retained in
`LICENSES/Apache-2.0.txt`.

FlashInfer bundles NVIDIA CUTLASS headers licensed under BSD-3-Clause. The
copyright, SPDX identifier, redistribution conditions, and disclaimer copied
from the pinned `cutlass/cutlass.h` are retained in
`LICENSES/BSD-3-Clause.txt`.

`DEPENDENCIES.sha256` pins the directly consumed FlashInfer template/config
headers and a CUTLASS license/header anchor. It is not a complete hash manifest
of CUTLASS's transitive include tree. A distributable vendored dependency must
retain upstream notices and bind the complete included tree.
