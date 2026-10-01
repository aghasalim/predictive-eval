#!/usr/bin/env bash
# measurement-db is gated: accept its terms on Hugging Face and log in first
# (hf auth login). The data stays in mdb/ and is not part of this repository.
set -euo pipefail
hf download aims-foundations/measurement-db --repo-type dataset --local-dir mdb
# The empirical mean baseline is the organisers' own code.
[ -d paiec_baseline ] || git clone --depth 1 https://github.com/aims-foundations/paiec_baseline.git
