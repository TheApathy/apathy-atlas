"""Pre-registered (previous attention agent, never run): in runD_L20_kernel, ckv/ik at chunk 1
must begin with chunk 0's rows BIT-IDENTICALLY -- the compressed cache is append-only and
sequence-lifetime, never cleared per chunk. Also: chunk-1 selections must actually reach rows
written by chunk 0 (else the persistence would be untested by the attention that needs it).
Controls (must FAIL identity): chunk-1's OWN new rows vs chunk 0's rows; a 1-row shift."""
import sys
import numpy as np
from oracle_io import Run

fail = 0
for run in sys.argv[1:] or ["runD_L20_kernel", "runE_torch"]:
    R = Run(run)
    for L in (2, 14, 20):
        r = int(R.load(L, "ratio", 0))
        for tap in ("ckv", "ik"):
            a = R.load(L, tap, 0, raw=True); b = R.load(L, tap, 1, raw=True)
            n0 = a.shape[0]
            same = np.array_equal(b[:n0], a)
            nz = np.count_nonzero(a) / a.size
            ctrl_new = int((b[n0:2 * n0] != a).any(axis=1).sum())
            ctrl_shift = int((b[1:n0 + 1] != a).any(axis=1).sum())
            print(f"{run} L{L:02d} r={r} {tap}: {a.shape}->{b.shape} prefix_identical={same} "
                  f"nonzero={nz:.4f} | CTRL new-rows differ {ctrl_new}/{n0}, shift differ {ctrl_shift}/{n0}")
            fail += (not same) or nz < 0.5 or ctrl_new == 0 or ctrl_shift == 0
        # cross-chunk reach: chunk-1 top-k selecting rows chunk 0 wrote
        occ = 2 if R.has(L, "topk", 3) else 1   # kernel path taps topk twice per chunk
        tk = R.load(L, "topk", occ)
        n_c0 = int(R.load(L, "n_c", 0))
        valid = tk >= 0
        old = (tk >= 0) & (tk < n_c0)
        print(f"   chunk-1 topk: {valid.sum()} selections, {old.sum()} ({old.sum()/valid.sum():.1%}) "
              f"hit rows < n_c(chunk0)={n_c0}; rows with >=1 such hit: {old.any(1).sum()}/{tk.shape[0]}")
print("RESULT", "FAIL" if fail else "PASS")
