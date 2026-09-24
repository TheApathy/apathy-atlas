#!/usr/bin/env python3
"""make_evidence.py <gate_run_dir> <binary> <build_commit> -> evidence.json (stdout: the Rust const)

Assembles the target-only admission record from the scored arms of one gate run:
prefill/gate.json, decode/gate.json, control256/gate.json and control-decode/gate.json.
Refuses unless prefill and decode pass and both controls fail. The Rust constant
it prints must be pasted into target_only_admission.rs; the unit test there
requires the two to match field for field.
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FIELDS = ("prompts", "positions", "argmax_agreement_excl_ties", "mean_kl", "top1_delta_abs", "nll_delta_rel")


def metrics(run, arm):
    g = json.load(open(os.path.join(run, arm, "gate.json")))
    return g, {k: g["metrics"].get(k) for k in FIELDS}


def rust(m):
    opt = lambda v: "None" if v is None else f"Some({v!r})"
    return (f"Glm53GateMetrics {{ prompts: {m['prompts']}, positions: {m['positions']}, "
            f"argmax_agreement_excl_ties: {m['argmax_agreement_excl_ties']!r}, mean_kl: {m['mean_kl']!r}, "
            f"top1_delta_abs: {opt(m['top1_delta_abs'])}, nll_delta_rel: {opt(m['nll_delta_rel'])} }}")


def main():
    run, binary, commit = sys.argv[1:4]
    gp, prefill = metrics(run, "prefill")
    gd, decode = metrics(run, "decode")
    gc, control = metrics(run, "control256")
    gcd, control_decode = metrics(run, "control-decode")
    if gp["fails"] or gd["fails"]:
        sys.exit(f"refusing: prefill fails={gp['fails']} decode fails={gd['fails']}")
    if not gc["fails"] or not gcd["fails"]:
        sys.exit("refusing: a negative control PASSED; the gate cannot fail")
    for arm in ("prefill", "decode", "control256", "control-decode"):
        r = json.load(open(os.path.join(run, arm, "run.json")))
        if not r.get("environ_matches_launch"):
            sys.exit(f"refusing: {arm} server environ did not match its recipe launch")
    sha = hashlib.sha256(open(binary, "rb").read()).hexdigest()
    prompts_sha = hashlib.sha256(open(os.path.join(HERE, "prompts_1024.json"), "rb").read()).hexdigest()
    reference = "exllamav3 e648f1a1 glm5_next, streaming uncached"
    record = {
        "status": "pass",
        "build_commit": commit,
        "binary_sha256": sha,
        "reference": reference,
        "prompts_sha256": prompts_sha,
        "bounds": {"min_argmax_agreement_excl_ties": 0.90, "max_mean_kl": 0.05,
                   "max_top1_delta_abs": 0.01, "max_nll_delta_rel": 0.02, "min_prompts": 3},
        "prefill": prefill,
        "decode": decode,
        "control": control,
        "control_arm": "ATLAS_GLM53_NEGATIVE_CONTROL=skip-kda-commit, 256-row prefill chunks",
        "control_decode": control_decode,
        "control_fails": gc["fails"],
        "control_decode_fails": gcd["fails"],
        "secondary_views": {"prefill": gp.get("views"), "decode": gd.get("views")},
        "reference_self_disagreement_1024": {
            "note": "same inputs, two reference runs: the reference's own noise floor",
            "argmax_agreement_excl_ties": 0.96977, "mean_kl": 0.01068},
    }
    json.dump(record, open(os.path.join(HERE, "evidence.json"), "w"), indent=2)
    print(f"""pub(crate) const GLM53_EXL3_TARGET_ONLY_EVIDENCE: Option<Glm53TargetOnlyEvidence> =
    Some(Glm53TargetOnlyEvidence {{
        build_commit: "{commit}",
        binary_sha256: "{sha}",
        reference: "{reference}",
        prompts_sha256: "{prompts_sha}",
        prefill: {rust(prefill)},
        decode: {rust(decode)},
        control: {rust(control)},
    }});""")


if __name__ == "__main__":
    main()
