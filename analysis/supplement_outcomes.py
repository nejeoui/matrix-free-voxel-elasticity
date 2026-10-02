"""Regenerate outcome/scale/manifest tables from retained evidence (CPU only).

Run: python3 analysis/supplement_outcomes.py
Raw results, frozen runners and protocols are read-only inputs. Null/NR means not
recorded; values are never inferred from a different host or solver configuration.
"""
import csv
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/supplement"


def read(path):
    return json.loads(Path(path).read_text())


def outcome_run(r, kind, stop_change=None):
    """One observation, retaining the history's timing and state conventions."""
    keys = ("case", "schedule", "arm", "repetition", "steps", "wall_seconds",
            "total_outer_iterations", "not_converged_solves", "final_compliance")
    row = {k: r.get(k) for k in keys}
    row["trajectory_kind"] = kind
    history = r.get("history", [])
    last = history[-1] if history else {}
    row.update(final_volume=last.get("volume"), final_design_change=last.get("change"),
               design_change_threshold=stop_change)
    if "skipped" in r:
        reason = "skipped: " + r["skipped"]
    elif kind in ("fixed_length", "auxiliary_diagnostic"):
        reason = "fixed_length"
    elif last.get("change") is not None and last["change"] <= stop_change:
        reason = "design_change_threshold"
    else:
        reason = "iteration_cap"
    row["stopping_reason"] = reason
    for source, target in (("gmg_setup_ms", "hierarchy_resetup_seconds"),
                           ("cg_ms", "cg_seconds"), ("sensitivity_ms", "sensitivity_seconds"),
                           ("diag_ms", "diagonal_seconds")):
        row[target] = sum(h[source] for h in history) / 1000 if history and all(h.get(source) is not None for h in history) else None
    # Construction precedes the per-step profiling timers; it was not separately timed.
    row["initial_hierarchy_construction_seconds"] = None
    return row


def optimization():
    rows, paired = [], []
    for session in ("g", "h"):
        protocol = read(ROOT / f"experiments/session_{session}/protocol.json")
        for path in sorted((ROOT / "results").glob(f"session-{session}-host*/remote/run-01/result.json")):
            host = path.parents[2].name
            d = read(path)
            for group, kind in (("runs", "fixed_length"), ("complete", "capped")):
                for i, r in enumerate(d[group]):
                    row = outcome_run(r, kind, protocol["complete_optimization"]["stop_change"] if group == "complete" else None)
                    row.update(session=session.upper(), host=host, raw_source=str(path.relative_to(ROOT)), raw_pointer=f"/{group}/{i}")
                    rows.append(row)
            if "their_e3_diagnostic" in d:
                r = outcome_run(d["their_e3_diagnostic"], "auxiliary_diagnostic")
                r.update(session=session.upper(), host=host, raw_source=str(path.relative_to(ROOT)), raw_pointer="/their_e3_diagnostic")
                rows.append(r)
            for kind in ("fixed_length", "capped"):
                matching = [r for r in rows if r["host"] == host and r["trajectory_kind"] == kind]
                cases = sorted({(r["case"], r["schedule"]) for r in matching})
                for case, schedule in cases:
                    by = {(r["repetition"], r["arm"]): r for r in matching if r["case"] == case and r["schedule"] == schedule}
                    reps = sorted({k[0] for k in by}, key=lambda v: -1 if v is None else v)
                    ratios, observations, work_matches = [], [], []
                    for rep in reps:
                        f, m = (by.get((rep, arm)) for arm in ("fused_ai_fp64", "modal8_muladd"))
                        if f and m and f["wall_seconds"] is not None and m["wall_seconds"] is not None:
                            ratio = f["wall_seconds"] / m["wall_seconds"]
                            ratios.append(ratio)
                            work_matches.append(abs(f["total_outer_iterations"] - m["total_outer_iterations"]) <= .03 * f["total_outer_iterations"])
                            observations.append({"repetition": rep, "fused_seconds": f["wall_seconds"], "parity_seconds": m["wall_seconds"], "fused_over_parity": ratio})
                    if ratios:
                        paired.append(dict(session=session.upper(), host=host, trajectory_kind=kind, case=case,
                                           schedule=schedule, paired_observations=observations,
                                           work_matched=all(work_matches), median_ratio=statistics.median(ratios), minimum_ratio=min(ratios), maximum_ratio=max(ratios)))
    return rows, paired


def scale():
    rows, fixed, products = [], [], []
    for path in sorted((ROOT / "results").glob("session-i-*/remote/run-01/result.json")):
        d, host = read(path), path.parents[2].name
        cases = {c["case"]: c for c in d["cases"]}
        for i, t in enumerate(d["to_tolerance"]):
            c = cases[t["case"]]
            nx, ny, nz = c["shapes"][0]
            # make_inputs.py clamps all 3 displacement components on the x=0 face.
            total = 3 * (nx + 1) * (ny + 1) * (nz + 1)
            rows.append(dict(host=host, gpu=d["device"]["name"], family="rediscretized", case=t["case"], arm=t["arm"],
                             repetition=t["rep"], elements=nx * ny * nz, total_dofs=total,
                             free_dofs=total - 3 * (ny + 1) * (nz + 1), hierarchy_levels=len(c["shapes"]),
                             iterations=t["iterations"], warm_solve_seconds=t["solver_seconds"],
                             shared_setup_seconds=c["shared_setup_seconds"], arm_setup_seconds=c["arm_setup_seconds"][t["arm"]],
                             independent_true_residual=t["independent_true_residual"], physical_passed=t["physical_passed"],
                             stopping_reason="residual_tolerance" if t["native_converged"] else "iteration_cap",
                             device_used_bytes=None, pool_total_bytes=None, live_array_bytes=None, peak_bytes=None,
                             raw_source=str(path.relative_to(ROOT)), raw_pointer=f"/to_tolerance/{i}"))
        for i, t in enumerate(d["fixed_work"]):
            fixed.append(dict(host=host, gpu=d["device"]["name"], **t, raw_source=str(path.relative_to(ROOT)), raw_pointer=f"/fixed_work/{i}"))
        for t in d["products"]:
            for rnd in t["rounds"]:
                for arm, ms in rnd["ms_per_product"].items():
                    products.append(dict(host=host, gpu=d["device"]["name"], case=t["case"], round=rnd["round"], arm=arm, ms_per_product=ms))
    return rows, fixed, products


def galerkin():
    rows = []
    for path in sorted((ROOT / "results").glob("session-f-*/remote/run-01/result.json")):
        for i, r in enumerate(read(path)["stages"]["sizing"]["rows"]):
            rows.append(dict(host=path.parents[2].name, family="Galerkin feasibility", **r,
                             live_array_bytes=None, peak_bytes=None, hierarchy_levels=4,
                             raw_source=str(path.relative_to(ROOT)), raw_pointer=f"/stages/sizing/rows/{i}"))
    for path in sorted((ROOT / "results").glob("session-i-*/remote/w2a-01/result.json")):
        for i, r in enumerate(read(path).get("records", [])):
            dims = r.get("dims")
            rows.append(dict(host=path.parents[2].name, family="Galerkin fixed-work", **r,
                             elements=math.prod(dims) if dims else None,
                             free_dofs=r["ndof"] - 3 * (dims[1] + 1) * (dims[2] + 1) if dims else None,
                             hierarchy_levels=4, live_array_bytes=None, peak_bytes=None, pool_total_bytes=None,
                             raw_source=str(path.relative_to(ROOT)), raw_pointer=f"/records/{i}"))
    return rows


def installation(host):
    base = host / "remote/installation"
    cfg = base / "cupy-config.txt"
    nvidia = base / "nvidia-smi.txt"
    attrs = dict(line.split(":", 1) for line in cfg.read_text().splitlines() if ":" in line) if cfg.exists() else {}
    attrs = {k.strip(): v.strip() for k, v in attrs.items()}
    text = nvidia.read_text() if nvidia.exists() else ""
    def smi(field):
        m = re.search(r"^\s*" + re.escape(field) + r"\s*:\s*(.+)$", text, re.M)
        return m.group(1).strip() if m else None
    return dict(gpu=attrs.get("Device 0 Name") or smi("Product Name"),
                gpu_capacity=smi("Total"), driver=smi("Driver Version"),
                cuda_runtime=attrs.get("CUDA Runtime Version"), cupy=attrs.get("CuPy Version"),
                nvrtc=attrs.get("NVRTC Version"), python=attrs.get("Python Version"),
                compiler="NVRTC " + attrs["NVRTC Version"] if attrs.get("NVRTC Version") else None,
                host_compiler_version=None, installation_source=str(base.relative_to(ROOT)))


def manifest():
    rows = []
    for host in sorted((ROOT / "results").glob("session-*")):
        result = host / "remote/run-01/result.json"
        # Failed/aborted attempts without a result are not counted as measured hosts.
        if not result.exists():
            continue
        s = host.name.split("-")[1]
        protocol_path = ROOT / f"experiments/session_{s}/protocol.json"
        if not protocol_path.exists():
            continue
        p, d = read(protocol_path), read(result)
        rental_path = host / "rental.json"
        rental = read(rental_path) if rental_path.exists() else {}
        row = installation(host)
        dev = d.get("device", {})
        offer = rental.get("offer", {})
        telemetry = d.get("telemetry", [])
        fields = telemetry[0].get("output", "").split(", ") if telemetry else []
        if len(fields) >= 4:
            row["driver"] = row["driver"] or fields[2]
            row["gpu_capacity"] = row["gpu_capacity"] or fields[3]
        row.update(session=s.upper(), host=host.name, cpu=offer.get("cpu_name"),
                   machine_id=offer.get("machine_id"), gpu=row["gpu"] or dev.get("name"),
                   cuda_device_total_bytes=dev.get("total_memory"), started_utc=d.get("started_utc"),
                   protocol=str(protocol_path.relative_to(ROOT)), raw_result=str(result.relative_to(ROOT)),
                   declared_cases=p.get("cases", p.get("sizing")),
                   arms=p.get("arms", p.get("products", {}).get("paths", p.get("paths_fp64", []) + p.get("paths_fp32", []))),
                   products=p.get("products"), fixed_work=p.get("fixed_work"),
                   to_tolerance=p.get("to_tolerance"), repetitions=p.get("repetitions"))
        row["metadata_sources_checked"] = [str(f.relative_to(ROOT)) for f in
                                           (rental_path, host / "preflight.json", result, host / "remote/installation/cupy-config.txt",
                                            host / "remote/installation/nvidia-smi.txt", host / "remote/installation/uname.txt") if f.exists()]
        if s == "a":
            row["missing_metadata"] = "CPU model absent: manual rental preflight records only 32 CPUs/188 GB RAM; no lscpu/model record in retained rental, installation or result files."
        if s in "gh":
            row.update(warmup="one SIMP step for every case/schedule/arm before repetitions",
                       timing_boundary="synchronized wall timer around solver construction and all SIMP steps; final state saving and allocator cleanup excluded",
                       precisions="FP64 outer; fp64 or mixed Galerkin schedule", sizes="216000, 512000, 514500, 499125 elements",
                       observed_repetitions=sorted({r["repetition"] for r in d["runs"]}))
        elif s in "dei":
            row.update(warmup="one untimed two-iteration solve per arm; E/I raw-product validation call per path before product timing",
                       timing_boundary="warm synchronized solve, including solver vector allocation and true-residual recomputation; hierarchy, spectral estimate, coarse factor, host transfers and replay excluded",
                       precisions="FP64 outer; FP64 or FP32 V-cycle", sizes=[c.get("case", c.get("id")) for c in d.get("cases", [])])
        elif s in "abc":
            row.update(warmup="one PCG iteration per FP64 path; one action per FP32 path, during setup before measurement",
                       timing_boundary="resident products or synchronized Jacobi-PCG/refinement solves; setup and CPU physical replay excluded",
                       precisions="FP64 and FP32 products; FP64 PCG or mixed refinement", sizes=[c["id"] for c in p["cases"]])
        elif s == "f":
            row.update(warmup="feasibility probes are cold subprocesses; not warm-solve measurements",
                       timing_boundary="separately recorded constructor/setup and capped solve in feasibility subprocess",
                       precisions="FP64/mixed per stage", sizes="2.000376 M success on large cards; 5.029452 M allocation failure")
        rows.append(row)
    counts = defaultdict(int)
    for r in rows:
        counts[r["session"]] += 1
    for r in rows:
        r["measured_allocations_in_family"] = counts[r["session"]]
    # These synthetic-product experiments precede the lettered sessions and are
    # the only direct RTX 3090 measurements used in the hardware product table.
    synthetic_protocol = read(ROOT / "evidence/kernel-rtx4090/protocol.json")
    for name in ("kernel-rtx4090", "kernel-rtx3090-replication"):
        path = ROOT / "evidence" / name / "result.json"
        d = read(path)
        dev = d["device"]
        telemetry = d["telemetry"][0]["output"].split(", ")
        tag = "synthetic-3090" if "3090" in name else "synthetic-4090"
        meta_dir = ROOT / "provenance/supplement/manifest-metadata"
        rent_path = meta_dir / f"{tag}-rental.json"
        config_path = meta_dir / f"{tag}-cupy-config.txt"
        rental = read(rent_path)
        config = {k.strip(): v.strip() for k, v in
                  (line.split(":", 1) for line in config_path.read_text().splitlines() if ":" in line)}
        rows.append(dict(session="synthetic", host=name, gpu=dev["name"], gpu_capacity=telemetry[3],
                         cpu=rental["offer"]["cpu_name"], driver=telemetry[2], cuda_runtime=config["CUDA Runtime Version"],
                         cupy=dev["cupy"], nvrtc=config["NVRTC Version"], compiler="NVRTC " + config["NVRTC Version"],
                         host_compiler_version=None, machine_id=rental["offer"]["machine_id"],
                         cuda_device_total_bytes=dev["total_memory"], started_utc=d["started_utc"],
                         protocol="evidence/kernel-rtx4090/protocol.json", raw_result=str(path.relative_to(ROOT)),
                         arms=synthetic_protocol["paths"], products=synthetic_protocol["benchmarks"],
                         measured_allocations_in_family=2, precisions="FP64; parity plain only",
                         warmup="five calls per path", repetitions="11 paired rounds; 20 calls per batch",
                         sizes=synthetic_protocol["benchmarks"]["shapes"],
                         timing_boundary=synthetic_protocol["benchmarks"]["timed_scope"],
                         metadata_sources_checked=[str(p.relative_to(ROOT)) for p in (path, rent_path, config_path)]))
    jhost = ROOT / "results/session-j-rtx4090vm-20260926-attempt07"
    jp = read(ROOT / "experiments/session_j/protocol.json")
    j = installation(jhost)
    jr = read(jhost / "rental.json")
    j.update(session="J", host=jhost.name, cpu=jr["offer"].get("cpu_name"), machine_id=jr["offer"]["machine_id"],
             protocol="experiments/session_j/protocol.json", raw_result=str((jhost / "remote/ncu/ncu-modal8_muladd.csv").relative_to(ROOT)),
             cupy="13.6.0", cuda_runtime="12.2.140 installed cudart; linked version NR", nvrtc="12.2.140 installed",
             compiler="NVRTC 12.2.140", compiler_note="installed library; loaded-version query not retained", host_compiler_version=None,
             arms=jp["ncu"]["kernels"], precisions="FP64", sizes=jp["ncu"]["grid"],
             repetitions="one selected launch per kernel; twelve profiler replay passes", measured_allocations_in_family=1,
             warmup="two matching launches skipped before selected launch",
             timing_boundary="Nsight Compute selected-kernel replay duration; excludes wrapper zeroing, projection and host overhead",
             nsight_compute="2023.2.2.0", missing_metadata="Installed CUDA/NVRTC version retained; loaded-version query and achieved clock lock not retained; see algebra-counters.md",
             metadata_sources_checked=[str((jhost / p).relative_to(ROOT)) for p in
                                       ("rental.json", "remote/installation/nvidia-smi.txt", "remote/installation/pip-freeze.txt",
                                        "remote/installation/nvrtc-location.txt", "remote/installation/cuda-runtime-install.log")])
    rows.append(j)
    base = ROOT / "evidence/companion/results/rescope/hex-modal-libceed-resident-20260924"
    cp = read(base / "protocol.json")
    metadata = ROOT / "provenance/supplement/manifest-metadata"
    gpu_text = read(metadata / "libceed-gpu-before.json")["stdout"]
    cpu_text = read(metadata / "libceed-cpu.json")["stdout"]
    capacity = re.search(r"FB Memory Usage\s+Total\s*:\s*(.+)", gpu_text).group(1).strip()
    cpu = re.search(r"Model name:\s*(.+)", cpu_text).group(1).strip()
    driver = re.search(r"Driver Version\s*:\s*(.+)", gpu_text).group(1).strip()
    rows.append(dict(session="libCEED", host="libceed-resident", gpu="NVIDIA GeForce RTX 4090", gpu_capacity=capacity,
                     cpu=cpu, driver=driver, cuda_runtime="CUDA 12.6 SDK/header probe", cupy="native harness", nvrtc=None,
                     compiler="nvcc 12.6.85; GCC 11.4.0", compiler_flags=cp["compiler"], host_compiler_version="GCC 11.4.0", machine_id=None,
                     protocol=str((base / "protocol.json").relative_to(ROOT)), raw_result=str((base / "rental-01/local-verification.json").relative_to(ROOT)),
                     arms=cp["paths"], precisions="FP64", sizes=[c["shape"] for c in cp["cases"] if c["performance"]],
                     measured_allocations_in_family=1, repetitions=cp["benchmark"], warmup="five seconds; twelve blocks of 256 actions per path/case",
                     timing_boundary=cp["measurement"], missing_metadata="NVRTC loaded-version query not recorded separately; native nvcc/GCC compiler versions retained.",
                     metadata_sources_checked=[str(p.relative_to(ROOT)) for p in sorted(metadata.glob("libceed-*.json"))]))
    return rows


def write_csv(name, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open("w", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v for k, v in r.items()})


def fmt(v):
    if v is None:
        return "NR"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v).replace("|", "/")


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] +
                     ["| " + " | ".join(fmt(v) for v in r) + " |" for r in rows])


def latex_tables(outcomes, pairs, scales, gal, env):
    """Readable split longtables for the self-contained PDF supplement."""
    def esc(v):
        s = fmt(v)
        return "".join({"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#",
                        "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}"}.get(c, c) for c in s)
    def number(v, digits=3):
        return "NR" if v is None else f"{v:.{digits}f}"
    def host(s):
        if s == "libceed-resident":
            return "CEED"
        for family in "gh":
            for n in (1, 2):
                if f"session-{family}-host{n}" in s:
                    return family.upper() + str(n)
        for family in "fi":
            if s.startswith(f"session-{family}-"):
                return family.upper() + ("A" if "a100" in s else "H" if "h100" in s else "R")
        if s.startswith("session-e-host"):
            return "E" + s.split("host")[1][0]
        if s.startswith("session-"):
            return s.split("-")[1].upper()
        return "S3090" if "3090" in s else "S4090"
    arms = {"stock": "S", "fused_ai_fp64": "F", "modal8_muladd": "P", "node_fp64": "N",
            "mg64-fused_ai": "64F", "mg64-node": "64N", "mg64-modal8_muladd": "64P",
            "mg32-node64-node32": "32N", "mg32-fused_ai64-node32": "32F", "mg32-modal8_muladd-node32": "32P"}
    cases = {"cantilever-216k": "C216", "cantilever-512k": "C512", "mbb-514k": "M514", "torsion-499k": "T499"}
    lines = [r"% Generated by analysis/supplement_outcomes.py; do not edit.",
             r"\section{Optimization outcomes and scale tables}",
             r"\label{sec:outcomes-scale-tables}",
             r"The tables regenerate from retained raw records. NR means not recorded, not zero. Exact per-repetition records, source paths and JSON pointers are in \texttt{optimization-outcomes.csv}, \texttt{scale-solves.csv} and \texttt{outcomes-scale.json}. No new GPU experiment is represented here.",
             r"Host keys G1/G2 and H1/H2 identify the two RTX 4090 hosts in Sessions G/H. IA/IH/IR identify Session I A100 PCIe, H100 80GB HBM3 and RTX 4090; FA/FH/FR identify Session F A100 PCIe, H100 NVL and RTX 4090. C216/C512 are cantilevers, M514 the MBB case and T499 torsion. S = stock gather/DGEMM/scatter, F = fused-ai FP64, P = parity mul-add, N = node FP64. Schedule 64 = all FP64; 32 = mixed. Scale arms 64F/64N/64P use FP64 throughout; 32F/32N/32P use a node FP32 V-cycle with the named FP64 outer product.",
             r"\subsection{Every optimization case, schedule and arm}",
             r"Each fixed-length row aggregates three repetitions by separate column medians. Capped rows and auxiliary diagnostics have n=1. Exit L = prescribed fixed length; C = 150-step iteration cap; D = auxiliary OC diagnostic. K is total Krylov iterations. An L exit is not a convergence test. Every C exit missed design change 0.01. O IDs match across the two tables.",
             r"\begingroup\small\setlength{\tabcolsep}{3pt}"]
    def tab(caption, headers, data, spec=None):
        if spec is None:
            spec = "@{}" + "l" * len(headers) + "@{}"
        lines.extend([r"\begin{longtable}{" + spec + "}", r"\caption{" + caption + r"}\\",
                      r"\toprule", " & ".join(headers) + r"\\", r"\midrule\endfirsthead",
                      r"\toprule", " & ".join(headers) + r"\\", r"\midrule\endhead",
                      r"\bottomrule\endfoot"])
        lines.extend(" & ".join(esc(v) for v in row) + r"\\" for row in data)
        lines.append(r"\end{longtable}")
    groups = defaultdict(list)
    for r in outcomes:
        groups[(r["host"], r["trajectory_kind"], r["case"], r["schedule"], r["arm"])].append(r)
    first, second = [], []
    for index, (key, runs) in enumerate(groups.items(), 1):
        def med(col):
            return statistics.median(r[col] for r in runs) if all(r[col] is not None for r in runs) else None
        label = f"O{index}"
        h, kind, c, schedule, arm = key
        first.append((label, host(h), cases[c], "64" if schedule == "fp64" else "32", arms[arm], len(runs),
                      int(med("steps")), {"fixed_length": "L", "capped": "C", "auxiliary_diagnostic": "D"}[kind],
                      int(med("total_outer_iterations")), number(med("wall_seconds")),
                      sum(r["not_converged_solves"] for r in runs)))
        second.append((label, number(med("cg_seconds")), number(med("hierarchy_resetup_seconds")),
                       number(med("final_compliance"), 6), number(med("final_volume"), 6), number(med("final_design_change"), 6)))
    tab("Optimization identity, stopping outcome and wall time. Failed-solve count is summed over repetitions.",
        ["ID", "Host", "Case", "Sch.", "Arm", "n", "Steps", "Exit", "K", "Wall s", "Failed"], first)
    lines.append(r"CG and re-setup are sums of per-step timers, then medians across repetitions. Initial hierarchy construction is included in wall time but separately NR for every row. Logged compliance precedes the final OC update; volume and change follow it. Change is the maximum absolute unfiltered design-variable update. Density-field agreement was not computed.")
    tab("Optimization solve/setup and final logged quantities (O IDs as above).",
        ["ID", "CG s", "Re-setup s", "Compliance", "Volume", "Change"], second)
    lines.append(r"The following paired ratios divide fused-ai wall time by parity wall time within a repetition. L rows have three ratios; C rows have one. Work=yes requires total Krylov counts within 3\% in every pair. Work=no rows remain descriptive and are not qualified fixed-work speedup claims. Median-threshold decisions additionally require the stated compliance/volume/solve gates; these ranges are not bootstrap intervals.")
    tab("All paired wall-time ratios, their observed ranges and Krylov work qualification.",
        ["Host", "Case", "Sch.", "Kind", "n", "Median", "Min", "Max", "Work"],
        [(host(p["host"]), cases[p["case"]], "64" if p["schedule"] == "fp64" else "32",
          "L" if p["trajectory_kind"] == "fixed_length" else "C", len(p["paired_observations"]),
          number(p["median_ratio"], 4), number(p["minimum_ratio"], 4), number(p["maximum_ratio"], 4),
          "yes" if p["work_matched"] else "no") for p in pairs])
    lines.extend([r"\endgroup", r"\subsection{Scale: every rediscretized solver configuration}",
                  r"n=2 warm tolerance solves per row; all passed the independent physical check. Elements and total/free DOFs follow the recorded grid and x=0 clamp. Both repetition times remain in CSV. The table median is not a total setup-plus-solve time. All six arms were resident in the measured harness. Live-array and transient peak memory are NR for every row. R IDs match the timing split below.",
                  r"\begingroup\small\setlength{\tabcolsep}{3pt}"])
    groups = defaultdict(list)
    for r in scales:
        groups[(r["host"], r["case"], r["arm"])].append(r)
    first, second = [], []
    for index, (key, runs) in enumerate(groups.items(), 1):
        r = runs[0]
        label = f"R{index}"
        first.append((label, host(key[0]), key[1], arms[key[2]], r["elements"], r["total_dofs"], r["free_dofs"],
                      r["hierarchy_levels"], "/".join(str(n) for n in sorted({x["iterations"] for x in runs})),
                      number(statistics.median(x["warm_solve_seconds"] for x in runs), 4)))
        second.append((label, number(r["shared_setup_seconds"], 4), number(r["arm_setup_seconds"], 4),
                       f"{max(x['independent_true_residual'] for x in runs):.2e}", "NR", "NR"))
    tab("Rediscretized scale dimensions and warm time to the requested residual tolerance.",
        ["ID", "Host", "Case", "Arm", "Elements", "DOFs", "Free DOFs", "Lvls", "K", "Warm s"], first)
    lines.append(r"Shared setup is once per case: reference hierarchy, host coarsest assembly/Cholesky, transfers and spectral estimation. Arm setup is the incremental recorded construction with cached compilation possible. Shared values are repeated for lookup, not additive across arms. Noncoarsest levels are matrix-free; the coarsest operator is assembled and factored.")
    tab("Scale setup and observed independent residual (R IDs as above).",
        ["ID", "Shared setup s", "Arm setup s", "Max. residual", "Live bytes", "Peak bytes"], second)
    lines.extend([r"\endgroup", r"\subsection{Galerkin capacity and capped solves}",
                  r"Session F uses four-level Galerkin with the arms/schedules shown. An OK row denotes allocation/execution success, not convergence: each retained 40-iteration solve was capped. Device used is CUDA total-minus-free; pool reserved includes allocator caches. Live arrays and construction peaks are NR. The 2 M success and 5 M failure are observations; 3 M is only an estimate. The 2 M grid is 2,000,376 elements / 6,169,152 total DOFs / 6,144,768 free DOFs. The 5 M grid is 5,029,452 / 15,397,956 / 15,353,064.",
                  r"\begingroup\small\setlength{\tabcolsep}{3pt}"])
    capacity = []
    for r in gal:
        if r["family"] == "Galerkin feasibility":
            capacity.append((host(r["host"]), arms[r["arm"]], "64" if r["schedule"] == "fp64" else "32", r["size"],
                             "OK" if r["status"] == "ok" else "OOM", r.get("iterations"), number(r.get("setup_seconds")),
                             number(r.get("solve_seconds")), number(r.get("gpu_used_after_bytes", 0) / 2**30) if "gpu_used_after_bytes" in r else "NR",
                             number(r["pool_total_bytes"] / 2**30) if r.get("pool_total_bytes") else "NR"))
    tab("Cold feasibility setup and capped solve; all successful rows have K=40 and did not converge.",
        ["Host", "Arm", "Sch.", "Size", "Status", "K", "Setup s", "Solve s", "Used GiB", "Pool GiB"], capacity)
    lines.append(r"W2a below records every block and arm. Each block has three fixed 50-iteration solves; the table shows their median. The separate tolerance attempt occurs only in block 0; NR in block 1 is not a failure. The 1 M grid has 1,000,000 elements / 3,106,053 total DOFs / 3,090,600 free DOFs; the 2 M grid has 2,000,376 / 6,169,152 / 6,144,768. Both use four levels. Setup brackets density-dependent setup only; construction before it is excluded.")
    w2a = []
    for r in gal:
        if r["family"] == "Galerkin fixed-work":
            t = r.get("to_tolerance", {})
            w2a.append((host(r["host"]), r["size"], "64" if r["schedule"] == "fp64" else "32", arms[r["arm"]], r["block"],
                        number(r.get("setup_seconds")), number(statistics.median(t["seconds"] for t in r["fixed_work"])) if "fixed_work" in r else "NR",
                        t.get("iterations"), "yes" if t.get("converged") else ("no" if "converged" in t else "NR"), number(t.get("seconds")),
                        number(r["gpu_used_bytes"] / 2**30) if "gpu_used_bytes" in r else "NR"))
    tab("Galerkin W2a fixed-work and tolerance-attempt outcomes; used memory is a post-setup sample, not peak.",
        ["Host", "Size", "Sch.", "Arm", "Block", "Setup s", "50 iter. s", "Tol. K", "Conv.", "Tol. s", "Used GiB"], w2a)
    lines.extend([r"\endgroup", r"\subsection{Hardware/software and experiment manifest}",
                  r"The synthetic rows S4090/S3090 precede Sessions A--I and directly measured only plain parity. H100 NVL (FH) and H100 80GB HBM3 (IH) are distinct. Every A100 measured here is PCIe; the historical A100 SXM roofline specification is not measured hardware. CPU NR means the record did not retain a model name. Every row is one allocation.",
                  r"\begingroup\small\setlength{\tabcolsep}{3pt}"])
    tab("Measured allocation identities; complete filenames and machine identifiers are in experiment-manifest.csv.",
        ["Key", "GPU", "Capacity", "CPU"],
        [(host(r["host"]), r["gpu"], r["gpu_capacity"], r["cpu"]) for r in env], "@{}lp{40mm}p{20mm}p{70mm}@{}")
    tab("Software from installation records. Installed and loaded-version queries are distinguished. Native libCEED nvcc/GCC versions come from retained version-command output.",
        ["Key", "Driver", "CUDA runtime", "CuPy", "CUDA/host compiler"],
        [(host(r["host"]), r["driver"], r["cuda_runtime"], r["cupy"], r.get("compiler")) for r in env], "@{}llp{56mm}lp{35mm}@{}")
    lines.extend([r"\endgroup",
                  r"Precision, kernels, sizes, repeated work and timing-boundary definitions are explicit in the Timing, optimization outcomes and measured scale narrative and in the structured manifest CSV. In particular: synthetic products use five warmups and 11 rounds of 20 calls; A--C use 11 product rounds, 10 fixed-work rounds, and 3/3/2 tolerance repetitions; D/E use 5 tolerance and 10 fixed-work repetitions (E has 11 product rounds); G/H use 3 paired trajectories plus one capped trajectory per arm/schedule; I uses 5 product rounds of 10 calls, 2 tolerance repetitions and 3 fixed-work rounds. G/H warm one full step per arm; A--C warm one FP64 PCG iteration or FP32 action; D/E/I warm two solver iterations per arm. Setup, transfers, masks and zeroing depend on the endpoint as specified in the narrative, not on a universal timer boundary."])
    (OUT / "outcomes-scale-tables.tex").write_text("\n".join(lines) + "\n")


def main():
    OUT.mkdir(exist_ok=True)
    outcomes, pairs = optimization()
    scales, fixed, products = scale()
    gal = galerkin()
    env = manifest()
    latex_tables(outcomes, pairs, scales, gal, env)
    ladder = [dict(host=p.parents[2].name, raw_source=str(p.relative_to(ROOT)), **row)
              for p in sorted((ROOT / "results").glob("session-i-*/remote/run-01/result.json")) for row in read(p)["ladder"]]
    data = dict(schema_version=1, optimization=outcomes, optimization_paired=pairs,
                scale=scales, scale_fixed_work=fixed, scale_products=products, galerkin_capacity=gal,
                scale_ladder=ladder, experiment_manifest=env)
    sources = sorted({r["raw_source"] for r in outcomes + scales + fixed + gal} |
                     {r[k] for r in env for k in ("protocol", "raw_result")} |
                     {s for r in env for s in r.get("metadata_sources_checked", [])})
    data["source_sha256"] = {s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest() for s in sources}
    (OUT / "outcomes-scale.json").write_text(json.dumps(data, indent=2) + "\n")
    for name, rows in (("optimization-outcomes.csv", outcomes), ("optimization-paired.csv", pairs),
                       ("scale-solves.csv", scales), ("scale-fixed-work.csv", fixed),
                       ("scale-products.csv", products), ("galerkin-capacity.csv", gal), ("experiment-manifest.csv", env)):
        write_csv(name, rows)
    md = ["# Generated outcome and capacity tables", "", "Regenerate with `python3 analysis/supplement_outcomes.py`. "
          "See [methods and interpretation](outcomes-scale.md). NR = not recorded, not zero. "
          "CSV/JSON files retain every repetition and exact source record; medians below are descriptive.", "", "## Optimization outcomes", "",
          "Every row identifies its host, schedule and arm; three fixed-length repetitions are aggregated by the median separately for each numerical column. "
          "Single capped trajectories have n=1. Setup is the sum of per-step `gmg_setup_ms`, including density-dependent re-setup; initial construction is included in wall time but not separately recorded. "
          "CG/setup/compliance/volume/change columns follow the retained per-step logging convention.", ""]
    groups = defaultdict(list)
    for r in outcomes:
        groups[(r["host"], r["trajectory_kind"], r["case"], r["schedule"], r["arm"])].append(r)
    display = []
    cols = ("steps", "total_outer_iterations", "wall_seconds", "cg_seconds", "hierarchy_resetup_seconds", "final_compliance", "final_volume", "final_design_change")
    for key, runs in groups.items():
        vals = [statistics.median(r[c] for r in runs) if all(r[c] is not None for r in runs) else None for c in cols]
        display.append((*key, len(runs), ",".join(sorted({r["stopping_reason"] for r in runs})), *vals))
    md += [table(["host", "kind", "case", "schedule", "arm", "n", "exit", "steps", "Krylov total", "wall s", "CG s", "re-setup s", "compliance", "volume", "change"], display), "", "## Paired wall-time observations", ""]
    md += [table(["host", "kind", "case", "schedule", "fused/parity ratios in repetition order", "median", "work matched within 3%"],
                 [(p["host"], p["trajectory_kind"], p["case"], p["schedule"], ", ".join(f"{o['fused_over_parity']:.6f}" for o in p["paired_observations"]), p["median_ratio"], p["work_matched"]) for p in pairs]), "", "## Rediscretized scale, all configurations", "",
           "n=2 warm solves per row; time is the median. Shared setup is paid once per case and arm setup is the recorded incremental setup in execution order, with cached compilation possible. "
           "No per-route live/peak memory measurement was recorded; neither periodic device telemetry nor an OOM bound supplies one. "
           "Total/free DOFs are calculated from recorded dimensions and the fully clamped x=0 face.", ""]
    groups = defaultdict(list)
    for r in scales:
        groups[(r["host"], r["case"], r["arm"])].append(r)
    display = []
    for key, runs in groups.items():
        r = runs[0]
        display.append((*key, r["elements"], r["total_dofs"], r["free_dofs"], r["hierarchy_levels"],
                        ",".join(str(n) for n in sorted({x["iterations"] for x in runs})),
                        statistics.median(x["warm_solve_seconds"] for x in runs), r["shared_setup_seconds"], r["arm_setup_seconds"], "NR / NR"))
    md += [table(["host", "case", "arm", "elements", "DOFs", "free DOFs", "levels", "iterations", "warm solve s", "shared setup s", "arm setup s", "live/peak memory"], display), "", "Ladder outcomes (including unsuccessful construction):", "",
           table(["host", "case", "status"], [(r["host"], r["case"], r["status"]) for r in ladder]), "", "## Galerkin feasibility and retained memory", ""]
    display = []
    for r in gal:
        if r["family"] == "Galerkin feasibility":
            display.append((r["host"], r["size"], r["arm"], r["schedule"], r["n_elem"], r["ndof"], r["n_free"], r["status"], r.get("iterations"),
                            r.get("converged"), r.get("setup_seconds"), r.get("solve_seconds"), (r["gpu_used_after_bytes"] / 2**30) if "gpu_used_after_bytes" in r else None,
                            (r["pool_total_bytes"] / 2**30) if r.get("pool_total_bytes") else None))
    md += [table(["host", "size", "arm", "schedule", "elements", "DOFs", "free DOFs", "status", "iterations", "converged", "setup s", "capped solve s", "device used GiB", "pool reserved GiB"], display), "", "## Galerkin W2a: every block and arm", "",
           "Each block has three 50-iteration fixed-work solves; their median is shown. Only block 0 records the separate 1000-iteration tolerance attempt; its outcome is retained, including failures. Device used is CUDA total-minus-free after setup, not peak or live arrays.", ""]
    display = []
    for r in gal:
        if r["family"] == "Galerkin fixed-work":
            t = r.get("to_tolerance", {})
            display.append((r["host"], r["size"], r["schedule"], r["arm"], r["block"], r.get("elements"), r.get("ndof"), r.get("free_dofs"), 4,
                            r.get("setup_seconds"), statistics.median(t["seconds"] for t in r["fixed_work"]) if "fixed_work" in r else None,
                            t.get("iterations"), t.get("converged"), t.get("seconds"), r.get("gpu_used_bytes", 0) / 2**30 if "gpu_used_bytes" in r else None))
    md += [table(["host", "size", "schedule", "arm", "block", "elements", "DOFs", "free DOFs", "levels", "setup s", "50 iterations s", "tol. iterations", "converged", "tol. attempt s", "device used GiB"], display), "", "## Hardware and software manifest", "",
           "One row per measured allocation; full protocol cases, arms, precision, repetitions, warmup and boundaries are in experiment-manifest.csv and outcomes-scale.json. "
           "Driver and runtime versions are separate. NR compiler metadata is not guessed from the Python compiler string.", ""]
    md += [table(["session/host", "GPU", "capacity", "CPU", "driver", "CUDA runtime", "CuPy", "CUDA/host compiler"],
                 [(r["host"], r["gpu"], r["gpu_capacity"], r["cpu"], r["driver"], r["cuda_runtime"], r["cupy"], r.get("compiler")) for r in env])]
    (OUT / "outcomes-scale-tables.md").write_text("\n".join(md) + "\n")
    print(json.dumps({"optimization_rows": len(outcomes), "paired_rows": len(pairs), "scale_solve_rows": len(scales), "scale_fixed_rows": len(fixed), "galerkin_rows": len(gal), "manifest_rows": len(env)}, sort_keys=True))


if __name__ == "__main__":
    main()
