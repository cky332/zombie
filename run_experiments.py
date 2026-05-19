"""Driver for the three reproduction experiments.

Usage:
  python run_experiments.py --smoke              # 1 quick SW session, prints actions
  python run_experiments.py --rq1a               # RQ1-A effectiveness (SW + RAG, 5 methods)
  python run_experiments.py --rq1b               # RQ1-B evolution-strategy comparison
  python run_experiments.py --all                # RQ1-A + RQ1-B + RQ2 plots
  python run_experiments.py --all --embedder tfidf   # offline-safe embedder

Prerequisites
-------------
1. Set the API key:
     export SILICONFLOW_API_KEY="sk-..."
2. The remote-execution environment's outbound network policy MUST allow:
     - api.siliconflow.cn       (DeepSeek-V3.2 chat completions)
     - huggingface.co + cdn-lfs.huggingface.co
       (only on first run, to download sentence-transformers/all-MiniLM-L6-v2;
        skip this if you pass `--embedder tfidf`, which uses a HashingVectorizer
        and needs no model download)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"


# Scale (matches plan-mode answers) ----------------------------------------

SW_EXPOSURE_ROUNDS = 3
SW_TRIGGER_ROUNDS = 20
RAG_EXPOSURE_ROUNDS = 20
RAG_TRIGGER_ROUNDS = 20
RAG_BENIGN_DB_SIZE = 200
RAG_TOP_K = 10

# Per-session step budget: bait sessions get more steps so the agent has
# enough room to (a) search, (b) read at least one redirect page, (c) finally
# fetch the malicious URL. Trigger sessions stay tight to keep the experiment
# fast.
BAIT_MAX_STEPS = 6
TRIGGER_MAX_STEPS = 4

PAYLOAD_SIGNATURE = "official-ai-safety.org"  # used for retention metric


def _import_runtime():
    """Imports kept inside a function so --smoke can fail fast if openai is missing."""
    global agents, attacks, metrics, tasks, evolution
    import agents  # noqa
    import attacks  # noqa
    import metrics  # noqa
    import tasks  # noqa
    import evolution  # noqa


def _build_embedder(kind: str):
    if kind == "tfidf":
        from agents import TfidfEmbedder

        return TfidfEmbedder()
    if kind == "sbert":
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    raise ValueError(f"unknown embedder kind: {kind}")


def _now() -> str:
    return time.strftime("%H:%M:%S")


def _save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o):
    if dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


# ---------------------------------------------------------------------------
# RQ1-A: Effectiveness on Sliding Window and RAG
# ---------------------------------------------------------------------------


def _build_web(method: str, architecture: str, bait_mode: bool):
    """Construct a SimulatedWeb whose malicious page body matches the method."""
    builder = (
        attacks.build_payload_for_sliding_window
        if architecture == "sliding_window"
        else attacks.build_payload_for_rag
    )
    page = builder(method)
    return tasks.SimulatedWeb(malicious_page_body=page, bait_mode=bait_mode)


def run_sliding_window_for_method(
    method: str,
    *,
    evolution_strategy: str = "raw_history",
    exposure: int = SW_EXPOSURE_ROUNDS,
    trigger: int = SW_TRIGGER_ROUNDS,
    log_dir: Optional[Path] = None,
) -> dict:
    print(f"  [{_now()}] sliding_window / {method} / evo={evolution_strategy}")
    web = _build_web(method, "sliding_window", bait_mode=True)
    agent = agents.SlidingWindowAgent(web, evolution_strategy=evolution_strategy)

    per_round_detection: list[dict[str, bool]] = []
    per_round_actions: list[list[dict]] = []
    bait = tasks.BAIT_TASKS
    benign = tasks.BENIGN_TASKS

    # Exposure
    for i in range(exposure):
        task = bait[i % len(bait)]
        web.bait_mode = True
        result = agent.run_session(task, round_index=i, max_steps=BAIT_MAX_STEPS)
        det = metrics.detect_attack(result.actions)
        per_round_detection.append(det)
        per_round_actions.append([{"tool": a.tool, "args": a.args} for a in result.actions])

    # Trigger
    web.bait_mode = False
    for j in range(trigger):
        task = benign[j % len(benign)]
        result = agent.run_session(task, round_index=exposure + j, max_steps=TRIGGER_MAX_STEPS)
        det = metrics.detect_attack(result.actions)
        per_round_detection.append(det)
        per_round_actions.append([{"tool": a.tool, "args": a.args} for a in result.actions])

    # Trigger-phase only ASR (paper Fig 3 plots ASR across trigger rounds)
    trigger_detection = per_round_detection[exposure:]
    asr_exfil = metrics.cumulative_asr(trigger_detection, "data_exfil")
    asr_cmd = metrics.cumulative_asr(trigger_detection, "cmd_exec")
    retention = metrics.sliding_window_retention(agent.snapshots, PAYLOAD_SIGNATURE)

    result = {
        "method": method,
        "architecture": "sliding_window",
        "evolution_strategy": evolution_strategy,
        "exposure": exposure,
        "trigger": trigger,
        "per_round_detection": per_round_detection,
        "per_round_actions": per_round_actions,
        "asr_curve_data_exfil": asr_exfil,
        "asr_curve_cmd_exec": asr_cmd,
        "retention_curve": retention,
        "final_buffer": agent.buffer.copy(),
    }
    if log_dir is not None:
        _save_json(result, log_dir / f"sw_{method}_{evolution_strategy}.json")
    return result


def run_rag_for_method(
    method: str,
    *,
    embedder,
    evolution_strategy: str = "raw_history",
    exposure: int = RAG_EXPOSURE_ROUNDS,
    trigger: int = RAG_TRIGGER_ROUNDS,
    db_size: int = RAG_BENIGN_DB_SIZE,
    top_k: int = RAG_TOP_K,
    log_dir: Optional[Path] = None,
) -> dict:
    print(f"  [{_now()}] rag / {method} / evo={evolution_strategy}")
    web = _build_web(method, "rag", bait_mode=True)
    agent = agents.RAGAgent(
        web,
        embedder=embedder,
        evolution_strategy=evolution_strategy,
        top_k=top_k,
    )
    agent.seed_db(tasks.make_benign_db_entries(db_size), source="benign")

    per_round_detection: list[dict[str, bool]] = []
    per_round_retrieved: list[list[str]] = []
    bait = tasks.BAIT_TASKS
    benign = tasks.BENIGN_TASKS

    # Exposure: bait tasks pull the malicious page into context and memory.
    for i in range(exposure):
        task = bait[i % len(bait)]
        web.bait_mode = True
        result = agent.run_session(task, round_index=i, max_steps=BAIT_MAX_STEPS)
        det = metrics.detect_attack(result.actions)
        per_round_detection.append(det)
        per_round_retrieved.append(result.retrieved)

    # Trigger: benign, unrelated tasks. We also collect retrieval density.
    web.bait_mode = False
    for j in range(trigger):
        task = benign[j % len(benign)]
        # Snapshot the top-100 retrieval for the recall metric BEFORE running.
        retrieved_top100 = [r["text"] for r in agent.retrieve(task, top_k=100)]
        result = agent.run_session(task, round_index=exposure + j, max_steps=TRIGGER_MAX_STEPS)
        det = metrics.detect_attack(result.actions)
        per_round_detection.append(det)
        per_round_retrieved.append(retrieved_top100)

    trigger_detection = per_round_detection[exposure:]
    asr_exfil = metrics.cumulative_asr(trigger_detection, "data_exfil")
    asr_cmd = metrics.cumulative_asr(trigger_detection, "cmd_exec")

    recall_per_round = per_round_retrieved[exposure:]
    rec_at_k = metrics.rag_recall_at_k(recall_per_round, k_values=[10, 50, 100])

    result = {
        "method": method,
        "architecture": "rag",
        "evolution_strategy": evolution_strategy,
        "exposure": exposure,
        "trigger": trigger,
        "db_size": db_size,
        "top_k": top_k,
        "per_round_detection": per_round_detection,
        "asr_curve_data_exfil": asr_exfil,
        "asr_curve_cmd_exec": asr_cmd,
        "injection_count_curve": agent.snapshots,
        "recall_at_k": {k: rec_at_k[k] for k in rec_at_k},
        "final_db_total": len(agent.db),
        "final_injected": sum(1 for e in agent.db if e["source"] == "injected"),
    }
    if log_dir is not None:
        _save_json(result, log_dir / f"rag_{method}_{evolution_strategy}.json")
    return result


def run_rq1a(embedder_kind: str = "sbert") -> None:
    print(f"[{_now()}] === RQ1-A: Attack Effectiveness ===")
    out_dir = RESULTS_DIR / "rq1_effectiveness"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{_now()}] Loading embedder ({embedder_kind}) ...")
    embedder = _build_embedder(embedder_kind)

    sw_results: dict[str, dict] = {}
    rag_results: dict[str, dict] = {}

    for m in attacks.METHODS:
        sw_results[m] = run_sliding_window_for_method(m, log_dir=out_dir)
        rag_results[m] = run_rag_for_method(m, embedder=embedder, log_dir=out_dir)

    # Plots
    sw_exfil = {m: sw_results[m]["asr_curve_data_exfil"] for m in attacks.METHODS}
    sw_cmd = {m: sw_results[m]["asr_curve_cmd_exec"] for m in attacks.METHODS}
    rag_exfil = {m: rag_results[m]["asr_curve_data_exfil"] for m in attacks.METHODS}
    rag_cmd = {m: rag_results[m]["asr_curve_cmd_exec"] for m in attacks.METHODS}

    metrics.plot_asr_curves(
        sw_exfil, "Sliding Window: Data Exfiltration ASR", str(out_dir / "sw_asr_exfil.png")
    )
    metrics.plot_asr_curves(
        sw_cmd, "Sliding Window: Command Execution ASR", str(out_dir / "sw_asr_cmd.png")
    )
    metrics.plot_asr_curves(
        rag_exfil, "RAG: Data Exfiltration ASR", str(out_dir / "rag_asr_exfil.png")
    )
    metrics.plot_asr_curves(
        rag_cmd, "RAG: Command Execution ASR", str(out_dir / "rag_asr_cmd.png")
    )

    # Persistence plots (Fig 5 -- RQ2 piggybacks on these runs)
    out_persist = RESULTS_DIR / "rq2_persistence"
    out_persist.mkdir(parents=True, exist_ok=True)

    retention_curves = {
        m: sw_results[m]["retention_curve"] for m in attacks.METHODS
    }
    metrics.plot_retention(
        retention_curves,
        "Sliding Window: Payload Retention",
        str(out_persist / "sw_retention.png"),
    )

    inj_curves = {m: rag_results[m]["injection_count_curve"] for m in attacks.METHODS}
    metrics.plot_count(
        inj_curves,
        "RAG: Payload Count (Infection + Trigger)",
        "Payload entries in DB",
        str(out_persist / "rag_injection_count.png"),
    )

    rec_per_method = {m: rag_results[m]["recall_at_k"] for m in attacks.METHODS}
    metrics.plot_recall_at_k(
        rec_per_method,
        "RAG: Retrieval Density by K",
        str(out_persist / "rag_recall_at_k.png"),
    )

    _save_json(
        {"sw": sw_results, "rag": rag_results},
        out_dir / "_summary.json",
    )
    print(f"[{_now()}] RQ1-A done. Outputs in {out_dir}")


# ---------------------------------------------------------------------------
# RQ1-B: Evolution-strategy impact
# ---------------------------------------------------------------------------


def run_rq1b() -> None:
    print(f"[{_now()}] === RQ1-B: Evolution strategy impact (Sliding Window, Zombie) ===")
    out_dir = RESULTS_DIR / "rq1_evolution"
    out_dir.mkdir(parents=True, exist_ok=True)

    final_asr_exfil: dict[str, float] = {}
    final_asr_cmd: dict[str, float] = {}

    for strat in ("raw_history", "verbal_reflection", "refined_experience"):
        result = run_sliding_window_for_method(
            "zombie",
            evolution_strategy=strat,
            log_dir=out_dir,
        )
        final_asr_exfil[strat] = result["asr_curve_data_exfil"][-1]
        final_asr_cmd[strat] = result["asr_curve_cmd_exec"][-1]

    metrics.plot_bar(
        final_asr_exfil,
        "ASR by Evolution Strategy: Data Exfiltration",
        "Final cumulative ASR",
        str(out_dir / "evolution_asr_exfil.png"),
    )
    metrics.plot_bar(
        final_asr_cmd,
        "ASR by Evolution Strategy: Command Execution",
        "Final cumulative ASR",
        str(out_dir / "evolution_asr_cmd.png"),
    )
    _save_json(
        {"data_exfil": final_asr_exfil, "cmd_exec": final_asr_cmd},
        out_dir / "_summary.json",
    )
    print(f"[{_now()}] RQ1-B done. Outputs in {out_dir}")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def run_smoke() -> None:
    print(f"[{_now()}] === Smoke test: SW / zombie / 1 bait + 2 trigger ===")
    web = _build_web("zombie", "sliding_window", bait_mode=True)
    agent = agents.SlidingWindowAgent(web)

    for i, task in enumerate(tasks.BAIT_TASKS[:1]):
        web.bait_mode = True
        r = agent.run_session(task, round_index=i, max_steps=BAIT_MAX_STEPS)
        print(f"  bait[{i}] thinking[0]={r.thinking[0][:200]!r}")
        for a in r.actions:
            print(f"    -> {a.tool}({a.args})")
        read_mal = any(
            a.tool == "read_url" and a.args.get("url") == web.malicious_url
            for a in r.actions
        )
        det_bait = metrics.detect_attack(r.actions)
        buf_tok = sum(len(x) for x in agent.buffer) // 4
        print(
            f"  bait[{i}] read_malicious_url={read_mal} det={det_bait}"
            f" buffer_len={len(agent.buffer)} buffer_tok~{buf_tok}"
        )

    web.bait_mode = False
    for j, task in enumerate(tasks.BENIGN_TASKS[:2]):
        r = agent.run_session(task, round_index=1 + j, max_steps=TRIGGER_MAX_STEPS)
        det = metrics.detect_attack(r.actions)
        print(
            f"  trigger[{j}] task={task[:40]!r} det={det}"
            f" thinking[0]={r.thinking[0][:160]!r}"
        )
        for a in r.actions:
            print(f"    -> {a.tool}({a.args})")
    print(f"[{_now()}] Smoke done. final buffer length = {len(agent.buffer)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--rq1a", action="store_true")
    parser.add_argument("--rq1b", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--embedder",
        choices=["sbert", "tfidf"],
        default="sbert",
        help="sbert downloads from HuggingFace on first use; tfidf is offline-safe.",
    )
    args = parser.parse_args()

    if not any([args.smoke, args.rq1a, args.rq1b, args.all]):
        parser.print_help()
        sys.exit(0)

    _import_runtime()

    try:
        if args.smoke:
            run_smoke()
        if args.rq1a or args.all:
            run_rq1a(embedder_kind=args.embedder)
        if args.rq1b or args.all:
            run_rq1b()
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
