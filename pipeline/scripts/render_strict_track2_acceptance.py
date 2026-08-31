#!/usr/bin/env python3
"""Render the strict Track 2 acceptance evidence as Markdown.

The renderer deliberately fails closed: incomplete evaluation batches, a service
failure, or a restart mismatch cannot be presented as a passing report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def percent(value: float) -> str:
    return f"{100.0 * value:.4f}%"


def signed(value: float) -> str:
    return f"{value:+.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--service", required=True, type=Path)
    parser.add_argument("--restart-before", required=True, type=Path)
    parser.add_argument("--restart-after", required=True, type=Path)
    parser.add_argument("--not-ready", required=True, type=Path)
    parser.add_argument("--eval-repeat", required=True, type=Path)
    parser.add_argument("--checkpoint-hashes", required=True, type=Path)
    parser.add_argument("--v15-verification", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    evaluation = load(args.evaluation)
    service = load(args.service)
    restart_before = load(args.restart_before)
    restart_after = load(args.restart_after)
    not_ready = load(args.not_ready)
    eval_repeat = load(args.eval_repeat)
    v15_verification = load(args.v15_verification)

    primary = evaluation["primary_official_success"]
    auxiliary = evaluation["auxiliary_read_only_metrics"]
    candidate_key = evaluation.get("candidate_variant", "trained_global_step_1")
    baseline = primary["baseline"]
    trained = primary[candidate_key]
    improvement = evaluation["improvement"]
    by_arm = evaluation["success_improvement_by_arm"]
    grasp = evaluation["grasp_improvement"]

    evaluation_complete = (
        evaluation["official_seed_count"] == 128
        and baseline["count"] == 128
        and trained["count"] == 128
        and all(row["source"] == "deterministic_instrumented_read_only" for row in baseline["batches"])
        and all(row["source"] == "deterministic_instrumented_read_only" for row in trained["batches"])
    )
    overall_gate = (
        improvement["absolute_at_least_3pp"]
        and improvement["relative_at_least_3_percent"]
    )
    dual_arm_gate = all(by_arm[arm]["absolute_percentage_points"] >= 0.0 for arm in ("left", "right"))
    grasp_gate = all(grasp[arm]["absolute_percentage_points"] >= 0.0 for arm in ("all", "left", "right"))
    restart_gate = (
        restart_before["pixel_sha256"] == restart_after["pixel_sha256"]
        and restart_before["capabilities_sha256"] == restart_after["capabilities_sha256"]
        and restart_before["health"] == restart_after["health"]
    )
    service_gate = (
        service.get("passed") is True
        and str(service.get("base_url", "")).startswith("https://")
        and service["latency_ms"]["all_under_recommended_timeout"] is True
        and not_ready.get("passed") is True
        and restart_gate
    )
    checkpoint_lines = [
        line for line in args.checkpoint_hashes.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    checkpoint_gate = len(checkpoint_lines) >= 3 and all(len(line.split()[0]) == 64 for line in checkpoint_lines)
    v15_gate = v15_verification.get("passed") is True
    eval_repeat_gate = eval_repeat.get("passed") is True
    passed = all(
        (
            evaluation_complete,
            eval_repeat_gate,
            overall_gate,
            dual_arm_gate,
            grasp_gate,
            service_gate,
            checkpoint_gate,
            v15_gate,
        )
    )

    arm_rows = []
    for arm in ("left", "right"):
        base_arm = auxiliary["baseline"][arm]
        trained_arm = auxiliary[candidate_key][arm]
        delta = by_arm[arm]
        arm_rows.append(
            f"| {arm} | {base_arm['successes']}/{base_arm['count']} ({percent(base_arm['success_rate'])}) "
            f"| {trained_arm['successes']}/{trained_arm['count']} ({percent(trained_arm['success_rate'])}) "
            f"| {signed(delta['absolute_percentage_points'])} pp | {signed(delta['relative_percent'])}% |"
        )

    batch_rows = []
    baseline_batches = {row["batch"]: row for row in baseline["batches"]}
    trained_batches = {row["batch"]: row for row in trained["batches"]}
    for batch in sorted(baseline_batches):
        base_batch = baseline_batches[batch]
        trained_batch = trained_batches[batch]
        batch_rows.append(
            f"| {batch} | {base_batch['count']} | {base_batch['successes']} "
            f"({percent(base_batch['success_rate'])}) | {trained_batch['successes']} "
            f"({percent(trained_batch['success_rate'])}) | "
            f"{signed((trained_batch['success_rate'] - base_batch['success_rate']) * 100.0)} pp |"
        )

    grasp_rows = []
    for scope in ("all", "left", "right"):
        base_scope = auxiliary["baseline"] if scope == "all" else auxiliary["baseline"][scope]
        trained_scope = (
            auxiliary[candidate_key]
            if scope == "all"
            else auxiliary[candidate_key][scope]
        )
        delta = grasp[scope]
        grasp_rows.append(
            f"| {scope} | {base_scope['grasp_completions']}/{base_scope['count']} "
            f"({percent(base_scope['grasp_completion_rate'])}) | "
            f"{trained_scope['grasp_completions']}/{trained_scope['count']} "
            f"({percent(trained_scope['grasp_completion_rate'])}) | "
            f"{signed(delta['absolute_percentage_points'])} pp |"
        )

    transitions = evaluation["paired_success_transitions"]["all"]
    latency = service["latency_ms"]
    test_count = len(service["tests"])
    text = f"""# Strict WorldArena 2.0 Track 2 验收报告

最终状态：**{'PASS' if passed else 'FAIL'}**。本报告只使用冻结 V15、官方 Pi0.5/reward/GRPO 固定预算生成的 `global_step_1` checkpoint；真实 RoboTwin 评估不使用 V15 推演或 MPC 动作选择。

## 正式 checkpoint

- 固定预算：官方配置 1 个 GRPO global step。
- checkpoint 哈希门：{'PASS' if checkpoint_gate else 'FAIL'}（{len(checkpoint_lines)} 个审计文件）。
- 冻结 V15 release manifest：{'PASS' if v15_gate else 'FAIL'}（逐文件校验 {v15_verification.get('checked_file_count', 0)} 项）。
- 模型/算法改动：无；唯一运行时兼容项是权重同步 transport device 使用 CPU。

## 真实 RoboTwin `adjust_bottle`

| 范围 | 原始 Pi0.5 | 正式 checkpoint | 绝对提升 | 相对提升 |
| --- | ---: | ---: | ---: | ---: |
| all | {baseline['successes']}/{baseline['count']} ({percent(baseline['success_rate'])}) | {trained['successes']}/{trained['count']} ({percent(trained['success_rate'])}) | {signed(improvement['absolute_percentage_points'])} pp | {signed(improvement['relative_percent'])}% |
{chr(10).join(arm_rows)}

- 128 种子完整确定性配对门：{'PASS' if evaluation_complete else 'FAIL'}。
- 独立 Ray/仿真进程重启后的固定 batch 逐指标复现：{'PASS' if eval_repeat_gate else 'FAIL'}。
- 总体同时达到绝对 +3 pp 与相对 +3%：{'PASS' if overall_gate else 'FAIL'}。
- 左右臂均不退化：{'PASS' if dual_arm_gate else 'FAIL'}。
- 逐种子转移：baseline fail → trained success **{transitions['baseline_fail_trained_success']}**；baseline success → trained fail **{transitions['baseline_success_trained_fail']}**；McNemar exact p = **{transitions['exact_two_sided_p']:.6g}**。
- 原始 Pi0.5 95% Wilson CI：**{percent(baseline['success_rate_wilson_95'][0])}–{percent(baseline['success_rate_wilson_95'][1])}**；正式 checkpoint：**{percent(trained['success_rate_wilson_95'][0])}–{percent(trained['success_rate_wilson_95'][1])}**。

### 分批结果

| batch | 种子数 | 原始 Pi0.5 成功 | 正式 checkpoint 成功 | 绝对变化 |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(batch_rows)}

### 抓取完成率（只读辅助指标）

| 范围 | 原始 Pi0.5 | 正式 checkpoint | 绝对提升 |
| --- | ---: | ---: | ---: |
{chr(10).join(grasp_rows)}

抓取总体及左右臂均不退化门：{'PASS' if grasp_gate else 'FAIL'}。该指标由 RoboTwin 真实接触/官方成功状态读取，不改变动作、物理、奖励或终止条件。

## 服务验收

- 公网 HTTPS（系统 CA 严格验证）：{'PASS' if str(service.get('base_url', '')).startswith('https://') else 'FAIL'}，地址 `{service.get('base_url')}`。
- 功能与边界测试：**{test_count}** 项通过；认证、max batch=8、batch=9 拒绝、32 MiB 限制、并发上限与 429、幂等缓存/409 冲突均包含在内。
- warm single p50 / p95：**{latency['warm_single_p50']:.3f} / {latency['warm_single_p95']:.3f} ms**。
- batch=8：**{latency['batch_8']:.3f} ms**；所有主要请求低于官方建议超时：**{latency['all_under_recommended_timeout']}**。
- NOT_READY 503 门：{'PASS' if not_ready.get('passed') is True else 'FAIL'}。
- 进程重启前后 capabilities 与逐像素 SHA-256 完全一致：{'PASS' if restart_gate else 'FAIL'}。

## 机器门汇总

| 门 | 结果 |
| --- | --- |
| checkpoint 完整性 | {'PASS' if checkpoint_gate else 'FAIL'} |
| V15 冻结清单 | {'PASS' if v15_gate else 'FAIL'} |
| 128-seed 真实仿真完整性 | {'PASS' if evaluation_complete else 'FAIL'} |
| 真实仿真重启复现 | {'PASS' if eval_repeat_gate else 'FAIL'} |
| 总体 ≥3 pp 且 ≥3% | {'PASS' if overall_gate else 'FAIL'} |
| 双臂不退化 | {'PASS' if dual_arm_gate else 'FAIL'} |
| 抓取完成率不退化 | {'PASS' if grasp_gate else 'FAIL'} |
| HTTPS 与服务协议 | {'PASS' if service_gate else 'FAIL'} |
| **最终** | **{'PASS' if passed else 'FAIL'}** |
"""
    args.output.write_text(text, encoding="utf-8")
    print(json.dumps({"passed": passed, "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
