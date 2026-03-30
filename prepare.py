from __future__ import annotations

import argparse
import json
import random
from datetime import date
from pathlib import Path

from scripts.deepseek_distill import DeepSeekDistiller
from scripts.prepare_mlx_dataset import prepare_dataset


DEFAULT_TEACHER_MODEL = "deepseek-chat"
DEFAULT_PROMPT_TEMPLATE_VERSION = "v2.deepseek_distill"
DEFAULT_SEED_FILE = Path("data/seed_questions.txt")

BASE_SEED_QUESTIONS = [
    "What is a Kubernetes deployment?",
    "How does a service differ from an ingress?",
    "Why use a liveness probe?",
    "What problem does HorizontalPodAutoscaler solve?",
    "How should secrets be handled in production clusters?",
    "When should you use a StatefulSet instead of a Deployment?",
    "How do readiness probes affect traffic routing?",
    "What causes CrashLoopBackOff and how do you debug it?",
    "How should ConfigMaps be managed across environments?",
    "What is the difference between requests and limits in Kubernetes?",
    "How do rolling updates work in Kubernetes?",
    "How should you expose an internal service securely?",
]

DISTILLATION_VARIANTS = (
    "foundation",
    "practical",
    "comparison",
    "failure_mode",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare DeepSeek-distilled dataset for MLX autoresearch.")
    parser.add_argument("--questions", default=str(DEFAULT_SEED_FILE))
    parser.add_argument("--num-samples", type=int, default=120)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variants-per-question", type=int, default=4)
    parser.add_argument("--teacher-model", default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--prompt-template-version", default=DEFAULT_PROMPT_TEMPLATE_VERSION)
    parser.add_argument("--base-url", help="OpenAI-compatible base URL for DeepSeek, e.g. https://api.deepseek.com/v1")
    parser.add_argument("--api-key", help="API key for the OpenAI-compatible DeepSeek endpoint")
    return parser.parse_args()


def ensure_seed_file(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(BASE_SEED_QUESTIONS) + "\n", encoding="utf-8")


def load_questions(path: Path) -> list[str]:
    ensure_seed_file(path)
    questions = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if questions:
        return questions
    return list(BASE_SEED_QUESTIONS)


def classify_topic(question: str) -> dict:
    q = question.lower()
    if "deployment" in q and "statefulset" not in q:
        return {
            "topic": "deployment",
            "core": "A Deployment keeps a declarative replica set running and manages rolling updates for stateless workloads.",
            "example": "A web API can be rolled from image v1 to v2 without replacing the Service endpoint manually.",
            "caveat": "It is the wrong controller when pod identity, ordered rollout, or persistent identity must be preserved.",
        }
    if "statefulset" in q:
        return {
            "topic": "statefulset",
            "core": "A StatefulSet gives pods stable network identity and stable persistent volume claims across restarts and reschedules.",
            "example": "Databases such as PostgreSQL or Kafka brokers often need predictable pod identity and durable storage.",
            "caveat": "StatefulSet adds operational cost and slower rollout behavior, so it should not replace a Deployment for stateless apps.",
        }
    if "service differ from an ingress" in q or ("service" in q and "ingress" in q):
        return {
            "topic": "service_ingress",
            "core": "A Service gives stable discovery and load-balancing to pods, while an Ingress defines HTTP routing rules from outside the cluster.",
            "example": "A Service can front multiple API pods internally, and an Ingress can route `/api` traffic to that Service.",
            "caveat": "Ingress is not a Service replacement because east-west traffic and non-HTTP traffic still need a Service.",
        }
    if "liveness probe" in q:
        return {
            "topic": "liveness_probe",
            "core": "A liveness probe tells Kubernetes when a container is unhealthy enough to restart instead of serving degraded traffic forever.",
            "example": "If an app deadlocks but the process still exists, a failing HTTP liveness probe can trigger a restart.",
            "caveat": "Aggressive thresholds create restart loops and can hide slow startup problems that should be handled with a startup probe.",
        }
    if "readiness probe" in q:
        return {
            "topic": "readiness_probe",
            "core": "A readiness probe controls whether a pod receives traffic, so it protects users from pods that are up but not ready to serve.",
            "example": "A pod can wait for database migrations to finish before its readiness probe returns success.",
            "caveat": "Using readiness as a health restart signal is a mistake because failed readiness removes traffic but does not restart the pod.",
        }
    if "horizontalpodautoscaler" in q or "autoscaler" in q:
        return {
            "topic": "hpa",
            "core": "HorizontalPodAutoscaler increases or decreases replicas based on observed metrics such as CPU, memory, or custom workload signals.",
            "example": "An API can scale from 3 to 10 replicas when request volume drives CPU past its target utilization.",
            "caveat": "HPA reacts poorly when requests and limits are wrong or when the chosen metric does not reflect saturation.",
        }
    if "secrets" in q:
        return {
            "topic": "secrets",
            "core": "Kubernetes Secrets store sensitive configuration separately from images and support controlled injection into pods.",
            "example": "Database credentials can be mounted as env vars or volumes and rotated without rebuilding the container image.",
            "caveat": "Base64 is encoding, not encryption, so production setups still need RBAC, secret rotation, and preferably an external secret manager.",
        }
    if "configmap" in q:
        return {
            "topic": "configmap",
            "core": "ConfigMaps separate non-sensitive runtime configuration from the application image so the same image can run across environments.",
            "example": "A service can load log level and upstream URLs from a ConfigMap while using the same container image in staging and prod.",
            "caveat": "Large or fast-changing ConfigMaps create drift and restart surprises if rollout strategy is not explicit.",
        }
    if "crashloopbackoff" in q:
        return {
            "topic": "crashloopbackoff",
            "core": "CrashLoopBackOff is a symptom that the container keeps exiting and Kubernetes is applying backoff before retrying.",
            "example": "A missing env var can make the app crash instantly and the pod will repeatedly restart with increasing delay.",
            "caveat": "Treating CrashLoopBackOff as the root cause is a mistake because the real issue is usually in logs, events, config, or startup dependencies.",
        }
    if "requests and limits" in q:
        return {
            "topic": "resources",
            "core": "Requests drive scheduling guarantees and limits cap usage, so together they shape placement, eviction risk, and throttling behavior.",
            "example": "A pod with CPU request 250m and limit 500m reserves enough scheduling capacity while allowing short bursts.",
            "caveat": "Setting CPU limits too low can throttle latency-sensitive workloads, while missing memory limits increases node pressure risk.",
        }
    if "rolling updates" in q:
        return {
            "topic": "rolling_update",
            "core": "Rolling updates replace pods gradually so a new version can be released without taking the whole service down at once.",
            "example": "A Deployment can scale up new pods before terminating old ones according to maxSurge and maxUnavailable.",
            "caveat": "A rollout can still fail if readiness probes are weak or if the app is not backward compatible with live traffic.",
        }
    if "internal service securely" in q:
        return {
            "topic": "internal_service_security",
            "core": "Internal exposure should prefer cluster-local networking, network policy, and identity-aware access instead of public endpoints.",
            "example": "A ClusterIP plus NetworkPolicy can expose an internal API only to a specific namespace or workload set.",
            "caveat": "Security groups alone are not enough if application credentials, namespace boundaries, and east-west policy are ignored.",
        }
    return {
        "topic": "kubernetes_general",
        "core": "The answer should explain the Kubernetes concept clearly, connect it to cluster operations, and show how it behaves under load or failure.",
        "example": "A production team typically validates the behavior with events, rollout status, logs, and service-level impact.",
        "caveat": "Conceptual knowledge alone is insufficient if the operator cannot map it to concrete debugging and deployment decisions.",
    }


def make_variant_prompt(question: str, variant: str) -> str:
    variant_suffix = {
        "foundation": "Answer for a junior platform engineer.",
        "practical": "Include one production example and one operational checklist.",
        "comparison": "Contrast it with the closest alternative and explain the trade-off.",
        "failure_mode": "Focus on what breaks in production and how to debug it.",
    }
    return f"{question} {variant_suffix[variant]}"


def build_teacher_completion(question: str, variant: str, topic_info: dict) -> str:
    core = topic_info["core"]
    example = topic_info["example"]
    caveat = topic_info["caveat"]

    variant_block = {
        "foundation": (
            "Reasoning: Start from the control-plane behavior, then connect it to what an on-call engineer observes.\n"
            f"Answer: {core}\n"
            f"Example: {example}\n"
            f"Operational caveat: {caveat}"
        ),
        "practical": (
            "Reasoning: Optimize for production usefulness rather than textbook completeness.\n"
            f"Answer: {core}\n"
            f"Practical example: {example}\n"
            "Checklist: Verify manifests, rollout status, events, and dependency readiness before calling the change healthy.\n"
            f"Operational caveat: {caveat}"
        ),
        "comparison": (
            "Reasoning: The learner needs the boundary of the concept, not just its definition.\n"
            f"Answer: {core}\n"
            f"Comparison: {example}\n"
            f"Trade-off: {caveat}"
        ),
        "failure_mode": (
            "Reasoning: Explain the normal path first, then identify how it fails under bad config or dependency pressure.\n"
            f"Answer: {core}\n"
            f"Failure signal: {caveat}\n"
            "Troubleshooting: Inspect pod events, container logs, rollout status, recent config changes, and upstream dependencies before restarting blindly.\n"
            f"Operational example: {example}"
        ),
    }

    return (
        f"Question: {question}\n"
        "Teacher model: deepseek-template-fallback\n"
        f"Distillation style: {variant}\n"
        f"{variant_block[variant]}"
    )


def expand_questions(base_questions: list[str], limit: int, seed: int, variants_per_question: int) -> list[tuple[str, str, int]]:
    rng = random.Random(seed)
    selected_variants = DISTILLATION_VARIANTS[: max(1, min(variants_per_question, len(DISTILLATION_VARIANTS)))]

    expanded: list[tuple[str, str, int]] = []
    for seed_index, question in enumerate(base_questions, start=1):
        expanded.append((question, "foundation", seed_index))
        for variant in selected_variants:
            if variant == "foundation":
                continue
            expanded.append((question, variant, seed_index))

    rng.shuffle(expanded)
    expanded.sort(key=lambda item: (item[2], DISTILLATION_VARIANTS.index(item[1]) if item[1] in DISTILLATION_VARIANTS else 99))
    return expanded[:limit]


def build_records(
    questions: list[str],
    limit: int,
    augment: bool,
    seed: int,
    teacher_model: str,
    prompt_template_version: str,
    variants_per_question: int,
    distiller: DeepSeekDistiller,
) -> list[dict]:
    rng = random.Random(seed)
    records: list[dict] = []
    expanded = expand_questions(questions, limit=limit, seed=seed, variants_per_question=variants_per_question)

    for idx, (seed_question, variant, seed_index) in enumerate(expanded, start=1):
        topic_info = classify_topic(seed_question)
        prompt = make_variant_prompt(seed_question, variant)
        distill_result = distiller.distill(
            question=seed_question,
            prompt=prompt,
            variant=variant,
            topic_info=topic_info,
        )
        completion = distill_result.content or build_teacher_completion(seed_question, variant, topic_info)
        text = f"{prompt}\n{completion}"
        record_teacher_model = distill_result.teacher_model_version if distill_result.used_api else "deepseek-template-fallback"

        record = {
            "id": f"distill_{idx:05d}",
            "prompt": prompt,
            "completion": completion,
            "text": text,
            "source": distill_result.source,
            "teacher_model": record_teacher_model,
            "teacher_model_version": record_teacher_model,
            "distillation_variant": variant,
            "prompt_template_version": prompt_template_version,
            "distillation_date": date.today().isoformat(),
            "seed_question": seed_question,
            "seed_index": seed_index,
            "seed_group": topic_info["topic"],
            "seed_trace": {
                "seed": seed,
                "split_seed": None,
                "variant": variant,
            },
        }
        records.append(record)

        if augment:
            records.append(
                {
                    "id": f"distill_{idx:05d}_aug",
                    "prompt": f"{prompt} Add one anti-pattern and one safer alternative.",
                    "completion": (
                        f"{completion}\n"
                        "Anti-pattern: Treat the first symptom as the root cause and restart without verifying rollout state, events, or dependency health.\n"
                        "Safer alternative: Reconstruct the failure timeline from config changes, probes, events, logs, and service impact before remediation."
                    ),
                    "text": (
                        f"{prompt} Add one anti-pattern and one safer alternative.\n"
                        f"{completion}\n"
                        "Anti-pattern: Treat the first symptom as the root cause and restart without verifying rollout state, events, or dependency health.\n"
                        "Safer alternative: Reconstruct the failure timeline from config changes, probes, events, logs, and service impact before remediation."
                    ),
                    "source": f"{distill_result.source}_augment",
                    "teacher_model": record_teacher_model,
                    "teacher_model_version": record_teacher_model,
                    "distillation_variant": f"{variant}_augment",
                    "prompt_template_version": prompt_template_version,
                    "distillation_date": date.today().isoformat(),
                    "seed_question": seed_question,
                    "seed_index": seed_index,
                    "seed_group": topic_info["topic"],
                    "seed_trace": {
                        "seed": seed,
                        "split_seed": None,
                        "variant": f"{variant}_augment",
                        "augmentation_seed": rng.randint(0, 10_000),
                    },
                }
            )

    return records


def main() -> None:
    args = parse_args()
    questions = load_questions(Path(args.questions))
    distiller = DeepSeekDistiller(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.teacher_model,
    )
    records = build_records(
        questions=questions,
        limit=args.num_samples,
        augment=args.augment,
        seed=args.seed,
        teacher_model=args.teacher_model,
        prompt_template_version=args.prompt_template_version,
        variants_per_question=args.variants_per_question,
        distiller=distiller,
    )

    for record in records:
        record["seed_trace"]["split_seed"] = args.split_seed

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "distilled_latest.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    metadata = prepare_dataset(
        raw_path=raw_path,
        output_dir=Path("data"),
        split_seed=args.split_seed,
    )
    print(f"dataset_id:      {metadata['dataset_id']}")
    print(f"distill_mode:    {'api' if distiller.is_configured else 'template_fallback'}")
    print(f"teacher_model:   {metadata['teacher_model_version']}")
    print(f"train_samples:   {metadata['train_samples']}")
    print(f"valid_samples:   {metadata['valid_samples']}")
    print(f"test_samples:    {metadata['test_samples']}")


if __name__ == "__main__":
    main()
