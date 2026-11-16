#!/usr/bin/env python
"""Batch image-to-image: one FLUX generation per prompt file in a folder.

Point it at a folder of prompts and a reference image, and it produces one
image per prompt:

    python scripts/batch_i2i.py prompts/ --image assets/room.png --out out/

Every prompt file in the folder becomes one request. References are the
image-to-image half, and come from one of two places:

    --image PATH        used for every prompt (repeatable, up to 8)
    (nothing)           paired by filename - `sofa.txt` picks up `sofa.png`,
                        plus `sofa_2.png`, `sofa_3.png` ... if they exist,
                        from the prompt folder or from --images-dir

The two combine: --image references come first, paired ones after, which is
the order FLUX sees them as `image 1`, `image 2` and so on.

The work is the same submit/poll/download the service already does, so this
reuses `app.flux.FluxClient` rather than re-implementing it - the API key,
endpoint and timeouts all come from the usual .env via `config.Config`.

Runs are resumable: a prompt whose output already exists is skipped unless
--overwrite is passed, and every attempt is appended to `manifest.jsonl` in the
output folder.

NOTE ON MODELS
    The reference is sent as `input_image` (`input_image_2` ...), which is what
    flux-2-pro and flux-kontext-pro accept. FLUX_API_URL decides the model, and
    --model overrides it for one run. flux-pro-1.1 is not image-to-image in this
    sense - it takes `image_prompt` instead - so it would ignore the reference.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.errors import AppError, UpstreamError  # noqa: E402
from app.flux import (  # noqa: E402
    MAX_REFERENCES,
    ContentModerated,
    FluxClient,
    GenerationTimeout,
    ModelUnavailable,
)
from config import Config  # noqa: E402

log = logging.getLogger("batch_i2i")

# A prompt file is text: a known suffix, or no suffix at all - the master prompt
# set in app/ is stored without one, and that is a normal way to keep them.
PROMPT_SUFFIXES = {".txt", ".md", ".prompt", ""}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# FLUX.2's text encoder truncates at 512 tokens; roughly four characters to the
# token, so anything past this is being silently cut and is worth saying so.
PROMPT_WARN_CHARS = 2000

# Retried with backoff. Moderation and a missing key are not - they will not
# come good on a second attempt.
RETRYABLE = (UpstreamError, GenerationTimeout)


@dataclass
class Job:
    prompt_path: Path
    prompt: str
    out_path: Path
    references: list = field(default_factory=list)  # list[Path]
    seed: int = None
    variant: int = 1


@dataclass
class Result:
    job: Job
    status: str  # "ok" | "skipped" | "failed"
    seconds: float = 0.0
    message: str = ""
    attempts: int = 1


# -- discovery -----------------------------------------------------------


def find_prompts(folder, recursive, pattern):
    if not folder.is_dir():
        raise SystemExit(f"not a folder: {folder}")

    if pattern:
        found = folder.rglob(pattern) if recursive else folder.glob(pattern)
        files = [p for p in found if p.is_file()]
    else:
        walk = folder.rglob("*") if recursive else folder.iterdir()
        files = [
            p
            for p in walk
            if p.is_file()
            and p.suffix.lower() in PROMPT_SUFFIXES
            and not p.name.startswith(".")
        ]
    return sorted(files)


def paired_references(prompt_path, images_dir):
    """Images belonging to one prompt: sofa.png, sofa_2.png, sofa-b.jpg.

    Anything in `images_dir` whose name starts with the prompt's stem counts, so
    a prompt can carry several references without a naming scheme to remember.
    """
    matches = [
        p
        for p in images_dir.glob(f"{prompt_path.stem}*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]
    return sorted(matches)


def load_prompt(path):
    # Prompts are hand-written and land in whatever the editor saved them as;
    # utf-8-sig also eats the BOM Notepad leaves behind.
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=encoding).strip()
        except UnicodeDecodeError:
            continue
    raise SystemExit(f"could not decode {path} as text")


def build_jobs(args, config):
    prompts = find_prompts(args.prompts, args.recursive, args.pattern)
    if not prompts:
        raise SystemExit(f"no prompt files found in {args.prompts}")

    images_dir = args.images_dir or args.prompts
    # Outputs written next to the references would be picked up as references
    # themselves on the next run, quietly feeding generations back in.
    for folder, what in ((args.prompts, "prompt"), (images_dir, "reference")):
        if args.out.resolve() == folder.resolve():
            raise SystemExit(
                f"--out is the {what} folder; write somewhere else so generated "
                "images are not read back in as references"
            )

    shared = [Path(p) for p in args.image]
    for path in shared:
        if not path.is_file():
            raise SystemExit(f"--image does not exist: {path}")

    extension = (args.format or config.FLUX_OUTPUT_FORMAT).lower()
    jobs, skipped_empty, skipped_unreferenced = [], [], []

    for prompt_path in prompts:
        text = load_prompt(prompt_path)
        if not text:
            skipped_empty.append(prompt_path)
            continue

        references = shared + [
            p for p in paired_references(prompt_path, images_dir) if p not in shared
        ]
        if not references and not args.text_only:
            skipped_unreferenced.append(prompt_path)
            continue
        if len(references) > MAX_REFERENCES:
            log.warning(
                "%s has %d references, keeping the first %d",
                prompt_path.name,
                len(references),
                MAX_REFERENCES,
            )
            references = references[:MAX_REFERENCES]

        if len(text) > PROMPT_WARN_CHARS:
            log.warning(
                "%s is %d characters; FLUX truncates the prompt at 512 tokens",
                prompt_path.name,
                len(text),
            )

        # Relative to the prompt folder, so --recursive keeps its subfolders.
        stem_path = prompt_path.relative_to(args.prompts).with_suffix("")
        for variant in range(1, args.variants + 1):
            suffix = "" if args.variants == 1 else f"_{variant}"
            jobs.append(
                Job(
                    prompt_path=prompt_path,
                    prompt=text,
                    out_path=args.out / f"{stem_path}{suffix}.{extension}",
                    references=references,
                    seed=None if args.seed is None else args.seed + variant - 1,
                    variant=variant,
                )
            )

    for path in skipped_empty:
        log.warning("skipping %s: empty", path.name)
    for path in skipped_unreferenced:
        log.warning(
            "skipping %s: no reference image (looked for %s.* in %s; "
            "pass --image, or --text-only to generate from the prompt alone)",
            path.name,
            path.stem,
            images_dir,
        )
    return jobs


# -- generation ----------------------------------------------------------

_local = threading.local()


def client_for(config):
    """One client - and so one requests.Session - per worker thread."""
    client = getattr(_local, "client", None)
    if client is None:
        client = _local.client = FluxClient(config)
    return client


def run_job(job, config, args):
    if job.out_path.exists() and not args.overwrite:
        return Result(job, "skipped", message="output exists")

    references = []
    try:
        for path in job.references:
            with Image.open(path) as image:
                image.load()
                references.append(image.copy())
    except Exception as exc:  # noqa: BLE001 - a bad file should not kill the run
        return Result(job, "failed", message=f"could not read reference: {exc}")

    client = client_for(config)
    started = time.monotonic()

    for attempt in range(1, args.retries + 2):
        try:
            image = client.generate(
                job.prompt,
                references=references,
                seed=job.seed,
                aspect_ratio=args.aspect_ratio,
                output_format=args.format or config.FLUX_OUTPUT_FORMAT,
                safety_tolerance=args.safety_tolerance,
            )
        except RETRYABLE as exc:
            if attempt > args.retries:
                return Result(job, "failed", time.monotonic() - started, str(exc), attempt)
            backoff = args.retry_wait * (2 ** (attempt - 1))
            log.warning("%s failed (%s), retrying in %.0fs", job.out_path.name, exc, backoff)
            time.sleep(backoff)
        except ContentModerated as exc:
            return Result(job, "failed", time.monotonic() - started, str(exc), attempt)
        except AppError as exc:
            return Result(job, "failed", time.monotonic() - started, str(exc), attempt)
        else:
            job.out_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(job.out_path)
            if args.save_prompt:
                job.out_path.with_suffix(".prompt.txt").write_text(job.prompt, encoding="utf-8")
            return Result(job, "ok", time.monotonic() - started, attempts=attempt)

    return Result(job, "failed", time.monotonic() - started, "exhausted retries")


# -- reporting -----------------------------------------------------------


class Manifest:
    """Append-only record of every attempt, one JSON object per line."""

    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()

    def write(self, result):
        row = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "prompt_file": str(result.job.prompt_path),
            "output": str(result.job.out_path),
            "references": [str(p) for p in result.job.references],
            "seed": result.job.seed,
            "status": result.status,
            "seconds": round(result.seconds, 1),
            "attempts": result.attempts,
        }
        if result.message:
            row["message"] = result.message
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def report(index, total, result):
    name = result.job.out_path.name
    if result.status == "ok":
        print(f"[{index}/{total}] ok      {name}  ({result.seconds:.0f}s)")
    elif result.status == "skipped":
        print(f"[{index}/{total}] skip    {name}  ({result.message})")
    else:
        print(f"[{index}/{total}] FAILED  {name}  {result.message}")


# -- cli -----------------------------------------------------------------

EPILOG = """examples:
  # one room photo, every prompt in the folder
  python scripts/batch_i2i.py prompts/ --image assets/room.png --out out/

  # each prompt paired with its own image by filename
  python scripts/batch_i2i.py prompts/ --out out/ --workers 4

  # three seeded variants each, resumable
  python scripts/batch_i2i.py prompts/ --image room.png --variants 3 --seed 1000
"""


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate one FLUX image per prompt file in a folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG,
    )
    parser.add_argument("prompts", type=Path, help="folder of prompt files")
    parser.add_argument(
        "--out", type=Path, default=Path("out"), help="output folder (default: out)"
    )
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        metavar="PATH",
        help="reference image used for every prompt; repeat for up to 8",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        help="where per-prompt references live (default: the prompt folder)",
    )
    parser.add_argument(
        "--pattern",
        help="glob for prompt files, e.g. *.txt (default: every text file)",
    )
    parser.add_argument("--recursive", action="store_true", help="descend into subfolders")
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="generate prompts that have no reference instead of skipping them",
    )
    parser.add_argument("--variants", type=int, default=1, help="images per prompt (default: 1)")
    parser.add_argument("--seed", type=int, help="base seed; variants take seed, seed+1, ...")
    parser.add_argument("--aspect-ratio", help="e.g. 16:9 (default: follow the input image)")
    parser.add_argument("--format", choices=("png", "jpeg"), help="output format")
    parser.add_argument("--safety-tolerance", type=int, help="0 (strictest) to 6, per the API")
    parser.add_argument(
        "--model",
        help="model slug appended to https://api.bfl.ai/v1/, e.g. flux-2-pro; "
        "overrides FLUX_API_URL",
    )
    parser.add_argument("--api-url", help="full endpoint URL, overriding --model")
    parser.add_argument("--workers", type=int, default=3, help="parallel requests (default: 3)")
    parser.add_argument("--retries", type=int, default=2, help="retries per image (default: 2)")
    parser.add_argument("--retry-wait", type=float, default=5.0, help="first backoff, doubling")
    parser.add_argument(
        "--overwrite", action="store_true", help="regenerate images that already exist"
    )
    parser.add_argument(
        "--save-prompt", action="store_true", help="write the prompt beside each image"
    )
    parser.add_argument("--limit", type=int, help="stop after this many jobs")
    parser.add_argument("--dry-run", action="store_true", help="list what would be generated")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    config = Config()
    if args.api_url:
        config.FLUX_API_URL = args.api_url
    elif args.model:
        config.FLUX_API_URL = f"https://api.bfl.ai/v1/{args.model.strip('/')}"

    jobs = build_jobs(args, config)
    if args.limit:
        jobs = jobs[: args.limit]
    if not jobs:
        raise SystemExit("nothing to do")

    print(f"{len(jobs)} image(s) from {args.prompts} -> {args.out}")
    print(f"model: {config.FLUX_API_URL}")

    if args.dry_run:
        for job in jobs:
            refs = ", ".join(p.name for p in job.references) or "none"
            seed = "" if job.seed is None else f"  seed={job.seed}"
            state = "  [exists]" if job.out_path.exists() and not args.overwrite else ""
            print(f"  {job.out_path}  <- {job.prompt_path.name}  refs: {refs}{seed}{state}")
        return 0

    # Fail before any work - but a fully finished run has nothing to submit, so
    # resuming one does not need a key.
    pending = [j for j in jobs if args.overwrite or not j.out_path.exists()]
    if pending and not FluxClient(config).configured:
        raise SystemExit(
            "FLUX_API_KEY is not set. Get a key from https://dashboard.bfl.ai "
            "and put it in .env"
        )

    manifest = Manifest(args.out / "manifest.jsonl")
    counts = {"ok": 0, "skipped": 0, "failed": 0}
    started = time.monotonic()

    workers = max(1, min(args.workers, len(jobs)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_job, job, config, args): job for job in jobs}
        try:
            for index, future in enumerate(as_completed(futures), start=1):
                try:
                    result = future.result()
                except ModelUnavailable as exc:
                    # The key is bad or gone; every remaining job says the same.
                    print(f"\naborting: {exc}")
                    for pending in futures:
                        pending.cancel()
                    return 1
                counts[result.status] += 1
                report(index, len(jobs), result)
                manifest.write(result)
        except KeyboardInterrupt:
            print("\ninterrupted; finishing in-flight requests")
            for pending in futures:
                pending.cancel()
            return 130

    elapsed = time.monotonic() - started
    print(
        f"\ndone in {elapsed:.0f}s: {counts['ok']} generated, "
        f"{counts['skipped']} skipped, {counts['failed']} failed"
    )
    print(f"manifest: {manifest.path}")
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
