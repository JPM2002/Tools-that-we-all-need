#!/usr/bin/env python3
"""
Token Cost Calculator + JSON test-cases runner

Adds:
- A folder of JSON "test cases" you can run quickly
- Each JSON can reference a pricing profile + token counts OR text blocks
- Batch runner prints per-case cost and grand totals

Folder layout (suggested):
.
├── token_cost_calc.py   (this file)
├── pricing_profiles.json
└── cases/
    ├── demo_gpt-4.1-mini.json
    ├── demo_gpt-4o-mini.json
    └── ...

Run:
  python token_cost_calc.py

Or run cases directly:
  python token_cost_calc.py --run-cases cases
  python token_cost_calc.py --run-cases cases --pattern gpt-4.1

Case JSON schema (either tokens or texts):
{
  "name": "My test",
  "profile": "gpt-4.1-mini",
  "mode": "tokens" | "texts",

  // if mode="tokens"
  "input_tokens": 1600,
  "cached_input_tokens": 800,
  "output_tokens": 200,

  // if mode="texts"
  "system": "...",
  "history": "...",
  "user": "...",
  "assistant_sample": "...",
  "cached_input_tokens": 0,            // optional; default: estimated system tokens

  // optional scaling
  "requests_per_day": 10000,
  "days": 30
}
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Any, List, Tuple


DEFAULT_STORE = "pricing_profiles.json"


# =========================
# Pricing Profiles
# =========================

@dataclass
class PricingProfile:
    name: str
    input_per_million: float
    cached_input_per_million: float
    output_per_million: float


def load_profiles(path: str = DEFAULT_STORE) -> Dict[str, PricingProfile]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out: Dict[str, PricingProfile] = {}
    for name, data in raw.items():
        out[name] = PricingProfile(
            name=name,
            input_per_million=float(data["input_per_million"]),
            cached_input_per_million=float(data["cached_input_per_million"]),
            output_per_million=float(data["output_per_million"]),
        )
    return out


def save_profiles(profiles: Dict[str, PricingProfile], path: str = DEFAULT_STORE) -> None:
    raw = {name: asdict(p) for name, p in profiles.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2, ensure_ascii=False)


# =========================
# Cost math
# =========================

def dollars_for_tokens(tokens: int, per_million: float) -> float:
    return (tokens / 1_000_000.0) * per_million


def estimate_tokens_from_text(text: str) -> int:
    # Rough rule of thumb: ~4 chars/token for English-ish text.
    chars = len(text or "")
    return max(0, round(chars / 4))


def compute_cost(
    profile: PricingProfile,
    input_tokens_total: int,
    cached_input_tokens: int,
    output_tokens: int,
) -> Dict[str, float]:
    input_tokens_total = max(0, int(input_tokens_total))
    cached_input_tokens = max(0, min(int(cached_input_tokens), input_tokens_total))
    output_tokens = max(0, int(output_tokens))

    noncached_input_tokens = input_tokens_total - cached_input_tokens

    cost_cached = dollars_for_tokens(cached_input_tokens, profile.cached_input_per_million)
    cost_input = dollars_for_tokens(noncached_input_tokens, profile.input_per_million)
    cost_output = dollars_for_tokens(output_tokens, profile.output_per_million)

    total = cost_cached + cost_input + cost_output

    return {
        "cached_input_tokens": float(cached_input_tokens),
        "noncached_input_tokens": float(noncached_input_tokens),
        "output_tokens": float(output_tokens),
        "cost_cached_input_usd": cost_cached,
        "cost_input_usd": cost_input,
        "cost_output_usd": cost_output,
        "total_cost_usd": total,
    }


def money(x: float) -> str:
    return f"${x:,.6f}"


def print_cost_breakdown(result: Dict[str, float]) -> None:
    print("\n--- Cost breakdown ---")
    print(f"Cached input tokens:     {int(result['cached_input_tokens']):,}")
    print(f"Non-cached input tokens: {int(result['noncached_input_tokens']):,}")
    print(f"Output tokens:           {int(result['output_tokens']):,}")
    print()
    print(f"Cached input cost: {money(result['cost_cached_input_usd'])}")
    print(f"Input cost:        {money(result['cost_input_usd'])}")
    print(f"Output cost:       {money(result['cost_output_usd'])}")
    print("----------------------")
    print(f"Total per request: {money(result['total_cost_usd'])}")


# =========================
# CLI prompts (interactive)
# =========================

def prompt_float(msg: str, default: Optional[float] = None) -> float:
    while True:
        s = input(msg).strip()
        if not s and default is not None:
            return default
        try:
            return float(s)
        except ValueError:
            print("Enter a number (e.g., 0.40).")


def prompt_int(msg: str, default: Optional[int] = None) -> int:
    while True:
        s = input(msg).strip()
        if not s and default is not None:
            return default
        try:
            return int(s)
        except ValueError:
            print("Enter an integer (e.g., 1600).")


def choose_profile(profiles: Dict[str, PricingProfile]) -> PricingProfile:
    if not profiles:
        seeded = PricingProfile(
            name="gpt-4.1-mini",
            input_per_million=0.40,
            cached_input_per_million=0.10,
            output_per_million=1.60,
        )
        profiles[seeded.name] = seeded
        save_profiles(profiles)
        print(f"Created default profile '{seeded.name}' in {DEFAULT_STORE}.")

    names = sorted(profiles.keys())
    print("\nAvailable pricing profiles:")
    for i, n in enumerate(names, 1):
        p = profiles[n]
        print(
            f"{i}. {n}  (input={p.input_per_million}/1M, "
            f"cached={p.cached_input_per_million}/1M, output={p.output_per_million}/1M)"
        )

    idx = prompt_int("Choose a profile number: ")
    idx = max(1, min(idx, len(names)))
    return profiles[names[idx - 1]]


def add_or_update_profile(profiles: Dict[str, PricingProfile]) -> None:
    name = input("Profile name (e.g., gpt-4.1-mini): ").strip()
    if not name:
        print("Name cannot be empty.")
        return
    inp = prompt_float("Input $ per 1M tokens: ")
    cached = prompt_float("Cached input $ per 1M tokens: ")
    out = prompt_float("Output $ per 1M tokens: ")

    profiles[name] = PricingProfile(
        name=name,
        input_per_million=inp,
        cached_input_per_million=cached,
        output_per_million=out,
    )
    save_profiles(profiles)
    print(f"Saved profile '{name}' to {DEFAULT_STORE}.")


# =========================
# JSON Cases
# =========================

def ensure_cases_folder(folder: str) -> None:
    os.makedirs(folder, exist_ok=True)


def write_sample_case(folder: str) -> str:
    ensure_cases_folder(folder)
    path = os.path.join(folder, "demo_gpt-4.1-mini.json")
    if os.path.exists(path):
        return path

    sample = {
        "name": "Demo: gpt-4.1-mini tokens",
        "profile": "gpt-4.1-mini",
        "mode": "tokens",
        "input_tokens": 1600,
        "cached_input_tokens": 800,
        "output_tokens": 200,
        "requests_per_day": 10000,
        "days": 30
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)
    return path


def load_case(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def case_to_tokens(case: Dict[str, Any]) -> Tuple[int, int, int]:
    mode = (case.get("mode") or "tokens").lower().strip()

    if mode == "tokens":
        input_tokens = int(case.get("input_tokens", 0))
        cached_tokens = int(case.get("cached_input_tokens", 0))
        output_tokens = int(case.get("output_tokens", 0))
        return input_tokens, cached_tokens, output_tokens

    if mode == "texts":
        system = case.get("system", "") or ""
        history = case.get("history", "") or ""
        user = case.get("user", "") or ""
        assistant = case.get("assistant_sample", "") or ""

        system_t = estimate_tokens_from_text(system)
        history_t = estimate_tokens_from_text(history)
        user_t = estimate_tokens_from_text(user)
        output_t = estimate_tokens_from_text(assistant)

        input_total = system_t + history_t + user_t

        # default cached input tokens to "system prompt tokens" (common scenario)
        cached_default = system_t
        cached_tokens = int(case.get("cached_input_tokens", cached_default))

        return input_total, cached_tokens, output_t

    raise ValueError(f"Unknown case mode: {mode!r}. Use 'tokens' or 'texts'.")


def list_case_files(folder: str, pattern: Optional[str] = None) -> List[str]:
    if not os.path.isdir(folder):
        return []
    files = []
    for name in os.listdir(folder):
        if not name.lower().endswith(".json"):
            continue
        if pattern and pattern.lower() not in name.lower():
            continue
        files.append(os.path.join(folder, name))
    return sorted(files)


def run_cases(folder: str, profiles: Dict[str, PricingProfile], pattern: Optional[str]) -> None:
    files = list_case_files(folder, pattern=pattern)
    if not files:
        print(f"No JSON cases found in: {folder}")
        print("Tip: create one with: --init-cases cases")
        return

    grand_total = 0.0
    print(f"Running {len(files)} case(s) from: {folder}\n")

    for fp in files:
        case = load_case(fp)
        name = case.get("name") or os.path.basename(fp)
        profile_name = case.get("profile")

        if not profile_name or profile_name not in profiles:
            print(f"[SKIP] {name} - unknown profile: {profile_name!r}")
            continue

        profile = profiles[profile_name]
        input_t, cached_t, output_t = case_to_tokens(case)
        result = compute_cost(profile, input_t, cached_t, output_t)

        # optional scaling
        rpd = int(case.get("requests_per_day", 0) or 0)
        days = int(case.get("days", 0) or 0)
        scaled = None
        if rpd > 0 and days > 0:
            scaled = result["total_cost_usd"] * rpd * days

        grand_total += result["total_cost_usd"]

        print(f"Case: {name}")
        print(f"  File: {fp}")
        print(f"  Profile: {profile.name}")
        print(f"  Input: {input_t:,}  (cached {cached_t:,})")
        print(f"  Output: {output_t:,}")
        print(f"  Total/request: {money(result['total_cost_usd'])}")
        if scaled is not None:
            print(f"  Scaled: {rpd:,} req/day × {days} days = {money(scaled)}")
        print()

    print(f"Grand total (sum of per-request totals across cases): {money(grand_total)}")


# =========================c# Interactive calculator
# =========================

def run_interactive() -> None:
    profiles = load_profiles()

    while True:
        print("\n=== Token Cost Calculator ===")
        print("1) Estimate cost (enter token counts)")
        print("2) Estimate cost (paste text, rough token estimate)")
        print("3) Add/update pricing profile")
        print("4) Quit")

        choice = input("Select: ").strip()

        if choice == "4":
            return

        if choice == "3":
            add_or_update_profile(profiles)
            continue

        profile = choose_profile(profiles)

        if choice == "1":
            print("\nEnter token counts (system+history+user).")
            input_tokens = prompt_int("Total input tokens: ")
            cached_tokens = prompt_int("Cached input tokens: ", default=0)
            output_tokens = prompt_int("Output tokens: ")

        elif choice == "2":
            print("\nPaste the texts. Press Enter on an empty line to finish each section.\n")

            def read_multiline(label: str) -> str:
                print(f"{label} (multiline):")
                lines = []
                while True:
                    line = input()
                    if line == "":
                        break
                    lines.append(line)
                return "\n".join(lines)

            system = read_multiline("SYSTEM PROMPT")
            history = read_multiline("HISTORY")
            user = read_multiline("USER MESSAGE")
            assistant = read_multiline("ASSISTANT REPLY (expected/sample)")

            system_t = estimate_tokens_from_text(system) if system else 0
            history_t = estimate_tokens_from_text(history) if history else 0
            user_t = estimate_tokens_from_text(user) if user else 0
            output_tokens = estimate_tokens_from_text(assistant) if assistant else 0

            input_tokens = system_t + history_t + user_t

            print("\n--- Estimated tokens (rough) ---")
            print(f"System:  {system_t:,}")
            print(f"History: {history_t:,}")
            print(f"User:    {user_t:,}")
            print(f"Input total:  {input_tokens:,}")
            print(f"Output total: {output_tokens:,}")

            cached_tokens = prompt_int(
                f"How many input tokens are cached? (common: system={system_t}) ",
                default=system_t,
            )

        else:
            print("Invalid selection.")
            continue

        result = compute_cost(profile, input_tokens, cached_tokens, output_tokens)
        print_cost_breakdown(result)

        req_per_day = prompt_int("\nRequests per day for scaling estimate (0 to skip): ", default=0)
        if req_per_day > 0:
            days = prompt_int("Days to estimate (e.g., 30): ", default=30)
            total = result["total_cost_usd"] * req_per_day * days
            print(f"\nEstimated total for {req_per_day:,} req/day over {days} days: {money(total)}")


# =========================
# Main
# =========================

def main() -> None:
    parser = argparse.ArgumentParser(description="Token cost calculator + JSON case runner")
    parser.add_argument("--run-cases", metavar="FOLDER", help="Run all JSON cases in a folder (non-interactive)")
    parser.add_argument("--pattern", metavar="SUBSTRING", help="Only run case files whose filename contains this substring")
    parser.add_argument("--init-cases", metavar="FOLDER", help="Create a cases folder and write a sample case JSON")
    args = parser.parse_args()

    if args.init_cases:
        folder = args.init_cases
        ensure_cases_folder(folder)
        sample_path = write_sample_case(folder)
        print(f"Created sample case: {sample_path}")
        print("Now run:")
        print(f"  python {os.path.basename(__file__)} --run-cases {folder}")
        return

    if args.run_cases:
        profiles = load_profiles()
        if not profiles:
            # seed default so runner can work out-of-the-box
            profiles["gpt-4.1-mini"] = PricingProfile(
                name="gpt-4.1-mini",
                input_per_million=0.40,
                cached_input_per_million=0.10,
                output_per_million=1.60,
            )
            save_profiles(profiles)

        run_cases(args.run_cases, profiles, pattern=args.pattern)
        return

    run_interactive()


if __name__ == "__main__":
    main()
