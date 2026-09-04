"""Generate the GR00T-H N1.7 language-analysis report.

Outputs include standalone SVG figures, an interactive HTML gallery, and a CSV
of representative instructions by semantic length. The visualizer uses only
the Python standard library.

Usage:
    python visualize_language_analysis.py
    python visualize_language_analysis.py --top-n 20
    python visualize_language_analysis.py --input-dir extracted_language \
        --output-dir extracted_language/visualizations
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from html import escape
import math
from pathlib import Path
import re
import textwrap


# -----------------------------------------------------------------------------
# Visual system
# -----------------------------------------------------------------------------

BG = "#F3F0E8"
CARD = "#FFFEFB"
INK = "#18323E"
MUTED = "#667780"
GRID = "#DCE3E1"
TEAL = "#17736D"
MINT = "#68B8A8"
BLUE = "#477EA8"
CORAL = "#DC765F"
AMBER = "#DEA94A"
PURPLE = "#796DAA"
PALE_TEAL = "#D9ECE7"
PALE_CORAL = "#F7DFD8"
FONT = "Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif"

LEXICAL_COLORS = [TEAL, CORAL, BLUE, PURPLE]


# -----------------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------------

def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run extract_gr00t_h_n17_training_language.py "
            "and analyse_language.py first."
        )
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], column: str) -> float:
    value = float(row[column])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite value in {column}: {row[column]!r}")
    return value


def as_int(row: dict[str, str], column: str) -> int:
    return int(row[column])


def require_unit_sum(rows: list[dict[str, str]], column: str, label: str) -> None:
    total = sum(as_float(row, column) for row in rows)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"{label} weights sum to {total:.12f}, not 1")


def title_case_identifier(value: str) -> str:
    aliases = {
        "cmr": "CMR",
        "jhu": "JHU",
        "lcsr": "LCSR",
        "lscr": "LSCR",
        "dvrk": "dVRK",
        "ucsd": "UCSD",
        "ucb": "UC Berkeley",
        "tud": "TUD",
        "ustc": "USTC",
        "n17": "N1.7",
    }
    words = value.replace("_", " ").split()
    return " ".join(aliases.get(word.lower(), word.capitalize()) for word in words)


def format_rule_label(rule: str) -> str:
    labels = {
        "strip_cmr_state_template": "Remove CMR state template",
        "strip_cmr_do_carrier": "Remove CMR 'do a/an' carrier",
        "strip_rob_tool_template": "Remove Rob tool template",
        "strip_rob_process_carrier": "Remove Rob process carrier",
        "strip_rob_run_identifier": "Remove Rob run identifier",
        "split_rob_concatenated_phases": "Split joined Rob phases",
        "insert_space_after_punctuation": "Repair punctuation spacing",
        "strip_terminal_punctuation": "Remove terminal punctuation",
        "split_camel_case": "Split CamelCase",
        "replace_underscores": "Replace underscores",
        "normalize_whitespace": "Normalize whitespace",
        "lowercase_semantic_text": "Lowercase task text",
        "drop_invalid_placeholder": "Drop invalid placeholder",
    }
    if rule in labels:
        return labels[rule]
    if ":" in rule:
        operation, transformation = rule.split(":", 1)
        source, separator, target = transformation.partition("->")
        if operation == "lemmatize_inflection" and separator:
            return f"Lemmatize {source} → {target}"
        if operation == "canonicalize_action_nominalization" and separator:
            return f"Action alias {source} → {target}"
        if operation == "expand_compound":
            return f"Split compound {transformation}"
    return title_case_identifier(rule)


def shorten(value: str, limit: int) -> str:
    value = " ".join(value.split())
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"


def fmt_percent(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}%"


def fmt_compact(value: float) -> str:
    absolute = abs(value)
    if absolute >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if absolute >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"


# -----------------------------------------------------------------------------
# SVG primitives
# -----------------------------------------------------------------------------

def svg_document(width: int, height: int, body: list[str]) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">\n'
        f'<rect width="{width}" height="{height}" fill="{BG}"/>\n'
        "<style>\n"
        f"text {{ font-family: {FONT}; fill: {INK}; }}\n"
        ".title { font-size: 29px; font-weight: 750; letter-spacing: -0.4px; }\n"
        ".subtitle { font-size: 14px; fill: #667780; }\n"
        ".panel-title { font-size: 17px; font-weight: 700; }\n"
        ".label { font-size: 12px; }\n"
        ".small { font-size: 10.5px; fill: #667780; }\n"
        ".value { font-size: 11.5px; font-weight: 650; }\n"
        "</style>\n"
        + "\n".join(body)
        + "\n</svg>\n"
    )


def rect(
    x: float,
    y: float,
    width: float,
    height: float,
    fill: str,
    radius: float = 0,
    stroke: str | None = None,
) -> str:
    stroke_attr = f' stroke="{stroke}"' if stroke else ""
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(width, 0):.2f}" '
        f'height="{max(height, 0):.2f}" rx="{radius:.2f}" fill="{fill}"{stroke_attr}/>'
    )


def text(
    x: float,
    y: float,
    value: str,
    css_class: str = "label",
    anchor: str = "start",
    fill: str | None = None,
) -> str:
    fill_attr = f' fill="{fill}"' if fill else ""
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" class="{css_class}" '
        f'text-anchor="{anchor}"{fill_attr}>{escape(value)}</text>'
    )


def line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: str = GRID,
    width: float = 1,
    dash: str | None = None,
) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{color}" stroke-width="{width}"{dash_attr}/>'
    )


def circle(x: float, y: float, radius: float, fill: str, stroke: str | None = None) -> str:
    stroke_attr = f' stroke="{stroke}" stroke-width="1.5"' if stroke else ""
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{fill}"{stroke_attr}/>'


def add_header(body: list[str], title_value: str, subtitle: str, width: int) -> None:
    body.append(text(58, 62, title_value, "title"))
    body.append(text(58, 90, subtitle, "subtitle"))
    body.append(line(58, 112, width - 58, 112, GRID, 1))


def add_footer(body: list[str], width: int, height: int, note: str) -> None:
    body.append(line(58, height - 44, width - 58, height - 44, GRID, 1))
    body.append(text(58, height - 20, note, "small"))
    body.append(text(width - 58, height - 20, "GR00T-H N1.7 language report", "small", "end"))


def add_bar_panel(
    body: list[str],
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    panel_title: str,
    panel_subtitle: str,
    rows: list[tuple[str, float, str]],
    color: str,
    label_width: float | None = None,
) -> None:
    body.append(rect(x, y, width, height, CARD, 14, GRID))
    body.append(text(x + 24, y + 35, panel_title, "panel-title"))
    body.append(text(x + 24, y + 57, panel_subtitle, "small"))

    if not rows:
        body.append(text(x + 24, y + 100, "No data", "subtitle"))
        return

    label_width = label_width or width * 0.34
    chart_x = x + 24 + label_width
    chart_width = width - label_width - 108
    chart_top = y + 78
    row_height = (height - 100) / len(rows)
    maximum = max(value for _, value, _ in rows) or 1.0

    for index, (label_value, value, display_value) in enumerate(rows):
        center_y = chart_top + index * row_height + row_height / 2
        bar_height = min(15, row_height * 0.48)
        body.append(text(chart_x - 12, center_y + 4, label_value, "label", "end"))
        body.append(rect(chart_x, center_y - bar_height / 2, chart_width, bar_height, "#E9EEEC", 6))
        bar_width = chart_width * value / maximum
        body.append(rect(chart_x, center_y - bar_height / 2, bar_width, bar_height, color, 6))
        body.append(text(chart_x + chart_width + 10, center_y + 4, display_value, "value"))


# -----------------------------------------------------------------------------
# Figure 1: training mixture and reconstruction coverage
# -----------------------------------------------------------------------------

def draw_mixture_coverage(rows: list[dict[str, str]], output: Path) -> None:
    rows = sorted(rows, key=lambda row: as_float(row, "normalized_training_mixture_weight"), reverse=True)
    width = 1540
    height = 170 + len(rows) * 66 + 70
    body: list[str] = []
    add_header(
        body,
        "Training mixture & reconstruction coverage",
        "Exact YAML mixture weights alongside current-public row coverage; >100% flags snapshot/filter differences.",
        width,
    )

    name_x = 62
    mix_x, mix_width = 390, 410
    coverage_x, coverage_width = 980, 290
    body.append(text(name_x, 145, "Training group", "small"))
    body.append(text(mix_x, 145, "Normalized mixture weight", "small"))
    body.append(text(coverage_x, 145, "Public rows ÷ expected training rows", "small"))
    body.append(text(width - 60, 145, "Gaps", "small", "end"))

    max_mix = max(as_float(row, "normalized_training_mixture_weight") for row in rows)
    max_coverage = max(
        as_float(row, "current_public_raw_rows_mapped_entries")
        / as_float(row, "expected_training_frames_yaml")
        for row in rows
    )
    max_coverage = max(2.0, math.ceil(max_coverage * 10) / 10)

    for index, row in enumerate(rows):
        y = 178 + index * 66
        mix = as_float(row, "normalized_training_mixture_weight")
        expected = as_float(row, "expected_training_frames_yaml")
        current = as_float(row, "current_public_raw_rows_mapped_entries")
        coverage = current / expected if expected else 0.0
        represented = row["language_distribution_reconstructed"].lower() == "true"
        missing = as_int(row, "missing_entries")
        color = TEAL if represented else CORAL

        body.append(text(name_x, y + 17, title_case_identifier(row["group"]), "label"))
        body.append(text(name_x, y + 36, f"{as_int(row, 'mapped_entries')} mapped entries", "small"))

        body.append(rect(mix_x, y + 4, mix_width, 16, "#E2E9E6", 7))
        body.append(rect(mix_x, y + 4, mix_width * mix / max_mix, 16, color, 7))
        body.append(text(mix_x + mix_width + 12, y + 17, fmt_percent(mix, 2), "value"))

        body.append(rect(coverage_x, y + 4, coverage_width, 16, "#E2E9E6", 7))
        body.append(
            rect(
                coverage_x,
                y + 4,
                coverage_width * min(coverage, max_coverage) / max_coverage,
                16,
                BLUE if coverage <= 1.05 else AMBER,
                7,
            )
        )
        expected_x = coverage_x + coverage_width / max_coverage
        body.append(line(expected_x, y, expected_x, y + 25, INK, 1.2, "3 3"))
        body.append(text(coverage_x + coverage_width + 12, y + 17, fmt_percent(coverage), "value"))
        body.append(
            text(
                coverage_x,
                y + 38,
                f"{fmt_compact(current)} public / {fmt_compact(expected)} expected",
                "small",
            )
        )

        gap_label = f"{missing} missing" if missing else "complete mapping"
        body.append(text(width - 60, y + 17, gap_label, "value", "end", CORAL if missing else TEAL))

    add_footer(
        body,
        width,
        height,
        "Dashed marker = 100% row coverage. Mixture bars use exact normalized mix_ratio values, not rounded comments.",
    )
    output.write_text(svg_document(width, height, body), encoding="utf-8")


# -----------------------------------------------------------------------------
# Figure 2: semantic prompt distribution
# -----------------------------------------------------------------------------

def draw_semantic_prompts(rows: list[dict[str, str]], output: Path, top_n: int) -> None:
    data = sorted(
        (
            (row["semantic_language"], as_float(row, "normalized_training_mixture_weight"))
            for row in rows
        ),
        key=lambda item: item[1],
        reverse=True,
    )[:top_n]

    width = 1500
    height = 175 + len(data) * 34 + 58
    body: list[str] = []
    add_header(
        body,
        "Most common normalized task instructions",
        "Training-mixture share after VLA input is converted to normalized task text.",
        width,
    )

    chart_x = 610
    chart_width = 690
    maximum = max(value for _, value in data)
    for index, (label_value, value) in enumerate(data):
        y = 145 + index * 34
        label_lines = textwrap.wrap(label_value, width=66) or [label_value]
        body.append(text(chart_x - 18, y + 14, shorten(label_lines[0], 70), "label", "end"))
        body.append(rect(chart_x, y + 2, chart_width, 13, "#E2E9E6", 6))
        body.append(rect(chart_x, y + 2, chart_width * value / maximum, 13, TEAL, 6))
        body.append(text(chart_x + chart_width + 14, y + 13, fmt_percent(value, 2), "value"))

    shown_mass = sum(value for _, value in data)
    body.append(
        text(
            60,
            height - 58,
            f"Top {len(data)} instructions account for {fmt_percent(shown_mass, 1)} of normalized language exposure.",
            "subtitle",
        )
    )
    add_footer(body, width, height, "Long labels are truncated only in the chart; source CSV retains full text.")
    output.write_text(svg_document(width, height, body), encoding="utf-8")


# -----------------------------------------------------------------------------
# Figure 3: lexical overview
# -----------------------------------------------------------------------------

def frequency_rows(rows: list[dict[str, str]], label_col: str, top_n: int) -> list[tuple[str, float, str]]:
    return [
        (
            shorten(row[label_col], 27),
            as_float(row, "token_share"),
            fmt_percent(as_float(row, "token_share"), 1),
        )
        for row in rows[:top_n]
    ]


def draw_lexical_overview(
    words: list[dict[str, str]],
    verbs: list[dict[str, str]],
    nouns: list[dict[str, str]],
    bigrams: list[dict[str, str]],
    output: Path,
    top_n: int,
) -> None:
    panel_n = min(top_n, 12)
    width, height = 1580, 1000
    body: list[str] = []
    add_header(
        body,
        "Normalized vocabulary",
        "Weighted statistics from normalized task text; bigrams contain only adjacent words.",
        width,
    )

    panels = [
        ("Content words", "Share of non-stopword occurrences", words, "word", TEAL),
        ("Verbs", "Share of lemmatized verb occurrences", verbs, "verb", CORAL),
        ("Nouns", "Share of lemmatized noun occurrences", nouns, "noun", BLUE),
        ("Adjacent bigrams", "No stopword, punctuation, or sentence bridging", bigrams, "bigram", PURPLE),
    ]
    positions = [(50, 120), (805, 120), (50, 545), (805, 545)]
    for (panel_title, subtitle, rows, label_col, color), (x, y) in zip(panels, positions):
        add_bar_panel(
            body,
            x=x,
            y=y,
            width=725,
            height=390,
            panel_title=panel_title,
            panel_subtitle=subtitle,
            rows=frequency_rows(rows, label_col, panel_n),
            color=color,
            label_width=205,
        )

    add_footer(body, width, height, "Directional left/right terms remain content words but are excluded from verb/noun counts by audited POS rules.")
    output.write_text(svg_document(width, height, body), encoding="utf-8")


# -----------------------------------------------------------------------------
# Figure 4: weighted instruction lengths
# -----------------------------------------------------------------------------

def weighted_quantile(points: list[tuple[int, float]], q: float) -> int:
    threshold = q * sum(weight for _, weight in points)
    cumulative = 0.0
    for length, weight in sorted(points):
        cumulative += weight
        if cumulative >= threshold:
            return length
    return max(length for length, _ in points)


def formalized_vlm_length_points(
    audit: list[dict[str, str]],
) -> list[tuple[int, float]]:
    """Aggregate model-facing lengths without expanding frame-level rows.

    ``vlm_formalized_text`` matches N1.7's lowercase/punctuation-removal step.
    The length remains an alphabetic-token proxy, not the VLM tokenizer's
    subword-token count.
    """
    weights: dict[int, float] = defaultdict(float)
    for row in audit:
        weight = as_float(row, "semantic_analysis_weight")
        if row["semantic_text"] and weight > 0:
            weights[as_int(row, "vlm_alphabetic_token_length")] += weight
    points = sorted(weights.items())
    if not points:
        raise ValueError("No retained model-facing language found in normalization audit")
    total = sum(weight for _, weight in points)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"Model-facing length weights sum to {total:.12f}, not 1")
    return points


def draw_instruction_lengths(
    rows: list[dict[str, str]],
    audit: list[dict[str, str]],
    output: Path,
) -> None:
    semantic_points = sorted(
        (
            as_int(row, "alphabetic_token_length"),
            as_float(row, "training_mixture_weight"),
        )
        for row in rows
    )
    vlm_points = formalized_vlm_length_points(audit)

    semantic_total = sum(weight for _, weight in semantic_points)
    vlm_total = sum(weight for _, weight in vlm_points)
    semantic_mean = sum(length * weight for length, weight in semantic_points) / semantic_total
    vlm_mean = sum(length * weight for length, weight in vlm_points) / vlm_total
    semantic_median = weighted_quantile(semantic_points, 0.5)
    vlm_median = weighted_quantile(vlm_points, 0.5)

    width, height = 1450, 760
    body: list[str] = []
    add_header(
        body,
        "Instruction length: VLA input vs normalized task text",
        "Training-mixture-weighted alphabetic-token proxy; examples are available for every normalized length.",
        width,
    )

    chart_x, chart_y = 95, 205
    chart_width, chart_height = 1230, 390
    maximum = max(weight for _, weight in semantic_points + vlm_points)
    min_length = min(semantic_points[0][0], vlm_points[0][0])
    max_length = max(semantic_points[-1][0], vlm_points[-1][0])
    x_step = chart_width / max(1, max_length - min_length)

    for tick in range(0, 6):
        y = chart_y + chart_height - tick * chart_height / 5
        value = maximum * tick / 5
        body.append(line(chart_x, y, chart_x + chart_width, y, GRID, 1))
        body.append(text(chart_x - 12, y + 4, fmt_percent(value), "small", "end"))

    def add_series(
        observed: list[tuple[int, float]],
        color: str,
        pale_color: str,
    ) -> None:
        weights = dict(observed)
        line_points = []
        for length in range(min_length, max_length + 1):
            x = chart_x + (length - min_length) * x_step
            y = chart_y + chart_height - chart_height * weights.get(length, 0.0) / maximum
            line_points.append((x, y))
        polygon_points = [
            (chart_x, chart_y + chart_height),
            *line_points,
            (chart_x + chart_width, chart_y + chart_height),
        ]
        point_string = " ".join(f"{x:.2f},{y:.2f}" for x, y in polygon_points)
        body.append(f'<polygon points="{point_string}" fill="{pale_color}" opacity="0.45"/>')
        line_string = " ".join(f"{x:.2f},{y:.2f}" for x, y in line_points)
        body.append(f'<polyline points="{line_string}" fill="none" stroke="{color}" stroke-width="3"/>')
        for length, weight in observed:
            x = chart_x + (length - min_length) * x_step
            y = chart_y + chart_height - chart_height * weight / maximum
            body.append(circle(x, y, 3.5, CARD, color))

    add_series(vlm_points, CORAL, PALE_CORAL)
    add_series(semantic_points, TEAL, PALE_TEAL)

    for length in range(min_length, max_length + 1):
        if length == min_length or length == max_length or length % 2 == 0:
            x = chart_x + (length - min_length) * x_step
            body.append(text(x, chart_y + chart_height + 28, str(length), "small", "middle"))

    for value, color in [(vlm_mean, CORAL), (semantic_mean, TEAL)]:
        x = chart_x + (value - min_length) * x_step
        body.append(line(x, chart_y, x, chart_y + chart_height, color, 1.5, "6 5"))

    body.append(line(chart_x, 151, chart_x + 32, 151, TEAL, 4))
    body.append(
        text(
            chart_x + 42,
            156,
            f"Normalized task text: mean {semantic_mean:.1f}, median {semantic_median}",
            "value",
            fill=TEAL,
        )
    )
    body.append(line(chart_x + 450, 151, chart_x + 482, 151, CORAL, 4))
    body.append(
        text(
            chart_x + 492,
            156,
            f"VLA input: mean {vlm_mean:.1f}, median {vlm_median}",
            "value",
            fill=CORAL,
        )
    )

    body.append(
        text(
            chart_x + chart_width / 2,
            chart_y + chart_height + 58,
            "Alphabetic tokens per instruction (word-level proxy, not tokenizer IDs)",
            "subtitle",
            "middle",
        )
    )
    add_footer(
        body,
        width,
        height,
        "VLA input is model-facing; normalized task text is analysis-only. Invalid placeholders are excluded.",
    )
    output.write_text(svg_document(width, height, body), encoding="utf-8")


# -----------------------------------------------------------------------------
# Figure 5: training-entry contributions
# -----------------------------------------------------------------------------

def draw_entry_contributions(rows: list[dict[str, str]], output: Path, top_n: int) -> None:
    weights: dict[tuple[str, str, str], float] = defaultdict(float)
    for row in rows:
        key = (row["entry_id"], row["group"], row["historical_training_path"])
        weights[key] += as_float(row, "normalized_reconstructed_weight")

    ranked = sorted(weights.items(), key=lambda item: item[1], reverse=True)[:top_n]
    data = []
    for (_, group, path), value in ranked:
        short_path = path.replace("${OPEN_H_DATA_PATH}/", "")
        label_value = f"{title_case_identifier(group)} · {shorten(short_path, 48)}"
        data.append((label_value, value, fmt_percent(value, 2)))

    width = 1580
    height = 190 + len(data) * 45 + 65
    body: list[str] = []
    add_header(
        body,
        "Largest reconstructed training-entry contributions",
        "Entry-level probability after applying group mixture weights and within-group public row proportions.",
        width,
    )
    add_bar_panel(
        body,
        x=48,
        y=132,
        width=1484,
        height=height - 210,
        panel_title=f"Top {len(data)} mapped entries",
        panel_subtitle="These are reconstructed exposure estimates, not raw frame shares.",
        rows=data,
        color=BLUE,
        label_width=625,
    )
    add_footer(body, width, height, "Missing historical entries are reported in gr00t_h_n17_missing_training_entries.csv.")
    output.write_text(svg_document(width, height, body), encoding="utf-8")


# -----------------------------------------------------------------------------
# Figure 6: normalization QA
# -----------------------------------------------------------------------------

def draw_normalization_qa(
    rules: list[dict[str, str]],
    pos_rows: list[dict[str, str]],
    audit: list[dict[str, str]],
    output: Path,
) -> None:
    width, height = 1580, 980
    body: list[str] = []
    add_header(
        body,
        "Data cleaning pipeline audit",
        "Rule reach is measured against VLA-input mixture weight; overlapping rules are not additive.",
        width,
    )

    rule_data = [
        (
            shorten(
                f"{format_rule_label(row['rule'])} ({as_int(row, 'source_rows_affected'):,} rows)",
                42,
            ),
            as_float(row, "source_mixture_weight_affected"),
            fmt_percent(as_float(row, "source_mixture_weight_affected"), 2),
        )
        for row in rules
    ]
    add_bar_panel(
        body,
        x=48,
        y=132,
        width=910,
        height=790,
        panel_title="Cleaning-rule reach",
        panel_subtitle="VLA-input probability affected by each rule",
        rows=rule_data,
        color=TEAL,
        label_width=295,
    )

    pos_aggregated: dict[str, float] = defaultdict(float)
    for row in pos_rows:
        label_value = f"{row['word']}: {row['nltk_tag']} → {row['corrected_tag']}"
        pos_aggregated[label_value] += as_float(row, "expected_corrections_per_training_sample")
    pos_data = [
        (label_value, value, f"{value:.4f}")
        for label_value, value in sorted(pos_aggregated.items(), key=lambda item: item[1], reverse=True)
    ]
    add_bar_panel(
        body,
        x=982,
        y=132,
        width=550,
        height=455,
        panel_title="Audited POS corrections",
        panel_subtitle="NLTK tag → corrected analysis tag",
        rows=pos_data,
        color=CORAL,
        label_width=165,
    )

    source_weight = sum(as_float(row, "source_training_mixture_weight") for row in audit)
    retained_weight = sum(
        as_float(row, "source_training_mixture_weight")
        for row in audit
        if row["semantic_text"]
    )
    dropped = source_weight - retained_weight
    body.append(rect(982, 612, 550, 310, CARD, 14, GRID))
    body.append(text(1006, 648, "Weight conservation", "panel-title"))
    body.append(text(1006, 672, "Original weights remain visible in the row-level audit", "small"))

    stats = [
        ("VLA input", source_weight, TEAL),
        ("Valid normalized task text", retained_weight, BLUE),
        ("Invalid input placeholder", dropped, CORAL),
    ]
    for index, (label_value, value, color) in enumerate(stats):
        y = 714 + index * 62
        body.append(text(1006, y, label_value, "label"))
        body.append(text(1498, y, fmt_percent(value, 4), "value", "end", color))
        body.append(rect(1006, y + 12, 492, 12, "#E2E9E6", 5))
        visible_width = 492 * value
        if value > 0:
            visible_width = max(visible_width, 2)
        body.append(rect(1006, y + 12, visible_width, 12, color, 5))

    add_footer(body, width, height, "The audit CSV retains VLA input, clean task text, normalized task text, rules, and weights.")
    output.write_text(svg_document(width, height, body), encoding="utf-8")


# -----------------------------------------------------------------------------
# Interactive verb -> nearest-following-noun drill-down
# -----------------------------------------------------------------------------

def html_identifier(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "item"


def render_verb_noun_drilldown(
    verbs: list[dict[str, str]],
    relationships: list[dict[str, str]],
    examples: list[dict[str, str]],
) -> str:
    relationships_by_verb: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in relationships:
        relationships_by_verb[row["verb"]].append(row)
    for rows in relationships_by_verb.values():
        rows.sort(
            key=lambda row: as_float(
                row,
                "pair_expected_occurrences_per_training_sample",
            ),
            reverse=True,
        )

    examples_by_pair: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in examples:
        examples_by_pair[(row["verb"], row["following_noun"])].append(row)

    verb_buttons = []
    verb_panels = []
    for verb_row in verbs:
        verb = verb_row["verb"]
        verb_id = f"verb-{html_identifier(verb)}"
        token_share = as_float(verb_row, "token_share")
        expected = as_float(verb_row, "expected_occurrences_per_training_sample")
        relation_rows = relationships_by_verb.get(verb, [])
        matched = (
            as_float(
                relation_rows[0],
                "matched_noun_expected_occurrences_per_training_sample",
            )
            if relation_rows
            else 0.0
        )
        match_rate = min(1.0, matched / expected) if expected else 0.0
        top_noun = relation_rows[0]["following_noun"] if relation_rows else "none found"

        verb_buttons.append(
            f'<button type="button" class="verb-button" '
            f'data-verb-target="{escape(verb_id)}" aria-controls="{escape(verb_id)}" '
            f'aria-expanded="false">{escape(verb)}</button>'
        )

        table_rows = []
        for relation in relation_rows:
            noun = relation["following_noun"]
            pair_examples = examples_by_pair.get((verb, noun), [])
            example_items = "".join(
                f"<li><b>{escape(row['current_public_path'])}</b> — "
                f"{escape(row['canonical_semantic_text'])}</li>"
                for row in pair_examples
            )
            if not example_items:
                example_items = "<li>No provenance example available</li>"
            table_rows.append(
                f"""
                <tr>
                  <td><b>{escape(noun)}</b></td>
                  <td>{escape(fmt_percent(as_float(relation, 'pair_share_of_all_verb_occurrences'), 1))}</td>
                  <td>{escape(fmt_percent(as_float(relation, 'pair_share_among_matched_nouns'), 1))}</td>
                  <td>{as_float(relation, 'pair_expected_occurrences_per_training_sample'):.5f}</td>
                  <td><ul class="relation-examples">{example_items}</ul></td>
                </tr>
                """
            )

        if table_rows:
            content = f"""
              <div class="relation-table-wrap">
                <table class="relation-table">
                  <thead><tr>
                    <th>Nearest following noun</th>
                    <th>All uses of verb</th>
                    <th>Matched uses only</th>
                    <th>Expected / sample</th>
                    <th>Representative instructions and datasets</th>
                  </tr></thead>
                  <tbody>{''.join(table_rows)}</tbody>
                </table>
              </div>
            """
        else:
            content = (
                '<p class="no-relation">No noun occurs after this verb before '
                "another verb or hard clause boundary in the normalized task text.</p>"
            )

        verb_panels.append(
            f"""
            <section class="verb-panel" id="{escape(verb_id)}" hidden>
              <div class="verb-panel-meta">
                <span class="verb-name">{escape(verb)}</span>
                <span>{escape(fmt_percent(token_share, 2))} of verb tokens</span>
                <span>{escape(fmt_percent(match_rate, 1))} matched to a following noun</span>
                <span>top noun: <b>{escape(top_noun)}</b></span>
              </div>
              {content}
            </section>
            """
        )

    return f"""
      <div class="verb-drilldown">
        <div class="verb-intro">
          <h3>What noun follows each verb?</h3>
          <p>Select a verb to show its nearest following nouns in normalized task text. Matching stops at another verb or a hard clause boundary, so this answers questions such as “go where?” without claiming a grammatical dependency.</p>
          <p class="download"><a href="../gr00t_h_n17_verb_following_noun.csv">Download all verb–noun statistics</a></p>
        </div>
        <div class="verb-buttons" role="group" aria-label="Select a verb">{''.join(verb_buttons)}</div>
        <div class="verb-panels">{''.join(verb_panels)}</div>
      </div>
    """


# -----------------------------------------------------------------------------
# Representative examples by semantic instruction length
# -----------------------------------------------------------------------------

LENGTH_EXAMPLE_FIELDS = [
    "semantic_alphabetic_token_length",
    "length_training_mixture_weight",
    "canonical_prompt_training_mixture_weight",
    "example_rank",
    "current_public_path",
    "historical_training_path",
    "mechanism",
    "semantic_surface_text",
    "semantic_text",
    "vlm_formalized_text",
    "original_text",
    "semantic_analysis_weight",
    "vlm_alphabetic_token_length",
]


def build_length_examples(
    audit: list[dict[str, str]],
    length_rows: list[dict[str, str]],
    examples_per_length: int = 3,
) -> list[dict[str, str | int | float]]:
    """Choose distinct, weighted concepts for every observed length."""
    length_weights = {
        as_int(row, "alphabetic_token_length"): as_float(row, "training_mixture_weight")
        for row in length_rows
    }
    candidates: dict[int, list[dict[str, str]]] = defaultdict(list)
    canonical_weights: dict[str, float] = defaultdict(float)
    for row in audit:
        if row["semantic_text"] and as_float(row, "semantic_analysis_weight") > 0:
            candidates[as_int(row, "semantic_alphabetic_token_length")].append(row)
            canonical_weights[row["semantic_text"]] += as_float(
                row,
                "semantic_analysis_weight",
            )

    missing = sorted(set(length_weights).difference(candidates))
    if missing:
        raise ValueError(f"No audit examples found for semantic lengths: {missing}")

    output_rows: list[dict[str, str | int | float]] = []
    for length in sorted(length_weights):
        rows_by_canonical: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in candidates[length]:
            rows_by_canonical[row["semantic_text"]].append(row)
        ranked_canonical = sorted(
            rows_by_canonical,
            key=lambda semantic: canonical_weights[semantic],
            reverse=True,
        )
        selected: list[dict[str, str]] = []
        seen_datasets: set[str] = set()

        # Rank distinct concepts by their aggregate mixture weight. For each
        # concept, choose its strongest source row, preferring a dataset that
        # has not already appeared in this length group.
        for semantic in ranked_canonical:
            source_rows = sorted(
                rows_by_canonical[semantic],
                key=lambda row: as_float(row, "semantic_analysis_weight"),
                reverse=True,
            )
            row = next(
                (
                    candidate
                    for candidate in source_rows
                    if candidate["current_public_path"] not in seen_datasets
                ),
                source_rows[0],
            )
            selected.append(row)
            seen_datasets.add(row["current_public_path"])
            if len(selected) == examples_per_length:
                break

        for rank, row in enumerate(selected, 1):
            output_rows.append(
                {
                    "semantic_alphabetic_token_length": length,
                    "length_training_mixture_weight": length_weights[length],
                    "canonical_prompt_training_mixture_weight": canonical_weights[
                        row["semantic_text"]
                    ],
                    "example_rank": rank,
                    "current_public_path": row["current_public_path"],
                    "historical_training_path": row["historical_training_path"],
                    "mechanism": row["mechanism"],
                    "semantic_surface_text": row["semantic_surface_text"],
                    "semantic_text": row["semantic_text"],
                    "vlm_formalized_text": row["vlm_formalized_text"],
                    "original_text": row["original_text"],
                    "semantic_analysis_weight": as_float(row, "semantic_analysis_weight"),
                    "vlm_alphabetic_token_length": as_int(row, "vlm_alphabetic_token_length"),
                }
            )
    return output_rows


def write_length_examples_csv(
    rows: list[dict[str, str | int | float]],
    output: Path,
) -> None:
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LENGTH_EXAMPLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def render_length_examples(rows: list[dict[str, str | int | float]]) -> str:
    grouped: dict[int, list[dict[str, str | int | float]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["semantic_alphabetic_token_length"])].append(row)

    lengths = sorted(grouped)
    default_length = max(
        lengths,
        key=lambda length: float(grouped[length][0]["length_training_mixture_weight"]),
    )
    default_index = lengths.index(default_length)
    default_unit = "token" if default_length == 1 else "tokens"
    panels = []
    for length, examples in sorted(grouped.items()):
        mass = float(examples[0]["length_training_mixture_weight"])
        noun = "token" if length == 1 else "tokens"
        example_cards = []
        for row in examples:
            dataset = str(row["current_public_path"])
            historical = str(row["historical_training_path"])
            mechanism = title_case_identifier(str(row["mechanism"]))
            semantic_surface_text = str(row["semantic_surface_text"])
            semantic_text = str(row["semantic_text"])
            vlm_text = str(row["vlm_formalized_text"])
            original_text = str(row["original_text"])
            vlm_length = int(row["vlm_alphabetic_token_length"])
            row_weight = float(row["semantic_analysis_weight"])
            concept_weight = float(row["canonical_prompt_training_mixture_weight"])
            example_cards.append(
                f"""
                <article class="example">
                  <div class="example-meta">
                    <span class="dataset">Dataset · {escape(dataset)}</span>
                    <span>{escape(mechanism)}</span>
                    <span>instruction weight {escape(fmt_percent(concept_weight, 3))}</span>
                    <span>source weight {escape(fmt_percent(row_weight, 3))}</span>
                  </div>
                  <dl>
                    <dt>Normalized task text</dt><dd>{escape(semantic_text)}</dd>
                    <dt>Clean task text</dt><dd>{escape(semantic_surface_text)}</dd>
                    <dt>VLA input <small>({vlm_length} alphabetic tokens)</small></dt>
                    <dd><code>{escape(vlm_text)}</code></dd>
                  </dl>
                  <details class="source-detail">
                    <summary>Source prompt and provenance</summary>
                    <p><b>Historical training path:</b> {escape(historical)}</p>
                    <p><b>Extracted prompt before VLA formalization:</b> {escape(original_text)}</p>
                  </details>
                </article>
                """
            )
        hidden_attribute = "" if length == default_length else " hidden"
        panels.append(
            f"""
            <section class="length-panel" data-length="{length}"{hidden_attribute}>
              <div class="length-result-meta">
                <b>{length} normalized {noun}</b>
                <span>{escape(fmt_percent(mass, 3))} of reconstructed language</span>
                <span>{len(examples)} representative example{'s' if len(examples) != 1 else ''}</span>
              </div>
              <div class="examples-grid">{''.join(example_cards)}</div>
            </section>
            """
        )

    encoded_lengths = ",".join(str(length) for length in lengths)
    return f"""
      <div class="length-examples">
        <div class="length-intro">
          <h3>Typical instructions at every normalized length</h3>
          <p>Examples are distinct normalized task instructions ranked by training-mixture weight. <b>VLA input</b> is the model-facing string, <b>clean task text</b> removes templates and boundary artifacts, and <b>normalized task text</b> standardizes case and morphology for analysis. Lengths are alphabetic-word proxies, not tokenizer IDs.</p>
        </div>
        <div class="length-control">
          <div class="length-control-heading">
            <label for="length-selector">Select normalized task length</label>
            <output id="length-output" for="length-selector" aria-live="polite">{default_length} {default_unit}</output>
          </div>
          <input id="length-selector" type="range" min="0" max="{len(lengths) - 1}"
                 value="{default_index}" step="1" data-lengths="{encoded_lengths}">
          <div class="length-scale">
            <span>{lengths[0]} token</span>
            <span>{len(lengths)} observed lengths</span>
            <span>{lengths[-1]} tokens</span>
          </div>
        </div>
        <div class="length-panels">{''.join(panels)}</div>
      </div>
    """


# -----------------------------------------------------------------------------
# HTML gallery
# -----------------------------------------------------------------------------

def write_gallery(
    output_dir: Path,
    figures: list[tuple[str, str, str]],
    length_examples: list[dict[str, str | int | float]],
    verbs: list[dict[str, str]],
    verb_relationships: list[dict[str, str]],
    verb_examples: list[dict[str, str]],
) -> None:
    length_examples_html = render_length_examples(length_examples)
    verb_drilldown_html = render_verb_noun_drilldown(
        verbs,
        verb_relationships,
        verb_examples,
    )
    figure_titles = {filename: title for filename, title, _ in figures}

    def render_images(filenames: list[str]) -> str:
        return "".join(
            f'<img src="{escape(filename)}" alt="{escape(figure_titles[filename])}" '
            f'loading="{"eager" if filename.startswith("02_") else "lazy"}">'
            for filename in filenames
        )

    def render_section(
        section_id: str,
        title_value: str,
        description: str,
        filenames: list[str],
        *,
        before: str = "",
        after: str = "",
        open_by_default: bool = False,
        secondary: bool = False,
    ) -> str:
        open_attribute = " open" if open_by_default else ""
        secondary_class = " secondary" if secondary else ""
        return f"""
          <details class="card{secondary_class}" id="{escape(section_id)}"{open_attribute}>
            <summary class="section-summary">
              <span class="summary-copy">
                <span class="section-title">{escape(title_value)}</span>
                <span class="section-description">{escape(description)}</span>
              </span>
              <span class="section-toggle" aria-hidden="true"></span>
            </summary>
            <div class="section-body">
              {before}
              {render_images(filenames)}
              {after}
            </div>
          </details>
        """

    cleaning_pipeline = """
      <div class="pipeline">
        <h3>From model input to reported statistics</h3>
        <div class="pipeline-grid">
          <div class="pipeline-stage">
            <span class="stage-number">1</span>
            <b>VLA input</b>
            <p>The actual formalized language string paired with images and passed into the VLM.</p>
          </div>
          <div class="pipeline-stage">
            <span class="stage-number">2</span>
            <b>Clean task text</b>
            <p>State/tool templates, carrier phrases, run IDs, and token-boundary artifacts are removed.</p>
          </div>
          <div class="pipeline-stage">
            <span class="stage-number">3</span>
            <b>Normalized task text</b>
            <p>Case and morphology are standardized, with explicit compound, action, and POS rules.</p>
          </div>
          <div class="pipeline-stage">
            <span class="stage-number">4</span>
            <b>Weighted statistics</b>
            <p>NLTK counts are aggregated using reconstructed GR00T-H training-mixture weights.</p>
          </div>
        </div>
        <p class="pipeline-note">The source prompt, every intermediate text view, each applied rule, and all weights remain available in the row-level audit CSV.</p>
      </div>
    """

    cards = [
        render_section(
            "task-distribution",
            "Normalized task distribution",
            "The task instructions with the greatest reconstructed training exposure.",
            ["02_semantic_prompt_distribution.svg"],
            open_by_default=True,
        ),
        render_section(
            "lexical-overview",
            "Normalized vocabulary",
            "Weighted words, verbs, nouns, adjacent bigrams, and selectable verb–noun context.",
            ["03_lexical_overview.svg"],
            after=verb_drilldown_html,
        ),
        render_section(
            "instruction-lengths",
            "Instruction lengths",
            "VLA input compared with normalized task text, with dataset-labeled examples.",
            ["04_instruction_length_distribution.svg"],
            after=length_examples_html,
        ),
        render_section(
            "reconstruction-notes",
            "Data reconstruction notes",
            "Supporting audit of mapped training entries, mixture weights, coverage, and known gaps.",
            [
                "05_training_entry_contributions.svg",
                "01_training_mixture_and_coverage.svg",
            ],
            secondary=True,
        ),
        render_section(
            "cleaning-pipeline",
            "Data cleaning pipeline",
            "How VLA input becomes normalized task text before weighted language analysis.",
            ["06_normalization_audit.svg"],
            before=cleaning_pipeline,
            secondary=True,
        ),
    ]

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GR00T-H N1.7 Language Report</title>
  <style>
    :root {{ color-scheme: light; --bg:#F3F0E8; --ink:#18323E; --muted:#667780; --card:#FFFEFB; }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.48 Inter,ui-sans-serif,system-ui,sans-serif; }}
    header {{ max-width:1260px; margin:0 auto; padding:38px 22px 18px; }}
    h1 {{ margin:0 0 8px; font-size:clamp(30px,4vw,48px); letter-spacing:-1.2px; line-height:1.05; }}
    header p {{ color:var(--muted); max-width:800px; font-size:15px; }}
    main {{ max-width:1260px; margin:auto; padding:8px 22px 48px; display:grid; gap:14px; }}
    .card {{ background:var(--card); border:1px solid #DCE3E1; border-radius:12px; overflow:hidden; box-shadow:0 6px 20px rgba(24,50,62,.045); }}
    .card.secondary {{ background:#FAF9F5; }}
    .section-summary {{ cursor:pointer; list-style:none; display:flex; align-items:center; justify-content:space-between; gap:20px; padding:15px 18px; }}
    .section-summary::-webkit-details-marker {{ display:none; }}
    .section-summary:hover {{ background:#F0F5F2; }}
    .summary-copy {{ display:grid; gap:3px; }}
    .section-title {{ font-size:20px; font-weight:750; letter-spacing:-.15px; }}
    .section-description {{ color:var(--muted); font-size:13.5px; }}
    .section-toggle {{ flex:0 0 auto; width:24px; height:24px; border-radius:50%; border:1px solid #C9D7D3; position:relative; }}
    .section-toggle::before, .section-toggle::after {{ content:""; position:absolute; background:#176B66; left:6px; right:6px; top:11px; height:1.5px; }}
    .section-toggle::after {{ transform:rotate(90deg); transition:transform .15s ease; }}
    .card[open] > .section-summary .section-toggle::after {{ transform:rotate(0); }}
    .section-body {{ border-top:1px solid #E3E8E6; }}
    h3 {{ margin:0 0 6px; font-size:18px; }}
    p {{ margin:0; color:var(--muted); line-height:1.45; }}
    img {{ width:100%; display:block; }}
    .section-body > img + img {{ border-top:1px solid #E3E8E6; }}
    a {{ color:#176B66; text-underline-offset:3px; }}
    .verb-drilldown {{ border-top:1px solid #DCE3E1; padding:20px 22px 24px; }}
    .verb-intro {{ max-width:950px; }}
    .download {{ margin-top:6px; font-size:13px; }}
    .verb-buttons {{ display:flex; flex-wrap:wrap; gap:6px; margin:14px 0 0; }}
    .verb-button {{ appearance:none; border:1px solid #CDE0DB; color:#176B66; background:#E8F2EF; border-radius:999px; padding:5px 10px; font:650 13px/1.3 inherit; cursor:pointer; }}
    .verb-button:hover {{ background:#D9ECE7; }}
    .verb-button[aria-expanded="true"] {{ color:#fff; background:#176B66; border-color:#176B66; }}
    .verb-panel {{ border:1px solid #DCE3E1; border-radius:9px; background:#F9F8F3; margin-top:12px; overflow:hidden; }}
    .verb-panel[hidden] {{ display:none; }}
    .verb-panel-meta {{ display:flex; flex-wrap:wrap; gap:8px 18px; align-items:center; padding:11px 13px; color:var(--muted); font-size:13px; }}
    .verb-name {{ font-size:16px; font-weight:750; color:#176B66; }}
    .relation-table-wrap {{ overflow-x:auto; padding:0 10px 10px; }}
    .relation-table {{ width:100%; border-collapse:collapse; background:var(--card); border:1px solid #E1E6E4; font-size:12.5px; }}
    .relation-table th {{ color:var(--muted); font-size:10.5px; text-transform:uppercase; letter-spacing:.04em; text-align:left; background:#F0F4F2; }}
    .relation-table th, .relation-table td {{ padding:8px 9px; border-bottom:1px solid #E1E6E4; vertical-align:top; }}
    .relation-table th:nth-child(2), .relation-table th:nth-child(3), .relation-table th:nth-child(4), .relation-table td:nth-child(2), .relation-table td:nth-child(3), .relation-table td:nth-child(4) {{ white-space:nowrap; }}
    .relation-examples {{ margin:0; padding-left:15px; min-width:280px; }}
    .relation-examples li {{ margin:0 0 4px; line-height:1.35; }}
    .no-relation {{ padding:0 12px 12px; }}
    .length-examples {{ border-top:1px solid #DCE3E1; padding:20px 22px 24px; }}
    .length-intro {{ margin-bottom:14px; max-width:950px; }}
    .length-control {{ max-width:780px; border:1px solid #D8E3DF; border-radius:9px; background:#F7FAF8; padding:12px 14px 9px; }}
    .length-control-heading {{ display:flex; align-items:baseline; justify-content:space-between; gap:12px; }}
    .length-control-heading label {{ font-weight:700; }}
    .length-control-heading output {{ color:#176B66; font-weight:750; }}
    #length-selector {{ width:100%; margin:10px 0 2px; accent-color:#176B66; cursor:pointer; }}
    .length-scale {{ display:flex; justify-content:space-between; gap:10px; color:var(--muted); font-size:11.5px; }}
    .length-panels {{ margin-top:12px; }}
    .length-panel {{ border:1px solid #DCE3E1; border-radius:9px; background:#F9F8F3; overflow:hidden; }}
    .length-panel[hidden] {{ display:none; }}
    .length-result-meta {{ display:flex; flex-wrap:wrap; gap:8px 18px; align-items:center; padding:11px 12px; color:var(--muted); font-size:13px; }}
    .length-result-meta b {{ color:var(--ink); font-size:15px; }}
    .examples-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:10px; padding:0 10px 10px; }}
    .example {{ background:var(--card); border:1px solid #E1E6E4; border-radius:8px; padding:11px; min-width:0; }}
    .example-meta {{ display:flex; flex-wrap:wrap; gap:5px; margin-bottom:9px; }}
    .example-meta span {{ background:#EDF2EF; color:var(--muted); border-radius:999px; padding:3px 6px; font-size:11px; overflow-wrap:anywhere; }}
    .example-meta .dataset {{ background:#D9ECE7; color:#175C58; font-weight:650; }}
    dl {{ margin:0; }}
    dt {{ color:var(--muted); font-size:11px; font-weight:650; margin:8px 0 2px; text-transform:uppercase; letter-spacing:.04em; }}
    dd {{ margin:0; line-height:1.4; overflow-wrap:anywhere; }}
    code {{ display:block; color:#663D34; background:#FAEAE5; border-radius:6px; padding:7px 8px; white-space:pre-wrap; overflow-wrap:anywhere; font-family:ui-monospace,SFMono-Regular,Consolas,monospace; font-size:12px; }}
    .source-detail {{ margin-top:9px; color:var(--muted); font-size:12px; }}
    .source-detail summary {{ cursor:pointer; }}
    .source-detail p {{ margin-top:5px; overflow-wrap:anywhere; }}
    .pipeline {{ padding:20px 22px 22px; background:#F7FAF8; }}
    .pipeline-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:8px; margin-top:12px; }}
    .pipeline-stage {{ min-width:0; border:1px solid #D8E3DF; border-radius:8px; background:var(--card); padding:11px; }}
    .pipeline-stage b {{ display:block; margin:7px 0 4px; font-size:14px; }}
    .pipeline-stage p {{ font-size:12.5px; }}
    .stage-number {{ display:grid; place-items:center; width:22px; height:22px; border-radius:50%; color:#fff; background:#176B66; font-size:11px; font-weight:750; }}
    .pipeline-note {{ margin-top:10px; font-size:12.5px; }}
    @media (max-width:800px) {{
      header, main {{ padding-left:14px; padding-right:14px; }}
      .section-description {{ display:none; }}
      .pipeline-grid {{ grid-template-columns:1fr 1fr; }}
    }}
    @media (max-width:480px) {{ .pipeline-grid {{ grid-template-columns:1fr; }} }}
    footer {{ max-width:1260px; margin:auto; padding:0 22px 32px; color:var(--muted); font-size:12px; }}
  </style>
</head>
<body>
  <header>
    <h1>GR00T-H N1.7 Language Report</h1>
    <p>Training-mixture-weighted results from normalized task text, with VLA input retained for comparison and audit.</p>
  </header>
  <main>{''.join(cards)}</main>
  <footer>Generated by visualize_language_analysis.py. Results are reconstructed from mapped public data and the N1.7 training recipe.</footer>
  <script>
    if (window.location.hash) {{
      const requestedSection = document.querySelector(window.location.hash);
      if (requestedSection?.matches('details.card')) {{
        document.querySelectorAll('main > details.card').forEach((section) => {{
          section.open = false;
        }});
        requestedSection.open = true;
      }}
    }}
    const verbButtons = [...document.querySelectorAll('.verb-button')];
    const verbPanels = [...document.querySelectorAll('.verb-panel')];
    verbButtons.forEach((button) => {{
      button.addEventListener('click', () => {{
        const shouldOpen = button.getAttribute('aria-expanded') !== 'true';
        verbButtons.forEach((item) => item.setAttribute('aria-expanded', 'false'));
        verbPanels.forEach((panel) => {{ panel.hidden = true; }});
        if (shouldOpen) {{
          const target = document.getElementById(button.dataset.verbTarget);
          if (target) {{
            target.hidden = false;
            button.setAttribute('aria-expanded', 'true');
          }}
        }}
      }});
    }});
    const lengthSelector = document.getElementById('length-selector');
    const lengthOutput = document.getElementById('length-output');
    const lengthPanels = [...document.querySelectorAll('.length-panel')];
    if (lengthSelector && lengthOutput) {{
      const observedLengths = lengthSelector.dataset.lengths.split(',');
      const showSelectedLength = () => {{
        const selectedLength = observedLengths[Number(lengthSelector.value)];
        const unit = selectedLength === '1' ? 'token' : 'tokens';
        lengthOutput.value = `${{selectedLength}} ${{unit}}`;
        lengthPanels.forEach((panel) => {{
          panel.hidden = panel.dataset.length !== selectedLength;
        }});
      }};
      lengthSelector.addEventListener('input', showSelectedLength);
      showSelectedLength();
    }}
  </script>
</body>
</html>
"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("extracted_language"),
        help="Directory containing extractor and NLTK CSV outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Figure destination (default: INPUT_DIR/visualizations).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Number of prompts and training entries to show (default: 15).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_n < 5:
        raise ValueError("--top-n must be at least 5")

    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    group_coverage = read_csv(input_dir / "gr00t_h_n17_group_coverage.csv")
    semantic = read_csv(input_dir / "gr00t_h_n17_semantic_language.csv")
    words = read_csv(input_dir / "gr00t_h_n17_nltk_word_frequency.csv")
    verbs = read_csv(input_dir / "gr00t_h_n17_nltk_verb_frequency.csv")
    nouns = read_csv(input_dir / "gr00t_h_n17_nltk_noun_frequency.csv")
    bigrams = read_csv(input_dir / "gr00t_h_n17_nltk_bigram_frequency.csv")
    lengths = read_csv(input_dir / "gr00t_h_n17_nltk_instruction_length.csv")
    public_counts = read_csv(input_dir / "gr00t_h_n17_public_language_counts.csv")
    rules = read_csv(input_dir / "gr00t_h_n17_language_normalization_rule_summary.csv")
    pos = read_csv(input_dir / "gr00t_h_n17_pos_correction_summary.csv")
    audit = read_csv(input_dir / "gr00t_h_n17_language_normalization_audit.csv")
    verb_relationships = read_csv(input_dir / "gr00t_h_n17_verb_following_noun.csv")
    verb_examples = read_csv(input_dir / "gr00t_h_n17_verb_following_noun_examples.csv")

    require_unit_sum(
        group_coverage,
        "normalized_training_mixture_weight",
        "training group",
    )
    require_unit_sum(semantic, "normalized_training_mixture_weight", "semantic prompt")
    require_unit_sum(words, "token_share", "word token-share")
    require_unit_sum(verbs, "token_share", "verb token-share")
    require_unit_sum(nouns, "token_share", "noun token-share")
    require_unit_sum(bigrams, "token_share", "bigram token-share")
    require_unit_sum(lengths, "training_mixture_weight", "instruction length")
    require_unit_sum(public_counts, "normalized_reconstructed_weight", "source prompt")

    figures = [
        (
            "01_training_mixture_and_coverage.svg",
            "Training reconstruction audit",
            "Training-group weights, public-data coverage, and missing-entry warnings.",
        ),
        (
            "02_semantic_prompt_distribution.svg",
            "Normalized task distribution",
            "The most exposed normalized task instructions under the reconstructed mixture.",
        ),
        (
            "03_lexical_overview.svg",
            "Normalized vocabulary",
            "Weighted words, verbs, nouns, bigrams, and verb–noun context.",
        ),
        (
            "04_instruction_length_distribution.svg",
            "Instruction lengths",
            "VLA input versus normalized task-text lengths.",
        ),
        (
            "05_training_entry_contributions.svg",
            "Reconstructed training-entry contributions",
            "Mapped training entries contributing the most reconstructed probability.",
        ),
        (
            "06_normalization_audit.svg",
            "Data cleaning pipeline audit",
            "Cleaning-rule reach, POS corrections, and weight conservation.",
        ),
    ]

    draw_mixture_coverage(group_coverage, output_dir / figures[0][0])
    draw_semantic_prompts(semantic, output_dir / figures[1][0], args.top_n)
    draw_lexical_overview(words, verbs, nouns, bigrams, output_dir / figures[2][0], args.top_n)
    length_examples = build_length_examples(audit, lengths)
    length_examples_path = output_dir / "instruction_length_examples.csv"
    write_length_examples_csv(length_examples, length_examples_path)

    draw_instruction_lengths(lengths, audit, output_dir / figures[3][0])
    draw_entry_contributions(public_counts, output_dir / figures[4][0], args.top_n)
    draw_normalization_qa(rules, pos, audit, output_dir / figures[5][0])
    write_gallery(
        output_dir,
        figures,
        length_examples,
        verbs,
        verb_relationships,
        verb_examples,
    )

    print("Generated GR00T-H N1.7 language visualizations:")
    for filename, _, _ in figures:
        print(f"  {output_dir / filename}")
    print(f"  {length_examples_path}")
    print(f"  {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
