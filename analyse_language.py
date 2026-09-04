"""Run mixture-weighted NLTK analysis on reconstructed GR00T-H N1.7 language.

The input contains one compact row per prompt/source combination, so this script
never expands or loads the underlying 124M+ frame-level observations. Each row
contributes in proportion to its normalized reconstructed training-mixture weight.
"""

from collections import Counter
import csv
import math
from pathlib import Path
import re

import nltk
from nltk import pos_tag, sent_tokenize, word_tokenize
from nltk.corpus import stopwords

from language_normalization import canonicalize_word, correct_pos_tag, normalize_language


INPUT_CSV = Path("extracted_language/gr00t_h_n17_public_language_counts.csv")
OUTPUT_DIR = Path("extracted_language")
WEIGHT_COLUMN = "normalized_reconstructed_weight"


# -----------------------------------------------------------------------------
# NLTK resources
# -----------------------------------------------------------------------------

NLTK_RESOURCES = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
    "stopwords": "corpora/stopwords",
}

for resource, location in NLTK_RESOURCES.items():
    try:
        nltk.data.find(location)
    except LookupError:
        try:
            nltk.data.find(f"{location}.zip")
        except LookupError:
            if not nltk.download(resource, quiet=True):
                raise RuntimeError(f"Could not install required NLTK resource: {resource}")

stop_words = set(stopwords.words("english"))


# -----------------------------------------------------------------------------
# Compact weighted corpus loading
# -----------------------------------------------------------------------------

def load_weighted_prompts(path: Path) -> tuple[list[dict], float, float]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run extract_gr00t_h_n17_training_language.py first."
        )

    audit_rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {
            "entry_id",
            "group",
            "mechanism",
            "historical_training_path",
            "current_public_path",
            "language",
            WEIGHT_COLUMN,
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        for line_no, row in enumerate(reader, 2):
            try:
                weight = float(row[WEIGHT_COLUMN])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid weight at {path}:{line_no}") from exc

            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"Invalid weight {weight!r} at {path}:{line_no}")

            normalized = normalize_language(row["language"], row["mechanism"])
            vlm_formalized_text = re.sub(
                r"[^\w\s]",
                "",
                normalized.original_text.lower(),
            )
            audit_rows.append(
                {
                    "entry_id": row["entry_id"],
                    "group": row["group"],
                    "mechanism": row["mechanism"],
                    "historical_training_path": row["historical_training_path"],
                    "current_public_path": row["current_public_path"],
                    "original_text": normalized.original_text,
                    "vlm_formalized_text": vlm_formalized_text,
                    "semantic_surface_text": normalized.semantic_surface_text,
                    "semantic_text": normalized.semantic_text,
                    "vlm_alphabetic_token_length": sum(
                        token.isalpha() for token in word_tokenize(vlm_formalized_text)
                    ),
                    "semantic_alphabetic_token_length": sum(
                        token.isalpha() for token in word_tokenize(normalized.semantic_text)
                    ),
                    "removed_template_context": normalized.removed_context,
                    "rules_applied": " | ".join(normalized.rules_applied),
                    "source_training_mixture_weight": weight,
                }
            )

    if not audit_rows:
        raise ValueError(f"No valid weighted prompts found in {path}")

    source_weight = sum(row["source_training_mixture_weight"] for row in audit_rows)
    if abs(source_weight - 1.0) > 1e-9:
        raise ValueError(f"Source prompt weights sum to {source_weight}, not 1")

    # Invalid placeholders are retained in the audit but excluded from NLTK.
    retained_weight = sum(
        row["source_training_mixture_weight"]
        for row in audit_rows
        if row["semantic_text"]
    )
    if retained_weight <= 0:
        raise ValueError("Retained prompt weights do not sum to a positive value")

    for row in audit_rows:
        row["semantic_analysis_weight"] = (
            row["source_training_mixture_weight"] / retained_weight
            if row["semantic_text"]
            else 0.0
        )

    return audit_rows, source_weight, retained_weight


def weighted_quantile(length_weights: Counter, quantile: float) -> int:
    threshold = quantile * sum(length_weights.values())
    cumulative = 0.0
    for length, weight in sorted(length_weights.items()):
        cumulative += weight
        if cumulative >= threshold:
            return length
    return max(length_weights)


def print_counter(title: str, counter: Counter, n: int = 30) -> None:
    total = sum(counter.values())
    print("=" * 78)
    print(title)
    print("=" * 78)
    print(f"{'item':<36} {'expected/sample':>18} {'token share':>14}")
    for item, expected_count in counter.most_common(n):
        label = " ".join(item) if isinstance(item, tuple) else item
        share = expected_count / total if total else 0.0
        print(f"{label:<36} {expected_count:>18.8f} {share:>13.4%}")
    print()


def write_counter(path: Path, item_column: str, counter: Counter) -> None:
    total = sum(counter.values())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [item_column, "expected_occurrences_per_training_sample", "token_share"]
        )
        for item, expected_count in counter.most_common():
            label = " ".join(item) if isinstance(item, tuple) else item
            writer.writerow(
                [label, expected_count, expected_count / total if total else 0.0]
            )


HARD_CLAUSE_BOUNDARIES = frozenset({".", "!", "?", ";", ":"})


def tag_semantic_sentence(
    sentence: str,
) -> list[tuple[str, str, str, str, str | None]]:
    """Return source token, NLTK tag, corrected tag, lemma, and audit rule."""
    tagged_tokens = []
    tokens = [
        token.lower() if token.isalpha() else token
        for token in word_tokenize(sentence)
    ]
    initial_tags = pos_tag(tokens)
    for index, (word, nltk_tag) in enumerate(initial_tags):
        previous_word = initial_tags[index - 1][0] if index else None
        next_word = (
            initial_tags[index + 1][0]
            if index + 1 < len(initial_tags)
            else None
        )
        next_next_word = (
            initial_tags[index + 2][0]
            if index + 2 < len(initial_tags)
            else None
        )
        tag, correction_rule = correct_pos_tag(
            word,
            nltk_tag,
            next_word=next_word,
            previous_word=previous_word,
            next_next_word=next_next_word,
        )
        lemma = (
            canonicalize_word(
                word,
                nltk_tag,
                [],
                next_word=next_word,
                previous_word=previous_word,
                next_next_word=next_next_word,
            )
            if word.isalpha()
            else word.lower()
        )
        tagged_tokens.append((word, nltk_tag, tag, lemma, correction_rule))
    return tagged_tokens


def following_noun_head_pairs(
    tagged_tokens: list[tuple[str, str, str, str, str | None]],
) -> list[tuple[str, str]]:
    """Pair each verb with the head of its first following noun phrase.

    This is a local head-final noun-sequence heuristic, not a dependency parse.
    Scanning stops at another verb or a hard punctuation boundary. Within the
    first contiguous noun sequence, the final noun is treated as its head.
    """
    pairs = []
    for index, (word, _, tag, lemma, _) in enumerate(tagged_tokens):
        if not word.isalpha() or not tag.startswith("VB"):
            continue
        noun_head = None
        for next_word, _, next_tag, next_lemma, _ in tagged_tokens[index + 1 :]:
            if next_word in HARD_CLAUSE_BOUNDARIES or next_tag.startswith("VB"):
                break
            if next_word.isalpha() and next_tag.startswith("NN"):
                noun_head = next_lemma
                continue
            if noun_head is not None:
                break
        if noun_head is not None:
            pairs.append((lemma, noun_head))
    return pairs


def write_verb_following_noun_outputs(
    relationship_path: Path,
    example_path: Path,
    verb_counter: Counter,
    pair_counter: Counter,
    prompt_pair_counter: Counter,
    audit_rows: list[dict],
) -> None:
    """Write compact relationship statistics and provenance-rich examples."""
    pair_totals_by_verb = Counter()
    for (verb, _), expected_count in pair_counter.items():
        pair_totals_by_verb[verb] += expected_count

    relationship_fields = [
        "verb",
        "following_noun",
        "verb_expected_occurrences_per_training_sample",
        "matched_noun_expected_occurrences_per_training_sample",
        "pair_expected_occurrences_per_training_sample",
        "pair_share_of_all_verb_occurrences",
        "pair_share_among_matched_nouns",
    ]
    with relationship_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=relationship_fields)
        writer.writeheader()
        for (verb, noun), pair_weight in sorted(
            pair_counter.items(),
            key=lambda item: (verb_counter[item[0][0]], item[1]),
            reverse=True,
        ):
            verb_weight = verb_counter[verb]
            matched_weight = pair_totals_by_verb[verb]
            writer.writerow(
                {
                    "verb": verb,
                    "following_noun": noun,
                    "verb_expected_occurrences_per_training_sample": verb_weight,
                    "matched_noun_expected_occurrences_per_training_sample": matched_weight,
                    "pair_expected_occurrences_per_training_sample": pair_weight,
                    "pair_share_of_all_verb_occurrences": pair_weight / verb_weight,
                    "pair_share_among_matched_nouns": pair_weight / matched_weight,
                }
            )

    audit_by_semantic: dict[str, list[dict]] = {}
    for row in audit_rows:
        if row["semantic_text"]:
            audit_by_semantic.setdefault(row["semantic_text"], []).append(row)
    for source_rows in audit_by_semantic.values():
        source_rows.sort(
            key=lambda row: row["semantic_analysis_weight"],
            reverse=True,
        )

    example_fields = [
        "verb",
        "following_noun",
        "example_rank",
        "canonical_semantic_text",
        "current_public_path",
        "historical_training_path",
        "prompt_pair_expected_occurrences_per_training_sample",
        "source_row_training_mixture_weight",
    ]
    prompts_by_pair: dict[tuple[str, str], list[tuple[str, float]]] = {}
    for (verb, noun, text), expected_count in prompt_pair_counter.items():
        prompts_by_pair.setdefault((verb, noun), []).append((text, expected_count))

    with example_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=example_fields)
        writer.writeheader()
        for pair in sorted(
            prompts_by_pair,
            key=lambda item: pair_counter[item],
            reverse=True,
        ):
            candidates = sorted(
                prompts_by_pair[pair],
                key=lambda item: item[1],
                reverse=True,
            )
            used_datasets: set[str] = set()
            selected = []
            for text, prompt_weight in candidates:
                source_rows = audit_by_semantic[text]
                source = next(
                    (
                        row
                        for row in source_rows
                        if row["current_public_path"] not in used_datasets
                    ),
                    source_rows[0],
                )
                selected.append((text, prompt_weight, source))
                used_datasets.add(source["current_public_path"])
                if len(selected) == 3:
                    break

            verb, noun = pair
            for rank, (text, prompt_weight, source) in enumerate(selected, 1):
                writer.writerow(
                    {
                        "verb": verb,
                        "following_noun": noun,
                        "example_rank": rank,
                        "canonical_semantic_text": text,
                        "current_public_path": source["current_public_path"],
                        "historical_training_path": source["historical_training_path"],
                        "prompt_pair_expected_occurrences_per_training_sample": prompt_weight,
                        "source_row_training_mixture_weight": source[
                            "semantic_analysis_weight"
                        ],
                    }
                )


def main() -> None:
    audit_rows, source_weight, retained_weight = load_weighted_prompts(INPUT_CSV)

    # Aggregate canonical prompts for the prompt-distribution output.
    semantic_weights = Counter()
    for row in audit_rows:
        if row["semantic_text"]:
            semantic_weights[row["semantic_text"]] += row["semantic_analysis_weight"]

    if abs(sum(semantic_weights.values()) - 1.0) > 1e-9:
        raise RuntimeError("Semantic prompt weights do not sum to 1")

    word_counter = Counter()
    verb_counter = Counter()
    noun_counter = Counter()
    bigram_counter = Counter()
    length_weights = Counter()
    pos_correction_counts = Counter()
    pos_correction_weights = Counter()
    verb_noun_counter = Counter()
    verb_noun_prompt_counter = Counter()

    for row in audit_rows:
        text = row["semantic_text"]
        if not text:
            continue
        surface_text = row["semantic_surface_text"]
        weight = row["semantic_analysis_weight"]
        tokens = word_tokenize(text)
        words = [word.lower() for word in tokens if word.isalpha()]

        length_weights[len(words)] += weight
        for word in words:
            if word not in stop_words:
                word_counter[word] += weight

        # POS-tag the clean task text before lemmatization changes its syntax.
        for sentence in sent_tokenize(surface_text):
            tagged_tokens = tag_semantic_sentence(sentence)
            for word, nltk_tag, tag, lemma, correction_rule in tagged_tokens:
                if not word.isalpha():
                    continue
                if correction_rule is not None:
                    key = (correction_rule, word.lower(), nltk_tag, tag)
                    pos_correction_counts[key] += 1
                    pos_correction_weights[key] += weight
                if tag.startswith("VB"):
                    verb_counter[lemma] += weight
                elif tag.startswith("NN"):
                    noun_counter[lemma] += weight

            for verb, noun in following_noun_head_pairs(tagged_tokens):
                verb_noun_counter[(verb, noun)] += weight
                verb_noun_prompt_counter[(verb, noun, text)] += weight

        # Bigrams use normalized task text and never cross sentence boundaries.
        for sentence in sent_tokenize(text):
            sentence_tokens = word_tokenize(sentence)
            for first, second in zip(sentence_tokens[:-1], sentence_tokens[1:]):
                if not first.isalpha() or not second.isalpha():
                    continue
                first = first.lower()
                second = second.lower()
                if first not in stop_words and second not in stop_words:
                    bigram_counter[(first, second)] += weight

    expected_length = sum(
        length * weight for length, weight in length_weights.items()
    )
    length_variance = sum(
        weight * (length - expected_length) ** 2
        for length, weight in length_weights.items()
    )

    print("=" * 78)
    print("GR00T-H N1.7 SEMANTIC TRAINING-MIXTURE LANGUAGE")
    print("=" * 78)
    print(f"Input:                         {INPUT_CSV}")
    print(f"Source weighted rows:          {len(audit_rows):,}")
    print(f"Unique semantic prompts:       {len(semantic_weights):,}")
    print(f"Source prompt weight:          {source_weight:.12f}")
    print(f"Retained semantic weight:      {retained_weight:.12f}")
    print(f"Dropped invalid weight:        {source_weight - retained_weight:.12f}")
    print(f"Normalized analysis weight:    {sum(semantic_weights.values()):.12f}")
    print("View:                          canonical task semantics; surface/VLM text audited")
    print()

    print("=" * 78)
    print("WEIGHTED INSTRUCTION LENGTH (alphabetic tokens)")
    print("=" * 78)
    print(f"mean:   {expected_length:.4f}")
    print(f"std:    {math.sqrt(length_variance):.4f}")
    print(f"min:    {min(length_weights)}")
    print(f"25%:    {weighted_quantile(length_weights, 0.25)}")
    print(f"median: {weighted_quantile(length_weights, 0.50)}")
    print(f"75%:    {weighted_quantile(length_weights, 0.75)}")
    print(f"max:    {max(length_weights)}")
    print()

    print_counter("TOP SEMANTIC CONTENT WORDS (MIXTURE-WEIGHTED)", word_counter)
    print_counter("TOP SEMANTIC VERBS, LEMMATIZED (MIXTURE-WEIGHTED)", verb_counter)
    print_counter("TOP SEMANTIC NOUNS, LEMMATIZED (MIXTURE-WEIGHTED)", noun_counter)
    print_counter("TOP ADJACENT SEMANTIC BIGRAMS (MIXTURE-WEIGHTED)", bigram_counter)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    word_path = OUTPUT_DIR / "gr00t_h_n17_nltk_word_frequency.csv"
    verb_path = OUTPUT_DIR / "gr00t_h_n17_nltk_verb_frequency.csv"
    noun_path = OUTPUT_DIR / "gr00t_h_n17_nltk_noun_frequency.csv"
    bigram_path = OUTPUT_DIR / "gr00t_h_n17_nltk_bigram_frequency.csv"
    length_path = OUTPUT_DIR / "gr00t_h_n17_nltk_instruction_length.csv"
    semantic_path = OUTPUT_DIR / "gr00t_h_n17_semantic_language.csv"
    audit_path = OUTPUT_DIR / "gr00t_h_n17_language_normalization_audit.csv"
    rule_path = OUTPUT_DIR / "gr00t_h_n17_language_normalization_rule_summary.csv"
    pos_path = OUTPUT_DIR / "gr00t_h_n17_pos_correction_summary.csv"
    verb_noun_path = OUTPUT_DIR / "gr00t_h_n17_verb_following_noun.csv"
    verb_noun_example_path = OUTPUT_DIR / "gr00t_h_n17_verb_following_noun_examples.csv"

    write_counter(word_path, "word", word_counter)
    write_counter(verb_path, "verb", verb_counter)
    write_counter(noun_path, "noun", noun_counter)
    write_counter(bigram_path, "bigram", bigram_counter)

    with length_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["alphabetic_token_length", "training_mixture_weight"])
        writer.writerows(sorted(length_weights.items()))

    with semantic_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["semantic_language", "normalized_training_mixture_weight"])
        writer.writerows(semantic_weights.most_common())

    with audit_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
        writer.writeheader()
        writer.writerows(audit_rows)

    rule_counts = Counter()
    rule_weights = Counter()
    for row in audit_rows:
        for rule in filter(None, row["rules_applied"].split(" | ")):
            rule_counts[rule] += 1
            rule_weights[rule] += row["source_training_mixture_weight"]

    with rule_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rule", "source_rows_affected", "source_mixture_weight_affected"])
        for rule, count in rule_counts.most_common():
            writer.writerow([rule, count, rule_weights[rule]])

    with pos_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rule",
                "word",
                "nltk_tag",
                "corrected_tag",
                "source_row_occurrences",
                "expected_corrections_per_training_sample",
            ]
        )
        for key, count in pos_correction_counts.most_common():
            rule, word, nltk_tag, corrected_tag = key
            writer.writerow(
                [rule, word, nltk_tag, corrected_tag, count, pos_correction_weights[key]]
            )

    write_verb_following_noun_outputs(
        verb_noun_path,
        verb_noun_example_path,
        verb_counter,
        verb_noun_counter,
        verb_noun_prompt_counter,
        audit_rows,
    )

    print("Saved:")
    for path in [
        word_path,
        verb_path,
        noun_path,
        bigram_path,
        length_path,
        semantic_path,
        audit_path,
        rule_path,
        pos_path,
        verb_noun_path,
        verb_noun_example_path,
    ]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
