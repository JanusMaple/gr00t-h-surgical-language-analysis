"""Auditable normalization rules for GR00T-H N1.7 language analysis.

Source prompts remain unchanged in the extracted corpus. This module creates a
separate semantic view and records every applied rule. It removes CMR state and
Rob tool templates, repairs observed token-boundary artifacts, applies POS-aware
lemmatization, and uses explicit corpus-audited aliases and POS overrides.
"""

from dataclasses import dataclass
import re

from nltk import pos_tag, sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordDetokenizer


CMR_MECHANISM = "CMR_STATE_PROMPT"
ROB_MECHANISM = "ROB_TOOL_PROMPT"

INVALID_LANGUAGE = frozenset({"", "unknown", "none", "n/a", "nan"})


# CMR state template; the final capture contains the procedure/task text.
CMR_PROMPT_PATTERN = re.compile(
    r"^arm left: ([^.]+)\. "
    r"left instrument: ([^.]+)\. "
    r"arm right: ([^.]+)\. "
    r"right instrument: ([^.]+)\. "
    r"(.+)$",
    flags=re.IGNORECASE,
)
CMR_CARRIER_PATTERN = re.compile(r"^do\s+(?:a|an)\s+", flags=re.IGNORECASE)


# Rob tool template. Observed tool values contain no periods.
ROB_PROMPT_PATTERN = re.compile(
    r"^left tool: ([^.]+)\. "
    r"right tool: ([^.]+)\. "
    r"aux tool: ([^.]+)\.\s*"
    r"(.*)$",
    flags=re.IGNORECASE,
)
ROB_CARRIER_PATTERN = re.compile(
    r"^The surgery process is\s+(?:a|an)\s+",
    flags=re.IGNORECASE,
)

# Run labels following Rob procedure names are metadata.
ROB_PROCEDURE_AND_RUN_PATTERN = re.compile(
    r"^(hemicolectomy|hysterectomy)"
    r"(?:\s+(InVivo\s+\d+|\d{4}-\d{2}-\d{2}_\d+))?"
    r"(?=\.|$)",
    flags=re.IGNORECASE,
)


# Lowercase compounds cannot be detected by the CamelCase rule.
LOWERCASE_COMPOUND_REPLACEMENTS = {
    "needlethreading": "needle threading",
    "needlepassing": "needle passing",
    "pegtransfer": "peg transfer",
    "suturepulling": "suture pulling",
}


# Explicit derivational aliases observed as surgical actions in this corpus.
# A general ``-tion`` suffix rule would create false matches.
ACTION_NOMINALIZATION_ALIASES = {
    "dissection": "dissect",
    "insertion": "insert",
    "retraction": "retract",
}

# Observed adjectival/domain forms that must not receive the verb fallback.
INFLECTION_LEMMA_EXCEPTIONS = frozenset(
    {
        "interrupted",
        "penetrating",
        "training",
        "underlying",
    }
)


MISSING_PUNCTUATION_SPACE = re.compile(r"(?<=[.!?;])(?=[A-Za-z])")
CAMEL_CASE_BOUNDARY = re.compile(r"(?<=[a-z])(?=[A-Z])")
UNDERSCORE_RUN = re.compile(r"_+")
WHITESPACE_RUN = re.compile(r"\s+")


# All 42 audited prompts use ``left`` and ``right`` directionally.
DIRECTIONAL_POS_OVERRIDES = {
    "left": "JJ",
    "right": "JJ",
}

# These forms modify nouns in every audited corpus occurrence.
ADJECTIVAL_POS_OVERRIDES = {
    "interrupted": "JJ",
    "penetrating": "JJ",
    "training": "JJ",
    "underlying": "JJ",
}


@dataclass(frozen=True)
class NormalizedLanguage:
    """One prompt's original, surface/canonical semantics, and audit trail."""

    original_text: str
    semantic_surface_text: str
    semantic_text: str
    removed_context: str
    rules_applied: tuple[str, ...]


def correct_pos_tag(word: str, nltk_tag: str) -> tuple[str, str | None]:
    """Apply narrowly scoped, corpus-audited POS corrections.

    Returns the corrected tag and the audit rule name, or ``None`` when the
    original NLTK tag is retained.
    """

    lowered = word.lower()
    corrected = DIRECTIONAL_POS_OVERRIDES.get(lowered)
    rule = "pos_override_directional_left_right"
    if corrected is None:
        corrected = ADJECTIVAL_POS_OVERRIDES.get(lowered)
        rule = "pos_override_corpus_adjective"
    if corrected is None or corrected == nltk_tag:
        return nltk_tag, None
    return corrected, rule


_LEMMATIZER = WordNetLemmatizer()
_DETOKENIZER = TreebankWordDetokenizer()


def _append_rule_once(rules: list[str], rule: str) -> None:
    if rule not in rules:
        rules.append(rule)


def _canonicalize_word(word: str, nltk_tag: str, rules: list[str]) -> str:
    """Return a readable base form while recording every transformation."""
    lowered = word.lower()
    if lowered != word:
        _append_rule_once(rules, "lowercase_semantic_text")

    corrected_tag, _ = correct_pos_tag(lowered, nltk_tag)
    if lowered in INFLECTION_LEMMA_EXCEPTIONS:
        lemma = lowered
    elif corrected_tag.startswith("VB"):
        lemma = _LEMMATIZER.lemmatize(lowered, "v")
    elif corrected_tag.startswith("NN"):
        lemma = _LEMMATIZER.lemmatize(lowered, "n")
    else:
        lemma = lowered

    # Recover action lemmas when short labels cause gerunds/participles to be
    # tagged as nouns or adjectives.
    if (
        lowered not in INFLECTION_LEMMA_EXCEPTIONS
        and lowered.endswith(("ing", "ed"))
    ):
        verb_lemma = _LEMMATIZER.lemmatize(lowered, "v")
        if verb_lemma != lowered:
            lemma = verb_lemma

    if lemma != lowered:
        _append_rule_once(rules, f"lemmatize_inflection:{lowered}->{lemma}")

    canonical = ACTION_NOMINALIZATION_ALIASES.get(lemma, lemma)
    if canonical != lemma:
        _append_rule_once(
            rules,
            f"canonicalize_action_nominalization:{lemma}->{canonical}",
        )
    return canonical


def _canonicalize_semantic_text(text: str, rules: list[str]) -> str:
    """Build the canonical prompt used by all aggregate NLTK statistics."""
    canonical_sentences: list[str] = []
    for sentence in sent_tokenize(text):
        tokens = word_tokenize(sentence)
        canonical_tokens = [
            _canonicalize_word(token, tag, rules) if token.isalpha() else token.lower()
            for token, tag in pos_tag(tokens)
        ]
        canonical_sentences.append(_DETOKENIZER.detokenize(canonical_tokens))

    canonical = " ".join(canonical_sentences).strip()
    without_terminal_punctuation = canonical.rstrip(".!?;:")
    if without_terminal_punctuation != canonical:
        _append_rule_once(rules, "strip_terminal_punctuation")
    return without_terminal_punctuation


def _replace_lowercase_compounds(text: str, rules: list[str]) -> str:
    for source, replacement in LOWERCASE_COMPOUND_REPLACEMENTS.items():
        pattern = re.compile(rf"\b{re.escape(source)}\b", flags=re.IGNORECASE)
        text, replacements = pattern.subn(replacement, text)
        if replacements:
            rules.append(f"expand_compound:{source}")
    return text


def _normalize_general_text(text: str, rules: list[str]) -> str:
    updated, replacements = MISSING_PUNCTUATION_SPACE.subn(" ", text)
    if replacements:
        rules.append("insert_space_after_punctuation")
    text = updated

    updated, replacements = UNDERSCORE_RUN.subn(" ", text)
    if replacements:
        rules.append("replace_underscores")
    text = updated

    updated, replacements = CAMEL_CASE_BOUNDARY.subn(" ", text)
    if replacements:
        rules.append("split_camel_case")
    text = updated

    text = _replace_lowercase_compounds(text, rules)

    updated = WHITESPACE_RUN.sub(" ", text).strip()
    if updated != text:
        rules.append("normalize_whitespace")
    return updated


def normalize_language(text: str, mechanism: str) -> NormalizedLanguage:
    """Create a semantic task view using auditable source-specific rules.

    Unexpected CMR or Rob templates raise ``ValueError`` rather than silently
    leaking template tokens into the semantic analysis.
    """

    original = "" if text is None else str(text).strip()
    rules: list[str] = []
    semantic = original
    removed_context = ""

    if original.lower() in INVALID_LANGUAGE:
        return NormalizedLanguage(
            original_text=original,
            semantic_surface_text="",
            semantic_text="",
            removed_context="",
            rules_applied=("drop_invalid_placeholder",),
        )

    if mechanism == CMR_MECHANISM:
        match = CMR_PROMPT_PATTERN.fullmatch(original)
        if match is None:
            raise ValueError(f"Unrecognized CMR prompt template: {original!r}")

        left_arm, left_instrument, right_arm, right_instrument, semantic = match.groups()
        removed_context = (
            f"arm left: {left_arm}. left instrument: {left_instrument}. "
            f"arm right: {right_arm}. right instrument: {right_instrument}"
        )
        rules.append("strip_cmr_state_template")

        semantic, replacements = CMR_CARRIER_PATTERN.subn("", semantic)
        if replacements != 1:
            raise ValueError(f"CMR procedure lacks expected 'do a/an' carrier: {original!r}")
        rules.append("strip_cmr_do_carrier")

    elif mechanism == ROB_MECHANISM:
        match = ROB_PROMPT_PATTERN.fullmatch(original)
        if match is None:
            raise ValueError(f"Unrecognized Rob tool template: {original!r}")

        left_tool, right_tool, aux_tool, semantic = match.groups()
        removed_context = (
            f"left tool: {left_tool}. right tool: {right_tool}. aux tool: {aux_tool}"
        )
        rules.append("strip_rob_tool_template")

        semantic, replacements = ROB_CARRIER_PATTERN.subn("", semantic)
        if replacements != 1:
            raise ValueError(f"Rob prompt lacks expected surgery-process carrier: {original!r}")
        rules.append("strip_rob_process_carrier")

        procedure_match = ROB_PROCEDURE_AND_RUN_PATTERN.match(semantic)
        if procedure_match is None:
            raise ValueError(f"Unrecognized Rob procedure prefix: {original!r}")
        procedure, run_identifier = procedure_match.groups()
        semantic = procedure + semantic[procedure_match.end():]
        if run_identifier is not None:
            rules.append("strip_rob_run_identifier")

        # Observed lower-to-upper transitions join Rob phase labels.
        semantic, replacements = CAMEL_CASE_BOUNDARY.subn(". ", semantic)
        if replacements:
            rules.append("split_rob_concatenated_phases")

    semantic_surface = _normalize_general_text(semantic, rules)
    semantic = _canonicalize_semantic_text(semantic_surface, rules)

    return NormalizedLanguage(
        original_text=original,
        semantic_surface_text=semantic_surface,
        semantic_text=semantic,
        removed_context=removed_context,
        rules_applied=tuple(rules),
    )
