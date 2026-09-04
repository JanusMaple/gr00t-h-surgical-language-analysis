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

# Contexts observed in the public Surgical corpus. ``ex vivo`` modifies tissue,
# while color and shape terms modify the head noun in ``<color> <shape> peg``.
EX_VIVO_HEAD_NOUNS = frozenset({"tissue"})
PEG_COLOR_MODIFIERS = frozenset({"black", "green", "orange", "white"})
PEG_SHAPE_MODIFIERS = frozenset({"cylinder", "triangular"})
TUNDRA_GRASPED_OBJECTS = frozenset(
    {"colon", "gallbladder", "intestine", "omentum", "stomach"}
)
SENTENCE_INITIAL_IMPERATIVE_VERBS = frozenset({"approach", "perform"})


@dataclass(frozen=True)
class NormalizedLanguage:
    """One prompt's original, surface/canonical semantics, and audit trail."""

    original_text: str
    semantic_surface_text: str
    semantic_text: str
    removed_context: str
    rules_applied: tuple[str, ...]


def correct_pos_tag(
    word: str,
    nltk_tag: str,
    next_word: str | None = None,
    previous_word: str | None = None,
    next_next_word: str | None = None,
) -> tuple[str, str | None]:
    """Apply narrowly scoped, corpus-audited POS corrections.

    Returns the corrected tag and the audit rule name, or ``None`` when the
    original NLTK tag is retained.
    """

    lowered = word.lower()
    previous = (previous_word or "").lower()
    following = (next_word or "").lower()
    following_after = (next_next_word or "").lower()

    # Both tokens in the audited phrase "ex vivo tissue" modify ``tissue``.
    if (
        lowered == "ex"
        and following == "vivo"
        and following_after in EX_VIVO_HEAD_NOUNS
    ) or (
        lowered == "vivo"
        and previous == "ex"
        and following in EX_VIVO_HEAD_NOUNS
    ):
        if nltk_tag != "JJ":
            return "JJ", "pos_override_ex_vivo_modifier"
        return nltk_tag, None

    # NLTK treats ``orange`` and ``cylinder`` as nouns in the audited peg
    # instructions. They are attributes of the object whose head noun is peg.
    if lowered in PEG_COLOR_MODIFIERS and following in PEG_SHAPE_MODIFIERS:
        if nltk_tag != "JJ":
            return "JJ", "pos_override_peg_color_modifier"
        return nltk_tag, None
    if lowered in PEG_SHAPE_MODIFIERS and following == "peg":
        if nltk_tag != "JJ":
            return "JJ", "pos_override_peg_shape_modifier"
        return nltk_tag, None

    # ``knot tying`` names a task/action, so ``tying`` is nominal rather than
    # an independent verb. This still canonicalizes to the readable base form
    # ``tie`` while remaining a noun in POS-dependent statistics.
    if lowered == "tying" and previous == "knot":
        if nltk_tag != "NN":
            return "NN", "pos_override_nominal_knot_tying"
        return nltk_tag, None

    if lowered == "plastic" and following == "phantom":
        if nltk_tag != "JJ":
            return "JJ", "pos_override_plastic_phantom_modifier"
        return nltk_tag, None

    # TUNDRA uses passive-like prompts such as ``the intestine grasped by the
    # surgeon``. For this report, the requested semantic event head is
    # canonicalized as the noun ``grasp`` following ``approach``.
    if (
        lowered == "grasped"
        and previous in TUNDRA_GRASPED_OBJECTS
        and following == "by"
    ):
        if nltk_tag != "NN":
            return "NN", "pos_override_tundra_grasp_event_noun"
        return nltk_tag, None

    # NLTK sometimes tags sentence-initial imperatives as nouns after the
    # casing normalization used for stable corpus-wide POS tagging.
    if lowered in SENTENCE_INITIAL_IMPERATIVE_VERBS and not previous:
        if nltk_tag != "VB":
            return "VB", "pos_override_sentence_initial_imperative"
        return nltk_tag, None

    # Audited JHU prompts use "cutting position" as a destination noun phrase.
    # Other cutting/clipping forms retain their ordinary action interpretation.
    if lowered == "cutting" and following == "position":
        if nltk_tag != "JJ":
            return "JJ", "pos_override_attributive_cutting_position"
        return nltk_tag, None

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


def canonicalize_word(
    word: str,
    nltk_tag: str,
    rules: list[str],
    next_word: str | None = None,
    previous_word: str | None = None,
    next_next_word: str | None = None,
) -> str:
    """Return a readable base form while recording every transformation."""
    lowered = word.lower()
    if lowered != word:
        _append_rule_once(rules, "lowercase_semantic_text")

    corrected_tag, correction_rule = correct_pos_tag(
        lowered,
        nltk_tag,
        next_word=next_word,
        previous_word=previous_word,
        next_next_word=next_next_word,
    )
    preserve_attributive_form = (
        correction_rule == "pos_override_attributive_cutting_position"
    )
    if preserve_attributive_form:
        _append_rule_once(rules, "preserve_attributive_cutting_position")
    elif correction_rule == "pos_override_ex_vivo_modifier":
        _append_rule_once(rules, "preserve_ex_vivo_modifier")
    elif correction_rule in {
        "pos_override_peg_color_modifier",
        "pos_override_peg_shape_modifier",
    }:
        _append_rule_once(rules, "preserve_peg_attribute_modifiers")
    elif correction_rule == "pos_override_nominal_knot_tying":
        _append_rule_once(rules, "canonicalize_nominal_knot_tying")
    elif correction_rule == "pos_override_plastic_phantom_modifier":
        _append_rule_once(rules, "preserve_plastic_phantom_modifier")
    elif correction_rule == "pos_override_tundra_grasp_event_noun":
        _append_rule_once(rules, "canonicalize_tundra_grasp_event_noun")
    elif correction_rule == "pos_override_sentence_initial_imperative":
        _append_rule_once(rules, "preserve_sentence_initial_imperative_verb")

    preserve_nominal_inflection = correction_rule in {
        "pos_override_nominal_knot_tying",
        "pos_override_tundra_grasp_event_noun",
    }
    if correction_rule == "pos_override_nominal_knot_tying":
        lemma = "tie"
    elif correction_rule == "pos_override_tundra_grasp_event_noun":
        lemma = "grasp"
    elif lowered in INFLECTION_LEMMA_EXCEPTIONS:
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
        and not preserve_attributive_form
        and not preserve_nominal_inflection
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
        tagged_tokens = pos_tag(tokens)
        canonical_tokens = []
        for index, (token, tag) in enumerate(tagged_tokens):
            previous_word = tagged_tokens[index - 1][0] if index else None
            next_word = (
                tagged_tokens[index + 1][0]
                if index + 1 < len(tagged_tokens)
                else None
            )
            next_next_word = (
                tagged_tokens[index + 2][0]
                if index + 2 < len(tagged_tokens)
                else None
            )
            canonical_tokens.append(
                canonicalize_word(
                    token,
                    tag,
                    rules,
                    next_word=next_word,
                    previous_word=previous_word,
                    next_next_word=next_next_word,
                )
                if token.isalpha()
                else token.lower()
            )
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
