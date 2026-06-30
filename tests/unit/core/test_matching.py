"""Unit tests for the matching utility module.

Tests for has_pattern, resolve_patterns, and resolve_include_exclude
which implement fnmatch-based wildcard support for column selection.
"""

from dqm_ml_core.utils.matching import has_pattern, resolve_include_exclude, resolve_patterns


class TestHasPattern:
    """Tests for has_pattern function detecting fnmatch wildcards.

    Covers recognition of *, ?, [...], [!...] patterns and rejection of literals.
    """

    def test_star_is_pattern(self):
        """Verify asterisk is recognized as a pattern."""
        assert has_pattern("*") is True

    def test_question_is_pattern(self):
        """Verify question mark is recognized as a pattern."""
        assert has_pattern("?") is True

    def test_char_class_is_pattern(self):
        """Verify character class brackets are recognized as a pattern."""
        assert has_pattern("[abc]") is True

    def test_negated_class_is_pattern(self):
        """Verify negated character class is recognized as a pattern."""
        assert has_pattern("[!abc]") is True

    def test_literal_not_pattern(self):
        """Verify literal string without wildcards is not a pattern."""
        assert has_pattern("hello") is False

    def test_empty_not_pattern(self):
        """Verify empty string is not a pattern."""
        assert has_pattern("") is False


class TestResolvePatterns:
    """Tests for resolve_patterns function matching patterns against candidate list.

    Covers literal pass-through, wildcard matching, character classes, mixed patterns,
    deduplication, and order preservation with first-match-wins semantics.
    """

    def test_all_literal_pass_through(self):
        """Verify literal patterns pass through matching candidates unchanged."""
        assert resolve_patterns(["a", "b"], ["a", "b", "c"]) == ["a", "b"]

    def test_star_matches_everything(self):
        """Verify star pattern matches all candidates."""
        assert resolve_patterns(["*"], ["a", "b"]) == ["a", "b"]

    def test_prefix_wildcard(self):
        """Verify prefix wildcard matches candidates with matching prefix."""
        assert resolve_patterns(["img_*"], ["img_a", "img_b", "other"]) == ["img_a", "img_b"]

    def test_question_matches_single_char(self):
        """Verify question mark matches exactly one character."""
        assert resolve_patterns(["img_?"], ["img_a", "img_ab"]) == ["img_a"]

    def test_char_class(self):
        """Verify character class matches any single character in class."""
        assert resolve_patterns(["[ab]"], ["a", "b", "c"]) == ["a", "b"]

    def test_mixed_literal_and_wildcard(self):
        """Verify mixed literal and wildcard patterns work together."""
        assert resolve_patterns(["id", "meta_*"], ["id", "meta_x", "meta_y", "other"]) == [
            "id",
            "meta_x",
            "meta_y",
        ]

    def test_no_match_returns_empty(self):
        """Verify pattern with no matches returns empty list."""
        assert resolve_patterns(["z_*"], ["a", "b"]) == []

    def test_duplicates_deduplicated(self):
        """Verify duplicate matches are deduplicated in result."""
        assert resolve_patterns(["*", "a"], ["a", "b"]) == ["a", "b"]

    def test_order_preserved_first_match_wins(self):
        """Verify result order follows pattern order with first match winning."""
        assert resolve_patterns(["b", "*"], ["a", "b"]) == ["b", "a"]


class TestResolveIncludeExclude:
    """Tests for resolve_include_exclude combining include and exclude patterns.

    Covers None/empty includes, literal and wildcard excludes, combined include/exclude
    scenarios, and edge cases with no matches.
    """

    def test_include_none_returns_all(self):
        """Verify None include returns all candidates when no exclude."""
        assert resolve_include_exclude(None, None, ["a", "b"]) == ["a", "b"]

    def test_include_empty_returns_empty(self):
        """Verify empty include list returns empty result."""
        assert resolve_include_exclude([], None, ["a", "b"]) == []

    def test_literal_exclude(self):
        """Verify literal exclude removes exact matches from include list."""
        assert resolve_include_exclude(["a", "b", "c"], ["b"], ["a", "b", "c"]) == ["a", "c"]

    def test_wildcard_exclude(self):
        """Verify wildcard exclude removes pattern matches from include list."""
        assert resolve_include_exclude(["a", "meta_x"], ["meta_*"], ["a", "meta_x"]) == ["a"]

    def test_include_none_with_exclude(self):
        """Verify None include with exclude removes matches from all candidates."""
        assert resolve_include_exclude(None, ["meta_*"], ["a", "meta_x"]) == ["a"]

    def test_exclude_no_match(self):
        """Verify exclude with no matches leaves include list unchanged."""
        assert resolve_include_exclude(["a", "b"], ["z"], ["a", "b"]) == ["a", "b"]

    def test_wildcard_include_and_literal_exclude(self):
        """Verify wildcard include combined with literal exclude works correctly."""
        assert resolve_include_exclude(["img_*"], ["img_bad"], ["img_a", "img_bad", "img_c"]) == [
            "img_a",
            "img_c",
        ]
