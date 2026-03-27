from __future__ import annotations

import pytest
from biogeme.latent_variables.tex_utils import tex_escape, tex_identifier


def test_tex_escape_returns_empty_string_for_empty_input() -> None:
    assert tex_escape("") == ""


def test_tex_escape_returns_same_text_when_nothing_needs_escaping() -> None:
    assert tex_escape("abcXYZ123") == "abcXYZ123"


def test_tex_escape_escapes_backslash() -> None:
    assert tex_escape("\\") == r"\textbackslash\{\}"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("#", r"\#"),
        ("$", r"\$"),
    ],
)
def test_tex_escape_escapes_each_special_character(raw: str, expected: str) -> None:
    assert tex_escape(raw) == expected


def test_tex_escape_escapes_all_supported_special_characters_in_one_string() -> None:
    raw = r"\_{}%&#$"
    expected = r"\textbackslash\{\}" r"\_" r"\{" r"\}" r"\%" r"\&" r"\#" r"\$"

    assert tex_escape(raw) == expected


def test_tex_escape_preserves_regular_characters_while_escaping_special_ones() -> None:
    raw = r"alpha_beta%42&x#y${z}\path"
    expected = r"alpha\_beta\%42\&x\#y\$\{z\}" r"\textbackslash\{\}path"

    assert tex_escape(raw) == expected


def test_tex_escape_order_handles_backslash_before_braces() -> None:
    raw = r"\{"
    expected = r"\textbackslash\{\}\{"

    assert tex_escape(raw) == expected


def test_tex_escape_order_does_not_reescape_inserted_backslash_sequences() -> None:
    raw = "_"
    result = tex_escape(raw)

    assert result == r"\_"
    assert r"\textbackslash{}_" not in result


def test_tex_identifier_wraps_plain_text_in_mathrm() -> None:
    assert tex_identifier("alpha") == r"\mathrm{alpha}"


def test_tex_identifier_uses_tex_escape_for_special_characters() -> None:
    raw = r"a_b%$"
    expected = r"\mathrm{a\_b\%\$}"

    assert tex_identifier(raw) == expected


def test_tex_identifier_with_empty_string() -> None:
    assert tex_identifier("") == r"\mathrm{}"


def test_tex_identifier_with_backslash_and_braces() -> None:
    raw = r"\name{value}"
    expected = r"\mathrm{\textbackslash\{\}name\{value\}}"

    assert tex_identifier(raw) == expected
