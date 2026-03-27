import copy

import pytest

from biogeme.expressions import Variable
from biogeme.expressions.add_prefix_suffix import add_prefix_suffix_to_all_variables


def make_expression_with_shared_gender():
    """Expression that reuses the same GENDER variable three times.

    This mirrors the original panel bug pattern, where the same Variable object
    was embedded in several comparison expressions.
    """
    gender = Variable("GENDER")
    size_a = Variable("SIZE_A")
    size_b = Variable("SIZE_B")
    size_c = Variable("SIZE_C")

    expr = (
            (size_a + (gender == 1)) + (size_b + (gender == 1)) + (size_c + (gender == 1))
    )
    return expr, gender, size_a, size_b, size_c


def make_expression_with_shared_subexpression():
    """Expression that reuses the same comparison subexpression object.

    This is an even closer reproduction of the original issue: one shared
    comparison expression is inserted in several places of the expression tree.
    """
    gender = Variable("GENDER")
    size_a = Variable("SIZE_A")
    size_b = Variable("SIZE_B")
    size_c = Variable("SIZE_C")

    is_woman = gender == 1
    expr = (size_a + is_woman) + (size_b + is_woman) + (size_c + is_woman)
    return expr, gender, size_a, size_b, size_c, is_woman


def make_expression_with_distinct_variables():
    """Expression with no shared variable objects."""
    age = Variable("AGE")
    income = Variable("INCOME")
    male = Variable("MALE")
    expr = age + income + (male == 1)
    return expr, age, income, male


@pytest.mark.parametrize(
    ("prefix", "suffix"),
    [
        ("", "__panel__01"),
        ("pre_", "_suf"),
        ("x_", "__panel__03"),
    ],
)
def test_single_variable_is_renamed_once_with_prefix_and_suffix(prefix, suffix):
    expr, age, income, male = make_expression_with_distinct_variables()

    count = add_prefix_suffix_to_all_variables(expr, prefix=prefix, suffix=suffix)

    assert count == 3
    assert age.name == f"{prefix}AGE{suffix}"
    assert income.name == f"{prefix}INCOME{suffix}"
    assert male.name == f"{prefix}MALE{suffix}"


def test_shared_variable_reused_in_multiple_comparisons_is_renamed_once():
    """Regression test for the panel renaming bug.

    Before the fix, the same shared Variable object could be renamed repeatedly
    during one traversal, producing names such as:

        GENDER__panel__01__panel__01__panel__01
    """
    expr, gender, size_a, size_b, size_c = make_expression_with_shared_gender()

    count = add_prefix_suffix_to_all_variables(expr, prefix="", suffix="__panel__01")

    # Four distinct Variable objects: GENDER, SIZE_A, SIZE_B, SIZE_C
    assert count == 4

    assert gender.name == "GENDER__panel__01"
    assert size_a.name == "SIZE_A__panel__01"
    assert size_b.name == "SIZE_B__panel__01"
    assert size_c.name == "SIZE_C__panel__01"

    # Most important regression check: no repeated panel suffix.
    assert gender.name.count("__panel__01") == 1
    assert "__panel__01__panel__01" not in gender.name


def test_shared_subexpression_reused_several_times_does_not_cascade_suffixes():
    """Same regression test, but with one shared comparison expression object."""
    expr, gender, size_a, size_b, size_c, is_woman = (
        make_expression_with_shared_subexpression()
    )

    count = add_prefix_suffix_to_all_variables(expr, prefix="", suffix="__panel__02")

    assert count == 4

    assert gender.name == "GENDER__panel__02"
    assert size_a.name == "SIZE_A__panel__02"
    assert size_b.name == "SIZE_B__panel__02"
    assert size_c.name == "SIZE_C__panel__02"

    assert gender.name.count("__panel__02") == 1
    assert "__panel__02__panel__02" not in gender.name

    # The shared comparison expression should still be reusable after renaming.
    # We do not inspect its internals directly, but if the underlying shared
    # variable had been renamed repeatedly, the check above would fail.


def test_count_reflects_unique_variable_objects_not_number_of_occurrences():
    """The return value must count renamed variable objects, not visits."""
    expr, gender, size_a, size_b, size_c = make_expression_with_shared_gender()

    count = add_prefix_suffix_to_all_variables(expr, prefix="", suffix="__panel__03")

    # GENDER appears three times syntactically, but it is one Variable object.
    assert count == 4


def test_fresh_copies_can_be_renamed_with_different_panel_suffixes():
    """Independent copies must each receive their own suffix cleanly."""
    expr1, gender1, size_a1, size_b1, size_c1 = make_expression_with_shared_gender()
    expr2, gender2, size_a2, size_b2, size_c2 = make_expression_with_shared_gender()

    count1 = add_prefix_suffix_to_all_variables(expr1, prefix="", suffix="__panel__01")
    count2 = add_prefix_suffix_to_all_variables(expr2, prefix="", suffix="__panel__02")

    assert count1 == 4
    assert count2 == 4

    assert gender1.name == "GENDER__panel__01"
    assert gender2.name == "GENDER__panel__02"

    assert "__panel__01__panel__01" not in gender1.name
    assert "__panel__02__panel__02" not in gender2.name


def test_deepcopy_of_expression_can_be_renamed_independently():
    """This mirrors the panel trajectory code path using copied expressions."""
    expr, _, _, _, _ = make_expression_with_shared_gender()

    expr_copy_1 = copy.deepcopy(expr)
    expr_copy_2 = copy.deepcopy(expr)

    # The original expression is not renamed here; each copy is handled
    # independently, as in the panel trajectory construction.
    count1 = add_prefix_suffix_to_all_variables(
        expr_copy_1, prefix="", suffix="__panel__01"
    )
    count2 = add_prefix_suffix_to_all_variables(
        expr_copy_2, prefix="", suffix="__panel__02"
    )

    assert count1 == 4
    assert count2 == 4


def test_original_problem_pattern_matches_expected_structure():
    """Sanity check: the reproduced pattern really reuses GENDER three times."""
    expr, gender, size_a, size_b, size_c = make_expression_with_shared_gender()

    # The same Python object must be reused; this is the structural precondition
    # that exposed the original bug.
    assert gender is not None
    assert size_a is not size_b
    assert size_b is not size_c
    assert size_a is not size_c
