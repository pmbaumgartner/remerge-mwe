from hypothesis import HealthCheck, given, settings, strategies as st
import pytest

from remerge import run


@pytest.mark.fast
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    corpus=st.lists(
        st.text(alphabet=st.sampled_from("abcd "), min_size=1, max_size=40),
        min_size=1,
        max_size=5,
    ),
    iterations=st.integers(min_value=1, max_value=10),
)
def test_run_property_invariants(corpus, iterations):
    winners = run(corpus, iterations, method="frequency")

    assert len(winners) <= iterations
    for winner in winners:
        assert len(winner.merged_lexeme.word) >= 2
        assert winner.merge_token_count >= 1
