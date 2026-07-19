from hypothesis import HealthCheck, given, settings, strategies as st

from remerge import run


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
    repeated = run(corpus, iterations, method="frequency")
    source_tokens = {token for document in corpus for token in document.split()}

    assert winners == repeated
    assert len(winners) <= iterations
    for winner in winners:
        assert len(winner.merged_lexeme.word) >= 2
        assert set(winner.merged_lexeme.word) <= source_tokens
        assert winner.merge_token_count >= 1
