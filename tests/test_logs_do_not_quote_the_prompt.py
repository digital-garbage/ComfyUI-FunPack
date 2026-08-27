"""No log line prints the user's own words back at them.

The log is the thing that gets copied out and sent somewhere. Phrase memory holds the user's
prompt, so a line naming three learned phrases puts prompt text into every paste — which the
user then has to strip by hand, every time, and will eventually forget to.

Counts and weights say everything the line is for. The phrases say nothing that a count does
not, and they are the half that cannot be un-sent.
"""
import inspect
import re
import sys

sys.path.insert(0, ".")


def _source():
    import conditioning
    return inspect.getsource(conditioning.FunPackVideoRefinerV2._v2_apply_h3_token_weights)


def test_the_emphasis_line_reports_counts_not_phrases():
    src = _source()
    assert "of {len(weighted)} learned" in src          # the count survives
    assert "strongest x" in src                          # so does the magnitude
    assert "{t!r}" not in src                            # the phrases do not
    assert "e.g." not in src


def test_no_log_line_in_the_refiner_joins_phrase_text():
    """A guard on the shape, not on one message: `", ".join(...)` over phrase-ish values
    inside a log call is how this came back the first time."""
    import conditioning
    src = inspect.getsource(conditioning)
    offenders = [m.group(0) for m in re.finditer(r'"\s*\(e\.g\.\s*"\s*\+\s*", "\.join', src)]
    assert offenders == []


def test_the_sampler_reports_the_same_way():
    """The sampler prints its own emphasis line; it must follow the same rule."""
    import samplers
    src = inspect.getsource(samplers)
    assert "strongest x{strongest:.2f}" in src
    assert "token span(s) biased" in src
