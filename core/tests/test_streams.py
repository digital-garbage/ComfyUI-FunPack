"""model_args: an APPLY_MODEL wrapper reads its call the same whichever way it came."""

from core.streams import model_args


def test_positional_as_comfyui_calls_it():
    # model_base.apply_model: execute(x, t, c_concat, c_crossattn, control, transformer_options, **kw)
    named = model_args((None, "c", None, {"sigmas": 1}), {"minimax_payload": {}})
    assert named["transformer_options"] == {"sigmas": 1}
    assert named["c_crossattn"] == "c" and "minimax_payload" in named


def test_by_name_as_an_outer_wrapper_may_call_on():
    assert model_args((), {"transformer_options": {"a": 1}})["transformer_options"] == {"a": 1}
