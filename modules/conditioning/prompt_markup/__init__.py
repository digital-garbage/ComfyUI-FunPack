"""Prompt markup -- (word:1.5), [phrase@2-4], [a|b]. A node-only module.

Two nodes, because the markup has to be split around the encoder: it must come
OUT before the encoder sees the text (Qwen reads "(word:1.5)" as punctuation in
your sentence), and it can only be applied AFTER, once there is a conditioning
to find the words in and a model to bias.

    text -> Prompt Markup -> clean -> (any H3 encode node) -> positive
                          -> markup ---------------------> Apply Prompt Markup
                                                           (+ model, clip, latent)

Where the words sit, and how seconds map to rows, is the MODEL's business: model
modules announce `text_tokenizer`, `prompt_rows` and `video_time_rows`. With
none that recognise the encoder, the markup is stripped and said to be unused --
never silently ignored.

The weight is an attention BIAS, and SLA runs any biased call dense, so a run
with weights or timed phrases runs dense attention.
"""

from .nodes import FunPackApplyPromptMarkup, FunPackPromptMarkup

ID = "conditioning_prompt_markup"
TITLE = "Prompt markup"
STAGE = "conditioning"
CATEGORY = "conditioning"
STATUS = "proven"

NODES = [FunPackPromptMarkup, FunPackApplyPromptMarkup]
