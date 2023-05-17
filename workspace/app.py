import logging

import gradio as gr
from nemochat import *

logging.basicConfig(format="%(asctime)s %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)

md_header = """
# NeMo Megatron-GPT 20B
"""
md_footer = """
<style>
img {
 display: inline;
}
</style>

[![Model architecture](https://img.shields.io/badge/Model%20Arch-Transformer%20Decoder-green)](https://github.com/NVIDIA/NeMo)[![Model size](https://img.shields.io/badge/Params-20B-green)](https://arxiv.org/pdf/1909.08053.pdf)[![Language](https://img.shields.io/badge/Language-en--US-lightgrey#model-badge)](https://arxiv.org/abs/2101.00027)

## Model Description

Megatron-GPT 20B is a transformer-based language model. GPT refers to a class of transformer decoder-only models similar to GPT-2 and 3 while 20B refers to the total trainable parameter count (20 Billion) [1, 2].

This model was trained with [NeMo Megatron](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/nlp/nemo_megatron/intro.html).

## Training Data

The model was trained on ["The Piles" dataset prepared by Eleuther.AI](https://pile.eleuther.ai/). [4]

## Evaluation results

_Zero-shot performance._ Evaluated using [LM Evaluation Test Suite from AI21](https://github.com/AI21Labs/lm-evaluation)

| ARC-Challenge | ARC-Easy | RACE-middle | RACE-high | Winogrande | RTE    | BoolQA | HellaSwag | PiQA   |
| ------------- | -------- | ----------- | --------- | ---------- | ------ | ------ | --------- | ------ |
| 0.4403        | 0.6141   | 0.5188      | 0.4277    | 0.659      | 0.5704 | 0.6954 | 0.721     | 0.7688 |

## Limitations

The model was trained on the data originally crawled from the Internet. This data contains toxic language and societal biases. Therefore, the model may amplify those biases and return toxic responses especially when prompted with toxic prompts.

## References

[1] [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)

[2] [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/pdf/1909.08053.pdf)

[3] [NVIDIA NeMo Toolkit](https://github.com/NVIDIA/NeMo)

[4] [The Pile: An 800GB Dataset of Diverse Text for Language Modeling](https://arxiv.org/abs/2101.00027)

## Licence

License to use this model is covered by the [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). By downloading the public and release version of the model, you accept the terms and conditions of the [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) license.
"""

css = """
body {
        background-image: url('file=/workspace/content/trianglify.svg');
        background-size: cover;
    } 
footer {
    visibility: hidden
    }
"""
theme = gr.themes.Base(primary_hue="green", text_size="lg").set(
    body_background_fill="none",
    body_background_fill_dark="none",
    button_primary_background_fill="#76B900",
    button_primary_background_fill_hover="#569700",
    button_primary_background_fill_hover_dark="#569700",
    button_primary_text_color="none",
    button_primary_text_color_dark="none",
)


with gr.Blocks(theme=theme, css=css, title="NVIDIA NeMo") as demo:
    gr.Markdown(md_header)
    chat_history = [
        [
            (
                None,
                "Alex is a cheerful AI assistant and companion, created by NVIDIA engineers. Alex is clever and helpful, and will do everything it can to cheer you up. 🤗",
            )
        ]
    ]
    chatbot = gr.Chatbot(chat_history[-1])
    msg = gr.Textbox()
    clear = gr.Button("Clear")
    # gr.Markdown(md_footer)

    def respond(message, chat_history):
        prompt = parse_history(chat_history) + "You: {}".format(message)
        inputs = prepare_inputs([[prompt]])
        result = CLIENT.infer(MODEL, inputs)
        completions = prepare_outputs(result)[0]
        chat_history = completions_to_chat_history(completions)
        logging.info(chat_history[-1])
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

demo.queue().launch(favicon_path="/workspace/content/faviconV2.png")
