import re

import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype

URL = "localhost:8000"
MODEL = "ensemble"
IS_RETURN_LOG_PROBS = True
START_ID = 1
END_ID = 50256
RANDOM_SEED = 42
STOP_WORDS_LIST = ["You:"]
BAD_WORDS_LIST = [""]
TOKENS_TO_GENERATE = 40
TEMPERATURE = 0.5
RUNTIME_TOP_K = 2
RUNTIME_TOP_P = 1.0
BEAM_SEARCH_DIVERSITY_RATE = 0.0
BEAM_WIDTH = 1
REPETITION_PENALTY = 1.0
LEN_PENALTY = 1.0


CLIENT = httpclient.InferenceServerClient(URL, concurrency=1, verbose=False)


def parse_history(chat_history: list):
    input = ""
    for turn in chat_history:
        if not turn[0]:
            input += turn[1] + "\n"
        else:
            input += "You: " + turn[0] + "\n" + turn[1] + "\n"
    return input


def completions_to_chat_history(completions):
    chat_pattern = re.compile(r"\n|You:|Alex:")
    chat_history_list = chat_pattern.split(completions)
    chat_history_list = [i.strip() for i in chat_history_list if not (i == """""")]
    chat_history_list.insert(0, None)

    chunked_chat_history = []
    for i in range(0, len(chat_history_list), 2):
        chunked_chat_history.append(chat_history_list[i : i + 2])

    return chunked_chat_history


def prepare_tensor(name, input):
    tensor = httpclient.InferInput(name, input.shape, np_to_triton_dtype(input.dtype))
    tensor.set_data_from_numpy(input)
    return tensor


def prepare_inputs(prompt):
    input0_data = np.array(prompt).astype(object)
    output0_len = np.ones_like(prompt).astype(np.uint32) * TOKENS_TO_GENERATE

    bad_words_list = np.array([BAD_WORDS_LIST], dtype=object)
    stop_words_list = np.array([STOP_WORDS_LIST], dtype=object)

    runtime_top_k = (RUNTIME_TOP_K * np.ones([input0_data.shape[0], 1])).astype(
        np.uint32
    )
    runtime_top_p = RUNTIME_TOP_P * np.ones([input0_data.shape[0], 1]).astype(
        np.float32
    )

    beam_search_diversity_rate = BEAM_SEARCH_DIVERSITY_RATE * np.ones(
        [input0_data.shape[0], 1]
    ).astype(np.float32)

    temperature = TEMPERATURE * np.ones([input0_data.shape[0], 1]).astype(np.float32)

    len_penalty = LEN_PENALTY * np.ones([input0_data.shape[0], 1]).astype(np.float32)

    repetition_penalty = REPETITION_PENALTY * np.ones([input0_data.shape[0], 1]).astype(
        np.float32
    )

    random_seed = RANDOM_SEED * np.ones([input0_data.shape[0], 1]).astype(np.uint64)

    is_return_log_probs = IS_RETURN_LOG_PROBS * np.ones(
        [input0_data.shape[0], 1]
    ).astype(bool)

    beam_width = (BEAM_WIDTH * np.ones([input0_data.shape[0], 1])).astype(np.uint32)
    start_id = START_ID * np.ones([input0_data.shape[0], 1]).astype(np.uint32)
    end_id = END_ID * np.ones([input0_data.shape[0], 1]).astype(np.uint32)

    inputs = [
        prepare_tensor("INPUT_0", input0_data),
        prepare_tensor("INPUT_1", output0_len),
        prepare_tensor("INPUT_2", bad_words_list),
        prepare_tensor("INPUT_3", stop_words_list),
        prepare_tensor("runtime_top_k", runtime_top_k),
        prepare_tensor("runtime_top_p", runtime_top_p),
        prepare_tensor("beam_search_diversity_rate", beam_search_diversity_rate),
        prepare_tensor("temperature", temperature),
        prepare_tensor("len_penalty", len_penalty),
        prepare_tensor("repetition_penalty", repetition_penalty),
        prepare_tensor("random_seed", random_seed),
        prepare_tensor("is_return_log_probs", is_return_log_probs),
        prepare_tensor("beam_width", beam_width),
        prepare_tensor("start_id", start_id),
        prepare_tensor("end_id", end_id),
    ]
    return inputs


def prepare_outputs(result):
    completions = result.as_numpy("OUTPUT_0")
    formatted_completions = []
    for completion in completions:
        tmp_string = completion.decode("utf-8")
        tmp_string = re.sub("<\|endoftext\|>", "", tmp_string)
        formatted_completions.append(tmp_string)

    return formatted_completions
