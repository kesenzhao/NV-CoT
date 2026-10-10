import os,random,re,threading
from openai import OpenAI
from gemini_api_judge import GeminiJudgeClient

def get_chat_template():
    chat_template = """
Below are two answers to a question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Note that [Model Answer] is consistent with [Standard Answer] whenever they are essentially the same. If the meaning is expressed in the same way, it is considered consistent, for example, 'pink' and 'it is pink'.
If they are consistent, Judement is 1; if they are different, Judement is 0. Just output Judement and don't output anything else.\n\n
"""
    return chat_template

def get_gpt4_score_ICE():
    example_1 = """
[Question]: Is the countertop tan or blue?
[Standard Answer]: The countertop is tan.
[Model_answer] : tan
Judgement: 1
""" # noqa

    example_2 = """
[Question]: On which side of the picture is the barrier?
[Standard Answer]: The barrier is on the left side of the picture.
[Model_answer] : left
Judgement: 1
""" # noqa

    example_3 = """
[Question]: Is the kite brown and large?
[Standard Answer]: Yes, the kite is brown and large.
[Model_answer] : Yes
Judgement: 1
""" # noqa

    example_4 = """
[Question]: Are the spots on a giraffe?
[Standard Answer]: No, the spots are on a banana.
[Model_answer] : no
Judgement: 1
""" # noqa

    example_5 = """
[Question]: Who is wearing pants?
[Standard Answer]: The boy is wearing pants.
[Model_answer] : The person in the picture is wearing pants.
Judgement: 1
""" # noqa

    example_6 = """
[Question]: Is the man phone both blue and closed?
[Standard Answer]: Yes, the man phone is both blue and closed.
[Model_answer] : No.
Judgement: 0
""" # noqa

    example_7 = """
[Question]: What color is the towel in the center of the picture?
[Standard Answer]: The towel in the center of the picture is blue.
[Model_answer] : The towel in the center of the picture is pink.
Judgement: 0
""" # noqa

    return [example_1, example_2, example_3, example_4, example_5, example_6, example_7]

def get_prompt(predict_str, ground_truth, question):
    examples = get_gpt4_score_ICE()
    chat_template = get_chat_template()
    demo_prompt = chat_template
    for example in examples:
        demo_prompt += example + '\n\n'
    test_prompt = f"""
[Question]: {question}
[Standard Answer]: {ground_truth}
[Model_answer] : {predict_str}
Judgement:"""
    full_prompt = f'{demo_prompt}{test_prompt}'


    return full_prompt

def _tool_reward_terms(tool_used: bool, tool_count: float, acc_reward: float):
    """Return configurable success bonus and call cost.

    The two-stage experiment keeps the historical +1.2 correct-tool bonus and
    zero call cost. Environment variables make that objective explicit in the
    launcher instead of baking an untraceable experiment choice into code.
    """
    success_bonus = float(os.environ.get("NV_COT_TOOL_SUCCESS_BONUS", "1.2"))
    call_cost = float(os.environ.get("NV_COT_TOOL_CALL_COST", "0.0"))
    bonus = success_bonus if tool_used and acc_reward > 0.5 else 0.0
    cost = call_cost * max(float(tool_count), 0.0)
    return bonus, cost

def judge_answer(answer_text, ground_truth, question_text, *, strict=False):
    """Judge answer correctness only; trajectory-specific shaping stays separate."""
    full_prompt = get_prompt(answer_text, ground_truth, question_text)

    client_list, model_name_list = _get_judge_clients()
    client_idx = random.randint(0, len(client_list) - 1)
    client = client_list[client_idx]
    model_name = model_name_list[client_idx]

    chat_response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": full_prompt},
        ],
        seed = random.randint(0, 1000000),
        temperature=0.3,
    )
    response = chat_response.choices[0].message.content.strip()
    if strict:
        match = re.fullmatch(r"(?:(?:Judgement|Judgment|Judement)\s*:\s*)?([01])\s*\.?", response, re.IGNORECASE)
        if match is None:
            raise ValueError(f"Malformed binary judge response: {response[:160]!r}")
        return float(match.group(1))
    # print(response)
    if 'Judgement:' in response:
        response = response.split('Judgement:')[-1].strip()
        if '1' in response:
            acc_reward = 1.0
        elif '0' in response:
            acc_reward = 0.0
        else:
            print(f' [WARNING] resp format error {response=}')
            acc_reward = 0.0
    else:
        if response == '1':
            acc_reward = 1.0
        elif response == '0':
            acc_reward = 0.0
        else:
            print(f' [WARNING] resp format error {response=}')
            acc_reward = 0.0

    return acc_reward

_local = threading.local()
def _get_judge_clients():
    if not hasattr(_local, 'client'):
        backend = os.environ.get('JUDGE_BACKEND', 'openai')
        key = os.environ['JUDGE_API_KEY']
        base = os.environ['JUDGE_BASE_URL']
        model = os.environ['LLM_AS_A_JUDGE_MODEL']
        if backend == 'gemini':
            client = GeminiJudgeClient(api_key=key, base_url=base, max_output_tokens=2048)
        elif backend == 'openai':
            client = OpenAI(api_key=key, base_url=base, timeout=120., max_retries=2)
        else:
            raise ValueError('JUDGE_BACKEND must be openai or gemini')
        _local.client, _local.model = client, model
    return [_local.client], [_local.model]
