"""DeepEyes default correct-tool shaping, with strict rollout validity.

Reference: verl/utils/reward_score/vl_agent.py::_tool_reward_terms and
compute_score. Freeze the default coefficients here rather than allowing
ambient environment variables to silently change an experiment's objective.
"""
REWARD_SPEC = {
    'version': 'deepeyes_correct_executed_tool_v1',
    'correctness_weight': 0.8,
    'protocol_failure_penalty': 0.2,
    'tool_success_bonus': 1.2,
    'tool_call_cost': 0.0,
    'tool_evidence': 'successfully executed crop actions, never generated text',
}


def reward_components(correctness, protocol_failed, executed_tools):
    if correctness not in (0., 1.):
        raise ValueError('Correctness must be binary')
    if isinstance(executed_tools, bool) or not isinstance(executed_tools, int) or executed_tools < 0:
        raise ValueError('Executed tool count must be a nonnegative integer')
    # Keep the existing strict protocol gate: failed trajectories cannot get
    # correctness or a tool bonus even if they contain a plausible answer.
    correct = float(correctness) if not protocol_failed else 0.
    answer_reward = REWARD_SPEC['correctness_weight'] * correct
    format_reward = -REWARD_SPEC['protocol_failure_penalty'] if protocol_failed else 0.
    tool_bonus = REWARD_SPEC['tool_success_bonus'] if executed_tools > 0 and correct > .5 else 0.
    tool_cost = REWARD_SPEC['tool_call_cost'] * executed_tools
    return dict(answer_reward=answer_reward, format_reward=format_reward,
                tool_bonus=tool_bonus, tool_cost=tool_cost,
                total=answer_reward + format_reward + tool_bonus - tool_cost)
