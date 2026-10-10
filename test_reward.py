"""CPU-only reward regression tests; no model load, credentials, or API calls."""
import ast
import concurrent.futures
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
from reward import reward_components

ROOT = Path(__file__).resolve().parent


class RewardTest(unittest.TestCase):
    def test_truth_table(self):
        for correct, failed, tools, expected in [
            (1, False, 0, .8), (1, False, 1, 2.), (1, False, 3, 2.),
            (0, False, 0, 0.), (0, False, 3, 0.),
            (0, True, 0, -.2), (1, True, 3, -.2),
        ]:
            with self.subTest(correct=correct, failed=failed, tools=tools):
                self.assertAlmostEqual(reward_components(correct, failed, tools)['total'], expected)

    def test_matches_local_deepeyes_default_tool_terms(self):
        source = ROOT / 'judge.py'
        tree = ast.parse(source.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_tool_reward_terms')
        ns = {'os': os}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), 'exec'), ns)
        with patch.dict(os.environ, {}, clear=True):
            for correct in (0., 1.):
                for tools in (0, 1, 3):
                    bonus, cost = ns['_tool_reward_terms'](tools > 0, tools, correct)
                    actual = reward_components(correct, False, tools)
                    self.assertEqual((actual['tool_bonus'], actual['tool_cost']), (bonus, cost))

    def test_invalid_inputs(self):
        for correct, tools in [(.5, 1), (1, -1), (1, 1.5), (1, True)]:
            with self.assertRaises(ValueError):
                reward_components(correct, False, tools)

    def test_real_judge_group_integration(self):
        # Extract the actual function to avoid importing the CUDA/model stack.
        tree = ast.parse((ROOT / 'rollout.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'judge_group')
        ns = dict(concurrent=concurrent, reward_components=reward_components,
                  time=types.SimpleNamespace(sleep=lambda _: None))
        exec(compile(ast.Module(body=[fn], type_ignores=[]), 'rollout.py', 'exec'), ns)
        calls = []
        def judge(answer, *args, **kwargs):
            calls.append(answer)
            self.assertTrue(kwargs['strict'])
            return float(answer == 'right')
        fake = types.ModuleType('judge')
        fake.judge_answer = judge
        trajectories = [dict(answer=answer, error=error, actions=[{}]*tools)
                        for answer, error, tools in [('right', None, 0), ('right', None, 2),
                                                    ('wrong', None, 1), ('right', 'bad tool', 1)]]
        with patch.dict(sys.modules, {'judge': fake}):
            scores = ns['judge_group'](trajectories, dict(ground_truth='right', question='q'))
            self.assertEqual(scores, [.8, 2., 0., -.2])
            self.assertEqual(sorted(calls), ['right', 'wrong'])
            self.assertEqual(trajectories[1]['reward_components']['tool_bonus'], 1.2)
            fake.judge_answer = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('unavailable'))
            with self.assertRaisesRegex(RuntimeError, 'no substitute reward'):
                ns['judge_group']([dict(answer='right', error=None, actions=[])],
                                  dict(ground_truth='right', question='q'))


if __name__ == '__main__':
    unittest.main()
