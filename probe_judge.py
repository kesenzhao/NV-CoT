from config import ROOT
"""Credential-safe binary API preflight; synthetic questions only."""
import json
import os
from pathlib import Path
from judge import judge_answer

results=[]
for answer, expected in [('red',1),('blue',0)]:
    try:
        score=judge_answer(answer,'red','What color is the red ball?',strict=True)
        results.append(dict(expected=expected,actual=score,passed=score==expected))
    except Exception as exc:
        # Never serialize request headers, response bodies, or credentials.
        results.append(dict(expected=expected,passed=False,error_type=type(exc).__name__))
report=dict(model=os.environ['LLM_AS_A_JUDGE_MODEL'],results=results,passed=all(x['passed'] for x in results))
(ROOT/'judge_preflight.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
raise SystemExit(0 if report['passed'] else 1)
