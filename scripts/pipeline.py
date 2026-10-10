"""Two-node shared-filesystem coordinator. Launch once on each node."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ROOT, CODE

p=argparse.ArgumentParser()
p.add_argument('--rank',type=int,choices=(0,1),required=True)
p.add_argument('--id',required=True,help='same NEW identifier on both nodes')
p.add_argument('--start',type=int,choices=(0,32,64,96),default=0)
p.add_argument('--until',type=int,choices=(32,64,96,128),default=128)
a=p.parse_args()
assert a.start<a.until and re.fullmatch(r'[A-Za-z0-9_-]+',a.id)
assert (ROOT/'plan.json').is_file(), 'Prepare data, preflight, smoke and plan first'
assert os.environ.get('MASTER_ADDR'), 'Set MASTER_ADDR on both nodes'
receipts=ROOT/'pipeline'/a.id
receipts.mkdir(parents=True,exist_ok=True)
# Never accept stale completion receipts for a restarted pipeline.
(receipts/f'node{a.rank}.started').open('x').close()

def healthy():
    if list(receipts.glob('*.failed')):
        raise RuntimeError('A pipeline node failed; inspect node logs')

def run(cmd):
    process=subprocess.Popen(cmd,cwd=CODE,start_new_session=True)
    try:
        while process.poll() is None:
            healthy()
            time.sleep(2)
        if process.returncode:
            raise RuntimeError(f'Stage exited with code {process.returncode}')
    finally:
        if process.poll() is None:
            import signal
            os.killpg(process.pid,signal.SIGTERM)
            try: process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                process.wait()

def barrier(label):
    (receipts/f'{label}.{a.rank}.done').touch()
    deadline=time.monotonic()+48*3600
    while not all((receipts/f'{label}.{rank}.done').exists() for rank in (0,1)):
        healthy()
        if time.monotonic()>deadline: raise TimeoutError(label)
        time.sleep(2)
    healthy()

def phase(name,step):
    print(name,step,'node',a.rank,flush=True)
    run(['bash',str(CODE/'scripts/run_node.sh'),str(a.rank),name,str(step)])
    barrier(f'{name}{step}')

try:
    barrier('startup')
    phase('eval',a.start)
    if a.rank==0:
        run([sys.executable,'summarize.py','--step',str(a.start)])
    barrier(f'summary{a.start}')
    for step in range(a.start+32,a.until+1,32):
        phase('train',step)
        phase('eval',step)
        if a.rank==0:
            run([sys.executable,'summarize.py','--step',str(step)])
            run([sys.executable,'curves.py'])
        barrier(f'summary{step}')
except BaseException:
    (receipts/f'node{a.rank}.failed').touch()
    raise
