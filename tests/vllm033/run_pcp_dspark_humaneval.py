"""Run the existing full HumanEval client against PCP8 static EPLB32 DSpark5."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request

import validate_pcp_dspark_gpu as setup


ROOT = setup.ROOT
OUT = setup.OUT
MODEL_NAME = 'dsv41-flash-eplb32-dspark5'


def save(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def client_environment():
    env = os.environ.copy()
    for key in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'LD_PRELOAD'):
        env.pop(key, None)
    env.update(HCU_ADAPTER_HOST='none', VLLM_PLUGINS='__disabled__',
               HIP_VISIBLE_DEVICES='', CUDA_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='')
    return env


def stop_process(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=90)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def main():
    evaluator = [os.environ.get('HCU_HUMANEVAL_PYTHON', 'python3'), '-B',
                 str(Path(__file__).with_name('evaluate_humaneval.py')),
                 '--api-url', setup.BASE_URL + '/v1', '--model', MODEL_NAME,
                 '--batch-size', '16', '--max-tokens', '2048', '--limit', '164']
    args = setup.command(True)
    args[args.index('--served-model-name') + 1] = MODEL_NAME
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', help='print commands without starting services')
    options = parser.parse_args()
    if options.dry_run:
        print(json.dumps({'server': args, 'evaluator': evaluator}, indent=2))
        return
    OUT.mkdir(parents=True, exist_ok=False)
    print('RUN_DIR=' + str(OUT), flush=True)
    with socket.socket() as check:
        check.bind(('127.0.0.1', setup.PORT))
    client_env = client_environment()
    save('status.json', {'state': 'preflight'})
    with (OUT / 'preflight.log').open('w') as log:
        subprocess.run(evaluator + ['--check'], env=client_env, stdout=log,
                       stderr=subprocess.STDOUT, check=True, timeout=180)
    manifest = {'server_command': args, 'evaluator_command': evaluator,
                'model': setup.MODEL, 'parallelism': 'TP1/DP1/PCP8/EP8',
                'dspark_tokens': 5, 'redundant_experts': 32,
                'swa_replay': False, 'layer_split': False,
                'expert_map_path': setup.MAP,
                'expert_map_sha256': hashlib.sha256(Path(setup.MAP).read_bytes()).hexdigest()}
    save('manifest.json', manifest)
    service = None
    client = None
    try:
        with (OUT / 'server.log').open('w') as log:
            service = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        save('status.json', {'state': 'starting', 'server_pid': service.pid})
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            if service.poll() is not None:
                raise RuntimeError('Server exited {}'.format(service.returncode))
            try:
                listed = setup.request('/v1/models', timeout=3)
                if any(row['id'] == MODEL_NAME for row in listed['data']):
                    break
            except (OSError, ValueError):
                pass
            time.sleep(3)
        else:
            raise TimeoutError('Server startup exceeded 1800 seconds')

        save('status.json', {'state': 'warmup', 'server_pid': service.pid})
        warmup = {'model': MODEL_NAME, 'messages': [
            {'role': 'user', 'content': 'Write a Python function that returns the sum of two integers.'}],
            'temperature': 0, 'max_tokens': 64,
            'chat_template_kwargs': {'thinking': False}}
        for repeat in range(2):
            with ThreadPoolExecutor(max_workers=16) as pool:
                responses = list(pool.map(
                    lambda index: setup.request('/v1/chat/completions', warmup, timeout=600),
                    range(16)))
            save('warmup_{}.json'.format(repeat), responses)

        save('status.json', {'state': 'evaluating', 'server_pid': service.pid,
                             'total': 164, 'concurrency': 16})
        with (OUT / 'evaluator.log').open('w') as log:
            client = subprocess.Popen(evaluator + ['--output', str(OUT / 'evalscope')],
                                      env=client_env, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True)
            deadline = time.monotonic() + 14400
            while client.poll() is None:
                if service.poll() is not None:
                    raise RuntimeError('Server exited during HumanEval')
                if time.monotonic() > deadline:
                    raise TimeoutError('HumanEval exceeded 4 hours')
                time.sleep(5)
            if client.returncode:
                raise RuntimeError('HumanEval client exited {}'.format(client.returncode))
        with urllib.request.urlopen(setup.BASE_URL + '/metrics') as response:
            (OUT / 'metrics.txt').write_bytes(response.read())
        save('status.json', {'state': 'evaluation_finished', 'evaluator_exit_code': 0})
    finally:
        if client is not None:
            stop_process(client)
        if service is not None:
            stop_process(service)


if __name__ == '__main__':
    def stop_requested(signum, frame):
        raise KeyboardInterrupt('Stopped by signal {}'.format(signum))

    signal.signal(signal.SIGTERM, stop_requested)
    try:
        main()
    except BaseException as error:
        if OUT.exists():
            save('status.json', {'state': 'failed', 'error': repr(error)})
        raise
