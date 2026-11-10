"""Compare eager PCP8 static EPLB decoding with and without DSpark."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request


ROOT = Path(os.environ.get('HCU_VALIDATION_ROOT', Path(__file__).resolve().parents[2]))
OUT = Path(os.environ.get('HCU_VALIDATION_OUTPUT', ROOT / 'test-results' / (
    'pcp_dspark_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + str(os.getpid()))))
PORT = int(os.environ.get('HCU_VALIDATION_PORT', '16541'))
BASE_URL = 'http://127.0.0.1:{}'.format(PORT)
MODEL = os.environ.get('HCU_VALIDATION_MODEL', '')
MAP = os.environ.get('HCU_VALIDATION_EXPERT_MAP', '')


def save(name, payload):
    (OUT / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')


def request(path, payload=None, timeout=600):
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(BASE_URL + path, body, {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def command(speculative):
    if not MODEL or not MAP:
        raise ValueError('Set HCU_VALIDATION_MODEL and HCU_VALIDATION_EXPERT_MAP')
    args = [sys.executable, '-m', 'vllm.entrypoints.cli.main', 'serve', MODEL,
            '--tensor-parallel-size', '1', '--data-parallel-size', '1',
            '--prefill-context-parallel-size', '8', '--enable-expert-parallel',
            '--language-model-only', '--moe-backend', 'deep_gemm',
            '--all2all-backend', 'deepep_high_throughput', '--enforce-eager',
            '--no-swa-bounded-replay', '--gpu-memory-utilization', '0.85',
            '--block-size', '64', '--max-num-seqs', '16',
            '--max-num-batched-tokens', '8192', '--max-model-len', '16384',
            '--no-enable-prefix-caching', '--skip-mm-profiling',
            '--engram-config', json.dumps({'cpu_offload': True, 'dp_shared_memory': False,
                                          'embedding_across_dp': False, 'use_thp': False}),
            '--additional-config', json.dumps({'hcu_static_eplb': {
                'expert_map_path': MAP, 'num_redundant_experts': 32,
                'static_dispatch_policy': 'nearest'}}),
            '--host', '127.0.0.1', '--port', str(PORT), '--served-model-name', 'dspark-check']
    if speculative:
        args += ['--speculative-config', json.dumps({
            'method': 'dspark', 'num_speculative_tokens': 5, 'enforce_eager': True,
            'moe_backend': 'deep_gemm'})]
    return args


def phase(name, speculative):
    args = command(speculative)
    save(name + '_command.json', args)
    with (OUT / (name + '_server.log')).open('w') as log:
        service = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        save('status.json', {'phase': name, 'state': 'starting', 'pid': service.pid})
        try:
            deadline = time.monotonic() + 1800
            while time.monotonic() < deadline:
                if service.poll() is not None:
                    raise RuntimeError('{} service exited {}'.format(name, service.returncode))
                try:
                    request('/v1/models', timeout=3)
                    break
                except Exception:
                    time.sleep(3)
            else:
                raise TimeoutError('{} startup timed out'.format(name))
            save('status.json', {'phase': name, 'state': 'inference', 'pid': service.pid})
            prompts = [
                'Compute 12 times 13 and explain the calculation briefly. Answer in English.',
                ('The experiment compares deterministic model decoding across ranks. ' * 37)
                + '\nSummarize the experiment in one sentence.',
                ('春天的河水流过山谷，研究员记录水温与流速。' * 256)
                + '\n请简要说明研究员记录了什么。',
            ]
            payload = {'model': 'dspark-check', 'prompt': prompts, 'temperature': 0,
                       'seed': 42, 'max_tokens': 32, 'ignore_eos': True, 'logprobs': 1}
            save('request.json', payload)
            results = []
            for repeat in range(2):
                started = time.monotonic()
                response = request('/v1/completions', payload)
                elapsed = time.monotonic() - started
                save('{}_response_{}.json'.format(name, repeat), response)
                if len(response.get('choices', [])) != len(prompts):
                    raise AssertionError('Missing completion choices')
                if response['usage']['completion_tokens'] != 32 * len(prompts):
                    raise AssertionError('Not all requested output tokens were generated')
                results.append({'duration_s': elapsed, 'response': response})
            try:
                with urllib.request.urlopen(BASE_URL + '/metrics') as response:
                    (OUT / (name + '_metrics.txt')).write_bytes(response.read())
            except Exception as error:
                (OUT / (name + '_metrics_error.txt')).write_text(str(error))
            return results
        finally:
            if service.poll() is None:
                os.killpg(service.pid, signal.SIGTERM)
                try:
                    service.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    os.killpg(service.pid, signal.SIGKILL)
                    service.wait(timeout=30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', help='print commands without starting services')
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps({'baseline': command(False), 'dspark': command(True)}, indent=2))
        return
    command(True)
    OUT.mkdir(parents=True, exist_ok=False)
    with socket.socket() as check:
        check.bind(('127.0.0.1', PORT))
    baseline = phase('baseline', False)
    speculative = phase('dspark', True)
    comparisons = []
    for repeat, (plain, draft) in enumerate(zip(baseline, speculative)):
        left = sorted(plain['response']['choices'], key=lambda item: item['index'])
        right = sorted(draft['response']['choices'], key=lambda item: item['index'])
        for a, b in zip(left, right):
            comparisons.append({'repeat': repeat, 'index': a['index'],
                                'text_equal': a['text'] == b['text'],
                                'tokens_equal': a['logprobs']['tokens'] == b['logprobs']['tokens']})
    tokens_equal = all(row['tokens_equal'] for row in comparisons)
    repeat_comparisons = {}
    for name, results in (('baseline', baseline), ('dspark', speculative)):
        first, second = [sorted(result['response']['choices'], key=lambda row: row['index'])
                         for result in results]
        repeat_comparisons[name] = [
            left['logprobs']['tokens'] == right['logprobs']['tokens']
            for left, right in zip(first, second)]
    save('summary.json', {'comparisons': comparisons,
                          'execution_completed': True,
                          'all_tokens_equal': tokens_equal,
                          'token_equivalence_passed': tokens_equal,
                          'within_phase_repeat_tokens_equal': repeat_comparisons,
                          'baseline_duration_s': [r['duration_s'] for r in baseline],
                          'dspark_duration_s': [r['duration_s'] for r in speculative]})
    save('status.json', {'state': 'finished', 'execution_completed': True,
                         'token_equivalence_passed': tokens_equal, 'services_stopped': True})


if __name__ == '__main__':
    def stop_requested(signum, frame):
        raise KeyboardInterrupt('Validation stopped by signal {}'.format(signum))

    signal.signal(signal.SIGTERM, stop_requested)
    try:
        main()
    except BaseException as error:
        if OUT.exists():
            save('status.json', {'state': 'failed', 'error': repr(error)})
        raise
