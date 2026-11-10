#!/usr/bin/env python3
"""Evaluate an existing FP8 server with EvalScope HumanEval."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-seconds', type=int, default=3600, help='maximum wait for an existing server')
    parser.add_argument('--check', action='store_true', help='load all data; send no requests')
    parser.add_argument('--api-url', default='http://127.0.0.1:16541/v1')
    parser.add_argument('--model', default='dsv41-flash-eplb32-dspark5')
    parser.add_argument('--data', type=Path, default=os.environ.get('HCU_HUMANEVAL_DATA'),
                        help='directory containing openai_humaneval/test-00000-of-00001.parquet')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-tokens', type=int, default=2048)
    parser.add_argument('--limit', type=int, default=164)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if min(args.batch_size, args.max_tokens, args.limit) < 1 or args.limit > 164:
        parser.error('Positive parameters required; limit must be at most 164')
    # The evaluator is an HTTP client, with no local model/GPU/plugin initialization.
    os.environ['HCU_ADAPTER_HOST'] = 'none'
    os.environ['VLLM_PLUGINS'] = '__disabled__'
    for key in ('HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'):
        os.environ[key] = ''
    os.environ['NO_PROXY'] = '127.0.0.1,localhost'
    os.environ['no_proxy'] = os.environ['NO_PROXY']
    if args.data is None:
        parser.error('Set --data or HCU_HUMANEVAL_DATA to the local HumanEval dataset')
    data = args.data.resolve()
    parquet = data / 'openai_humaneval/test-00000-of-00001.parquet'
    if not parquet.is_file():
        parser.error(f'Missing local dataset: {parquet}')
    from evalscope.config import TaskConfig
    from evalscope.api.registry import get_benchmark
    import evalscope.benchmarks.humaneval.humaneval_adapter  # register standard adapter
    config = TaskConfig(
        model=args.model,
        api_url=args.api_url,
        api_key='EMPTY',
        eval_type='openai_api',
        datasets=['humaneval'],
        dataset_args={'humaneval': {'local_path': str(data)}},
        dataset_dir=str(data.parent / 'cache'),
        eval_batch_size=args.batch_size,
        limit=args.limit,
        repeats=1,
        seed=42,
        generation_config={
            'temperature': 0,
            'max_tokens': args.max_tokens,
            'timeout': 1800,
            'stream': True,
            'extra_body': {'chat_template_kwargs': {'thinking': False}},
        },
        work_dir=str(args.output or Path.cwd() / 'test-results' /
                     datetime.datetime.now().strftime('humaneval_%Y%m%d_%H%M%S')),
        no_timestamp=True,
        ignore_errors=False,
    )
    adapter = get_benchmark('humaneval', config)
    datasets = adapter.load_dataset()
    samples = [sample for dataset in datasets.values() for sample in dataset]
    ids = [sample.metadata['task_id'] for sample in samples]
    expected = [f'HumanEval/{i}' for i in range(args.limit)]
    if ids != expected:
        raise RuntimeError(f'Unexpected HumanEval task IDs: {ids}')
    print(f'Dataset verified: {len(ids)} samples, {ids[0]} .. {ids[-1]}', flush=True)
    if args.check:
        print('Check passed; no API requests sent and service untouched.', flush=True)
        return
    output = Path(config.work_dir)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        'dataset_source': str(parquet),
        'dataset_sha256': hashlib.sha256(parquet.read_bytes()).hexdigest(),
        'task_ids': ids,
        'config': config.model_dump(mode='json'),
        'service_lifecycle': 'external; evaluator never starts or stops service',
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(f'Output: {output}', flush=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + args.wait_seconds
    while True:
        try:
            with opener.open(args.api_url.rstrip('/') + '/models', timeout=10) as response:
                models = json.load(response)
            if not any(item.get('id') == args.model for item in models.get('data', [])):
                raise RuntimeError(f'Model {args.model!r} not listed by server')
            break
        except (OSError, RuntimeError) as error:
            if time.monotonic() >= deadline:
                raise TimeoutError('Server did not become ready within the configured wait') from error
            print('Waiting for existing server to become ready...', flush=True)
            time.sleep(15)
    print('Server ready; starting HumanEval evaluation.', flush=True)
    from evalscope import run_task
    run_task(task_cfg=config)
    print(f'HumanEval evaluation finished. Service retained. Results: {output}', flush=True)


if __name__ == '__main__':
    main()
