"""Bounded addition experiment with held-out exact-match evaluation."""
import json
import random
import shutil
import time
from pathlib import Path

import torch

from matrix_network import MatrixNetwork
from matrix_network_addition import EOS, random_problem, vocab_for
from matrix_network_optimizer import MatrixNetworkOptimizer
from matrix_network_training import apply_batch_update
from matrix_network_utils import save_checkpoint


def heldout_problems(count, seed, excluded=()):
    rng = random.Random(seed)
    problems = {}
    excluded = set(excluded)
    while len(problems) < count:
        prompt, answer = random_problem(rng, 3, 10)
        if prompt not in excluded:
            problems[prompt] = answer
    return list(problems.items())


@torch.no_grad()
def exact_evaluate(model, problems):
    correct = 0
    generated_correct = teacher_correct = teacher_digit_correct = 0
    digit_total = teacher_total = stopped_count = 0
    examples = []
    for prompt, answer in problems:
        prediction, stopped = model.generate(prompt, max_len=5)
        ok = stopped and prediction == answer
        correct += int(ok)
        stopped_count += int(stopped)
        digit_total += len(answer)
        generated_correct += sum(a == b for a, b in zip(prediction, answer))
        model.reset_state()
        model.apply_context(model.encode(prompt))
        for token in answer + EOS:
            target_id = model.stoi[token]
            hit = int(model.predict() == target_id)
            teacher_correct += hit
            teacher_digit_correct += hit if token != EOS else 0
            teacher_total += 1
            model.apply_context([target_id])
        if ok and len(examples) < 10:
            examples.append({'prompt': prompt, 'answer': answer, 'prediction': prediction})
    return {'exact_match': correct / len(problems), 'correct': correct,
            'samples': len(problems), 'correct_examples': examples,
            'token_acc': generated_correct / digit_total,
            'teacher_forced_token_acc': teacher_correct / teacher_total,
            'teacher_forced_digit_acc': teacher_digit_correct / digit_total,
            'stop_rate': stopped_count / len(problems)}


def run_experiment(output_dir, *, n=64, batch_size=16, max_iters=200000,
                   max_seconds=1800, eval_every=5000, eval_samples=1000,
                   confirmation_samples=2000, seed=0, target_accuracy=0.5,
                   on_save=None, resume_from=None, update_noise_scale=0.01,
                   historical=False, stop_on_target=True, ablation=None):
    if ablation not in (None, 'objective', 'plain-one-hot', 'read', 'noise',
                        'no-softmax', 'no-hidden', 'no-normalization'):
        raise ValueError(f'Unknown ablation: {ablation}')
    if ablation is not None and not historical:
        raise ValueError('Ablations must start from historical=True')
    for name, value in {
        'batch_size': batch_size, 'max_iters': max_iters, 'max_seconds': max_seconds,
        'eval_every': eval_every, 'eval_samples': eval_samples,
        'confirmation_samples': confirmation_samples,
    }.items():
        if value <= 0:
            raise ValueError(f'{name} must be positive')
    if not 0 <= target_accuracy <= 1:
        raise ValueError('target_accuracy must be between zero and one')
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    validation = heldout_problems(eval_samples, 812341)
    confirmation = heldout_problems(confirmation_samples, 912341, dict(validation))
    excluded = set(dict(validation)) | set(dict(confirmation))
    rng = random.Random(seed)
    model_class = MatrixNetwork
    optimizer_class = MatrixNetworkOptimizer
    update_batch = apply_batch_update
    optimizer_options = {}
    architecture = 'double-right'
    if historical:
        from experiments.historical_97fe187.adapter import MatrixNetwork as model_class
        from experiments.historical_97fe187.matrix_network_optimizer import MatrixNetworkOptimizer as optimizer_class
        from experiments.historical_97fe187.matrix_network_training import apply_batch_update as update_batch
        optimizer_options = {'momentum_decay': 0.9, 'momentum_weight': 0.0}
        architecture = 'single-left-97fe187'
        if ablation == 'objective':
            from experiments.ablations import objective_only_update as update_batch
        elif ablation == 'plain-one-hot':
            from experiments.ablations import plain_one_hot_update as update_batch
        elif ablation in ('no-softmax', 'no-hidden', 'no-normalization'):
            from functools import partial
            from experiments.ablations import target_component_update
            update_batch = partial(target_component_update, remove=ablation)
        elif ablation == 'read':
            model_class = MatrixNetwork
            from experiments.ablations import read_only_update as update_batch
            architecture = 'double-right-historical-target'
        elif ablation == 'noise':
            optimizer_class = MatrixNetworkOptimizer
            optimizer_options = {}
    model = model_class(n=n, vocab=vocab_for(10), eos_token=EOS, device='cpu')
    optimizer = optimizer_class(model, base_lr=0.1, token_lr=1.0,
                                update_noise_scale=update_noise_scale, orthogonalize_period=100,
                                **optimizer_options)
    config = dict(n=n, batch_size=batch_size, max_iters=max_iters, max_seconds=max_seconds,
                  seed=seed, target_accuracy=target_accuracy, eval_every=eval_every,
                  eval_samples=eval_samples, confirmation_samples=confirmation_samples,
                  base_lr=0.1, token_lr=1.0, update_noise_scale=update_noise_scale,
                  architecture=architecture, historical=historical, stop_on_target=stop_on_target,
                  ablation=ablation,
                  objective=('margin-one-hot' if ablation == 'objective' else
                             'historical-softmax' if historical and ablation != 'plain-one-hot' else 'one-hot'),
                  noise_scaling='current' if ablation == 'noise' or not historical else 'historical')
    if ablation in ('no-softmax', 'no-hidden', 'no-normalization'):
        config['objective'] = f'historical-{ablation}'
    if ablation == 'objective':
        config['correct_margin'] = 0.1
    status = {'config': config, 'status': 'running', 'history': [], 'best_exact_match': -1.0}
    best_state = None
    start_iteration = 0
    elapsed_before = 0.0
    if resume_from is not None:
        source = Path(resume_from)
        checkpoint = torch.load(source / 'latest.pt', map_location='cpu', weights_only=False)
        if checkpoint['metadata'].get('architecture') != architecture:
            raise ValueError('Cannot resume a checkpoint from a different architecture')
        if checkpoint['metadata']['config'].get('ablation') != ablation:
            raise ValueError('Cannot resume a checkpoint from a different ablation')
        if checkpoint['metadata']['config'].get('objective') != config['objective']:
            raise ValueError('Cannot resume a checkpoint from a different objective')
        previous_config = checkpoint['metadata']['config']
        for key in ('n', 'batch_size', 'seed', 'update_noise_scale',
                    'eval_samples', 'confirmation_samples'):
            if key in previous_config and previous_config[key] != config[key]:
                raise ValueError(f'Cannot resume with a different {key}')
        model.load_state_dict(checkpoint['model_state'])
        model.reset_state()
        start_iteration = checkpoint['metadata']['iteration']
        previous = json.loads((source / 'status.json').read_text())
        status.update(history=previous['history'], best_exact_match=previous['best_exact_match'],
                      best_iteration=previous['best_iteration'], resumed_from=str(source))
        elapsed_before = previous['elapsed_seconds']
        best_state = torch.load(source / 'best.pt', map_location='cpu', weights_only=False)['model_state']
        if source.resolve() != output.resolve():
            shutil.copyfile(source / 'best.pt', output / 'best.pt')
            shutil.copyfile(source / 'latest.pt', output / 'latest.pt')
        # Older checkpoints lack RNG state. Advance their sampler to avoid
        # repeating the initial training examples; noise then starts a new stream.
        metadata = checkpoint['metadata']
        if 'random_state' in metadata:
            rng.setstate(metadata['random_state'])
            torch.set_rng_state(metadata['torch_rng_state'])
            optimizer.update_count = metadata['update_count']
        else:
            for _ in range(start_iteration * batch_size):
                while random_problem(rng, 3, 10)[0] in excluded:
                    pass
            optimizer.update_count = start_iteration
    start = time.monotonic()

    def persist():
        tmp = output / 'status.json.tmp'
        tmp.write_text(json.dumps(status, indent=2))
        tmp.replace(output / 'status.json')
        if on_save:
            on_save()

    if resume_from is not None:
        row = {'iteration': start_iteration, 'elapsed_seconds': elapsed_before,
               **exact_evaluate(model, validation)}
        status['history'].append(row)
        print(json.dumps(row), flush=True)
    print(json.dumps({'event': 'training_started', 'config': config}), flush=True)
    persist()
    for it in range(start_iteration + 1, max_iters + 1):
        sequences, starts = [], []
        while len(sequences) < batch_size:
            prompt, answer = random_problem(rng, 3, 10)
            if prompt in excluded:
                continue
            sequences.append(model.encode(prompt + answer + EOS))
            starts.append(len(prompt))
        update_batch(model, optimizer, sequences, starts, 1.0)
        expired = elapsed_before + time.monotonic() - start >= max_seconds
        if it % eval_every != 0 and it != max_iters and not expired:
            continue
        metrics = exact_evaluate(model, validation)
        row = {'iteration': it, 'elapsed_seconds': elapsed_before + time.monotonic() - start, **metrics}
        status['history'].append(row)
        status['iteration'] = it
        status['elapsed_seconds'] = row['elapsed_seconds']
        print(json.dumps(row), flush=True)
        if metrics['exact_match'] > status['best_exact_match']:
            status['best_exact_match'] = metrics['exact_match']
            status['best_iteration'] = it
            best_state = {key: value.clone() for key, value in model.state_dict().items()}
            save_checkpoint(model, str(output / 'best.pt'), metadata={
                'task': 'addition', 'number_base': 10, 'addend_digits': 3,
                'architecture': architecture, 'iteration': it, 'config': config})
        save_checkpoint(model, str(output / 'latest.pt'), metadata={
            'task': 'addition', 'number_base': 10, 'addend_digits': 3,
            'architecture': architecture, 'iteration': it, 'config': config,
            'random_state': rng.getstate(), 'torch_rng_state': torch.get_rng_state(),
            'update_count': optimizer.update_count})
        persist()
        if (stop_on_target and metrics['exact_match'] >= target_accuracy) or expired:
            break
    model.load_state_dict(best_state)
    model.reset_state()
    status['confirmation'] = exact_evaluate(model, confirmation)
    status['target_met'] = (status['best_exact_match'] >= target_accuracy
                            and status['confirmation']['exact_match'] >= target_accuracy)
    status['status'] = 'completed'
    status['elapsed_seconds'] = elapsed_before + time.monotonic() - start
    persist()
    print(json.dumps(status), flush=True)
    return status
