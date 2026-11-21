import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from torch import nn

from training.backbone_benchmark import runner as reference
from training.backbone_benchmark.models import A3, TCN, build_model, logits_of
from training.backbone_benchmark.supplement.models import GRU, SmallGRU
from training.component_ablation import training as original
from training.component_ablation.features import sha256
from training.component_ablation.models import make_loader, set_seed


ROOT = Path(__file__).resolve().parents[2]
MODELS = (A3, TCN, GRU)
SEEDS = (42, 142, 242)
SOURCE_FILES = (
    'training/a3_tuning/runner.py', 'training/a3_tuning/test_runner.py',
    'training/component_ablation/features.py', 'training/component_ablation/models.py',
    'training/component_ablation/training.py', 'training/backbone_benchmark/models.py',
    'training/backbone_benchmark/runner.py', 'training/backbone_benchmark/supplement/models.py',
    'docs/research/a3-tuning-plan.md', 'docs/research/a3-tuning-execution.md',
)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def recipes():
    configurations = []
    for name, rate, decay, dropout, scheduler, smoothing in (
        ('C0', .001, .0001, .30, None, 0.),
        ('C1', .0003, .0001, .30, None, 0.),
        ('C2', .001, .001, .30, None, 0.),
        ('C3', .0003, .001, .15, None, 0.),
        ('C4', .0003, .001, .30, 'cosine', 0.),
        ('C5', .0003, .001, .30, 'cosine', .05),
    ):
        configuration = original.default_config('full')
        configuration.update(candidate=name, learning_rate=rate, weight_decay=decay,
                             dropout=dropout, scheduler=scheduler, label_smoothing=smoothing,
                             cosine_t_max=50, cosine_eta_min=.00001, checkpoint_interval=5)
        configurations.append(configuration)
    return configurations


def model_for(variant, dropout):
    return SmallGRU(dropout) if variant == GRU else build_model(variant, dropout)


def predict(model, data, indices, config, device):
    loader = make_loader(data, indices, original.VARIANTS[A3], config, False, 0)
    model.eval()
    probabilities, log_probabilities = [], []
    with torch.no_grad():
        for features, _ in loader:
            logits = logits_of(model, features.to(device))
            probabilities.extend(logits.softmax(1)[:, 1].cpu().tolist())
            log_probabilities.extend(logits.log_softmax(1).cpu().tolist())
    if not np.isfinite(probabilities).all() or not np.isfinite(log_probabilities).all():
        raise ValueError('Nonfinite inference')
    return np.asarray(probabilities), np.asarray(log_probabilities)


def training_rng(loader):
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(),
                loader=loader.generator.get_state(), augmentation=loader.dataset.rng.bit_generator.state)


def restore_rng(state, loader):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])
    loader.generator.set_state(state['loader'])
    loader.dataset.rng.bit_generator.state = state['augmentation']


def fit(variant, data, train, validation, config, seed, device, folder,
        run_signature, fixed_epochs=None, interrupt_after=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    metadata = dict(variant=variant, train=train.tolist(),
                    validation=None if validation is None else validation.tolist(),
                    config=config, seed=seed, fixed_epochs=fixed_epochs,
                    run_signature=run_signature)
    fit_signature = signature(metadata)
    state_path = folder / 'state.pt'
    done_path = folder / 'fit.json'
    if done_path.exists():
        done = read_json(done_path)
        if done['signature'] != fit_signature or sha256(state_path) != done['checkpoint_sha256']:
            raise ValueError('Completed fit signature/checkpoint mismatch')
        state = torch.load(state_path, map_location='cpu', weights_only=False)
        model = model_for(variant, config['dropout']).to(device)
        model.load_state_dict(state['best_state'])
        return model, done
    set_seed(seed)
    model = model_for(variant, config['dropout']).to(device)
    loader = make_loader(data, train, original.VARIANTS[A3], config, True, seed)
    counts = np.bincount(data['labels'][train], minlength=2)
    if np.any(counts == 0) or (validation is not None and set(train) & set(validation)):
        raise ValueError('Invalid fit split/classes')
    weights = len(train) / (2 * counts.astype(np.float32))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device),
                                    label_smoothing=config['label_smoothing'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['cosine_t_max'], eta_min=config['cosine_eta_min'])
        if config['scheduler'] == 'cosine' else None)
    history, best_state, best_loss, best_epoch, stale = [], None, float('inf'), 0, 0
    if state_path.exists():
        state = torch.load(state_path, map_location='cpu', weights_only=False)
        if state['signature'] != fit_signature:
            raise ValueError('Resume signature mismatch')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(state['scheduler'])
        history, best_state = state['history'], state['best_state']
        best_loss, best_epoch, stale = state['best_loss'], state['best_epoch'], state['stale']
        restore_rng(state['rng'], loader)
    limit = config['max_epochs'] if fixed_epochs is None else fixed_epochs
    for epoch in range(len(history) + 1, limit + 1):
        if validation is not None and stale >= config['patience']:
            break
        model.train()
        total_loss, learning_rate = 0., optimizer.param_groups[0]['lr']
        for features, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(logits_of(model, features.to(device)), targets.to(device))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
            optimizer.step()
            total_loss += loss.item() * len(targets)
        entry = dict(epoch=epoch, train_weighted_loss=total_loss / len(train), learning_rate=learning_rate)
        if validation is not None:
            scores, _ = predict(model, data, validation, config, device)
            scores = np.clip(scores, 1e-7, 1 - 1e-7)
            labels = data['labels'][validation]
            value = float(-(labels * np.log(scores) + (1 - labels) * np.log(1 - scores)).mean())
            entry['validation_unweighted_loss'] = value
            if value < best_loss - 1e-6:
                best_loss, best_epoch, stale = value, epoch, 0
                best_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
            else:
                stale += 1
        else:
            best_epoch = epoch
        history.append(entry)
        if scheduler is not None:
            scheduler.step()
        finished = epoch == limit or (validation is not None and stale >= config['patience'])
        if epoch % config['checkpoint_interval'] == 0 or finished or epoch == interrupt_after:
            if validation is None:
                best_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
            temporary = folder / 'state.pending'
            torch.save(dict(signature=fit_signature, model=model.state_dict(), optimizer=optimizer.state_dict(),
                            scheduler=None if scheduler is None else scheduler.state_dict(), history=history,
                            best_state=best_state, best_loss=best_loss, best_epoch=best_epoch, stale=stale,
                            rng=training_rng(loader)), temporary)
            temporary.replace(state_path)
        if epoch == interrupt_after:
            raise InterruptedError('Test interruption after saved epoch')
    model.load_state_dict(best_state)
    record = dict(signature=fit_signature, metadata=metadata, history=history, selected_epoch=best_epoch,
                  checkpoint_sha256=sha256(state_path), completed=True)
    if validation is not None:
        scores, log_probabilities = predict(model, data, validation, config, device)
        record.update(indices=validation.tolist(), labels=data['labels'][validation].tolist(),
                      scores=scores.tolist(), log_probabilities=log_probabilities.tolist())
    write_json(done_path, record)
    return model, record


def choose_candidate(candidates):
    ranked = []
    for order, candidate in enumerate(candidates):
        labels = np.asarray(candidate['labels'], dtype=int)
        scores = np.asarray(candidate['scores'])
        log_probabilities = np.asarray(candidate['log_probabilities'])
        indices = candidate['indices']
        if len(set(indices)) != len(indices) or log_probabilities.shape != (len(labels), 2):
            raise ValueError('Invalid inner coverage')
        values = original.metrics(labels, scores >= .5)
        cross_entropy = float(-log_probabilities[np.arange(len(labels)), labels].mean())
        rank = (values['accuracy'], values['balanced_accuracy'], -cross_entropy, -order)
        ranked.append((rank, candidate, dict(metrics=values, cross_entropy=cross_entropy)))
    _, selected, _ = max(ranked, key=lambda item: item[0])
    return selected, [dict(candidate=item[1]['config']['candidate'], **item[2]) for item in ranked]


def full_fingerprint(data, device):
    return dict(schema='a3-tuning-v1', recipes=recipes(), variants=list(MODELS), seeds=list(SEEDS),
                source={name: sha256(ROOT / name) for name in SOURCE_FILES},
                inputs={key: value['sha256'] for key, value in data['provenance'].items()},
                runtime=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                             pandas=pd.__version__, sklearn=sklearn.__version__, cuda=torch.version.cuda,
                             cudnn=torch.backends.cudnn.version(), threads=torch.get_num_threads(),
                             device_type=device.type,
                             gpu=[torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]),
                selection='pooled inner accuracy; balanced accuracy; logit CE; fixed candidate order',
                epoch_selection='unsmoothed unweighted clipped-score CE; improvement > 1e-6',
                policies=['fixed'], outer_subjects=[1, 2, 3, 4], budget=dict(inner=648, refit=36),
                checkpoint_interval=5, claims='exploratory; old subjects already observed')


def smoke(data, device, output):
    fingerprint = full_fingerprint(data, device)
    digest = signature(fingerprint)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / 'locked_protocol.json'
    if lock_path.exists() and read_json(lock_path) != fingerprint:
        raise ValueError('Smoke lock changed; use a new output')
    write_json(lock_path, fingerprint)
    train, test, inner = reference.splits(data, 1)
    _, fit_indices, validation = inner[0]
    reports = []
    for variant in MODELS:
        for candidate in (recipes()[0], recipes()[4], recipes()[5]):
            config = dict(candidate, max_epochs=1)
            _, record = fit(variant, data, fit_indices, validation, config, 1062, device,
                            output / variant / config['candidate'], digest)
            reports.append(record)
        model, _ = fit(variant, data, train, None, recipes()[0], 2040, device,
                       output / variant / 'refit', digest, fixed_epochs=1)
        scores, _ = predict(model, data, test, recipes()[0], device)
        if scores.shape != (len(test),):
            raise ValueError('Smoke test coverage mismatch')
    inventory = {path.relative_to(output).as_posix(): sha256(path)
                 for path in output.rglob('*') if path.is_file() and path.name in {'fit.json', 'state.pt'}}
    report = dict(status='SMOKE_PASS', full_fingerprint=digest, fits=12, artifacts=inventory,
                  inner_fits=len(reports), synthetic=False, code=fingerprint['source'])
    write_json(output / 'smoke.json', report)
    print(json.dumps(report), flush=True)
    return report


def tasks():
    return [(variant, seed, outer) for seed in SEEDS for outer in range(1, 5) for variant in MODELS]


def task_name(variant, seed, outer):
    return f'{variant}_seed{seed}_subject{outer}'


def run_fold(data, variant, seed, outer, device, output, digest):
    folder = Path(output) / 'folds' / task_name(variant, seed, outer)
    if (folder / 'fold.json').exists():
        saved = read_json(folder / 'fold.json')
        if saved['decision']['run_signature'] != digest or sha256(folder / 'model.pt') != saved['checkpoint_sha256']:
            raise ValueError('Completed outer signature/checkpoint mismatch')
        return
    train, test, inner = reference.splits(data, outer)
    candidates = []
    for config in recipes():
        candidate = dict(config=config, indices=[], labels=[], scores=[], log_probabilities=[], epochs=[])
        for person, fit_indices, validation in inner:
            _, record = fit(variant, data, fit_indices, validation, config,
                            seed + outer * 1000 + person * 10, device,
                            folder / config['candidate'] / f'inner{person}', digest)
            for key in ('indices', 'labels', 'scores', 'log_probabilities'):
                candidate[key].extend(record[key])
            candidate['epochs'].append(record['selected_epoch'])
        if set(candidate['indices']) != set(train):
            raise ValueError('Inner predictions do not cover outer-train')
        candidates.append(candidate)
        print(f"INNER_DONE {task_name(variant, seed, outer)} {config['candidate']}", flush=True)
    selected, ranking = choose_candidate(candidates)
    decision = dict(run_signature=digest, variant=variant, seed=seed, subject=outer,
                    selected_candidate=selected['config']['candidate'], ranking=ranking,
                    selected_epochs=selected['epochs'], refit_epochs=int(np.median(selected['epochs'])))
    decision_path = folder / 'selection.json'
    if decision_path.exists() and read_json(decision_path) != decision:
        raise ValueError('Selection changed during resume')
    write_json(decision_path, decision)
    model, refit = fit(variant, data, train, None, selected['config'], seed + outer * 1000 + 999,
                       device, folder / 'refit', digest, fixed_epochs=decision['refit_epochs'])
    scores, _ = predict(model, data, test, selected['config'], device)
    checkpoint = folder / 'model.pt'
    temporary = folder / 'model.pending'
    torch.save(dict(variant=variant, dropout=selected['config']['dropout'],
                    state={key: tensor.detach().cpu() for key, tensor in model.state_dict().items()},
                    decision=decision), temporary)
    temporary.replace(checkpoint)
    rows = [dict(variant=variant, seed=seed, subject=outer, sample_index=int(index),
                 video_id=str(data['video_ids'][index]), label=int(data['labels'][index]),
                 score=float(score), prediction=int(score >= .5), threshold=.5, policy='fixed')
            for index, score in zip(test, scores)]
    write_json(folder / 'fold.json', dict(decision=decision, predictions=rows,
               checkpoint_sha256=sha256(checkpoint), refit_sha256=refit['checkpoint_sha256']))
    print(f'OUTER_DONE {task_name(variant, seed, outer)}', flush=True)


def summarize(output, complete=False):
    output = Path(output)
    rows = []
    for path in sorted((output / 'folds').glob('*/fold.json')):
        fold = read_json(path)
        if sha256(path.parent / 'model.pt') != fold['checkpoint_sha256']:
            raise ValueError('Outer checkpoint corruption')
        rows.extend(fold['predictions'])
    progress = dict(status='RUNNING', completed_fits=len(list((output / 'folds').rglob('fit.json'))),
                    total_fits=684, completed_outer=len(list((output / 'folds').glob('*/fold.json'))),
                    total_outer=36, updated_unix=time.time())
    if rows:
        frame = pd.DataFrame(rows)
        if frame.duplicated(['variant', 'seed', 'sample_index']).any():
            raise ValueError('Duplicate OOF row')
        frame.to_csv(output / 'predictions_partial.csv', index=False)
        metrics = []
        for (variant, seed), group in frame.groupby(['variant', 'seed']):
            metrics.append(dict(variant=variant, seed=int(seed), samples=len(group),
                                complete=len(group) == 160,
                                **original.metrics(group.label, group.prediction)))
        pd.DataFrame(metrics).to_csv(output / 'metrics_partial.csv', index=False)
        if complete:
            if len(frame) != 1440 or len(metrics) != 9 or not all(item['complete'] for item in metrics):
                raise ValueError('Incomplete primary OOF inventory')
            for _, group in frame.groupby(['variant', 'seed']):
                if set(group.sample_index) != set(range(160)):
                    raise ValueError('OOF samples missing')
            frame.to_csv(output / 'predictions.csv', index=False)
            frame[frame.label != frame.prediction].to_csv(output / 'errors.csv', index=False)
            pooled = pd.DataFrame(metrics)
            pooled.to_csv(output / 'pooled_by_seed.csv', index=False)
            pooled.groupby('variant')[['accuracy', 'balanced_accuracy', 'sensitivity', 'specificity', 'f1']].agg(
                ['mean', 'std']).to_csv(output / 'seed_mean_std.csv')
            subject_rows = []
            for (variant, seed, subject), group in frame.groupby(['variant', 'seed', 'subject']):
                subject_rows.append(dict(variant=variant, seed=int(seed), subject=int(subject),
                                         **original.metrics(group.label, group.prediction)))
            pd.DataFrame(subject_rows).to_csv(output / 'by_subject.csv', index=False)
    if complete:
        if progress['completed_fits'] != 684 or progress['completed_outer'] != 36:
            raise ValueError('Incomplete fit budget')
        progress['status'] = 'FULL_COMPLETE'
        write_json(output / 'completion.json', progress)
    write_json(output / 'progress.json', progress)
    return progress


def full(data, device, output, smoke_evidence, workers):
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise ValueError('Full execution requires CUDA; CPU is limited to tests/smoke')
    fingerprint = full_fingerprint(data, device)
    evidence = read_json(Path(smoke_evidence) / 'smoke.json')
    if (evidence['status'] != 'SMOKE_PASS' or evidence['full_fingerprint'] != signature(fingerprint)
            or evidence['fits'] != 12 or evidence['synthetic'] or len(evidence['artifacts']) != 24):
        raise ValueError('Missing or incompatible real smoke')
    for relative, expected in evidence['artifacts'].items():
        if sha256(Path(smoke_evidence) / relative) != expected:
            raise ValueError('Smoke evidence corruption')
    if workers not in range(1, torch.cuda.device_count() + 1):
        raise ValueError('Worker count exceeds GPUs')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    lock = dict(fingerprint=fingerprint, smoke_sha256=sha256(Path(smoke_evidence) / 'smoke.json'),
                workers=workers)
    lock_path = output / 'locked_protocol.json'
    if lock_path.exists() and read_json(lock_path) != lock:
        raise ValueError('Full code/data/config/runtime lock mismatch')
    write_json(lock_path, lock)
    children, handles = [], []
    try:
        for worker in range(workers):
            handle = (output / f'worker{worker}.log').open('a', encoding='utf-8')
            handles.append(handle)
            command = [sys.executable, '-u', '-m', 'training.a3_tuning.runner', '--mode', 'worker',
                       '--input-root', str(Path(data['provenance']['pose']['path']).parent),
                       '--output', str(output), '--device', 'cuda', '--workers', str(workers),
                       '--worker-index', str(worker), '--smoke-evidence', str(smoke_evidence)]
            children.append(subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT))
        while any(child.poll() is None for child in children):
            if any(child.poll() not in (None, 0) for child in children):
                raise RuntimeError('A GPU worker failed; inspect worker logs; existing checkpoints retained')
            print(json.dumps(summarize(output)), flush=True)
            time.sleep(15)
        if any(child.returncode != 0 for child in children):
            raise RuntimeError('A GPU worker failed; inspect logs')
        print(json.dumps(summarize(output, complete=True)), flush=True)
    except BaseException:
        write_json(output / 'interrupted.json', dict(status='INTERRUPTED', time=time.time()))
        raise
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
                child.wait()
        for handle in handles:
            handle.close()


def main():
    parser = argparse.ArgumentParser(description='Six-recipe paired A3/TCN/GRU nested LOSO')
    parser.add_argument('--mode', choices=['smoke', 'full', 'worker'], default='smoke')
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke-evidence', type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--worker-index', type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(2)
    data = reference.load_verified(args.input_root)
    device = torch.device(args.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA unavailable')
        torch.cuda.set_device(args.worker_index if args.mode == 'worker' else 0)
        device = torch.device('cuda', torch.cuda.current_device())
    if args.mode == 'smoke':
        smoke(data, device, args.output)
    elif args.mode == 'full':
        full(data, device, args.output, args.smoke_evidence, args.workers)
    else:
        lock = read_json(args.output / 'locked_protocol.json')
        fingerprint = full_fingerprint(data, device)
        if lock['fingerprint'] != fingerprint or lock['workers'] != args.workers:
            raise ValueError('Worker lock mismatch')
        if sha256(args.smoke_evidence / 'smoke.json') != lock['smoke_sha256']:
            raise ValueError('Worker smoke evidence mismatch')
        for index, (variant, seed, outer) in enumerate(tasks()):
            if index % args.workers == args.worker_index:
                run_fold(data, variant, seed, outer, device, args.output, signature(lock))


if __name__ == '__main__':
    main()
