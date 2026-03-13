#!/usr/bin/env python3
import argparse
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch


EOS_TOKEN = "~"
OUTPUT_VOCAB = f"{EOS_TOKEN}0123456789"
VOCAB = OUTPUT_VOCAB + "+="
EPS = 1e-12
BASE_MODES = ("learned", "identity_fixed")
UPDATE_MODES = ("outer_diff", "skew")
Problem = Tuple[int, int, str, str]


def normalize_vector(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    return x / (x.norm() + eps)


def manual_rotation_delta(u: torch.Tensor, v: torch.Tensor, update_mode: str) -> torch.Tensor:
    if update_mode == "outer_diff":
        return torch.outer(v - u, u)
    if update_mode == "skew":
        return torch.outer(v, u) - torch.outer(u, v)
    raise ValueError(f"Unsupported update_mode={update_mode!r}; expected one of {UPDATE_MODES}")


def orthogonalize_newton_schulz(w: torch.Tensor, steps: int) -> torch.Tensor:
    out = w
    for _ in range(steps):
        out = 1.5 * out - 0.5 * (out @ out.transpose(-1, -2) @ out)
    return out


def maybe_orthogonalize(w: torch.Tensor, steps: int) -> torch.Tensor:
    if steps <= 0:
        return w
    return orthogonalize_newton_schulz(w, steps)


def random_rotation_perturbation(
    n: int,
    device: torch.device,
    scale: float,
    orthogonalize_steps: int,
    batch: int | None = None,
) -> torch.Tensor:
    shape = (batch, n, n) if batch is not None else (n, n)
    noise = torch.randn(*shape, device=device) * scale
    return orthogonalize_newton_schulz(torch.eye(n, device=device).expand(shape) + noise, orthogonalize_steps)


def init_orthogonal_from_identity_plus_random(
    n: int,
    device: torch.device,
    sigma: float,
    orthogonalize_steps: int,
    batch: int | None = None,
) -> torch.Tensor:
    shape = (batch, n, n) if batch is not None else (n, n)
    std = (1.0 / n) ** 0.5
    eye = torch.eye(n, device=device).expand(shape)
    noise = torch.randn(*shape, device=device) * std
    return orthogonalize_newton_schulz(eye + sigma * noise, orthogonalize_steps)


def random_problem(rng: random.Random, addend_digits: int) -> Problem:
    max_val = (10**addend_digits) - 1
    a = rng.randint(0, max_val)
    b = rng.randint(0, max_val)
    lhs = f"{a:0{addend_digits}d}+{b:0{addend_digits}d}="
    rhs = str(a + b)
    return a, b, lhs, rhs


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ManualRotationMatrixNetwork:
    def __init__(
        self,
        *,
        n: int,
        device: torch.device,
        base_mode: str = "learned",
        init_sigma: float = 0.0,
        orthogonalize_steps: int = 1,
    ):
        if base_mode not in BASE_MODES:
            raise ValueError(f"Unsupported base_mode={base_mode!r}; expected one of {BASE_MODES}")
        if n < len(VOCAB):
            raise ValueError(f"n must be >= {len(VOCAB)} to fit fixed one-hot heads, got {n}")

        self.n = n
        self.device = device
        self.base_mode = base_mode
        self.learn_base_mat = base_mode == "learned"
        self.init_sigma = init_sigma
        self.vocab = VOCAB
        self.output_vocab = OUTPUT_VOCAB
        self.vocab_size = len(self.vocab)
        self.output_vocab_size = len(self.output_vocab)
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos: Dict[int, str] = {i: ch for ch, i in self.stoi.items()}

        self.query = torch.zeros(n, device=device)
        self.query[0] = 1.0
        self.unembed = torch.eye(n, device=device)[: self.output_vocab_size]

        self.base_mat = (
            init_orthogonal_from_identity_plus_random(n, device, init_sigma, orthogonalize_steps)
            if self.learn_base_mat
            else torch.eye(n, device=device)
        )
        self.token_mats = init_orthogonal_from_identity_plus_random(
            n,
            device,
            init_sigma,
            orthogonalize_steps,
            batch=self.vocab_size,
        )

    def eye(self) -> torch.Tensor:
        return torch.eye(self.n, device=self.device)

    def base_matrix(self) -> torch.Tensor:
        return self.base_mat if self.learn_base_mat else self.eye()

    def encode(self, s: str) -> List[int]:
        return [self.stoi[ch] for ch in s]

    def prefix_state_ids(self, token_ids: Sequence[int]) -> torch.Tensor:
        v = self.query
        for tid in reversed(token_ids):
            v = self.token_mats[tid] @ v
        return self.base_matrix() @ v

    def predict_next_id(self, token_ids: Sequence[int]) -> int:
        state = self.prefix_state_ids(token_ids)
        return int(state[: self.output_vocab_size].argmax().item())

    def predict_next(self, prefix: str) -> str:
        return self.itos[self.predict_next_id(self.encode(prefix))]

    def state_dict(self) -> Dict[str, torch.Tensor | str | int]:
        out: Dict[str, torch.Tensor | str | int] = {
            "n": self.n,
            "vocab": self.vocab,
            "output_vocab": self.output_vocab,
            "base_mode": self.base_mode,
            "init_sigma": self.init_sigma,
            "token_mats": self.token_mats,
        }
        if self.learn_base_mat:
            out["base_mat"] = self.base_mat
        return out

    @classmethod
    def from_checkpoint(cls, path: str, device: torch.device) -> Tuple["ManualRotationMatrixNetwork", int | None]:
        try:
            ckpt = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location=device)
        n = int(ckpt["n"])
        base_mode = str(ckpt["base_mode"])
        init_sigma = float(ckpt.get("init_sigma", 0.0))
        model = cls(n=n, device=device, base_mode=base_mode, init_sigma=init_sigma)
        token_mats = ckpt["token_mats"].to(device)
        if token_mats.shape != (model.vocab_size, n, n):
            raise ValueError(f"token_mats shape mismatch: expected {(model.vocab_size, n, n)}, got {tuple(token_mats.shape)}")
        model.token_mats = token_mats
        if model.learn_base_mat:
            base_mat = ckpt.get("base_mat")
            if base_mat is None or tuple(base_mat.shape) != (n, n):
                raise ValueError("Learned-base checkpoint missing valid base_mat")
            model.base_mat = base_mat.to(device)
        addend_digits = ckpt.get("addend_digits")
        if addend_digits is not None:
            addend_digits = int(addend_digits)
        return model, addend_digits


def save_checkpoint(model: ManualRotationMatrixNetwork, save_path: str, addend_digits: int) -> None:
    path = Path(save_path)
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **model.state_dict(),
            "addend_digits": addend_digits,
        },
        path,
    )


def generate_until_eos(model: ManualRotationMatrixNetwork, lhs: str, max_len: int) -> Tuple[str, bool]:
    pred = ""
    for _ in range(max_len):
        next_ch = model.predict_next(lhs + pred)
        if next_ch == EOS_TOKEN:
            return pred, True
        pred += next_ch
    return pred, False


def evaluate(
    model: ManualRotationMatrixNetwork,
    eval_samples: int,
    seed: int,
    max_gen_len: int,
    addend_digits: int,
) -> Tuple[float, float, float]:
    rng = random.Random(seed)
    exact = 0
    tf_correct = 0
    tf_total = 0
    stopped = 0

    for _ in range(eval_samples):
        _, _, lhs, rhs = random_problem(rng, addend_digits)
        pred, did_stop = generate_until_eos(model, lhs, max_gen_len)
        stopped += int(did_stop)
        if did_stop and pred == rhs:
            exact += 1

        target_seq = rhs + EOS_TOKEN
        for i, ch in enumerate(target_seq):
            pred_id = model.predict_next_id(model.encode(lhs + target_seq[:i]))
            tf_correct += int(pred_id == model.stoi[ch])
            tf_total += 1

    return exact / max(eval_samples, 1), tf_correct / max(tf_total, 1), stopped / max(eval_samples, 1)


def show_samples(model: ManualRotationMatrixNetwork, seed: int, max_gen_len: int, addend_digits: int, count: int = 10) -> None:
    rng = random.Random(seed)
    for _ in range(count):
        a, b, lhs, rhs = random_problem(rng, addend_digits)
        pred, did_stop = generate_until_eos(model, lhs, max_gen_len)
        ok = "OK" if (did_stop and pred == rhs) else "XX"
        stop_txt = "eos" if did_stop else "max"
        print(f"{lhs}{rhs:>4s} | pred={pred:>4s} ({stop_txt}) [{ok}]   ({a}+{b})")


@torch.no_grad()
def apply_batch_update(
    model: ManualRotationMatrixNetwork,
    prefixes: Sequence[Sequence[int]],
    target_ids: Sequence[int],
    learning_rate: float,
    orthogonalize_steps: int,
    orthogonalize_every: int,
    count_power: float,
    iter_idx: int,
    context_length_power: float,
    update_mode: str,
) -> Tuple[float, float]:
    n = model.n
    base_delta = torch.zeros(n, n, device=model.device)
    base_count = 0
    token_delta = torch.zeros(model.vocab_size, n, n, device=model.device)
    token_count = torch.zeros(model.vocab_size, device=model.device)
    correct = 0
    total = 0
    mean_target_score = 0.0

    base = model.base_matrix()
    target_basis = model.unembed

    for prefix_ids, target_id in zip(prefixes, target_ids):
        state = model.prefix_state_ids(prefix_ids)
        correct += int(int(state[: model.output_vocab_size].argmax().item()) == target_id)
        total += 1
        mean_target_score += float(state[target_id].item())

        target_vec = target_basis[target_id]
        context_scale = float(len(prefix_ids) ** (-context_length_power)) if len(prefix_ids) > 0 else 1.0
        suffix_inputs: List[torch.Tensor] = [torch.empty(0, device=model.device) for _ in prefix_ids]
        suffix = model.query
        for idx in range(len(prefix_ids) - 1, -1, -1):
            suffix_inputs[idx] = suffix
            suffix = model.token_mats[prefix_ids[idx]] @ suffix

        base_input = suffix
        if model.learn_base_mat:
            u_base = normalize_vector(base @ base_input)
            v_base = target_vec
            base_delta.add_(context_scale * manual_rotation_delta(u_base, v_base, update_mode))
            base_count += context_scale

        left_target = base.transpose(-1, -2) @ target_vec
        for idx, tid in enumerate(prefix_ids):
            x_in = suffix_inputs[idx]
            u = normalize_vector(model.token_mats[tid] @ x_in)
            v = normalize_vector(left_target)
            weight = context_scale
            token_delta[tid].add_(weight * manual_rotation_delta(u, v, update_mode))
            token_count[tid] += weight
            left_target = model.token_mats[tid].transpose(-1, -2) @ left_target

    eye = model.eye()
    do_orth = orthogonalize_every > 0 and (iter_idx % orthogonalize_every == 0)
    if model.learn_base_mat and base_count > 0:
        scaled_delta = base_delta / (float(base_count) ** count_power)
        updated = (eye + learning_rate * scaled_delta) @ model.base_mat
        model.base_mat = maybe_orthogonalize(updated, orthogonalize_steps) if do_orth else updated

    active = token_count > 0
    if active.any():
        scaled_delta = token_delta[active] / token_count[active].pow(count_power).view(-1, 1, 1)
        updated = (eye.unsqueeze(0) + learning_rate * scaled_delta) @ model.token_mats[active]
        model.token_mats[active] = maybe_orthogonalize(updated, orthogonalize_steps) if do_orth else updated

    return mean_target_score / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def inject_token_rotation_noise(
    model: ManualRotationMatrixNetwork,
    noise_scale: float,
    orthogonalize_steps: int,
) -> None:
    if noise_scale <= 0.0:
        return
    perturb = random_rotation_perturbation(
        model.n,
        model.device,
        scale=noise_scale,
        orthogonalize_steps=orthogonalize_steps,
        batch=model.vocab_size,
    )
    model.token_mats = perturb @ model.token_mats


def scheduled_noise_scale(iter_idx: int, total_iters: int, noise_scale: float, noise_scale_final: float, noise_decay_iters: int) -> float:
    if noise_scale_final < 0.0:
        return noise_scale
    if noise_decay_iters <= 0:
        decay_iters = total_iters
    else:
        decay_iters = noise_decay_iters
    t = min(max(iter_idx - 1, 0), decay_iters) / max(decay_iters, 1)
    return (1.0 - t) * noise_scale + t * noise_scale_final


def train(
    *,
    model: ManualRotationMatrixNetwork,
    iters: int,
    batch_size: int,
    learning_rate: float,
    addend_digits: int,
    seed: int,
    log_every: int,
    eval_every: int,
    eval_samples: int,
    max_gen_len: int,
    orthogonalize_steps: int,
    orthogonalize_every: int,
    noise_every: int,
    noise_scale: float,
    noise_scale_final: float,
    noise_decay_iters: int,
    count_power: float,
    context_length_power: float,
    update_mode: str,
) -> ManualRotationMatrixNetwork:
    rng = random.Random(seed)

    for iter_idx in range(1, iters + 1):
        prefixes: List[List[int]] = []
        target_ids: List[int] = []

        for _ in range(batch_size):
            _, _, lhs, rhs = random_problem(rng, addend_digits)
            target_seq = rhs + EOS_TOKEN
            for i, ch in enumerate(target_seq):
                prefixes.append(model.encode(lhs + target_seq[:i]))
                target_ids.append(model.stoi[ch])

        mean_target_score, token_acc = apply_batch_update(
            model,
            prefixes=prefixes,
            target_ids=target_ids,
            learning_rate=learning_rate,
            orthogonalize_steps=orthogonalize_steps,
            orthogonalize_every=orthogonalize_every,
            count_power=count_power,
            iter_idx=iter_idx,
            context_length_power=context_length_power,
            update_mode=update_mode,
        )

        current_noise_scale = scheduled_noise_scale(
            iter_idx=iter_idx,
            total_iters=iters,
            noise_scale=noise_scale,
            noise_scale_final=noise_scale_final,
            noise_decay_iters=noise_decay_iters,
        )
        if noise_every > 0 and current_noise_scale > 0.0 and iter_idx % noise_every == 0:
            inject_token_rotation_noise(model, noise_scale=current_noise_scale, orthogonalize_steps=orthogonalize_steps)

        if iter_idx % log_every == 0 or iter_idx == 1:
            print(
                f"iter={iter_idx:5d} mean_target_score={mean_target_score:.4f} "
                f"token_acc={token_acc:.3f} noise_scale={current_noise_scale:.6f}"
            )

        if iter_idx % eval_every == 0 or iter_idx == iters:
            exact, tf_acc, stop_rate = evaluate(
                model,
                eval_samples=eval_samples,
                seed=seed + iter_idx,
                max_gen_len=max_gen_len,
                addend_digits=addend_digits,
            )
            print(f"  eval exact_match={exact:.3f} teacher_forced_token_acc={tf_acc:.3f} stop_rate={stop_rate:.3f}")

    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dense matrix network with manual rotation updates and fixed query/unembedding")
    p.add_argument("--n", type=int, default=16, help="Square matrix dimension; must be >= vocab size")
    p.add_argument("--iters", type=int, default=1500, help="Training iterations")
    p.add_argument("--batch-size", type=int, default=64, help="Problems per iteration")
    p.add_argument("--learning-rate", type=float, default=0.01, help="Manual rotation step size")
    p.add_argument("--base-mode", type=str, default="learned", choices=list(BASE_MODES), help="Base matrix behavior")
    p.add_argument("--addend-digits", type=int, default=3, help="Digits for each addend in a+b")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-samples", type=int, default=300)
    p.add_argument("--init-sigma", type=float, default=0.0, help="Initialize matrices as orthogonalize(I + sigma * random), with random having variance 1/n")
    p.add_argument("--orthogonalize-steps", type=int, default=1, help="Newton-Schulz orthogonalization steps after each update")
    p.add_argument("--orthogonalize-every", type=int, default=1, help="Apply Newton-Schulz re-orthogonalization every N iterations; 0 disables it")
    p.add_argument("--noise-every", type=int, default=10, help="Inject a small random orthogonal perturbation into token matrices every N iterations; 0 disables it")
    p.add_argument("--noise-scale", type=float, default=1e-3, help="Scale of random perturbation before Newton-Schulz re-orthogonalization")
    p.add_argument("--noise-scale-final", type=float, default=-1.0, help="Optional final noise scale for linear decay; negative keeps constant noise")
    p.add_argument("--noise-decay-iters", type=int, default=0, help="Iterations over which to linearly decay noise_scale to noise_scale_final; 0 means use full training length")
    p.add_argument("--count-power", type=float, default=0.5, help="Scale accumulated updates by count**count_power; 0=sum, 0.5=sum/sqrt(count), 1=mean")
    p.add_argument("--context-length-power", type=float, default=0.0, help="Scale each prediction context by len(prefix)**(-power); 1.0 gives each context a fixed total update budget")
    p.add_argument("--update-mode", type=str, default="outer_diff", choices=list(UPDATE_MODES), help="Local manual update rule: original outer-difference or skew-symmetric first-order rotation")
    p.add_argument("--load-path", type=str, default=None, help="Optional checkpoint to continue training from")
    p.add_argument("--save-path", type=str, default="checkpoints/matrix_network_manual_rotation.pt", help="Checkpoint path")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be > 0")
    if args.init_sigma < 0.0:
        raise ValueError("--init-sigma must be >= 0")
    if args.orthogonalize_steps < 0:
        raise ValueError("--orthogonalize-steps must be >= 0")
    if args.orthogonalize_every < 0:
        raise ValueError("--orthogonalize-every must be >= 0")
    if args.noise_every < 0:
        raise ValueError("--noise-every must be >= 0")
    if args.noise_scale < 0.0:
        raise ValueError("--noise-scale must be >= 0")
    if args.noise_scale_final < 0.0 and args.noise_scale_final != -1.0:
        raise ValueError("--noise-scale-final must be >= 0 or -1 for constant noise")
    if args.noise_decay_iters < 0:
        raise ValueError("--noise-decay-iters must be >= 0")
    if args.count_power < 0.0:
        raise ValueError("--count-power must be >= 0")
    if args.context_length_power < 0.0:
        raise ValueError("--context-length-power must be >= 0")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = pick_device(args.device)
    print(f"device={device}")
    print(f"iters={args.iters} learning_rate={args.learning_rate}")
    print(f"base_mode={args.base_mode} addend_digits={args.addend_digits}")
    print(
        f"init_sigma={args.init_sigma} init_random_var={1.0 / args.n:.6f} "
        f"orthogonalize_steps={args.orthogonalize_steps} orthogonalize_every={args.orthogonalize_every}"
    )
    print(
        f"noise_every={args.noise_every} noise_scale={args.noise_scale} "
        f"noise_scale_final={args.noise_scale_final} noise_decay_iters={args.noise_decay_iters}"
    )
    print(f"update_mode={args.update_mode}")
    print(f"count_power={args.count_power}")
    print(f"context_length_power={args.context_length_power}")

    if args.load_path is None:
        model = ManualRotationMatrixNetwork(
            n=args.n,
            device=device,
            base_mode=args.base_mode,
            init_sigma=args.init_sigma,
            orthogonalize_steps=max(1, args.orthogonalize_steps),
        )
        addend_digits = args.addend_digits
    else:
        model, loaded_addend_digits = ManualRotationMatrixNetwork.from_checkpoint(args.load_path, device)
        if model.n != args.n:
            print(f"loaded_n={model.n}; overriding --n={args.n}")
        if model.base_mode != args.base_mode:
            print(f"loaded_base_mode={model.base_mode}; overriding --base-mode={args.base_mode}")
        addend_digits = int(loaded_addend_digits or args.addend_digits)
        if loaded_addend_digits is not None and loaded_addend_digits != args.addend_digits:
            print(f"loaded_addend_digits={loaded_addend_digits}; overriding --addend-digits={args.addend_digits}")

    max_gen_len = addend_digits + 2
    print(f"max_gen_len={max_gen_len}")
    print(f"output_vocab={OUTPUT_VOCAB}")

    model = train(
        model=model,
        iters=args.iters,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        addend_digits=addend_digits,
        seed=args.seed,
        log_every=args.log_every,
        eval_every=args.eval_every,
        eval_samples=args.eval_samples,
        max_gen_len=max_gen_len,
        orthogonalize_steps=args.orthogonalize_steps,
        orthogonalize_every=args.orthogonalize_every,
        noise_every=args.noise_every,
        noise_scale=args.noise_scale,
        noise_scale_final=args.noise_scale_final,
        noise_decay_iters=args.noise_decay_iters,
        count_power=args.count_power,
        context_length_power=args.context_length_power,
        update_mode=args.update_mode,
    )

    save_checkpoint(model, args.save_path, addend_digits=addend_digits)
    print(f"saved_checkpoint={args.save_path}")
    print("\nSample predictions:")
    show_samples(model, seed=args.seed + 999, max_gen_len=max_gen_len, addend_digits=addend_digits)


if __name__ == "__main__":
    main()
