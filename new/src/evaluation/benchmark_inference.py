"""
بنچمارک زمان استنتاج (Inference Time) مدل PPO روی CPU
"""

import sys
from pathlib import Path
import time
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))

from stable_baselines3 import PPO
from configs.paths import get_paths

PATHS = get_paths()


def benchmark_inference(
    model_path: Path,
    n_warmup: int = 100,
    n_iterations: int = 10_000,
    use_real_env: bool = False
):
    """
    زمان استنتاج مدل را روی CPU اندازه‌گیری می‌کند.
    
    Args:
        model_path: مسیر مدل (مثلاً best_model یا final_model)
        n_warmup: تعداد اجرای گرم‌کردن (اولین اجراها کندترند)
        n_iterations: تعداد تکرار برای میانگین‌گیری
        use_real_env: اگر True باشد از محیط واقعی برای observation استفاده می‌شود
    """
    print("\n" + "=" * 60)
    print("⏱️  بنچمارک زمان استنتاج (Inference Time) روی CPU")
    print("=" * 60)
    
    # بارگذاری مدل روی CPU
    full_path = Path(model_path)
    if not full_path.is_absolute():
        full_path = PATHS['models'] / model_path
    if not full_path.exists():
        zip_path = full_path.with_suffix('.zip') if full_path.suffix != '.zip' else full_path
        if not zip_path.exists():
            zip_path = Path(str(full_path) + ".zip")
        if zip_path.exists():
            full_path = zip_path
        else:
            raise FileNotFoundError(f"مدل یافت نشد: {full_path}")
    
    print(f"📂 بارگذاری مدل: {full_path}")
    model = PPO.load(str(full_path), device='cpu')
    
    # ایجاد observation معتبر (۲۹ فیچر مطابق abr_multi_env)
    obs_dim = 29
    if use_real_env:
        from src.environment.abr_multi_env import ABREnv
        env = ABREnv(
            video_names='bigbuckbunny',
            trace_dir=str(PATHS['train_traces']),
            vmaf_dir=str(PATHS['vmaf_scores']),
            siti_dir=str(PATHS['content_features']),
            max_chunks=48
        )
        obs, _ = env.reset()
    else:
        # مشاهده‌ی شبیه‌سازی‌شده (محدودهٔ معمول)
        np.random.seed(42)
        obs = np.random.randn(obs_dim).astype(np.float32) * 0.3 + 0.5
        obs = np.clip(obs, 0, 1)
    
    # Warmup
    print(f"🔥 Warmup: {n_warmup} اجرا...")
    for _ in range(n_warmup):
        _ = model.predict(obs, deterministic=True)
    
    # بنچمارک اصلی
    print(f"⏱️  بنچمارک: {n_iterations:,} تکرار...")
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = model.predict(obs, deterministic=True)
    elapsed = time.perf_counter() - start
    
    # نتایج
    mean_ms = (elapsed / n_iterations) * 1000
    throughput = n_iterations / elapsed
    
    print("\n" + "-" * 60)
    print("📊 نتایج:")
    print(f"   • کل زمان:        {elapsed:.3f} ثانیه")
    print(f"   • میانگین هر پیش‌بینی:  {mean_ms:.4f} ms")
    print(f"   • Throughput:     {throughput:,.0f} inferences/ثانیه")
    print("-" * 60)
    print(f"\n✅ زمان استنتاج روی CPU: ~{mean_ms:.3f} ms per inference\n")
    
    return {
        'mean_ms': mean_ms,
        'throughput_per_sec': throughput,
        'total_sec': elapsed
    }


def compare_models(
    model_names: list,
    n_warmup: int = 100,
    n_iterations: int = 10_000
):
    """Compare inference time of multiple models on CPU. Prints results in English."""
    results = []
    np.random.seed(42)

    print("\n" + "=" * 70)
    print("       INFERENCE TIME COMPARISON (CPU)")
    print("=" * 70)

    for name in model_names:
        try:
            model_path = find_model_path(name)
            print(f"\n  Loading: {name}...")
            model = PPO.load(str(model_path), device='cpu')

            # Use each model's observation space (e.g. 29 for Proposed, 18 for Pensieve)
            obs_dim = model.observation_space.shape[0]
            obs = np.random.randn(obs_dim).astype(np.float32) * 0.3 + 0.5
            obs = np.clip(obs, 0, 1)

            # Warmup
            for _ in range(n_warmup):
                _ = model.predict(obs, deterministic=True)

            # Benchmark
            start = time.perf_counter()
            for _ in range(n_iterations):
                _ = model.predict(obs, deterministic=True)
            elapsed = time.perf_counter() - start

            mean_ms = (elapsed / n_iterations) * 1000
            throughput = n_iterations / elapsed
            results.append({
                'name': name,
                'mean_ms': mean_ms,
                'throughput': throughput,
                'total_sec': elapsed
            })
            print(f"    OK - {mean_ms:.4f} ms/inference")

        except FileNotFoundError as e:
            print(f"    SKIP - Model not found: {e}")
        except Exception as e:
            print(f"    ERROR - {e}")

    # Print comparison table
    if not results:
        print("\n  No models could be benchmarked.")
        return

    print("\n" + "-" * 70)
    print("  RESULTS")
    print("-" * 70)
    print(f"  {'Model':<40} {'Mean (ms)':<12} {'Throughput (inf/s)':<18}")
    print("-" * 70)
    for r in results:
        print(f"  {r['name']:<40} {r['mean_ms']:<12.4f} {r['throughput']:>,.0f}")
    print("-" * 70)

    if len(results) >= 2:
        fastest = min(results, key=lambda x: x['mean_ms'])
        slowest = max(results, key=lambda x: x['mean_ms'])
        speedup = slowest['mean_ms'] / fastest['mean_ms']
        print(f"\n  Fastest:  {fastest['name']} ({fastest['mean_ms']:.4f} ms)")
        print(f"  Slowest:  {slowest['name']} ({slowest['mean_ms']:.4f} ms)")
        print(f"  Speedup:  {speedup:.2f}x")
    print("=" * 70 + "\n")
    return results


def find_model_path(base_name: str) -> Path:
    """یافتن مسیر مدل: اول best_model، در غیر این صورت final_model"""
    base = PATHS['models'] / base_name
    best = base / 'best_model' / 'best_model'
    final = base / 'final_model'
    if best.with_suffix('.zip').exists():
        return best
    if (base / 'best_model').exists():
        return base / 'best_model' / 'best_model'
    if final.with_suffix('.zip').exists():
        return final
    if final.exists():
        return final
    raise FileNotFoundError(f"مدل در {base} یافت نشد")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Inference time benchmark on CPU')
    parser.add_argument('--model', type=str, default='ppo_abr_multi_dynamic_22',
                        help='Model folder name')
    parser.add_argument('--compare', action='store_true',
                        help='Compare Proposed vs Pensieve models')
    parser.add_argument('--iterations', type=int, default=10_000,
                        help='Benchmark iterations')
    parser.add_argument('--warmup', type=int, default=100,
                        help='Warmup runs')
    parser.add_argument('--real-env', action='store_true',
                        help='Use real environment observations')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_models(
            model_names=['ppo_abr_multi_dynamic_22', 'pensieve_multi_vmaf_new_14'],
            n_warmup=args.warmup,
            n_iterations=args.iterations
        )
    else:
        model_path = find_model_path(args.model)
        benchmark_inference(
            model_path=model_path,
            n_warmup=args.warmup,
            n_iterations=args.iterations,
            use_real_env=args.real_env
        )
