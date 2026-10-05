"""Benchmark the four existing myconv implementations on CUDA.

Run from the repository root (with nvcc and ninja on PATH):
    python gpu/benchmark_myconv.py --output-dir benchmarks
Quick check of all four backends:
    python gpu/benchmark_myconv.py --image-sizes 8 --filter-sizes 3 --repeats 3
Replot a saved CSV without rerunning kernels:
    python gpu/benchmark_myconv.py --output-dir benchmarks --plot-only

Default images: 16x16, 32x32, 64x64. Filters: 3x3, 7x7, 21x21.
Batch=1, Cin=Cout=8, float32, stride=1, padding=filter_size//2, zero bias.
Compilation, warmup, input transfers and correctness checks are not timed.
PyTorch/CUDA use events inside a replayed CUDA graph (no Python launch gaps).
JAX records events inside its compiled computation, on its own CUDA stream.
Times are GPU elapsed time for the entire forward, including all its kernels
and device-side gaps, not a sum of individual kernel durations.

Each case gets a fresh process and a configurable timeout. Failures are saved
in the CSV/logs and left missing in plots, never represented as zero runtime.
Requires CUDA PyTorch, CUDA JAX with jax.ffi, numpy, matplotlib, nvcc and ninja.
"""

import argparse
import csv
import ctypes
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys


BACKENDS = ("pytorch", "inductor", "jax", "cuda")
LABELS = ("PyTorch", "PyTorch Inductor", "JAX", "CUDA")


def cuda_extension(with_events=False):
    from torch.utils.cpp_extension import load

    sources = [str(Path(__file__).with_name("myconv_kernel.cu"))]
    includes = []
    if with_events:
        import jax.ffi

        sources.append(str(Path(__file__).with_name("benchmark_jax_events.cc")))
        includes.append(jax.ffi.include_dir())
    return load(
        name="myconv_benchmark_events" if with_events else "myconv_benchmark",
        sources=sources, extra_include_paths=includes,
    )


def time_torch(run, warmup, repeats):
    import torch

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            output = run()
    stream.synchronize()
    # External events become real event-record nodes in the captured graph.
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        start.record()
        output = run()
        end.record()
    samples = []
    for _ in range(repeats):
        graph.replay()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return output, samples


def time_jax(x, weight, bias, padding, warmup, repeats):
    import jax
    import jax.ffi
    import numpy as np
    import torch
    from myconv_jax import conv2d_manual_jax

    module = cuda_extension(with_events=True)
    library = ctypes.CDLL(module.__file__)
    jax.ffi.register_ffi_target(
        "myconv_record_event", jax.ffi.pycapsule(library.myconv_record_event),
        platform="CUDA",
    )
    jax.config.update("jax_default_matmul_precision", "highest")
    inputs = jax.device_put(
        tuple(t.cpu().numpy() for t in (x, weight, bias)), jax.devices("gpu")[0],
    )
    jax.block_until_ready(inputs)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # Initialize the event handles before passing them to the FFI handler.
    start.record()
    end.record()
    end.synchronize()

    def record(event, buffers, copy_before):
        specs = tuple(jax.ShapeDtypeStruct(b.shape, b.dtype) for b in buffers)
        return jax.ffi.ffi_call("myconv_record_event", specs, has_side_effect=True)(
            *buffers, event=np.int64(event.cuda_event), copy_before=copy_before,
        )

    @jax.jit
    def run(x, weight, bias):
        x, weight, bias = record(start, (x, weight, bias), True)
        output = conv2d_manual_jax(x, weight, bias, stride=1, padding=padding)
        return record(end, (output,), False)[0]

    for _ in range(warmup):
        run(*inputs).block_until_ready()
    samples = []
    for _ in range(repeats):
        output = run(*inputs).block_until_ready()
        samples.append(start.elapsed_time(end))
    return torch.from_numpy(np.array(output)).to(x.device), samples


def benchmark_case(backend, dim, kernel, warmup, repeats):
    # Set before JAX is imported/initialized in this fresh process.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import torch
    from myconv import ConvModel

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    padding = kernel // 2
    with torch.inference_mode():
        x = torch.randn(1, 8, dim, dim, device="cuda")
        model = ConvModel(dim, dim, 8, 8, kernel, stride=1, padding=padding).cuda().eval()
        # No explicit im2col buffer is needed by the fused CUDA implementation.
        if backend == "pytorch":
            needed = x.element_size() * 8 * dim**2 * kernel**2
            free, _ = torch.cuda.mem_get_info()
            if needed > free:
                raise torch.OutOfMemoryError(
                    f"im2col alone requires {needed / 2**30:.3f} GiB; "
                    f"only {free / 2**30:.3f} GiB is free"
                )
        if backend == "jax":
            output, samples = time_jax(
                x, model.weight, model.bias, padding, warmup, repeats,
            )
        else:
            if backend == "cuda":
                module = cuda_extension()
                run = lambda: module.conv_cuda(x, model.weight, 1, padding)
            else:
                if backend == "inductor":
                    model = torch.compile(model, backend="inductor", fullgraph=True)
                run = lambda: model(x)
            output, samples = time_torch(run, warmup, repeats)
        # Runnable correctness check on every measured shape, outside timing.
        reference = torch.nn.functional.conv2d(x, model.weight, model.bias, padding=padding)
        assert output.shape == (1, 8, dim, dim)
        torch.testing.assert_close(output, reference, rtol=1e-3, atol=1e-2)
        assert all(math.isfinite(t) and t > 0 for t in samples), samples
    return {"median_ms": statistics.median(samples), "samples_ms": samples,
            "gpu": torch.cuda.get_device_name()}


def plot_results(rows, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullLocator
    import numpy as np

    dims = sorted({int(r["image_dim"]) for r in rows})
    filters = sorted({int(r["filter_dim"]) for r in rows})
    values = {(r["backend"], int(r["image_dim"]), int(r["filter_dim"])):
              float(r["median_ms"]) if r["status"] == "ok" else math.nan for r in rows}
    finite = [math.log10(v) for v in values.values() if math.isfinite(v)]
    lo, hi = (min(finite), max(finite)) if finite else (-3, 0)
    margin = max((hi - lo) * 0.08, 0.1)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(1, len(filters), figsize=(5 * len(filters), 4),
                             sharey=True, squeeze=False, layout="constrained")
    for ax, kernel in zip(axes[0], filters):
        for backend, label, color in zip(BACKENDS, LABELS, colors):
            ax.plot(dims, [values.get((backend, d, kernel), math.nan) for d in dims],
                    "o-", label=label, color=color)
        ax.set(xscale="log", yscale="log", title=f"{kernel}×{kernel} filter",
               xlabel="Image dimension (H = W)", ylim=(10**(lo - margin), 10**(hi + margin)))
        ax.set_xticks(dims, [str(d) for d in dims])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.grid(True, alpha=0.25)
    axes[0, 0].set_ylabel("Median GPU runtime (ms)")
    axes[0, -1].legend()
    fig.suptitle("CUDA-event runtime · missing points indicate failed cases")
    fig.savefig(output_dir / "runtime_2d.png", dpi=160)
    plt.close(fig)

    # Four matched panels avoid hiding one backend's points behind another.
    # Explicit log coordinates work reliably with Matplotlib's 3D axes.
    fig = plt.figure(figsize=(12, 9), layout="constrained")
    for i, (backend, label, color) in enumerate(zip(BACKENDS, LABELS, colors), 1):
        ax = fig.add_subplot(2, 2, i, projection="3d")
        for j, kernel in enumerate(filters):
            z = [math.log10(v) if math.isfinite(v) else math.nan
                 for d in dims for v in [values.get((backend, d, kernel), math.nan)]]
            ax.plot(np.log2(dims), [j] * len(dims), z, "o-", color=color)
        ax.set(title=label, xlabel="Image dim (log₂)", ylabel="Filter dim",
               zlabel="Runtime (ms, log₁₀)", zlim=(lo - margin, hi + margin))
        ax.set_xticks(np.log2(dims), [str(d) for d in dims])
        ax.set_yticks(range(len(filters)), [str(k) for k in filters])
        ax.zaxis.set_major_formatter(FuncFormatter(lambda z, _: f"{10**z:.3g}"))
        ax.view_init(elev=25, azim=-55)
    fig.suptitle("Batch 1 · 8 input/output channels · same padding\n"
                 f"{len(dims) * len(filters)} configurations per backend; failed cases are omitted")
    fig.savefig(output_dir / "runtime_3d.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks"))
    parser.add_argument("--image-sizes", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--filter-sizes", type=int, nargs="+", default=[3, 7, 21])
    parser.add_argument("--backends", choices=BACKENDS, nargs="+", default=list(BACKENDS))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=300,
                        help="Seconds per case, including compilation; 0 disables the limit")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--worker", nargs=3, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 1 or args.timeout < 0:
        parser.error("warmup/repeats must be positive and timeout nonnegative")
    if any(d < 1 for d in args.image_sizes) or any(k < 1 or k % 2 == 0 for k in args.filter_sizes):
        parser.error("image sizes must be positive and filter sizes positive and odd")
    if args.worker:
        backend, dim, kernel = args.worker
        print(json.dumps(benchmark_case(backend, int(dim), int(kernel), args.warmup, args.repeats)))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "runtimes.csv"
    if args.plot_only:
        with csv_path.open(newline="") as f:
            plot_results(list(csv.DictReader(f)), args.output_dir)
        return

    rows = []
    fields = ["backend", "image_dim", "filter_dim", "padding", "median_ms",
              "status", "samples_ms", "gpu", "detail"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for kernel in args.filter_sizes:
            for dim in args.image_sizes:
                for backend in args.backends:
                    row = dict.fromkeys(fields, "")
                    row.update(backend=backend, image_dim=dim, filter_dim=kernel,
                               padding=kernel // 2, status="ok")
                    print(f"{backend:9s} image={dim:4d} filter={kernel:2d}: ", end="", flush=True)
                    command = [sys.executable, str(Path(__file__).resolve()),
                               "--worker", backend, str(dim), str(kernel),
                               "--warmup", str(args.warmup), "--repeats", str(args.repeats)]
                    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                          text=True, start_new_session=True) as process:
                        try:
                            stdout, stderr = process.communicate(timeout=args.timeout or None)
                        except subprocess.TimeoutExpired:
                            # Inductor/nvcc may have compiler children; stop the whole case.
                            os.killpg(process.pid, signal.SIGKILL)
                            stdout, stderr = process.communicate()
                            row.update(status="timeout", detail=f"Exceeded {args.timeout:g} seconds")
                        except BaseException:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                            raise
                        log = stdout + stderr
                        if row["status"] == "ok":
                            if process.returncode:
                                row.update(status="error", detail=log.strip().splitlines()[-1]
                                           if log.strip() else f"exit code {process.returncode}")
                            else:
                                row.update(json.loads(stdout.strip().splitlines()[-1]))
                                row["samples_ms"] = json.dumps(row["samples_ms"])
                    (args.output_dir / f"{backend}_{dim}_{kernel}.log").write_text(log)
                    writer.writerow(row)
                    f.flush()
                    rows.append(row)
                    print(f"{row['median_ms']:.6f} ms" if row["status"] == "ok"
                          else f"{row['status']}: {row['detail']}")
    plot_results(rows, args.output_dir)
    failed = sum(row["status"] != "ok" for row in rows)
    if failed:
        print(f"{failed}/{len(rows)} cases failed or timed out; see CSV and per-case logs.")
    print(f"Saved {csv_path}, runtime_2d.png and runtime_3d.png")


if __name__ == "__main__":
    main()
