"""Run: python gpu/profile_myconv.py --output-dir traces

Open the resulting JSON files at https://ui.perfetto.dev.
Requires CUDA-enabled PyTorch/JAX, with nvcc and ninja on PATH.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.cpp_extension import load

from myconv import ConvModel


def trace(name, run, output_dir):
    print(f"trace starting for {name}")
    # Compile and warm up before recording.
    for _ in range(3):
        run()
    torch.cuda.synchronize()
    print("warmups complete")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for i in range(3):
            with record_function(f"{name}/iteration_{i}"):
                run()
                torch.cuda.synchronize()

    path = output_dir / f"{name}.json"
    prof.export_chrome_trace(str(path))
    # Check that CUPTI captured GPU work, including kernels launched by JAX.
    events = json.loads(path.read_text())["traceEvents"]
    assert any(event.get("cat") == "kernel" for event in events), (
        f"No GPU kernels captured in {path}; check PyTorch's CUPTI support."
    )
    print(path)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("traces"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    N, C, H, W = 1, 8, 32, 32
    x = torch.randn(N, C, H, W, device="cuda")
    # Zero bias (ConvModel's default) matches the bias-free CUDA kernel.
    model = ConvModel(H, W, C, 8, 3, stride=1, padding=1).cuda().eval()

    trace("pytorch", lambda: model(x), args.output_dir)
    compiled = torch.compile(model, backend="inductor")
    trace("inductor", lambda: compiled(x), args.output_dir)

    # Let JAX share GPU memory with PyTorch instead of reserving most of it.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    from myconv_jax import conv2d_manual_jax

    x_jax, weight_jax, bias_jax = jax.device_put(
        tuple(t.detach().cpu().numpy() for t in (x, model.weight, model.bias)),
        jax.devices("gpu")[0],
    )
    conv_jax = jax.jit(conv2d_manual_jax, static_argnames=("stride", "padding"))
    trace(
        "jax",
        lambda: conv_jax(x_jax, weight_jax, bias_jax).block_until_ready(),
        args.output_dir,
    )

    conv_cuda = load(
        name="myconv",
        sources=[str(Path(__file__).with_name("myconv_kernel.cu"))],
    )
    trace("cuda", lambda: conv_cuda.conv_cuda(x, model.weight, 1, 1), args.output_dir)


if __name__ == "__main__":
    main()
