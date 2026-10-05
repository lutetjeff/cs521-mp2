// Record CUDA events on JAX's execution stream, with data dependencies that
// keep the convolution between them. Boundary copies are outside the interval.
#include <cuda_runtime_api.h>
#include <xla/ffi/api/ffi.h>

namespace ffi = xla::ffi;

static ffi::Error RecordEvent(cudaStream_t stream, int64_t event,
                              bool copy_before, ffi::RemainingArgs args,
                              ffi::RemainingRets rets) {
    if (args.size() != rets.size()) {
        return ffi::Error::InvalidArgument("Input/output count mismatch");
    }
    auto copy = [&]() -> ffi::Error {
        for (size_t i = 0; i < args.size(); ++i) {
            auto src = args.get<ffi::AnyBuffer>(i);
            auto dst = rets.get<ffi::AnyBuffer>(i);
            if (!src) return src.error();
            if (!dst) return dst.error();
            if (src->size_bytes() != (*dst)->size_bytes()) {
                return ffi::Error::InvalidArgument("Input/output size mismatch");
            }
            auto status = cudaMemcpyAsync((*dst)->untyped_data(),
                src->untyped_data(), src->size_bytes(), cudaMemcpyDeviceToDevice, stream);
            if (status != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(status));
        }
        return ffi::Error::Success();
    };
    if (copy_before) {
        auto error = copy();
        if (error.failure()) return error;
    }
    // XLA may capture this handler in a CUDA graph. Keep actual event nodes
    // rather than allowing capture to turn the records into graph dependencies.
    cudaStreamCaptureStatus capture;
    auto status = cudaStreamIsCapturing(stream, &capture);
    if (status != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(status));
    status = cudaEventRecordWithFlags(reinterpret_cast<cudaEvent_t>(event), stream,
        capture == cudaStreamCaptureStatusActive ? cudaEventRecordExternal : cudaEventRecordDefault);
    if (status != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(status));
    return copy_before ? ffi::Error::Success() : copy();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(myconv_record_event, RecordEvent,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("event")
        .Attr<bool>("copy_before")
        .RemainingArgs()
        .RemainingRets(),
    {ffi::Traits::kCmdBufferCompatible});
