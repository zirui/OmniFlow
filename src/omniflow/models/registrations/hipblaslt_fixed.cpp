#include <torch/extension.h>

#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPGuard.h>
#include <c10/hip/HIPStream.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <string>
#include <unordered_map>
#include <vector>

namespace {

#define HIPBLASLT_CHECK(expr) \
    TORCH_CHECK((expr) == HIPBLAS_STATUS_SUCCESS, "hipBLASLt call failed: ", #expr)

struct CachedProblem {
    hipblasLtMatmulDesc_t desc;
    hipblasLtMatrixLayout_t mat_a;
    hipblasLtMatrixLayout_t mat_b;
    hipblasLtMatrixLayout_t mat_d;
    hipblasLtMatmulAlgo_t algo;
    const void* scale_a;
    const void* scale_b;
};

thread_local std::unordered_map<int64_t, CachedProblem> problem_cache;

hipDataType fp8_type(at::ScalarType type) {
    if (type == at::kFloat8_e4m3fn)
        return HIP_R_8F_E4M3;
    if (type == at::kFloat8_e5m2)
        return HIP_R_8F_E5M2;
    TORCH_CHECK(false, "only FP8 E4M3 and E5M2 inputs are supported");
}

at::Tensor fixed_scaled_mm(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& a_scale,
    const at::Tensor& b_scale,
    int64_t solution) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "inputs must be CUDA/HIP tensors");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.size(1) == b.size(0), "invalid GEMM shapes");
    TORCH_CHECK(a.is_contiguous(), "A must be row-major contiguous");
    TORCH_CHECK(b.stride(0) == 1 && b.stride(1) == b.size(0), "B must be column-major");
    TORCH_CHECK(a_scale.numel() == 1 && b_scale.numel() == 1, "scales must be scalar tensors");

    c10::hip::HIPGuard guard(a.get_device());
    const int64_t m = a.size(0), n = b.size(1), k = a.size(1);
    auto out = at::empty({m, n}, a.options().dtype(at::kBFloat16));
    auto handle = at::cuda::getCurrentCUDABlasLtHandle();
    const int64_t cache_key = solution | (m / 256 << 20) | (n / 256 << 28) |
        (k / 256 << 36) | (static_cast<int64_t>(a.get_device()) << 52) |
        (static_cast<int64_t>(a.scalar_type() == at::kFloat8_e5m2) << 56) |
        (static_cast<int64_t>(b.scalar_type() == at::kFloat8_e5m2) << 57);
    auto cached = problem_cache.find(cache_key);

    const void* scale_a = b_scale.data_ptr();
    const void* scale_b = a_scale.data_ptr();
    const float alpha = 1.0f, beta = 0.0f;
    if (cached == problem_cache.end()) {
        int version = 0;
        HIPBLASLT_CHECK(hipblasLtGetVersion(handle, &version));
        TORCH_CHECK(version == 100401, "fixed solutions require hipBLASLt 1.4.1, found ", version);
        const std::string arch = at::cuda::getCurrentDeviceProperties()->gcnArchName;
        TORCH_CHECK(arch.rfind("gfx950", 0) == 0, "fixed solutions require gfx950, found ", arch);

        CachedProblem problem;
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&problem.mat_a, fp8_type(b.scalar_type()), k, n, k));
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&problem.mat_b, fp8_type(a.scalar_type()), k, m, k));
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&problem.mat_d, HIP_R_16BF, n, m, n));
        HIPBLASLT_CHECK(hipblasLtMatmulDescCreate(&problem.desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
        const hipblasOperation_t trans_a = HIPBLAS_OP_T;
        const hipblasOperation_t trans_b = HIPBLAS_OP_N;
        HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(
            problem.desc, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
        HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(
            problem.desc, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));
        HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(
            problem.desc, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scale_a, sizeof(scale_a)));
        HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(
            problem.desc, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scale_b, sizeof(scale_b)));
        problem.scale_a = scale_a;
        problem.scale_b = scale_b;

        std::vector<int> indices{static_cast<int>(solution)};
        std::vector<hipblasLtMatmulHeuristicResult_t> results;
        HIPBLASLT_CHECK(hipblaslt_ext::getAlgosFromIndex(handle, indices, results));
        TORCH_CHECK(results.size() == 1, "hipBLASLt solution not found: ", solution);
        problem.algo = results[0].algo;
        size_t workspace_size = 0;
        HIPBLASLT_CHECK(hipblaslt_ext::matmulIsAlgoSupported(
            handle, problem.desc, &alpha, problem.mat_a, problem.mat_b, &beta,
            problem.mat_d, problem.mat_d, problem.algo, workspace_size));
        TORCH_CHECK(workspace_size <= at::cuda::getCUDABlasLtWorkspaceSize(),
                    "hipBLASLt solution needs too much workspace");
        cached = problem_cache.emplace(cache_key, problem).first;
    }

    auto& problem = cached->second;
    if (problem.scale_a != scale_a) {
        HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(
            problem.desc, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scale_a, sizeof(scale_a)));
        problem.scale_a = scale_a;
    }
    if (problem.scale_b != scale_b) {
        HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(
            problem.desc, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scale_b, sizeof(scale_b)));
        problem.scale_b = scale_b;
    }

    HIPBLASLT_CHECK(hipblasLtMatmul(
        handle, problem.desc, &alpha, b.data_ptr(), problem.mat_a, a.data_ptr(), problem.mat_b,
        &beta, out.data_ptr(), problem.mat_d, out.data_ptr(), problem.mat_d, &problem.algo,
        at::cuda::getCUDABlasLtWorkspace(), at::cuda::getCUDABlasLtWorkspaceSize(),
        c10::hip::getCurrentHIPStream()));
    return out;
}

at::Tensor fixed_scaled_mm_meta(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor&,
    const at::Tensor&,
    int64_t) {
    return at::empty({a.size(0), b.size(1)}, a.options().dtype(at::kBFloat16));
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(omniflow, module) {
    module.def("flux_hipblaslt_scaled_mm(Tensor a, Tensor b, Tensor a_scale, Tensor b_scale, int solution) -> Tensor");
}

TORCH_LIBRARY_IMPL(omniflow, CUDA, module) {
    module.impl("flux_hipblaslt_scaled_mm", fixed_scaled_mm);
}

TORCH_LIBRARY_IMPL(omniflow, Meta, module) {
    module.impl("flux_hipblaslt_scaled_mm", fixed_scaled_mm_meta);
}
