/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "xla/tsl/platform/default/dso_loader.h"

#include <stdlib.h>

#include <initializer_list>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "third_party/gpus/cuda/cuda_config.h"
#include "third_party/nccl/nccl_config.h"
#include "third_party/nvshmem/nvshmem_config.h"
#include "xla/tsl/platform/logging.h"
#include "tsl/platform/load_library.h"
#include "tsl/platform/path.h"
#include "tsl/platform/platform.h"
#include "third_party/tensorrt/tensorrt_config.h"

#if TENSORFLOW_USE_ROCM
#include "rocm/rocm_config.h"
#endif

namespace tsl {
namespace internal {

namespace {
absl::string_view GetCudaVersion() { return TF_CUDA_VERSION; }
absl::string_view GetCudaRtVersion() { return TF_CUDART_VERSION; }
absl::string_view GetCuptiVersion() { return TF_CUPTI_VERSION; }
absl::string_view GetCudnnVersion() { return TF_CUDNN_VERSION; }
absl::string_view GetCublasVersion() { return TF_CUBLAS_VERSION; }
absl::string_view GetCusolverVersion() { return TF_CUSOLVER_VERSION; }
absl::string_view GetCufftVersion() { return TF_CUFFT_VERSION; }
absl::string_view GetCusparseVersion() { return TF_CUSPARSE_VERSION; }
absl::string_view GetNcclVersion() { return TF_NCCL_VERSION; }
absl::string_view GetTensorRTVersion() { return TF_TENSORRT_VERSION; }
absl::string_view GetNvshmemVersion() { return XLA_NVSHMEM_VERSION; }

absl::StatusOr<void*> GetDsoHandle(const std::string& name,
                                   absl::string_view version) {
  auto filename =
      tsl::internal::FormatLibraryFileName(name, std::string(version));
  void* dso_handle;
  absl::Status status =
      tsl::internal::LoadDynamicLibrary(filename.c_str(), &dso_handle);
  if (status.ok()) {
    VLOG(1) << "Successfully opened dynamic library " << filename;
    return dso_handle;
  }

  auto message = absl::StrCat("Could not load dynamic library '", filename,
                              "'; dlerror: ", status.message());
#if !defined(PLATFORM_WINDOWS)
  if (const char* ld_library_path = getenv("LD_LIBRARY_PATH")) {
    absl::StrAppend(&message, "; LD_LIBRARY_PATH: ", ld_library_path);
  }
#endif
  VLOG(1) << message;
  return absl::Status(absl::StatusCode::kFailedPrecondition, message);
}
}  // namespace

#if defined(PLATFORM_WINDOWS)
absl::StatusOr<void*> GetWinCudaDsoHandle(
    const std::string& name, absl::string_view primary_version,
    std::initializer_list<absl::string_view> fallback_versions = {}) {
  absl::StatusOr<void*> primary_res = nullptr;
  bool tried_primary = false;
  if (!primary_version.empty()) {
    primary_res = GetDsoHandle(name + "64_", primary_version);
    if (primary_res.ok()) return primary_res;
    tried_primary = true;
  }
  for (auto ver : fallback_versions) {
    if (ver.empty() || ver == primary_version) continue;
    auto fb = GetDsoHandle(name + "64_", ver);
    if (fb.ok()) return fb;
    if (!tried_primary) {
      primary_res = fb;
      tried_primary = true;
    }
  }
  auto legacy = GetDsoHandle(name, primary_version);
  if (legacy.ok()) return legacy;
  return tried_primary ? primary_res : legacy;
}
#endif

namespace DsoLoader {
absl::StatusOr<void*> GetCudaDriverDsoHandle() {
#if defined(PLATFORM_WINDOWS)
  return GetDsoHandle("nvcuda", "");
#elif defined(__APPLE__)
  // On Mac OS X, CUDA sometimes installs libcuda.dylib instead of
  // libcuda.1.dylib.
  auto handle_or = GetDsoHandle("cuda", "");
  if (handle_or.ok()) {
    return handle_or;
  }
#endif
  return GetDsoHandle("cuda", "1");
}

absl::StatusOr<void*> GetNvmlDsoHandle() {
#if defined(PLATFORM_WINDOWS)
  return GetDsoHandle("nvml", "");
#else
  return GetDsoHandle("nvidia-ml", "1");
#endif
}

absl::StatusOr<void*> GetNvrtcDsoHandle() {
#if defined(PLATFORM_WINDOWS)
  // On Windows, NVRTC DLLs follow nvrtc64_<major>0_0.dll (e.g.,
  // nvrtc64_130_0.dll for CUDA 13.x, nvrtc64_120_0.dll for CUDA 12.x).
  absl::string_view cudart_version = GetCudaRtVersion();
  absl::string_view major = cudart_version.substr(0, cudart_version.find('.'));
  if (!major.empty()) {
    std::string nvrtc_version = absl::StrCat(major, "0_0");
    return GetWinCudaDsoHandle("nvrtc", nvrtc_version,
                               {cudart_version, "130_0", "120_0"});
  }
  return GetWinCudaDsoHandle("nvrtc", "130_0", {"120_0"});
#else
  return GetDsoHandle("nvrtc", GetCudaRtVersion());
#endif
}

absl::StatusOr<void*> GetCudaRuntimeDsoHandle() {
#if defined(PLATFORM_WINDOWS)
  return GetWinCudaDsoHandle("cudart", GetCudaRtVersion(), {"12"});
#else
  return GetDsoHandle("cudart", GetCudaRtVersion());
#endif
}

absl::StatusOr<void*> GetCublasDsoHandle() {
#if defined(PLATFORM_WINDOWS)
  return GetWinCudaDsoHandle("cublas", GetCublasVersion(), {"12"});
#else
  return GetDsoHandle("cublas", GetCublasVersion());
#endif
}

absl::StatusOr<void*> GetCublasLtDsoHandle() {
#if defined(PLATFORM_WINDOWS)
  return GetWinCudaDsoHandle("cublasLt", GetCublasVersion(), {"12"});
#else
  return GetDsoHandle("cublasLt", GetCublasVersion());
#endif
}

absl::StatusOr<void*> GetCufftDsoHandle() {
#if defined(PLATFORM_WINDOWS)
  return GetWinCudaDsoHandle("cufft", GetCufftVersion(), {"12", "11"});
#else
  return GetDsoHandle("cufft", GetCufftVersion());
#endif
}

absl::StatusOr<void*> GetCusolverDsoHandle() {
#if defined(PLATFORM_WINDOWS)
  return GetWinCudaDsoHandle("cusolver", GetCusolverVersion(), {"12", "11"});
#else
  return GetDsoHandle("cusolver", GetCusolverVersion());
#endif
}

absl::StatusOr<void*> GetCusparseDsoHandle() {
#if defined(PLATFORM_WINDOWS)
  return GetWinCudaDsoHandle("cusparse", GetCusparseVersion(), {"12", "11"});
#else
  return GetDsoHandle("cusparse", GetCusparseVersion());
#endif
}

absl::StatusOr<void*> GetCuptiDsoHandle() {
#if defined(PLATFORM_WINDOWS)
  // CUDA 13+ ships cupti64_<major>.dll (e.g., cupti64_13.dll); older Windows
  // toolkits used date-versioned names (cupti64_<YYYY.M.P>.dll) or cupti.dll.
  auto status_or_handle =
      GetWinCudaDsoHandle("cupti", GetCuptiVersion(), {"13", "12"});
  if (status_or_handle.ok()) return status_or_handle;
  return GetDsoHandle("cupti", "");
#else
  // Load specific version of CUPTI this is built.
  auto status_or_handle = GetDsoHandle("cupti", GetCuptiVersion());
  if (status_or_handle.ok()) return status_or_handle;
  // Load whatever libcupti.so user specified.
  return GetDsoHandle("cupti", "");
#endif
}

absl::StatusOr<void*> GetCudnnDsoHandle() {
#if defined(PLATFORM_WINDOWS)
  return GetWinCudaDsoHandle("cudnn", GetCudnnVersion(), {"9", "8"});
#else
  return GetDsoHandle("cudnn", GetCudnnVersion());
#endif
}

absl::StatusOr<void*> GetNcclDsoHandle() {
  return GetDsoHandle("nccl", GetNcclVersion());
}

absl::StatusOr<void*> GetNvshmemDsoHandle() {
  return GetDsoHandle("nvshmem_host", GetNvshmemVersion());
}

absl::StatusOr<void*> GetNvInferDsoHandle() {
#if defined(PLATFORM_WINDOWS)
  return GetDsoHandle("nvinfer", "");
#else
  return GetDsoHandle("nvinfer", GetTensorRTVersion());
#endif
}

absl::StatusOr<void*> GetNvInferPluginDsoHandle() {
#if defined(PLATFORM_WINDOWS)
  return GetDsoHandle("nvinfer_plugin", "");
#else
  return GetDsoHandle("nvinfer_plugin", GetTensorRTVersion());
#endif
}

}  // namespace DsoLoader

namespace CachedDsoLoader {
absl::StatusOr<void*> GetCudaDriverDsoHandle() {
  static auto result = new auto(DsoLoader::GetCudaDriverDsoHandle());
  return *result;
}

absl::StatusOr<void*> GetCudaRuntimeDsoHandle() {
  static auto result = new auto(DsoLoader::GetCudaRuntimeDsoHandle());
  return *result;
}

absl::StatusOr<void*> GetCublasDsoHandle() {
  static auto result = new auto(DsoLoader::GetCublasDsoHandle());
  return *result;
}

absl::StatusOr<void*> GetCublasLtDsoHandle() {
  static auto result = new auto(DsoLoader::GetCublasLtDsoHandle());
  return *result;
}

absl::StatusOr<void*> GetCufftDsoHandle() {
  static auto result = new auto(DsoLoader::GetCufftDsoHandle());
  return *result;
}

absl::StatusOr<void*> GetCusolverDsoHandle() {
  static auto result = new auto(DsoLoader::GetCusolverDsoHandle());
  return *result;
}

absl::StatusOr<void*> GetCusparseDsoHandle() {
  static auto result = new auto(DsoLoader::GetCusparseDsoHandle());
  return *result;
}

absl::StatusOr<void*> GetCuptiDsoHandle() {
  static auto result = new auto(DsoLoader::GetCuptiDsoHandle());
  return *result;
}

absl::StatusOr<void*> GetCudnnDsoHandle() {
  static auto result = new auto(DsoLoader::GetCudnnDsoHandle());
  return *result;
}

}  // namespace CachedDsoLoader
}  // namespace internal
}  // namespace tsl
