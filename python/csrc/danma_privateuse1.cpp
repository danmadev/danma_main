#include <ATen/EmptyTensor.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/core/Allocator.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/core/impl/alloc_cpu.h>
#include <torch/extension.h>

#include <atomic>
#include <cstring>

namespace at {
namespace detail {
C10_REGISTER_GUARD_IMPL(
    PrivateUse1,
    c10::impl::NoOpDeviceGuardImpl<DeviceType::PrivateUse1>);
} // namespace detail
} // namespace at

namespace {

std::atomic<uint64_t> allocation_count{0};
std::atomic<uint64_t> copy_count{0};

struct DanmaPrivateUse1Hooks final : at::PrivateUse1HooksInterface {
  bool isBuilt() const override {
    return true;
  }

  bool isAvailable() const override {
    return true;
  }

  bool hasPrimaryContext(at::DeviceIndex device_index) const override {
    return device_index == 0;
  }

  at::DeviceIndex deviceCount() const override {
    return 1;
  }

  void setCurrentDevice(at::DeviceIndex device) const override {
    TORCH_CHECK(device == 0, "DANMA exposes only device index 0");
  }

  at::DeviceIndex getCurrentDevice() const override {
    return 0;
  }

  at::DeviceIndex exchangeDevice(at::DeviceIndex device) const override {
    TORCH_CHECK(device == 0, "DANMA exposes only device index 0");
    return 0;
  }

  at::DeviceIndex maybeExchangeDevice(at::DeviceIndex device) const override {
    TORCH_CHECK(device == 0 || device == -1, "DANMA exposes only device index 0");
    return 0;
  }

  at::Device getDeviceFromPtr(void* data) const override {
    (void)data;
    return at::Device(at::DeviceType::PrivateUse1, 0);
  }
};

static bool register_hooks [[maybe_unused]] = []() {
  at::RegisterPrivateUse1HooksInterface(new DanmaPrivateUse1Hooks());
  return true;
}();

struct DanmaStagingAllocator final : at::Allocator {
  at::DataPtr allocate(size_t nbytes) override {
    allocation_count.fetch_add(1, std::memory_order_relaxed);
    void* data = c10::alloc_cpu(nbytes);
    return {
        data,
        data,
        &DanmaStagingAllocator::release,
        at::Device(at::DeviceType::PrivateUse1, 0)};
  }

  static void release(void* ptr) {
    if (ptr != nullptr) {
      c10::free_cpu(ptr);
    }
  }

  at::DeleterFnPtr raw_deleter() const override {
    return &DanmaStagingAllocator::release;
  }

  void copy_data(void* dest, const void* src, std::size_t count) const final {
    default_copy_data(dest, src, count);
  }
};

DanmaStagingAllocator global_allocator;
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &global_allocator);

void check_supported_tensor(const at::Tensor& tensor) {
  TORCH_CHECK(
      tensor.is_cpu() ||
          tensor.device().type() == c10::DeviceType::PrivateUse1,
      "DANMA PrivateUse1 supports only CPU <-> DANMA staging copies");
  TORCH_CHECK(
      tensor.scalar_type() == c10::ScalarType::Float,
      "DANMA PrivateUse1 MVP supports only float32 tensors");
  TORCH_CHECK(
      tensor.is_contiguous(),
      "DANMA PrivateUse1 MVP supports only contiguous tensors");
}

at::Tensor custom_empty_memory_format(
    at::IntArrayRef size,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory,
    std::optional<at::MemoryFormat> memory_format) {
  TORCH_CHECK(
      c10::layout_or_default(layout) == c10::Layout::Strided,
      "DANMA PrivateUse1 supports only strided tensors");
  TORCH_CHECK(
      !pin_memory.value_or(false),
      "DANMA PrivateUse1 does not support pinned device memory");
  const auto scalar_type = c10::dtype_or_default(dtype);
  TORCH_CHECK(
      scalar_type == c10::ScalarType::Float,
      "DANMA PrivateUse1 MVP supports only float32 tensors");
  constexpr c10::DispatchKeySet keys(c10::DispatchKey::PrivateUse1);
  return at::detail::empty_generic(
      size, &global_allocator, keys, scalar_type, memory_format);
}

at::Tensor custom_empty_strided(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory) {
  TORCH_CHECK(
      c10::layout_or_default(layout) == c10::Layout::Strided,
      "DANMA PrivateUse1 supports only strided tensors");
  TORCH_CHECK(
      !pin_memory.value_or(false),
      "DANMA PrivateUse1 does not support pinned device memory");
  const auto scalar_type = c10::dtype_or_default(dtype);
  TORCH_CHECK(
      scalar_type == c10::ScalarType::Float,
      "DANMA PrivateUse1 MVP supports only float32 tensors");
  constexpr c10::DispatchKeySet keys(c10::DispatchKey::PrivateUse1);
  return at::detail::empty_strided_generic(
      size, stride, &global_allocator, keys, scalar_type);
}

at::Tensor& custom_fill_scalar(at::Tensor& self, const at::Scalar& value) {
  check_supported_tensor(self);
  auto* data = static_cast<float*>(self.mutable_data_ptr());
  const float fill = value.toFloat();
  for (int64_t index = 0; index < self.numel(); ++index) {
    data[index] = fill;
  }
  return self;
}

at::Tensor custom_copy_from(
    const at::Tensor& self,
    const at::Tensor& dst,
    bool non_blocking) {
  (void)non_blocking;
  check_supported_tensor(self);
  check_supported_tensor(dst);
  TORCH_CHECK(self.sizes() == dst.sizes(), "DANMA copy shape mismatch");
  TORCH_CHECK(
      self.scalar_type() == dst.scalar_type(),
      "DANMA copy dtype mismatch");

  copy_count.fetch_add(1, std::memory_order_relaxed);
  std::memcpy(
      dst.storage().data_ptr().get(),
      self.storage().data_ptr().get(),
      self.storage().nbytes());
  return dst;
}

at::Tensor custom_to_device(
    const at::Tensor& self,
    at::Device device,
    at::ScalarType dtype,
    bool non_blocking,
    bool copy,
    std::optional<at::MemoryFormat> memory_format) {
  (void)non_blocking;
  (void)copy;
  check_supported_tensor(self);
  TORCH_CHECK(
      device.is_cpu() ||
          device.type() == c10::DeviceType::PrivateUse1,
      "DANMA PrivateUse1 can copy only between CPU and DANMA");
  TORCH_CHECK(
      dtype == c10::ScalarType::Float,
      "DANMA PrivateUse1 MVP supports only float32 tensors");

  auto out = at::empty(
      self.sizes(),
      dtype,
      self.options().layout(),
      device,
      false,
      memory_format);
  check_supported_tensor(out);
  copy_count.fetch_add(1, std::memory_order_relaxed);
  std::memcpy(out.mutable_data_ptr(), self.mutable_data_ptr(), self.nbytes());
  return out;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("empty.memory_format", &custom_empty_memory_format);
  m.impl("empty_strided", &custom_empty_strided);
  m.impl("fill_.Scalar", &custom_fill_scalar);
  m.impl("_copy_from", &custom_copy_from);
  m.impl("to.Device", &custom_to_device);
}

// Intentionally no catch-all PrivateUse1 fallback. Unsupported operations must
// fail instead of silently running an ATen CPU implementation.

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("device_count", []() { return 1; });
  m.def("current_device", []() { return 0; });
  m.def("is_available", []() { return true; });
  m.def(
      "allocation_count",
      []() { return allocation_count.load(std::memory_order_relaxed); });
  m.def(
      "copy_count",
      []() { return copy_count.load(std::memory_order_relaxed); });
}
