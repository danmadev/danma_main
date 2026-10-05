#include <ATen/EmptyTensor.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/core/Allocator.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <torch/extension.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace {

constexpr char kMagic[4] = {'D', 'N', 'T', '1'};
constexpr uint8_t kAlloc = 1;
constexpr uint8_t kFree = 2;
constexpr uint8_t kUpload = 3;
constexpr uint8_t kDownload = 4;
constexpr uint8_t kAdd = 5;
constexpr uint8_t kMm = 6;
constexpr uint8_t kFill = 7;
constexpr uint8_t kCopy = 8;
constexpr uint8_t kSumRows = 9;
constexpr uint8_t kStats = 10;
constexpr size_t kMaxFrameBytes = 256 * 1024;

std::mutex endpoint_mutex;
uint16_t current_port = 0;
std::atomic<uint64_t> next_tensor_id{1};
std::atomic<uint64_t> allocation_count{0};
std::atomic<uint64_t> copy_count{0};

struct RemoteHandle {
  uint64_t id;
  size_t nbytes;
  uint16_t port;
};

void append_u64(std::vector<uint8_t>& out, uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8) {
    out.push_back(static_cast<uint8_t>((value >> shift) & 0xff));
  }
}

uint64_t read_u64(const std::vector<uint8_t>& input, size_t& offset) {
  TORCH_CHECK(offset + 8 <= input.size(), "DANMA tensor reply is truncated");
  uint64_t value = 0;
  for (int i = 0; i < 8; ++i) {
    value = (value << 8) | input[offset++];
  }
  return value;
}

void append_f32(std::vector<uint8_t>& out, float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  out.push_back(static_cast<uint8_t>(bits & 0xff));
  out.push_back(static_cast<uint8_t>((bits >> 8) & 0xff));
  out.push_back(static_cast<uint8_t>((bits >> 16) & 0xff));
  out.push_back(static_cast<uint8_t>((bits >> 24) & 0xff));
}

float read_f32(const std::vector<uint8_t>& input, size_t& offset) {
  TORCH_CHECK(offset + 4 <= input.size(), "DANMA tensor reply is truncated");
  uint32_t bits =
      static_cast<uint32_t>(input[offset]) |
      (static_cast<uint32_t>(input[offset + 1]) << 8) |
      (static_cast<uint32_t>(input[offset + 2]) << 16) |
      (static_cast<uint32_t>(input[offset + 3]) << 24);
  offset += 4;
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

void send_all(int fd, const uint8_t* data, size_t length) {
  size_t sent = 0;
  while (sent < length) {
    const auto n = ::send(fd, data + sent, length - sent, MSG_NOSIGNAL);
    TORCH_CHECK(n > 0, "DANMA tensor transport send failed: ", std::strerror(errno));
    sent += static_cast<size_t>(n);
  }
}

void recv_all(int fd, uint8_t* data, size_t length) {
  size_t received = 0;
  while (received < length) {
    const auto n = ::recv(fd, data + received, length - received, 0);
    TORCH_CHECK(n > 0, "DANMA tensor transport receive failed");
    received += static_cast<size_t>(n);
  }
}

std::vector<uint8_t> remote_request(
    uint16_t port,
    uint8_t opcode,
    const std::vector<uint8_t>& body) {
  TORCH_CHECK(port != 0, "DANMA remote tensor endpoint is not configured");

  std::vector<uint8_t> payload;
  payload.reserve(5 + body.size());
  payload.insert(payload.end(), kMagic, kMagic + 4);
  payload.push_back(opcode);
  payload.insert(payload.end(), body.begin(), body.end());
  TORCH_CHECK(
      payload.size() <= kMaxFrameBytes,
      "DANMA tensor request exceeds protocol frame limit");

  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  TORCH_CHECK(fd >= 0, "cannot create DANMA tensor socket");

  timeval tv{};
  tv.tv_sec = 4;
  tv.tv_usec = 0;
  ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(port);
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

  if (::connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
    const std::string message = std::strerror(errno);
    ::close(fd);
    TORCH_CHECK(false, "DANMA tensor transport connect failed: ", message);
  }

  const uint32_t length = htonl(static_cast<uint32_t>(payload.size()));
  try {
    send_all(fd, reinterpret_cast<const uint8_t*>(&length), sizeof(length));
    send_all(fd, payload.data(), payload.size());

    uint32_t reply_length_network = 0;
    recv_all(
        fd,
        reinterpret_cast<uint8_t*>(&reply_length_network),
        sizeof(reply_length_network));
    const size_t reply_length = ntohl(reply_length_network);
    TORCH_CHECK(
        reply_length > 0 && reply_length <= kMaxFrameBytes,
        "invalid DANMA tensor response size");
    std::vector<uint8_t> reply(reply_length);
    recv_all(fd, reply.data(), reply.size());
    ::close(fd);

    TORCH_CHECK(!reply.empty(), "empty DANMA tensor response");
    if (reply[0] != 0) {
      const std::string code(reply.begin() + 1, reply.end());
      TORCH_CHECK(false, "DANMA tensor runtime rejected request: ", code);
    }
    return reply;
  } catch (...) {
    ::close(fd);
    throw;
  }
}

uint16_t configured_port() {
  std::lock_guard<std::mutex> guard(endpoint_mutex);
  TORCH_CHECK(current_port != 0, "call enable_privateuse1(host, port) before creating DANMA tensors");
  return current_port;
}

void configure_endpoint(const std::string& host, int64_t port) {
  TORCH_CHECK(
      host == "127.0.0.1" || host == "localhost",
      "DANMA tensor v1 only supports loopback");
  TORCH_CHECK(port >= 1 && port <= 65535, "DANMA tensor port must be 1..65535");
  std::lock_guard<std::mutex> guard(endpoint_mutex);
  current_port = static_cast<uint16_t>(port);
}

void remote_alloc(RemoteHandle* handle) {
  TORCH_CHECK(handle->nbytes % sizeof(float) == 0, "DANMA only supports float32 storage");
  std::vector<uint8_t> body;
  append_u64(body, handle->id);
  append_u64(body, handle->nbytes / sizeof(float));
  remote_request(handle->port, kAlloc, body);
}

void remote_free(RemoteHandle* handle) {
  std::vector<uint8_t> body;
  append_u64(body, handle->id);
  remote_request(handle->port, kFree, body);
}

void remote_upload(RemoteHandle* handle, const float* data, size_t numel) {
  TORCH_CHECK(numel * sizeof(float) == handle->nbytes, "DANMA upload size mismatch");
  std::vector<uint8_t> body;
  body.reserve(16 + numel * sizeof(float));
  append_u64(body, handle->id);
  append_u64(body, numel);
  for (size_t i = 0; i < numel; ++i) {
    TORCH_CHECK(std::isfinite(data[i]), "DANMA upload requires finite float32 values");
    append_f32(body, data[i]);
  }
  remote_request(handle->port, kUpload, body);
}

void remote_download(RemoteHandle* handle, float* data, size_t numel) {
  std::vector<uint8_t> body;
  append_u64(body, handle->id);
  auto reply = remote_request(handle->port, kDownload, body);
  size_t offset = 1;
  const auto returned = read_u64(reply, offset);
  TORCH_CHECK(returned == numel, "DANMA download size mismatch");
  for (size_t i = 0; i < numel; ++i) {
    data[i] = read_f32(reply, offset);
  }
  TORCH_CHECK(offset == reply.size(), "DANMA download has trailing bytes");
}

void remote_fill(RemoteHandle* handle, float value) {
  TORCH_CHECK(std::isfinite(value), "DANMA fill requires finite scalar");
  std::vector<uint8_t> body;
  append_u64(body, handle->id);
  append_f32(body, value);
  remote_request(handle->port, kFill, body);
}

void remote_copy(RemoteHandle* src, RemoteHandle* dst) {
  TORCH_CHECK(src->port == dst->port, "DANMA tensors belong to different nodes");
  TORCH_CHECK(src->nbytes == dst->nbytes, "DANMA copy size mismatch");
  std::vector<uint8_t> body;
  append_u64(body, src->id);
  append_u64(body, dst->id);
  remote_request(src->port, kCopy, body);
}

RemoteHandle* remote_handle(const at::Tensor& tensor) {
  TORCH_CHECK(
      tensor.device().type() == c10::DeviceType::PrivateUse1,
      "expected DANMA tensor");
  TORCH_CHECK(tensor.storage_offset() == 0, "DANMA MVP does not support storage offsets");
  auto* handle = static_cast<RemoteHandle*>(tensor.storage().data_ptr().get());
  TORCH_CHECK(handle != nullptr, "DANMA tensor has no remote handle");
  return handle;
}

void ensure_remote_pair(const at::Tensor& lhs, const at::Tensor& rhs) {
  TORCH_CHECK(lhs.device().type() == c10::DeviceType::PrivateUse1, "lhs must be DANMA");
  TORCH_CHECK(rhs.device().type() == c10::DeviceType::PrivateUse1, "rhs must be DANMA");
  TORCH_CHECK(lhs.scalar_type() == at::kFloat && rhs.scalar_type() == at::kFloat, "DANMA supports float32 only");
  TORCH_CHECK(lhs.is_contiguous() && rhs.is_contiguous(), "DANMA supports contiguous tensors only");
  TORCH_CHECK(remote_handle(lhs)->port == remote_handle(rhs)->port, "DANMA tensors belong to different nodes");
}

void remote_add_into(
    const at::Tensor& lhs,
    const at::Tensor& rhs,
    at::Tensor& out,
    float alpha) {
  ensure_remote_pair(lhs, rhs);
  auto* lh = remote_handle(lhs);
  auto* rh = remote_handle(rhs);
  auto* oh = remote_handle(out);
  TORCH_CHECK(lh->port == oh->port, "DANMA output belongs to another node");
  TORCH_CHECK(lhs.numel() >= 0 && rhs.numel() > 0, "invalid DANMA add sizes");
  std::vector<uint8_t> body;
  append_u64(body, lh->id);
  append_u64(body, rh->id);
  append_u64(body, oh->id);
  append_u64(body, static_cast<uint64_t>(lhs.numel()));
  append_u64(body, static_cast<uint64_t>(rhs.numel()));
  append_f32(body, alpha);
  remote_request(lh->port, kAdd, body);
}

at::Tensor remote_add(
    const at::Tensor& lhs,
    const at::Tensor& rhs,
    float alpha) {
  ensure_remote_pair(lhs, rhs);
  TORCH_CHECK(
      lhs.sizes() == rhs.sizes() ||
          (rhs.dim() == 1 && lhs.dim() == 2 && rhs.size(0) == lhs.size(1)),
      "DANMA add supports equal shapes or 2D + 1D bias broadcast");
  auto out = at::empty(lhs.sizes(), lhs.options());
  remote_add_into(lhs, rhs, out, alpha);
  return out;
}

at::Tensor remote_mm(
    const at::Tensor& lhs,
    const at::Tensor& rhs,
    bool lhs_transposed,
    bool rhs_transposed) {
  ensure_remote_pair(lhs, rhs);
  TORCH_CHECK(lhs.dim() == 2 && rhs.dim() == 2, "DANMA mm requires rank-2 tensors");

  const int64_t lhs_rows = lhs_transposed ? lhs.size(1) : lhs.size(0);
  const int64_t lhs_cols = lhs_transposed ? lhs.size(0) : lhs.size(1);
  const int64_t rhs_rows = rhs_transposed ? rhs.size(1) : rhs.size(0);
  const int64_t rhs_cols = rhs_transposed ? rhs.size(0) : rhs.size(1);
  TORCH_CHECK(lhs_cols == rhs_rows, "DANMA mm dimension mismatch");

  auto out = at::empty({lhs_rows, rhs_cols}, lhs.options());
  auto* lh = remote_handle(lhs);
  auto* rh = remote_handle(rhs);
  auto* oh = remote_handle(out);
  std::vector<uint8_t> body;
  for (auto id : {lh->id, rh->id, oh->id}) {
    append_u64(body, id);
  }
  for (auto dim : {lhs_rows, lhs_cols, rhs_cols}) {
    append_u64(body, static_cast<uint64_t>(dim));
  }
  body.push_back(lhs_transposed ? 1 : 0);
  body.push_back(rhs_transposed ? 1 : 0);
  remote_request(lh->port, kMm, body);
  return out;
}

at::Tensor remote_sum_rows(const at::Tensor& input) {
  TORCH_CHECK(
      input.device().type() == c10::DeviceType::PrivateUse1 &&
          input.scalar_type() == at::kFloat && input.dim() == 2 &&
          input.is_contiguous(),
      "DANMA row sum requires contiguous rank-2 float32 tensor");
  auto out = at::empty({input.size(1)}, input.options());
  auto* ih = remote_handle(input);
  auto* oh = remote_handle(out);
  std::vector<uint8_t> body;
  append_u64(body, ih->id);
  append_u64(body, oh->id);
  append_u64(body, static_cast<uint64_t>(input.size(0)));
  append_u64(body, static_cast<uint64_t>(input.size(1)));
  remote_request(ih->port, kSumRows, body);
  return out;
}

struct DanmaDeviceGuard final : c10::impl::DeviceGuardImplInterface {
  c10::DeviceType type() const override { return c10::DeviceType::PrivateUse1; }
  c10::Device exchangeDevice(c10::Device device) const override {
    TORCH_CHECK(
        device.type() == c10::DeviceType::PrivateUse1 &&
            (device.index() == 0 || device.index() == -1),
        "DANMA exposes only device index 0");
    return c10::Device(c10::DeviceType::PrivateUse1, 0);
  }
  c10::Device getDevice() const override {
    return c10::Device(c10::DeviceType::PrivateUse1, 0);
  }
  void setDevice(c10::Device device) const override {
    TORCH_CHECK(
        device.type() == c10::DeviceType::PrivateUse1 &&
            (device.index() == 0 || device.index() == -1),
        "DANMA exposes only device index 0");
  }
  void uncheckedSetDevice(c10::Device) const noexcept override {}
  c10::Stream getStream(c10::Device) const noexcept override {
    return c10::Stream(
        c10::Stream::DEFAULT,
        c10::Device(c10::DeviceType::PrivateUse1, 0));
  }
  c10::Stream getDefaultStream(c10::Device) const override { return getStream(getDevice()); }
  c10::Stream getNewStream(c10::Device, int priority = 0) const override {
    (void)priority;
    return getStream(getDevice());
  }
  c10::Stream exchangeStream(c10::Stream) const noexcept override { return getStream(getDevice()); }
  c10::DeviceIndex deviceCount() const noexcept override { return 1; }
  void record(void** event, const c10::Stream&, const c10::DeviceIndex, const c10::EventFlag) const override {
    if (*event == nullptr) *event = new bool(true);
    else *static_cast<bool*>(*event) = true;
  }
  void block(void*, const c10::Stream&) const override {}
  bool queryEvent(void* event) const override { return event == nullptr || *static_cast<bool*>(event); }
  void destroyEvent(void* event, const c10::DeviceIndex) const noexcept override {
    delete static_cast<bool*>(event);
  }
  bool queryStream(const c10::Stream&) const override { return true; }
  void synchronizeStream(const c10::Stream&) const override {}
  void synchronizeEvent(void*) const override {}
  void synchronizeDevice(const c10::DeviceIndex) const override {}
  double elapsedTime(void*, void*, const c10::DeviceIndex) const override { return 0.0; }
};

} // namespace

C10_REGISTER_GUARD_IMPL(PrivateUse1, DanmaDeviceGuard);

namespace {

struct DanmaPrivateUse1Hooks final : at::PrivateUse1HooksInterface {
  bool isBuilt() const override { return true; }
  bool isAvailable() const override { return true; }
  bool hasPrimaryContext(at::DeviceIndex device_index) const override { return device_index == 0; }
  at::DeviceIndex deviceCount() const override { return 1; }
  void setCurrentDevice(at::DeviceIndex device) const override {
    TORCH_CHECK(device == 0, "DANMA exposes only device index 0");
  }
  at::DeviceIndex getCurrentDevice() const override { return 0; }
  at::DeviceIndex exchangeDevice(at::DeviceIndex device) const override {
    TORCH_CHECK(device == 0, "DANMA exposes only device index 0");
    return 0;
  }
  at::DeviceIndex maybeExchangeDevice(at::DeviceIndex device) const override {
    TORCH_CHECK(device == 0 || device == -1, "DANMA exposes only device index 0");
    return 0;
  }
  at::Device getDeviceFromPtr(void*) const override {
    return at::Device(at::DeviceType::PrivateUse1, 0);
  }
};

static bool register_hooks [[maybe_unused]] = []() {
  at::RegisterPrivateUse1HooksInterface(new DanmaPrivateUse1Hooks());
  return true;
}();

struct DanmaRemoteAllocator final : at::Allocator {
  at::DataPtr allocate(size_t nbytes) override {
    TORCH_CHECK(nbytes % sizeof(float) == 0, "DANMA allocator supports float32 storage only");
    auto* handle = new RemoteHandle{
        next_tensor_id.fetch_add(1, std::memory_order_relaxed),
        nbytes,
        configured_port()};
    try {
      remote_alloc(handle);
    } catch (...) {
      delete handle;
      throw;
    }
    allocation_count.fetch_add(1, std::memory_order_relaxed);
    return {
        handle,
        handle,
        &DanmaRemoteAllocator::release,
        at::Device(at::DeviceType::PrivateUse1, 0)};
  }

  static void release(void* ptr) {
    auto* handle = static_cast<RemoteHandle*>(ptr);
    if (handle == nullptr) return;
    try {
      remote_free(handle);
    } catch (...) {
      // A dying client cannot safely throw from a DataPtr deleter. The remote
      // tensor may leak until the development node restarts.
    }
    delete handle;
  }

  at::DeleterFnPtr raw_deleter() const override { return &DanmaRemoteAllocator::release; }

  void copy_data(void*, const void*, std::size_t) const final {
    TORCH_CHECK(false, "DANMA remote storage must use registered copy operators");
  }
};

DanmaRemoteAllocator global_allocator;
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &global_allocator);

void check_tensor_metadata(const at::Tensor& tensor) {
  TORCH_CHECK(
      tensor.is_cpu() || tensor.device().type() == c10::DeviceType::PrivateUse1,
      "DANMA supports only CPU <-> remote tensor transfers");
  TORCH_CHECK(tensor.scalar_type() == at::kFloat, "DANMA supports float32 only");
  if (tensor.device().type() == c10::DeviceType::PrivateUse1) {
    TORCH_CHECK(
        tensor.is_contiguous(),
        "DANMA remote tensor kernels currently require contiguous device tensors");
  }
}

at::Tensor custom_empty_memory_format(
    at::IntArrayRef size,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device>,
    std::optional<bool> pin_memory,
    std::optional<at::MemoryFormat> memory_format) {
  TORCH_CHECK(c10::layout_or_default(layout) == c10::Layout::Strided, "DANMA supports strided tensors only");
  TORCH_CHECK(!pin_memory.value_or(false), "DANMA does not support pinned device memory");
  const auto scalar_type = c10::dtype_or_default(dtype);
  TORCH_CHECK(scalar_type == at::kFloat, "DANMA supports float32 only");
  constexpr c10::DispatchKeySet keys(c10::DispatchKey::PrivateUse1);
  return at::detail::empty_generic(size, &global_allocator, keys, scalar_type, memory_format);
}

at::Tensor custom_empty_strided(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device>,
    std::optional<bool> pin_memory) {
  TORCH_CHECK(c10::layout_or_default(layout) == c10::Layout::Strided, "DANMA supports strided tensors only");
  TORCH_CHECK(!pin_memory.value_or(false), "DANMA does not support pinned device memory");
  const auto scalar_type = c10::dtype_or_default(dtype);
  TORCH_CHECK(scalar_type == at::kFloat, "DANMA supports float32 only");
  constexpr c10::DispatchKeySet keys(c10::DispatchKey::PrivateUse1);
  return at::detail::empty_strided_generic(size, stride, &global_allocator, keys, scalar_type);
}

at::Tensor& custom_fill_scalar(at::Tensor& self, const at::Scalar& value) {
  check_tensor_metadata(self);
  remote_fill(remote_handle(self), value.toFloat());
  return self;
}

at::Tensor custom_copy_from(
    const at::Tensor& self,
    const at::Tensor& dst,
    bool non_blocking) {
  (void)non_blocking;
  check_tensor_metadata(self);
  check_tensor_metadata(dst);
  TORCH_CHECK(self.sizes() == dst.sizes(), "DANMA copy shape mismatch");

  if (self.is_cpu() && dst.device().type() == c10::DeviceType::PrivateUse1) {
    // Autograd commonly hands device copies expanded/strided CPU gradients.
    // Materialize only a transient CPU transfer view; DANMA device payload
    // remains exclusively in Rust remote storage.
    const auto host = self.is_contiguous() ? self : self.contiguous();
    remote_upload(
        remote_handle(dst),
        host.const_data_ptr<float>(),
        static_cast<size_t>(host.numel()));
  } else if (
      self.device().type() == c10::DeviceType::PrivateUse1 && dst.is_cpu()) {
    TORCH_CHECK(dst.is_contiguous(), "DANMA download destination must be contiguous");
    remote_download(
        remote_handle(self),
        dst.mutable_data_ptr<float>(),
        static_cast<size_t>(self.numel()));
  } else if (
      self.device().type() == c10::DeviceType::PrivateUse1 &&
      dst.device().type() == c10::DeviceType::PrivateUse1) {
    remote_copy(remote_handle(self), remote_handle(dst));
  } else {
    TORCH_CHECK(false, "unsupported DANMA copy direction");
  }
  copy_count.fetch_add(1, std::memory_order_relaxed);
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
  check_tensor_metadata(self);
  TORCH_CHECK(
      device.is_cpu() || device.type() == c10::DeviceType::PrivateUse1,
      "DANMA can transfer only between CPU and danma:0");
  TORCH_CHECK(dtype == at::kFloat, "DANMA supports float32 only");

  auto out = at::empty(
      self.sizes(),
      dtype,
      self.options().layout(),
      device,
      false,
      memory_format);
  if (self.is_cpu() && device.type() == c10::DeviceType::PrivateUse1) {
    const auto host = self.is_contiguous() ? self : self.contiguous();
    remote_upload(
        remote_handle(out),
        host.const_data_ptr<float>(),
        static_cast<size_t>(host.numel()));
  } else if (self.device().type() == c10::DeviceType::PrivateUse1 && device.is_cpu()) {
    remote_download(remote_handle(self), out.mutable_data_ptr<float>(), static_cast<size_t>(self.numel()));
  } else if (
      self.device().type() == c10::DeviceType::PrivateUse1 &&
      device.type() == c10::DeviceType::PrivateUse1) {
    remote_copy(remote_handle(self), remote_handle(out));
  } else {
    TORCH_CHECK(false, "unsupported DANMA to.Device direction");
  }
  copy_count.fetch_add(1, std::memory_order_relaxed);
  return out;
}

at::Tensor custom_add(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
  return remote_add(self, other, alpha.toFloat());
}

at::Tensor& custom_add_inplace(at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
  TORCH_CHECK(
      self.sizes() == other.sizes(),
      "DANMA add_ currently requires equal shapes");
  remote_add_into(self, other, self, alpha.toFloat());
  return self;
}

at::Tensor custom_mm(const at::Tensor& self, const at::Tensor& mat2) {
  return remote_mm(self, mat2, false, false);
}

at::Tensor custom_linear(
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias) {
  TORCH_CHECK(input.dim() == 2 && weight.dim() == 2, "DANMA linear MVP supports rank-2 input");
  auto out = remote_mm(input, weight, false, true);
  if (bias.has_value() && bias->defined()) {
    out = remote_add(out, *bias, 1.0f);
  }
  return out;
}

class RemoteMmAutograd : public torch::autograd::Function<RemoteMmAutograd> {
 public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      at::Tensor self,
      at::Tensor other) {
    ctx->save_for_backward({self, other});
    return remote_mm(self, other, false, false);
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto grad = grad_outputs[0];
    at::Tensor grad_self;
    at::Tensor grad_other;
    if (ctx->needs_input_grad(0)) {
      grad_self = remote_mm(grad, saved[1], false, true);
    }
    if (ctx->needs_input_grad(1)) {
      grad_other = remote_mm(saved[0], grad, true, false);
    }
    return {grad_self, grad_other};
  }
};

class RemoteAddAutograd : public torch::autograd::Function<RemoteAddAutograd> {
 public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      at::Tensor self,
      at::Tensor other) {
    ctx->saved_data["bias_broadcast"] =
        other.dim() == 1 && self.dim() == 2 && other.size(0) == self.size(1);
    return remote_add(self, other, 1.0f);
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    auto grad = grad_outputs[0];
    at::Tensor grad_self = ctx->needs_input_grad(0) ? grad : at::Tensor();
    at::Tensor grad_other;
    if (ctx->needs_input_grad(1)) {
      grad_other = ctx->saved_data["bias_broadcast"].toBool()
          ? remote_sum_rows(grad)
          : grad;
    }
    return {grad_self, grad_other};
  }
};

class RemoteLinearAutograd : public torch::autograd::Function<RemoteLinearAutograd> {
 public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      at::Tensor input,
      at::Tensor weight,
      at::Tensor bias) {
    ctx->save_for_backward({input, weight});
    ctx->saved_data["has_bias"] = bias.defined();
    auto out = remote_mm(input, weight, false, true);
    if (bias.defined()) {
      out = remote_add(out, bias, 1.0f);
    }
    return out;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto grad = grad_outputs[0];
    at::Tensor grad_input;
    at::Tensor grad_weight;
    at::Tensor grad_bias;
    if (ctx->needs_input_grad(0)) {
      grad_input = remote_mm(grad, saved[1], false, false);
    }
    if (ctx->needs_input_grad(1)) {
      grad_weight = remote_mm(grad, saved[0], true, false);
    }
    if (ctx->saved_data["has_bias"].toBool() && ctx->needs_input_grad(2)) {
      grad_bias = remote_sum_rows(grad);
    }
    return {grad_input, grad_weight, grad_bias};
  }
};

at::Tensor autograd_mm(const at::Tensor& self, const at::Tensor& other) {
  return RemoteMmAutograd::apply(self, other);
}

at::Tensor autograd_add(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha) {
  TORCH_CHECK(alpha.toDouble() == 1.0, "DANMA differentiable add currently requires alpha=1");
  return RemoteAddAutograd::apply(self, other);
}

at::Tensor autograd_linear(
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias) {
  return RemoteLinearAutograd::apply(
      input,
      weight,
      bias.has_value() ? *bias : at::Tensor());
}

std::vector<uint64_t> remote_stats() {
  auto reply = remote_request(configured_port(), kStats, {});
  size_t offset = 1;
  std::vector<uint64_t> stats;
  while (offset + 8 <= reply.size()) {
    stats.push_back(read_u64(reply, offset));
  }
  return stats;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("empty.memory_format", &custom_empty_memory_format);
  m.impl("empty_strided", &custom_empty_strided);
  m.impl("fill_.Scalar", &custom_fill_scalar);
  m.impl("_copy_from", &custom_copy_from);
  m.impl("to.Device", &custom_to_device);
  m.impl("add.Tensor", &custom_add);
  m.impl("add_.Tensor", &custom_add_inplace);
  m.impl("mm", &custom_mm);
  m.impl("linear", &custom_linear);
}

TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
  m.impl("add.Tensor", &autograd_add);
  m.impl("mm", &autograd_mm);
  m.impl("linear", &autograd_linear);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("configure_endpoint", &configure_endpoint);
  m.def("device_count", []() { return 1; });
  m.def("current_device", []() { return 0; });
  m.def("is_available", []() { return true; });
  m.def("allocation_count", []() { return allocation_count.load(std::memory_order_relaxed); });
  m.def("copy_count", []() { return copy_count.load(std::memory_order_relaxed); });
  m.def("remote_stats", &remote_stats);
  m.def("host_payload_bytes", []() { return 0; });
}
