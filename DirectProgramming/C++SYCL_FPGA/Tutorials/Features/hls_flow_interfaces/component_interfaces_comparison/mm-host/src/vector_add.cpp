#include <iostream>

// oneAPI headers
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

// #include "annotated_class_util.hpp"
#include <sycl/ext/intel/prototype/annotated_class_util.hpp>
#include "exception_handler.hpp"

// Buffer locations for MM Host interfaces
constexpr int kBL1 = 1;
constexpr int kBL2 = 2;
constexpr int kBL3 = 3;

// Forward declare the kernel name in the global scope. This is an FPGA best
// practice that reduces name mangling in the optimization reports.
class SimpleVAdd;

// Create type alias for the type of kernel argument `c_out`, so it can be
// reused in the annotated memory allocation in the host code
using a_in_t = sycl::ext::oneapi::experimental::annotated_arg<
      int *, decltype(sycl::ext::oneapi::experimental::properties{
                 sycl::ext::intel::experimental::buffer_location<kBL1>,
                 sycl::ext::intel::experimental::dwidth<32>,
                 sycl::ext::intel::experimental::latency<0>,
                 sycl::ext::intel::experimental::read_write_mode_read,
                 sycl::ext::oneapi::experimental::alignment<4>})>;

using b_in_t = sycl::ext::oneapi::experimental::annotated_arg<
      int *, decltype(sycl::ext::oneapi::experimental::properties{
                 sycl::ext::intel::experimental::buffer_location<kBL2>,
                 sycl::ext::intel::experimental::dwidth<32>,
                 sycl::ext::intel::experimental::latency<0>,
                 sycl::ext::intel::experimental::read_write_mode_read,
                 sycl::ext::oneapi::experimental::alignment<4>})>;

using c_out_t = sycl::ext::oneapi::experimental::annotated_arg<
    int *, decltype(sycl::ext::oneapi::experimental::properties{
               sycl::ext::intel::experimental::buffer_location<kBL3>,
               sycl::ext::intel::experimental::dwidth<32>,
               sycl::ext::intel::experimental::latency<0>,
               sycl::ext::intel::experimental::read_write_mode_write,
               sycl::ext::oneapi::experimental::alignment<4>})>;
struct SimpleVAddKernel {
  a_in_t a_in;
  b_in_t b_in;
  c_out_t c_out;

  int len;

  void operator()() const {
    for (int idx = 0; idx < len; idx++) {
      int a_val = a_in[idx];
      int b_val = b_in[idx];
      int sum = a_val + b_val;
      c_out[idx] = sum;
    }
  }
};

constexpr int kVectorSize = 256;

int main() {
  try {
    // Use compile-time macros to select either:
    //  - the FPGA emulator device (CPU emulation of the FPGA)
    //  - the FPGA device (a real FPGA)
    //  - the simulator device
#if FPGA_SIMULATOR
    auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
    auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

    // create the device queue
    sycl::queue q(selector, fpga_tools::exception_handler);

    auto device = q.get_device();

    std::cout << "Running on device: "
              << device.get_info<sycl::info::device::name>().c_str()
              << std::endl;

    int count = kVectorSize;  // pass array size by value

    // declare arrays and fill them
    // Allocate USM shared memory using the utility function `alloc_annotated`
    // (defined in "annotated_class_util.hpp"), which takes an annotated_arg
    // type as the template parameter and returns an instance of such
    // annotated_arg. This ensures the properties of the returned memory
    // (for example, buffer location and alignment) match with the annotations
    // on the kernel arguments
    a_in_t a = fpga_tools::alloc_annotated<a_in_t>(count, q);
    b_in_t b = fpga_tools::alloc_annotated<b_in_t>(count, q);
    c_out_t c = fpga_tools::alloc_annotated<c_out_t>(count, q);

    for (int i = 0; i < count; i++) {
      a[i] = i;
      b[i] = (count - i);
    }

    std::cout << "Add two vectors of size " << count << std::endl;

    q.single_task<SimpleVAdd>(SimpleVAddKernel{a, b, c, count}).wait();

    // verify that VC is correct
    bool passed = true;
    for (int i = 0; i < count; i++) {
      int expected = a[i] + b[i];
      if (c[i] != expected) {
        std::cout << "idx=" << i << ": result " << c[i] << ", expected ("
                  << expected << ") A=" << a[i] << " + B=" << b[i] << std::endl;
        passed = false;
      }
    }

    std::cout << (passed ? "PASSED" : "FAILED") << std::endl;

    sycl::free(a, q);
    sycl::free(b, q);
    sycl::free(c, q);

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;

  } catch (sycl::exception const &e) {
    std::cerr << "Caught a synchronous SYCL exception: " << e.what()
              << std::endl;
    std::cerr << "   If you are targeting an FPGA hardware, "
                 "ensure that your system is plugged to an FPGA board that is "
                 "set up correctly"
              << std::endl;
    std::terminate();
  }
}