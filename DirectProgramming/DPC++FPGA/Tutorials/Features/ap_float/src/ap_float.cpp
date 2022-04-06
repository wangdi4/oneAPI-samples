// clang-format off
#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/ac_types/ap_float.hpp>
#include <sycl/ext/intel/ac_types/ap_float_math.hpp>
#include <iomanip> // for std::setprecision

#define _USE_MATH_DEFINES // need to define this for Windows
#include <math.h>
// clang-format on

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#include "dpc_common.hpp"

// Include some helper functions for this tutorial
#include "util.hpp"
using namespace sycl;

// Forward declare the kernel name in the global scope.
// This is a FPGA best practice that reduces name mangling in the optimization
// reports.
class ApproximateSineWithDouble;
class ApproximateSineWithAPFloat;

class ConversionKernelA;
class ConversionKernelB;
class ConversionKernelC;

class SimpleQuadraticEqnSolverKernel;
class SpecializedQuadraticEqnSolverKernel;

// The number of terms in the polynomial approximation of the sine function
constexpr int kSineApproximateTermsCount = 10;

constexpr double kSineApproximationEpsilon = 1e-13;

// ap_float< 8,23> has the same number of exponent and mantissa bits as native
// float type
using APFloatType = ihc::ap_float<8, 23>;

// ap_float<11,52> has the same number of exponent and mantissa bits as native
// double type
using APDoubleType = ihc::ap_float<11, 52>;

using PairAPDoubleType = std::pair<APDoubleType, APDoubleType>;

// Now we are changing the rounding mode on APFloatType and APDoubleType
constexpr auto kRoundingModeRZERO = ihc::fp_config::FP_Round::RZERO;

// ap_float< 8,23> has the same number of exponent and mantissa bits as float
using APFloatTypeB = ihc::ap_float<8, 23, kRoundingModeRZERO>;

// ap_float<11,52> has the same number of exponent and mantissa bits as double
using APDoubleTypeB = ihc::ap_float<11, 52, kRoundingModeRZERO>;

constexpr auto kRoundingModeRNE = ihc::fp_config::FP_Round::RNE;

// -------------------------------------------------------------------------- //
// Polynomial Sine Approximation example
// -------------------------------------------------------------------------- //

// The function template to generate sine-approximation kernels with different
// floating data types
template <typename T, class KernelTag>
void SineApproximationKernel(queue &q, const T &input, T &output) {
  buffer<T, 1> inp_buffer(&input, 1);
  buffer<T, 1> res_buffer(&output, 1);

  q.submit([&](handler &h) {
    accessor x{inp_buffer, h, read_only};
    accessor retval{res_buffer, h, write_only, no_init};

    h.single_task<KernelTag>([=] {
      T res = 0.0;
      T sign = 1.0;
      T term = x[0];
      T numerator = x[0];
      T denom = 1.0;

#pragma unroll
      for (int i = 1; i <= kSineApproximateTermsCount; ++i) {
        res += term;
        sign = -sign;
        denom *= 2 * i * (2 * i + 1);
        numerator *= x[0] * x[0];
        term = sign * numerator / denom;
      }
      retval[0] = res;
    });
  });
}

bool TestSineApproximation(queue &q) {
  bool passed_native = false, passed_non_native = false;

  std::cout << "Testing basic arithmetic operators to approximate the sine "
               "function\n\n";

  double input = M_PI_4;  // pi / 4
  double expected =
      M_SQRT1_2;  // 1/square_root(2), it is the value of sin(input);
  double double_result;

  // Approximate with native double type
  SineApproximationKernel<double, ApproximateSineWithDouble>(q, input,
                                                             double_result);

  // Approximate with ap_float type
  // We set the rounding mode to RZERO (truncate to zero) because this allows us
  // to generate compile-time ap_float constants from double type literals shown
  // below, which eliminates the area usage for initialization.
  using APDoubleTypeC = ihc::ap_float<11, 44, kRoundingModeRZERO>;

  APDoubleTypeC ap_float_input = (APDoubleTypeC)input;
  APDoubleTypeC ap_float_result;

  SineApproximationKernel<APDoubleTypeC, ApproximateSineWithAPFloat>(
      q, ap_float_input, ap_float_result);

  double difference_a = std::abs(double_result - expected);
  double difference_b = std::abs((double)ap_float_result - expected);

  std::cout << "Native Type Result:\n";
  std::cout << "Result     = " << std::setprecision(3) << (double)double_result
            << "\n";
  std::cout << "Expected   = " << std::setprecision(3) << (double)expected
            << "\n";
  std::cout << "Difference = " << std::setprecision(3) << (double)difference_a
            << "\n\n";

  std::cout << "Non Native Type Result:\n";
  std::cout << "Result     = " << std::setprecision(3)
            << (double)ap_float_result << "\n";
  std::cout << "Expected   = " << std::setprecision(3) << (double)expected
            << "\n";
  std::cout << "Difference = " << std::setprecision(3) << (double)difference_b
            << "\n";

  passed_native = (difference_a < kSineApproximationEpsilon);
  passed_non_native = (difference_b < kSineApproximationEpsilon);

  std::cout << "\nSine Approximation: ";
  if (passed_native && passed_non_native) {
    std::cout << "PASSED\n\n";
    return true;
  } else {
    std::cout << "FAILED\n\n";
    return false;
  }
}

// -------------------------------------------------------------------------- //
// Rounding Mode and native type to ap_float type conversion examples
// -------------------------------------------------------------------------- //

// The default rounding mode when converting from other types to APFloatType and
// APDoubleType is RNE (round to nearest) This rounding mode provides better
// accuracy but can be more area intensive than RZERO(truncate to zero)
void TestConversionKernelA(queue &q, const APFloatType &num,
                           APDoubleType &res) {
  buffer<APFloatType, 1> inp_buffer(&num, 1);
  buffer<APDoubleType, 1> res_buffer(&res, 1);

  q.submit([&](handler &h) {
    accessor num_accessor{inp_buffer, h, read_only};
    accessor res_accessor{res_buffer, h, write_only, no_init};

    h.single_task<ConversionKernelA>([=] {
      // x and y will be compile time constants and hence no cast operation will
      // be generated for it.
      const APFloatType x = 3.1f;
      const APDoubleType y = 4.1;

      // This is not free, construction will result in a cast block in RTL from
      // double to float. Constant propagation will not be able to remove this
      // block since the rounding logic for RNE is quite complicated
      const APFloatType z = 4.1;

      // When mixing types in arithmetic operations, rounding operations are
      // needed to promote different types to the same:
      // - x and num are of the same type, so no conversion is required
      // - y and num are not of the same type, num will be promoted to the more
      //   dominant APDoubleType type and this will result in generation of a
      //   cast operation
      // - result of x * num will be promoted to APDoubleType before being added
      // to y * num, this will generate a cast operation
      // - z will be promoted to APDoubleType before being added to the rest,
      // requiring another cast operation
      auto res = x * num_accessor[0] + y * num_accessor[0] + z;
      res_accessor[0] = res;
    });
  });
}

// The rounding mode when converting from other types to APFloatTypeB and
// APDoubleTypeB is RZERO (truncate towards). This rounding mode is simpler and
// can be constant-propagated
void TestConversionKernelB(queue &q, const APFloatTypeB &num,
                           APDoubleTypeB &res) {
  buffer<APFloatTypeB, 1> inp_buffer(&num, 1);
  buffer<APDoubleTypeB, 1> res_buffer(&res, 1);

  q.submit([&](handler &h) {
    accessor num_accessor{inp_buffer, h, read_only};
    accessor res_accessor{res_buffer, h, write_only, no_init};

    h.single_task<ConversionKernelB>([=] {
      const APFloatTypeB x = 3.1f;
      const APDoubleTypeB y = 4.1;

      // Constant propagation will be able to make z a compile-time constant
      // with rounding mode RZERO
      const APFloatTypeB z = 4.1;

      // - x * num : the result of the multiply is cast (promoted) using RNE,
      // resulting in a cast block
      // - y * num : num is cast (promoted) using RZERO which doesn't need an
      // explicit cast block in hardware
      // - z : cast version of z is also a compile time constant so no hardware
      // is generated for the conversion
      auto res = x * num_accessor[0] + y * num_accessor[0] + z;
      res_accessor[0] = res;
    });
  });
}

// For Kernel C, we are using RNE for the conversion on both types. However,
// sometimes we still want to deploy other modes of conversion, especially for
// constructing and casting constants.
void TestConversionKernelC(queue &q, const APFloatType &num,
                           APDoubleType &res) {
  buffer<APFloatType, 1> inp_buffer(&num, 1);
  buffer<APDoubleType, 1> res_buffer(&res, 1);

  q.submit([&](handler &h) {
    accessor num_accessor{inp_buffer, h, read_only};
    accessor res_accessor{res_buffer, h, write_only, no_init};

    h.single_task<ConversionKernelC>([=] {
      const APFloatType x = 3.1f;
      const APDoubleType y = 4.1;

      // y is a compile time constant, so converting y to z using RZERO will
      // also produce a compile time constant.
      const APFloatType z = y.convert_to<8, 23, kRoundingModeRZERO>();

      // The convert_to function allows you to convert ap_float of different
      // precisions using different modes, but you must make sure the receiving
      // type of the convert_to function matches the exponent and mantissa width
      // of the convert_to arguments.
      auto res =
          (x * num_accessor[0])
              .convert_to<11, 52,
                          kRoundingModeRNE>() +  // This conversion generates
                                                 // a cast operation
          y * num_accessor[0] +  // The conversion of num to APDoubleType
                                 // creates a cast operation
          z.convert_to<11, 52,
                       kRoundingModeRZERO>();  // the result of this conversion
                                               // is a compile time constant
      res_accessor[0] = res;
    });
  });
}

template <typename T1, typename T2>
bool RunSpecifiedConversionKernel(queue &q,
                                  void (*kernel_func)(queue &, const T1 &,
                                                      T2 &)) {
  constexpr double kConversionKernelEpsilon = 1e-5;

  T1 input = (10.1f);
  T2 res;
  kernel_func(q, input, res);

  double expected = (3.1 * input) + (4.1 * input) + 4.1;
  double difference = (res - expected).abs();

  std::cout << "Result     = " << std::setprecision(3) << (double)res << "\n";
  std::cout << "Expected   = " << std::setprecision(3) << (double)expected
            << "\n";
  std::cout << "Difference = " << std::setprecision(3) << (double)difference
            << "\n\n";

  return difference < kConversionKernelEpsilon;
}

bool TestAllConversionKernels(queue &q) {
  std::cout << "Testing conversions in ap_float\n";
  bool passed_A = RunSpecifiedConversionKernel<APFloatType, APDoubleType>(
      q, TestConversionKernelA);

  std::cout << "Testing conversions in ap_float with rounding mode RZERO\n";
  bool passed_B = RunSpecifiedConversionKernel<APFloatTypeB, APDoubleTypeB>(
      q, TestConversionKernelB);

  std::cout
      << "Testing conversions in ap_float using the convert_to function\n";
  bool passed_C = RunSpecifiedConversionKernel<APFloatType, APDoubleType>(
      q, TestConversionKernelC);

  std::cout << "Conversion: ";
  if (passed_A && passed_B && passed_C) {
    std::cout << "PASSED\n\n";
    return true;
  } else {
    std::cout << "FAILED\n\n";
    return false;
  }
}

// -------------------------------------------------------------------------- //
// Quadratic Equation Solver example
// -------------------------------------------------------------------------- //

// This kernel computes the two roots from a quadratic equation with
// coefficient a, b, and c, for real numbers only, using the simple mathematical
// operators *, / etc.
void TestSimpleQuadraticEqnSolver(queue &q, const float A, const float B,
                                  const float C, PairAPDoubleType &r) {
  APDoubleType root1, root2;

  {
    buffer<float, 1> inp1_buffer(&A, 1);
    buffer<float, 1> inp2_buffer(&B, 1);
    buffer<float, 1> inp3_buffer(&C, 1);

    buffer<APDoubleType, 1> root1_buffer(&root1, 1);
    buffer<APDoubleType, 1> root2_buffer(&root2, 1);

    q.submit([&](handler &h) {
      accessor x{inp1_buffer, h, read_only};
      accessor y{inp2_buffer, h, read_only};
      accessor z{inp3_buffer, h, read_only};
      accessor r1{root1_buffer, h, write_only, no_init};
      accessor r2{root2_buffer, h, write_only, no_init};

      h.single_task<SimpleQuadraticEqnSolverKernel>([=] {
        APDoubleType a(x[0]), b(y[0]), c(z[0]);
        auto rooted = b * b - 4.0 * a * c;
        PairAPDoubleType ret;
        if (rooted > 0.0 || rooted.abs() < 1e-20) {
          if (rooted < 0.0) {
            rooted = -rooted;
          }
          auto root = ihc::ihc_sqrt(rooted);
          r1[0] = (-b + root) / (2.0 * a);
          r2[0] = (-b - root) / (2.0 * a);
        } else {
          r1[0] = APDoubleType::nan();
          r2[0] = APDoubleType::nan();
        }
      });
    });
  }

  r = std::make_pair(root1, root2);
}

// SimpleQuadraticEqnSolverKernel was relatively area intensive and there
// are many potential optimization opportunities if we fine tune the arithmetic
// instructions. In SpecializedQuadraticEqnSolverKernel we will use the explicit
// ap_float math functions and customize them to improve our quality of results
void TestSpecializedQuadraticEqnSolver(queue &q, const float A, const float B,
                                       const float C, PairAPDoubleType &r) {
  // Accuracy and Subnormal Options must be compile time constants
  constexpr auto kAccuracyLow = ihc::fp_config::FP_Accuracy::LOW;
  constexpr auto kSubnormalOff = ihc::fp_config::FP_Subnormal::OFF;
  constexpr auto kAccuracyHigh = ihc::fp_config::FP_Accuracy::HIGH;
  constexpr auto kSubnormalOn = ihc::fp_config::FP_Subnormal::ON;

  APDoubleType root1, root2;

  {
    buffer<float, 1> inp1_buffer(&A, 1);
    buffer<float, 1> inp2_buffer(&B, 1);
    buffer<float, 1> inp3_buffer(&C, 1);

    buffer<APDoubleType, 1> root1_buffer(&root1, 1);
    buffer<APDoubleType, 1> root2_buffer(&root2, 1);

    q.submit([&](handler &h) {
      accessor x{inp1_buffer, h, read_only};
      accessor y{inp2_buffer, h, read_only};
      accessor z{inp3_buffer, h, read_only};
      accessor r1{root1_buffer, h, write_only, no_init};
      accessor r2{root2_buffer, h, write_only, no_init};

      h.single_task<SpecializedQuadraticEqnSolverKernel>([=] {
        // Use a smaller type if possible, single precision vs double
        APFloatType a(x[0]), b(y[0]), c(z[0]);

        // By default subnormal number processing is off, but for the purpose of
        // demonstration, we also spell it out
        auto bsquare = APDoubleType::mul<kAccuracyLow, kSubnormalOff>(
            b, b);  // here we avoid one upcast from float to double
        auto fourA = APDoubleType::mul<kAccuracyLow, kSubnormalOff>(
            APFloatType(4.0f), a);  // here we avoid one upcast again
        auto fourAC = APDoubleType::mul<kAccuracyLow, kSubnormalOff>(
            fourA, c);  // here we avoid one upcast as well

        // For the subtraction operation, we want to have subnormal number
        // processed because the number can be really small we also want to have
        // a higher precision since we are dealing with small numbers on which
        // we make critical decisions on
        auto rooted =
            APDoubleType::sub<kAccuracyHigh, kSubnormalOn>(bsquare, fourAC);

        if (rooted > 0.0 || rooted.abs() < 1e-20) {
          if (rooted < 0.0) {
            rooted = -rooted;
          }
          auto root = ihc::ihc_sqrt(rooted);
          // divider is expensive, low accuracy would provide a significant area
          // gain. The default option for addition and multiplication (high
          // accuracy and no subnormal) is OK
          r1[0] = APDoubleType::div<kAccuracyLow, kSubnormalOff>(-b + root,
                                                                 2.0 * a);
          r2[0] = APDoubleType::div<kAccuracyLow, kSubnormalOff>(-b - root,
                                                                 2.0 * a);

        } else {
          r1[0] = APDoubleType::nan();
          r2[0] = APDoubleType::nan();
        }
      });
    });
  }

  r = std::make_pair(root1, root2);
}

bool RunSpecifiedQuadraticEqnSolverKernel(queue &q,
                                          void (*func)(queue &, const float,
                                                       const float, const float,
                                                       PairAPDoubleType &)) {
  constexpr double kQuadraticEqnEpsilon = 1e-6;
  constexpr size_t kQuadraticTestsCount = 3;

  double testvec[kQuadraticTestsCount][3] = {
      {1., -5.1, 6.}, {2., 4.1, 2.}, {1., 0.1, 0.}};

  DoublePair golden_results[sizeof(testvec)];
  PairAPDoubleType outputs[kQuadraticTestsCount];
  bool passed = true;

  for (int i = 0; i < kQuadraticTestsCount; ++i) {
    func(q, testvec[i][0], testvec[i][1], testvec[i][2], outputs[i]);
    golden_results[i] =
        quadratic_gold(testvec[i][0], testvec[i][1], testvec[i][2]);

    auto diff_root1 =
        std::fabs((double)outputs[i].first - golden_results[i].first);
    auto diff_root2 =
        std::fabs((double)outputs[i].second - golden_results[i].second);

    std::cout << "Result     = " << std::setprecision(3)
              << (double)outputs[i].first << " and " << std::setprecision(3)
              << (double)outputs[i].second << "\n";
    std::cout << "Expected   = " << std::setprecision(3)
              << (double)golden_results[i].first << " and "
              << std::setprecision(3) << (double)golden_results[i].second
              << "\n";
    std::cout << "Difference = " << std::setprecision(3) << (double)diff_root1
              << " and " << std::setprecision(3) << (double)diff_root2 << "\n";

    if (diff_root1 > kQuadraticEqnEpsilon ||
        diff_root2 > kQuadraticEqnEpsilon) {
      passed = false;
      std::cout << "failed! difference exceeds kQuadraticEqnEpsilon = "
                << kQuadraticEqnEpsilon << "\n";
    }

    std::cout << "\n";
  }

  // test the nan case
  PairAPDoubleType nan_pair;
  func(q, 1., 2., 4., nan_pair);
  std::cout << "Result     = " << nan_pair.first << " and " << nan_pair.second
            << "\n";
  std::cout << "Expected   = NaN and NaN\n";
  if (!(sycl::isnan((double)nan_pair.first) &&
        sycl::isnan((double)nan_pair.second))) {
    passed = false;
    std::cout << "failed! first or second is not a nan!"
              << "\n";
  }
  return passed;
}

bool TestQuadraticEquationSolverKernels(queue &q) {
  std::cout << "Calculating quadratic equation in higher precision\n";
  auto test_a =
      RunSpecifiedQuadraticEqnSolverKernel(q, TestSimpleQuadraticEqnSolver);
  std::cout << "\nCalculating quadratic equation with the optimized kernel\n";
  auto test_b = RunSpecifiedQuadraticEqnSolverKernel(
      q, TestSpecializedQuadraticEqnSolver);

  std::cout << "\nQuadratic Equation Solving: ";
  if (test_a && test_b) {
    std::cout << "PASSED\n";
    return true;
  } else {
    std::cout << "FAILED\n";
    return false;
  }
}

int main() {
#if defined(FPGA_EMULATOR)
  ext::intel::fpga_emulator_selector selector;
#else
  ext::intel::fpga_selector selector;
#endif

  bool passed = true;

  try {
    // Create the SYCL device queue
    queue q(selector, dpc_common::exception_handler);

    passed &= TestSineApproximation(q);
    passed &= TestAllConversionKernels(q);
    passed &= TestQuadraticEquationSolverKernels(q);

  } catch (sycl::exception const &e) {
    // Catches exceptions in the host code
    std::cerr << "Caught a SYCL host exception:\n" << e.what() << "\n";

    // Most likely the runtime couldn't find FPGA hardware!
    if (e.code().value() == CL_DEVICE_NOT_FOUND) {
      std::cerr << "If you are targeting an FPGA, please ensure that your "
                   "system has a correctly configured FPGA board.\n";
      std::cerr << "Run sys_check in the oneAPI root directory to verify.\n";
      std::cerr << "If you are targeting the FPGA emulator, compile with "
                   "-DFPGA_EMULATOR.\n";
    }
    std::terminate();
  }

  if (passed) {
    std::cout << "\nPASSED: all kernel results are correct.\n\n";
  } else {
    std::cout << "\nFAILED\n\n";
  }

  return 0;
}
