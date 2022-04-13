# Using the Algorithmic C Fixed Point Data Type 'ap_float'

This FPGA tutorial demonstrates how to use the Algorithmic C (AC) data type `ap_float` and some best practices.

***Documentation***:  The [DPC++ FPGA Code Samples Guide](https://software.intel.com/content/www/us/en/develop/articles/explore-dpcpp-through-intel-fpga-code-samples.html) helps you to navigate the samples and build your knowledge of DPC++ for FPGA. <br>
The [oneAPI DPC++ FPGA Optimization Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide) is the reference manual for targeting FPGAs through DPC++. <br>
The [oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) is a general resource for target-independent DPC++ programming.

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04/20.04, RHEL*/CentOS* 8, SUSE* 15; Windows* 10
| Hardware                          | Intel® Programmable Acceleration Card (PAC) with Intel Arria® 10 GX FPGA <br> Intel® FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix® 10 SX) <br> Intel® FPGA 3rd party / custom platforms with oneAPI support <br> *__Note__: Intel® FPGA PAC hardware is only compatible with Ubuntu 18.04* 
| Software                          | Intel® oneAPI DPC++ Compiler <br> Intel® FPGA Add-On for oneAPI Base Toolkit
| What you will learn               | Including and using the `ap_float` type <br> Using `ap_float` type to trade off mathematical accuracy for lesser resource utilization <br> Using various `ap_float` rounding modes and their effect on accuracy and resource utilization <br> Using the `ap_float `math functions for better quality of results
| Time to complete                  | 1 hour

## Purpose

This FPGA tutorial shows how to use the `ap_float` type with some simple examples and recommended best practices.

This data-type can be used in place of native floating point types to generate area efficient and optimized designs for the FPGA. For example, operations which do not utilize all the bits of the native types or designs which do not require all of the range and precision of native types are good candidates for replacement with the `ap_float` type.

This tutorial will present the following:
1. How to include the `ap_float` type and an overview of common `ap_float` use cases.
2. A Polynomial Sine Approximation example which illustrates how to trade off mathematical accuracy for lesser FPGA resource utilization.
3. Rounding Mode and native type to `ap_float` type conversion examples which describe various `ap_float` rounding modes and their effect on accuracy and FPGA resource utilization.
4. A Quadratic Equation Solver example which showcases explicit `ap_float` math functions and how they can be used to replace mathematical operators like `*, /, +` and `-` for better quality of results.

## Simple Code Example

An `ap_float` number can be defined as follows:

```cpp
ihc::ap_float<E, M> a;
```
which consists of `E+M+1` bits: one sign bit, `E` exponent bits and `M` mantissa bits. For example, `ap_float<8,23>` has the same number of exponent and mantissa bits as native `float`, and `ap_float<11,52>` has the same number of exponent and mantissa bits as native `double`.

Optionally, another template parameter can be specified to set the rounding mode. For more details please refer to the section [*Declare the ap_float Data Type*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide/top/optimize-your-design/resource-use/data-types-and-operations/var-prec-fp-sup/declare-and-use-the-ac-data-types/declare-the-ap-float-data-type.html) in the Intel® oneAPI DPC++ FPGA Optimization Guide.

To use this type in your code, you must include the following header:

```cpp
#include <sycl/ext/intel/ac_types/ap_float.hpp>
```

To use `ap_float` math functions, you must include the following header:

```cpp
#include <sycl/ext/intel/ac_types/ap_float_math.hpp>
```

Additionally, you must pass the flag `-qactypes` (Linux) / `/Qactypes` (Windows) to the `dpcpp` command when compiling your SYCL program in order to ensure that the headers are correctly included. Specify the flag to `dpcpp` if you are invoking `dpcpp` on the command line. The `CMake` file provided with this tutorial will do so automatically.

You can easily convert your existing designs that use native floating-point types to use `ap_float`: simply switch the original type. For math functions, `ap_float` has the "ihc_" prefix, you can simply switch your math functions accordingly, e.g. `sin(x)` should be changed to `ihc_sin(x)` for `ap_float`.

After the migration, you can use the area report to examine the area improvement of your design. In general, the line structure of the area report does not change. For example, instead of seeing a `X bit floating-point multiply` on the old design, the source line for the changed design would show `fpga.vpfp.mul`. 

You should confirm that the area used for the operation has indeed decreased from a Quartus compile. 

## Overview of Common Use Cases for `ap_float`

You should consider migrating to `ap_float` types when you have precision requirements that differ from native `float` and `double` types, including both the range (number of exponent bits) and precision (number of mantissa bits) metrics. 

Double precision operations cannot be placed into a single hardened DSP block like single-precision operations, so double precision operations are significantly more area intensive and use more hardware resources. Moreover, `float` only has 23 bits of mantissa while `double` has 52, this could be an overkill for applications that only seek a sweet spot in between.

Additionally, the built in subnormal support with native `double` type is area intensive and being able to turn subnormal support off can be great for reducing area utilization if the application does not consider very small subnormal numbers.

Finally, the various rounding modes offered along with the `ap_float` type can help trade-off mathematical accuracy for FPGA resource utilization.

## Trading Off Mathematical Accuracy for Better Resource Utilization

Two kernels `ApproximateSineWithDouble` and `ApproximateSineWithAPFloat`, instantiated from the template function `RunSineApproximationKernel()`, implement a simple polynomial approximation of the sine function with single and double precision respectively. 

The former uses `double` type to do so and the latter uses an `ap_float<11,44, Rnd>`. The `Rnd` rounding mode rounds towards zero. These two kernels will illustrate how to trade off accuracy for lesser FPGA resource utilization.

See the section *Examining the Reports* to go over the differences in resource utilization between these kernels. See the section *Example of Output* to see the difference in accuracy of results produced by these kernels.

Note how the kernel function within `RunSineApproximationKernel()` has been written once and the individual kernels are only differentiated by their input/output data types: `ApproximateSineWithDouble` uses `double` data type and `ApproximateSineWithAPFLoat` uses `ap_float` data type.

```cpp
// Approximate sine with native double type
RunSineApproximationKernel<double, ApproximateSineWithDouble>(q, input,
                                                              double_result);
...
constexpr auto Rnd = ihc::fp_config::FP_Round::RZERO;
using ap_float_double = ihc::ap_float<11, 44, Rnd>;

// Approximate sine with `ap_float` type
RunSineApproximationKernel<ap_float_double, ApproximateSineWithAPFloat>(
    q, ap_float_input, ap_float_result);
```

This code-reuse is because `ap_float` is designed to fully blend in with native C++ types for syntax and semantics.

## Conversion Between Native Types and `ap_float`

In normal floating-point FPGA applications, floating-point literals are represented as compile-time constants and implemented as tie-offs (wires that directly connects to Gnd/Vcc) in RTL. This allows the construction of a constant to use no hardware resources in the FPGA flow.

However, `ap_float` types that have non-standard exponent and mantissa widths cannot be trivially converted from C++ native `float` or `double` literals. As a result, the construction of an `ap_float` type may sometimes require FPGA logic resources to round the native floating-point constant to the specified `ap_float`. This is called 'intermediate conversion'.
 
It is important to understand when the intermediate conversions can occur. Conversion does not only happen when you are explicitly casting numbers: it can also happen when you perform arithmetic operations using `ap_float` types with different precisions. Intermediate conversions are necessary because the operation needs to unify the types of the operands by promoting the less "dominant" types (types that have lower representable range). This is demonstrated by the kernel code in the function `TestConversionKernelA`.

### Converting Native Numbers to `ap_float` Numbers with Minimal FPGA Hardware Resources

There are a few ways to generate compile-time `ap_float` constants that do not require any hardware implementation:
 
  1. Initializing `ap_float<8,23>` from `float` or `ap_float<11,52>` from `double` is just a direct bitwise copy (wires in RTL), so if the input `float`/`double` is a compile-time constant, the constructed `ap_float` is also a compile-time constant. You may want to extend these two types instead of the native `float` and `double` type if you want to use `ap_float` specific floating-point arithmetic controls (for example, the explicit binary operation presented in the next section *Using Explicit `ap_float` Math Functions in Place of Mathematical Operators*).
 
  2. Converting from a constant to another `ap_float` that has rounding mode `FP_Round::ZERO` also results in a compile time constant. This rounding mode is also respected in a binary operation when promotion rounding is required. This is demonstrated by the kernel code in the function `TestConversionKernelB()`.

  3. The `convert_to` method of an `ap_float` returns itself rounded to a different type, it accepts a rounding mode as either accurate and area-intensive `RNE` mode (rounds to nearest, tie breaks to even) or inaccurate and non area-intensive `RZERO` (truncate towards zero) mode. When using `RZERO`, the compiler will also be able to convert a constant at compile time. This conversion bypasses the original rounding mode of the `ap_float` type. It is demonstrated by the code in the function `TestConversionKernelC`.

The kernel code in this tutorial contains comments that describe which operations result in generation of explicit cast operations and which do not.

Note:
  1. When assigning the result of the `convert_to` function to another `ap_float`, if the left hand side of the assignment has different exponent or mantissa widths than the ones specified in the `convert_to` function on the right hand side, another conversion can occur. 
 
  2. If your code performs computations on constant/literal native floating-point values, the compiler can sometimes combine them at compile time and save area. This is a compiler optimization technique called 'constant folding' or 'constant propagation'. Please note that this optimization does not work for `ap_float` even when the operands are constant. You should compute your constant arithmetics in native types or pre-compute them by hand.

## Using Explicit `ap_float` Math Functions in Place of Mathematical Operators

In C++ applications, the basic binary operations have little expressiveness. On the contrary, FPGAs implement these operations using configurable logic, so you can improve your design's performance by fine-tuning the floating-point operations since they are usually area and latency intensive.

The general form of explicit operations provided in the `ap_float` math functions are as follows:
 
For addition, subtraction and division, the syntax is:

```
  ihc::ap_float<E, M>::add/sub/div<AccuracyOption, SubnormalOption>(op1, op2)
```

Usage:

* Rounds `op1` and `op2` to the specified `E` (exponent) and `M` (mantissa) widths
* Implements the operation with the provided accuracy and subnormal options.
* Returns the result with type `ap_float<E, M>`
 
For multiplication, the syntax is:

```
  ihc::ap_float<E, M>::mul<AccuracyOption, SubnormalOption>(op1, op2)
```

Usage:

* Leaves `op1` and `op2` intact
* Implements the operation with the provided accuracy and subnormal options.
* Returns the result with type `ap_float<E, M>`
 
The accuracy setting is optional and can be one of the `enum`s below:
```
  ihc::fp_config::FP_Accuracy::HIGH
  ihc::fp_config::FP_Accuracy::LOW
```
 
The subnormal setting is optional can be one of the `enum`s below:
```
  ihc::fp_config::FP_Subnormal::ON
  ihc::fp_config::FP_Subnormal::OFF
```

Note:
* Both `enum`s need to be compile time constants.
* You must specify the accuracy setting if you want to specify the subnormal setting.

The kernel code in the function `TestSpecializedQuadraticEqnSolver()` demonstrates how to use the explicit versions of `ap_float` binary operators to perform floating-point arithmetic operations based on your need.

You can fine-tune the floating-point arithmetic operations when you are multiplying numbers with different precisions and/or outputting the
result of the multiply with a different precision.

You may also want to fine-tune arithmetic operations when you are not concerned about the accuracy of the operation, or when you expect your values to easily fall into the subnormal range and you do not wish to flush them to zero when that happens.
 
To address these use cases, `ap_float` provides an explicit version of binary operators using template functions. The explicit operators provide 3 more main features in addition to basic binary operators.
 
1. Allow inputs and outputs with different precisions in the multiplication.

2. Tweak the area and accuracy trade off of the binary operations.

    The binary operations have high accuracy by default and produce results that are 0.5 ULP off from the most correct result. Users can override the default to choose an implementation with less area but also less precision (1 ULP).

3. Turn on/off subnormal support in the binary operations.

    To save area, subnormal support in the binary operators default to auto, this means it would be off unless there is direct hardened DSP support for it. Users can turn it on when the computation is expected to produce values close to 0, with some additional area.

After fine-tuning the operations, the overall structure of the area report would remain the same, but for each of the fine-tuned operation, you should see an area reduction on the same line if you have chosen to use the low accuracy variant of the operation, or an area increase if you decide to enable subnormal support on an operation.

### Code Example
This section of the tutorial corresponds to the functions `TestSimpleQuadraticEqnSolver` and `TestSpecializedQuadraticEqnSolver`, which contain the kernels `SimpleQuadraticEqnSolverKernel` and `SpecializedQuadraticEqnSolverKernel` respectively.
 
`SimpleQuadraticEqnSolverKernel` demonstrates a design that uses `ap_float` with arithmetic operators to compute the quadratic formula.
 
`SpecializedQuadraticEqnSolverKernel` implements the same design but with the explicit `ap_float` math functions instead of the binary operators. Please refer to the comments in the code to understand how each operation has been tweaked.

To compare the resource utilization between arithmetic operators and explicit math functions, see the section *Examining the reports for the Quadratic Equation Solver Kernels* below to know more about what to look for in the reports.

## Key Concepts
* `ap_float` can be used to improve the quality of results on the FPGA by leveraging various features like arbitrary precision, rounding modes, and explicit math functions.
* Use `ap_float` to reduce the range or precision of the operation as required as opposed to native floating point types which have fixed range and precision.
* Rounding mode `RZERO` produces simpler hardware at the cost of accuracy whereas the default rounding mode `RNE` produces more accurate results and uses more FPGA resources.
* The explicit math functions provided by `ap_float` can be used in place of binary math operators such as `+, -, *` and `/`. The functions provide template parameters for fine tuning the accuracy of the operation and turning subnormal number support on or off.

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.


## Building the `ap_float` Tutorial

### Include Files

The included header `dpc_common.hpp` is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.

### Running Samples in DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the type of compute node and whether to run in batch or interactive mode. Compiles to FPGA are only supported on fpga_compile nodes. Executing programs on FPGA hardware is only supported on fpga_runtime nodes of the appropriate type, such as fpga_runtime:arria10 or fpga_runtime:stratix10.  Neither compiling nor executing programs on FPGA hardware are supported on the login nodes. For more information, see the Intel® oneAPI Base Toolkit Get Started Guide ([https://devcloud.intel.com/oneapi/documentation/base-toolkit/](https://devcloud.intel.com/oneapi/documentation/base-toolkit/)).

When compiling for FPGA hardware, it is recommended to increase the job timeout to 12h.

### On a Linux* System

1. Install the design in `build` directory from the design directory by running `cmake`:

   ```bash
   mkdir build
   cd build
   ```

   If you are compiling for the Intel® PAC with Intel Arria® 10 GX FPGA, run `cmake` using the command:

   ```bash
   cmake ..
   ```

   Alternatively, to compile for the Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX), run `cmake` using the command:

   ```bash
   cmake .. -DFPGA_BOARD=intel_s10sx_pac:pac_s10
   ```
   You can also compile for a custom FPGA platform. Ensure that the board support package is installed on your system. Then run `cmake` using the command:
   ```bash
   cmake .. -DFPGA_BOARD=<board-support-package>:<board-variant>
   ```

2. Compile the design using the generated `Makefile`. The following four build targets are provided that match the recommended development flow:

   * Compile and run for emulation (fast compile time, targets emulates an FPGA device) using:

     ```bash
     make fpga_emu
     ```

   * Generate HTML optimization reports using:

     ```bash
     make report
     ```

   * Compile and run on FPGA hardware (longer compile time, targets an FPGA device) using:

     ```bash
     make fpga
     ```

3. (Optional) As the above hardware compile may take several hours to complete, FPGA precompiled binaries (compatible with Linux* Ubuntu* 18.04) can be downloaded <a href="https://iotdk.intel.com/fpga-precompiled-binaries/latest/ap_float.fpga.tar.gz" download>here</a>.

### On a Windows* System

1. Generate the `Makefile` by running `cmake`.
     ```
   mkdir build
   cd build
   ```
   To compile for the Intel® PAC with Intel Arria® 10 GX FPGA, run `cmake` using the command:  
    ```
    cmake -G "NMake Makefiles" ..
   ```
   Alternatively, to compile for the Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX), run `cmake` using the command:

   ```
   cmake -G "NMake Makefiles" .. -DFPGA_BOARD=intel_s10sx_pac:pac_s10
   ```
   You can also compile for a custom FPGA platform. Ensure that the board support package is installed on your system. Then run `cmake` using the command:
   ```
   cmake -G "NMake Makefiles" .. -DFPGA_BOARD=<board-support-package>:<board-variant>
   ```

2. Compile the design through the generated `Makefile`. The following build targets are provided, matching the recommended development flow:

   * Compile for emulation (fast compile time, targets emulated FPGA device): 
     ```
     nmake fpga_emu
     ```
   * Generate the optimization report: 
     ```
     nmake report
     ``` 
   * Compile for FPGA hardware (longer compile time, targets FPGA device):
     ```
     nmake fpga
     ``` 

*Note:* The Intel® PAC with Intel Arria® 10 GX FPGA and Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX) do not yet support Windows*. Compiling to FPGA hardware on Windows* requires a third-party or custom Board Support Package (BSP) with Windows* support.<br>
*Note:* If you encounter any issues with long paths when compiling under Windows*, you may have to create your ‘build’ directory in a shorter path, for example c:\samples\build.  You can then run cmake from that directory, and provide cmake with the full path to your sample directory.
 
### In Third-Party Integrated Development Environments (IDEs)

You can compile and run this tutorial in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*).
For instructions, refer to the following link: [Intel® oneAPI DPC++ FPGA Workflows on Third-Party IDEs](https://software.intel.com/en-us/articles/intel-oneapi-dpcpp-fpga-workflow-on-ide)

## Examining the Reports

Locate the pair of `report.html` files in either:

* **Report-only compile**:  `ap_float_report.prj`
* **FPGA hardware compile**: `ap_float.prj`

### Examining the Area Reports for the Sine Approximation Kernels

Navigate to the *Area Estimates* page. Click on the *Kernel System* line to expand it.

Observe the difference in resource utilization of the kernels `ApproximateSineWithDouble` and `ApproximateSineWithAPFloat`.

Expand the lines with the kernel names by clicking on them and expand the sub hierarchies to observe how the `add, mult` and `div`
operations use lesser resources for the `ApproximateSineWithAPFloat` kernel.

You should observe an area reduction in resource utilization of up to 30% for the binary operations.

### Examining the Reports for Conversion Kernels

You can find the usages of conversion in both the area report and the graph viewer. The name of the rounding block is "cast".
Let's look at the reports and analyze each kernel in the tutorial.

1. Kernel: `ConversionKernelA`
  This kernel uses the default rounding mode `RNE` - round to nearest.

  Navigate to the *System Viewer* report (*Views* > *System Viewer*) and on the left pane, click on the cluster under `ConversionKernelA`. You should see the conversion functions mentioned in the comments of the source code as "cast" nodes in the graph. The casts from literal types are eliminated at compile time.

  The 4 cast nodes correspond to the following pieces of code:

  ```cpp
  const floatTy z = 4.1;
  ```

  ```cpp
  auto res = x * num_accessor[0] + y * num_accessor[0] + z;
  ```

  The comments in the kernel code describe how these 2 lines generate 4 "cast" nodes.

2. Kernel: `ConversionKernelB`
  This kernel uses the simpler rounding mode `RZERO`.

  In the graph for the cluster under `ConversionKernelB`, you will find that it now only contains one "cast" node. This corresponds to the code:
  ```cpp
  x * num_accessor[0] + ...
  ```
  Although `x` and `num_accessor[0]` represent `ap_float`s constructed to use rounding mode `RZERO`, the result of this operation is cast to the higher precision `ap_float` using the default rounding mode `RNE` as the multiplication result is an operand for the next operation which uses higher precision.

  The other cast node is represented by a combination of `shift`, `select`, and `and` operations hence only one node labeled as "cast" is visible in the reports.
  
  Similarly, the reduction in the number of cast nodes as compared to `ConversionKernelA` results in reduction of hardware resources used by `ConversionKernelC`.
  
2. Kernel: `ConversionKernelC`
  This kernel shows how to use the `convert_to` function and modify the rounding mode for a specific operation.

  In the graph for the cluster under `Kernel_C`, you will find that it contains two "cast" nodes, corresponding to the conversions:
  ```cpp
  auto res = (x * num_accessor[0]).convert_to<11, 52, RndN>() + // This conversion is done explicitly
              y * num_accessor[0] + // This conversion is done explicitly
              z.convert_to<11, 52, RndZ>(); 
  ```


### Examining the Reports for the Quadratic Equation Solver Kernels

Navigate to the *Area Estimates* report and expand the *Kernel System* section. Observe the differences in area utilization for the two kernels `SimpleQuadraticEqnSolverKernel` and `SpecializedQuadraticEqnSolverKernel`. You should observe a decrease in area of the multiplier in the calculation of `b*b - 4*a*c` at their corresponding line numbers.

You should also observe a significant area estimation reduction of the divider from changing it to the low accuracy mode in the report. Also note that the area increase of the subtraction as we enable the subnormal support.

## Running the Sample

1. Run the sample on the FPGA emulator (the kernel executes on the CPU):

   ```bash
   ./ap_float.fpga_emu    # Linux
   ap_float.fpga_emu.exe  # Windows
   ```

2. Run the sample on the FPGA device

   ```bash
   ./ap_float.fpga             # Linux
   ```

### Example of Output

```txt
Testing basic arithmetic operators to approximate the sine function

Native Type Result:
Result     = 0.707
Expected   = 0.707
Difference = 1.11e-16

Non Native Type Result:
Result     = 0.707
Expected   = 0.707
Difference = 5.12e-14

PASSED

Testing conversions in ap_float
Result     = 76.8
Expected   = 76.8
Difference = 1.81e-06

Testing conversions in ap_float with rounding mode RZERO
Result     = 76.8
Expected   = 76.8
Difference = 1.81e-06

Testing conversions in ap_float using the convert_to function
Result     = 76.8
Expected   = 76.8
Difference = 1.81e-06

PASSED

Calculating quadratic equation in higher precision
Result     = 3.26 and 1.84
Expected   = 3.26 and 1.84
Difference = 2.19e-07 and 1.24e-07

Result     = -0.8 and -1.25
Expected   = -0.8 and -1.25
Difference = 8.48e-08 and 1.32e-07

Result     = 0 and -0.1
Expected   = 0 and -0.1
Difference = 0 and 1.49e-09

Result     = NaN and NaN
Expected   = NaN and NaN

Calculating quadratic equation with the optimized kernel
Result     = 3.26 and 1.84
Expected   = 3.26 and 1.84
Difference = 2.19e-07 and 1.24e-07

Result     = -0.8 and -1.25
Expected   = -0.8 and -1.25
Difference = 8.48e-08 and 1.32e-07

Result     = 0 and -0.1
Expected   = 0 and -0.1
Difference = 0 and 1.49e-09

Result     = NaN and NaN
Expected   = NaN and NaN

PASSED
```

### Discussion of Results
`ap_float` can be leveraged to improve the design performance and fine tune FPGA resource utilization.
