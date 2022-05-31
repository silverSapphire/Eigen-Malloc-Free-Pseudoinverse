# Malloc-Free Pseudoinverse Solver with Eigen C++ Template Library
This extended Eigen C++ template library and wrapper provides a malloc-free Moore-Penrose pseudoinverse solver. 

Standard Eigen pseudoinverse functions, even those with memory preallocation variants, perform heap allocation during execution. This runtime allocation is undesirable in certain cases, particularly in [embedded systems](https://en.wikipedia.org/wiki/The_Power_of_10:_Rules_for_Developing_Safety-Critical_Code). The [BDCSVD class](https://eigen.tuxfamily.org/dox/classEigen_1_1BDCSVD.html) has been extended to allow for up-front allocation of all used memory, with an ensuing malloc-free execution of the decomposition.

The development of this Eigen extension was presented at the Flight Software Workshop, and the talk can be viewed [here](https://www.youtube.com/watch?v=4bzlMmd67WM).

## Installation
- The modified Eigen directory can be used or installed exactly as the original Eigen: simply include the headers in your project, and you are good to go! No make required. See [here](https://eigen.tuxfamily.org/dox/GettingStarted.html) for the original Eigen documentation.

- Example pseudoinverse-solving code around the new Eigen functionality is located in PINV. This can be used as-is, requiring compilation and linking of PINV, or can be copied and pasted into your project as appropriate.

## Usage
```
#include "PINV.h"

int main() {
    PINV p; //Initial heap allocation

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> input =
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(3, 2);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> output =
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(input.cols(), input.rows());

    //fill in input...

    p.reset();
    p.bd.input.block(0, 0, input.rows(), input.cols()) = input;
    p.calculate();
    output = p->bd.output.block(0, 0, output.rows(), output.cols());
}
```

## Constraints
The malloc-free pseudoinverse solver is currently limited to problem sizes of 2500x17 or less. It is not currently templated and is restricted to doubles. Row major is used.

## Tests
Tests and more example usages can be found in [demo.cpp](demo.cpp). 

Eigen developer-specific [preprocessor directives](https://eigen.tuxfamily.org/dox/classEigen_1_1BDCSVD.html) are used to ensure that Eigen performs no heap allocation during execution, specifically EIGEN_RUNTIME_NO_MALLOC.

## Technical Details
Simply, the Eigen modifications allow for BDCSVD to calculate the relevant pieces of a pseudoinverse computation without heap allocation. PINV provides code that takes these pieces and uses them to find the actual pseudoinverse, also without heap allocation. Mathematically, the Moore-Penrose pseudoinverse is calculated via the singular value decomposition through bidiagonalization: BDCSVD inside Eigen performs the decomposition, and PINV uses the decomposition to calculate the pseudoinverse.

### Extensions
Modified Eigen files include:
- Core/MatrixBase
- Householder/Householder
- QR/ColPivHouseholderQR
- SVD/BDCSVD
- SVD/JacobiSVD
- SVD/UpperBidiagonalization

and more. See [this PR](https://github.com/silverSapphire/Eigen-Malloc-Free-Pseudoinverse/pull/2/files) for the full Eigen diff.

### Notes
There are numbers hardcoded into the PINV memory initializations for all of the temporary variables. These were determined by stepping through normal malloc'ing execution of BDCSVD with the original 2500x17 problem size and noting down the allocated sizes. These numbers would likely change if the temporaries were initialized with sizes other than 2500x17. This is also why problem sizes exceeding 2500x17 would likely fail to execute.

Problem sizes smaller than 2500x17 can still be computed accurately since the pseudoinverse is forgiving of 0's padding the matrices-- ie, these do not affect the correctness of the output. It is thus important to zero out the input before filling in the problem set.

## Licensing
This open-source code is forked from [Eigen 3.3.7](https://gitlab.com/libeigen/eigen/-/releases/3.3.7). For the Eigen MPL2 and other licensing information, see [here](https://eigen.tuxfamily.org/index.php?title=Main_Page#License).
