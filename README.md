# Malloc-Free Pseudoinverse Solver in Eigen C++ Template Library
This extended Eigen C++ template library provides a malloc-free pseudoinverse solver. 

Standard Eigen pseudoinverse functions, even those with memory preallocation variants, perform heap allocation during execution. This runtime allocation is undesirable in certain cases, specifically in embedded systems [[1]](#Links). The BDCSVD class [[4]](#Links) has been extended to allow for up-front allocation of all used memory, with an ensuing malloc-free execution of the pseudoinverse solver.

## Installation
The Eigen directory can be used or installed exactly as the original Eigen: simply include the headers in your project, and you are good to go! No make required. See [[2]](#Links) for the Eigen documentation.

## Usage

## Tests

## Constraints
The pseudoinverse solver is currently limited to problem sizes of 2500x17 or less. It is not currently templated and is restricted to doubles. Row major is used.

### Licensing
This open-source code is forked from Eigen 3.3.7: https://gitlab.com/libeigen/eigen/-/releases/3.3.7. For the Eigen MPL2 and other licensing information, see the Eigen directory.

### Links
1. https://en.wikipedia.org/wiki/The_Power_of_10:_Rules_for_Developing_Safety-Critical_Code
2. https://eigen.tuxfamily.org/dox/GettingStarted.html
3. https://eigen.tuxfamily.org/dox/TopicPreprocessorDirectives.html
4. https://eigen.tuxfamily.org/dox/classEigen_1_1BDCSVD.html