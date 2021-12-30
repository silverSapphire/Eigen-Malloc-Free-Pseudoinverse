#define EIGEN_DEFAULT_TO_ROW_MAJOR
#define EIGEN_RUNTIME_NO_MALLOC

#include "Eigen/Dense"
#include "Eigen/Core"
#include <iostream>

//Uses modified Eigen
int main() {

    uint NMODE = 17;
    uint NPIX = 2500;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> zmm_raw =
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Ones(NPIX, NMODE);
    Eigen::Ref< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> >
        zmm(zmm_raw.block(0, 0, NPIX, NMODE));
    for(int i = 0; i < zmm.rows(); i++) {
        for(int j = 0; j < zmm.cols(); j++) {
            zmm(i, j) = i;
        }
    }

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> copy =
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>::
        Zero(zmm.rows(), zmm.cols());

    //bidiagonal stuff...
    Eigen::internal::BandMatrix<double, -1, -1, 1, 0, 1> bdg(zmm.cols(), zmm.cols());

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> hh =
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>::Zero(2500, 17);
    Eigen::Matrix<double, Eigen::Dynamic, 1> col_temp =
        Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(2500, 1);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> ess_temp =
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>::Zero(2500, 1);
    //Should be column major XXX
    Eigen::internal::UpperBidiagonalization< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> > 
        bid(copy, &hh, &bdg, &col_temp, &ess_temp);

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> bid_dense =
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Zero(zmm.cols(), zmm.cols());

    //Create the svd object
    Eigen::BDCSVD< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> > svd
        (zmm.rows(), zmm.cols(), Eigen::ComputeThinU | Eigen::ComputeThinV);

    //Create the Jacobi object
    //Should be column=major XXX
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> jacobi_temp_member =
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>::Zero(8, 1);
    Eigen::JacobiSVD< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> > 
        jacobi
        (9, 8, &jacobi_temp_member, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> jacobi_temp =
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>::Zero(9, 8);

    //Stuff for the divide function, n=17
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> UofSVD =
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>::Zero(18, 18);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> VofSVD =
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>::Zero(17, 17);
    Eigen::Matrix<double, Eigen::Dynamic, 1> singVals =
        Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(17, 1);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> workspace =
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>::Zero(1, 17);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> q1 =
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>::Zero(9, 1);

    //Post-boot, no malloc!
    Eigen::internal::set_is_malloc_allowed(false);

    //this actually gets inside the function and runs, errors later but unrelated
    std::cout << "ACTUAL COMPUTE CALL\n";
    svd.compute(zmm_raw, &copy, &bid, &bid_dense, &jacobi, &jacobi_temp, 
            &singVals, &UofSVD, &VofSVD, &workspace, &q1, &ess_temp);
    std::cout << "singularValues" << svd.singularValues() << std::endl;
    for(int i = 0; i < svd.matrixV().rows(); i++) {
        for(int j = 0; j < svd.matrixV().cols(); j++) {
            std::cout << "(" << i << ", " << j << ") " <<svd.matrixV()(i, j) << std::endl;
        }
    }

}
