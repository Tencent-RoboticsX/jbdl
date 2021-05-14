#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "qpOASES.hpp"


namespace py = pybind11;

class QProblemWrapper
{
    public:
        QProblemWrapper(size_t nV, size_t nC){
            QuadProgDim = nV;
            ConstraintDim = nC;
            QProblemCore = qpOASES::QProblem(nV, nC, qpOASES::HST_UNKNOWN,
                               qpOASES::BT_TRUE);
        };
        int init(std::vector<qpOASES::real_t> H,
                 std::vector<qpOASES::real_t> g,
                 std::vector<qpOASES::real_t> A,
                 std::vector<qpOASES::real_t> lb,
                 std::vector<qpOASES::real_t> ub,
                 std::vector<qpOASES::real_t> lbA,
                 std::vector<qpOASES::real_t> ubA,
                 int nWSR){
            qpOASES::returnValue status = QProblemCore.init(H.data(), g.data(),
                A.data(), lb.data(), ub.data(), lbA.data(), ubA.data(), nWSR,  nullptr);
            return int(status);
        };

        std::vector<qpOASES::real_t> getPrimalSolution(){
            std::vector<qpOASES::real_t> QuadProgPrimalSol(QuadProgDim, 0);
            QProblemCore.getPrimalSolution(QuadProgPrimalSol.data());
            return QuadProgPrimalSol;
        };

        std::vector<qpOASES::real_t> getDualSolution(){
            std::vector<qpOASES::real_t> QuadProgDualSol(QuadProgDim + ConstraintDim, 0);
            QProblemCore.getDualSolution(QuadProgDualSol.data());
            return QuadProgDualSol;
        }

        qpOASES::real_t  getObjVal(){
            qpOASES::real_t ObjVal =  QProblemCore.getObjVal();
            return ObjVal;
        }



    private:
        size_t QuadProgDim, ConstraintDim;
        qpOASES::QProblem QProblemCore;

          

};

PYBIND11_MODULE(qpoases, m) {
    // optional module docstring
    m.doc() = "pybind11 example plugin";


    // bindings to Pet class
    py::class_<QProblemWrapper>(m, "QProblem")
        .def(py::init<int, int>())
        .def("init", &QProblemWrapper::init)
        .def("getPrimalSolution", &QProblemWrapper::getPrimalSolution)
        .def("getDualSolution", &QProblemWrapper::getDualSolution)
        .def("getObjVal", &QProblemWrapper::getObjVal);
        // .def("getHessian", &QProblemWrapper::getHessian);
        // .def("init", &qpOASES::QProblem::init);
}