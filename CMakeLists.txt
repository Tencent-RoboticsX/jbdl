cmake_minimum_required(VERSION 3.12...3.18)
project(jbdl LANGUAGES CXX)

message(STATUS "Using CMake version " ${CMAKE_VERSION})

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

set(SOURCE_DIR "src/jbdl/experimental")
include_directories("${SOURCE_DIR}/math/include")
include_directories("${SOURCE_DIR}/tools/include")
include_directories("${SOURCE_DIR}/qpoases/include")
include_directories("${SOURCE_DIR}/xla_binding_tools/include")


add_definitions(-D__SUPPRESSANYOUTPUT__)

# CPU op library
set(
    QPOASES_SRC
    "${SOURCE_DIR}/qpoases/src/BLASReplacement.cpp"
	"${SOURCE_DIR}/qpoases/src/Bounds.cpp"
	"${SOURCE_DIR}/qpoases/src/Constraints.cpp"
	"${SOURCE_DIR}/qpoases/src/Flipper.cpp"
	"${SOURCE_DIR}/qpoases/src/Indexlist.cpp"
	"${SOURCE_DIR}/qpoases/src/LAPACKReplacement.cpp"
	"${SOURCE_DIR}/qpoases/src/Matrices.cpp"
	"${SOURCE_DIR}/qpoases/src/MessageHandling.cpp"
	"${SOURCE_DIR}/qpoases/src/Options.cpp"
	"${SOURCE_DIR}/qpoases/src/OQPinterface.cpp"
	"${SOURCE_DIR}/qpoases/src/QProblem.cpp"
	"${SOURCE_DIR}/qpoases/src/QProblemB.cpp"
	"${SOURCE_DIR}/qpoases/src/SolutionAnalysis.cpp"
	"${SOURCE_DIR}/qpoases/src/SparseSolver.cpp"
	"${SOURCE_DIR}/qpoases/src/SQProblem.cpp"
	"${SOURCE_DIR}/qpoases/src/SQProblemSchur.cpp"
	"${SOURCE_DIR}/qpoases/src/SubjectTo.cpp"
	"${SOURCE_DIR}/qpoases/src/Utils.cpp"
)
pybind11_add_module(math "${SOURCE_DIR}/math/src/math.cpp"  "${SOURCE_DIR}/math_bindings.cpp")
pybind11_add_module(tools "${SOURCE_DIR}/tools/src/tools.cpp"  "${SOURCE_DIR}/tools_bindings.cpp")
pybind11_add_module(qpoases "${QPOASES_SRC}"  "${SOURCE_DIR}/qpoases_bindings.cpp")
pybind11_add_module(cpu_ops "${QPOASES_SRC}"  "${SOURCE_DIR}/cpu_ops_bindings.cpp")

install(TARGETS math tools qpoases cpu_ops DESTINATION jbdl/experimental)