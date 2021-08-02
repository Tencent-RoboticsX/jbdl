cmake_minimum_required(VERSION 3.12...3.18)
project(jbdl LANGUAGES CXX)
enable_language(C)

option (CUDA_SUPPORT "Enable CUDA support" OFF)
configure_file(jaxrbdl_config.h.in jaxrbdl_config.h)

message(STATUS "Using CMake version " ${CMAKE_VERSION})

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

set(SOURCE_DIR "src/jbdl/experimental")
include_directories(${SOURCE_DIR})
include_directories("${SOURCE_DIR}/math/include")
include_directories("${SOURCE_DIR}/tools/include")
include_directories("${SOURCE_DIR}/qpoases/include")
include_directories("${SOURCE_DIR}/xla_binding_tools/include")
include_directories("${SOURCE_DIR}/cuosqp/osqp_sources/include")
include_directories("${SOURCE_DIR}/cuosqp/osqp_sources/lin_sys/cuda/cuda_pcg")
include_directories("${SOURCE_DIR}/cuosqp/osqp_sources/lin_sys/cuda/cuda_pcg/include")
include_directories("${SOURCE_DIR}/cuosqp/osqp_sources/algebra/cuda/")
include_directories("${SOURCE_DIR}/cuosqp/osqp_sources/algebra/cuda/include/")
include_directories("/usr/local/cuda/samples/common/inc")
include_directories("/usr/local/cuda/targets/x86_64-linux/include")


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

if (CUDA_SUPPORT)
    enable_language(CUDA)
    find_package(CUDA)
    set(
        CUOSQP_SRC_gpu
        "${SOURCE_DIR}/lcp_gpu/kernels.cc.cu"
        "${SOURCE_DIR}/cuosqp/osqp_sources/src/auxil.c"
        "${SOURCE_DIR}/cuosqp/osqp_sources/src/ctrlc.c"
        "${SOURCE_DIR}/cuosqp/osqp_sources/src/error.c"
        "${SOURCE_DIR}/cuosqp/osqp_sources/src/lin_sys.c"
        "${SOURCE_DIR}/cuosqp/osqp_sources/src/osqp_api.c"
        "${SOURCE_DIR}/cuosqp/osqp_sources/src/polish.c"
        "${SOURCE_DIR}/cuosqp/osqp_sources/src/proj.c"
        "${SOURCE_DIR}/cuosqp/osqp_sources/src/scaling.c"
        "${SOURCE_DIR}/cuosqp/osqp_sources/src/util.c"
        "${SOURCE_DIR}/cuosqp/osqp_sources/algebra/cuda/algebra_libs.c"
        "${SOURCE_DIR}/cuosqp/osqp_sources/algebra/cuda/matrix.cu"
        "${SOURCE_DIR}/cuosqp/osqp_sources/algebra/cuda/vector.cu"
        "${SOURCE_DIR}/cuosqp/osqp_sources/algebra/cuda/src/cuda_csr.cu"
        "${SOURCE_DIR}/cuosqp/osqp_sources/algebra/cuda/src/cuda_handler.cu"
        "${SOURCE_DIR}/cuosqp/osqp_sources/algebra/cuda/src/cuda_lin_alg.cu"
        "${SOURCE_DIR}/cuosqp/osqp_sources/algebra/cuda/src/cuda_malloc.cu"
        "${SOURCE_DIR}/cuosqp/osqp_sources/lin_sys/cuda/cuda_pcg/cuda_pcg_interface.c"
        "${SOURCE_DIR}/cuosqp/osqp_sources/lin_sys/cuda/cuda_pcg/src/cuda_pcg.cu"
    )

    pybind11_add_module(gpu_ops SHARED "${CUOSQP_SRC_gpu}"  "${SOURCE_DIR}/lcp_gpu/gpu_ops.cc")
    target_link_libraries (gpu_ops PRIVATE cublas cusparse)

    install(TARGETS math tools qpoases cpu_ops gpu_ops DESTINATION jbdl/experimental)

    add_executable (test_gpu ${SOURCE_DIR}/lcp_gpu/test.cpp)
    target_link_libraries (test_gpu gpu_ops)

else()

    install(TARGETS math tools qpoases cpu_ops DESTINATION jbdl/experimental)

endif()
