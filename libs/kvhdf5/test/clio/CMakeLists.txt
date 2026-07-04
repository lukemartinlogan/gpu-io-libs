# CLIO dependency-contract canary.
#
# This is the ONE test that links clio directly and asserts the raw clio CTE
# producer-only contract kvhdf5 is built on — it includes NO kvhdf5 headers. The
# shared recipe (kvhdf5_add_gpu_test_target, in ../CMakeLists.txt) omits the src/
# include dir unless WITH_KVHDF5_HEADERS is passed, which it is NOT here, so this
# binary physically cannot see kvhdf5 code. If a future clio bump breaks this,
# the failure points at clio, not at our abstractions. The kvhdf5 end-to-end
# tests live in ../e2e; shared CTE server bring-up in ../support/cte_env.h.

if(NOT KVHDF5_HAS_CUDA)
    message(STATUS "clio_contract_tests: skipped (no CUDA compiler)")
    return()
endif()

kvhdf5_add_gpu_test_target(clio_contract_tests
    SRCS cte_putget_test.cu
)
