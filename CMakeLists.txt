cmake_minimum_required(VERSION 3.0.0)
project(SSRLS_Tracking VERSION 0.1.0)


find_package(OpenCV REQUIRED)

include_directories(
    ${BOOST_INCLUDE_DIRS}
)

include(CTest)
enable_testing()

add_executable(SSRLS_Tracking main.cpp)
target_link_libraries(SSRLS_Tracking PRIVATE  ${OpenCV_LIBS})


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
