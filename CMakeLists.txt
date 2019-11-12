cmake_minimum_required (VERSION 3.0)

add_subdirectory(thirdparty/volk)

project(VkRayTraceWeekend)

find_package(CUDA REQUIRED)

set (INCLUDE_DIR 
thirdparty/volk
thirdparty/Vulkan-Headers/include
thirdparty/glm
.
)

set (SOURCE
main.cpp
PathTracer.cpp
rand_state_init.cu
)

set (HEADER
context.inl
xor_wow_data.hpp
rand_state_init.hpp
PathTracer.h
)


if (WIN32) 
set (DEFINES  ${DEFINES}
-D"_CRT_SECURE_NO_DEPRECATE"  
-D"_SCL_SECURE_NO_DEPRECATE" 
-D"_CRT_SECURE_NO_WARNINGS"
)
else()
add_definitions(-std=c++14)
add_compile_options(-fPIC)
endif()

include_directories(${INCLUDE_DIR})
add_definitions(${DEFINES})

cuda_add_executable(test ${SOURCE} ${HEADER})
target_link_libraries(test volk)






