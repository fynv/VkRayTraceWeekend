cmake_minimum_required (VERSION 3.0)

project(VkRayTraceWeekend)

find_package(CUDA REQUIRED)

set (INCLUDE_DIR 
thirdparty/volk
thirdparty/Vulkan-Headers/include
thirdparty/glm/glm
.
)

include_directories(${INCLUDE_DIR})

if (WIN32) 
add_definitions(-D"VK_USE_PLATFORM_WIN32_KHR")
endif()

add_subdirectory(thirdparty/volk)

set (SOURCE
main.cpp
rand_state_init.cu
PathTracer.cpp
)

set (HEADER
context.inl
RNGState.h
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

add_definitions(${DEFINES})

cuda_add_executable(test ${SOURCE} ${HEADER})
target_link_libraries(test volk)






