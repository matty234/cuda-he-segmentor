cmake_minimum_required(VERSION 3.0.0)

# add cuda support
enable_language(CUDA)

# set the project name
project(HelloWorld LANGUAGES CXX CUDA)

# add the executable
add_executable(HelloWorld main.cu)
# add debug symbols
set_target_properties(HelloWorld PROPERTIES COMPILE_FLAGS "-g")
# add cuda debug symbols
set_target_properties(HelloWorld PROPERTIES CUDA_FLAGS "-g")
# set the c++ standard
set_property(TARGET HelloWorld PROPERTY CXX_STANDARD 11)

# set the cuda standard
set_property(TARGET HelloWorld PROPERTY CUDA_STANDARD 11)

# set the cuda standard
set_property(TARGET HelloWorld PROPERTY CUDA_STANDARD_REQUIRED ON)

# add libtiff
find_package(TIFF REQUIRED)
target_link_libraries(HelloWorld ${TIFF_LIBRARIES})
# open cv
find_package(OpenCV REQUIRED) 
target_link_libraries(HelloWorld ${OpenCV_LIBS})