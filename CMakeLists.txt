# clang++ main.cpp -o main -I/usr/local/include/opencv4 -L/usr/local/lib/ -lopencv_core -lopencv_imgcodecs -lopencv_highgui -fcolor-diagnostics -fansi-escape-codes -std=c++11 -rpath=/usr/local/lib 
cmake_minimum_required(VERSION 3.30.3)
project(sotuken CXX)

include_directories(/usr/local/include/opencv4)
link_directories(/usr/local/lib)
add_compile_options(-g -ggdb)

add_executable(main.o main.cpp)
set_property(TARGET main.o PROPERTY CXX_STANDARD 11)
target_link_libraries(main.o opencv_core opencv_imgcodecs opencv_highgui opencv_imgproc)
