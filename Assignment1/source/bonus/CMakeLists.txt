cmake_minimum_required(VERSION 2.8)
project(HW1_BONUS)

set(CMAKE_C_STANDARD 99)
AUX_SOURCE_DIRECTORY(. SRC)

add_executable(pstree ${SRC})