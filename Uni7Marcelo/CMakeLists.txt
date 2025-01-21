cmake_minimum_required(VERSION 3.12)

include(pico_sdk_import.cmake)

project(RP2040_MIDI_KEYBOARD C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

set(PICO_EXAMPLES_PATH ${PROJECT_SOURCE_DIR})

#target_include_directories(${PROJECT_NAME} .)

# initialize the Raspberry Pi Pico SDK
pico_sdk_init()

add_executable(
                ${PROJECT_NAME}
               # main.c
                )

target_sources(
                ${PROJECT_NAME} PUBLIC
                ${CMAKE_CURRENT_LIST_DIR}/main.c
                ${CMAKE_CURRENT_LIST_DIR}/usb_descriptors.c
                )

# Make sure TinyUSB can find tusb_config.h
target_include_directories(
                            ${PROJECT_NAME} PUBLIC
                            ${CMAKE_CURRENT_LIST_DIR})

# Link Libraries
target_link_libraries(
                        ${PROJECT_NAME}
                        pico_stdlib
                        pico_multicore
                        hardware_i2c
                        hardware_timer
                        hardware_pwm
                        tinyusb_device
                        tinyusb_board
                        )

# Do not use, tinyUSB replaces
#pico_enable_stdio_usb(${PROJECT_NAME} 1)
#pico_enable_stdio_uart(${PROJECT_NAME} 0)

# Create EXEC binaries like uf2
pico_add_extra_outputs(${PROJECT_NAME})
