if(WIN32)
    set(USERHOME $ENV{USERPROFILE})
else()
    set(USERHOME $ENV{HOME})
endif()
set(sdkVersion 2.1.0)
set(toolchainVersion 13_3_Rel1)
set(picotoolVersion 2.1.0)
set(picoVscode ${USERHOME}/.pico-sdk/cmake/pico-vscode.cmake)
if (EXISTS ${picoVscode})
    include(${picoVscode})
endif()
set(PICO_BOARD pico CACHE STRING "Board type")

cmake_minimum_required(VERSION 3.13)
include(pico_sdk_import.cmake)

project(blink_new C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
pico_sdk_init()

add_executable(mandelbrot-fixvfloat)

pico_generate_pio_header(mandelbrot-fixvfloat ${CMAKE_CURRENT_LIST_DIR}/hsync.pio)
pico_generate_pio_header(mandelbrot-fixvfloat ${CMAKE_CURRENT_LIST_DIR}/vsync.pio)
pico_generate_pio_header(mandelbrot-fixvfloat ${CMAKE_CURRENT_LIST_DIR}/rgb.pio)

pico_enable_stdio_usb(mandelbrot-fixvfloat 1)
pico_enable_stdio_uart(mandelbrot-fixvfloat 0)

target_sources(mandelbrot-fixvfloat PRIVATE mandelbrot_fixvfloat.c vga_graphics.c registers.h)

target_link_libraries(mandelbrot-fixvfloat PRIVATE pico_stdlib pico_multicore pico_bootsel_via_double_reset hardware_spi hardware_sync hardware_pio hardware_dma hardware_adc)

pico_add_extra_outputs(mandelbrot-fixvfloat)
