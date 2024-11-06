cmake_minimum_required(VERSION 3.10)

# Set the project name and version
project(filters VERSION 1.0)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Create a shared library
add_library(filters SHARED)

# Add source files
file(GLOB_RECURSE SOURCES "src/kalman/*cpp")
target_sources(filters PRIVATE ${SOURCES})

# Add include directory
target_include_directories(filters PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Generate export header
include(GenerateExportHeader)
generate_export_header(filters
    EXPORT_MACRO_NAME FILTERS_API
    EXPORT_FILE_NAME ${CMAKE_CURRENT_BINARY_DIR}/filters_export.h
)

# Installation rules
include(GNUInstallDirs)

# Install the library
install(TARGETS filters
    EXPORT filters-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install header files
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/filters
    FILES_MATCHING PATTERN "*.hpp"
)

# Install the export header
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/filters_export.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/filters
)

# Export the targets
install(EXPORT filters-targets
    FILE filters-targets.cmake
    NAMESPACE filters::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/filters
)

# Create a ConfigVersion.cmake file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/filtersConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

# Create a Config.cmake file
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/filtersConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/filters
)

# Install the Config.cmake and ConfigVersion.cmake files
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/filtersConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/filtersConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/filters
)
