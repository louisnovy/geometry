if(TARGET igl::core)
    return()
endif()

include(FetchContent)
FetchContent_Declare(
    libigl
    GIT_REPOSITORY https://github.com/libigl/libigl.git
    GIT_TAG 7b6cc272845d583934c6aa4d30cc3c7a31a8f11c
)
FetchContent_MakeAvailable(libigl)