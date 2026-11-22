if(NOT TARGET Rachis::RHI)
    find_package(Rachis 0.5 CONFIG REQUIRED)
endif()
add_fill_library(RachisShaderFirst)
