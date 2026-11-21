# Owner: C. inkling-server: OpenAI-compatible HTTP server. Links the real engine when the inkcore target
# exists (include this directory after src/engine), otherwise only the stub engine is available.
add_library(inkserver STATIC chat_template.cpp)
target_include_directories(inkserver PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(inkserver PUBLIC inktok)

add_executable(render_main ${CMAKE_SOURCE_DIR}/tests/server/render_main.cpp)
target_link_libraries(render_main inkserver)
add_executable(tok_main ${CMAKE_SOURCE_DIR}/tests/server/tok_main.cpp)
target_link_libraries(tok_main inktok)
