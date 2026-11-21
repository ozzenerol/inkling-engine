# Owner: C. inkling-server: OpenAI-compatible HTTP server. Links the real engine when the inkcore target
# exists (include this directory after the engine), otherwise only the stub engine is available.
add_library(inkserver STATIC chat_template.cpp output_parser.cpp prompt_cache.cpp http.cpp api.cpp
            stub_engine.cpp engine_real.cpp)
target_include_directories(inkserver PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(inkserver PUBLIC inktok)
option(INK_SERVER_ENGINE "link the real engine into inkling-server" ON)
if(INK_SERVER_ENGINE AND TARGET inkcore AND EXISTS ${CMAKE_SOURCE_DIR}/src/engine/engine.cpp)
  target_compile_definitions(inkserver PRIVATE INK_HAVE_ENGINE=1)
  target_link_libraries(inkserver PUBLIC inkcore)
endif()

add_executable(inkling-server main.cpp)
target_link_libraries(inkling-server inkserver)

add_executable(render_main ${CMAKE_SOURCE_DIR}/tests/server/render_main.cpp)
target_link_libraries(render_main inkserver)
add_executable(tok_main ${CMAKE_SOURCE_DIR}/tests/server/tok_main.cpp)
target_link_libraries(tok_main inktok)
add_executable(parser_test ${CMAKE_SOURCE_DIR}/tests/server/parser_test.cpp)
target_link_libraries(parser_test inkserver)
