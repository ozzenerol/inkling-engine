# Owner: C. Tokenizer + the JSON value it parses tokenizer.json with.
find_path(PCRE2_INCLUDE_DIR pcre2.h HINTS /opt/homebrew/include REQUIRED)
find_library(PCRE2_LIB pcre2-8 HINTS /opt/homebrew/lib REQUIRED)
add_library(inktok STATIC tokenizer.cpp ../server/json.cpp)
target_include_directories(inktok PUBLIC ${CMAKE_SOURCE_DIR}/src PRIVATE ${PCRE2_INCLUDE_DIR})
target_link_libraries(inktok PUBLIC ${PCRE2_LIB})
target_compile_features(inktok PUBLIC cxx_std_20)
