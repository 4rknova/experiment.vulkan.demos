CXX := g++
STD := -std=c++17
WARN := -Wall -Wextra -Wpedantic
CXXFLAGS := $(STD) $(WARN) -O2
LDLIBS := -lvulkan -lglfw

OUT := vulkan_glfw_quad
SRC := main.cpp
SHADERS := quad.vert.spv quad.frag.spv

all: $(OUT)

$(OUT): $(SRC) $(SHADERS)
	$(CXX) $(CXXFLAGS) -o $@ $(SRC) $(LDLIBS)

quad.vert.spv: quad.vert
	glslangValidator -V $< -o $@

quad.frag.spv: quad.frag
	glslangValidator -V $< -o $@

run: $(OUT)
	./$(OUT)

clean:
	rm -f $(OUT) *.spv