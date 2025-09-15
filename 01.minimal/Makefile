CXX := g++
# Common warnings + C++17
WARN := -Wall -Wextra -Wpedantic
STD  := -std=c++17

# Build types
CXXFLAGS_RELEASE := $(STD) $(WARN) -O2 -DNDEBUG
CXXFLAGS_DEBUG   := $(STD) $(WARN) -O0 -g

# Link Vulkan directly (no pkg-config)
LDLIBS := -lvulkan

SRC := main.cpp
OUT_RELEASE := hello_vulkan
OUT_DEBUG   := hello_vulkan_debug

.PHONY: all release debug run clean

all: release

release: $(OUT_RELEASE)
debug:   $(OUT_DEBUG)

$(OUT_RELEASE): $(SRC)
	$(CXX) $(CXXFLAGS_RELEASE) -o $@ $^ $(LDLIBS)

$(OUT_DEBUG): $(SRC)
	$(CXX) $(CXXFLAGS_DEBUG) -o $@ $^ $(LDLIBS)

run: $(OUT_RELEASE)
	./$(OUT_RELEASE)

clean:
	rm -f $(OUT_RELEASE) $(OUT_DEBUG)
