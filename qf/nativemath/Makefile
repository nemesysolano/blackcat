CC := g++
CFLAGS := -Wall -Wextra -Iinclude -O2 -fno-elide-constructors -std=c++23
SRC_DIR := src
OBJ_DIR := obj
BIN_DIR := bin
RESOURCE_DIR := resources

TENSORFLOW_LITE_BASE := /Volumes/APPFS512GB/workspaces/LiteRT
TENSORFLOW_LITE_BUILD := $(TENSORFLOW_LITE_BASE)/litert/cmake_build
TENSORFLOW_LITE_INC_DIRS := $(TENSORFLOW_LITE_BASE) $(TENSORFLOW_LITE_BASE)/include $(TENSORFLOW_LITE_BASE)/abseil-cpp $(TENSORFLOW_LITE_BASE)/flatbuffers/include $(TENSORFLOW_LITE_BUILD)/include $(TENSORFLOW_LITE_BUILD)/abseil-cpp
TENSORFLOW_LITE_LIB_DIR := $(TENSORFLOW_LITE_BUILD)/c

LITERT_FLAGS := -lLiteRt -framework Metal -framework CoreVideo -framework Foundation -framework Accelerate
RPATH_FLAGS := -Wl,-rpath,$(TENSORFLOW_LITE_LIB_DIR)

ifdef __TEST_MAIN__
    TARGET := $(BIN_DIR)/nativemath    
    CFLAGS += -D__TEST_MAIN__ $(addprefix -I, $(TENSORFLOW_LITE_INC_DIRS)) -Wno-mismatched-tags -Wno-unused-parameter -Wno-deprecated-declarations -Wno-sign-compare
    # Link the library, frameworks, and set the runtime search path
    LDFLAGS := -L$(TENSORFLOW_LITE_LIB_DIR) $(LITERT_FLAGS) $(RPATH_FLAGS)
else
    # Note: On macOS, shared libraries usually use .dylib, but LiteRT builds often use .so symlinks
    TARGET := $(BIN_DIR)/libnativemath.so
    CFLAGS += -fPIC $(addprefix -I, $(TENSORFLOW_LITE_INC_DIRS)) -Wno-mismatched-tags -Wno-unused-parameter -Wno-deprecated-declarations
    # Add -shared for the library target along with LiteRT flags
    LDFLAGS := -shared -L$(TENSORFLOW_LITE_LIB_DIR) $(LITERT_FLAGS) $(RPATH_FLAGS)
endif

# 3. Fixed file extension mismatch (*.c instead of *.cpp) [cite: 14]
SRCS := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(SRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

all: $(TARGET)

$(TARGET): $(OBJS) | $(BIN_DIR)
	$(CC) $(OBJS) $(LDFLAGS) -o $@ 
	@echo "Copying LiteRT libraries to $(BIN_DIR)..."
	cp $(TENSORFLOW_LITE_LIB_DIR)/libLiteRt.dylib $(BIN_DIR)/
	cp $(TENSORFLOW_LITE_LIB_DIR)/libLiteRtMetalAccelerator.dylib $(BIN_DIR)/
	cp $(RESOURCE_DIR)/HII.US.tflite $(BIN_DIR)/
	cp $(RESOURCE_DIR)/HII.US.csv $(BIN_DIR)/

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BIN_DIR) $(OBJ_DIR):
	mkdir -p $@

clean:
	@if [ -d $(OBJ_DIR) ]; then find $(OBJ_DIR) -type f ! -name '.gitkeep' -delete; fi
	@if [ -d $(BIN_DIR) ]; then find $(BIN_DIR) -type f ! -name '.gitkeep' -delete; fi

.PHONY: all clean