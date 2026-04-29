CC := g++
CFLAGS := -Wall -Wextra -Iinclude -O2 -fno-elide-constructors -std=c++23
SRC_DIR := src
OBJ_DIR := obj
BIN_DIR := bin

# 1. Check for a passed variable instead of MAKECMDGOALS [cite: 14]
ifdef __TEST_MAIN__
    # Change the target name from main to nativemath
    TARGET := $(BIN_DIR)/nativemath
    LDFLAGS :=
    # 2. Pass the macro to the compiler 
    CFLAGS += -D__TEST_MAIN__
else
    TARGET := $(BIN_DIR)/libnativemath.so
    CFLAGS += -fPIC
    LDFLAGS := -shared
endif

# 3. Fixed file extension mismatch (*.c instead of *.cpp) [cite: 14]
SRCS := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(SRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

all: $(TARGET)

$(TARGET): $(OBJS) | $(BIN_DIR)
	$(CC) $(OBJS) $(LDFLAGS) -o $@ 

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BIN_DIR) $(OBJ_DIR):
	mkdir -p $@

clean:
	@if [ -d $(OBJ_DIR) ]; then find $(OBJ_DIR) -type f ! -name '.gitkeep' -delete; fi
	@if [ -d $(BIN_DIR) ]; then find $(BIN_DIR) -type f ! -name '.gitkeep' -delete; fi

.PHONY: all clean