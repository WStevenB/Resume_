CC = g++

INC_DIR := incl
SRC_DIR := src
OBJ_DIR := obj
BIN_DIR := bin

EXE := $(BIN_DIR)/GunVision_3.0
SRC := $(wildcard $(SRC_DIR)/*.cpp)
OBJ := $(SRC:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

LDFLAGS := -framework OpenGL -framework OpenCL -framework GLUT `pkg-config --libs /opt/homebrew/Cellar/opencv/4.7.0_1/lib/pkgconfig/opencv4.pc`

CPPFLAGS := -w -std=gnu++11 -MMD -MP -Wall -g

EXTERNAL_INCLUDES := /opt/homebrew/Cellar/opencv/4.7.0_1/include/opencv4

INCLUDES := -I $(INC_DIR) $(foreach i, $(EXTERNAL_INCLUDES), -I$i)

.PHONY: all clean

all: $(EXE)

$(EXE): $(OBJ) | $(BIN_DIR)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CC) $(CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BIN_DIR) $(OBJ_DIR):
	mkdir -p $@

clean:
	@$(RM) -rv $(BIN_DIR) $(OBJ_DIR)

-include $(OBJ:.o=.d)
