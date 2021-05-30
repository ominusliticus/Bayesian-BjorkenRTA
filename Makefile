# TO DO: Switch to CMake

CC = g++ -std=c++17 -Wall#-g3 -fsanitize=address
OPT = -O2 -funroll-loops -finline-functions -fopenmp
LIBS = -lpthread -lfmt
INCLUDES = -I /usr/local/include

SRC = ./src/
INC = ./include/
OBJ = ./build/

FILES := $(shell find $(SRC) -name '*.cpp')
OBJ_FILES := $(patsubst $(SRC)%.cpp,$(OBJ)%.o,$(FILES))

CFLAGS = $(OPT)

EXE = ./build/exact_solution.x

all: $(EXE)

$(EXE): $(OBJ_FILES)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS) $(INCLUDE)

$(OBJ)%.o: $(SRC)%.cpp
	$(CC) $(CFLAGS) $(INCLUDE) -MMD -c -o $@ $< 

run:
	./$(EXE)

clean:
	rm -f build/*