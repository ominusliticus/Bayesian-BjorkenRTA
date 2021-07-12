# TO DO: Switch to CMake

SRC = ./src/
TST = ./test/
INC = ./include/
OBJ = ./build/
THIRD_PARTY = ./3rd_party/

# Armadillo instructions
ARMA_INC = $(THIRD_PARTY)armadillo/inlcude/
ARMA_LIB = -lopenblas -llapack

FILES := $(shell find $(SRC) -name '*.cpp') $(SRC)fmt/format.cc
OBJ_FILES := $(patsubst $(SRC)%.cpp,$(OBJ)%.o,$(FILES))

TST_FILES := $(TST)test_moments_of_dist_func.cpp $(filter-out $(SRC)main.cpp, $(wildcard $(SRC)*.cpp))
TST_OBJ_FILES := $(patsubst $(TST)%.cpp,$(OBJ)%.o,$(TST_FILES))

CC = g++ -std=c++17 -Wall#-g3 -fsanitize=address
OPT = -O3 -funroll-loops -finline-functions -fopenmp # -fno-stack-protector
LIBS = -lpthread $(ARMA_LIB)
INCLUDE = -I /usr/local/include -I $(ARMA_INC)

CFLAGS = $(OPT)

EXE = ./build/exact_solution.x
TEST_MOM = ./build/test_moments.x
TEST_EXACT = ./build/test_output_exact.x
TEST_EXACT_TUPLE = ./build/test_output_exact_tuple.x

# Tells make file to run commands even when files already exit
.PHONY: all

all: $(EXE)

$(EXE): $(OBJ_FILES)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS) $(INCLUDE)

$(OBJ)%.o: $(SRC)%.cpp Makefile
	$(CC) $(CFLAGS) $(INCLUDE) -MMD -c -o $@ $< 

run:
	$(EXE)

	

test: $(TEST_MOM)

$(TEST_MOM): $(TST_OBJ_FILES)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS) $(INCLUDE)

$(OBJ)%.o: $(TST)%.cpp
	$(CC) $(CFLAGS) $(INCLUDE) -MMD -c $< -o $@  

run_test:
	$(TEST_MOM)

clean:
	rm -f build/*