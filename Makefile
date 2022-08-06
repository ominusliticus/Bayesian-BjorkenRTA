# TO DO: Switch to CMake

SRC = ./src/
TST = ./test/
INC = ./include
OBJ = ./build/
THIRD_PARTY = ./3rd_party/

FILES := $(shell find $(SRC) -name '*.cpp') $(SRC)fmt/format.cc
OBJ_FILES := $(patsubst $(SRC)%.cpp,$(OBJ)%.o,$(FILES))

TST_MOMENTS := $(TST)test_moments_of_dist_func.cpp $(filter-out $(SRC)main.cpp, $(wildcard $(SRC)*.cpp)) $(SRC)fmt/format.cc
TST_MOMENTS_FILES := $(patsubst $(TST)%.cpp,$(OBJ)%.o,$(TST_MOMENTS))

TST_HYDROS := $(TST)test_run_all_hydros.cpp $(filter-out $(SRC)main.cpp, $(wildcard $(SRC)*.cpp)) $(SRC)fmt/format.cc
TST_HYDROS_FILES := $(patsubst $(TST)%.cpp,$(OBJ)%.o,$(TST_HYDROS))

TST_INVERSION := $(TST)test_variable_inversion.cpp $(filter-out $(SRC)main.cpp, $(wildcard $(SRC)*.cpp)) $(SRC)fmt/format.cc
TST_INVERSION_FILES := $(patsubst $(TST)%.cpp,$(OBJ)%.o,$(TST_INVERSION))

UNAME_S := $(shell uname -s)
$(info $$UNAME_S is [$(UNAME_S)])
ifeq ($(UNAME_S),Linux)
	CC = g++-11 -std=c++20 -Wall #-g3 -fsanitize=address
	OPT = -O3 -funroll-loops -finline-functions -fopenmp # -fno-stack-protector

	# Armadillo instructions
	ARMA_INC = $(THIRD_PARTY)armadillo/include/
	ARMA_LIB = -lopenblas -llapack
	
	LIBS = -lpthread $(ARMA_LIB)
	INCLUDE = -I /usr/local/include -I $(ARMA_INC) -I $(INC)
endif
ifeq ($(UNAME_S),Darwin)
	CC = clang++ -std=c++20 -Wall
	OPT = -O3 -funroll-loops -finline-functions -arch x86_64

	ARMA_INC = $(THIRD_PARTY)armadillo/include/
	ARMA_LIB = $(wildcard /usr/local/Cellar/openblas/*/lib/libopenblas.a) $(wildcard /usr/local/Cellar/lapack/*/lib/liblapack.a)

	LIBS = -lpthread $(ARMA_LIB) $(wildcard /usr/local/Cellar/libomp/*/lib/libomp.a)
	INCLUDE = -I /usr/local/include -I $(wildcard /usr/local/Cellar/libomp/*/include) -I $(wildcard /usr/local/Cellar/lapack/*/include) -I $(ARMA_INC) -I $(INC)
endif
CFLAGS = $(OPT)

EXE = ./build/exact_solution.x
TEST_MOM = ./build/test_moments.x
TEST_HYDRO = ./build/test_run_all_hydros.x
TEST_INVERSION = ./build/test_variable_inversion.x

#DEPS := $(OBJ_FILES:.o=.d)

-include $(DEPS)

# Tells make file to run commands even when files already exit
.PHONY: all

all: $(EXE)

$(EXE): $(OBJ_FILES)
	$(CC) $(CFLAGS) -MMD -o $@ $^ $(LIBS) $(INCLUDE)

$(OBJ)%.o: $(SRC)%.cpp Makefile
	$(CC) $(CFLAGS) $(INCLUDE) -MMD -c -o $@ $< 

run:
	$(EXE)

###########################################################
## Tests
###########################################################
# TODO: Write comments that describe each test

test_moments: $(TEST_MOM)

$(TEST_MOM): $(TST_MOMENTS_FILES)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS) $(INCLUDE)

$(OBJ)%.o: $(TST)%.cpp
	$(CC) $(CFLAGS) $(INCLUDE) -MMD -c $< -o $@  

run_test_moments:
	$(TEST_MOM)


test_hydros: $(TEST_HYDRO)

$(TEST_HYDRO): $(TST_HYDROS_FILES)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS) $(INCLUDE)

$(OBJ)%.o: $(TST)%.cpp
	$(CC) $(CFLAGS) $(INCLUDE) -MMD -c $< -o $@

run_test_hydros:
	$(TEST_HYDRO)


test_inversion: $(TEST_INVERSION)

$(TEST_INVERSION): $(TST_INVERSION_FILES)
	$(CC) $(CFLAGS) -MMD -o $@ $^ $(LIBS) $(INCLUDE)

$(OBJ)%.o: $(TST)%.cpp
	$(CC) $(CFLAGS) $(INCLUDE) -MMD -c -o $@ $< 

run_test_inversion:
	$(TEST_INVERSION)

clean:
	rm -f build/*
