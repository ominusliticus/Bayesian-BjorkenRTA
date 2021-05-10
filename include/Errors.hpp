#pragma once
#include <iostream>

#define DEBUG 1
#define PRINT(x) std::cout << x << std::endl;
#define ASSERT_ERROR(x) std::cout << x << std::endl;

#if DEBUG 
#define PRINT_DEBUG(n)  std::cout << n << std::endl;
#endif