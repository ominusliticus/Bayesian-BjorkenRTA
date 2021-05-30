//
// Author: Kevin Ingles

#ifndef ERRORS_HPP
#define ERRORS_HPP
#include <iostream>
// #include <fmt/core.h>
// #include <fmt/format.h>

// TODO: switch to using fmt library...
template<typename... Args>
void Print(std::ostream& out, Args&&... args)
{
    // Uses C++17 Folding Expressions
    ((out << std::forward<Args>(args) << '\t'), ...);
    out << '\n';
}

template<typename... Args>
void Print_Error(std::ostream& out, Args&&... args)
{
    // Uses C++17 Folding Expressions
    ((out << std::forward<Args>(args) << '\t'), ...);
    out << '\n';
}


#define DEBUG 0
#if DEBUG 
#define PRINT_DEBUG(n)  std::cout << n << std::endl;
#else
#define PRINT_DEBUG(x) 
#endif

#endif