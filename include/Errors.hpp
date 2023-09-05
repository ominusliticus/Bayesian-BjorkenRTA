//  Copyright 2021-2022 Kevin Ingles
//
//  Permission is hereby granted, free of charge, to any person obtaining
//  a copy of this software and associated documentation files (the
//  "Software"), to deal in the Software without restriction, including
//  without limitation the right to use, copy, modify, merge, publish,
//  distribute, sublicense, and/or sell copies of the Software, and to
//  permit persons to whom the Sofware is furnished to do so, subject to
//  the following conditions:
//
//  The above copyright notice and this permission notice shall be
//  included in all copies or substantial poritions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
//  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
//  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
//  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
//  CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
//  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
//  SOFTWARE OR THE USE OF OTHER DEALINGS IN THE SOFTWARE
//
// Author: Kevin Ingles
// File: Errors.hpp
// Description: Contains standard outputting rountines

#ifndef ERRORS_HPP
#define ERRORS_HPP
#include <fmt/core.h>
#include <fmt/format.h>
#include <iostream>

#define get_var_name(x) #x

// TODO: switch to using fmt library...
template <typename... Args>
void Print(std::ostream& out, Args&&... args)
{
    // Uses C++17 Folding Expressions
    ((out << std::forward<Args>(args) << '\t'), ...);
    out << '\n';
}

template <typename... Args>
void Print_Error(std::ostream& out, Args&&... args)
{
    // Uses C++17 Folding Expressions
    ((out << std::forward<Args>(args) << '\t'), ...);
    out << '\n';
}

#define DEBUG 0
#if DEBUG
#  define PRINT_DEBUG(n) std::cout << n << std::endl;
#else
#  define PRINT_DEBUG(x)
#endif

#endif
