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
// File: Integration.hpp
// Description: Header file implementation of a templated general purpose integration rountine
//              Integration use adaptive 48-point Gaus-Legendre integration method and 
//              variadic templates to handle arbitrary function calls


#ifndef INTEGRATION_HPP
#define INTEGRATION_HPP
#include "Errors.hpp"

constexpr double inf = std::numeric_limits<double>::infinity();

// Choice to generalize code to calculate number of points to use
constexpr int NSUM48 = 24;


// constexpr keyword is to ensure there are not defined in multiple translation units
constexpr double x48[NSUM48] = { 0.0323801709628694,
                    0.0970046992094627,
                    0.1612223560688917,
                    0.2247637903946891,
                    0.2873624873554556,
                    0.3487558862921608,
                    0.4086864819907167,
                    0.4669029047509584,
                    0.5231609747222330,
                    0.5772247260839727,
                    0.6288673967765136,
                    0.6778723796326639,
                    0.7240341309238146,
                    0.7671590325157404,
                    0.8070662040294426,
                    0.8435882616243935,
                    0.8765720202742479,
                    0.9058791367155696,
                    0.9313866907065543,
                    0.9529877031604309,
                    0.9705915925462473,
                    0.9841245837228269,
                    0.9935301722663508,
                    0.9987710072524261 };

constexpr double w48[NSUM48] = { 0.0647376968126839,
                    0.0644661644359501,
                    0.0639242385846482,
                    0.0631141922862540,
                    0.0620394231598927,
                    0.0607044391658939,
                    0.0591148396983956,
                    0.0572772921004032,
                    0.0551995036999842,
                    0.0528901894851937,
                    0.0503590355538545,
                    0.0476166584924905,
                    0.0446745608566943,
                    0.0415450829434647,
                    0.0382413510658307,
                    0.0347772225647704,
                    0.0311672278327981,
                    0.0274265097083569,
                    0.0235707608393244,
                    0.0196161604573555,
                    0.0155793157229438,
                    0.0114772345792345,
                    0.0073275539012763,
                    0.0031533460523058 };


///////////////////////////////////////////////////////
//     Prototyping so the functions see each other   //
///////////////////////////////////////////////////////
template<typename Functor, typename...Args>
double GausQuadAux(Functor&& func, double _low, double _high, double result, double tol, int depth, bool improper_top, Args&&... args);

template<typename Functor, typename...Args>
double GausQuad(Functor&& func, double _low, double _high, double tol, int maxDepth, Args&&... args);

///////////////////////////////////////////////////////
//              Defining implementation              //
///////////////////////////////////////////////////////
template<typename Functor, typename...Args>
double GausQuadAux(Functor&& func, double _low, double _high, double result, double tol, int depth, bool improper_top, Args&&... args)
{	
    // Quick check to ensure that we should do calculation
    if (depth <= 0)
        return result;

    double high = _high;
    double low  = _low;

    double middle = (high + low) / 2.0;
    double interval1_result = 0;
    double interval2_result = 0;

    double yneg, ypos;
    for (int i = 0; i < NSUM48; i++)
    {
        // Sum up areas using above weights and points
        yneg = ((middle - low) * (-x48[i]) + (middle + low)) / 2.0;
        ypos = ((middle - low) * (x48[i]) + (middle + low)) / 2.0;
        if (!improper_top) 
            interval1_result += w48[i] * func(yneg, std::forward<Args>(args)...) + w48[i] * func(ypos, std::forward<Args>(args)...);
        else
            interval1_result += w48[i] * func(1 / yneg, std::forward<Args>(args)...) / pow(yneg, 2.0) + w48[i] * func(1 / ypos, std::forward<Args>(args)...) / pow(ypos, 2.0);
     
        // Sum up areas using above weights and points
        yneg = ((high - middle) * (-x48[i]) + (high + middle)) / 2.0;
        ypos = ((high - middle) * (x48[i]) + (high + middle)) / 2.0;
        if (!improper_top) 
            interval2_result += w48[i] * func(yneg, std::forward<Args>(args)...) + w48[i] * func(ypos, std::forward<Args>(args)...);
        else
            interval2_result += w48[i] * func(1 / yneg, std::forward<Args>(args)...) / pow(yneg, 2.0) + w48[i] * func(1 / ypos, std::forward<Args>(args)...) / pow(ypos, 2.0);
    }
    interval1_result *= (middle - low) / 2.0;
    interval2_result *= (high - middle) / 2.0;

    double result2 = interval1_result + interval2_result;

    if (fabs(result - result2) / result < tol)
        return result;
    else 
        return GausQuadAux(func, low, middle, interval1_result, tol, depth-1, improper_top, std::forward<Args>(args)...) + GausQuadAux(func, middle, high, interval2_result, tol, depth-1, improper_top, std::forward<Args>(args)...);
}

template<typename Functor, typename...Args>
double GausQuad(Functor&& func, double _low, double _high, double tol, int maxDepth, Args&&... args)
{			  
    double result = 0;
	double yneg, ypos;
    double high = _high;
    double low  = _low;

    bool improper_top    = false;
    if (high == inf || low == -inf)
    {
        if (high == inf && low != -inf)  
        {
            if (low == 0) 
            {
                result += GausQuad(func, 0, 1, tol, maxDepth, std::forward<Args>(args)...);
                result += GausQuad(func, 1, high, tol, maxDepth, std::forward<Args>(args)...);
                return result;
            }
            else high = 1 / low;
            
            low = 0;
            improper_top = true;
        }
        else if (high != inf && low == -inf) 
        {
           if (high == 0) 
           {
                result += GausQuad(func, -1, 0, tol, maxDepth, std::forward<Args>(args)...);
                result += GausQuad(func, low, -1, tol, maxDepth, std::forward<Args>(args)...);
                return result;
           }
            else low = 1 / high;
            
            high = 0;
            improper_top = true;
        }
        else 
        {
            result += GausQuad(func, low, -1, tol, maxDepth, std::forward<Args>(args)...);
            result += GausQuad(func, -1, 1, tol, maxDepth, std::forward<Args>(args)...);
            result += GausQuad(func, 1, high, tol, maxDepth, std::forward<Args>(args)...);
            return result;
        }
    }
	
    for (int i = 0; i < NSUM48; i++)
    {
        // Sum up areas using above weights and points
        yneg = ((high - low)*(-x48[i]) + (high + low))/2.0;
        ypos = ((high - low)*(x48[i]) + (high + low))/2.0;
        if (!improper_top) 
        {
            result += w48[i] * func(yneg, std::forward<Args>(args)...) + w48[i] * func(ypos, std::forward<Args>(args)...);
        }
        else
        {
            result += w48[i] * func(1 / yneg, std::forward<Args>(args)...) / pow(yneg, 2.0) + w48[i] * func(1 / ypos, std::forward<Args>(args)...) / pow(ypos, 2.0);
        }
    }
    result *= (high - low)/2.0;

    return GausQuadAux(func, low, high, result, tol, maxDepth, improper_top, std::forward<Args>(args)...);
}

#endif
