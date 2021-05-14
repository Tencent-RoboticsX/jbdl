#include <pybind11/pybind11.h>

#include "tools.hpp"

int multiply(int i, int j)
{
    return i * j;
}

void Pet::go_for_a_walk() { hunger++; }

const std::string &Pet::get_name() const
{
    return name;
}

int Pet::get_hunger() const
{
    return hunger;
}
