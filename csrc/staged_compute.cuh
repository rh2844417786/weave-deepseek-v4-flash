#pragma once
#include <pybind11/pybind11.h>

// Registration is called by the single extension module in megakernel.cu.
void bind_staged(pybind11::module_& module);
