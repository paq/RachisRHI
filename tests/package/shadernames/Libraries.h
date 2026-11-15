#pragma once

#include <Rachis/Device.h>

#include <cstddef>
#include <span>

std::span<const std::byte> firstFillShader(rachis::Backend backend);
std::span<const std::byte> secondFillShader(rachis::Backend backend);
