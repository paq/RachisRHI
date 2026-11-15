#version 460

layout(location = 0) in vec4 color;
layout(location = 1) in vec2 uv;

layout(location = 0) out vec4 target;

void main()
{
    target = color * vec4(uv, 1.0, 1.0);
}
