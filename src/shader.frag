#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in float height;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 low   = vec3(0.0, 0.2, 0.0);
    vec3 high  = vec3(0.7, 1.0, 0.7);
    vec3 color = mix(low, high, height);

    outColor = vec4(color, 1.0);
}
