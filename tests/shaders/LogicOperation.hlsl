#ifdef __spirv__
struct LogicPushConstants
{
    uint source;
};
[[vk::push_constant]] LogicPushConstants logicPush;
#define source logicPush.source
#else
cbuffer PushConstants : register(b0, space4)
{
    uint source;
};
#endif

uint pixelMain() : SV_Target0
{
    return source;
}
