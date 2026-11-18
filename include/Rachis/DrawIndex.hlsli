#ifndef RACHIS_DRAW_INDEX_HLSLI
#define RACHIS_DRAW_INDEX_HLSLI

#if defined(__spirv__) && defined(__SLANG__)

uint rachisDrawIndex()
{
    return spirv_asm
    {
        OpCapability DrawParameters;
        result:$$uint = OpLoad builtin(DrawIndex:uint);
    };
}

#elif defined(__spirv__)

[[vk::ext_builtin_input(4426)]] static const uint rachisBuiltinDrawIndex;

uint rachisDrawIndex()
{
    return rachisBuiltinDrawIndex;
}

#else

cbuffer RachisDrawIndex : register(b0, space5)
{
    uint rachisDrawIndexValue;
}

uint rachisDrawIndex()
{
    return rachisDrawIndexValue;
}

#endif

#endif
