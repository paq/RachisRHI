cbuffer HullResources : register(b0, space0)
{
    float4 hullColor;
    float4 tessellationControl;
};

#ifdef __spirv__
struct DomainPushConstants
{
    float4 domainControl;
};
[[vk::push_constant]] DomainPushConstants domainPush;
#define domainControl domainPush.domainControl
#else
cbuffer PushConstants : register(b0, space4)
{
    float4 domainControl;
};
#endif

struct ControlPoint
{
    float4 position : POSITION;
};

struct PatchConstants
{
    float edges[3] : SV_TessFactor;
    float inside : SV_InsideTessFactor;
    float4 color : COLOR0;
};

struct DomainOutput
{
    float4 position : SV_Position;
    float4 color : COLOR0;
};

ControlPoint tessellationVertex(uint vertexId : SV_VertexID)
{
    const float2 positions[3] = {
        float2(0.0F, 0.75F),
        float2(0.75F, -0.75F),
        float2(-0.75F, -0.75F),
    };
    ControlPoint output;
    output.position = float4(positions[vertexId], 0.0F, 1.0F);
    return output;
}

PatchConstants tessellationFactors(InputPatch<ControlPoint, 3> patch)
{
    PatchConstants output;
    output.edges[0] = tessellationControl.x;
    output.edges[1] = tessellationControl.x;
    output.edges[2] = tessellationControl.x;
    output.inside = tessellationControl.x;
    output.color = hullColor;
    return output;
}

PatchConstants tessellationFactorsStatic(InputPatch<ControlPoint, 3> patch)
{
    PatchConstants output;
    output.edges[0] = 4.0F;
    output.edges[1] = 4.0F;
    output.edges[2] = 4.0F;
    output.inside = 4.0F;
    output.color = float4(0.0F, 1.0F, 0.0F, 1.0F);
    return output;
}

[domain("tri")]
[partitioning("integer")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(3)]
[patchconstantfunc("tessellationFactors")]
ControlPoint tessellationHull(InputPatch<ControlPoint, 3> patch, uint controlPointId : SV_OutputControlPointID)
{
    return patch[controlPointId];
}

[domain("tri")]
[partitioning("integer")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(3)]
[patchconstantfunc("tessellationFactorsStatic")]
ControlPoint tessellationHullStatic(InputPatch<ControlPoint, 3> patch, uint controlPointId : SV_OutputControlPointID)
{
    return patch[controlPointId];
}

[domain("tri")]
DomainOutput tessellationDomain(PatchConstants constants,
                                float3 barycentricCoordinates : SV_DomainLocation,
                                const OutputPatch<ControlPoint, 3> patch)
{
    DomainOutput output;
    const float4 position = patch[0].position * barycentricCoordinates.x
                            + patch[1].position * barycentricCoordinates.y
                            + patch[2].position * barycentricCoordinates.z;
    output.position = float4(position.xy * domainControl.x + domainControl.yz, position.zw);
    output.color = constants.color;
    return output;
}

[domain("tri")]
DomainOutput tessellationDomainStatic(PatchConstants constants,
                                      float3 barycentricCoordinates : SV_DomainLocation,
                                      const OutputPatch<ControlPoint, 3> patch)
{
    DomainOutput output;
    output.position = patch[0].position * barycentricCoordinates.x
                      + patch[1].position * barycentricCoordinates.y
                      + patch[2].position * barycentricCoordinates.z;
    output.color = constants.color;
    return output;
}

float4 tessellationPixel(DomainOutput input) : SV_Target0
{
    return input.color;
}
