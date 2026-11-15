# RHI shader fixtures

`BootstrapTriangle.slang` and `BootstrapCube.slang` are RHI-owned test inputs.
Their entry points and resource bindings preserve the pipeline, shader-module,
and compute contracts exercised by the RHI tests.

`LayoutFixture.slang` is the RHI reflection and binding-layout test input.
The HLSL workload files in this directory are used only by RHI tests.
