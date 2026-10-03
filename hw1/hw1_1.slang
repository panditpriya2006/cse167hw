[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float2 center,
                  uniform float radius,
                  uniform float3 color,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    image[tid] = float4(1.0);
    // TODO: your code here
    float2 cord = float2(tid.x + 0.5, height-tid.y-0.5);
    float dist = length(center - cord);
    if(dist <= radius){
        image[tid] = float4(color, 1.0);
    }
    else{
        image[tid] = float4(float3(0.5, 0.5, 0.5), 1.0);
    }
}
