import hw1_common;
float segment_distance_squared(float2 p, float2 a, float2 b)
{
    float2 ab = b - a;
    float denom = dot(ab, ab);
    float t = 0.0;
    if (denom > 1e-8)
    {
        t = clamp(dot(p - a, ab) / denom, 0.0, 1.0);
    }
    float2 closest = a + t * ab;
    float2 diff = p - closest;
    return dot(diff, diff);
}

bool point_in_polygon(float2 p,
                      StructuredBuffer<float2> points,
                      int pts_begin,
                      int pts_end)
{
    bool inside = false;

    for (int i = pts_begin, j = pts_end - 1;
         i < pts_end;
         j = i, i++)
    {
        float2 a = points[i];
        float2 b = points[j];
        bool crosses = ((a.y > p.y) != (b.y > p.y));
        if (crosses)
        {
            float x_intersection =
                (b.x - a.x) * (p.y - a.y) /
                (b.y - a.y) + a.x;

            if (p.x < x_intersection)
            {
                inside = !inside;
            }
        }
    }
    return inside;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(
    uint2 tid : SV_DispatchThreadID,
    uniform float3 background,
    uniform int /*samples_per_axis*/,
    StructuredBuffer<Shape> shapes,
    uniform int num_shapes,
    StructuredBuffer<Circle> circles,
    StructuredBuffer<Polyline> polylines,
    StructuredBuffer<QuadraticBezier> /*quadratic_beziers*/,
    StructuredBuffer<float2> points,
    RWTexture2D<float4> image)
{
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height)
    {
        return;
    }
    // Pixel center in screen coordinates.
    // The scene coordinate system has its y-axis pointing upward.
    float2 pixel = float2(
        float(tid.x) + 0.5,
        float(height) - (float(tid.y) + 0.5)
    );
    float4 color = float4(background, 1.0);
    // Process shapes in scene order so later shapes appear on top.
    for (int s = 0; s < num_shapes; s++)
    {
        Shape shape = shapes[s];
        if (shape.type == ShapeType::Circle) {
            Circle circle = circles[shape.index];

            float3 local_h = mul(
                shape.world_to_obj,
                float3(pixel, 1.0)
            );
            float2 local_p = local_h.xy / local_h.z;
            float2 diff = local_p - circle.center;
            float dist = length(diff);
            // Draw the fill if the pixel is inside the circle.
            if (shape.use_fill && dist <= circle.radius)
            {
                color = shape.fill_rgba;
            }
            // Draw the outline if the pixel is close to the circumference.
            if (shape.use_stroke)
            {
                float half_width = shape.stroke_width * 0.5;
                if (abs(dist - circle.radius) <= half_width)
                {
                    color = shape.stroke_rgba;
                }
            }
        }
        else if (shape.type == ShapeType::Polyline)
        {
            Polyline polyline = polylines[shape.index];
            float3 local_h = mul(
                shape.world_to_obj,
                float3(pixel, 1.0)
            );
            float2 p = local_h.xy / local_h.z;
            int n = polyline.pts_end - polyline.pts_begin;
            if (n < 2)
            {
                continue;
            }
            bool inside = false;
            if (shape.use_fill && polyline.closed)
            {
                inside = point_in_polygon(
                    p,
                    points,
                    polyline.pts_begin,
                    polyline.pts_end
                );
                if (inside)
                {
                    color = shape.fill_rgba;
                }
            }
            // Test the distance to each line segment.
            if (shape.use_stroke)
            {
                float min_dist_sq = 1e30;
                for (int i = 0; i < n - 1; i++)
                {
                    float2 a = points[polyline.pts_begin + i];
                    float2 b = points[polyline.pts_begin + i + 1];
                    float d_sq = segment_distance_squared(p, a, b);
                    min_dist_sq = min(min_dist_sq, d_sq);
                }
                // A closed polyline also has a final-to-first segment.
                if (polyline.closed)
                {
                    float2 a = points[polyline.pts_end - 1];
                    float2 b = points[polyline.pts_begin];

                    float d_sq = segment_distance_squared(p, a, b);
                    min_dist_sq = min(min_dist_sq, d_sq);
                }
                float half_width = shape.stroke_width * 0.5;
                if (min_dist_sq <= half_width * half_width)
                {
                    color = shape.stroke_rgba;
                }
            }
        }
        // QuadraticBezier is intentionally not implemented in this part.
    }
    image[tid] = color;
}
