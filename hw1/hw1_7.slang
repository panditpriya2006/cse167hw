
import hw1_common;

// Squared distance from a point to a line segment.
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

// Test whether a point is inside a closed polygon using ray casting.
bool point_in_polygon(
    float2 p,
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

// Composite a foreground RGBA color over a background RGB color.
float3 alpha_blend(float3 behind, float4 foreground)
{
    return foreground.rgb * foreground.a
         + behind * (1.0 - foreground.a);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(
    uint2 tid : SV_DispatchThreadID,
    uniform float3 background,
    uniform int samples_per_axis,
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

    // Ensure that at least one sample is taken per pixel.
    int spp = max(samples_per_axis, 1);
    float3 pixel_sum = float3(0.0, 0.0, 0.0);

    // Sample multiple positions inside each pixel.
    for (int sy = 0; sy < spp; sy++)
    {
        for (int sx = 0; sx < spp; sx++)
        {
            float2 pixel = float2(
                float(tid.x) + (float(sx) + 0.5) / float(spp),
                float(height) -
                    (float(tid.y) + (float(sy) + 0.5) / float(spp))
            );

            // Each sample starts with the background color.
            float3 color = background;

            // Process shapes in scene order.
            for (int s = 0; s < num_shapes; s++)
            {
                Shape shape = shapes[s];

                // Transform the sample into object coordinates.
                float3 local_h = mul(
                    shape.world_to_obj,
                    float3(pixel, 1.0)
                );

                float2 p = local_h.xy / local_h.z;

                if (shape.type == ShapeType::Circle)
                {
                    Circle circle = circles[shape.index];

                    float dist = length(p - circle.center);

                    // Fill and stroke are separate compositing layers.
                    if (shape.use_fill && dist <= circle.radius)
                    {
                        color = alpha_blend(color, shape.fill_rgba);
                    }

                    if (shape.use_stroke)
                    {
                        float half_width = shape.stroke_width * 0.5;

                        if (abs(dist - circle.radius) <= half_width)
                        {
                            color = alpha_blend(
                                color,
                                shape.stroke_rgba
                            );
                        }
                    }
                }
                else if (shape.type == ShapeType::Polyline)
                {
                    Polyline polyline = polylines[shape.index];

                    int n = polyline.pts_end - polyline.pts_begin;

                    if (n < 2)
                    {
                        continue;
                    }

                    // Composite the polygon fill first.
                    if (shape.use_fill && polyline.closed)
                    {
                        bool inside = point_in_polygon(
                            p,
                            points,
                            polyline.pts_begin,
                            polyline.pts_end
                        );

                        if (inside)
                        {
                            color = alpha_blend(
                                color,
                                shape.fill_rgba
                            );
                        }
                    }

                    // Find the closest edge for the stroke.
                    if (shape.use_stroke)
                    {
                        float min_dist_sq = 1e30;

                        for (int i = 0; i < n - 1; i++)
                        {
                            float2 a =
                                points[polyline.pts_begin + i];
                            float2 b =
                                points[polyline.pts_begin + i + 1];

                            float d_sq =
                                segment_distance_squared(p, a, b);

                            min_dist_sq = min(min_dist_sq, d_sq);
                        }

                        // Include the closing edge for closed polygons.
                        if (polyline.closed)
                        {
                            float2 a = points[polyline.pts_end - 1];
                            float2 b = points[polyline.pts_begin];

                            float d_sq =
                                segment_distance_squared(p, a, b);

                            min_dist_sq = min(min_dist_sq, d_sq);
                        }
                        float half_width = shape.stroke_width * 0.5;
                        if (min_dist_sq <= half_width * half_width)
                        {
                            color = alpha_blend(
                                color,
                                shape.stroke_rgba
                            );
                        }
                    }
                }
                // Quadratic Bézier curves are not implemented here.
            }
            // Accumulate this subpixel's final color.
            pixel_sum += color;
        }
    }
    // Average all subpixel colors to obtain the final pixel color.
    float sample_count = float(spp * spp);
    float3 final_color = pixel_sum / sample_count;
    image[tid] = float4(final_color, 1.0);
}
