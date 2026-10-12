import hw1_common;

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float3 background,
                  uniform int samples_per_axis,
                  StructuredBuffer<Shape> shapes,
                  uniform int num_shapes,
                  StructuredBuffer<Circle> circles,
                  StructuredBuffer<Polyline> polylines,
                  StructuredBuffer<QuadraticBezier> /*quadratic_beziers*/, // ignore for now
                  StructuredBuffer<float2> points,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    image[tid] = float4(1.0);
    // TODO: your code here
    int N = samples_per_axis;
    float3 accumulated_color = float3(0.0, 0.0, 0.0);

    // Loop over the N x N subpixel grid
    for (int sy = 0; sy < N; ++sy) {
        for (int sx = 0; sx < N; ++sx) {
            // Compute subpixel center in world space
            float sub_x = float(tid.x) + (float(sx) + 0.5) / float(N);
            float sub_y = float(height - 1 - tid.y) + (float(sy) + 0.5) / float(N);
            float2 p_world = float2(sub_x, sub_y);

            float3 sample_color = background;

            // Evaluate shapes in back-to-front order
            for (int s = num_shapes - 1; s >= 0; --s) {
                Shape shape = shapes[s];

                // Transform subpixel center into object space
                float3 p_obj_h = mul(shape.world_to_obj, float3(p_world, 1.0));
                float2 p = p_obj_h.xy;

                if (shape.type == ShapeType.Circle) {
                    Circle c = circles[shape.index];

                    // Circle Fill
                    if (shape.use_fill) {
                        float dist_sq = dot(p - c.center, p - c.center);
                        if (dist_sq <= c.radius * c.radius) {
                            sample_color = shape.fill_rgba.xyz;
                        }
                    }

                    // Circle Stroke
                    if (shape.use_stroke) {
                        float dist = length(p - c.center);
                        float half_w = shape.stroke_width * 0.5;
                        if (abs(dist - c.radius) <= half_w) {
                            sample_color = shape.stroke_rgba.xyz;
                        }
                    }
                } else if (shape.type == ShapeType.Polyline) {
                    Polyline poly = polylines[shape.index];
                    int num_poly_pts = poly.pts_end - poly.pts_begin;

                    // Polyline Fill (Non-zero winding number)
                    if (poly.closed && shape.use_fill && num_poly_pts >= 3) {
                        int winding_count = 0;
                        for (int i = 0; i < num_poly_pts; ++i) {
                            float2 p0 = points[poly.pts_begin + i];
                            float2 p1 = points[poly.pts_begin + ((i + 1) % num_poly_pts)];

                            if (p0.y <= p.y) {
                                if (p1.y > p.y) {
                                    float t = (p.y - p0.y) / (p1.y - p0.y);
                                    float x_int = p0.x + t * (p1.x - p0.x);
                                    if (p.x < x_int) {
                                        winding_count++;
                                    }
                                }
                            } else {
                                if (p1.y <= p.y) {
                                    float t = (p.y - p0.y) / (p1.y - p0.y);
                                    float x_int = p0.x + t * (p1.x - p0.x);
                                    if (p.x < x_int) {
                                        winding_count--;
                                    }
                                }
                            }
                        }
                        if (winding_count != 0) {
                            sample_color = shape.fill_rgba.xyz;
                        }
                    }

                    // Polyline Stroke (Flat caps + Round joins)
                    if (shape.use_stroke && num_poly_pts >= 2) {
                        float half_w = shape.stroke_width * 0.5;
                        float half_w_sq = half_w * half_w;
                        bool in_stroke = false;

                        int num_segments = poly.closed ? num_poly_pts : (num_poly_pts - 1);

                        // Line bodies (flat caps)
                        for (int i = 0; i < num_segments; ++i) {
                            float2 p0 = points[poly.pts_begin + i];
                            float2 p1 = points[poly.pts_begin + ((i + 1) % num_poly_pts)];

                            float2 d = p1 - p0;
                            float len_sq = dot(d, d);
                            if (len_sq > 0.0) {
                                float t = dot(p - p0, d) / len_sq;
                                if (t >= 0.0 && t <= 1.0) {
                                    float2 q_proj = p0 + t * d;
                                    if (dot(p - q_proj, p - q_proj) <= half_w_sq) {
                                        in_stroke = true;
                                        break;
                                    }
                                }
                            }
                        }

                        // Round joins
                        if (!in_stroke) {
                            int start_idx = poly.closed ? 0 : 1;
                            int end_idx   = poly.closed ? num_poly_pts : (num_poly_pts - 1);

                            for (int i = start_idx; i < end_idx; ++i) {
                                float2 v = points[poly.pts_begin + i];
                                if (dot(p - v, p - v) <= half_w_sq) {
                                    in_stroke = true;
                                    break;
                                }
                            }
                        }

                        if (in_stroke) {
                            sample_color = shape.stroke_rgba.xyz;
                        }
                    }
                }
            }

            accumulated_color += sample_color;
        }
    }

    // Average over all N x N samples
    float3 final_color = accumulated_color / float(N * N);
    image[tid] = float4(final_color, 1.0);
}
