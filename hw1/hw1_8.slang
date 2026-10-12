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
                  StructuredBuffer<QuadraticBezier> quadratic_beziers,
                  StructuredBuffer<float2> points,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    image[tid] = float4(1.0);
    
    int N = samples_per_axis;
    float3 accumulated_color = float3(0.0, 0.0, 0.0);

    for (int sy = 0; sy < N; ++sy) {
        for (int sx = 0; sx < N; ++sx) {
            float sub_x = float(tid.x) + (float(sx) + 0.5) / float(N);
            float sub_y = float(height - 1 - tid.y) + (float(sy) + 0.5) / float(N);
            float2 p_world = float2(sub_x, sub_y);

            float3 sample_color = background;

            for (int s = num_shapes - 1; s >= 0; --s) {
                Shape shape = shapes[s];
                float3 p_obj_h = mul(shape.world_to_obj, float3(p_world, 1.0));
                float2 p = p_obj_h.xy;

                if (shape.type == ShapeType.Circle) {
                    Circle c = circles[shape.index];
                    if (shape.use_fill) {
                        float dist_sq = dot(p - c.center, p - c.center);
                        if (dist_sq <= c.radius * c.radius) {
                            sample_color = shape.fill_rgba.xyz;
                        }
                    }
                    if (shape.use_stroke) {
                        float dist = length(p - c.center);
                        float half_w = shape.stroke_width * 0.5;
                        if (abs(dist - c.radius) <= half_w) {
                            sample_color = shape.stroke_rgba.xyz;
                        }
                    }
                } else if (shape.type == ShapeType.QuadraticBezier) {
                    QuadraticBezier bez = quadratic_beziers[shape.index];
                    int total_pts = bez.pts_end - bez.pts_begin;
                    int num_curves = bez.closed ? (total_pts / 2) : ((total_pts - 1) / 2);

                    // 1. Bezier Fill Check (Non-zero winding number)
                    if (bez.closed && shape.use_fill && num_curves >= 1) {
                        int winding_count = 0;
                        for (int i = 0; i < num_curves; ++i) {
                            int idx0 = bez.pts_begin + 2 * i;
                            int idx1 = idx0 + 1;
                            int idx2 = bez.pts_begin + ((2 * i + 2) % total_pts);

                            float2 p0 = points[idx0];
                            float2 p1 = points[idx1];
                            float2 p2 = points[idx2];

                            float2 A = p0 - 2.0 * p1 + p2;
                            float2 B = 2.0 * (p1 - p0);
                            float2 C = p0;

                            float coeff_a = A.y;
                            float coeff_b = B.y;
                            float coeff_c = C.y - p.y;

                            float roots_arr[2];
                            int valid_roots = 0;

                            // Handle linear vs quadratic cases robustly
                            if (abs(coeff_a) < 1e-6) {
                                if (abs(coeff_b) > 1e-6) {
                                    roots_arr[0] = -coeff_c / coeff_b;
                                    valid_roots = 1;
                                }
                            } else {
                                float2 q_roots;
                                if (solve_quadratic(coeff_a, coeff_b, coeff_c, q_roots)) {
                                    roots_arr[0] = q_roots.x;
                                    roots_arr[1] = q_roots.y;
                                    valid_roots = 2;
                                }
                            }

                            for (int r = 0; r < valid_roots; ++r) {
                                float t = roots_arr[r];
                                // Use half-open interval [0.0, 1.0) to prevent double counting shared endpoints
                                if (t >= 0.0 && t < 1.0) {
                                    float x_int = A.x * t * t + B.x * t + C.x;
                                    if (p.x < x_int) {
                                        float dy = 2.0 * A.y * t + B.y;
                                        if (dy > 0.0) {
                                            winding_count++;
                                        } else if (dy < 0.0) {
                                            winding_count--;
                                        }
                                    }
                                }
                            }
                        }
                        if (winding_count != 0) {
                            sample_color = shape.fill_rgba.xyz;
                        }
                    }

                    // 2. Bezier Stroke Check (Flat caps + Round joins)
                    if (shape.use_stroke && num_curves >= 1) {
                        float half_w = shape.stroke_width * 0.5;
                        float half_w_sq = half_w * half_w;
                        bool in_stroke = false;

                        for (int i = 0; i < num_curves; ++i) {
                            int idx0 = bez.pts_begin + 2 * i;
                            int idx1 = idx0 + 1;
                            int idx2 = bez.pts_begin + ((2 * i + 2) % total_pts);

                            float2 p0 = points[idx0];
                            float2 p1 = points[idx1];
                            float2 p2 = points[idx2];

                            float2 A = p0 - 2.0 * p1 + p2;
                            float2 B = 2.0 * (p1 - p0);
                            float2 D = p0 - p;

                            float cubic_a = 2.0 * dot(A, A);
                            float cubic_b = 3.0 * dot(A, B);
                            float cubic_c = 2.0 * dot(A, D) + dot(B, B);
                            float cubic_d = dot(D, B);

                            float3 roots;
                            int num_roots = solve_cubic(cubic_a, cubic_b, cubic_c, cubic_d, roots);

                            for (int r = 0; r < num_roots; ++r) {
                                float t = roots[r];
                                if (t >= 0.0 && t <= 1.0) {
                                    float2 q_t = A * t * t + B * t + p0;
                                    if (dot(p - q_t, p - q_t) <= half_w_sq) {
                                        in_stroke = true;
                                        break;
                                    }
                                }
                            }
                            if (in_stroke) break;
                        }

                        // Round joins at shared endpoints
                        if (!in_stroke) {
                            int start_join = bez.closed ? 0 : 1;
                            int end_join = bez.closed ? num_curves : (num_curves - 1);
                            for (int i = start_join; i <= end_join; ++i) {
                                int pt_idx = bez.pts_begin + ((2 * i) % total_pts);
                                float2 v = points[pt_idx];
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

    float3 final_color = accumulated_color / float(N * N);
    image[tid] = float4(final_color, 1.0);
}
