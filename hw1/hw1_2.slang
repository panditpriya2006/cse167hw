import hw1_common;

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  StructuredBuffer<float2> polyline,
                  uniform int num_polyline,
                  uniform bool is_closed,
                  uniform bool use_fill_color,
                  uniform float3 fill_color,
                  uniform bool use_stroke_color,
                  uniform float3 stroke_color,
                  uniform float stroke_width,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    image[tid] = float4(1.0);
    // TODO: your code here
    
    //pixel center
    float2 p = float2(tid.x + 0.5, height - tid.y - 0.5);
    
    //default bg color
    float3 color = float3(0.5, 0.5, 0.5);

    //fill logic (non-zero rule)
    bool is_filled = false;
    if (is_closed && use_fill_color && num_polyline >= 3) {
        int winding_count = 0;

        for (int i = 0; i < num_polyline; i++) {
            float2 p0 = polyline[i];
            float2 p1 = polyline[(i+1) % num_polyline];
            
            if (p0.y <= p.y) {
                if (p1.y > p.y) {
                    //upwards edge
                    float t = (p.y - p0.y) / (p1.y - p0.y);
                    float x = p0.x + t * (p1.x - p0.x);
                    if (p.x < x) {
                        winding_count++;
                    }
                }
            }
            else {
                if (p1.y <= p.y) {
                    float t = (p.y - p0.y) / (p1.y - p0.y);
                    float x = p0.x + t * (p1.x - p0.x);
                    if (p.x < x) {
                        winding_count--;
                    }
                }
            }
        }

        if (winding_count != 0) {
            is_filled = true;
            color = fill_color;
        }
	}

    //stroke logic
    if (use_stroke_color && num_polyline >= 2) {
        float half_width = stroke_width / 2;
        float half_width_sq = half_width * half_width;
        bool in_stroke = false;

        //determine num of segments
        int num_segments = is_closed ? num_polyline : (num_polyline - 1);

        //flat caps
        for (int i = 0; i < num_segments; i++) {
            float2 p0 = polyline[i];
            float2 p1 = polyline[(i+1) % num_polyline];

            float2 d = p1 - p0;
            float len_sq = dot(d, d);

            if (len_sq > 0.0) {
                // Projection parameter t (from 0 to 1 along the segment)
                float t = dot(p - p0, d) / len_sq;

                // Point must fall strictly within segment boundaries for a flat cap
                if (t >= 0.0 && t <= 1.0) {
                    float2 q_prime = p0 + t * d;
                    float dist_sq = dot(p - q_prime, p - q_prime);
                    if (dist_sq <= half_width_sq) {
                        in_stroke = true;
                        break;
                    }
                }
            }
        }

        //round joins
        if (!in_stroke) {
            // If closed: all vertices are joins.
            // If open: only interior vertices (1 through num_polyline - 2) are joins.
            int start_idx = is_closed ? 0 : 1;
            int end_idx   = is_closed ? num_polyline : (num_polyline - 1);

            for (int i = start_idx; i < end_idx; ++i) {
                float2 v = polyline[i];
                float dist_sq = dot(p - v, p - v);
                if (dist_sq <= half_width_sq) {
                    in_stroke = true;
                    break;
                }
            }
        }

        if (in_stroke) {
            color = stroke_color;
        }
        
    }

    // output final RGBA pixel color
    image[tid] = float4(color, 1.0);
    
}
