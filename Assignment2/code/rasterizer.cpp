// clang-format off
//
// Created by goksu on 4/6/19.
//

#include <algorithm>
#include <vector>
#include "rasterizer.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>


rst::pos_buf_id rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f> &positions)
{
    auto id = get_next_id();
    pos_buf.emplace(id, positions);

    return {id};
}

rst::ind_buf_id rst::rasterizer::load_indices(const std::vector<Eigen::Vector3i> &indices)
{
    auto id = get_next_id();
    ind_buf.emplace(id, indices);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_colors(const std::vector<Eigen::Vector3f> &cols)
{
    auto id = get_next_id();
    col_buf.emplace(id, cols);

    return {id};
}

auto to_vec4(const Eigen::Vector3f& v3, float w = 1.0f)
{
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}


static bool insideTriangle(float x, float y, const Vector3f* _v)
{   
    // TODO : Implement this function to check if the point (x, y) is inside the triangle represented by _v[0], _v[1], _v[2]
    Eigen::Vector3f point(x + 0.5f, y + 0.5f , 1.0f);

    Eigen::Vector3f AP = point - _v[0];
    Eigen::Vector3f BP = point - _v[1];
    Eigen::Vector3f CP = point - _v[2];
    auto AB = _v[1] - _v[0];
    auto CA = _v[0] - _v[2];
    auto BC = _v[2] - _v[1];

    auto res_1 = AB.cross(AP).z();
    auto res_2 = BC.cross(BP).z();
    auto res_3 = CA.cross(CP).z();

    if ((res_1 > 0 && res_2 > 0 && res_3 > 0) || (res_1 < 0 && res_2 < 0 && res_3 < 0)) return true;
    else return false;

}

static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector3f* v)
{
    float c1 = (x*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*y + v[1].x()*v[2].y() - v[2].x()*v[1].y()) / (v[0].x()*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*v[0].y() + v[1].x()*v[2].y() - v[2].x()*v[1].y());
    float c2 = (x*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*y + v[2].x()*v[0].y() - v[0].x()*v[2].y()) / (v[1].x()*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*v[1].y() + v[2].x()*v[0].y() - v[0].x()*v[2].y());
    float c3 = (x*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*y + v[0].x()*v[1].y() - v[1].x()*v[0].y()) / (v[2].x()*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*v[2].y() + v[0].x()*v[1].y() - v[1].x()*v[0].y());
    return {c1,c2,c3};
}

void rst::rasterizer::draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type)
{
    auto& buf = pos_buf[pos_buffer.pos_id];
    auto& ind = ind_buf[ind_buffer.ind_id];
    auto& col = col_buf[col_buffer.col_id];

    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;

    Eigen::Matrix4f mvp = projection * view * model;
    for (auto& i : ind)
    {
        Triangle t;
        Eigen::Vector4f v[] = {
                mvp * to_vec4(buf[i[0]], 1.0f),
                mvp * to_vec4(buf[i[1]], 1.0f),
                mvp * to_vec4(buf[i[2]], 1.0f)
        };
        //Homogeneous division
        for (auto& vec : v) {
            vec /= vec.w();
        }
        //Viewport transformation
        for (auto & vert : v)
        {
            vert.x() = 0.5*width*(vert.x()+1.0);
            vert.y() = 0.5*height*(vert.y()+1.0);
            vert.z() = vert.z() * f1 + f2;
        }

        for (int i = 0; i < 3; ++i)
        {
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
        }

        auto col_x = col[i[0]];
        auto col_y = col[i[1]];
        auto col_z = col[i[2]];

        t.setColor(0, col_x[0], col_x[1], col_x[2]);
        t.setColor(1, col_y[0], col_y[1], col_y[2]);
        t.setColor(2, col_z[0], col_z[1], col_z[2]);

        rasterize_triangle(t);
    }

    //downsampling
    for(int y = 0; y < height; y ++) {
        for(int x = 0; x < width; x++) {
            Eigen::Vector3f color{0,0,0};
            float infinity = std::numeric_limits<float>::infinity();

            for(float j = start_point; j < 1.0; j += pixel_size_sm) {
                for(float i = start_point; i < 1.0; i += pixel_size_sm) {
                    int index = get_index_ssaa(x,y,i,j);
                    color += frame_buf_ssaa[index];
                }
            }

            Eigen::Vector3f p;
            p << x , y , 0;
            set_pixel(p , color/(ssaa_h*ssaa_w));
        }
    }
}

//Screen space rasterization
void rst::rasterizer::rasterize_triangle(const Triangle& t) {
    auto v = t.toVector4();
    
    // TODO : Find out the bounding box of current triangle.
    float x_min = t.v[0].x();
    float x_max = t.v[0].x();
    float y_min = t.v[0].y();
    float y_max = t.v[0].y();

    for(auto point : t.v) {
        x_min = x_min > point.x() ? point.x() : x_min;
        x_max = x_max < point.x() ? point.x() : x_max;
        y_min = y_min > point.y() ? point.y() : y_min;
        y_max = y_max < point.y() ? point.y() : y_max;
    }

    // iterate through the pixel and find if the current pixel is inside the triangle
    for(int x = (int)x_min; x < (int)x_max; x++) {
        for (int y = (int)y_min; y < (int)y_max; y++) {
            int index = get_index(x,y);
            float cnt = 0.0;
            float max_cnt = ssaa_h * ssaa_w;
            
            //iterator the downsampling points
            for(float i = start_point; i < 1.0; i+= pixel_size_sm) {
                for(float j = start_point; j < 1.0; j += pixel_size_sm) {
                    if(insideTriangle(x+i, y+j, t.v)) {
                        cnt += 1.0;
                    }
                }
            }

            //2. SSAA
            for(float j = start_point; j < 1.0; j += pixel_size_sm) {
                for(float i = start_point; i <1.0; i += pixel_size_sm) {
                    if(insideTriangle(x+i, y+j, t.v)) {
                        // If so, use the following code to get the interpolated z value.
                        auto[alpha, beta, gamma] = computeBarycentric2D(x+i, y+j, t.v);
                        float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                        float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                        z_interpolated *= w_reciprocal;

                        // TODO : set the current pixel (use the set_pixel function) 
                        // to the color of the triangle (use getColor function) if it should be painted.
                        int index = get_index_ssaa(x,y,i,j);
                        if( z_interpolated < depth_buf_ssaa[index]) {
                            frame_buf_ssaa[index] = t.getColor();
                            depth_buf_ssaa[index] = z_interpolated;
                        }
                    }
                }
            }


            //3.basic method
            /*if(insideTriangle(x,y,t.v)) {
                // If so, use the following code to get the interpolated z value.
                auto[alpha, beta, gamma] = computeBarycentric2D(x, y, t.v);
                float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                z_interpolated *= w_reciprocal;

                // Question: how to get the buf_index and the color ?????
                // TODO : set the current pixel (use the set_pixel function) to the color of the triangle (use getColor function) if it should be painted.
                int buf_index = get_index(x,y);

                if(z_interpolated >= depth_buf[buf_index]) continue;
                depth_buf[buf_index] = z_interpolated;
                Vector3f color = t.getColor();
                Vector3f point(x,y,depth_buf[buf_index]);

                set_pixel(point,color);
            }*/
        }
    }
}

void rst::rasterizer::set_model(const Eigen::Matrix4f& m)
{
    model = m;
}

void rst::rasterizer::set_view(const Eigen::Matrix4f& v)
{
    view = v;
}

void rst::rasterizer::set_projection(const Eigen::Matrix4f& p)
{
    projection = p;
}

void rst::rasterizer::clear(rst::Buffers buff)
{
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color)
    {
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});
        std::fill(frame_buf_ssaa.begin(), frame_buf_ssaa.end(), Eigen::Vector3f{0, 0, 0});
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth)
    {
        std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());
        std::fill(depth_buf_ssaa.begin(), depth_buf_ssaa.end(), std::numeric_limits<float>::infinity());
    }
}

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h)
{
    frame_buf.resize(w * h);
    depth_buf.resize(w * h);
    frame_buf_ssaa.resize(w * ssaa_w * h * ssaa_h);
    depth_buf_ssaa.resize(w * ssaa_w * h * ssaa_h);
}

int rst::rasterizer::get_index_ssaa(int x, int y, float i, float j)
{
    int ssaa_height = height * ssaa_h;
    int ssaa_width = width * ssaa_w;

    i = int((i - start_point) / pixel_size_sm);
    j = int((j - start_point) / pixel_size_sm);

    return (ssaa_height-1-y*ssaa_h+j) * ssaa_width + x*ssaa_w + i;
}
int rst::rasterizer::get_index(int x, int y)
{
    return (height-1-y)*width + x;
}

void rst::rasterizer::set_pixel(const Eigen::Vector3f& point, const Eigen::Vector3f& color)
{
    //old index: auto ind = point.y() + point.x() * width;
    auto ind = (height-1-point.y())*width + point.x();
    frame_buf[ind] = color;

}

// clang-format on