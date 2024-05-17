#ifndef CYLINDER_STITCHER_TEXTSOURCEMANAGER_H
#define CYLINDER_STITCHER_TEXTSOURCEMANAGER_H

#include <ft2build.h>
#include FT_FREETYPE_H
#include "freetype/freetype.h"
#include <QByteArray>
#include <QImage>
#include <QOpenGLTexture>
#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <string>

/**
 * 结构体myCharacter用于表示一个字符的属性，包括在纹理图中的位置、尺寸、轴承点、进位等信息。
 * 
 * @param _top 字符在纹理图中的顶部坐标。
 * @param _left 字符在纹理图中的左侧坐标。
 * @param _width 字符的宽度。
 * @param _height 字符的高度。
 * @param _bearing_x 字符的轴承点（即字符底部中心点）到纹理图左侧的偏移量。
 * @param _bearing_y 字符的轴承点（即字符底部中心点）到纹理图顶部的偏移量。
 * @param _advance 字符的进位，即这个字符宽度加上它与其下一个字符中心点之间的间距。
 * @param _tot_width 纹理图的总宽度。
 * @param _tot_height 纹理图的总高度。
 */
struct myCharacter {
    myCharacter(int _top, int _left, int _width, int _height, int _bearing_x,
                int _bearing_y, int _advance, int _tot_width, int _tot_height)
        : top(_top), left(_left), rel_top((float)top / (float)_tot_height),
          rel_left((float)left / (float)_tot_width), width(_width),
          height(_height), rel_width((float)width / (float)_tot_width),
          rel_height((float)height / (float)_tot_height), bearing_x(_bearing_x),
          bearing_y(_bearing_y), advance(_advance) { }

    int top, left;   // 在纹理图中的位置
    float rel_top, rel_left; // 相对于纹理图总高度和宽度的相对位置

    int width, height; // 字符的宽度和高度
    float rel_width, rel_height; // 相对于纹理图总宽度和高度的相对尺寸
    int bearing_x, bearing_y; // 轴承点的坐标
    int advance; // 进位值，表示字符与其下一个字符中心点的距离
};

class TextSourceManager {
public:
    TextSourceManager();

    ~TextSourceManager();

    bool initialize(int _pixel_height = 100, int _pixel_width = 0,
                    const std::string& font_file =
                        "../assets/fonts/AbyssinicaSIL-Regular.ttf");

    std::map<char, myCharacter> character_map;

    QOpenGLTexture* getCharactersTexture() {
        return qtexture_characters;
    }

private:
    bool initialized = false;
    int pixel_width, pixel_height;
    int rows = 8, cols = 16;
    QImage qimage_characters;
    QOpenGLTexture* qtexture_characters = nullptr;
    cv::Mat mat_characters;
};

#endif   // CYLINDER_STITCHER_TEXTSOURCEMANAGER_H
