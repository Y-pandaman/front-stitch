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

struct myCharacter {
    myCharacter(int _top, int _left, int _width, int _height, int _bearing_x,
                int _bearing_y, int _advance, int _tot_width, int _tot_height)
        : top(_top), left(_left), rel_top((float)top / (float)_tot_height),
          rel_left((float)left / (float)_tot_width), width(_width),
          height(_height), rel_width((float)width / (float)_tot_width),
          rel_height((float)height / (float)_tot_height), bearing_x(_bearing_x),
          bearing_y(_bearing_y), advance(_advance) { }

    int top, left;   // 总纹理图中的位置
    float rel_top, rel_left;

    int width, height;
    float rel_width, rel_height;
    int bearing_x, bearing_y;
    int advance;
};

class TextSourceManager {
public:
    TextSourceManager();

    ~TextSourceManager();

    bool initialize(int _pixel_height = 100, int _pixel_width = 0,
                    const std::string& font_file =
                        "../ar_vis/fonts/AbyssinicaSIL-Regular.ttf");

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
