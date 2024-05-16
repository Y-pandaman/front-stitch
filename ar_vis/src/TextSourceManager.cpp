/*
 * @Author: 姚潘涛
 * @Date: 2024-05-08 10:14:31
 * @LastEditors: 姚潘涛
 * @LastEditTime: 2024-05-16 16:23:36
 * @Description:
 *
 * Copyright (c) 2024 by pandaman, All Rights Reserved.
 */
#include "TextSourceManager.h"

bool TextSourceManager::initialize(int _pixel_height, int _pixel_width,
                                   const std::string& font_file) {
    if (initialized) {
        return true;
    }
    pixel_height = _pixel_height;
    pixel_width  = _pixel_width;
    FT_Library ft;
    if (FT_Init_FreeType(&ft)) {
        std::cout << "ERROR::FREETYPE: Could not init FreeType Library"
                  << std::endl;
        return false;
    }
    FT_Face face;
    if (FT_New_Face(ft, font_file.c_str(), 0, &face)) {
        std::cout << "ERROR::FREETYPE: Failed to load font" << std::endl;
        return false;
    }
    FT_Set_Pixel_Sizes(face, _pixel_width, _pixel_height);   // set size

    int c_max_width = 0, c_max_height = 0;

    for (uchar c = 0; c < 128; c++) {
        if (FT_Load_Char(face, c, FT_LOAD_RENDER)) {
            std::cout << "ERROR::FREETYTPE: Failed to load Glyph" << std::endl;
            return false;
        }
        int c_height = face->glyph->bitmap.rows;
        int c_width  = face->glyph->bitmap.width;
        c_max_height = std::max(c_max_height, c_height);
        c_max_width  = std::max(c_max_width, c_width);
    }
    int tot_width  = c_max_width * cols;
    int tot_height = c_max_height * rows;

    // printf("c_max_height: %d c_max_width: %d\n", c_max_height, c_max_width);

    for (uchar c = 0; c < 128; c++) {
        if (FT_Load_Char(face, c, FT_LOAD_RENDER))   // load 'X'
        {
            std::cout << "ERROR::FREETYTPE: Failed to load Glyph" << std::endl;
            return false;
        }
        int c_height = face->glyph->bitmap.rows;
        int c_width  = face->glyph->bitmap.width;

        if (mat_characters.empty()) {
            mat_characters = cv::Mat::zeros(c_max_height * rows,
                                            c_max_width * cols, CV_8UC1);
        }
        cv::Mat cur_char = cv::Mat::zeros(c_height, c_width, CV_8UC1);
        memcpy(cur_char.data, face->glyph->bitmap.buffer,
               c_width * c_height * sizeof(uchar));

        int loc_row = c / cols, loc_col = c % cols;

        cv::Mat sub_loc = mat_characters(cv::Rect(
            loc_col * c_max_width, loc_row * c_max_height, c_width, c_height));

        cur_char.copyTo(sub_loc);

        myCharacter character(loc_row * c_max_height, loc_col * c_max_width,
                              c_width, c_height, face->glyph->bitmap_left,
                              face->glyph->bitmap_top, face->glyph->advance.x,
                              tot_width, tot_height);
        character_map.insert(std::make_pair(c, character));
    }

    std::vector<uchar> img_buf;
    cv::imencode(".bmp", mat_characters, img_buf);
    QByteArray img_byte_arr((char*)img_buf.data(),
                            static_cast<int>(img_buf.size()));
    qimage_characters.loadFromData(img_byte_arr, "BMP");
    qtexture_characters = new QOpenGLTexture(qimage_characters);
    initialized         = true;
    return true;
}

TextSourceManager::TextSourceManager() { }

TextSourceManager::~TextSourceManager() {
    //    delete[] characters_data;
}
