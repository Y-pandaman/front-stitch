/*
 * @Author: 姚潘涛
 * @Date: 2024-05-08 10:14:31
 * @LastEditors: 姚潘涛
 * @LastEditTime: 2024-05-17 09:38:13
 * @Description:
 *
 * Copyright (c) 2024 by pandaman, All Rights Reserved.
 */
#include "TextSourceManager.h"

/**
 * 初始化文本源管理器。
 *
 * 此函数负责初始化字体库，加载指定字体文件，并为ASCII字符集中的每个字符预渲染位图，
 * 之后将这些位图存储为一个大的图像矩阵，同时构建字符到其位图位置的映射。
 *
 * @param _pixel_height 字体的像素高度。
 * @param _pixel_width 字体的像素宽度。
 * @param font_file 要加载的字体文件的路径。
 * @return 初始化成功返回true，否则返回false。
 */
bool TextSourceManager::initialize(int _pixel_height, int _pixel_width,
                                   const std::string& font_file) {
    if (initialized) {
        // 如果已经初始化，则直接返回成功
        return true;
    }
    pixel_height = _pixel_height;
    pixel_width  = _pixel_width;

    // 初始化FreeType库
    FT_Library ft;
    if (FT_Init_FreeType(&ft)) {
        std::cout << "ERROR::FREETYPE: Could not init FreeType Library"
                  << std::endl;
        return false;
    }

    // 加载字体文件
    FT_Face face;
    if (FT_New_Face(ft, font_file.c_str(), 0, &face)) {
        std::cout << "ERROR::FREETYPE: Failed to load font" << std::endl;
        return false;
    }

    // 设置字体像素大小
    FT_Set_Pixel_Sizes(face, _pixel_width, _pixel_height);

    int c_max_width = 0, c_max_height = 0;

    // 遍历ASCII字符集，加载每个字符并确定最大宽度和高度
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

    // 计算存储所有字符所需的总宽度和高度
    int tot_width  = c_max_width * cols;
    int tot_height = c_max_height * rows;

    // 为每个字符的位图分配空间，并将它们复制到大的图像矩阵中
    for (uchar c = 0; c < 128; c++) {
        if (FT_Load_Char(face, c, FT_LOAD_RENDER))   // 加载字符的位图
        {
            std::cout << "ERROR::FREETYTPE: Failed to load Glyph" << std::endl;
            return false;
        }
        int c_height = face->glyph->bitmap.rows;
        int c_width  = face->glyph->bitmap.width;

        // 初始化或更新字符的位图矩阵
        if (mat_characters.empty()) {
            mat_characters = cv::Mat::zeros(c_max_height * rows,
                                            c_max_width * cols, CV_8UC1);
        }
        cv::Mat cur_char = cv::Mat::zeros(c_height, c_width, CV_8UC1);
        memcpy(cur_char.data, face->glyph->bitmap.buffer,
               c_width * c_height * sizeof(uchar));

        // 计算字符在存储矩阵中的位置，并将字符位图复制到该位置
        int loc_row = c / cols, loc_col = c % cols;
        cv::Mat sub_loc = mat_characters(cv::Rect(
            loc_col * c_max_width, loc_row * c_max_height, c_width, c_height));
        cur_char.copyTo(sub_loc);

        // 构建字符对象并插入到映射中
        myCharacter character(loc_row * c_max_height, loc_col * c_max_width,
                              c_width, c_height, face->glyph->bitmap_left,
                              face->glyph->bitmap_top, face->glyph->advance.x,
                              tot_width, tot_height);
        character_map.insert(std::make_pair(c, character));
    }

    // 将字符位图矩阵编码为BMP图像并加载到QImage中，之后创建OpenGL纹理以便于渲染
    std::vector<uchar> img_buf;
    cv::imencode(".bmp", mat_characters, img_buf);
    QByteArray img_byte_arr((char*)img_buf.data(),
                            static_cast<int>(img_buf.size()));
    qimage_characters.loadFromData(img_byte_arr, "BMP");
    qtexture_characters = new QOpenGLTexture(qimage_characters);

    // 标记初始化完成
    initialized = true;
    return true;
}

TextSourceManager::TextSourceManager() { }

TextSourceManager::~TextSourceManager() {
    //    delete[] characters_data;
}
