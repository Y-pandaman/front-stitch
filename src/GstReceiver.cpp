#include "GstReceiver.h"

/**
 * 初始化GstReceiver。
 * 该函数负责设置视频源URL和队列大小，初始化图像队列和互斥锁，并尝试打开视频源。
 * 如果视频源无法打开，则函数返回false，否则返回true。
 *
 * @param url_ 视频源的URL。
 * @param queue_size_ 图像队列的大小，决定了可以缓存多少帧视频。
 * @return 初始化成功返回true，失败返回false。
 */
bool GstReceiver::initialize(const std::string& url_, int queue_size_) {
    this->video_url  = url_;
    this->queue_size = queue_size_;

    // 如果已经初始化过，则先释放之前的资源
    if (status != UNINITIALIZED) {
        delete[] this->image_queue;
        delete[] this->mutex_on_image_queue;
    }

    // 初始化图像队列和对应的互斥锁
    this->image_queue          = new cv::Mat[queue_size_];
    this->mutex_on_image_queue = new std::mutex[queue_size_];

    // 尝试打开视频源
    video_capture.open(this->video_url);
    if (!video_capture.isOpened()) {
        printf("can't open %s \n", this->video_url.c_str());
        return false;
    }

    // 设置状态为已初始化
    status = INITIALIZED;
    return true;
}

/**
 * 启动接收工作线程。
 * 该函数用于启动一个工作线程，用于接收和处理视频数据。在启动前，它会检查当前状态，
 * 并确保视频源能够被正确打开。如果一切准备就绪，它将创建并启动一个工作线程。
 *
 * @return bool 返回true如果成功启动接收工作线程，否则返回false。
 */
bool GstReceiver::startReceiveWorker() {
    // 检查当前状态，未初始化则直接返回false
    if (status == UNINITIALIZED) {
        return false;
    }
    // 如果状态为初始化或停止，重置帧计数并尝试打开视频源
    if (status == INITIALIZED || status == STOP) {
        frame_count = 0;   // 重置帧计数
        // 尝试打开视频源，失败则打印错误信息并返回false
        if (!video_capture.isOpened()) {
            printf("can't open %s \n", this->video_url.c_str());
            return false;
        }
        // 获取视频高度和宽度
        video_height = video_capture.get(cv::CAP_PROP_FRAME_HEIGHT);
        video_width  = video_capture.get(cv::CAP_PROP_FRAME_WIDTH);
        p_write = p_read = 0;   // 重置指针位置
        // 创建并启动接收工作线程
        thread_receive_worker =
            new std::thread(&GstReceiver::receiveWorker, this);
        status = RUN;   // 更新状态为运行中
    }
    return true;   // 成功启动工作线程，返回true
}

/**
 * @brief 该函数是一个工作线程，用于从视频源持续读取图像帧并存储到队列中。
 * 它会不断地尝试从视频源读取帧，直到被明确停止。
 *
 * @note
 * 该函数没有参数和返回值，因为它是一个成员函数，通过类的实例访问成员变量。
 */
void GstReceiver::receiveWorker() {
    while (true) {
        // 检查是否收到了停止信号，并处理停止逻辑
        if (stop_flag) {
            stop_flag = false;   // 重置停止标志，以允许再次启动接收工作
            return;              // 退出工作线程
        }

        // 计算要写入的队列位置
        int pos = p_write % queue_size;
        // 加锁以保护图像队列的访问
        std::unique_lock<std::mutex> locker(mutex_on_image_queue[pos]);
        // 尝试从视频捕获对象中读取一帧图像
        bool read_flag = video_capture.read(image_queue[pos]);
        if (!read_flag) {
            locker.unlock();   // 读取失败时解锁，并退出循环
            break;
        }

        // 图像读取成功后，解锁，并通知等待中的处理线程
        locker.unlock();
        cond_image_queue_not_empty
            .notify_one();   // 通知一个等待的线程，队列中已经有新图像了
        std::this_thread::sleep_for(
            std::chrono::milliseconds(1));   // 为了不独占CPU，线程短暂休眠
        p_write = (p_write + 1);   // 更新下一个要写入的位置
    }
}

uchar* GstReceiver::getImageData() {
    int pos = p_read % queue_size;
    std::unique_lock<std::mutex> locker(mutex_on_image_queue[pos]);
    while (p_read >= p_write) {
        cond_image_queue_not_empty.wait(locker);
    }
    size_t buf_size = image_queue[pos].rows * image_queue[pos].cols *
                      image_queue[pos].elemSize();
    uchar* result = new uchar[buf_size];
    memcpy(result, image_queue[pos].data, buf_size);
    p_read = p_read + 1;
    locker.unlock();
    return result;
}

GstReceiver::~GstReceiver() {
    if (thread_receive_worker != nullptr) {
        thread_receive_worker->join();
    }
    if (thread_write_worker != nullptr) {
        thread_write_worker->join();
    }
    delete image_queue;
    delete mutex_on_image_queue;
    delete thread_receive_worker;
    delete thread_write_worker;
}

/**
 * 从图像队列中获取一张Mat图像。
 * 此函数是线程安全的，会等待直到队列中有图像可用。
 *
 * @return cv::Mat 返回从队列中获取到的图像Mat对象。
 */
cv::Mat GstReceiver::getImageMat() {
    cv::Mat result;

    // 如果队列中的图像数量大于等于1，则准备读取最新的一帧图像
    if (p_write >= 1)
        p_read = p_write - 1;

    // 计算当前要读取的队列位置
    int pos = p_read % queue_size;

    // 加锁以保护图像队列，确保线程安全
    std::unique_lock<std::mutex> locker(mutex_on_image_queue[pos]);

    // 等待直到队列中有图像可供读取
    while (p_read >= p_write) {
        cond_image_queue_not_empty.wait(locker);
    }

    // 从队列中克隆出图像，并更新读取位置
    result = image_queue[pos].clone();
    p_read = (p_read + 1);

    // 解锁图像队列
    locker.unlock();

    return result;
}

void GstReceiver::openEcalImageTopic(const std::string& ecal_topic_name) {
    ecal_image_sender.open(ecal_topic_name);
    flag_ecal_send = true;
}
