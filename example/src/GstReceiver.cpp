#include "GstReceiver.h"

bool GstReceiver::initialize(const std::string& url_, int queue_size_) {
    this->video_url  = url_;
    this->queue_size = queue_size_;
    if (status != UNINITIALIZED) {
        delete[] this->image_queue;
        delete[] this->mutex_on_image_queue;
    }
    this->image_queue          = new cv::Mat[queue_size_];
    this->mutex_on_image_queue = new std::mutex[queue_size_];

    video_capture.open(this->video_url);
    if (!video_capture.isOpened()) {
        printf("can't open %s \n", this->video_url.c_str());
        return false;
    }

    status = INITIALIZED;
    return true;
}

bool GstReceiver::startReceiveWorker() {
    if (status == UNINITIALIZED) {
        return false;
    }
    if (status == INITIALIZED || status == STOP) {
        frame_count = 0;
        if (!video_capture.isOpened()) {
            printf("can't open %s \n", this->video_url.c_str());
            return false;
        }
        video_height = video_capture.get(cv::CAP_PROP_FRAME_HEIGHT);
        video_width  = video_capture.get(cv::CAP_PROP_FRAME_WIDTH);
        p_write = p_read = 0;
        thread_receive_worker =
            new std::thread(&GstReceiver::receiveWorker, this);
        status = RUN;
    }
    return true;
}

void GstReceiver::receiveWorker() {
    while (true) {
        if (stop_flag) {
            stop_flag = false;
            return;
        }
        int pos = p_write % queue_size;
        std::unique_lock<std::mutex> locker(mutex_on_image_queue[pos]);
        bool read_flag = video_capture.read(image_queue[pos]);
        if (!read_flag) {
            locker.unlock();
            break;
        }

        locker.unlock();
        cond_image_queue_not_empty.notify_one();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        p_write = (p_write + 1);
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

cv::Mat GstReceiver::getImageMat() {
    cv::Mat result;
    if (p_write >= 1)
        p_read = p_write - 1;
    int pos = p_read % queue_size;
    std::unique_lock<std::mutex> locker(mutex_on_image_queue[pos]);
    while (p_read >= p_write) {
        cond_image_queue_not_empty.wait(locker);
    }
    result = image_queue[pos].clone();
    p_read = (p_read + 1);
    locker.unlock();
    return result;
}

void GstReceiver::openEcalImageTopic(const std::string& ecal_topic_name) {
    ecal_image_sender.open(ecal_topic_name);
    flag_ecal_send = true;
}
