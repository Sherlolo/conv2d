#include "Mat.h"

Mat::Mat(const Mat& other) : data(other.data), w(other.w), h(other.h), refcount(other.refcount)
{
    add_count();
}

Mat& Mat::operator=(const Mat& other)
{
    if(this == &other)
        return *this;
    this->sub_count();
    this->release();

    this->data = other.data;
    this->w = other.w;
    this->h = other.h;
    this->refcount = other.refcount;
    this->add_count();

    return *this;
}
