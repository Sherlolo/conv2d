#include <string.h>
#include <iostream>
#include <malloc.h>
#include <stdio.h>
class Mat
{
public:
    Mat() : data(nullptr), refcount(nullptr), w(0), h(0)
    {
        refcount = new int(1);
    }
    Mat(int _w, int _h) : w(_w), h(_h)
    {
        create();
    }
    Mat(int _w, int _h, float value) : w(_w), h(_h)
    {
        create();
        fill(value);
    }
    Mat(int _w, int _h, float* value) : w(_w), h(_h)
    {
        create();
        fill(value);
    }

    ~Mat()
    {
        *refcount -= 1;
        release();
    }
    void release()
    {
        if(*refcount == 0 && data!=nullptr)
        {
            delete refcount;
            delete [] data;
            //_aligned_free(data);
        }
    }


    Mat(const Mat& other);
    Mat& operator=(const Mat& other);

    //operator
    float& operator()(int i, int j)
    {
        return *(data + i*w + j);
    }

    float& operator()(int i, int j) const
    {
        return *(data + i*w + j);
    }
    void reshape(int _w, int _h)
    {
        w = _w;
        h = _h;
    }   

    //print
    friend std::ostream& operator<<(std::ostream & os,const Mat& m) // must do in class
    {
        for(int i = 0; i < m.h; ++i)
        {
            for(int j = 0; j < m.w; ++j)
                os << m(i,j) << " ";
            os << std::endl;
        }
        return os;
    }

    //private
    void add_count()
    {
        *refcount += 1;
    }

    void sub_count()
    {
        *refcount -= 1;
    }

    void create()
    {
        data = new float[w*h];
        //data = (float*)_aligned_malloc(w*h*sizeof(float), 32);
        refcount = new int(1);
    }
    void fill(float v)
    {   
        for(int i = 0; i < w*h; ++i)
            data[i] = v;
    }
    void fill(float* v)
    {
        memcpy(data, v, sizeof(float)*w*h);
    }
public:
    float* data;
    int w;
    int h;
    int* refcount;
}; // MAt