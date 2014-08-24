#ifndef CLCOMMONS_IMAGE_H
#define CLCOMMONS_IMAGE_H

#include "clcommons/common.h"

enum {
    ADDRESS_CLAMP = 0, //repeat border
    ADDRESS_ZERO = 1, //returns 0
    ADDRESS_REFLECT_BORDER_EXCLUSIVE = 2, //reflects at boundary and will not duplicate boundary elements
    ADDRESS_REFLECT_BORDER_INCLUSIVE = 3,//reflects at boundary and will duplicate boundary elements,
    ADDRESS_NOOP = 4//programmer guarantees no reflection necessary
};

//coordinate is c, r for compatibility with climage and CUDA
INLINE uint2 tex2D(const int rows, const int cols, const int _c, const int _r, const uint sample_method){
    int c = _c;
    int r = _r;
    if(sample_method == ADDRESS_REFLECT_BORDER_EXCLUSIVE){
        c = c < 0 ? -c : c;
        c = c >= cols ? cols - (c - cols) - 2: c;
        r = r < 0 ? -r : r;
        r = r >= rows ? rows - (r - rows) - 2: r;
    }else if(sample_method == ADDRESS_CLAMP){
        c = c < 0 ? 0 : c;
        c = c > cols - 1 ? cols - 1 : c;
        r = r < 0 ? 0 : r;
        r = r > rows - 1 ? rows - 1 : r;
    }else if(sample_method == ADDRESS_REFLECT_BORDER_INCLUSIVE){
        c = c < 0 ? -c - 1 : c;
        c = c >= cols ? cols - (c - cols) - 1: c;
        r = r < 0 ? -r - 1 : r;
        r = r >= rows ? rows - (r - rows) - 1: r;
    }else if(sample_method == ADDRESS_ZERO){
    }else if(sample_method == ADDRESS_NOOP){
    }else{
        assert(false);
    }
    assert_val(r >= 0 && r < rows, r);
    assert_val(c >= 0 && c < cols, c);
    return (uint2)(r, c);
}

INLINE __global uchar* image_line_at_(__global uchar *im_p, const uint im_rows, const uint im_cols, const uint image_pitch_p, const uint r){
    assert_val(r >= 0 && r < im_rows, r);
    (void) im_cols;
    return im_p + r * image_pitch_p;
}
#define image_line_at(PixelT, im_p, im_rows, im_cols, image_pitch, r) ((__global PixelT *) image_line_at_((__global uchar *) (im_p), (im_rows), (im_cols), (image_pitch), (r)))

INLINE __global uchar* image_pixel_at_(__global uchar *im_p, const uint im_rows, const uint im_cols, const uint image_pitch_p, const uint r, const uint c, const uint sizeof_pixel){
    assert_val(r >= 0 && r < im_rows, r);
    assert_val(c >= 0 && c < im_cols, c);
    return im_p + r * image_pitch_p + c * sizeof_pixel;
}
#define image_pixel_at(PixelT, im_p, im_rows, im_cols, image_pitch, r, c) (*((__global PixelT *) image_pixel_at_((__global uchar *)(im_p), (im_rows), (im_cols), (image_pitch), (r), (c), sizeof(PixelT))))

INLINE __global uchar* image_tex2D_(__global uchar *im_p, const uint im_rows, const uint im_cols, const uint image_pitch, const int r, const int c, const uint sizeof_pixel, const uint sample_method){
    const uint2 p2 = tex2D((int) im_rows, (int) im_cols, c, r, sample_method);
    return image_pixel_at_(im_p, im_rows, im_cols, image_pitch, p2.s0, p2.s1, sizeof_pixel);
}
#define image_tex2D(PixelT, im_p, im_rows, im_cols, image_pitch, r, c, sample_method) \
  (((sample_method) == ADDRESS_ZERO) & (((r) < 0) | ((r) >= (im_rows)) | ((c) < 0) | ((c) >= (im_cols))) ? 0 : \
  *(__global PixelT *) image_tex2D_((__global uchar *)(im_p), (im_rows), (im_cols), (image_pitch), (r), (c), sizeof(PixelT), (sample_method)))


#ifdef ENABLE_CL_CPP

struct CLImage{
    __global uchar *data;
    uint pitch;
    uint rows;
    uint cols;

    INLINE CLImage(uint rows, uint cols, __global uchar *data, uint pitch):data(data), pitch(pitch), rows(rows), cols(cols){}

    template<typename PixelT, typename IndexT = uint>
    INLINE IndexT offset(IndexT r, IndexT c) const{
        assert_val(r >= 0 && r < rows, r);
        assert_val(c >= 0 && c < cols, c);
        return r * pitch + c * sizeof(PixelT);
    }

    template<typename IndexT = uint>
    INLINE __global const uchar* _line(IndexT r)const{
        assert_val(r >= 0 && r < rows, r);
        return data + r * pitch;
    }

    template<typename IndexT = uint>
    INLINE __global uchar* _line(IndexT r){
        assert_val(r >= 0 && r < rows, r);
        return data + r * pitch;
    }

    template<typename PixelT, typename IndexT = uint>
    INLINE __global PixelT& _at(IndexT r, IndexT c){
        assert_val(r >= 0 && r < rows, r);
        assert_val(c >= 0 && c < cols, c);
        return *((__global PixelT *)(data + offset<PixelT, IndexT>(r, c)));
    }
    template<typename PixelT, typename IndexT = uint>
    INLINE __global const PixelT& _at(IndexT r, IndexT c) const{
        assert_val(r >= 0 && r < rows, r);
        assert_val(c >= 0 && c < cols, c);
        return *((__global const PixelT *)(data + offset<PixelT, IndexT>(r, c)));
    }

    //Note that this is r,c!
    template<typename PixelT>
    INLINE PixelT _tex2D(const int c, const int r, const uint sample_method = ADDRESS_CLAMP) const{
        uint2 p = tex2D(rows, cols, c, r, sample_method);
        return _at<PixelT, uint>(p.s0, p.s1);
    }
};

template<typename PixelT>
struct CLImageT : public CLImage{

    INLINE CLImageT(uint rows, uint cols, __global uchar *data):CLImage(rows, cols, data, sizeof(PixelT) * cols){}
    INLINE CLImageT(uint rows, uint cols, __global uchar *data, uint pitch):CLImage(rows, cols, data, pitch){}
    INLINE __global PixelT* p(){
        return data;
    }
    INLINE __global const PixelT* p() const{
        return data;
    }

    template<typename IndexT = uint>
    INLINE const __global PixelT* line(IndexT r) const{
        return (__global PixelT*) CLImage::_line<IndexT>(r);
    }

    template<typename IndexT = uint>
    INLINE __global PixelT* line(IndexT r){
        return (__global PixelT*) CLImage::_line<IndexT>(r);
    }

    template<typename IndexT = uint>
    INLINE __global const PixelT& at(uint r, uint c) const{
        return CLImage::_at<PixelT, IndexT>(r, c);
    }
    template<typename IndexT = uint>
    INLINE __global PixelT& at(uint r, uint c){
        return CLImage::_at<PixelT, IndexT>(r, c);
    }

    INLINE const PixelT tex2D(const int c, const int r, const uint sample_method = ADDRESS_REFLECT_BORDER_EXCLUSIVE){
        return CLImage::_tex2D<PixelT>(c, r, sample_method);
    }
    INLINE const PixelT tex2D(const int c, const int r, const uint sample_method = ADDRESS_REFLECT_BORDER_EXCLUSIVE) const{
        return CLImage::_tex2D<PixelT>(c, r, sample_method);
    }
    template<typename IndexT = uint>
    INLINE const PixelT& operator()(uint r, uint c) const{
        return this->at(r, c);
    }
    template<typename IndexT = uint>
    INLINE __global PixelT& operator()(uint r, uint c){
        return this->at(r, c);
    }
};

#endif

#endif
