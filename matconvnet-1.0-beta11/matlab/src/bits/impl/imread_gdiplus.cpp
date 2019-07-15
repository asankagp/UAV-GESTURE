// @file imread_win.cpp
// @brief Image reader based on Windows GDI+.
// @author Andrea Vedaldi

/*
Copyright (C) 2015 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "../imread.hpp"
#include "imread_helpers.hpp"

#include <windows.h>
#include <gdiplus.h>
#include <algorithm>

using namespace Gdiplus;
#pragma comment (lib,"Gdiplus.lib")

/* ---------------------------------------------------------------- */
/*                                       GDI+ reader implementation */
/* ---------------------------------------------------------------- */

#define check(x) \
if (!x) { image.error = 1 ; goto done ; }

class vl::ImageReader::Impl
{
public:
  Impl() ;
  ~Impl() ;
  GdiplusStartupInput gdiplusStartupInput;
  ULONG_PTR           gdiplusToken;
  vl::Image read(char const * filename, float * memory) ;
  vl::Image readDimensions(char const * filename) ;
} ;

vl::ImageReader::Impl::Impl()
{
  GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, NULL);
}

vl::ImageReader::Impl::~Impl()
{
  GdiplusShutdown(gdiplusToken);
}

static void getImagePropertiesHelper(vl::Image & image, Gdiplus::Bitmap & bitmap)
{
  // determine if the image is grayscale
  // this can either happen with an indexed images that has a grayscale palette
  bool grayscale = false ;
  Gdiplus::PixelFormat gdiPixelFormat = bitmap.GetPixelFormat();
  if (Gdiplus::IsIndexedPixelFormat(gdiPixelFormat)) {
    int paletteSize = bitmap.GetPaletteSize() ;
    Gdiplus::ColorPalette * palette =
      reinterpret_cast<Gdiplus::ColorPalette *>(new char[paletteSize]) ;
    bitmap.GetPalette(palette, paletteSize) ;
    grayscale = (palette->Flags & Gdiplus::PaletteFlagsGrayScale) != 0 ;
    delete[] reinterpret_cast<char *>(palette) ;
  }
  // or if the pixel type is as follows
  grayscale |= (gdiPixelFormat == PixelFormat16bppGrayScale) ;

  image.width = bitmap.GetWidth() ;
  image.height = bitmap.GetHeight() ;
  image.depth = grayscale ? 1 : 3 ;
}

vl::Image
vl::ImageReader::Impl::read(char const * filename, float * memory)
{
  // initialize the image as null
  Image image ;
  image.width = 0 ;
  image.height = 0 ;
  image.depth = 0 ;
  image.memory = NULL ;
  image.error = 0 ;

  Status status ;
  Rect rect ;
  bool grayscale = false ;

  wchar_t filenamew [1024*4] ;
  size_t n = 0 ;
  size_t convertedChars = 0 ;
  mbstowcs_s(&n, filenamew, sizeof(filenamew)/sizeof(wchar_t), filename, _TRUNCATE);

  BitmapData data ;
  Bitmap bitmap(filenamew);
  if (bitmap.GetLastStatus() != Ok) {
    image.error = 1 ;
    goto done ;
  }

  getImagePropertiesHelper(image, bitmap) ;

  if (memory == NULL) {
    image.memory = (float*)malloc(image.height * image.width * image.depth * sizeof(float)) ;
    check(image.memory) ;
  } else {
    image.memory = memory ;
  }

  // get the pixels
  rect = Rect(0,0,image.width,image.height);
  status = bitmap.LockBits(&rect,
                           ImageLockModeRead,
                           PixelFormat32bppRGB,
                           &data) ;
  if (status != Ok) {
    image.error = 1 ;
    goto done ;
  }

  // copy RGB to MATLAB format
  switch (image.depth) {
	case 3:
	  vl::impl::imageFromPixels<impl::pixelFormatBGRA>(image, (char unsigned const *)data.Scan0, data.Stride) ;
      break ;
	case 1:
	  vl::impl::imageFromPixels<impl::pixelFormatBGRAasL>(image, (char unsigned const *)data.Scan0, data.Stride) ;
	  break ;
  }

done:
  return image ;
}

vl::Image
vl::ImageReader::Impl::readDimensions(char const * filename)
{
  Image image ;
  image.width = 0 ;
  image.height = 0 ;
  image.depth = 0 ;
  image.memory = NULL ;
  image.error = 0 ;

  Status status ;

  wchar_t filenamew [1024*4] ;
  size_t n = 0 ;
  size_t convertedChars = 0 ;
  mbstowcs_s(&n, filenamew, sizeof(filenamew)/sizeof(wchar_t), filename, _TRUNCATE);

  Bitmap bitmap(filenamew);
  if (bitmap.GetLastStatus() != Ok) {
    image.error = 1 ;
    goto done ;
  }

  getImagePropertiesHelper(image, bitmap) ;

done:
  return image ;
}

/* ---------------------------------------------------------------- */
/*                                                      GDI+ reader */
/* ---------------------------------------------------------------- */

vl::ImageReader::ImageReader()
: impl(NULL)
{
  impl = new vl::ImageReader::Impl() ;
}

vl::ImageReader::~ImageReader()
{
  delete impl ;
}

vl::Image
vl::ImageReader::read(char const * filename, float * memory)
{
  return impl->read(filename, memory) ;
}

vl::Image
vl::ImageReader::readDimensions(char const * filename)
{
  return impl->readDimensions(filename) ;
}
