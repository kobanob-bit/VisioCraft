#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <vector>
#include <cstdint>
#include <libheif/heif.h>
#include <iostream>


using namespace emscripten;

// cv::getPerspectiveTransform の代替関数（4点の対応関係から3x3透視変換行列を計算）
cv::Mat getPerspectiveTransformCustom(const std::vector<cv::Point2f>& src, const std::vector<cv::Point2f>& dst) {
    cv::Mat A(8, 8, CV_64F);
    cv::Mat B(8, 1, CV_64F);

    for (int i = 0; i < 4; ++i) {
        double x = src[i].x;
        double y = src[i].y;
        double u = dst[i].x;
        double v = dst[i].y;

        A.at<double>(i * 2, 0) = x;
        A.at<double>(i * 2, 1) = y;
        A.at<double>(i * 2, 2) = 1;
        A.at<double>(i * 2, 3) = 0;
        A.at<double>(i * 2, 4) = 0;
        A.at<double>(i * 2, 5) = 0;
        A.at<double>(i * 2, 6) = -u * x;
        A.at<double>(i * 2, 7) = -u * y;
        B.at<double>(i * 2, 0) = u;

        A.at<double>(i * 2 + 1, 0) = 0;
        A.at<double>(i * 2 + 1, 1) = 0;
        A.at<double>(i * 2 + 1, 2) = 0;
        A.at<double>(i * 2 + 1, 3) = x;
        A.at<double>(i * 2 + 1, 4) = y;
        A.at<double>(i * 2 + 1, 5) = 1;
        A.at<double>(i * 2 + 1, 6) = -v * x;
        A.at<double>(i * 2 + 1, 7) = -v * y;
        B.at<double>(i * 2 + 1, 0) = v;
    }

    cv::Mat h(8, 1, CV_64F);
    cv::solve(A, B, h, cv::DECOMP_LU);

    cv::Mat H(3, 3, CV_64F);
    H.at<double>(0, 0) = h.at<double>(0);
    H.at<double>(0, 1) = h.at<double>(1);
    H.at<double>(0, 2) = h.at<double>(2);
    H.at<double>(1, 0) = h.at<double>(3);
    H.at<double>(1, 1) = h.at<double>(4);
    H.at<double>(1, 2) = h.at<double>(5);
    H.at<double>(2, 0) = h.at<double>(6);
    H.at<double>(2, 1) = h.at<double>(7);
    H.at<double>(2, 2) = 1.0;

    return H;
}

class ImageProcessor {
private:
    cv::Mat srcMat;
    cv::Mat dstMat;

    bool loadWithStandardDecoder(const uint8_t* data, size_t size) {
        try {
            if (!data || size == 0) return false;
            std::vector<uint8_t> buffer(data, data + size);
            srcMat = cv::imdecode(buffer, cv::IMREAD_COLOR);
            return !srcMat.empty();
        } catch (...) {
            return false;
        }
    }

public:
    ImageProcessor() {
        heif_init(nullptr);
    }

    ~ImageProcessor() {
        heif_deinit();
    }


	// HEIFデコード用の補助関数
	bool decodeHEIFMemory(const uint8_t* data, size_t size, cv::Mat& outMat) {
	    heif_context* ctx = heif_context_alloc();
	    if (!ctx) return false;

	    // ★重要: Wasm環境でスレッド作成エラーを出さないよう、最大スレッド数を0(シングルスレッド)に制限
	    heif_context_set_max_decoding_threads(ctx, 0);

	    heif_error err = heif_context_read_from_memory(ctx, data, size, nullptr);
	    if (err.code != heif_error_Ok) {
	        std::cout << "HEIF Read Error: " << err.message << std::endl;
	        heif_context_free(ctx);
	        return false;
	    }

	    heif_image_handle* handle = nullptr;
	    err = heif_context_get_primary_image_handle(ctx, &handle);
	    if (err.code != heif_error_Ok) {
	        std::cout << "HEIF Get Handle Error: " << err.message << std::endl;
	        heif_context_free(ctx);
	        return false;
	    }

	    heif_image* img = nullptr;
	    err = heif_decode_image(handle, &img, heif_colorspace_RGB, heif_chroma_interleaved_RGBA, nullptr);
	    if (err.code != heif_error_Ok) {
	        std::cout << "HEIF Decode Image Error: " << err.message << std::endl;
	        heif_image_handle_release(handle);
	        heif_context_free(ctx);
	        return false;
	    }

	    int width = heif_image_get_width(img, heif_channel_interleaved);
	    int height = heif_image_get_height(img, heif_channel_interleaved);
	    int stride = 0;
	    const uint8_t* plane = heif_image_get_plane_readonly(img, heif_channel_interleaved, &stride);

	    // RGBA -> BGR 変換
	    cv::Mat rgba(height, width, CV_8UC4, (void*)plane, stride);
	    cv::cvtColor(rgba, outMat, cv::COLOR_RGBA2BGR);

	    heif_image_release(img);
	    heif_image_handle_release(handle);
	    heif_context_free(ctx);
	    return true;
	}
	
	// 画像読み込みメイン関数
	bool loadImageFromMemory(uintptr_t bufferPtr, size_t size) {
	    const uint8_t* data = reinterpret_cast<const uint8_t*>(bufferPtr);
	    std::vector<uint8_t> vecData(data, data + size);

	    // 1. まずOpenCVの標準デコーダー（JPG/PNG等）を試す
	    cv::Mat decoded = cv::imdecode(vecData, cv::IMREAD_COLOR);
	    if (!decoded.empty()) {
	        std::cout << "Standard decoder result: 1" << std::endl;
	        // ※既存の画像セット処理（例: srcMat = decoded; 等）を行なって true を返す
	        srcMat = decoded.clone();
	        return true;
	    }

	    std::cout << "Standard decoder result: 0 (Trying HEIF...)" << std::endl;

	    // 2. 標準デコーダーで失敗した場合は libheif でデコードを試す
	    cv::Mat heifMat;
	    if (decodeHEIFMemory(data, size, heifMat) && !heifMat.empty()) {
	        std::cout << "HEIF decoder result: 1" << std::endl;
	        srcMat = heifMat.clone();
	        return true;
	    }

	    std::cout << "HEIF decoder result: 0" << std::endl;
	    return false;
	}

    int getWidth() const { return srcMat.cols; }
    int getHeight() const { return srcMat.rows; }

    uintptr_t getSrcPixels() {
        if (srcMat.empty()) return 0;
        static cv::Mat srcRgba;
        cv::cvtColor(srcMat, srcRgba, cv::COLOR_BGR2RGBA);
        return reinterpret_cast<uintptr_t>(srcRgba.data);
    }

    uintptr_t getDstPixels() {
        if (dstMat.empty()) return 0;
        static cv::Mat dstRgba;
        cv::cvtColor(dstMat, dstRgba, cv::COLOR_BGR2RGBA);
        return reinterpret_cast<uintptr_t>(dstRgba.data);
    }

    int getDstWidth() const { return dstMat.cols; }
    int getDstHeight() const { return dstMat.rows; }

    void rotateImage(int direction) {
        if (srcMat.empty()) return;
        if (direction == 0) {
            cv::rotate(srcMat, srcMat, cv::ROTATE_90_COUNTERCLOCKWISE);
        } else {
            cv::rotate(srcMat, srcMat, cv::ROTATE_90_CLOCKWISE);
        }
    }

	bool transformPerspective(emscripten::val srcPoints, int outWidth, int outHeight) {
	        if (srcMat.empty()) return false;

	        std::vector<cv::Point2f> pts(8);
	        for (int i = 0; i < 8; ++i) {
	            pts[i] = cv::Point2f(srcPoints[i]["x"].as<float>(), srcPoints[i]["y"].as<float>());
	        }

	        int halfH = outHeight / 2;

	        // std::vector<cv::Point2f> として定義（getPerspectiveTransformCustom の型に合致させる）
	        std::vector<cv::Point2f> srcTop = { pts[0], pts[1], pts[2], pts[3] };
	        std::vector<cv::Point2f> dstTop = {
	            cv::Point2f(0.0f, 0.0f), cv::Point2f((float)outWidth, 0.0f),
	            cv::Point2f(0.0f, (float)halfH), cv::Point2f((float)outWidth, (float)halfH)
	        };

	        std::vector<cv::Point2f> srcBottom = { pts[4], pts[5], pts[6], pts[7] };
	        std::vector<cv::Point2f> dstBottom = {
	            cv::Point2f(0.0f, 0.0f), cv::Point2f((float)outWidth, 0.0f),
	            cv::Point2f(0.0f, (float)halfH), cv::Point2f((float)outWidth, (float)halfH)
	        };

	        dstMat = cv::Mat::zeros(outHeight, outWidth, srcMat.type());

	        // 自作関数に vector を渡して行列を取得
	        cv::Mat M1 = getPerspectiveTransformCustom(srcTop, dstTop);
	        cv::Mat topRes;
	        cv::warpPerspective(srcMat, topRes, M1, cv::Size(outWidth, halfH));
	        topRes.copyTo(dstMat(cv::Rect(0, 0, outWidth, halfH)));

	        cv::Mat M2 = getPerspectiveTransformCustom(srcBottom, dstBottom);
	        cv::Mat bottomRes;
	        cv::warpPerspective(srcMat, bottomRes, M2, cv::Size(outWidth, halfH));
	        bottomRes.copyTo(dstMat(cv::Rect(0, halfH, outWidth, halfH)));

	        return true;
    }
};

EMSCRIPTEN_BINDINGS(visiocraft_module) {
    class_<ImageProcessor>("ImageProcessor")
        .constructor<>()
        .function("loadImageFromMemory", &ImageProcessor::loadImageFromMemory, allow_raw_pointers())
        .function("getWidth", &ImageProcessor::getWidth)
        .function("getHeight", &ImageProcessor::getHeight)
        .function("getSrcPixels", &ImageProcessor::getSrcPixels, allow_raw_pointers())
        .function("getDstPixels", &ImageProcessor::getDstPixels, allow_raw_pointers())
        .function("getDstWidth", &ImageProcessor::getDstWidth)
        .function("getDstHeight", &ImageProcessor::getDstHeight)
        .function("rotateImage", &ImageProcessor::rotateImage)
        .function("transformPerspective", &ImageProcessor::transformPerspective)
        ;
}