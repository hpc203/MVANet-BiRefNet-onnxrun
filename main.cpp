#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>

using namespace cv;
using namespace std;
using namespace Ort;


class MVANet
{
public:
	MVANet(string modelpath);
	Mat detect(const Mat& frame, const float score_th=0);
private:
	vector<float> input_image;

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "Multi-view Aggregation Network for Dichotomous Image Segmentation");
	Ort::Session *ort_session = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	const vector<const char*> input_names = {"input_image"};
	const vector<const char*> output_names = {"output_image"};
    int inpWidth;
	int inpHeight;
    void preprocess(const Mat& frame);
	const float mean_[3] = {0.485, 0.456, 0.406};
	const float std_[3] = {0.229, 0.224, 0.225};
};

MVANet::MVANet(string model_path)
{
	//OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
    sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

    // std::wstring widestr = std::wstring(model_path.begin(), model_path.end());   ////windows写法
	// ort_session = new Session(env, widestr.c_str(), sessionOptions);           ////windows写法
    ort_session = new Session(env, model_path.c_str(), sessionOptions);          ////linux写法

    size_t numInputNodes = ort_session->GetInputCount();
    AllocatorWithDefaultOptions allocator;
    vector<vector<int64_t>> input_node_dims;
	for (int i = 0; i < numInputNodes; i++)
	{
		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		input_node_dims.push_back(input_dims);
	}

    this->inpHeight = input_node_dims[0][2];
	this->inpWidth = input_node_dims[0][3];
}

void MVANet::preprocess(const Mat& frame)
{
    Mat img;
    cvtColor(frame, img, COLOR_BGR2RGB);
	resize(img, img, Size(this->inpWidth, this->inpHeight));

	vector<Mat> rgbChannels(3);
    split(img, rgbChannels);
	for (int c = 0; c < 3; c++)
	{
		rgbChannels[c].convertTo(rgbChannels[c], CV_32FC1, 1.0 / (255.0* this->std_[c]), (0.0 - this->mean_[c]) / this->std_[c]);
	}
	const int image_area = img.rows * img.cols;
    this->input_image.clear();
	this->input_image.resize(1 * 3 * image_area);
    int single_chn_size = image_area * sizeof(float);
	memcpy(this->input_image.data(), (float *)rgbChannels[0].data, single_chn_size);
    memcpy(this->input_image.data() + image_area, (float *)rgbChannels[1].data, single_chn_size);
    memcpy(this->input_image.data() + image_area * 2, (float *)rgbChannels[2].data, single_chn_size);
}

Mat MVANet::detect(const Mat& frame, const float score_th)
{
	this->preprocess(frame);

	array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };
    auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, this->input_image.data(), this->input_image.size(), input_shape_.data(), input_shape_.size());

	// 开始推理
	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());   // 开始推理
	
    std::vector<int64_t> out_shape = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape();  ////输出形状是1,1,1024,1024
    const int out_h = out_shape[2];
    const int out_w = out_shape[3];
	float* pred = ort_outputs[0].GetTensorMutableData<float>();       
    Mat out = Mat(out_h, out_w, CV_32FC1, pred);
	Mat mask;
	cv::exp(-out, mask);
	mask = 1.f / (1.f + mask);
    if(score_th > 0)
	{
		mask.setTo(0, mask < score_th);
		mask.setTo(1, mask >= score_th);
	}
	mask *= 255;
	mask.convertTo(mask, CV_8UC1);
	resize(mask, mask, Size(frame.cols, frame.rows));
    return mask;
}


int main()
{
	MVANet mynet("mvanet_1024x1024.onnx");  
	string imgpath = "testimgs/3.jpg";  ///文件路径写正确，程序才能正常运行的
	const float score_th=0.9;

	Mat srcimg = imread(imgpath);

	Mat mask = mynet.detect(srcimg, score_th);
	Mat dstimg = srcimg.clone();
	dstimg.setTo(Scalar(255, 255, 255), mask==0);
	
	// imwrite("mask.jpg", mask);
	// imwrite("dstimg.jpg", dstimg);

    namedWindow("srcimg", WINDOW_NORMAL);
	imshow("srcimg", srcimg);
	namedWindow("mask", WINDOW_NORMAL);
	imshow("mask", mask);
	namedWindow("dstimg", WINDOW_NORMAL);
	imshow("dstimg", dstimg);
	waitKey(0);
	destroyAllWindows();
}