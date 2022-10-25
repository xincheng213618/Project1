#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <iostream>
#include<math.h>
#include<time.h>

//CUDA
#include <cuda_runtime_api.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cufft.h>
#include "my_cuda.h"

using namespace std;

static bool g_bPCPnEnable = false;

Mycuda cuda_Function;

int num = 0;
int num_1 = 0;
std::mutex mtx;

cv::Mat Mat2Gray(cv::Mat _Ic, float _low, float _high, int out_putmode) {
	//___________________________________________________________
		//_Ic input CV_32FC1 random data range
		//out_putmode = 0: no change data type
		//out_putmode = 1; normalized to 0 - 1 CV_32FC1
		//out_putmode = 2; normalized to 0 - 255 CV_8UC1
	//___________________________________________________________
	cv::Mat Im_dst, Ic_src;
	_Ic.copyTo(Im_dst);
	_Ic.copyTo(Ic_src);
	for (int row = 0; row < Ic_src.rows; row++)
	{
		float* vptr0 = Im_dst.ptr<float>(row);
		for (int col = 0; col < Ic_src.cols; col++)
		{
			if (vptr0[col] < _low) {
				vptr0[col] = _low;
			}
			if (vptr0[col] > _high) {
				vptr0[col] = _high;
			}
		}
	}

	if (out_putmode == 1) {
		Im_dst = Im_dst - _low;
		Im_dst.convertTo(Im_dst, CV_32FC1, 1 / (_high - _low), 0);
	}
	else if (out_putmode == 2)
	{
		Im_dst = Im_dst - _low;
		Im_dst.convertTo(Im_dst, CV_32FC1, 1 / (_high - _low), 0);
		Im_dst.convertTo(Im_dst, CV_8UC1, 255, 0);
	}
	return Im_dst;
}

cv::Mat Phase2PC(cv::Mat _phi, float Max_frequency, float Pixelsize, float Filter) {//, float Gamma, float Gain, 
	mtx.lock();
	int i = num;
	num++;
	mtx.unlock();

	printf("num=%d\r\n", i);
	int nW;
	int nH;
	int nH_extend;
	int nW_extend;
	float delta_x;
	float delta_y;
	float a = 0.7f; // 决定生成光强均值

	float ee = 0.03f;//图像延拓比例（图像延拓去滤波伪影）
	nH = _phi.rows + int(round(ee * _phi.rows) * 2);//行
	nW = _phi.cols + int(round(ee * _phi.cols) * 2);//列

	//网格计算标志位
	int grid_filter_flag = 0;//1:行列有变化;2:滤波有变化

	delta_x = 1 / Pixelsize / nW;//列像素数
	delta_y = 1 / Pixelsize / nH;//行像素数

	clock_t start = 0, stop = 0;

	cv::Mat rho, weight, filter2, Ipc, Phi, real_filter, imag_filter;
	cv::Mat Show_PC;

	_phi.copyTo(Phi);

	cuda_Function.grid_creat_now(&rho, nH, nW, delta_x, delta_y, i);
	cuda_Function.circ_creat(rho, weight, filter2, Max_frequency, Filter, i);
	cv::GaussianBlur(weight, weight, cv::Size(15, 15), 60, 60);
	cv::GaussianBlur(filter2, filter2, cv::Size(31, 31), 300, 300);
	cuda_Function.filter_creat_now(weight, filter2, real_filter, imag_filter, g_bPCPnEnable, i);

	real_filter = Mat2Gray(real_filter, 0.25f, 0.9f, 2);
	real_filter.convertTo(real_filter, CV_8UC1, 1, 0);
	return real_filter;
}

int main()
{
	cv::Mat image[3];
	cv::Mat image_out;
	float Max_frequency = 1.0f;
	float Filter = sqrt(0.0008f);
	float Pixelsize = 1.0f;
	float Denoise = 1.0f;
	cv::Mat Show_PC;
	clock_t start[3] = {}, stop[3] = {};
	char str[100];

	image[0] = cv::imread("1.tiff");
	cv::cvtColor(image[0], image[0], cv::COLOR_BGR2GRAY);
	image[0].convertTo(image[0], CV_32FC1, 1.0 / 255.0);
	image[1] = cv::imread("ceshi3.bmp");
	cv::cvtColor(image[1], image[1], cv::COLOR_BGR2GRAY);
	image[1].convertTo(image[1], CV_32FC1, 1.0 / 255.0);
	image[2] = cv::imread("ceshi3.bmp");
	cv::cvtColor(image[2], image[2], cv::COLOR_BGR2GRAY);
	image[2].convertTo(image[2], CV_32FC1, 1.0 / 255.0);
	cuda_Function.Init_ALL();
	//time start
	start[0] = clock();
	for (int i = 0; i < 2; i++) {
		int j = i % 3;
		Show_PC = Phase2PC(image[j], Max_frequency, Pixelsize, Filter);
		/*std::thread([&]() {
			
			Show_PC = Phase2PC(image[j], Max_frequency, Pixelsize, Filter);
			mtx.lock();
			num_1++;
			sprintf_s(str, sizeof(str), "E:\\实验室学习\\Task4-22.08.04\\CV\\封装\\old\\write\\%d.jpg", num_1);
			cv::imwrite(str, Show_PC);
			mtx.unlock();
			}).join();*/
	}
	std::this_thread::sleep_for(std::chrono::milliseconds(10000));

	//start[0] = clock();
	//Show_PC = Phase2PC(image[0], Max_frequency, Pixelsize, Filter);

	

	stop[0] = clock();
	double endtime = (double)(stop[0] - start[0]) / CLOCKS_PER_SEC;
	std::cout << "time: " << endtime << "s" << "\r\n\r\n";

	//imshow("Show_PC", Show_PC);
	//imwrite("E:\\实验室学习\\C_3.bmp", Show_PC);
	/*if (Init_State == true) {
		cuda_Function.Free_ALL();
		Init_State = false;
	}*/
	cv::waitKey(0);

	return 0;
}

