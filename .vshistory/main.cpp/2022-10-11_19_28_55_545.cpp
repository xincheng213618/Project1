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

cv::Mat Phase2PC(cv::Mat _phi, float Max_frequency, float Pixelsize, float Filter) {//, float Gamma, float Gain, 

	int nW;
	int nH;
	int nH_extend;
	int nW_extend;
	float delta_x;
	float delta_y;
	float a = 0.7f; // �������ɹ�ǿ��ֵ

	float ee = 0.03f;//ͼ�����ر�����ͼ������ȥ�˲�αӰ��
	nH = _phi.rows + int(round(ee * _phi.rows) * 2);//��
	nW = _phi.cols + int(round(ee * _phi.cols) * 2);//��

	//��������־λ
	int grid_filter_flag = 0;//1:�����б仯;2:�˲��б仯

	delta_x = 1 / Pixelsize / nW;//��������
	delta_y = 1 / Pixelsize / nH;//��������

	clock_t start = 0, stop = 0;

	cv::Mat rho, weight, filter2, Ipc, Phi, real_filter, imag_filter;
	cv::Mat Show_PC;

	_phi.copyTo(Phi);

	Phi.convertTo(Phi, CV_8UC1, 1, 0);
	return Phi;
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

	image[0] = cv::imread("..\\1.tiff");
	cv::cvtColor(image[0], image[0], cv::COLOR_BGR2GRAY);
	image[0].convertTo(image[0], CV_32FC1, 1.0 / 255.0);
	image[1] = cv::imread("..\\ceshi3.bmp");
	cv::cvtColor(image[1], image[1], cv::COLOR_BGR2GRAY);
	image[1].convertTo(image[1], CV_32FC1, 1.0 / 255.0);
	image[2] = cv::imread("..\\ceshi3.bmp");
	cv::cvtColor(image[2], image[2], cv::COLOR_BGR2GRAY);
	image[2].convertTo(image[2], CV_32FC1, 1.0 / 255.0);

	//time start
	start[0] = clock();
	for (int i = 0; i < 10; i++) {
		int j = i % 3;

		std::thread([&]() {
			//std::this_thread::sleep_for(20000ms);
			Show_PC = Phase2PC(image[j], Max_frequency, Pixelsize, Filter);
			sprintf_s(str, sizeof(str), "E:\\ʵ����ѧϰ\\Task4-22.08.04\\CV\\��װ\\old\\write\\%d.jpg", i);
			cv::imwrite(str, Show_PC);
			}).detach();
	}

	//start[0] = clock();
	//Show_PC = Phase2PC(image[0], Max_frequency, Pixelsize, Filter);

	

	stop[0] = clock();
	double endtime = (double)(stop[0] - start[0]) / CLOCKS_PER_SEC;
	std::cout << "time: " << endtime << "s" << "\r\n\r\n";

	imshow("Show_PC", Show_PC);
	//imwrite("E:\\ʵ����ѧϰ\\C_3.bmp", Show_PC);
	/*if (Init_State == true) {
		cuda_Function.Free_ALL();
		Init_State = false;
	}*/
	cv::waitKey(0);

	return 0;
}
