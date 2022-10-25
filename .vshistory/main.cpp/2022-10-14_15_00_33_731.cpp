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

using namespace cv;
using namespace std;

#define PI 3.14159265358979323846

/*********************�ҵı������*************************/
//�ɵ�����
int nW_old = 0;
int nH_old = 0;
//ȫ�ֱ���Max_frequency_old(Phase2PC)
float Max_frequency_old = 0.0f;
//ȫ�ֱ���Filter_old(Phase2PC)
float Filter_old = 0.0f;

bool Init_State = false;

int test;

//����һ��ȫ��rho(Phase2PC)
cv::Mat rho_old;
//����һ��ȫ��real(Phase2PC)
cv::Mat real_old;
//����һ��ȫ��imag(Phase2PC)
cv::Mat imag_old;

enum PC_state
{
	no_change,
	size_change,
	filter_change
};

Mycuda cuda_Function;

int num_1 = 0;
int num = 0;
std::mutex mtx;
/*********************�ҵı������*************************/

//�������
bool g_bPCPnEnable = false;

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

/***********************************�ҵĺ���***********************************/

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

	//int i = 1;
	//num++;
	mtx.lock();
	int i = num;
	num++;
	if (num == 4) { num = 0; }
	mtx.unlock();

	mtx.lock();
	//�����ж�
	if (nH != nH_old || nW != nW_old) {
		nH_old = nH;
		nW_old = nW;
		Max_frequency_old = Max_frequency;
		Filter_old = Filter;
		grid_filter_flag = 1;
	}
	if (grid_filter_flag == 0 && (Max_frequency != Max_frequency_old || Filter != Filter_old)) {
		Max_frequency_old = Max_frequency;
		Filter_old = Filter;
		grid_filter_flag = 2;
	}
	test = grid_filter_flag;
	cout << "\r\ntest=" << test << "\r\n";

	_phi.copyTo(Phi);

	Max_frequency = Max_frequency_old;
	Filter = Filter_old;
	//��ʼ���������ڴ�;��

	switch (grid_filter_flag)
	{
	case size_change://��ͼ�ߴ緢���仯
		cuda_Function.grid_creat_now(&rho, nH, nW, delta_x, delta_y, i);//�������񲢼���ƽ����
		//filter_creat(nH, nW, rho, Max_frequency, Filter);
		rho_old = rho;
		cuda_Function.circ_creat(rho, weight, filter2, Max_frequency, Filter, i);
		//cuda_Function.gaussianBlur_gpu(weight, weight, 15, 60);
		//cuda_Function.gaussianBlur_gpu(filter2, filter2, 31, 300);
		cv::GaussianBlur(weight, weight, cv::Size(15, 15), 60, 60);
		cv::GaussianBlur(filter2, filter2, cv::Size(31, 31), 300, 300);
		cuda_Function.filter_creat_now(weight, filter2, real_old, imag_old, g_bPCPnEnable, i);
		break;
	case filter_change://�˲��������仯����ͼ�ߴ粻��
		//0.088
		rho = rho_old;
		//filter_creat(nH, nW, rho, Max_frequency, Filter);
		cuda_Function.circ_creat(rho, weight, filter2, Max_frequency, Filter, i);
		cv::GaussianBlur(weight, weight, cv::Size(15, 15), 60, 60);
		cv::GaussianBlur(filter2, filter2, cv::Size(31, 31), 300, 300);
		cuda_Function.filter_creat_now(weight, filter2, real_old, imag_old, g_bPCPnEnable, i);
		break;
	case no_change:
		//0s
		rho = rho_old;
		break;
	}
	mtx.unlock();

	real_filter = real_old;
	imag_filter = imag_old;

	nH_extend = int(round(ee * _phi.rows));
	nW_extend = int(round(ee * _phi.cols));

	//copyMakeBorder��OpenCV�����㣬�ٶȲ�࣬��Ҫ��һ��ʼ������Opencv�����д�ó���
	copyMakeBorder(Phi, Phi, nH_extend, nH_extend, nW_extend, nW_extend, cv::BORDER_REPLICATE);

	cuda_Function.ALL_calculate(Phi, real_filter, imag_filter, &Ipc, nH_extend, nW_extend, i);

	cv::Scalar mean1 = cv::mean(Ipc);
	Ipc = Mat2Gray(Ipc, 0.25f, 0.9f, 2);
	Ipc.convertTo(Show_PC, CV_8UC1, 1, 0);

	return Show_PC;
}


int main()
{
	Mat image[3];
	Mat image_out;
	float Max_frequency = 1.0f;
	float Filter = sqrt(0.0008f);
	float Pixelsize = 1.0f;
	float Denoise = 1.0f;
	cv::Mat Show_PC;
	clock_t start = 0, stop = 0;
	char str[100];
	double endtime;

	image[0] = imread("1.tiff");
	cv::cvtColor(image[0], image[0], COLOR_BGR2GRAY);
	image[0].convertTo(image[0], CV_32FC1, 1.0 / 255.0);
	image[1] = imread("ceshi3.bmp");
	cv::cvtColor(image[1], image[1], COLOR_BGR2GRAY);
	image[1].convertTo(image[1], CV_32FC1, 1.0 / 255.0);
	image[2] = imread("ceshi3.bmp");
	cv::cvtColor(image[2], image[2], COLOR_BGR2GRAY);
	image[2].convertTo(image[2], CV_32FC1, 1.0 / 255.0);
	cuda_Function.Init_ALL();
	//time start
	for (int i = 0; i < 50; i++) {
		int j = i % 3;
		start = clock();
		Show_PC = Phase2PC(image[j], Max_frequency, Pixelsize, Filter);
		stop = clock();
		double endtime += (double)(stop - start) / CLOCKS_PER_SEC;
		std::cout << "time: " << endtime << "s" << "\r\n\r\n";
		//std::thread([&]() {
		//	//std::this_thread::sleep_for(20000ms);
		//	Show_PC = Phase2PC(image[j], Max_frequency, Pixelsize, Filter);
		//	mtx.lock();
		//	num_1++;
		//	sprintf_s(str, sizeof(str), "E:\\ʵ����ѧϰ\\Task4-22.08.04\\CV\\��װ\\old\\write\\%d.jpg", num_1);
		//	cv::imwrite(str, Show_PC);
		//	mtx.unlock();
		//	}).detach();
	}

	//start[0] = clock();
	//Show_PC = Phase2PC(image[0], Max_frequency, Pixelsize, Filter);

	std::this_thread::sleep_for(15000ms);

	//imshow("Show_PC", Show_PC);

	cuda_Function.Free_ALL();
	//imwrite("E:\\ʵ����ѧϰ\\C_3.bmp", Show_PC);
	/*if (Init_State == true) {
		cuda_Function.Free_ALL();
		Init_State = false;
	}*/
	waitKey(0);

	return 0;
}



