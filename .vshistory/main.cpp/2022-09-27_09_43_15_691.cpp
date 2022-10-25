#include"iostream"
#include"cuda_runtime_api.h"
#include"device_launch_parameters.h"
#include"cufft.h"

#include <iostream>
#include<math.h>
#include<time.h>
using namespace std;


//FFT���任�����ڹ淶���ĺ���
__global__ void normalizing(cufftComplex* data, int data_len)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	data[idx].x /= data_len;
	data[idx].y /= data_len;
}
void Check(cudaError_t status)
{
	if (status != cudaSuccess)
	{
		cout << "�к�:" << __LINE__ << endl;
		cout << "����:" << cudaGetErrorString(status) << endl;
	}
}
int main()
{
	const int Nt = 256;
	const int BATCH = 1;
	//BATCH������������һ��һά���ݣ���BATCH=2ʱ
	//��0-1024��1024-2048��Ϊ����һά�ź���FFT�����任
	cufftComplex* host_in, * host_out, * device_in, * device_out;
	//�����ڴ����뼰��ʼ��--������ҳ�ڴ�
	Check(cudaMallocHost((void**)&host_in, Nt * sizeof(cufftComplex)));
	Check(cudaMallocHost((void**)&host_out, Nt * sizeof(cufftComplex)));
	for (int i = 0; i < Nt; i++)
	{
		host_in[i].x = i + 1;
		host_in[i].y = i + 1;
	}
	//�豸�ڴ�����
	Check(cudaMalloc((void**)&device_in, Nt * sizeof(cufftComplex)));
	Check(cudaMalloc((void**)&device_out, Nt * sizeof(cufftComplex)));
	//���ݴ���--H2D
	Check(cudaMemcpy(device_in, host_in, Nt * sizeof(cufftComplex), cudaMemcpyHostToDevice));

	//����cufft���
	cufftHandle cufftForwrdHandle, cufftInverseHandle;
	cufftPlan1d(&cufftForwrdHandle, Nt, CUFFT_C2C, BATCH);
	cufftPlan1d(&cufftInverseHandle, Nt, CUFFT_C2C, BATCH);

	//ִ��fft���任
	cufftExecC2C(cufftForwrdHandle, device_in, device_out, CUFFT_FORWARD);

	//���ݴ���--D2H
	Check(cudaMemcpy(host_out, device_out, Nt * sizeof(cufftComplex), cudaMemcpyDeviceToHost));

	//�����������--���任������
	cout << "���任���:" << endl;
	cout.setf(20);
	for (int i = 0; i < Nt; i++)
	{
		cout << host_out[i].x << "+j*" << host_out[i].y << endl;
	}

	//ִ��fft���任
	cufftExecC2C(cufftInverseHandle, device_out, device_in, CUFFT_INVERSE);

	//IFFT�������ֵ��N�������Ҫ��/N����
	dim3 grid(Nt / 128);
	dim3 block(128);
	normalizing << <grid, block >> > (device_in, Nt);

	//���ݴ���--D2H
	Check(cudaMemcpy(host_in, device_in, Nt * sizeof(cufftComplex), cudaMemcpyDeviceToHost));

	//�����������--���任������
	cout << "���任���:" << endl;
	cout.setf(20);
	for (int i = 0; i < Nt; i++)
	{
		cout << host_in[i].x << "+j*" << host_in[i].y << endl;
	}
	cin.get();
	return 0;
}