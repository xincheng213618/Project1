#include "my_cuda.cuh"

using namespace std;

static float(*device1_float_out)[4] = NULL, (*device2_float_out)[4] = NULL, (*device3_float_out)[4] = NULL, (*device4_float_out)[4] = NULL;
static float(*Amp_real_out)[4] = NULL, (*Phi_imag_out)[4] = NULL;
static cufftComplex(*U0_device_out)[4] = NULL;
cudaStream_t stream[4];

#define PI 3.14159265358979323846

//std::mutex mtx_1;

static void Check(cudaError_t status)
{
	if (status != cudaSuccess)
	{
		cout << "�к�:" << __LINE__ << endl;
		cout << "����:" << cudaGetErrorString(status) << endl;
	}
}

__global__ void Mat2complex_kernel(float* src_1, float* src_2, cufftComplex* U0, bool phase_mode, int N)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < N) {
		if (phase_mode) {
			U0[idx].x = src_1[idx] * cos(src_2[idx]);
			U0[idx].y = src_1[idx] * sin(src_2[idx]);
		}
		else {
			//�о���ط�û��ҪCUDA���٣�PC����Ҳû�õ�
			U0[idx].x = src_1[idx];
			U0[idx].y = src_2[idx];
		}
	}
}

__global__ void Mat2complex_kernel(float* src_2, cufftComplex* U0, float a, int N)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < N) {
		U0[idx].x = a * cos(src_2[idx]);
		U0[idx].y = a * sin(src_2[idx]);
	}
}

__global__ void Complex2mat_kernel(cufftComplex* U0, float* dst_1, float* dst_2, bool phase_mode, int N)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < N) {
		if (phase_mode) {
			dst_1[idx] = sqrt(U0[idx].x * U0[idx].x + U0[idx].y * U0[idx].y);
			dst_2[idx] = atan2(U0[idx].y, U0[idx].x);
		}
		else {
			//��������ܼ����𣿲��ᷴ�������ٶ���
			dst_1[idx] = U0[idx].x;
			dst_2[idx] = U0[idx].y;
		}
	}
}

__global__ void grid_creat_kernel(float* dst, int nH, int nW, float delta_x, float delta_y, int N)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < N) {
		dst[idx] = sqrt((-nW / 2.0f * delta_x + idx % nW * delta_x)\
					  * (-nW / 2.0f * delta_x + idx % nW * delta_x)\
					  + (-nH / 2.0f * delta_y + idx / nW * delta_y)\
					  * (-nH / 2.0f * delta_y + idx / nW * delta_y));
	}
}

__global__ void CUDA_mul_kernel_2(float* src1, float* src2, float* src3, float* src4, float* dst1, float* dst2, int N)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < N) {
		dst1[idx] = src1[idx] * src3[idx] - src2[idx] * src4[idx];
		dst2[idx] = src1[idx] * src4[idx] + src2[idx] * src3[idx];
	}
}

__global__ void CUDA_mul_kernel(float* src1, float* src2, float* dst, int N)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < N) {
		dst[idx] = src1[idx] * src2[idx];
	}
}

//FFT���任�����ڹ淶���ĺ���
__global__ void normalizing(cufftComplex* data, int data_len)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	data[idx].x /= data_len;
	data[idx].y /= data_len;
}

__global__ void filter_creat_kernel(float* src, float* dst1, float* dst2, float Max_frequency, float Filter, float Denoise_Radius, int N)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < N) {
		dst1[idx] = (float)(src[idx] < Max_frequency * Filter);
		dst2[idx] = (float)(src[idx] < Denoise_Radius);
	}
}
//���� ʵ��fftshift
__global__ void fftshift_step1_kernel(float* src, float* dst1, float* dst2, float* dst3, float* dst4, int nH, int nW, int N)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < N) {
		if ((idx % nW < nW / 2) && (idx / nW < nH / 2))
		{
			dst1[idx / nW * (nW / 2) + idx % nW] = src[idx];
		}
		else if ((idx % nW >= nW / 2) && (idx / nW < nH / 2))
		{
			dst2[idx / nW * (nW / 2) + (idx - nW / 2) % nW] = src[idx];
		}
		else if ((idx % nW < nW / 2) && (idx / nW >= nH / 2))
		{
			dst3[idx / nW * (nW / 2) - nH / 2 * nW / 2 + idx % nW] = src[idx];
		}
		else
		{
			dst4[idx / nW * (nW / 2) - nH / 2 * nW / 2 + (idx - nW / 2) % nW] = src[idx];
		}
	}
}

__global__ void fftshift_step2_kernel(float* tmp1, float* tmp2, float* tmp3, float* tmp4, float* Phi, int nH, int nW, int N)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < N) {
		if ((idx % nW < nW / 2) && (idx / nW < nH / 2))
		{
			Phi[idx] = tmp4[idx / nW * (nW / 2) + idx % nW];
		}
		else if ((idx % nW >= nW / 2) && (idx / nW < nH / 2))
		{
			Phi[idx] = tmp3[idx / nW * (nW / 2) + (idx - nW / 2) % nW];
		}
		else if ((idx % nW < nW / 2) && (idx / nW >= nH / 2))
		{
			Phi[idx] = tmp2[idx / nW * (nW / 2) - nH / 2 * nW / 2 + idx % nW];
		}
		else
		{
			Phi[idx] = tmp1[idx / nW * (nW / 2) - nH / 2 * nW / 2 + (idx - nW / 2) % nW];
		}
	}
}

__global__ void Phase_delay(float* data, bool flag, int N)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < N) {
		if (flag){
			data[idx] = data[idx] * PI / 2;
		}
		else {
			data[idx] = -data[idx] * PI / 2;
		}
	}
}

__global__ void rect_kernel(float* src, float* dst, int nH, int nW, int x, int y, int high, int width, int N)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < N && idx % nW>y - 1 && idx % nW<y + width && idx / nW>x - 1 && idx / nW < x + high) {
		dst[(idx / nW - x) * width + (idx - idx / nW * nW - y)] = src[idx];
	}
}

void CUDA_Init_ALL(int nH, int nW)
{
	size_t a, b;
	Check(cudaMalloc((void**)&device1_float_out, 4 * nH * nW * sizeof(float)));
	Check(cudaMalloc((void**)&device2_float_out, 4 * nH * nW * sizeof(float)));
	Check(cudaMalloc((void**)&device3_float_out, 4 * nH * nW * sizeof(float)));
	Check(cudaMalloc((void**)&device4_float_out, 4 * nH * nW * sizeof(float)));
	Check(cudaMalloc((void**)&Amp_real_out, 4 * nH * nW * sizeof(float)));
	Check(cudaMalloc((void**)&Phi_imag_out, 4 * nH * nW * sizeof(float)));
	Check(cudaMalloc((void**)&U0_device_out, 4 * nH * nW * sizeof(cufftComplex)));
	
	cudaStreamCreate(&stream[0]);
	cudaStreamCreate(&stream[1]);
	cudaStreamCreate(&stream[2]);
	cudaStreamCreate(&stream[3]);

}
//
void CUDA_Free_ALL()
{
	Check(cudaFree(device1_float_out));
	Check(cudaFree(device2_float_out));
	Check(cudaFree(device3_float_out));
	Check(cudaFree(device4_float_out));
	Check(cudaFree(Amp_real_out));
	Check(cudaFree(Phi_imag_out));
	Check(cudaFree(U0_device_out));
}

//�������񲢼���ƽ����
void CUDA_grid_creat_now(cv::Mat* dst, int nH, int nW, float delta_x, float delta_y, int i)
{
	printf("abc = %d", i);
	float* device1_float = device1_float_out[i];
	cv::Mat rho = cv::Mat::ones(nH, nW, CV_32FC1);
	dim3 grid((nH * nW + 1024 - 1) / 1024);//(1280*960+1024-1)/1024
	dim3 block(1024);//1024
	grid_creat_kernel << <grid, block >> > (device1_float, nH, nW, delta_x, delta_y, nH * nW);
	cudaDeviceSynchronize();
	Check(cudaMemcpy(rho.data, (uchar*)device1_float, nH * nW * sizeof(float), cudaMemcpyDeviceToHost));
	*dst = rho;
	device1_float = NULL;
	//cudaStreamSynchronize(0);
}

void CUDA_circ_creat(cv::Mat src, cv::Mat& dst1, cv::Mat& dst2, float Max_frequency, float Filter, int i)
{
	int nH = src.rows;
	int nW = src.cols;
	int Nt = nH * nW;
	/*�ݶ�����*/
	int PS_Gas_Radius = 15;// PS_Gas_RadiusΪ��˹�˲��뾶���ø�˹�˲�ʵ����ֺ���ƣ�
	int PS_Gas_var = 60;// PS_Gas_varΪ��˹�˲�����ø�˹�˲�ʵ����ֺ���ƣ�
	float Denoise_Radius = (1.0f / 3.0f);;// Denoise_Radius��ȥ��뾶
	int G_R = 31; // gaussian�˲��뾶
	int G_V = 300; // gaussian�˲�����

	float* device1_float = device1_float_out[i];
	float* device2_float = device2_float_out[i];
	float* device3_float = device3_float_out[i];

	cv::Mat weight = cv::Mat::zeros(nH, nW, CV_32FC1);
	cv::Mat filter2 = cv::Mat::zeros(nH, nW, CV_32FC1);

	Check(cudaMemcpy(device3_float, src.data, nH * nW * sizeof(float), cudaMemcpyHostToDevice));
	dim3 grid((nH * nW + 1024 - 1) / 1024);//(1280*960+1024-1)/1024
	dim3 block(1024);//1024
	filter_creat_kernel << <grid, block >> > (device3_float, device1_float, device2_float, Max_frequency, Filter, Denoise_Radius, Nt);
	cudaDeviceSynchronize();
	Check(cudaMemcpy(weight.data, (uchar*)device1_float, nH * nW * sizeof(float), cudaMemcpyDeviceToHost));
	Check(cudaMemcpy(filter2.data, (uchar*)device2_float, nH * nW * sizeof(float), cudaMemcpyDeviceToHost));
	dst1 = weight;
	dst2 = filter2;
	//cudaStreamSynchronize(0);
}

void CUDA_filter_creat_now(cv::Mat src1, cv::Mat src2, cv::Mat& dst1, cv::Mat& dst2, bool flag, int i)
{
	int nH = src2.rows;
	int nW = src2.cols;
	int Nt = nH * nW;
	
	float* device1_float = device1_float_out[i];
	float* device2_float = device2_float_out[i];
	float* device3_float = device3_float_out[i];
	float* device4_float = device4_float_out[i];
	float* Amp_real = Amp_real_out[i];
	float* Phi_imag = Phi_imag_out[i];
	cufftComplex* U0_device = U0_device_out[i];

	cv::Mat real_out = cv::Mat::zeros(nH, nW, CV_32FC1);
	cv::Mat imag_out = cv::Mat::zeros(nH, nW, CV_32FC1);
	Check(cudaMemcpy(device1_float, src1.data, nH * nW * sizeof(float), cudaMemcpyHostToDevice));
	Check(cudaMemcpy(device2_float, src2.data, nH * nW * sizeof(float), cudaMemcpyHostToDevice));

	dim3 grid((nH * nW + 1024 - 1) / 1024);//(1280*960+1024-1)/1024
	dim3 block(1024);//1024
	Phase_delay << <grid, block >> > (device1_float, flag, Nt);
	Mat2complex_kernel << <grid, block >> > (device1_float, U0_device, 1.0f, Nt);
	Complex2mat_kernel << <grid, block >> > (U0_device, Amp_real, Phi_imag, false, Nt);
	CUDA_mul_kernel << <grid, block >> > (Amp_real, device2_float, Amp_real, Nt);
	CUDA_mul_kernel << <grid, block >> > (Phi_imag, device2_float, Phi_imag, Nt);
	
	//fftshift real
	fftshift_step1_kernel << <grid, block >> > (Amp_real, device1_float, device2_float, device3_float, device4_float, nH, nW, Nt);
	fftshift_step2_kernel << <grid, block >> > (device1_float, device2_float, device3_float, device4_float, Amp_real, nH, nW, Nt);
	Check(cudaMemcpy(real_out.data, (uchar*)Amp_real, nH * nW * sizeof(float), cudaMemcpyDeviceToHost));
	dst1 = real_out;
	//fftshift imag
	fftshift_step1_kernel << <grid, block >> > (Phi_imag, device1_float, device2_float, device3_float, device4_float, nH, nW, Nt);
	fftshift_step2_kernel << <grid, block >> > (device1_float, device2_float, device3_float, device4_float, Phi_imag, nH, nW, Nt);
	Check(cudaMemcpy(imag_out.data, (uchar*)Phi_imag, nH * nW * sizeof(float), cudaMemcpyDeviceToHost));
	dst2 = imag_out;
	//cudaStreamSynchronize(0);
}

void CUDA_ALL_calculate(cv::Mat Phi, cv::Mat real_filter, cv::Mat imag_filter, cv::Mat* Ipc, int nH_extend, int nW_extend, int i)
{
	int nH = real_filter.rows;
	int nW = real_filter.cols;
	int nH_old = nH - nH_extend * 2;
	int nW_old = nW - nW_extend * 2;
	int Nt = nH * nW;
	float a = 0.7f;

	float* device1_float = device1_float_out[i];
	float* device2_float = device2_float_out[i];
	float* device3_float = device3_float_out[i];
	float* device4_float = device4_float_out[i];
	float* Amp_real = Amp_real_out[i];
	float* Phi_imag = Phi_imag_out[i];
	cufftComplex* U0_device = U0_device_out[i];

	cufftHandle cufftForwrdHandle, cufftInverseHandle;

	cufftPlan2d(&cufftForwrdHandle, nH, nW, CUFFT_C2C);
	cufftPlan2d(&cufftInverseHandle, nH, nW, CUFFT_C2C);

	cufftSetStream(cufftForwrdHandle, stream[i]);
	cufftSetStream(cufftInverseHandle, stream[i]);

	cv::Mat Ipc_out = cv::Mat::zeros(nH_old, nW_old, CV_32FC1);

	Check(cudaMemcpyAsync(Phi_imag, Phi.data, nH * nW * sizeof(float), cudaMemcpyHostToDevice, stream[i]));
	Check(cudaMemcpyAsync(Amp_real, real_filter.data, nH * nW * sizeof(float), cudaMemcpyHostToDevice, stream[i]));

	dim3 grid((nH * nW + 1024 - 1) / 1024);//(1280*960+1024-1)/1024
	dim3 block(1024);//1024
	Mat2complex_kernel << <grid, block, 0, stream[i] >> > (Phi_imag, U0_device, a, Nt);
	//ִ��fft���任
	cufftExecC2C(cufftForwrdHandle, U0_device, U0_device, CUFFT_FORWARD);
	
	Complex2mat_kernel << <grid, block, 0, stream[i] >> > (U0_device, device1_float, device2_float, false, Nt);

	Check(cudaMemcpyAsync(Phi_imag, imag_filter.data, nH * nW * sizeof(float), cudaMemcpyHostToDevice, stream[i]));
	
	CUDA_mul_kernel_2 << <grid, block, 0, stream[i] >> > (Amp_real, Phi_imag, device1_float, device2_float, device3_float, device4_float, Nt);

	Mat2complex_kernel << <grid, block, 0, stream[i] >> > (device3_float, device4_float, U0_device, false, Nt);

	cufftExecC2C(cufftForwrdHandle, U0_device, U0_device, CUFFT_INVERSE);
	
	normalizing << <grid, block, 0, stream[i] >> > (U0_device, Nt);
	Complex2mat_kernel << <grid, block, 0, stream[i] >> > (U0_device, Amp_real, Phi_imag, true, Nt);
	
	CUDA_mul_kernel << <grid, block, 0, stream[i] >> > (Amp_real, Amp_real, device1_float, Nt);

	rect_kernel << <grid, block, 0, stream[i] >> > (device1_float, device3_float, nH, nW, nH_extend, nW_extend, nH_old, nW_old, Nt);

	Check(cudaMemcpyAsync(Ipc_out.data, (uchar*)device3_float, nH_old * nW_old * sizeof(float), cudaMemcpyDeviceToHost, stream[i]));
	*Ipc = Ipc_out;
	
}

//void CUDA_gaussianBlur_gpu(cv::Mat & src, cv::Mat & dst, int Gas_Radius, int Gas_var)
//{
//	cv::cuda::GpuMat src_gpu, dst_gpu;
//
//	src_gpu.upload(src);
//
//	cv::Ptr<cv::cuda::Filter> filter;
//	filter = cv::cuda::createGaussianFilter(CV_32FC1, CV_32FC1, cv::Size(Gas_Radius, Gas_Radius), Gas_var, Gas_var);
//	filter->apply(src_gpu, dst_gpu);
//	dst_gpu.download(dst);
//}