#include "Exercise.hpp"
#include "include/chronoGPU.hpp"
#include "thrust/host_vector.h"
#include "thrust/device_vector.h"
#include "thrust/transform.h"

void Exercise::Question1(const thrust::host_vector<int>& A,
												const thrust::host_vector<int>& B, 
												thrust::host_vector<int>&C) const
{
// TODO: addition of two vectors using thrust
	ChronoGPU chrUP, chrDOWN, chrGPU;

	// thrust::device_vector<int> d_A(A);
	// thrust::device_vector<int> d_B(B);
	// thrust::device_vector<int> d_C(A.size());

	for (int i=3; i--; ){
		chrUP.start();
		thrust::device_vector<int> d_A(A);
		thrust::device_vector<int> d_B(B);
		thrust::device_vector<int> d_C(A.size());
		chrUP.stop();

		chrGPU.start();
		thrust::transform(
			d_A.begin(), d_A.end(), 
			d_B.begin(),
			d_C.begin(),
			thrust::placeholders::_1+ thrust::placeholders::_2
		);
		chrGPU.stop();

		chrDOWN.start();
		C=d_C;
		chrDOWN.stop();
	}
	float elapsed = chrUP.elapsedTime() + chrDOWN.elapsedTime() + chrGPU.elapsedTime();
	std::cout << "Question1 done in " << elapsed << std::endl;
	std::cout <<" - UP time : " << chrUP.elapsedTime() << std::endl;
	std::cout <<" - GPU time : "<< chrGPU.elapsedTime() << std::endl;
	std::cout <<" - DOWN time : " << chrDOWN.elapsedTime() << std::endl;
}


void Exercise::Question2(thrust::host_vector<int>&A) const 
{
	ChronoGPU chrUP, chrDOWN, chrGPU;
  // TODO: addition using ad hoc iterators
	for (int i=3; i--; ){
		chrUP.start();
		thrust::counting_iterator<int>X(1);
		thrust::constant_iterator<int>Y(4);
		thrust::device_vector<int> gpuA(A.size());
		chrUP.stop();
		chrGPU.start();
		thrust::transform(
			X, X + A.size(),
		 	Y, gpuA.begin(),
		 	thrust::placeholders::_1+ thrust::placeholders::_2
		);
		chrGPU.stop();
		chrDOWN.start();
		A = gpuA;
		chrDOWN.stop();
	}
	float elapsed = chrUP.elapsedTime() + chrDOWN.elapsedTime() + chrGPU.elapsedTime();
	std::cout << "Question1 done in " << elapsed << std::endl;
	std::cout <<" - UP time : " << chrUP.elapsedTime() << std::endl;
	std::cout <<" - GPU time : "<< chrGPU.elapsedTime() << std::endl;
	std::cout <<" - DOWN time : " << chrDOWN.elapsedTime() << std::endl;
}



typedef thrust::tuple<int, int, int> myInt3;
class additionFunctor3: public thrust::unary_function<myInt3, int>{
	public:
		__device__ int operator()(const myInt3 &tuple){
			return thrust::get<0>(tuple) + thrust::get<1>(tuple) + thrust::get<2>(tuple);
		}
};

void Exercise::Question3(const thrust::host_vector<int>& A,
												const thrust::host_vector<int>& B, 
												const thrust::host_vector<int>& C, 
												thrust::host_vector<int>&D) const 
{
  // TODO
	ChronoGPU chrUP, chrDOWN, chrGPU;
	for (int i=3; i--; ){
		thrust::device_vector<int> gpuA = A;
		thrust::device_vector<int> gpuB = B;
		thrust::device_vector<int> gpuC = C;
		thrust::device_vector<int> gpuD(D.size());
		thrust::transform(
			thrust::make_zip_iterator(thrust::make_tuple(gpuA.begin(), gpuB.begin(), gpuC.begin())),
			thrust::make_zip_iterator(thrust::make_tuple(gpuA.end(), gpuB.end(), gpuC.end())), 
			gpuD.begin(), 
			additionFunctor3()
			// thrust::placeholders::_1+ thrust::placeholders::_2+ thrust::placeholders::_3
		);
		D=gpuD(D);
	}
}
