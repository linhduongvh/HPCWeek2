#include <iostream>
#include "student.hpp"

// do not forget to add the needed included files
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/scatter.h>
#include <thrust/iterator/transform_iterator.h>

// ==========================================================================================
// Exercise 1
// Feel free to add any function you need, it is your file ;-)
struct scanFunction : public thrust::unary_function<ColoredObject,unsigned>
{
	__host__ __device__ unsigned operator()(const ColoredObject& object)
  {
    return unsigned(object == ColoredObject::BLUE);
  }
};


// mandatory function: should returns the blue objects contained in the input parameter
thrust::host_vector<ColoredObject> compactBlue( const thrust::host_vector<ColoredObject>& input ) {
	// thrust::host_vector<ColoredObject> answer;

	thrust::device_vector<ColoredObject> dInput(input);
	thrust::device_vector<ColoredObject> dScan(input.size());

	const int size = input.size();
	// std::cout << size;

	thrust::inclusive_scan(
		thrust::make_transform_iterator(
			dInput.begin(),
			scanFunction()
			),
		thrust::make_transform_iterator(
			dInput.end(),
			scanFunction()
			),
		dScan.begin()
	);
	const unsigned noBlueObjects = dScan[size - 1];
	thrust::device_vector<ColoredObject> answer(noBlueObjects);
	thrust::scatter_if(
		dInput.begin(),
		dInput.end(),
		thrust::make_transform_iterator(dScan.begin(),thrust::placeholders::_1-1 ),
		thrust::make_transform_iterator(dInput.begin(), scanFunction() ),
		answer.begin()
	);

	return answer;
}

// ==========================================================================================
// Exercise 2
// Feel free to add any function you need, it is your file ;-)
struct WhereToGoFunctor: public thrust::unary_function<thrust::tuple<unsigned,unsigned,unsigned>, unsigned
{
	
};
thrust::host_vector<int> radixSort( const thrust::host_vector<int>& h_input ) {
	thrust::host_vector<int> answer;


	return answer;
}

// ==========================================================================================
// Exercise 3
// Feel free to add any function you need, it is your file ;-)
thrust::host_vector<int> quickSort( const thrust::host_vector<int>& h_input ) {
	thrust::host_vector<int> answer;
	return answer;
}
