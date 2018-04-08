#include "mnist_loader.h"
#include "node.h"

#include <iostream>
#include <Eigen/Core>

#define EIGEN_MPL2_ONLY

#define INPUT_SIZE  784
#define HIDDEN_SIZE 50
#define OUTPUT_SIZE 10

Eigen::VectorXf Arrayi2VectorXf(uint8_t *array, int size)
{
	Eigen::VectorXf ret(size);

	// this conversion needs due to different data format
	for(int i = 0 ; i < size ; i++)
	{
		ret[i] = array[i];
	}

	return ret;
}

Eigen::VectorXf Predict(Node &node1, Node &node2, Eigen::VectorXf input)
{
		Eigen::VectorXf temp   = node1.Calc(input);
		Eigen::VectorXf output = node2.Calc(temp);

		return output;
}

float SquareError(Eigen::VectorXf vec1, Eigen::VectorXf vec2)
{
	assert(vec1.rows() == vec2.rows());
	assert(vec1.cols() == vec2.cols());
	assert(vec1.cols() == 1);

	float error = 0;
	for(int i = 0 ; i < vec1.rows() ; i++)
	{
		float v = vec1[i] - vec2[i];
		error += v * v;
	}

	return error;
}

int MaxIndex(Eigen::VectorXf vec)
{
	int max_index = 0;
	for(int i = 0 ; i < vec.rows() ; i++)
	{
		if(vec[max_index] < vec[i])
		{
			max_index = i;
		}
	}
	return max_index;
}

int main()
{
	std::vector<std::shared_ptr<Image> > train_image_list;
	std::vector<uint8_t>                 train_label_list;
	std::vector<std::shared_ptr<Image> > test_image_list;
	std::vector<uint8_t>                 test_label_list;

	printf("[INFO] load image...\n");
	std::shared_ptr<MnistLoader> loader = std::make_shared<MnistLoader>();
	if(loader->LoadImage("../train-images-idx3-ubyte", train_image_list) == false)
	{
		printf("[ERROR] load data error.\n");
	}

	if(loader->LoadLabel("../train-labels-idx1-ubyte", train_label_list) == false)
	{
		printf("[ERROR] load data error.\n");
	}

	if(loader->LoadImage("../t10k-images-idx3-ubyte", test_image_list) == false)
	{
		printf("[ERROR] load data error.\n");
	}

	if(loader->LoadLabel("../t10k-labels-idx1-ubyte", test_label_list) == false)
	{
		printf("[ERROR] load data error.\n");
	}

	Node node1(INPUT_SIZE, HIDDEN_SIZE);
	Node node2(HIDDEN_SIZE, OUTPUT_SIZE);

	printf("[INFO] training...\n");

	printf("[INFO] test...\n");
	std::vector<uint8_t> result_list;
	for(int image_index = 0 ; image_index < test_image_list.size() ; image_index++)
	{
		std::shared_ptr<Image> image = test_image_list[image_index];
		Eigen::VectorXf        input = Arrayi2VectorXf(image->image, image->image_size);

		input.normalize();
		Eigen::VectorXf output = Predict(node1, node2, input);

		int result = MaxIndex(output);
		result_list.push_back(result);
	}

	assert(result_list.size() == test_label_list.size());
	int success_count = 0;
	int fail_count    = 0;
	for(int i = 0 ; i < result_list.size() ; i++)
	{
		if(result_list[i] == test_label_list[i])
		{
			success_count++;
		}else
		{
			fail_count++;
		}
	}
	float accuracy = success_count / (float)(success_count + fail_count);
	std::cout << "accuracy = " << accuracy << std::endl;

	return 0;
}
