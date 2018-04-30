#include "mnist_loader.h"
#include "node.h"

#include <iostream>
#include <memory>
#include <Eigen/Core>

#define EIGEN_MPL2_ONLY

#define INPUT_SIZE  784
#define HIDDEN_SIZE 50
#define OUTPUT_SIZE 10

#define VAR_DIFF (double)(1.0E-6)

namespace {
	Eigen::VectorXf ReLu(Eigen::VectorXf &val)
	{
		Eigen::VectorXf ret(val.rows());

		for(int i = 0 ; i < val.rows() ; i++)
		{
			ret[i] = fmax(0, val[i]);
		}

		return ret;
	}

	Eigen::VectorXf Softmax(Eigen::VectorXf &val)
	{
		Eigen::VectorXf ret(val.rows());

		double sum = 0;
		for(int i = 0 ; i < val.rows() ; i++)
		{
			sum = exp(val[i]);
		}

		for(int i = 0 ; i < val.rows() ; i++)
		{
			ret[i] = exp(val[i]) / sum;
		}

		return ret;
	}

	Eigen::VectorXf MakeOnehotVector(int size, int index)
	{
		Eigen::VectorXf ret = Eigen::VectorXf::Zero(size);
		ret[index] = 1;
		return ret;
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
};

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
	Eigen::VectorXf temp;

	temp = node1.Calc(input);
	temp = ReLu(temp);
	temp = node2.Calc(temp);
	temp = Softmax(temp);

	return temp;
}

double CalcError(Node &node1, Node &node2, Eigen::VectorXf input, Eigen::VectorXf ans)
{
	Eigen::VectorXf output     = Predict(node1, node2, input);
	double          error_rate = SquareError(output, ans);

	return error_rate;
}

int main()
{
	std::vector<std::shared_ptr<Image> > train_image_list;
	std::vector<uint8_t>                 train_label_list;
	std::vector<std::shared_ptr<Image> > test_image_list;
	std::vector<uint8_t>                 test_label_list;

	Eigen::initParallel();
	Eigen::setNbThreads(4);

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
	if(node1.Load("weight_node1") == false)
	{
		printf("[INFO] node1 weight value initialize with random value.\n");
	}
	Node node2(HIDDEN_SIZE, OUTPUT_SIZE);
	if(node2.Load("weight_node2") == false)
	{
		printf("[INFO] node2 weight value initialize with random value.\n");
	}

	printf("[INFO] training...\n");
	for(int image_index = 0 ; image_index < train_image_list.size() ; image_index++)
	{
		if(image_index % 100 == 0)
		{
			printf("[INFO] exec %d/%d\n", image_index, (int)train_label_list.size());
		}
		std::shared_ptr<Image> image = train_image_list[image_index];
		Eigen::VectorXf        input = Arrayi2VectorXf(image->image, image->image_size);
		int                    ans   = train_label_list[image_index];
		Eigen::VectorXf        ans_vec = MakeOnehotVector(OUTPUT_SIZE, ans);

		input.normalize();

		double *node1_mod = new double[node1.GetMaxWeightIndex()];
		double *node2_mod = new double[node2.GetMaxWeightIndex()];

		for(int i = 0 ; i < node1.GetMaxWeightIndex() ; i++)
		{
			double dyp = 0;
			double dyn = 0;

			node1.PushWeightDiff(i, VAR_DIFF);
			dyp = CalcError(node1, node2, input, ans_vec);
			node1.PopWeightDiff();

			node1.PushWeightDiff(i, -VAR_DIFF);
			dyn = CalcError(node1, node2, input, ans_vec);
			node1.PopWeightDiff();

			node1_mod[i] = (dyp + dyn) / (2 * VAR_DIFF);
		}

		for(int i = 0 ; i < node2.GetMaxWeightIndex() ; i++)
		{
			double dyp = 0;
			double dyn = 0;

			node2.PushWeightDiff(i, VAR_DIFF);
			dyp = CalcError(node1, node2, input, ans_vec);
			node2.PopWeightDiff();

			node2.PushWeightDiff(i, -VAR_DIFF);
			dyn = CalcError(node1, node2, input, ans_vec);
			node2.PopWeightDiff();

			node2_mod[i] = (dyp + dyn) / (2 * VAR_DIFF);
		}

		for(int i = 0 ; i < node1.GetMaxWeightIndex() ; i++)
		{
			node1.AddWeight(i, node1_mod[i]);
		}

		for(int i = 0 ; i < node2.GetMaxWeightIndex() ; i++)
		{
			node2.AddWeight(i, node2_mod[i]);
		}

		delete[] node1_mod;
		delete[] node2_mod;

		if(image_index == 10)
		{
			// optimization test
			return 1;
		}
	}

	node1.Save("weight_node1");
	node2.Save("weight_node2");

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
