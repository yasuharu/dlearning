
#include <Eigen/Core>

class Node
{
	public:
		Node(int input_size, int output_size)
			: input_size_(input_size), output_size_(output_size)
		{
			Initialize();
		}

		Eigen::MatrixXf Calc(Eigen::VectorXf input);

	private:
		void Initialize();

		const int input_size_;
		const int output_size_;

		Eigen::MatrixXf weight_;
};
