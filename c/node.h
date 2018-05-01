
#include <Eigen/Core>

class Node
{
	public:
		Node(int input_size, int output_size)
			: input_size_(input_size), output_size_(output_size),
			  weight_diff_index(-1), inxout_size_(input_size * output_size),
				weight_size_(input_size * output_size + output_size)
		{
			Initialize();
		}

		Eigen::MatrixXf Calc(Eigen::VectorXf input);

		int GetMaxWeightIndex()
		{
			return weight_size_;
		}

		void PushWeightDiff(int index, double diff);
		void PopWeightDiff();
		void AddWeight(int index, double var);

		bool Load(const char* file_name);
		bool Save(const char* file_name);

	private:
		void Initialize();

		int input_size_;
		int output_size_;

		// input_size * output_size
		int inxout_size_;

		// input_size * output_size + output_size
		int weight_size_;

		Eigen::MatrixXf weight_;
		Eigen::VectorXf offset_;

		int    weight_diff_index;
		double weight_diff_temp;

		double GetWeight(int index);
		void   SetWeight(int index, double value);
};
