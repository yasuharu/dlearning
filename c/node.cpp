
#include "node.h"

#include "assert.h"

void Node::Initialize()
{
	weight_ = Eigen::MatrixXf::Random(output_size_, input_size_);
}

Eigen::MatrixXf Node::Calc(Eigen::VectorXf input)
{
	assert(input.rows() == weight_.cols());
	return weight_ * input;
}

