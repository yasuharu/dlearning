
#include "node.h"

#include <stdio.h>

#include "assert.h"

void Node::Initialize()
{
	weight_ = Eigen::MatrixXf::Random(output_size_, input_size_);
	offset_ = Eigen::VectorXf::Random(output_size_);
}

Eigen::MatrixXf Node::Calc(Eigen::VectorXf input)
{
	assert(input.rows() == weight_.cols());
	return weight_ * input + offset_;
}

int Node::GetMaxWeightIndex()
{
	return input_size_ * output_size_ + output_size_;
}

void Node::PushWeightDiff(int index, double diff)
{
	assert(weight_diff_index == -1);
	assert(index < GetMaxWeightIndex());

	weight_diff_index = index;
	weight_diff_temp  = GetWeight(index);

	SetWeight(index, weight_diff_temp + diff);
}

void Node::PopWeightDiff()
{
	assert(weight_diff_index != -1);

	SetWeight(weight_diff_index, weight_diff_temp);

	weight_diff_index = -1;
}

void Node::AddWeight(int index, double var)
{
	assert(index < GetMaxWeightIndex());

	SetWeight(index, GetWeight(index) + var);
}

double Node::GetWeight(int index)
{
	if(index < input_size_ * output_size_)
	{
		int row_index = index / input_size_;
		int col_index = index % input_size_;

		return weight_(row_index, col_index);
	}else
	{
		int row_index = index - (input_size_ * output_size_);

		return offset_[row_index];
	}
}

void Node::SetWeight(int index, double value)
{
	if(index < input_size_ * output_size_)
	{
		int row_index = index / input_size_;
		int col_index = index % input_size_;

		weight_(row_index, col_index) = value;
	}else
	{
		int row_index = index - (input_size_ * output_size_);

		offset_[row_index] = value;
	}
}

bool Node::Load(const char *file_name)
{
	FILE *fp = fopen(file_name, "r");
	if(fp == NULL)
	{
		return false;
	}

	int input_size, output_size;
	fscanf(fp, "%d,%d,", &input_size, &output_size);

	if(input_size_ != input_size || output_size_ != output_size)
	{
		printf("[ERROR] Matrix size is not same.\n");
		return false;
	}

	input_size_  = input_size;
	output_size_ = output_size;

	for(int i = 0 ; i < GetMaxWeightIndex() ; i++)
	{
		double val;
		fscanf(fp, "%lf,", &val);
		SetWeight(i, val);
	}

	fclose(fp);

	return true;
}

bool Node::Save(const char *file_name)
{
	FILE *fp = fopen(file_name, "w");
	if(fp == NULL)
	{
		return false;
	}

	fprintf(fp, "%d,%d,", input_size_, output_size_);

	for(int i = 0 ; i < GetMaxWeightIndex() ; i++)
	{
		fprintf(fp, "%lf,", GetWeight(i));
	}

	fclose(fp);

	return true;
}

