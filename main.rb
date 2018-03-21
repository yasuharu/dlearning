require "mnist"

class Node
  def initialize(input_size, output_size)
    @weight_size = input_size * output_size
    @weight      = Array.new(@weight_size, 0)
		init_weight

    @input_size  = input_size
    @output_size = output_size
  end

	def init_weight
		@weight_size.times do |index|
			@weight[index] = Random.rand(1.0)
		end
	end

  def calc(input)
    if input.size != @input_size
      p input.size
      p @input_size
      raise ArgumentException
    end

    output = Array.new(@output_size, 0)
    @output_size.times do |output_index|
      @input_size.times do |input_index|
        weight_index = output_index * @input_size + input_index
        if weight_index >= @weight_size
          p weight_index
          p @weight_size
          raise Exception
        end
        weight = @weight[weight_index]

        output[output_index] += weight * input[input_index]
      end
    end

    return output
  end

	def get_weight_size
		return @weight_size
	end

	def get_weight(index)
		raise ArgumentException if index >= @weight_size
		return @weight[index]
	end

	def set_weight(index, val)
		raise ArgumentException if index >= @weight_size
		@weight[index] = val
	end
end

def normalize(vec)
  sum      = vec.inject(0) { |s, v| s + (v * v) }
  vec_size = Math.sqrt(sum)
  vec.map{ |v| v / vec_size }
end

def max_index(vec)
  index = 0
  vec.each_with_index do |v, i|
    if v > vec[index]
      index = i
    end
  end
  return index
end

def softmax(vec)
  sum_exp = 0
	vec.map! do |v|
		v = Math.exp(v)
		sum_exp += v
		v
	end

	vec.map! { |v| v / sum_exp }
end

def square_error(result, expect)
	if result.size != expect.size
		raise ArgumentException
	end

	sum = 0
	result.size.times do |index|
		v = (result[index] - expect[index])
		sum += (v * v)
	end

	return sum
end

def one_hot(size, one_hot_index)
	a = Array.new(size, 0)
	a[one_hot_index] = 1
	return a
end

# main

INPUT_SIZE  = 784
HIDDEN_SIZE = 50
OUTPUT_SIZE = 10
GRAD_DELTA  = 0.0001
node1 = Node.new(INPUT_SIZE , HIDDEN_SIZE)
node2 = Node.new(HIDDEN_SIZE, OUTPUT_SIZE)

train_images = Mnist.load_images('train-images-idx3-ubyte.gz')[2]
train_labels = Mnist.load_labels('train-labels-idx1-ubyte.gz')

def predict(node1, node2, input)
  temp    = node1.calc input
#	printf "[INFO] node1 output : "
#	p temp
  output  = node2.calc temp

#	printf "[INFO] predict before softmax : "
#	p output
	output = softmax output
end

def calc_error(node1, node2, input, expect)
	output = predict(node1, node2, input)
#	printf "[INFO] predict : "
#	p output
	error  = square_error(output, expect)
#	printf "[INFO] error : "
#	p error
end

Random.new(1)

# training
train_images.each_with_index do |input, train_index|
  input_i = input.unpack("C*")
  input_n = normalize input_i

	node_list = [node1, node2]
	grad_list = [Array.new(node1.get_weight_size, 0), Array.new(node2.get_weight_size, 0)]

	# calc node1 grad
	node_list.size.times do |node_index|
		node = node_list[node_index]
		node.get_weight_size.times do |weight_index|
			temp   = node.get_weight(weight_index)
			expect = one_hot OUTPUT_SIZE, train_labels[train_index]

#			p expect

			# calc f(x+h)
			node.set_weight(weight_index, temp + GRAD_DELTA)
			val1 = calc_error(node1, node2, input_n, expect)

			# calc f(x-h)
			node.set_weight(weight_index, temp - GRAD_DELTA)
			val2 = calc_error(node1, node2, input_n, expect)

			# calc grad
			grad_list[node_index][weight_index] = (val1 + val2) / 2

			printf "[INFO] grad = %f, node = %d, weight = %d\n",
				grad_list[node_index][weight_index], node_index, weight_index
		end
	end
end

# check
test_images = Mnist.load_images('t10k-images-idx3-ubyte.gz')[2]
test_labels = Mnist.load_labels('t10k-labels-idx1-ubyte.gz')

test_count    = test_images.size
success_count = 0
fail_count    = 0
test_images.each_with_index do |input, i|
  input_i = input.unpack("C*")
  input_n = normalize input_i

  temp    = node1.calc input_n
  output  = node2.calc temp

  result = max_index output
  expect = test_labels[i]

  if result == expect
    success_count += 1
  else
    fail_count    += 1
  end
  printf "[%d/%d] result = %d, expect = %d\n", (success_count + fail_count), test_count, result, expect
end

accuracy = success_count / (success_count + fail_count).to_f
printf "accuracy = %f\n", accuracy
