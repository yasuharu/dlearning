require "mnist"

train_images = Mnist.load_images('train-images-idx3-ubyte.gz')[2]
train_labels = Mnist.load_labels('train-labels-idx1-ubyte.gz')

class Node
  def initialize(input_size, output_size)
    @weight_size = input_size * output_size
    @weight      = Array.new(@weight_size, 0.1)

    @input_size  = input_size
    @output_size = output_size
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

# main

INPUT_SIZE  = 784
OUTPUT_SIZE = 10
node1 = Node.new(INPUT_SIZE, INPUT_SIZE)
node2 = Node.new(INPUT_SIZE, OUTPUT_SIZE)

# vec = normalize [2, 1]
# p vec
# p Math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
# exit
# train_images.each do |input|
#   input_i = input.unpack("C*")
#   input_n = normalize input_i
# 
#   temp    = node1.calc input_n
#   output  = node2.calc temp
# 
# #   p output
# end

test_images = Mnist.load_images('t10k-images-idx3-ubyte.gz')[2]
test_labels = Mnist.load_labels('t10k-labels-idx1-ubyte.gz')

success_count = 0
fail_count    = 0
test_images.each_with_index do |input, i|
  input_i = input.unpack("C*")
  input_n = normalize input_i

  temp    = node1.calc input_n
  output  = node2.calc temp

  p output
  result = max_index output
  expect = test_labels[i]

  if result == expect
    success_count += 1
  else
    fail_count    += 1
  end
  printf "result = %d, expect = %d\n", result, expect
end

accuracy = success_count / (success_count + fail_count)
printf "accuracy = %f\n", accuracy
