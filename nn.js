function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}
function dsigmoid(x) {
  return x * (1 - x);
}

class NeuralNetwork {
  constructor(input_nodes, hidden_nodes, output_nodes) {
    this.input_nodes = input_nodes;
    this.hidden_nodes = hidden_nodes;
    this.output_nodes = output_nodes;

    this.weights_ih = new Matrix(this.hidden_nodes, this.input_nodes);
    this.weights_ho = new Matrix(this.output_nodes, this.hidden_nodes);
    this.weights_ih.randomize();
    this.weights_ho.randomize();

    this.bias_h = new Matrix(this.hidden_nodes, 1);
    this.bias_o = new Matrix(this.output_nodes, 1);
    this.bias_h.randomize();
    this.bias_o.randomize();

    this.learning_rate = 0.1;

    console.log('Weights:');
    console.table(this.weights_ih.matrix);
    console.table(this.weights_ho.matrix);
    console.log(''); console.log('');
    console.log('Biases:');
    console.table(this.bias_h.matrix);
    console.table(this.bias_o.matrix);
    console.log(''); console.log('');
  }

  feedForward(input_array) {
    let inputs = Matrix.fromArray(input_array);

    //Dot product -> biases -> activation function (input to hidden)
    let hidden = Matrix.dot(this.weights_ih, inputs);
    hidden.add(this.bias_h);
    hidden = Matrix.map(hidden, sigmoid);

    //Dot product -> biases -> activation function (hidden to output)
    let output = Matrix.dot(this.weights_ho, hidden);
    output.add(this.bias_o);
    output = Matrix.map(output, sigmoid);

    return output.toArray();
  }

  train(inputs_array, targets_array) {
    let inputs = Matrix.fromArray(inputs_array);
    let targets = Matrix.fromArray(targets_array);

    //Feed forward
    let hidden = Matrix.dot(this.weights_ih, inputs);
    hidden.add(this.bias_h);
    hidden = Matrix.map(hidden, sigmoid);

    let outputs = Matrix.dot(this.weights_ho, hidden);
    outputs.add(this.bias_o);
    outputs = Matrix.map(outputs, sigmoid);

    //Back propagation (output to hidden)
    let output_errors = Matrix.subtract(targets, outputs);

    let gradients = Matrix.map(outputs, dsigmoid);
    gradients.multiply(output_errors);
    gradients.multiply(this.learning_rate);

    let hidden_t = Matrix.transpose(hidden);
    let weights_ho_deltas = Matrix.dot(gradients, hidden_t);

    this.weights_ho.add(weights_ho_deltas);
    this.bias_o.add(gradients);

    //Back propagation (hidden to input)
    let weights_ho_t = Matrix.transpose(this.weights_ho);
    let hidden_errors = Matrix.dot(weights_ho_t, output_errors);

    let hidden_gradients = Matrix.map(hidden, dsigmoid);
    hidden_gradients.multiply(hidden_errors);
    hidden_gradients.multiply(this.learning_rate);

    let inputs_t = Matrix.transpose(inputs);
    let weights_ih_deltas = Matrix.dot(hidden_gradients, inputs_t);

    this.weights_ih.add(weights_ih_deltas);
    this.bias_h.add(hidden_gradients);
  }
}
