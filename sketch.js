let nn = new NeuralNetwork(2, 2, 1);

let training_data = [
  {
    inputs: [0, 0],
    targets: [0]
  },
  {
    inputs: [0, 1],
    targets: [1]
  },
  {
    inputs: [1, 0],
    targets: [1]
  },
  {
    inputs: [1, 1],
    targets: [0]
  }
];

for (let i = 0; i <= 50000; i++) {
  const example = Math.floor(Math.random() * 4);
  nn.train(training_data[example].inputs, training_data[example].targets);
}

console.log('False xor False:');
console.log(nn.feedForward([0, 0], [0]));
console.log('False xor True:');
console.log(nn.feedForward([0, 1], [1]));
console.log('True xor False:');
console.log(nn.feedForward([1, 0], [1]));
console.log('True xor True:');
console.log(nn.feedForward([1, 1], [0]));
