from NeuralNetwork import Network
zero = [
  0, 1, 1, 0,
  1, 0, 0, 1,
  1, 0, 0, 1,
  1, 0, 0, 1,
  0, 1, 1, 0
]

one = [
  0, 0, 1, 0,
  0, 0, 1, 0,
  0, 0, 1, 0,
  0, 0, 1, 0,
  0, 0, 1, 0
]

two = [
  0, 1, 1, 0,
  1, 0, 0, 1,
  0, 0, 1, 0,
  0, 1, 0, 0,
  1, 1, 1, 1
]

three = [
  1, 1, 1, 1,
  0, 0, 0, 1,
  0, 1, 1, 1,
  0, 0, 0, 1,
  1, 1, 1, 1
]

exampleZero = [zero, [0, 0]]
exampleOne = [one, [0, 1]]
exampleTwo = [two, [1, 0]]
exampleThree = [three, [1, 1]]
examples = [exampleZero,exampleOne,exampleTwo,exampleThree]

network = Network()
network.addLayer(10, 20)
network.addLayer(2, None)
network.train(examples)

print network.process(zero)
print network.process(one)
print network.process(two)
print network.process(three)