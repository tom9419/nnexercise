#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven
from model.mlp import MultilayerPerceptron

from report.evaluator import Evaluator
from report.performance_plot import PerformancePlot


def main():
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000,
                                                    oneHot=False)
    myMLPClassifier = MultilayerPerceptron(data.trainingSet,
                                        data.validationSet,
                                        data.testSet,
                                        loss='ce',
                                        layers=[128, 150],
                                        learningRate=0.005,
                                        epochs=10)

    # Report the result #

    # Train the classifiers
    print("=========================")
    print("Training..")

    print("\nMultilayer Perceptron has been training..")
    myMLPClassifier.train()
    print("Done..")

    # Do the recognizer
    # Explicitly specify the test set to be evaluated
    mlpPred = myMLPClassifier.evaluate()
    
    # Report the result
    print("=========================")
    evaluator = Evaluator()

    print("\nResult of the Multi Layer Perceptron recognizer:")
    evaluator.printComparison(data.testSet, mlpPred)
    evaluator.printAccuracy(data.testSet, mlpPred)
    
    # Draw
    plot = PerformancePlot("Multi Layer Perceptron validation")
    plot.draw_performance_epoch(myMLPClassifier.performances,
                                myMLPClassifier.epochs)
    
    
if __name__ == '__main__':
    main()
