package dlTest;

import java.io.IOException;
import java.util.Collections;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public class Main {

  public static void log(String msg) {
    System.out.println(msg);
  }
  
  
  public static void main(String[] args) throws IOException {
    
    log("Configure model...");
    
    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(100)
            .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
            .gradientNormalizationThreshold(1.0d)
            .weightInit(WeightInit.XAVIER)
            .iterations(10)
            .momentum(0.5)
            .momentumAfter(Collections.singletonMap(3, 0.9))
            .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
            .list()
            .layer(0, new RBM.Builder()
                    .nIn(784).nOut(300)
                    .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                    .visibleUnit(RBM.VisibleUnit.BINARY)
                    .hiddenUnit(RBM.HiddenUnit.BINARY)
                    .build()
            )
            .layer(1, new RBM.Builder()
                    .nIn(300).nOut(300)
                    .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                    .visibleUnit(RBM.VisibleUnit.BINARY)
                    .hiddenUnit(RBM.HiddenUnit.BINARY)
                    .build()
            )
            .layer(2, new RBM.Builder()
                    .nIn(300).nOut(300)
                    .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                    .visibleUnit(RBM.VisibleUnit.BINARY)
                    .hiddenUnit(RBM.HiddenUnit.BINARY)
                    .build()
            )
            .layer(0, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                    .activation("softmax")
                    .nIn(300).nOut(300).build())
            .pretrain(true).backprop(false)
            .build();
    
    log("Init model...");
    
    MultiLayerNetwork model = new MultiLayerNetwork(conf);
    model.init();

    log("Loading data...");
    
    DataSetIterator dataIter = new MnistDataSetIterator(300, 100, true);
    
    log("Training model...");
    
    model.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(100)));
    model.fit(dataIter);
  
  }
  
}
