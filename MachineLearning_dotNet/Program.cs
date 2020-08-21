using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Reflection.Emit;
using CNTK;
using CNTKUtil;

namespace MachineLearning_dotNet
{
    class Program
    {
        static readonly int latentDimensions = 32;
        static readonly int imageHeight = 32;
        static readonly int imageWidth = 32;
        static readonly int channels = 3;
        static void Main(string[] args)
        {
            foreach (var item in CNTK.DeviceDescriptor.AllDevices())
            {
                Console.WriteLine($"{item.Id}: {item.Type.ToString()}");
            }
            /// Data loading
            if (!File.Exists("./x_channels_first_8_5.bin"))
            {
                Console.WriteLine("Unpacking archive...");
                ZipFile.ExtractToDirectory("../../../../Resources/frog_pictures.zip", "./");
            }
            Console.WriteLine("Loading data files..");
            float[][] trainingData = DataUtil.LoadBinary<float>("./x_channels_first_8_5.bin", 5000, channels * imageHeight * imageWidth);

            ///  Generator Input Variable
            Variable generatorVar = CNTK.Variable.InputVariable(new int[] { latentDimensions }, DataType.Float, "generator_input");

            // Generator Architecture
            Function generator = generatorVar.Dense(128 * 16 * 16, v => CNTK.CNTKLib.LeakyReLU(v, 0.1))
                                             .Reshape(new int[] { 16, 16, 128 })
                                             .Convolution2D(256, new int[] { 5, 5 }, padding: true, activation: v => CNTK.CNTKLib.LeakyReLU(v, 0.1))
                                             .ConvolutionTranspose(
                                                 filterShape: new int[] { 4, 4 },
                                                 numberOfFilters: 128,
                                                 strides: new int[] { 2, 2 },
                                                 outputShape: new int[] { 32, 32 },
                                                 padding: true,
                                                 activation: v => CNTK.CNTKLib.LeakyReLU(v, 0.1)
                                             )
                                             .Convolution2D(256, new int[] { 3, 3 }, padding: true, activation: v => CNTK.CNTKLib.LeakyReLU(v, 0.1))
                                             .Convolution2D(256, new int[] { 5, 5 }, padding: true, activation: v => CNTK.CNTKLib.LeakyReLU(v, 0.1))
                                             .Convolution2D(256, new int[] { 7, 7 }, padding: true, activation: v => CNTK.CNTKLib.Tanh(v))
                                             .Convolution2D(channels, new int[] { 7, 7 }, padding: true, activation: CNTK.CNTKLib.Tanh)
                                             .ToNetwork();
            //                             .Dense(128 * 16 * 16, var => CNTK.CNTKLib.LeakyReLU(var, 0.1))
            //                             .Reshape(new int[] { 16, 16, 128 })
            //                             .Convolution2D(256, new int[] { 5, 5 }, padding: true, activation: var => CNTK.CNTKLib.LeakyReLU(var, 0.1))
            //                             .ConvolutionTranspose(
            //                                filterShape: new int[] { 4, 4 },
            //                                numberOfFilters: 256,
            //                                strides: new int[] { 2, 2 },
            //                                outputShape: new int[] { 32, 32 },
            //                                padding: true,
            //                                activation: var => CNTK.CNTKLib.LeakyReLU(var, 0.1))
            //                             .Convolution2D(256, new int[] { 5, 5 }, padding: true, activation: var => CNTK.CNTKLib.LeakyReLU(var, 0.1))
            //                             .Convolution2D(256, new int[] { 5, 5 }, padding: true, activation: var => CNTK.CNTKLib.LeakyReLU(var, 0.1))
            //                             .Convolution2D(channels, new int[] { 7, 7 }, padding: true, activation: CNTK.CNTKLib.Tanh)
            //                             .ToNetwork();

            // Discriminator Input Variable
            Variable discriminatorVar = CNTK.Variable.InputVariable(new int[] { imageWidth, imageHeight, channels }, DataType.Float, name: "discriminator_input");

            // Discriminator Architecture
            Function discriminator = discriminatorVar.Convolution2D(32, new int[] { 7, 7 }, padding: true, activation: v => CNTK.CNTKLib.LeakyReLU(v, 0.1))
                                                     .Convolution2D(32, new int[] { 5, 5 }, padding: true, activation: v => CNTK.CNTKLib.LeakyReLU(v, 0.1))
                                                     .Pooling(PoolingType.Max, new int[] { 5, 5 }, new int[] { 1, 1 })
                                                     .Convolution2D(64, new int[] { 5, 5 }, padding: true, activation: v => CNTK.CNTKLib.LeakyReLU(v, 0.1))
                                                     .Convolution2D(64, new int[] { 5, 5 }, padding: true, activation: v => CNTK.CNTKLib.LeakyReLU(v, 0.1))
                                                     .Pooling(PoolingType.Max, new int[] { 5, 5 }, new int[] { 1, 1 })
                                                     .Convolution2D(128, new int[] { 4, 4 }, padding: true, strides: new int[] { 2, 2 }, activation: v => CNTK.CNTKLib.LeakyReLU(v, 0.1))
                                                     .Convolution2D(256, new int[] { 4, 4 }, padding: true, strides: new int[] { 2, 2 }, activation: v => CNTK.CNTKLib.LeakyReLU(v, 0.1))
                                                     .Pooling(PoolingType.Max, new int[] { 4, 4 }, new int[] { 1, 1 })
                                                     .Convolution2D(512, new int[] { 3, 3 }, strides: new int[] { 2, 2 }, activation: v => CNTK.CNTKLib.Tanh(v))
                                                     .Dropout(0.2)
                                                     .Dense(1, CNTK.CNTKLib.Sigmoid)
                                                     .ToNetwork();

            // +++ Create GAN +++
            Function gan = Gan.CreateGan(generator, discriminator);

            // Output of GAN
            Variable labelVar = CNTK.Variable.InputVariable(new NDShape(0), DataType.Float, "label_var");

            // Loss Function
            Function discriminatorLoss = CNTK.CNTKLib.BinaryCrossEntropy(discriminator, labelVar);
            Function ganLoss = CNTK.CNTKLib.BinaryCrossEntropy(gan, labelVar);


            // Set up the algorithms for traning discriminator and the GAN (learners)
            Learner discriminatorLearner = discriminator.GetAdaDeltaLearner(1);
            Learner ganLearner = gan.GetAdaDeltaLearner(0.25);

            // Set up the trainers for calculating [_ the discriminator and gan loss _] during each training epoch
            Trainer discriminatorTrainer = discriminator.GetTrainer(discriminatorLearner, discriminatorLoss, discriminatorLoss);
            Trainer ganTrainer = gan.GetTrainer(ganLearner, ganLoss, ganLoss);


            /// START Training!
            string outputFolder = "./images";
            if (!Directory.Exists(outputFolder))
                Directory.CreateDirectory(outputFolder);

            Console.WriteLine("Training GAN...");

            int numEpoches = 100_000;
            int batchSize = 15;
            int start = 0;
            for (int epoch = 0; epoch < numEpoches; epoch++)
            {
                // create a set of fake images..
                var generatedImages = Gan.GenerateImages(generator, batchSize, latentDimensions);
                start = Math.Min(start, trainingData.Length - batchSize);


                ///////////////////////COPIED 
                var batch = Gan.GetTrainingBatch(discriminatorVar, generatedImages, trainingData, batchSize, start);
                start += batchSize;
                if (start >= trainingData.Length)
                {
                    start = 0;
                }

                // train the discriminator
                //var discriminatorResult = discriminatorTrainer.TrainMinibatch(
                //    new Dictionary<Variable, Value> { { discriminatorVar, batch.featureBatch }, { labelVar, batch.labelBatch } }, false, CNTK.DeviceDescriptor.GPUDevice(0)
                //    );
                var discriminatorResult = discriminatorTrainer.TrainBatch(
                    new[] {
                            (discriminatorVar, batch.featureBatch),
                            (labelVar, batch.labelBatch)
                    }, true);

                // get a misleading batch: all fake images but labelled as real
                var misleadingBatch = Gan.GetMisleadingBatch(gan, batchSize, latentDimensions);

                // train the gan
                var ganResult = ganTrainer.TrainBatch(
                    new[] {
                            (gan.Arguments[0], misleadingBatch.featureBatch),
                            (labelVar, misleadingBatch.labelBatch)
                    }, true);
                // report result every 100 epochs
                if (epoch % 500 == 0)
                {
                    Console.WriteLine($"Epoch: {epoch}, Discriminator loss: {discriminatorResult.Loss}, Gan loss: {ganResult.Loss}");
                }
                // save files every 1000 epochs
                if (epoch % 500 == 0)
                {
                    // save a generated image
                    var path = Path.Combine(outputFolder, $"generated_frog_{epoch}.png");
                    Gan.SaveImage(generatedImages[0].ToArray(), imageWidth, imageHeight, path);
                    path = Path.Combine(outputFolder, $"sample_frog_{start}.png");
                    Gan.SaveImage(trainingData[start], imageWidth, imageHeight, path);

                    // save an actual image for comparison
                    // path = Path.Combine(outputFolder, $"actual_frog_{epoch}.png");
                    // Gan.SaveImage(trainingData[Math.Max(start - batchSize, 0)], imageWidth, imageHeight, path);
                }
                
            }
        }
    }
}
