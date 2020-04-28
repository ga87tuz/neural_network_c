#include <stdio.h>
#include <stdlib.h>
#include<math.h>
#include <time.h>

#define DEBUG 1

//config
//#define NR_OUTPUTS 1
#define ETA 0.01
#define NR_LAYERS 5
#define MAX_UNITS_PER_LAYER 5

#define MAX_NR_TRAINING_DATA 1000
#define MAX_NR_TEST_DATA 1000
#define NR_INPUTS 2 //programm needs to be adjusted

#define MAX_ITER 100000

int main()
{
    //config
    int maxUnitsLayerX[NR_LAYERS];
    maxUnitsLayerX[0] = 2;
    maxUnitsLayerX[1] = 3;
    maxUnitsLayerX[2] = 5;
    maxUnitsLayerX[3] = 3;
    maxUnitsLayerX[4] = 1;

    //srand(time(NULL));
    srand(1);


//-------------------------------Struct-----------------------
//    typedef struct {
//        double weights[100][100];
//        double In[100];
//        double Out[100];
//        double bias;
//    }NEURON;

    typedef struct {
        //NEURON neurons[100];
        double weights[MAX_UNITS_PER_LAYER+1][MAX_UNITS_PER_LAYER+1];
        double dWeights[MAX_UNITS_PER_LAYER+1][MAX_UNITS_PER_LAYER+1];
        double delta[MAX_UNITS_PER_LAYER+1];
        double in[MAX_UNITS_PER_LAYER+1];
        double out[MAX_UNITS_PER_LAYER+1];
        //double bias;
        int nrUnits;
    }LAYER;

    typedef struct {
        LAYER layers[NR_LAYERS];
        int nrLayer;
        double overAllError;
    }NN;
//-------------------------------END Struct-----------------------


//-------------------------------Functions-----------------------
    void setRandomWeights(NN *nn){
        int i=0;
        int j=0;
        int n=0;
        for(n=0; n<nn->nrLayer-1; n++){
            for(i=0; i<nn->layers[n].nrUnits+1;i++){
                for(j=0; j<nn->layers[n+1].nrUnits+1;j++){
                    nn->layers[n].weights[i][j]=((((double)rand()/(double)RAND_MAX)*2)-1)/10.0;
                }
            }
        }
    }

    void normalize(int nrTrainData, int nrTestData, double *trainData, double *testData){
        //find min and max of data
        double maxVal;
        double minVal;
        maxVal = trainData[0];
        minVal = trainData[0];
        int t=0;
        for(t=0;t<nrTrainData;t++){
            if(trainData[t]>maxVal){
                maxVal = trainData[t];
            }
            if(trainData[t]<minVal){
                minVal = trainData[t];
            }
        }
        for(t=0;t<nrTestData;t++){
            if(testData[t]>maxVal){
                maxVal = testData[t];
            }
            if(testData[t]<minVal){
                minVal = testData[t];
            }
        }

        //printf("NormalizeData: %lf     %lf\n", maxVal, minVal);
        //center and normalize to [-1, +1]
        double valueToCenterData = 0;

        if(minVal>0 && maxVal>0)
        {
            valueToCenterData = -(maxVal-minVal)/2 -minVal;
        }else if(minVal<0 && maxVal<0){
            valueToCenterData = (maxVal-minVal)/2 +maxVal;
        }else{
            valueToCenterData = -(maxVal+minVal)/2;
        }

        //center data and normalizeto [-1, +1]
         for(t=0;t<nrTrainData;t++){
            trainData[t] = (trainData[t] + valueToCenterData)/ ((maxVal-minVal)/2);
            //data[t] = data[t] / ((maxVal-minVal)/2); //without centering
         }
         for(t=0;t<nrTestData;t++){
            testData[t] = (testData[t] + valueToCenterData)/ ((maxVal-minVal)/2);
            //data[t] = data[t] / ((maxVal-minVal)/2); //without centering
         }
    }

    double actFunctionTanh(double net){
        double y=tanh(net);
        //printf("%lf", y);

        return y;
    }

    double derivativeActFunctionTanh(double net){
        double y=1-(tanh(net)*tanh(net));
        //printf("%lf", y);

        return y;
    }

    void calcError(LAYER* outputLayer, double *label, double *error){
        int j=1;
        for(j=1; j<=outputLayer->nrUnits; j++){
            //printf("label: %lf,   output: %lf",label[i-1], outputLayer->out[i]);
            *error += (label[j-1] - outputLayer->out[j])*(label[j-1] - outputLayer->out[j]);
            //printf(" error: %.15lf\n",*error);
            //calc delta for output layer
            outputLayer->delta[j] = 2*(label[j-1]-outputLayer->out[j]);
        }
    }

    void backpropagation(LAYER* layer1, LAYER* layer2){
        int j=0;
        int k=0;
        for( j = 1; j <= layer1->nrUnits; j++ ) {
            layer1->delta[j] = 0.0; //set delta zero
            for( k = 1; k <= layer2->nrUnits; k++ ) {
                layer1->delta[j] += layer1->weights[j][k] * layer2->delta[k]; //add for all units oflayer 2
            }
        layer1->delta[j] = layer1->delta[j] * derivativeActFunctionTanh(layer1->in[j]);
        }
    }

    void updateWeights(LAYER* layer1, LAYER* layer2){
        int j=0;
        int i=0;
        for( j = 1; j <= layer2->nrUnits; j++ ) {
            layer1->dWeights[0][j] = layer2->delta[j]; //bias delta * 1
            layer1->weights[0][j] += ETA * layer1->dWeights[0][j];//bias update weight
            for( i = 1; i <= layer1->nrUnits; i++ ) {
                layer1->dWeights[i][j] = layer2->delta[j] * layer1->out[i];
                layer1->weights[i][j] += ETA * layer1->dWeights[i][j];
            }
        }

    }

    void calcInOutHidden(LAYER* layer1, LAYER* layer2){
    int j=0;
    int i=0;
        layer2->out[0]=1; //bias =1;
        for( j = 1; j <= layer2->nrUnits; j++ ) {
            layer2->in[j] = layer1->weights[0][j]; //bias weight *1
            for( i = 1; i <= layer1->nrUnits; i++ ) {
                layer2->in[j] += layer1->out[i] * layer1->weights[i][j] ;
            }
        layer2->out[j] = actFunctionTanh(layer2->in[j]);
        }
    }

    void calcInOutOutputLayer(LAYER* layer1, LAYER* layer2){
    int j=0;
    int i=0;
        for( j = 1; j <= layer2->nrUnits; j++ ) {
            layer2->in[j] = layer1->weights[0][j]; //bias weight *1
            for( i = 1; i <= layer1->nrUnits; i++ ) {
                layer2->in[j] += layer1->out[i] * layer1->weights[i][j] ;
            }
        layer2->out[j] = layer2->in[j];
        //printf("output: %.15lf\n", layer2->out[j]);
        }
    }

     void classify(NN *nn, double *testData, int nrTest){
        //calc in out activation
        int t=0;
        for(t=0;t<nrTest;t++){
            int l=0;
            for(l=0;l<nn->nrLayer;l++){ //for all layers
                if(l==0){//input Layer
                    int p=0;
                    nn->layers[0].out[0] = 1;
                    for(p=1;p<=nn->layers[0].nrUnits;p++){
                        //printf("   inpLayer: %lf  %d  %d\n", (testData+ (MAX_NR_TEST_DATA* (p-1)) )[t],p, t);
                        nn->layers[0].out[p] = (testData+ (MAX_NR_TEST_DATA* (p-1)) )[t];
                    }

                }else if( (l>0) && (l<nn->nrLayer-1) ){//hidden layers
                    calcInOutHidden(&nn->layers[l-1], &nn->layers[l]);

                }else{//output layer
                    calcInOutOutputLayer(&nn->layers[l-1], &nn->layers[l]);
                    //TODO: for multiple output units
                    if(nn->layers[nn->nrLayer-1].out[1]>=0){
                        printf("+1\n");
                        //printf("+1   raw:%lf\n",nn->layers[nn->nrLayer-1].out[1]);
                    }else{
                        printf("-1\n");
                        //printf("-1   raw:%lf\n",nn->layers[nn->nrLayer-1].out[1]);
                    }
//                    printf("iter: %d,   %d \n",t,nrTest );
//                    printf("predicted: %lf\n", nn->layers[nn->nrLayer-1].out[1]);
                    //ErrorOverAllPoints += calcError(&nn.layers[l], &trainingLabel[t]);
                }
            }
        }
    }

     void regression(NN *nn, double *testData, int nrTest){
        //calc in out activation
        int t=0;
        for(t=0;t<nrTest;t++){
            int l=0;
            for(l=0;l<nn->nrLayer;l++){ //for all layers
                if(l==0){//input Layer
                    int p=0;
                    nn->layers[0].out[0] = 1;
                    for(p=1;p<=nn->layers[0].nrUnits;p++){
                        //printf("   inpLayer: %lf  %d  %d\n", (testData+ (MAX_NR_TEST_DATA* (p-1)) )[t],p, t);
                        nn->layers[0].out[p] = (testData+ (MAX_NR_TEST_DATA* (p-1)) )[t];
                    }

                }else if( (l>0) && (l<nn->nrLayer-1) ){//hidden layers
                    calcInOutHidden(&nn->layers[l-1], &nn->layers[l]);

                }else{//output layer
                    calcInOutOutputLayer(&nn->layers[l-1], &nn->layers[l]);
                    //TODO: for multiple output units
                    printf("%lf\n", nn->layers[l].out[1]);
//                    printf("iter: %d,   %d \n",t,nrTest );
//                    printf("predicted: %lf\n", nn->layers[nn->nrLayer-1].out[1]);
                    //ErrorOverAllPoints += calcError(&nn.layers[l], &trainingLabel[t]);
                }
            }
        }
    }

    //initialize layer
    void initLayer(NN* nn, LAYER *layer, int maxUnits){
        layer->nrUnits = maxUnits;
        setRandomWeights(nn);
    }
    //initialize NN
    void initNN(NN *nn, int *unitsPerLayerX){
        int i=0;
        nn->nrLayer = NR_LAYERS;
        for(i=0; i<nn->nrLayer;i++){
            initLayer(nn, &nn->layers[i], unitsPerLayerX[i]);
        }
    }

//-------------------------------END Functions-----------------------


//-------------------------------Main-----------------------
// create Neural Network
int nrTrain=0;
int nrTest=0;
double trainingData[NR_INPUTS][MAX_NR_TRAINING_DATA];
double trainingLabel[MAX_NR_TRAINING_DATA];
double testData[NR_INPUTS][MAX_NR_TEST_DATA];
NN nn;
initNN(&nn, maxUnitsLayerX);



//nn.nrLayer = NrLayers;
//nn.layers[0].bias = 15.3321;
//printf("NrUnits: %d\n",nn.layers[0].nrUnits);
//printf("%lf\n", nn.layers[0].bias);
//printf("TEST\n");
//
//nn.layers[0].nrUnits = 5;
//int i =0;
//for(i=0; i<nn.layers[0].nrUnits;i++){
//    nn.layers[0].in[i] = 0.0;
//}
//for(i=0; i<nn.layers[0].nrUnits;i++){
//    printf("%lf\n", nn.layers[0].in[i]);
//}





//Scan input
//#if (DEBUG ==0)
//    while (scanf("%lf,%lf,%lf", &data1, &data2, &data3) != EOF) {
//        if(data1==0 && data2==0 && data3==0){
//                //training finished
//                printf("training finished!\n\n");
//                while(scanf("%lf,%lf", &data1, &data2) != EOF){
//                    printf("data: %lf, ", data1);
//                    printf("data: %lf \n", data2);
//                }
//        break;
//        }
//        printf("data: %lf, ", data1);
//        printf("data: %lf, ", data2);
//        printf("data: %lf \n", data3);
//    }
//#endif

#if (DEBUG >=1)
    FILE * file;
    file = fopen ("../data/testInput11B.txt", "r");
    while (fscanf(file,"%lf,%lf,%lf", &trainingData[0][nrTrain], &trainingData[1][nrTrain], &trainingLabel[nrTrain]) != EOF) {
    #endif
    #if (DEBUG ==0)
    while (scanf("%lf,%lf,%lf", &trainingData[0][nrTrain], &trainingData[1][nrTrain], &trainingLabel[nrTrain]) != EOF) {
    #endif
    nrTrain++;
        if(trainingData[0][nrTrain-1]==0 && trainingData[1][nrTrain-1]==0 && trainingLabel[nrTrain-1]==0){
              //printf("training data finished!\n\n");
                #if (DEBUG >=1)
                while(fscanf(file,"%lf,%lf", &testData[0][nrTest], &testData[1][nrTest]) != EOF){
                #endif
                #if (DEBUG ==0)
                while(scanf("%lf,%lf", &testData[0][nrTest], &testData[1][nrTest]) != EOF){
                #endif
                    nrTest++;
                }


            break;
        }
    }

#if (DEBUG >=1)
    fclose(file);
#endif



    normalize(nrTrain-1,nrTest, trainingData[0],testData[0]);
    normalize(nrTrain-1,nrTest, trainingData[1],testData[1]);
//    normalize(nrTrain-1, trainingData[1]);
//    normalize(nrTest, testData[1]);
//    normalize(nrTest, testData[0]);
    //------------------start training
    int iter=0;
    for(iter=0; iter<MAX_ITER; iter++){//iterieren bis Grenze oder ziel error erreicht
        nn.overAllError=0;
        int t=0;
        for(t=0; t<nrTrain-1; t++){ //über training daten iterieren
            //calc in out activation
            int l=0;
            for(l=0;l<nn.nrLayer;l++){ //for all layers
                if(l==0){//input Layer
                    int p=0;
                    nn.layers[0].out[0] = 1;
                    for(p=1;p<=nn.layers[0].nrUnits;p++){
                        //printf("inpLayer: %lf  %d  %d\n", trainingData[p-1][t],p, t);
                        nn.layers[0].out[p] = trainingData[p-1][t];
                    }

                }else if( (l>0) && (l<nn.nrLayer-1) ){//hidden layers
                    calcInOutHidden(&nn.layers[l-1], &nn.layers[l]);

                }else{//output layer
                    calcInOutOutputLayer(&nn.layers[l-1], &nn.layers[l]);
                    //ErrorOverAllPoints += calcError(&nn.layers[l], &trainingLabel[t]);
                }
            }

            for(l=nn.nrLayer-1;l>0;l--){ //for all layers backward
                //backpropagation and weight update
                if(l==nn.nrLayer-1){//output layer
                    calcError(&nn.layers[l], &trainingLabel[t], &nn.overAllError); //calc overAll Error and delta for output layer
                    updateWeights(&nn.layers[l-1], &nn.layers[l]);  //weights between last hidden and output
                    backpropagation(&nn.layers[l-1], &nn.layers[l]);//delta of last hidden Layer
                }else{//hidden Layers
                    updateWeights(&nn.layers[l-1], &nn.layers[l]);
                    backpropagation(&nn.layers[l-1], &nn.layers[l]);
                }
            }
        }
        //    if(error<...){
        //        break;
        //    }

            if(iter%100==0){
                printf("iteration:%d,  error: %lf\n",iter ,nn.overAllError);
            }
    }


    //classify data points
    classify(&nn, &testData, nrTest);
    //regression(&nn, &testData, nrTest);


//    int t=0;
//    //print data
//    for(t=0;t<nrTrain-1;t++){
//        printf("%lf    %lf        %.1lf\n", trainingData[0][t], trainingData[1][t], trainingLabel[t]);
//    }

//    for(t=0;t<nrTest;t++){
//        printf("%lf    %lf\n", testData[0][t], testData[1][t]);
//    }














//-------------------------------END Main-----------------------

    return 0;
}
