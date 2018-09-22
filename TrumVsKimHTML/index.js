
const dict = extractDictionary;

const generateX = (dataArr, namesArr) => {
    let trainDat = [];
    dataArr.forEach((element, index) => {
        element.forEach(element => {
            //trainDat.push(tf.tensor1d(bow(element, voc)));
            trainDat.push(bow(element, voc));

        });
    });
    console.info("X Data generated");
    return trainDat;

};

const generateY = (dataArr) => {
    const YDat = [];
    let tmpSingleY = [];
    dataArr.forEach((element, index) => {
        if (index == 0)
            tmpSingleY = [1, 0];
        else
            tmpSingleY = [0, 1];
        element.forEach(() => {
            //YDat.push(tf.tensor1d(tmpSingleY));
            YDat.push(tmpSingleY);

        });
    });

    return YDat;
}

const cleanData = data => {
    data.map((element, index) => {
        if (element.indexOf('"@') >= 0) {
            data.splice(index, 1);
        }
    });
};



const clean = "clean";
console.time(clean)
cleanData(trumpTrain);
cleanData(trumpTest);
cleanData(kimTrain);
cleanData(kimTest);
console.timeEnd(clean);
console.info("data cleaned");

const allRawTrainData = trumpTrain.concat(
    kimTrain,
    trumpTest,
    kimTest
);
const voc = dict(allRawTrainData);


console.time("testData");
const datX = generateX(
    [trumpTrain, kimTrain]
);
const x = tf.tensor2d(datX);

const datY = generateY(
    [trumpTrain, kimTrain]);
console.timeEnd("testData");
const y = tf.tensor2d(datY);

const model = tf.sequential({
    layers: [tf.layers.dense({ units: 200, inputShape: [5954], batchInputShape: [null,5954] }), tf.layers.dense({units: 200}), tf.layers.dense({units: 2})]
});

async function asd() {
    model.compile({ optimizer: 'sgd', loss: 'meanSquaredError' });
    await model.fit(x, y, {
        batchSize: 32,
        epochs: 100
    });
    console.log("fitted");

}
asd();

predictText = (input) => {
    model.predict(tf.tensor2d(bow(input, voc), [1, 5954]), {batchSize:  32}).print();
}

testThisText = () => {
    let inp = document.getElementById("UserInputText");
    //console.log(inp.value);
    predictText(inp.value);
}