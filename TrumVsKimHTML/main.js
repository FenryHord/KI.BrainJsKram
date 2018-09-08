const dict = extractDictionary;

const generateTestData = (dataArr, namesArr) => {
    let trainDat = [];
    dataArr.forEach((element, index) => {
        element.forEach(element => {
            trainDat.push({
                input: bow(element, voc),
                output: { [namesArr[index]]: 1 }
            });
        });
    });
    console.info("Test Data generated");
    return trainDat;

};

const cleanData = data => {
    data.map((element, index) => {
        if (element.indexOf('"@') >= 0) {
            data.splice(index, 1);
        }
    });
};

const trainAndSave = () => {
    const MAXITERATIONS = 100;
    const conf = {
        iterations: MAXITERATIONS,
        log: true,
        logPeriod: 10,
        callback: () => {
            console.timeEnd("trainSpeed");
            console.time("trainSpeed");
          /* for (var i = generatedTrianigData.length - 1; i > 0; i--) {
            var j = Math.floor(Math.random() * (i + 1));
            var temp = generatedTrianigData[i];
            generatedTrianigData[i] = generatedTrianigData[j];
            generatedTrianigData[j] = temp;
          } */
        },
        callbackPerid: 10,
        //learningRate: 0.3,
        errorThresh: 0.000005
    };
    console.time("trainSpeed");
    net.train(generatedTrianigData, conf);

    let lastJSON = net.toJSON();
    //fs.writeFileSync('./JSON/lastJSON.json', JSON.stringify(lastJSON));
};

const testTraining = (versuche) => {
    let testResult = '';
    let fromTrump = true;
    let cntRichtig = versuche;
    for (let i = 0; i < versuche; i++) {
        if (Math.random() < 0.5) {
            fromTrump = true;
            testResult =
                trumpTest[Math.round(Math.random() * trumpTest.length)];
            console.log('Trump', testResult.trimRight());
        } else {
            fromTrump = false;
            testResult = kimTest[Math.round(Math.random() * kimTest.length)];
            console.log('Kim', testResult.trimRight());
        }
        let result = net.run(bow(testResult, voc));
        console.log('Result: ', result);
        if (fromTrump === true && result.Trump > result.Kim) {
            console.log('RICHTIG es war TRUMP');
        } else if (fromTrump === false && result.Trump < result.Kim) {
            console.log('RICHTIG es war KIM');
        } else {
            console.log('FALSCH');
            cntRichtig--;
        }

        console.log('----------------------------------------');
    }
    let prozentRichtig = cntRichtig / versuche * 100;
    console.log('+++++++++++++++++++ ' + prozentRichtig + ' % ++++++++++++++++++++');
};


let net = new brain.NeuralNetwork({ hiddenLayers: [20, 20] });
net.setActivation('sigmoid');

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
const generatedTrianigData = generateTestData(
    [trumpTrain, kimTrain],
    ['Trump', 'Kim']
);
console.timeEnd("testData");



trainAndSave();
testTraining(100);