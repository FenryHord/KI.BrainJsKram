let fs = require('fs');
let brain = require('brain.js');
let mimir = require('./mimirIndex'),
  bow = mimir.bow,
  dict = mimir.dict;
let trumpTrainData = JSON.parse(
  fs.readFileSync('./Rohdaten/trumpTrainingData.json')
);
let trumpTestData = JSON.parse(
  fs.readFileSync('./Rohdaten/trumpTestData.json')
);
let kimTrainData = JSON.parse(
  fs.readFileSync('./Rohdaten/kardashianTrainingData.json')
);
let kimTestData = JSON.parse(
  fs.readFileSync('./Rohdaten/kardashianTestData.json')
);

const generateTestData = (trumpData, KimData) => {
  let trainDat = [];
  trumpData.forEach(element => {
    trainDat.push({
      input: bow(element, voc),
      output: { Trump: 1 }
    });
  });
  KimData.forEach(element => {
    trainDat.push({
      input: bow(element, voc),
      output: { Kim: 1 }
    });
  });

  return trainDat;
};

const allRawTrainData = trumpTrainData.concat(kimTrainData, trumpTestData, kimTestData);
const voc = dict(allRawTrainData);
const generatedTrianigData = generateTestData(trumpTrainData, kimTrainData);


const net = new brain.NeuralNetwork({ hiddenLayers: [20] });
net.setActivation('sigmoid');
const MAXITERATIONS = 100;
const conf = {
  iterations: MAXITERATIONS,
  log: true,
  logPeriod: 10,
  callback: () => {
    for (var i = generatedTrianigData.length - 1; i > 0; i--) {
      var j = Math.floor(Math.random() * (i + 1));
      var temp = generatedTrianigData[i];
      generatedTrianigData[i] = generatedTrianigData[j];
      generatedTrianigData[j] = temp;
    }
  },
  callbackPerid: 10,
  //learningRate: 0.3,
  errorThresh: 0.000005
};
net.train(generatedTrianigData, conf);

let testResult = "";
let fromTrump = true;
for (let i = 0; i < 20; i++) {
  if (Math.random() < 0.5) {
    fromTrump = true;
    testResult = trumpTestData[Math.round(Math.random() * trumpTestData.length)];
    console.log("Trump", testResult.trimRight());
  }
  else {
    fromTrump = false;
    testResult = kimTestData[Math.round(Math.random() * kimTestData.length)];
    console.log("Kim", testResult.trimRight());
  }
  let result = net.run(bow(testResult, voc));
  console.log("Result: ", result);
  if (fromTrump === true && result.Trump > result.Kim) {
    console.log("RICHTIG es war TRUMP");
  }
  else if (fromTrump === false && result.Trump < result.Kim) {
    console.log("RICHTIG es war KIM");
  }
  else {
    console.log("FALSCH");
  }
  console.log("----------------------------------------");
}
//console.log(bow("congratulations", voc));
