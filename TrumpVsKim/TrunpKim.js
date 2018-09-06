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

const generateTestData = (trumpData, KimData) => {
  let trainDat = [];
  trumpData.forEach(element => {
    trainDat.push({
      input: bow(element, voc),
      output: [0]
    });
  });
  KimData.forEach(element => {
    trainDat.push({
      input: bow(element, voc),
      output: [1]
    });
  });

  return trainDat;
};

let allRawTrainData = [];
allRawTrainData = trumpTrainData.concat(kimTrainData);
console.log(allRawTrainData.length);
let voc = dict(allRawTrainData);

console.log(voc.words.length);
let finalTrainingData = generateTestData(trumpTrainData, kimTrainData);

const net = new brain.NeuralNetwork({ hiddenLayers: [10] });
net.setActivation('sigmoid');
const MAXITERATIONS = 100;
const conf = {
  iterations: MAXITERATIONS,
  log: true,
  logPeriod: 10,
  //learningRate: 0.3,
  errorThresh: 0.000005
};
net.train(finalTrainingData, conf);

console.log(net.run(bow(kimTrainData[0], voc)));
//console.log(bow("congratulations", voc));
